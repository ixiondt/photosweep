#!/usr/bin/env bash
# smoke.sh -- build the image, run the tests in it, then bring the compose stack
# up against three generated pictures and a stand-in model, and check the page
# end to end: sign-in, labelling, and one duplicate group; then again from
# empty with MODEL=none, where the copy must be found with nothing labelled.
# Needs Docker with the compose plugin. Leaves nothing behind.
#
#   scripts/smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
umask 022  # a root shell at 077 makes files the containers' user cannot read
WORK=$(mktemp -d)
chmod 755 "$WORK"
P=photosweep-smoke
PORT=18087
BASE="http://127.0.0.1:$PORT"
PW="smoke test passphrase $(date +%s)"
fail() { echo "FAIL: $*"; docker compose -p $P logs --tail 40 2>/dev/null || true; exit 1; }
cleanup() {
	docker compose -p $P -f "$ROOT/compose.yaml" -f "$WORK/override.yaml" down -v >/dev/null 2>&1 || true
	docker run --rm -v "$WORK:/w" alpine rm -rf /w/photos >/dev/null 2>&1 || true
	rm -rf "$WORK"
}
trap cleanup EXIT

echo "== build =="
docker build -q -t photosweep:latest . >/dev/null

echo "== tests, inside the image =="
docker run --rm --user root -v "$ROOT/tests:/app/tests:ro" -v "$ROOT/requirements-dev.txt:/app/requirements-dev.txt:ro" \
	-e DATA_DIR=/tmp/ps-test -e PICS_ROOT=/tmp photosweep:latest \
	sh -c 'pip install -q --root-user-action=ignore pytest==9.2.0 >/dev/null && python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1' \
	| tee "$WORK/pytest.txt"
grep -q " passed" "$WORK/pytest.txt" && ! grep -q "failed\|error" "$WORK/pytest.txt" || fail "tests"

echo "== three pictures: one, a smaller copy of it, and a different one =="
mkdir -p "$WORK/photos/2024"
docker run --rm --user root -v "$WORK/photos:/photos" photosweep:latest python -c '
from PIL import Image, ImageDraw
def scene(seed, size=(1600, 1200)):
    im = Image.new("RGB", size, (20 + seed * 70, 40, 90)); d = ImageDraw.Draw(im)
    for i in range(12):
        x, y = (i * 137 * (seed + 1)) % size[0], (i * 89 * (seed + 2)) % size[1]
        d.ellipse([x, y, x + 300, y + 300], fill=((i * 40) % 255, (i * 90 + seed * 50) % 255, (i * 20) % 255))
    return im
a = scene(1); a.save("/photos/2024/IMG_0001.jpg", quality=92)
a.resize((800, 600)).save("/photos/2024/IMG_0001-WA0001.jpg", quality=70)
scene(2).save("/photos/2024/IMG_0002.jpg", quality=92)
'
docker run --rm --user root -v "$WORK/photos:/photos" alpine chown -R 33:33 /photos

cat > "$WORK/stub.py" <<'PY'
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
ANSWER = json.dumps({"category": "photo", "confidence": 90, "reason": "stand-in", "blurry": False, "clutter": False})
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps({"message": {"role": "assistant", "content": ANSWER}, "done": True}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("0.0.0.0", 11434), H).serve_forever()
PY
cat > "$WORK/.env" <<EOF
PHOTOS_DIR=$WORK/photos
OLLAMA_URL=http://stub:11434
MODEL=stand-in
AUTH_MODE=password
NC_BASE=
PORT=$PORT
EOF
cat > "$WORK/override.yaml" <<EOF
services:
  web: { env_file: !override [$WORK/.env] }
  scanner: { env_file: !override [$WORK/.env] }
  stub:
    image: photosweep:latest
    command: ["python", "/stub.py"]
    user: "0:0"
    volumes: ["$WORK/stub.py:/stub.py:ro"]
    healthcheck: { disable: true }
EOF

echo "== up =="
export PHOTOS_DIR="$WORK/photos" PORT
docker compose -p $P -f compose.yaml -f "$WORK/override.yaml" up -d --quiet-pull >/dev/null 2>&1 \
	|| { docker compose -p $P -f compose.yaml -f "$WORK/override.yaml" up -d; fail "compose up"; }
for _ in $(seq 30); do curl -sf "$BASE/healthz" >/dev/null && break; sleep 1; done
curl -sf "$BASE/healthz" | grep -q '"ok":true' || fail "healthz"

echo "== signed out: only the sign-in page =="
[ "$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' "$BASE/")" = "303 $BASE/login" ] || fail "/ should send to /login"
[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/summary")" = 401 ] || fail "summary open while signed out"
curl -s "$BASE/api/auth" | grep -q '"passwordSet":false' || fail "fresh install should have no password"

echo "== set a password, sign in =="
printf '%s\n' "$PW" | docker compose -p $P exec -T web python -m photosweep.passwd --stdin
J=(-H "Origin: $BASE" -H "X-Photosweep: 1" -H "Content-Type: application/json")
[ "$(curl -s -o /dev/null -w '%{http_code}' "${J[@]}" -d '{"password":"not it at all, sorry"}' "$BASE/api/login")" = 401 ] || fail "wrong password accepted"
curl -s -c "$WORK/jar" "${J[@]}" -d "{\"password\":\"$PW\"}" "$BASE/api/login" | grep -q '"ok":true' || fail "sign-in"
[ "$(curl -s -o /dev/null -w '%{http_code}' -b "$WORK/jar" "$BASE/")" = 200 ] || fail "page after sign-in"

echo "== the scanner labels all three and groups the copy =="
for _ in $(seq 90); do
	S=$(curl -s -b "$WORK/jar" "$BASE/api/summary")
	echo "$S" | grep -q '"dupGroups":{"same":1' && break
	sleep 2
done
echo "$S" | python3 -c '
import json, sys
s = json.load(sys.stdin)
print("labelled", s["progress"]["labelled"], "of", s["progress"]["total"], "| dup groups", s["dupGroups"], "| extras", s["dupExtras"])
assert s["progress"]["labelled"] == 3, "not all labelled"
assert s["dupGroups"] == {"same": 1, "similar": 0}, "copy not grouped"
assert s["nextcloudReady"] is False and s["nextcloudSetUp"] is False
' || fail "scan"
curl -s -b "$WORK/jar" "$BASE/api/dupes?tier=same" | python3 -c '
import json, sys
g = json.load(sys.stdin)["groups"][0]["items"]
print("best copy:", g[0]["rel"])
assert g[0]["rel"] == "2024/IMG_0001.jpg", "the larger original should be the best copy"
' || fail "keeper"

echo "== deleting is refused without Nextcloud =="
ID=$(curl -s -b "$WORK/jar" "$BASE/api/dupes/extras?tier=same" | python3 -c 'import json,sys; print(json.load(sys.stdin)["ids"][0])')
curl -s -b "$WORK/jar" "${J[@]}" -d "{\"ids\":[$ID]}" "$BASE/api/dupes/delete-extras" | grep -q NEXTCLOUD_NOT_CONFIGURED || fail "delete without Nextcloud"

echo "== sign out =="
curl -s -b "$WORK/jar" -c "$WORK/jar" "${J[@]}" -d '{}' "$BASE/api/logout" >/dev/null
[ "$(curl -s -o /dev/null -w '%{http_code}' -b "$WORK/jar" "$BASE/api/summary")" = 401 ] || fail "still signed in"

echo "== no model: a fresh stack with MODEL=none finds the copy and labels nothing =="
docker compose -p $P -f compose.yaml -f "$WORK/override.yaml" down -v >/dev/null 2>&1
sed -i 's/^MODEL=.*/MODEL=none/' "$WORK/.env"
docker compose -p $P -f compose.yaml -f "$WORK/override.yaml" up -d --quiet-pull >/dev/null 2>&1 || fail "compose up (no model)"
for _ in $(seq 30); do curl -sf "$BASE/healthz" >/dev/null && break; sleep 1; done
printf '%s
' "$PW" | docker compose -p $P exec -T web python -m photosweep.passwd --stdin
curl -s -c "$WORK/jar" "${J[@]}" -d "{\"password\":\"$PW\"}" "$BASE/api/login" | grep -q '"ok":true' || fail "sign-in (no model)"
for _ in $(seq 60); do
	S=$(curl -s -b "$WORK/jar" "$BASE/api/summary")
	echo "$S" | grep -q '"dupGroups":{"same":1' && break
	sleep 2
done
echo "$S" | python3 -c '
import json, sys
s = json.load(sys.stdin)
p = s["progress"]
print("fingerprinted", p["fingerprinted"], "of", p["total"], "| labelled", p["labelled"], "| dup groups", s["dupGroups"])
assert s["labelling"] is False and s["model"] == "none"
assert p["fingerprinted"] == 3 and p["labelled"] == 0, "no model, so nothing labelled"
assert s["dupGroups"] == {"same": 1, "similar": 0}, "copy not grouped without a model"
' || fail "no-model scan"
curl -s -b "$WORK/jar" "$BASE/api/dupes?tier=same" | python3 -c '
import json, sys
g = json.load(sys.stdin)["groups"][0]["items"]
assert g[0]["rel"] == "2024/IMG_0001.jpg", "the larger original should be the best copy"
' || fail "no-model keeper"

echo "SMOKE OK"
