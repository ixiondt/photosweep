"""The gates: who may read, and what a write must prove. Module reset per test
so config (read at import) picks up each test's environment."""
import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

OP = "owner@example.com"
HOST = "testserver"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PICS_ROOT", str(tmp_path / "pics"))
    monkeypatch.setenv("OPERATORS", OP)
    monkeypatch.setenv("AUTH_MODE", "header")
    monkeypatch.setenv("NC_BASE", "https://cloud.example.com")
    monkeypatch.setenv("NC_USER", "alice")
    for name in [m for m in sys.modules if m.startswith("photosweep")]:
        del sys.modules[name]
    web = importlib.import_module("photosweep.web")
    from photosweep import db
    conn = db.connect(web.config.DB_PATH)
    conn.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence) VALUES (?,?,?,?,?,?)",
        [("2020/a.png", 1, "2020-01-01T00:00:00+00:00", "done", "screenshot", 90),
         ("2020/b.jpg", 1, "2020-01-01T00:00:00+00:00", "done", "photo", 40),
         ("2020/c.jpg", 1, "2020-01-01T00:00:00+00:00", "pending", None, None)])
    return TestClient(web.app)


def who(extra=None):
    h = {"Tailscale-User-Login": OP}
    h.update(extra or {})
    return h


WRITE = {"Origin": f"http://{HOST}", "X-Photosweep": "1", "Content-Type": "application/json"}


def test_no_identity_is_refused(client):
    for path in ("/", "/api/summary", "/api/items", "/thumb/1", "/view/1"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.json()["code"] == "UNAUTHORIZED"


def test_wrong_identity_is_refused(client):
    assert client.get("/api/summary", headers={"Tailscale-User-Login": "else@example.com"}).status_code == 401


def test_empty_allowlist_denies_everyone(tmp_path, monkeypatch, client):
    from photosweep import config
    monkeypatch.setattr(config, "OPERATORS", set())
    assert client.get("/api/summary", headers=who()).status_code == 401


def test_summary_counts(client):
    body = client.get("/api/summary", headers=who()).json()
    assert body["counts"]["screenshot"] == 1 and body["counts"]["photo"] == 1
    assert body["progress"] == {**body["progress"], "total": 3, "labelled": 2, "pending": 1}
    assert body["nextcloudReady"] is False


def test_items_clamps_and_validates(client):
    r = client.get("/api/items?tab=screenshot&limit=100000", headers=who())
    assert r.status_code == 200 and r.json()["total"] == 1
    assert client.get("/api/items?tab=nope", headers=who()).status_code == 400


def test_blurry_tab_lists_soft_photos_blurriest_first(client):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, sharpness) VALUES (?,?,?,?,?,?,?,?)",
        [("2020/soft.jpg", 1, "x", "done", "photo", 80, 0, 9.5),
         ("2020/softer.jpg", 1, "x", "done", "photo", 90, 0, 2.0),
         ("2020/fine.jpg", 1, "x", "done", "photo", 90, 0, 70.0),
         ("2020/kept.jpg", 1, "x", "done", "photo", 90, 1, 1.0),
         ("2020/meme.jpg", 1, "x", "done", "meme", 90, 0, 1.0),
         ("2020/unscored.jpg", 1, "x", "done", "photo", 90, 0, None)])
    body = client.get("/api/items?tab=blurry&min_conf=95", headers=who()).json()
    assert [i["rel"] for i in body["items"]] == ["2020/softer.jpg", "2020/soft.jpg"]  # min_conf ignored
    assert body["total"] == 2 and body["items"][0]["sharpness"] == 2.0
    s = client.get("/api/summary", headers=who()).json()
    assert s["counts"]["blurry"] == 2 and s["blurMax"] == config.BLUR_MAX


def test_model_blur_verdict_overrides_pixels_and_each_photo_is_in_one_tab(client):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, sharpness, blurry, blur_conf) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("2020/selfie.jpg", 1, "x", "done", "photo", 95, 0, 3.0, 0, 95),      # smooth, model: sharp
         ("2020/night.jpg", 1, "x", "done", "photo", 95, 0, 120.0, 1, 90),     # noisy, model: blurry
         ("2020/shaky.jpg", 1, "x", "done", "photo", 95, 0, 40.0, 1, 70),
         ("2020/unasked.jpg", 1, "x", "done", "photo", 95, 0, 4.0, None, None),
         ("2020/noanswer.jpg", 1, "x", "done", "photo", 95, 0, 6.0, -1, None),
         ("2020/noanswer-ok.jpg", 1, "x", "done", "photo", 95, 0, 60.0, -1, None),
         ("2020/nothing.jpg", 1, "x", "done", "photo", 95, 0, None, None, None)])
    blurry = client.get("/api/items?tab=blurry", headers=who()).json()
    assert [i["rel"] for i in blurry["items"]] == [
        "2020/night.jpg", "2020/shaky.jpg", "2020/unasked.jpg", "2020/noanswer.jpg"]  # surest, then pixels
    photos = client.get("/api/items?tab=photo&limit=200", headers=who()).json()
    names = {i["rel"] for i in photos["items"]}
    assert {"2020/selfie.jpg", "2020/noanswer-ok.jpg", "2020/nothing.jpg", "2020/b.jpg"} == names
    assert not names & {i["rel"] for i in blurry["items"]}
    s = client.get("/api/summary", headers=who()).json()
    assert s["counts"]["blurry"] == 4 and s["counts"]["photo"] == 4
    assert s["blurCheck"] == {"checked": 5, "total": 8}  # b.jpg from the fixture is a photo too


def test_security_headers(client):
    r = client.get("/api/summary", headers=who())
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'none'" in r.headers["Content-Security-Policy"]
    assert r.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("drop", ["Origin", "X-Photosweep"])
def test_write_needs_origin_and_header(client, drop):
    headers = who({k: v for k, v in WRITE.items() if k != drop})
    r = client.post("/api/keep", headers=headers, content='{"ids":[1]}')
    assert r.status_code == 403


def test_write_refuses_cross_origin(client):
    r = client.post("/api/keep", headers=who({**WRITE, "Origin": "https://evil.example"}), content='{"ids":[1]}')
    assert r.status_code == 403


def test_write_needs_json_content_type(client):
    r = client.post("/api/keep", headers=who({**WRITE, "Content-Type": "application/x-www-form-urlencoded"}),
                    content="ids=1")
    assert r.status_code == 415 and r.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.parametrize("body", ['{"ids":[]}', '{"ids":"1"}', '{"ids":[true]}', '[1]', 'nope'])
def test_write_validates_ids(client, body):
    assert client.post("/api/keep", headers=who(WRITE), content=body).status_code == 400


def test_keep_then_audit(client):
    r = client.post("/api/keep", headers=who(WRITE), content='{"ids":[1],"keep":true}')
    assert r.json() == {"updated": 1}
    s = client.get("/api/summary", headers=who()).json()
    assert s["counts"]["kept"] == 1 and s["counts"]["screenshot"] == 0
    from photosweep import config, db
    assert db.verify_audit(db.connect(config.DB_PATH)) == (True, None)


def test_delete_without_nextcloud_is_409(client):
    r = client.post("/api/delete", headers=who(WRITE), content='{"ids":[1]}')
    assert r.status_code == 409 and r.json()["code"] == "NEXTCLOUD_NOT_CONFIGURED"


def test_pause_round_trip(client):
    assert client.post("/api/pause", headers=who(WRITE), content='{"paused":true}').json() == {"paused": True}
    assert client.get("/api/summary", headers=who()).json()["paused"] is True


def test_recheck_relabels_errors_and_requeues_photo_blur_without_moving_them(client):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, attempts, error, blurry, blur_conf) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("2020/err.jpg", 1, "x", "error", None, None, 3, "model: not JSON: ''", None, None),
         ("2020/pic.jpg", 1, "x", "done", "photo", 95, 0, None, 0, 90),
         ("2020/meme.jpg", 1, "x", "done", "meme", 95, 0, None, None, None)])
    ids = [r["id"] for r in c.execute("SELECT id FROM files WHERE rel IN ('2020/err.jpg','2020/pic.jpg','2020/meme.jpg') ORDER BY rel")]
    r = client.post("/api/recheck", headers=who(WRITE), content=json.dumps({"ids": ids}))
    assert r.json() == {"relabel": 1, "blur": 1}  # the meme is neither
    rows = {x["rel"]: dict(x) for x in c.execute("SELECT * FROM files")}
    assert (rows["2020/err.jpg"]["status"], rows["2020/err.jpg"]["attempts"], rows["2020/err.jpg"]["error"]) == (
        "pending", 0, None)
    assert (rows["2020/pic.jpg"]["blurry"], rows["2020/pic.jpg"]["blur_recheck"]) == (0, 1)  # verdict stands
    assert rows["2020/pic.jpg"]["clutter_recheck"] == 1 and rows["2020/meme.jpg"]["clutter_recheck"] == 0
    assert rows["2020/meme.jpg"]["blur_recheck"] == 0
    assert db.verify_audit(c) == (True, None)
    assert client.post("/api/recheck", headers=who({**WRITE, "Origin": "https://evil.example"}),
                       content=json.dumps({"ids": ids})).status_code == 403


def test_delete_clears_errors_rows_and_a_failure_puts_each_row_back(client, monkeypatch):
    import contextlib
    from photosweep import config, db, nextcloud
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, error) VALUES (?,?,?,?,?)",
        [("2020/empty.jpg", 0, "x", "error", "decode: UnidentifiedImageError"),
         ("2020/stuck.jpg", 0, "x", "error", "decode: UnidentifiedImageError")])
    ids = {r["rel"]: r["id"] for r in c.execute("SELECT id, rel FROM files")}
    with open(config.NC_SECRET_FILE, "w") as f:
        f.write("app-password")
    monkeypatch.setattr(nextcloud, "client", lambda *_a: contextlib.nullcontext(None))
    monkeypatch.setattr(nextcloud, "delete",
                        lambda _c, _b, _u, _p, rel: "deleted" if rel == "2020/empty.jpg" else "error:500")
    body = json.dumps({"ids": [ids["2020/empty.jpg"], ids["2020/stuck.jpg"], ids["2020/b.jpg"]]})
    r = client.post("/api/delete", headers=who(WRITE), content=body).json()
    assert (r["deleted"], r["failed"]) == (1, 2)
    status = {x["rel"]: x["status"] for x in c.execute("SELECT rel, status FROM files")}
    assert status["2020/empty.jpg"] == "deleted"
    assert status["2020/stuck.jpg"] == "error"  # back to Errors, not promoted to labelled
    assert status["2020/b.jpg"] == "done"       # a labelled row goes back to labelled
    assert status["2020/c.jpg"] == "pending"    # not asked for, untouched
    assert db.verify_audit(c) == (True, None)


def test_clutter_outranks_blurry_and_each_photo_is_in_one_tab(client):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, sharpness, blurry, clutter, "
        "clutter_conf, camera, name_hint) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [("2020/floor.jpg", 1, "x", "done", "photo", 95, 0, 60.0, 0, 1, 80, "Apple", None),
         ("2020/smear.jpg", 1, "x", "done", "photo", 95, 0, 60.0, 1, 1, 95, None, "messaging"),  # both: Clutter
         ("2020/shaky.jpg", 1, "x", "done", "photo", 95, 0, 60.0, 1, 0, 90, "Apple", None),
         ("2020/kid.jpg", 1, "x", "done", "photo", 95, 0, 60.0, 0, 0, 90, None, "messaging"),
         ("2020/noanswer.jpg", 1, "x", "done", "photo", 95, 0, 60.0, 0, -1, None, "Apple", None),
         ("2020/kept.jpg", 1, "x", "done", "photo", 95, 1, 60.0, 0, 1, 99, None, None)])

    def tab(name):
        return client.get(f"/api/items?tab={name}&limit=200", headers=who()).json()
    clutter, blurry, photos = tab("clutter"), tab("blurry"), tab("photo")
    assert [i["rel"] for i in clutter["items"]] == ["2020/smear.jpg", "2020/floor.jpg"]  # surest first
    assert [i["rel"] for i in blurry["items"]] == ["2020/shaky.jpg"]
    names = [{i["rel"] for i in t["items"]} for t in (clutter, blurry, photos)]
    assert names[2] == {"2020/kid.jpg", "2020/noanswer.jpg", "2020/b.jpg"}
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
    s = client.get("/api/summary", headers=who()).json()
    assert s["counts"]["clutter"] == 2 and s["counts"]["blurry"] == 1 and s["counts"]["photo"] == 3
    assert s["clutterCheck"] == {"checked": 5, "total": 6}  # kept is not counted; b.jpg is unasked


def test_source_filter_narrows_any_tab_and_reports_each_count(client):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, sharpness, camera, name_hint) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [("2020/cam.jpg", 1, "x", "done", "photo", 95, 60.0, "Apple", None),
         ("2020/wa.jpg", 1, "x", "done", "photo", 95, 60.0, None, "messaging"),
         ("2020/fb.jpg", 1, "x", "done", "photo", 95, 60.0, None, "facebook")])

    def get(src):
        return client.get(f"/api/items?tab=photo&source={src}", headers=who()).json()
    body = get("nocamera")
    assert {i["rel"] for i in body["items"]} == {"2020/wa.jpg", "2020/fb.jpg", "2020/b.jpg"}
    assert body["total"] == 3
    assert body["sources"] == {"all": 4, "camera": 1, "nocamera": 3, "messaging": 1}
    assert [i["rel"] for i in get("messaging")["items"]] == ["2020/wa.jpg"]
    assert [i["rel"] for i in get("camera")["items"]] == ["2020/cam.jpg"]
    r = client.get("/api/items?tab=photo&source=x%27%20OR%201=1--", headers=who())
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


def test_status_counts_re_asks_as_unchecked_and_their_pace(client):
    """A re-ask is work still to do, and a photo in Clutter is never asked
    about blur, so neither may leave the status line stuck at 'complete'."""
    import time
    from photosweep import config, db
    before = client.get("/api/summary", headers=who()).json()
    recent = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 60))
    c = db.connect(config.DB_PATH)
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, blurry, clutter, "
        "clutter_recheck, asked_at) VALUES (?,1,'x','done','photo',95,0,?,?,?,?)",
        [("r/answered.jpg", 0, 0, 0, recent),      # both questions settled, answered a minute ago
         ("r/re-ask.jpg", 0, 1, 1, recent),        # waiting for a clutter re-ask
         ("r/clutter.jpg", None, 1, 0, None),      # in Clutter: never asked about blur
         ("r/old.jpg", 0, 0, 0, "2000-01-01T00:00:00")])
    s = client.get("/api/summary", headers=who()).json()
    b0, c0 = before["blurCheck"], before["clutterCheck"]
    # the two in Clutter (re-ask.jpg is clutter=1 until re-asked) are outside the blur total
    assert s["blurCheck"] == {"checked": b0["checked"] + 2, "total": b0["total"] + 2}
    assert s["clutterCheck"] == {"checked": c0["checked"] + 3, "total": c0["total"] + 4}
    q = s["questions"]
    assert q["rechecks"] == 1 and q["left"] == before["questions"]["left"] + 1
    assert q["perHour"] == 0 and q["etaHours"] is None  # two answers are too few for a pace
    c.executemany("INSERT INTO files (rel, size, mtime, status, category, asked_at) VALUES (?,1,'x','done','photo',?)",
                  [(f"r/n{k}.jpg", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 600 + k)))
                   for k in range(10)])
    q = client.get("/api/summary", headers=who()).json()["questions"]
    # twelve answers over the ten minutes since the first: about 72 an hour, not 24
    assert 70 <= q["perHour"] <= 73 and q["etaHours"] == round(q["left"] / q["perHour"], 1)
