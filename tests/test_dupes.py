"""Duplicates: the hash survives what a forward does to a file, linking finds
each pair once, groups are tiered and ranked, and only live files count."""
import importlib
import io
import random
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageEnhance

from photosweep import config, db, dupes, imaging

OP = "owner@example.com"


def _scene(seed, size=(1200, 900)):
    """A picture with structure at several scales, unlike any other seed's."""
    rnd = random.Random(seed)
    img = Image.new("RGB", size, tuple(rnd.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rnd.randrange(size[0]), rnd.randrange(size[1])
        r = rnd.randrange(30, 300)
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rnd.randrange(256) for _ in range(3)))
    return img


def _thumb_hash(img):
    """What the scanner does: open, thumbnail to 512 as JPEG, hash the stored JPEG."""
    data = imaging.jpeg_bytes(img, 512)
    with Image.open(io.BytesIO(data)) as stored:
        return imaging.dhash(stored)


def _reencoded(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_dhash_survives_a_forward_and_tells_pictures_apart():
    base = _scene(1)
    h = _thumb_hash(base)
    assert len(h) == 64
    variants = {
        "whatsapp-ish": _reencoded(base.resize((450, 338)), "JPEG", quality=40),
        "png": _reencoded(base, "PNG"),
        "half size": base.resize((600, 450)),
        "a little brighter": ImageEnhance.Brightness(base).enhance(1.08),
    }
    for name, v in variants.items():
        assert dupes.distance(h, _thumb_hash(v)) <= config.DUP_SAME, name
    for seed in range(2, 12):
        assert dupes.distance(h, _thumb_hash(_scene(seed))) > config.DUP_SIMILAR, seed


# ------------------------------------------------------------ fixtures ----

def _hash(seed, flip=0):
    """A 256-bit hash, and the same one with `flip` bits changed."""
    rnd = random.Random(seed)
    bits = rnd.getrandbits(256)
    for b in rnd.sample(range(256), flip):
        bits ^= 1 << b
    return f"{bits:064x}"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    return db.connect(config.DB_PATH)


def add(conn, rel, dhash, status="done", kept=0, w=1000, h=750, size=100_000, camera=None, mtime="2020-01-01"):
    cur = conn.execute(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, width, height, camera, dhash) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (rel, size, mtime, status, "photo", 90, kept, w, h, camera, dhash))
    return cur.lastrowid


def never():
    return False


# ------------------------------------------------------------- linking ----

def test_link_finds_each_pair_once_then_only_new_files(conn):
    a = add(conn, "a.jpg", _hash(1))
    b = add(conn, "b.jpg", _hash(1, 3))
    add(conn, "c.jpg", _hash(2))
    assert dupes.link(conn, never) == (3, 1)
    assert [tuple(r) for r in conn.execute("SELECT a, b, dist FROM dup_pairs")] == [(a, b, 3)]
    assert dupes.link(conn, never) == (0, 0)  # nothing new, nothing done
    d = add(conn, "d.jpg", _hash(1, 5))
    assert dupes.link(conn, never) == (1, 2)  # the newcomer against everyone, once
    assert {(r["a"], r["b"]) for r in conn.execute("SELECT a, b FROM dup_pairs")} == {(a, b), (a, d), (b, d)}


def test_link_stops_cleanly_and_resumes(conn):
    for i in range(6):
        add(conn, f"{i}.jpg", _hash(1, i))
    calls = iter([False, False, True])
    linked, _ = dupes.link(conn, lambda: next(calls, True), batch=1)
    assert linked == 2
    assert dupes.link(conn, never)[0] == 4
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 15  # all 6 choose 2, once each


def test_changing_the_cutoff_relinks_everything(conn, monkeypatch):
    add(conn, "a.jpg", _hash(1))
    add(conn, "b.jpg", _hash(1, 30))
    dupes.link(conn, never)
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 1
    monkeypatch.setattr(config, "DUP_SIMILAR", 20)
    assert dupes.link(conn, never) == (2, 0)  # the 30-bit pair is beyond the new cutoff
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 0


def test_backfill_hashes_from_thumbnails(conn, tmp_path):
    ok = add(conn, "a.jpg", None)
    missing = add(conn, "b.jpg", None)
    paths = {ok: tmp_path / "a.jpg", missing: tmp_path / "nope.jpg"}
    _scene(1).save(paths[ok], format="JPEG")
    assert dupes.backfill_hashes(conn, lambda i: str(paths[i]), never) == 1
    got = dict(conn.execute("SELECT id, dhash FROM files").fetchall())
    assert got[ok] and got[missing] is None


# -------------------------------------------------------------- groups ----

def test_groups_tier_rank_and_count_only_live_unsettled_files(conn):
    # copies of one picture: the camera original is best, whatever the order
    wa = add(conn, "IMG-20200807-WA0003.jpg", _hash(1, 3), w=900, h=1600, size=167_143)
    orig = add(conn, "PXL_20200807.jpg", _hash(1), w=2304, h=4096, size=2_841_199, camera="Google Pixel")
    re1 = add(conn, "PXL_20200807_1.jpg", _hash(1, 5), w=2304, h=4096, size=2_841_199, camera="Google Pixel")
    # a burst: same size and camera, joined through the first frame at 15 and 18 bits
    burst = [add(conn, f"CP1A50{i}.JPG", _hash(2, (0, 15, 18)[i]), w=8192, h=5464, size=9_000_000 + i, camera="Canon")
             for i in range(3)]
    # settled or gone: never shown
    add(conn, "gone-a.jpg", _hash(3), status="deleted")
    add(conn, "gone-b.jpg", _hash(3, 2))
    add(conn, "kept-a.jpg", _hash(4), kept=1)
    add(conn, "kept-b.jpg", _hash(4, 2), kept=1)
    # a kept file still anchors its group, and is its best copy
    small_kept = add(conn, "small.jpg", _hash(5, 2), kept=1, w=100, h=100)
    big = add(conn, "big.jpg", _hash(5), w=4000, h=3000)
    dupes.link(conn, never)

    got = {tuple(sorted(g["ids"])): g for g in dupes.groups(conn)}
    assert set(got) == {tuple(sorted([wa, orig, re1])), tuple(sorted(burst)), tuple(sorted([small_kept, big]))}
    copies = got[tuple(sorted([wa, orig, re1]))]
    assert copies["tier"] == "same" and copies["ids"][0] == orig and copies["ids"][1] == re1  # `_1` loses the tie
    assert copies["dist"][0] is None and copies["dist"][2] == 3
    bursts = got[tuple(sorted(burst))]
    assert bursts["tier"] == "similar" and bursts["ids"][0] == burst[2]  # the largest file
    assert got[tuple(sorted([small_kept, big]))]["ids"][0] == small_kept
    # deletable: Same extras minus the kept one is 2 (copies) + 1 (big, under a kept best) = 3
    assert dupes.counts(dupes.groups(conn)) == {"same": 2, "similar": 1, "extras": 5,
                                                "deletable": {"same": 3, "similar": 2}}
    assert set(dupes.extras(dupes.groups(conn), "same")) == {wa, re1, big}
    assert orig not in dupes.extras(dupes.groups(conn), "same")  # a best copy is never an extra

    # deleting the copies settles that group
    conn.execute("UPDATE files SET status='deleted' WHERE id IN (?, ?)", (wa, re1))
    assert all(orig not in g["ids"] for g in dupes.groups(conn))


def test_keeper_a_blur_verdict_does_not_beat_resolution_but_splits_a_burst():
    base = {"kept": 0, "blurry": 0, "width": 900, "height": 1600, "camera": None, "size": 200_000,
            "sharpness": 50, "mtime": "2020", "rel": "x.jpg"}
    original = {**base, "id": 1, "width": 2304, "height": 4096, "camera": "Pixel", "size": 2_800_000, "blurry": 1}
    forward = {**base, "id": 2}
    assert sorted([forward, original], key=dupes.keeper_key)[0]["id"] == 1  # the 2026-11-21 case
    frame_a = {**original, "id": 3, "blurry": 1, "size": 3_000_000}
    frame_b = {**original, "id": 4, "blurry": 0}
    assert sorted([frame_a, frame_b], key=dupes.keeper_key)[0]["id"] == 4  # equal frames: the sharp one


def test_keeper_prefers_the_plain_name_over_a_copy_name():
    base = {"kept": 0, "blurry": None, "width": 10, "height": 10, "camera": "x", "size": 5,
            "sharpness": 1, "mtime": "2020", "id": 1}
    for original, copy in (("IMG_4575.JPG", "IMG_4575 (1).JPG"), ("IMAG0205.jpg", "IMAG0205_1.jpg"),
                           ("DSC_0011.JPG", "DSC_0011 copy.JPG")):
        rows = [{**base, "rel": copy, "id": 1}, {**base, "rel": original, "id": 2}]
        assert sorted(rows, key=dupes.keeper_key)[0]["rel"] == original


# ----------------------------------------------------------------- web ----

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
    from photosweep import db as wdb, dupes as wdupes
    c = wdb.connect(web.config.DB_PATH)
    add(c, "a.jpg", _hash(1), w=2000, h=1500)
    add(c, "a-copy.jpg", _hash(1, 2), w=800, h=600)
    add(c, "b0.jpg", _hash(2))
    add(c, "b1.jpg", _hash(2, 25))
    wdupes.link(c, never)
    return TestClient(web.app)


def test_dupes_needs_an_operator_and_a_known_tier(client):
    assert client.get("/api/dupes").status_code == 401
    r = client.get("/api/dupes?tier=nope", headers={"Tailscale-User-Login": OP})
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


def test_dupes_pages_by_group_with_the_keeper_first(client):
    h = {"Tailscale-User-Login": OP}
    body = client.get("/api/dupes?tier=same&limit=100000", headers=h).json()
    assert body["total"] == 1 and body["counts"]["extras"] == 2
    items = body["groups"][0]["items"]
    assert [i["rel"] for i in items] == ["a.jpg", "a-copy.jpg"]
    assert items[0]["keeper"] is True and items[1]["dist"] == 2 and items[1]["size"] == 100_000
    similar = client.get("/api/dupes?tier=similar", headers=h).json()
    assert similar["total"] == 1 and similar["groups"][0]["tier"] == "similar"
    assert client.get("/api/dupes?tier=same&offset=1", headers=h).json()["groups"] == []
    summary = client.get("/api/summary", headers=h).json()
    assert summary["counts"]["dupes"] == 2 and summary["dupGroups"] == {"same": 1, "similar": 1}
    assert summary["dupExtras"] == {"same": 1, "similar": 1}


WRITE = {"Origin": "http://testserver", "X-Photosweep": "1", "Content-Type": "application/json"}


def test_select_all_extras_is_same_only(client):
    h = {"Tailscale-User-Login": OP}
    assert client.get("/api/dupes/extras?tier=similar", headers=h).status_code == 400
    body = client.get("/api/dupes/extras?tier=same", headers=h).json()
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    assert body == {"ids": [c.execute("SELECT id FROM files WHERE rel='a-copy.jpg'").fetchone()[0]], "count": 1}


def test_bulk_delete_takes_only_current_same_extras(client, monkeypatch):
    """The page may send anything; the server deletes only what is an extra of
    a Same group right now. A best copy, a Similar frame and a kept file are
    skipped, so no Same group can lose its last copy through this route."""
    import contextlib
    import json
    from photosweep import config, db, nextcloud
    c = db.connect(config.DB_PATH)
    ids = {r["rel"]: r["id"] for r in c.execute("SELECT id, rel FROM files")}
    with open(config.NC_SECRET_FILE, "w") as f:
        f.write("app-password")
    sent = []
    monkeypatch.setattr(nextcloud, "client", lambda *_a: contextlib.nullcontext(None))
    monkeypatch.setattr(nextcloud, "delete", lambda _c, _b, _u, _p, rel: sent.append(rel) or "deleted")
    asked = [ids["a.jpg"], ids["a-copy.jpg"], ids["b1.jpg"]]  # best copy, extra, Similar frame
    r = client.post("/api/dupes/delete-extras", headers={"Tailscale-User-Login": OP, **WRITE},
                    content=json.dumps({"ids": asked})).json()
    assert sent == ["a-copy.jpg"] and r["deleted"] == 1
    assert r["skipped"] == 2 and set(r["skippedIds"]) == {ids["a.jpg"], ids["b1.jpg"]}
    # asked again for the now-lone best copy: still refused, the picture stays
    r = client.post("/api/dupes/delete-extras", headers={"Tailscale-User-Login": OP, **WRITE},
                    content=json.dumps({"ids": [ids["a.jpg"]]})).json()
    assert r["deleted"] == 0 and r["skipped"] == 1 and sent == ["a-copy.jpg"]
    assert db.verify_audit(c) == (True, None)
    # and the plain gates still hold
    assert client.post("/api/dupes/delete-extras", headers={"Tailscale-User-Login": OP},
                       content=json.dumps({"ids": asked})).status_code == 403
