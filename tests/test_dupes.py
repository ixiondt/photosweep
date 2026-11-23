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
    assert dupes.counts(dupes.groups(conn)) == {"same": 2, "similar": 1, "mirrored": 0, "extras": 5,
                                                "deletable": {"same": 3, "similar": 2, "mirrored": 0}}
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
    assert summary["dupExtras"] == {"same": 1, "similar": 1, "mirrored": 0}


WRITE = {"Origin": "http://testserver", "X-Photosweep": "1", "Content-Type": "application/json"}


def test_select_all_extras_is_same_only(client):
    h = {"Tailscale-User-Login": OP}
    assert client.get("/api/dupes/extras?tier=similar", headers=h).status_code == 400
    assert client.get("/api/dupes/extras?tier=similar&mirrored=1", headers=h).json() == {"ids": [], "count": 0}
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


# ------------------------------------- fingerprinted but not labelled ----

def test_unlabelled_duplicates_group_keep_and_delete_like_labelled_ones(client, monkeypatch):
    """With MODEL=none (or before the model gets to them) a row stays pending
    with a hash. It must group, keep, delete, and on a failed delete go back to
    pending, not to Errors or to labelled."""
    import contextlib
    import json
    from photosweep import config, db, dupes as wdupes, nextcloud
    c = db.connect(config.DB_PATH)
    orig = c.execute(
        "INSERT INTO files (rel, size, mtime, status, width, height, dhash) VALUES (?,?,?,?,?,?,?)",
        ("u.jpg", 100_000, "2020-01-01", "pending", 3000, 2000, _hash(7))).lastrowid
    copy = c.execute(
        "INSERT INTO files (rel, size, mtime, status, width, height, dhash) VALUES (?,?,?,?,?,?,?)",
        ("u-copy.jpg", 50_000, "2020-01-01", "pending", 1000, 666, _hash(7, 1))).lastrowid
    unprinted = c.execute("INSERT INTO files (rel, size, mtime) VALUES ('n.jpg', 1, 'x')").lastrowid
    wdupes.link(c, never)
    h = {"Tailscale-User-Login": OP}
    same = client.get("/api/dupes?tier=same&limit=100", headers=h).json()
    assert [[i["id"] for i in g["items"]] for g in same["groups"] if g["items"][0]["id"] == orig] == [[orig, copy]]
    assert copy in client.get("/api/dupes/extras?tier=same", headers=h).json()["ids"]

    r = client.post("/api/keep", headers={**h, **WRITE}, content=json.dumps({"ids": [orig, copy]}))
    assert r.json()["updated"] == 2
    assert client.get("/api/summary", headers=h).json()["counts"]["kept"] == 2
    client.post("/api/keep", headers={**h, **WRITE}, content=json.dumps({"ids": [orig, copy], "keep": False}))

    with open(config.NC_SECRET_FILE, "w") as f:
        f.write("app-password")
    monkeypatch.setattr(nextcloud, "client", lambda *_a: contextlib.nullcontext(None))
    monkeypatch.setattr(nextcloud, "delete", lambda _c, _b, _u, _p, rel: "error:500")
    r = client.post("/api/delete", headers={**h, **WRITE}, content=json.dumps({"ids": [copy, unprinted]})).json()
    assert r["failed"] == 2
    st = {x["id"]: x["status"] for x in c.execute("SELECT id, status FROM files")}
    assert st[copy] == "pending" and st[unprinted] == "pending"  # back where they were
    monkeypatch.setattr(nextcloud, "delete", lambda _c, _b, _u, _p, rel: "deleted")
    client.post("/api/delete", headers={**h, **WRITE}, content=json.dumps({"ids": [copy]}))
    assert c.execute("SELECT status FROM files WHERE id=?", (copy,)).fetchone()[0] == "deleted"
    assert all(orig not in g["ids"] for g in wdupes.groups(c))  # its group is settled


def test_summary_with_no_model_says_so_and_counts_fingerprints(client, monkeypatch):
    from photosweep import config, db
    c = db.connect(config.DB_PATH)
    c.execute("INSERT INTO files (rel, size, mtime, status, dhash) VALUES ('p.jpg', 1, 'x', 'pending', ?)", (_hash(9),))
    c.execute("INSERT INTO files (rel, size, mtime) VALUES ('q.jpg', 1, 'x')")  # not reached yet
    monkeypatch.setattr(config, "LABELLING", False)
    monkeypatch.setattr(config, "ROUTER_URL", "https://api.example.com/v1")  # would be remote, if it were used
    body = client.get("/api/summary", headers={"Tailscale-User-Login": OP}).json()
    assert body["labelling"] is False
    assert body["modelRemote"]["remote"] is False  # no model, so nothing leaves
    p = body["progress"]
    assert (p["total"], p["fingerprinted"]) == (6, 5)  # four labelled + p.jpg; q.jpg still to do


# ------------------------------------------------------------- mirrors ----

def _flat_scene(seed):
    """Big flat areas, where neighbouring pixels tie: the case a hash worked
    out from the plain one got wrong (66 bits on 2026-11-23)."""
    img = _scene(seed)
    ImageDraw.Draw(img).rectangle((0, 0, 400, 900), fill=(255, 255, 255))
    return img


def _hashes(img):
    data = imaging.jpeg_bytes(img, 512)
    with Image.open(io.BytesIO(data)) as stored:
        return imaging.dhash(stored), imaging.dhash_flipped(stored)


def test_a_mirrored_copy_is_found_by_its_flipped_hash():
    for base in (_scene(7), _flat_scene(7)):
        h, f = _hashes(base)
        other, _ = _hashes(_reencoded(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((600, 450)),
                                      "JPEG", quality=50))
        assert dupes.distance(h, other) > config.DUP_SIMILAR  # the plain hash never pairs them
        assert dupes.mirrored(h, f, other) and dupes.distance(f, other) <= config.DUP_SAME
        assert not dupes.mirrored(h, f, _hashes(_scene(8))[0])
    assert not dupes.mirrored(h, None, other) and not dupes.mirrored(h, "", other)


def test_backfill_flips_from_thumbnails_once(conn, tmp_path):
    good = add(conn, "a.jpg", _hash(1))
    lost = add(conn, "b.jpg", _hash(2))
    add(conn, "c.jpg", None)
    _scene(1).save(tmp_path / f"{good}.jpg", "JPEG")

    def thumb(i):
        return str(tmp_path / f"{i}.jpg")
    assert dupes.backfill_flips(conn, thumb, never) == 2
    got = dict(conn.execute("SELECT id, dhash_flip FROM files"))
    assert len(got[good]) == 64 and got[lost] == "" and got[3] is None
    assert dupes.backfill_flips(conn, thumb, never) == 0  # nothing left, the lost one not retried


def add_flip(conn, rel, dhash, flip, **kw):
    i = add(conn, rel, dhash, **kw)
    conn.execute("UPDATE files SET dhash_flip=? WHERE id=?", (flip, i))
    return i


def test_similar_marks_mirrored_copies_and_can_show_only_them(conn, monkeypatch):
    monkeypatch.setattr(config, "DUP_EMBED", 0.96)
    best = add_flip(conn, "selfie.jpeg", _hash(5), _hash(6), w=2160, h=3840, size=2_400_000, camera="phone")
    flip = add_flip(conn, "selfie-forward.jpg", _hash(6, 9), _hash(5, 9), w=2160, h=3840, size=500_000)
    other = add_flip(conn, "burst-a.jpg", _hash(9), _hash(11), camera="phone")
    other2 = add_flip(conn, "burst-b.jpg", _hash(10), _hash(12), camera="phone")
    conn.executemany("INSERT INTO embed_pairs (a, b, sim) VALUES (?, ?, ?)",
                     [(best, flip, 0.976), (other, other2, 0.97)])
    gs = {g["ids"][0]: g for g in dupes.groups(conn)}
    assert gs[best]["tier"] == "similar" and gs[best]["mirror"] == [False, True] and gs[best]["mirrored"]
    assert not gs[other]["mirrored"]
    assert dupes.counts(list(gs.values()))["mirrored"] == 1
    cols = "id, rel"
    assert dupes.page(conn, "similar", 0, 10, cols)["total"] == 2
    only = dupes.page(conn, "similar", 0, 10, cols, only_mirrored=True)
    assert only["total"] == 1 and [i["mirror"] for i in only["groups"][0]["items"]] == [False, True]
    assert dupes.mirrored_extras(list(gs.values())) == [flip]
    conn.execute("UPDATE files SET kept=1 WHERE id=?", (flip,))
    # a Keep is never selected, and wins: the kept flip ranks first, so the
    # other copy becomes the mirrored one, as a Keep decides a Same group too
    assert dupes.mirrored_extras(dupes.groups(conn)) == [best]


def test_a_mirrored_pair_is_found_below_the_best_copy_too(conn, monkeypatch):
    """In a group whose best copy is a third picture, the flipped one of the
    pair is still marked (the synthetic page check, 2026-11-23)."""
    monkeypatch.setattr(config, "DUP_EMBED", 0.96)
    top = add_flip(conn, "big.jpg", _hash(20), _hash(30), w=4000, h=3000, camera="phone")
    a = add_flip(conn, "selfie.jpg", _hash(21), _hash(22), size=900_000, camera="phone")
    b = add_flip(conn, "selfie-forward.jpg", _hash(22, 4), _hash(21, 4), size=200_000)
    conn.executemany("INSERT INTO embed_pairs (a, b, sim) VALUES (?, ?, ?)", [(top, a, 0.97), (a, b, 0.98)])
    (g,) = dupes.groups(conn)
    assert g["ids"] == [top, a, b] and g["mirror"] == [False, False, True] and g["mirrored"]
    assert dupes.mirrored_extras([g]) == [b]  # not `a`, a different photo of the group
