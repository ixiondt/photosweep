"""The scan loop: prepare, prefetch, and the router-outage path. The model is
faked; everything else (SQLite, Pillow, the thumbnail tree) is real."""
import os

import pytest
from PIL import Image

from photosweep import classify, config, db, imaging, scanner


@pytest.fixture
def lib(tmp_path, monkeypatch):
    pics = tmp_path / "pics"
    for i in range(4):
        d = pics / "2026" / "01"
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (i * 40, 90, 120)).save(d / f"img{i}.jpg")
    (pics / "2026" / "01" / "broken.jpg").write_bytes(b"not a jpeg")
    token = tmp_path / "router.token"
    token.write_text("t" * 40)
    monkeypatch.setattr(config, "PICS_ROOT", str(pics))
    monkeypatch.setattr(config, "THUMB_DIR", str(tmp_path / "thumbs"))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(config, "ROUTER_TOKEN_FILE", str(token))
    monkeypatch.setattr(scanner, "_stop", False)
    return tmp_path


def rows(path):
    conn = db.connect(path)
    return {r["rel"].rsplit("/", 1)[1]: dict(r) for r in conn.execute("SELECT * FROM files")}


def test_prepare_writes_thumb_and_touches_no_db(lib):
    out = scanner.prepare(7, "2026/01/img1.jpg")
    assert out[0] == "ok" and out[3:5] == (64, 48)
    with Image.open(scanner.thumb_path(7)) as im:  # scored from the thumbnail as stored
        assert out[5] == imaging.sharpness(im)
    assert os.path.exists(scanner.thumb_path(7))
    assert not [p for p in os.listdir(os.path.dirname(scanner.thumb_path(7))) if p.endswith(".tmp")]


def test_prepare_classifies_its_own_failures(lib):
    assert scanner.prepare(1, "2026/01/missing.jpg") == ("gone",)
    assert scanner.prepare(1, "2026/01/broken.jpg")[0] == "bad"
    assert scanner.prepare(1, "../../etc/passwd") == ("bad", "path escapes the library")


def test_next_pending_excludes_the_row_in_hand(lib):
    conn = db.connect(config.DB_PATH)
    scanner.walk(conn)
    first = scanner.next_pending(conn)
    second = scanner.next_pending(conn, exclude=first["id"])
    assert second["id"] > first["id"]


def test_run_labels_everything_once_with_prefetch(lib, monkeypatch):
    seen = []

    def fake(_http, _url, _token, _model, b64, **_k):
        seen.append(b64)
        return ("screenshot", 90, "fake")
    monkeypatch.setattr(classify, "classify", fake)
    scanner.run(limit=10)
    got = rows(config.DB_PATH)
    assert {k: v["status"] for k, v in got.items()} == {
        "img0.jpg": "done", "img1.jpg": "done", "img2.jpg": "done", "img3.jpg": "done",
        "broken.jpg": "error"}  # retried up to the cap, then parked in Errors
    assert got["broken.jpg"]["attempts"] == config.MAX_ATTEMPTS
    assert len(seen) == 4 and len(set(seen)) == 4  # each image sent once, none twice


def test_run_respects_limit_without_overshooting(lib, monkeypatch):
    monkeypatch.setattr(classify, "classify", lambda *a, **k: ("photo", 95, "x"))
    scanner.run(limit=2)
    assert sum(r["status"] == "done" for r in rows(config.DB_PATH).values()) == 2


def test_router_outage_retries_the_same_image_and_costs_it_nothing(lib, monkeypatch):
    calls = []

    def flaky(_http, _url, _token, _model, b64, **_k):
        calls.append(b64)
        if len(calls) == 1:
            raise classify.RouterUnavailable("down", retry_after=0)
        return ("meme", 80, "x")
    monkeypatch.setattr(classify, "classify", flaky)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.run(limit=1)
    assert calls[0] == calls[1]  # the retry was the same image, from the kept prepare
    done = [r for r in rows(config.DB_PATH).values() if r["status"] == "done"]
    assert len(done) == 1 and done[0]["attempts"] == 0


def test_run_stores_sharpness_and_backfill_fills_older_rows(lib, monkeypatch):
    monkeypatch.setattr(classify, "classify", lambda *a, **k: ("photo", 95, "x"))
    scanner.run(limit=10)
    conn = db.connect(config.DB_PATH)
    labelled = conn.execute("SELECT id, sharpness FROM files WHERE status='done'").fetchall()
    assert labelled and all(r["sharpness"] is not None for r in labelled)
    # rows labelled before the column existed: wiped, then refilled from the thumbnails
    conn.execute("UPDATE files SET sharpness=NULL")
    os.remove(scanner.thumb_path(labelled[0]["id"]))  # a missing thumb stays NULL, no crash
    assert scanner.backfill_sharpness(conn) == len(labelled) - 1
    left = conn.execute("SELECT id FROM files WHERE status='done' AND sharpness IS NULL").fetchall()
    assert [r["id"] for r in left] == [labelled[0]["id"]]


def _labelled(lib, monkeypatch, reasons=None):
    """Label the fixture library as photos; returns the connection."""
    monkeypatch.setattr(classify, "classify", lambda *a, **k: ("photo", 95, "x"))
    scanner.run(limit=10)
    conn = db.connect(config.DB_PATH)
    for rel, reason in (reasons or {}).items():
        conn.execute("UPDATE files SET reason=? WHERE rel=?", (reason, rel))
    return conn


def test_next_blur_takes_said_blurry_first_then_lowest_score(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch, {"2026/01/img3.jpg": "blurry but real photo"})
    conn.execute("UPDATE files SET sharpness=50 WHERE status='done'")
    conn.execute("UPDATE files SET sharpness=5 WHERE rel='2026/01/img1.jpg'")
    conn.execute("UPDATE files SET kept=1 WHERE rel='2026/01/img0.jpg'")
    order = []
    while (row := scanner.next_blur(conn)) is not None:
        order.append(row["rel"].rsplit("/", 1)[1])
        conn.execute("UPDATE files SET blurry=0 WHERE id=?", (row["id"],))
    assert order == ["img3.jpg", "img1.jpg", "img2.jpg"]  # kept and errored rows never asked


def test_judge_one_records_verdict_and_parks_bad_answers(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    ids = [r["id"] for r in conn.execute("SELECT id FROM files WHERE status='done' ORDER BY id")]
    answers = iter([(True, 88), classify.BadVerdict("not JSON: ''"), classify.BadVerdict("not JSON: ''")])

    def fake(*_a, **_k):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(classify, "judge_blur", fake)
    rows_ = {i: conn.execute("SELECT id, rel, blur_recheck FROM files WHERE id=?", (i,)).fetchone() for i in ids}
    assert scanner.judge_one(conn, None, "t", rows_[ids[0]]) == 1
    assert scanner.judge_one(conn, None, "t", rows_[ids[1]]) == -1
    os.remove(scanner.thumb_path(ids[2]))
    assert scanner.judge_one(conn, None, "t", rows_[ids[2]]) == -1  # no thumbnail: no model call
    got = {r["id"]: (r["blurry"], r["blur_conf"]) for r in conn.execute("SELECT id, blurry, blur_conf FROM files")}
    assert got[ids[0]] == (1, 88) and got[ids[1]] == (-1, None) and got[ids[2]] == (-1, None)


def test_service_alternates_labels_and_side_questions_then_idles(lib, monkeypatch):
    calls = []
    monkeypatch.setattr(classify, "classify", lambda *a, **k: calls.append("label") or ("photo", 95, "x"))
    monkeypatch.setattr(classify, "judge_blur", lambda *a, **k: calls.append("blur") or (False, 90))
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: calls.append("clutter") or (False, 90, "a dog"))

    def sleep(s):
        if s == 60:  # nothing left to label or check: the idle sleep ends the test
            scanner._stop = True
    monkeypatch.setattr(scanner.time, "sleep", sleep)
    scanner.run()
    got = rows(config.DB_PATH)
    assert all(r["blurry"] == 0 and r["clutter"] == 0 for k, r in got.items() if k != "broken.jpg")
    assert calls.count("label") == 4 and calls.count("blur") == 4 and calls.count("clutter") == 4
    # interleaved, not side-questions-after-all; and a photo's clutter question
    # comes before any blur one (failed tries on broken.jpg also earn side turns,
    # so the exact rhythm is not fixed)
    assert calls[:2] == ["label", "clutter"]
    assert calls.index("clutter") < calls.index("blur") and "clutter" in calls[:calls.index("label", 1) + 1]
    last_label = len(calls) - 1 - calls[::-1].index("label")
    assert calls.index("clutter") < last_label


def test_blur_check_survives_a_router_outage(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    n = conn.execute("SELECT count(*) FROM files WHERE status='done'").fetchone()[0]
    seen = []

    def flaky(*a, **_k):
        seen.append(a[-1])
        if len(seen) == 1:
            raise classify.RouterUnavailable("down", retry_after=0)
        return (True, 80)
    monkeypatch.setattr(classify, "judge_blur", flaky)
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: (False, 90, "x"))
    monkeypatch.setattr(scanner.time, "sleep", lambda s: setattr(scanner, "_stop", s == 60))
    scanner.run()
    assert seen[0] == seen[1] and len(seen) == n + 1  # the same photo again, then the rest once each
    assert conn.execute("SELECT count(*) FROM files WHERE blurry=1").fetchone()[0] == n


def test_a_label_retry_samples_and_the_first_try_does_not(lib, monkeypatch):
    modes = []

    def fake(*_a, sample=False):
        modes.append(sample)
        if len(modes) == 1:
            raise classify.BadVerdict("not JSON: ''")  # a greedy loop that never answered
        return ("photo", 90, "x")
    monkeypatch.setattr(classify, "classify", fake)
    scanner.run(limit=1)
    assert modes == [False, True]  # the same image, retried with sampling


def test_blur_check_gets_one_sampled_retry(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    modes = []

    def fake(*_a, sample=False):
        modes.append(sample)
        if not sample:
            raise classify.BadVerdict("not JSON: ''")
        return (False, 80)
    monkeypatch.setattr(classify, "judge_blur", fake)
    row = scanner.next_blur(conn)
    assert scanner.judge_one(conn, None, "t", row) == 0 and modes == [False, True]


def test_recheck_reads_the_original_larger_and_jumps_the_queue(lib, monkeypatch):
    big = lib / "pics" / "2026" / "01" / "img2.jpg"
    Image.new("RGB", (2000, 1500), (10, 200, 30)).save(big)  # larger than the thumbnail
    conn = _labelled(lib, monkeypatch)
    conn.execute("UPDATE files SET blurry=0, blur_conf=90 WHERE status='done'")
    conn.execute("UPDATE files SET blur_recheck=1 WHERE rel='2026/01/img2.jpg'")
    seen = []

    def fake(_h, _u, _t, _m, b64, sample=False):
        import base64
        import io
        seen.append(Image.open(io.BytesIO(base64.b64decode(b64))).size)
        return (True, 77)
    monkeypatch.setattr(classify, "judge_blur", fake)
    row = scanner.next_blur(conn)
    assert row["rel"] == "2026/01/img2.jpg"  # the only candidate: the others already have verdicts
    assert scanner.judge_one(conn, None, "t", row) == 1
    assert max(seen[0]) == config.RECHECK_PX
    got = conn.execute("SELECT blurry, blur_conf, blur_recheck FROM files WHERE id=?", (row["id"],)).fetchone()
    assert tuple(got) == (1, 77, 0) and scanner.next_blur(conn) is None


def test_migration_adds_sharpness_to_an_old_database(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript(db.SCHEMA)  # the schema as first deployed, which has no sharpness column
    assert "sharpness" not in {r[1] for r in old.execute("PRAGMA table_info(files)")}
    old.execute("INSERT INTO files (rel, size, mtime) VALUES ('a.jpg', 1, 'x')")
    old.commit()
    old.close()
    conn = db.connect(path)
    db.connect(path)  # a second process opening it must not trip over the first
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
    assert {"sharpness", "blurry", "blur_conf", "blur_recheck", "clutter", "clutter_conf", "clutter_reason"} <= cols
    assert tuple(conn.execute("SELECT sharpness, blurry, blur_conf FROM files").fetchone()) == (None, None, None)


def test_next_clutter_asks_cameraless_first_and_blurry_last(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    conn.execute("UPDATE files SET camera='Apple iPhone 12' WHERE status='done'")
    conn.execute("UPDATE files SET camera=NULL WHERE rel='2026/01/img2.jpg'")
    conn.execute("UPDATE files SET blurry=1 WHERE rel='2026/01/img0.jpg'")
    conn.execute("UPDATE files SET kept=1 WHERE rel='2026/01/img3.jpg'")
    order = []
    while (row := scanner.next_clutter(conn)) is not None:
        order.append(row["rel"].rsplit("/", 1)[1])
        conn.execute("UPDATE files SET clutter=0 WHERE id=?", (row["id"],))
    assert order == ["img2.jpg", "img1.jpg", "img0.jpg"]  # kept never asked


def test_clutter_one_records_verdict_reason_and_parks_bad_answers(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    ids = [r["id"] for r in conn.execute("SELECT id FROM files WHERE status='done' ORDER BY id")]
    answers = iter([(True, 88, "bare floor"), classify.BadVerdict("x"), classify.BadVerdict("x"),
                    classify.BadVerdict("x"), (False, 70, "a child")])
    modes = []

    def fake(*_a, sample=False):
        modes.append(sample)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(classify, "judge_clutter", fake)

    def row(i):
        return conn.execute("SELECT id, rel, clutter_recheck FROM files WHERE id=?", (i,)).fetchone()
    assert scanner.clutter_one(conn, None, "t", row(ids[0])) == 1
    assert scanner.clutter_one(conn, None, "t", row(ids[1])) == -1  # greedy, then sampled: both bad
    assert scanner.clutter_one(conn, None, "t", row(ids[2])) == 0   # bad, then the sampled retry answers
    assert modes == [False, False, True, False, True]
    os.remove(scanner.thumb_path(ids[3]))
    assert scanner.clutter_one(conn, None, "t", row(ids[3])) == -1  # no thumbnail: no model call
    got = {r["id"]: tuple(r)[1:] for r in conn.execute("SELECT id, clutter, clutter_conf, clutter_reason FROM files")}
    assert got[ids[0]] == (1, 88, "bare floor") and got[ids[1]] == (-1, None, None)
    assert got[ids[2]] == (0, 70, "a child") and got[ids[3]] == (-1, None, None)


def test_side_turns_recheck_first_then_clutter_then_blur(lib, monkeypatch):
    conn = _labelled(lib, monkeypatch)
    assert scanner.next_side(conn)[0] == "clutter"
    conn.execute("UPDATE files SET blur_recheck=1 WHERE rel='2026/01/img1.jpg'")
    kind, row = scanner.next_side(conn)
    assert kind == "blur" and row["rel"] == "2026/01/img1.jpg"
    conn.execute("UPDATE files SET blur_recheck=0, clutter=0")
    assert scanner.next_side(conn)[0] == "blur"
    # a photo already called clutter is in Clutter whatever its blur, so it is not asked
    conn.execute("UPDATE files SET clutter=1")
    assert scanner.next_blur(conn) is None and scanner.next_side(conn) is None


def test_clutter_recheck_reads_the_original_larger_and_jumps_the_queue(lib, monkeypatch):
    big = lib / "pics" / "2026" / "01" / "img2.jpg"
    Image.new("RGB", (2000, 1500), (10, 200, 30)).save(big)
    conn = _labelled(lib, monkeypatch)
    conn.execute("UPDATE files SET camera='Apple'")  # so the queue order would not pick img2 first
    conn.execute("UPDATE files SET clutter=0, clutter_conf=90, clutter_reason='a person' "
                 "WHERE rel != '2026/01/img0.jpg'")
    conn.execute("UPDATE files SET clutter_recheck=1, blur_recheck=1 WHERE rel='2026/01/img2.jpg'")
    seen = []

    def fake(_h, _u, _t, _m, b64, sample=False):
        import base64
        import io
        seen.append(Image.open(io.BytesIO(base64.b64decode(b64))).size)
        return (True, 88, "a laptop screen")
    monkeypatch.setattr(classify, "judge_clutter", fake)
    kind, row = scanner.next_side(conn)
    assert (kind, row["rel"]) == ("clutter", "2026/01/img2.jpg")  # ahead of unasked img0 and of the blur recheck
    assert scanner.clutter_one(conn, None, "t", row) == 1
    assert max(seen[0]) == config.RECHECK_PX
    got = conn.execute("SELECT clutter, clutter_conf, clutter_reason, clutter_recheck FROM files WHERE id=?",
                       (row["id"],)).fetchone()
    assert tuple(got) == (1, 88, "a laptop screen", 0)
    kind, row = scanner.next_side(conn)
    assert (kind, row["rel"]) == ("blur", "2026/01/img2.jpg")  # the blur recheck is still honoured


def test_service_hashes_and_links_what_it_labels(lib, monkeypatch):
    monkeypatch.setattr(classify, "classify", lambda *a, **k: ("photo", 95, "x"))
    monkeypatch.setattr(classify, "judge_blur", lambda *a, **k: (False, 90))
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: (False, 90, "x"))
    monkeypatch.setattr(scanner.time, "sleep", lambda s: setattr(scanner, "_stop", s == 60))
    scanner.run()
    conn = db.connect(config.DB_PATH)
    done = conn.execute("SELECT dhash, dup_linked FROM files WHERE status='done'").fetchall()
    assert len(done) == 4 and all(r["dhash"] and r["dup_linked"] == 1 for r in done)
    # the fixture's four frames are each one flat colour, which is one picture
    # as far as a difference hash can tell: every pair, once
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 6


# --------------------------------------------- fingerprinting, no model ----

def _until_idle(monkeypatch):
    """The service's 60 s idle sleep ends the test."""
    monkeypatch.setattr(scanner.time, "sleep", lambda s: setattr(scanner, "_stop", s == 60))


def test_model_none_fingerprints_everything_and_finds_duplicates(lib, monkeypatch):
    monkeypatch.setattr(config, "LABELLING", False)
    asked = []
    monkeypatch.setattr(classify, "classify", lambda *a, **k: asked.append(1) or ("photo", 90, "x"))
    _until_idle(monkeypatch)
    scanner.run()
    got = rows(config.DB_PATH)
    assert not asked  # no model was ever asked
    pics = {k: v for k, v in got.items() if k != "broken.jpg"}
    assert all(r["status"] == "pending" and r["dhash"] and r["sharpness"] is not None
               and r["width"] == 64 and r["category"] is None for r in pics.values())
    assert all(os.path.exists(scanner.thumb_path(r["id"])) for r in pics.values())
    assert got["broken.jpg"]["status"] == "error"  # unreadable, parked as before
    conn = db.connect(config.DB_PATH)
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 6  # four flat frames, every pair
    assert len(scanner.dupes.groups(conn)) == 1  # and the page sees them, unlabelled


def test_fingerprint_first_then_label_decodes_each_picture_once(lib, monkeypatch):
    decoded, order = [], []
    real = scanner.prepare

    def counting(file_id, rel):
        decoded.append(rel)
        order.append("print")
        return real(file_id, rel)
    monkeypatch.setattr(scanner, "prepare", counting)
    monkeypatch.setattr(classify, "classify", lambda *a, **k: order.append("label") or ("photo", 95, "x"))
    monkeypatch.setattr(classify, "judge_blur", lambda *a, **k: (False, 90))
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: (False, 90, "x"))
    _until_idle(monkeypatch)
    scanner.run()
    good = [r for r in decoded if r != "2026/01/broken.jpg"]
    assert sorted(good) == sorted(set(good)) and len(good) == 4  # one decode each; labelling read the stored thumb
    assert order.index("label") >= 4 + config.MAX_ATTEMPTS  # every picture printed before the first label
    assert all(r["status"] == "done" for k, r in rows(config.DB_PATH).items() if k != "broken.jpg")


def test_a_lost_thumbnail_is_decoded_again_for_labelling(lib, monkeypatch):
    monkeypatch.setattr(config, "LABELLING", False)
    _until_idle(monkeypatch)
    scanner.run()
    conn = db.connect(config.DB_PATH)
    row = scanner.next_pending(conn)
    os.remove(scanner.thumb_path(row["id"]))
    prepared = scanner.load_or_prepare(row)
    assert prepared[0] == "ok" and os.path.exists(scanner.thumb_path(row["id"]))
    with open(scanner.thumb_path(row["id"]), "rb") as f:
        assert scanner.load_or_prepare(row) == ("stored", f.read())


def test_a_refused_model_still_gets_every_duplicate(lib, monkeypatch):
    # scanner's own reference: other test files re-import photosweep, so a fresh
    # `from photosweep import remote` here can be a different module object
    monkeypatch.setattr(scanner.remote, "allowed", lambda url=None: False)
    asked = []
    monkeypatch.setattr(classify, "classify", lambda *a, **k: asked.append(1) or ("photo", 90, "x"))
    _until_idle(monkeypatch)
    scanner.run()
    conn = db.connect(config.DB_PATH)
    assert conn.execute("SELECT count(*) FROM files WHERE dhash IS NOT NULL").fetchone()[0] == 4
    assert conn.execute("SELECT count(*) FROM dup_pairs").fetchone()[0] == 6
    assert conn.execute("SELECT count(*) FROM files WHERE status='done'").fetchone()[0] == 0
    assert not asked
