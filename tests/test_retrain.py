"""The automatic retrain: the gate's arithmetic on hand-made answers, then the
whole loop (due, train in a child process, judge, swap, roll-back copy) on a
synthetic library whose classes a fresh head can learn and a blind one cannot."""
import json
import os

import numpy as np
import pytest

from photosweep import classifier, config, db, heads_gate, retrain, scanner, train_heads

DIM = classifier.DIM
OLD, MARK, NEW = "2026-01-01T00:00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00"


def blind_heads(path):
    """Heads that know nothing: every label equally likely, clutter and blur
    a coin toss for every picture. Any head that learned anything beats them."""
    z2 = np.zeros((DIM, 2), np.float32)
    np.savez(path, label_W=np.zeros((DIM, 5), np.float32), label_b=np.zeros(5, np.float32),
             label_scale=np.float32(1), label_classes=np.array(train_heads.LABELS),
             clutter_W=z2, clutter_b=np.zeros(2, np.float32), clutter_scale=np.float32(1),
             clutter_classes=np.array(["keeper", "clutter"]),
             blur_W=z2, blur_b=np.zeros(2, np.float32), blur_scale=np.float32(1),
             blur_classes=np.array(["sharp", "blurry"]))
    return str(path)


def row(**kw):
    base = {"scanned_at": OLD, "clutter_at": None, "blur_at": None, "kept": 0, "kept_at": None,
            "status": "done", "clutter": None, "deleted_at": None}
    return {**base, **kw}


def clutter_case(n, p_new, kept=(), asked=NEW):
    """n photos, alternately keeper and clutter, the new heads' clutter
    probability for each from p_new(i, y); label answers all old."""
    rows = [row(clutter_at=asked, blur_at=asked, clutter=i % 2) for i in range(n)]
    for i in kept:
        rows[i].update(kept=1, kept_at=NEW)
    y = np.array([0 if r["kept"] else r["clutter"] for r in rows], int)
    p = np.array([p_new(i, y[i]) for i in range(n)])
    idx = np.arange(n)
    oof = {"label": (idx, np.full(n, 4), np.ones(n), train_heads.LABELS, np.full((n, 5), 0.2)),
           "clutter": (idx, y, np.ones(n), ["keeper", "clutter"], np.stack([1 - p, p], 1)),
           "blur": (idx, y, np.ones(n), ["sharp", "blurry"], np.full((n, 2), 0.5))}
    return rows, np.zeros((n, DIM), np.float32), oof


# ---------------------------------------------------------------- gate ----

def test_decided_at_dates_each_answer_by_who_gave_it():
    assert heads_gate.decided_at(row(scanned_at="L"), "label") == "L"
    assert heads_gate.decided_at(row(kept=1, kept_at="K", clutter_at="C"), "clutter") == "K"
    assert heads_gate.decided_at(row(status="deleted", clutter=1, deleted_at="D", clutter_at="C"), "clutter") == "D"
    # deleted but not called clutter (a duplicate copy): the model's answer is what is learned
    assert heads_gate.decided_at(row(status="deleted", clutter=0, deleted_at="D", clutter_at="C"), "clutter") == "C"
    # found on a real library: a clutter re-ask must not make the old blur answer look new
    assert heads_gate.decided_at(row(clutter_at=NEW, blur_at=OLD), "blur") == OLD


def test_heads_that_learned_beat_blind_ones_on_fresh_answers(tmp_path):
    rows, X, oof = clutter_case(200, lambda i, y: 0.9 if y else 0.1)
    v = heads_gate.judge(rows, X, oof, blind_heads(tmp_path / "h.npz"), MARK, 0.95)
    assert v["swap"] and v["reasons"] == []
    q = v["questions"]["clutter"]
    assert q["verdict"] == "better" and q["fresh"] == 200 and q["new_loss"] < q["old_loss"]
    assert v["questions"]["label"]["verdict"] == "not judged"  # its answers are all old


def test_heads_that_got_worse_are_refused(tmp_path):
    rows, X, oof = clutter_case(200, lambda i, y: 0.1 if y else 0.9)  # confidently wrong
    v = heads_gate.judge(rows, X, oof, blind_heads(tmp_path / "h.npz"), MARK, 0.95)
    assert not v["swap"] and v["questions"]["clutter"]["verdict"] == "worse"
    assert "log-loss" in v["reasons"][0]


def test_heads_that_would_file_more_fresh_keeps_as_clutter_are_refused(tmp_path):
    # better on the whole, but sure that one of the owner's new keeps is clutter:
    # the blind heads file none at the trust level
    rows, X, oof = clutter_case(200, lambda i, y: 0.97 if (y or i == 0) else 0.03, kept=(0,))
    v = heads_gate.judge(rows, X, oof, blind_heads(tmp_path / "h.npz"), MARK, 0.95)
    q = v["questions"]["clutter"]
    assert q["new_loss"] < q["old_loss"]
    assert (q["kept"], q["kept_flagged_new"], q["kept_flagged_old"]) == (1, 1, 0)
    assert not v["swap"] and "new keeps" in v["reasons"][0]


def test_answers_the_heads_in_use_may_have_seen_are_not_judged(tmp_path):
    rows, X, oof = clutter_case(200, lambda i, y: 0.9 if y else 0.1, asked=OLD)
    v = heads_gate.judge(rows, X, oof, blind_heads(tmp_path / "h.npz"), MARK, 0.95)
    assert not v["swap"] and v["questions"]["clutter"] == {"fresh": 0, "verdict": "not judged"}
    assert "too few new answers" in v["reasons"][0]


def test_a_question_the_new_heads_dropped_is_refused(tmp_path):
    rows, X, oof = clutter_case(200, lambda i, y: 0.9 if y else 0.1)
    del oof["blur"]  # train_heads skips a question with too few answers
    v = heads_gate.judge(rows, X, oof, blind_heads(tmp_path / "h.npz"), MARK, 0.95)
    assert not v["swap"] and v["questions"]["blur"] == {"verdict": "missing"}


# ---------------------------------------------------------------- loop ----

def separable_library(path, when=NEW):
    """300 pictures in five separable label clusters; the 180 photos split
    along one direction for clutter and another for blur. Every answer was
    given at `when`, by the vision model."""
    conn = db.connect(str(path))
    rng = np.random.default_rng(7)
    centres = {c: rng.standard_normal(DIM) for c in train_heads.LABELS}
    cl_dir, bl_dir = rng.standard_normal(DIM), rng.standard_normal(DIM)
    n = 0
    for c in train_heads.LABELS:
        for k in range(180 if c == "photo" else 30):
            n += 1
            cl, bl = (k % 2, (k // 2) % 2) if c == "photo" else (None, None)
            v = centres[c] + 0.3 * rng.standard_normal(DIM) + 2.0 * (cl or 0) * cl_dir + 2.0 * (bl or 0) * bl_dir
            conn.execute("INSERT INTO files (rel, size, mtime, status, category, clutter, blurry, label_src, "
                         "clutter_src, blur_src, scanned_at, clutter_at, blur_at, embedding) VALUES (?,1,'x','done',?,?,?,"
                         "'model',?,?,?,?,?,?)",
                         (f"{n}.jpg", c, cl, bl, "model" if cl is not None else None,
                          "model" if bl is not None else None, when, when if cl is not None else None,
                          when if bl is not None else None,
                          classifier.pack(v / np.linalg.norm(v))))
    return conn


def test_training_reads_a_library_the_current_code_has_not_migrated(tmp_path):
    # found on a real library: train_heads opens read-only, so the column the
    # scanner would have added on start is not there yet
    conn = separable_library(tmp_path / "old.db")
    conn.execute("ALTER TABLE files DROP COLUMN kept_at")
    conn.close()
    rows, X, _ = train_heads.load(str(tmp_path / "old.db"))
    assert len(rows) == 300 and all(r["kept_at"] is None for r in rows)


@pytest.fixture
def loop(tmp_path, monkeypatch):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    own = tmp_path / "data" / "heads.npz"
    own.parent.mkdir()
    blind_heads(own)
    monkeypatch.setattr(config, "CLASSIFIER", "siglip")
    monkeypatch.setattr(config, "CLASSIFIER_MODEL", str(model))
    monkeypatch.setattr(config, "CLASSIFIER_HEADS", str(own))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(config, "AUTO_RETRAIN", True)
    monkeypatch.setattr(config, "RETRAIN_MIN_NEW", 200)
    monkeypatch.setattr(config, "RETRAIN_EVERY_S", 3600)
    return own


def test_the_first_look_only_sets_the_mark(loop):
    conn = separable_library(config.DB_PATH)
    assert retrain.due(conn) is False
    assert db.get_setting(conn, retrain.MARK)
    # answers given before the mark are not new: nothing to retrain on yet
    assert retrain.due(conn) is False


def test_new_answers_count_what_the_heads_can_learn_from(loop):
    conn = separable_library(config.DB_PATH)
    # 300 labels, 180 clutter answers and 180 blur answers, all at NEW
    assert retrain.new_answers(conn, MARK) == 660
    assert retrain.new_answers(conn, NEW) == 0
    conn.execute("UPDATE files SET label_src='classifier' WHERE id <= 10")  # its own answers never count
    conn.execute("UPDATE files SET kept=1, kept_at='2026-03-02T00:00:00+00:00' WHERE id=300")
    assert retrain.new_answers(conn, NEW) == 1


def test_off_or_too_soon_is_never_due(loop, monkeypatch):
    conn = separable_library(config.DB_PATH)
    db.set_setting(conn, retrain.MARK, MARK)
    assert retrain.due(conn) is True
    db.set_setting(conn, retrain.TRIED, db.now())
    assert retrain.due(conn) is False  # tried within RETRAIN_EVERY_S
    db.set_setting(conn, retrain.TRIED, OLD)
    monkeypatch.setattr(config, "RETRAIN_MIN_NEW", 10_000)
    assert retrain.due(conn) is False  # not enough new answers
    monkeypatch.setattr(config, "RETRAIN_MIN_NEW", 200)
    monkeypatch.setattr(config, "AUTO_RETRAIN", False)
    assert retrain.due(conn) is False


def test_better_heads_replace_the_old_and_keep_a_copy(loop):
    conn = separable_library(config.DB_PATH)
    db.set_setting(conn, retrain.MARK, MARK)
    old = loop.read_bytes()
    assert retrain.run(conn, lambda: False) is True
    assert (loop.parent / "heads.npz.prev.npz").read_bytes() == old
    assert loop.read_bytes() != old and not os.path.exists(str(loop) + ".candidate.npz")
    assert set(classifier.Heads(str(loop)).q) == {"label", "clutter", "blur"}
    last = json.loads(db.get_setting(conn, retrain.LAST))
    assert last["outcome"] == "replaced"
    assert {q["verdict"] for q in last["questions"].values()} == {"better"}
    # the mark moves to when the new heads read the library
    assert db.get_setting(conn, retrain.MARK) > MARK


def test_with_nothing_new_the_heads_in_use_stay(loop):
    conn = separable_library(config.DB_PATH, when=OLD)
    db.set_setting(conn, retrain.MARK, MARK)
    old = loop.read_bytes()
    assert retrain.run(conn, lambda: False) is False
    assert loop.read_bytes() == old and not (loop.parent / "heads.npz.prev.npz").exists()
    last = json.loads(db.get_setting(conn, retrain.LAST))
    assert last["outcome"] == "kept" and "too few new answers" in last["reasons"][0]
    assert db.get_setting(conn, retrain.MARK) == MARK  # judged on more next time


def test_a_training_that_fails_is_recorded_and_changes_nothing(loop):
    conn = db.connect(config.DB_PATH)  # an empty library: train_heads exits non-zero
    db.set_setting(conn, retrain.MARK, MARK)
    old = loop.read_bytes()
    assert retrain.run(conn, lambda: False) is False
    assert loop.read_bytes() == old
    assert json.loads(db.get_setting(conn, retrain.LAST))["outcome"] == "failed"


def test_stopping_abandons_the_training(loop):
    conn = separable_library(config.DB_PATH)
    db.set_setting(conn, retrain.MARK, MARK)
    old = loop.read_bytes()
    assert retrain.run(conn, lambda: True) is False
    assert loop.read_bytes() == old and not os.path.exists(str(loop) + ".candidate.npz")
    assert json.loads(db.get_setting(conn, retrain.LAST))["outcome"] == "abandoned"


def test_reload_heads_swaps_the_heads_and_keeps_the_model(loop, monkeypatch):
    class Session:
        def get_outputs(self):
            return []
    clf = classifier.Classifier("unused", str(loop), session=Session())
    monkeypatch.setattr(classifier, "_loaded", clf)
    session = clf.session
    z = np.load(str(loop))
    arrays = {k: z[k] for k in z.files}
    arrays["label_b"] = np.array([0, 0, 0, 0, 5], np.float32)  # now sure of photo
    np.savez(str(loop), **arrays)
    assert clf.heads.verdicts(np.zeros(DIM, np.float32))["label_p"] < 0.5
    classifier.reload_heads()
    assert classifier._loaded is clf and clf.session is session
    assert clf.heads.verdicts(np.zeros(DIM, np.float32))["label"] == "photo"
    assert clf.heads_file == str(loop)


def test_the_scanner_reloads_and_rescores_after_a_swap_and_survives_a_failure(loop, monkeypatch):
    conn = separable_library(config.DB_PATH)
    calls = []
    monkeypatch.setattr(retrain, "due", lambda c: True)
    monkeypatch.setattr(retrain, "run", lambda c, stopping: True)
    monkeypatch.setattr(classifier, "reload_heads", lambda: calls.append("reload"))
    monkeypatch.setattr(scanner, "rescore", lambda c: calls.append("rescore"))
    assert scanner._retrain_if_due(conn) is True and calls == ["reload", "rescore"]

    def boom(c):
        raise RuntimeError("disk full")
    monkeypatch.setattr(retrain, "due", boom)
    assert scanner._retrain_if_due(conn) is False  # logged, and the scan goes on
