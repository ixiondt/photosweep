"""Stage 2, the image classifier: preprocessing, heads, the scanner's use of
it, and training. The ONNX session is faked (a fixed projection of the
pixels), so these run without the 100 MB model; the real one is exercised by
scripts/smoke.sh inside the image."""
import os

import numpy as np
import pytest
from PIL import Image

from photosweep import classifier, classify, config, db, scanner, train_heads

DIM = classifier.DIM


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for onnxruntime: the embedding is the image's mean colour
    spread over DIM numbers, so a red picture and a blue one land far apart."""

    def get_outputs(self):
        return [FakeOutput("last_hidden_state"), FakeOutput("pooler_output")]

    def run(self, _names, feeds):
        x = feeds["pixel_values"]
        assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
        rgb = x.mean(axis=(2, 3))[0]
        v = np.zeros(DIM, np.float32)
        v[:3] = rgb + 1.5
        return [np.zeros((1, 196, DIM), np.float32), v[None]]


def heads_file(path, label_W=None):
    """Heads that call red pictures screenshots and everything else a photo,
    sure of both; blue photos clutter, surely; and every photo a little more
    sharp than blurry (about 0.4), which is never sure."""
    W = np.zeros((DIM, 5), np.float32)
    W[0, 0], W[2, 4], W[1, 4] = 4.0, 4.0, 4.0
    cW = np.zeros((DIM, 2), np.float32)
    cW[2, 1], cW[0, 0], cW[1, 0] = 4.0, 4.0, 4.0
    bW = np.zeros((DIM, 2), np.float32)
    bW[:3, 0] = 0.01
    np.savez(path, label_W=label_W if label_W is not None else W, label_b=np.zeros(5, np.float32),
             label_scale=np.float32(10), label_classes=np.array(train_heads.LABELS),
             clutter_W=cW, clutter_b=np.zeros(2, np.float32), clutter_scale=np.float32(10),
             clutter_classes=np.array(["keeper", "clutter"]),
             blur_W=bW, blur_b=np.zeros(2, np.float32), blur_scale=np.float32(10),
             blur_classes=np.array(["sharp", "blurry"]))
    return str(path)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    pics = tmp_path / "pics" / "2026"
    pics.mkdir(parents=True)
    Image.new("RGB", (64, 48), (250, 10, 10)).save(pics / "red.jpg")
    Image.new("RGB", (64, 48), (10, 10, 250)).save(pics / "blue.jpg")
    Image.new("RGB", (64, 48), (10, 250, 10)).save(pics / "green.jpg")
    monkeypatch.setattr(config, "PICS_ROOT", str(tmp_path / "pics"))
    monkeypatch.setattr(config, "THUMB_DIR", str(tmp_path / "thumbs"))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(config, "ROUTER_TOKEN_FILE", "")
    monkeypatch.setattr(scanner, "_stop", False)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(config, "CLASSIFIER", "siglip")
    monkeypatch.setattr(config, "CLASSIFIER_MODEL", str(model))
    hf = heads_file(tmp_path / "heads.npz")
    monkeypatch.setattr(classifier, "HEADS_FILE", hf)
    monkeypatch.setattr(classifier, "_loaded", classifier.Classifier(str(model), hf, session=FakeSession()))
    return tmp_path


def by_name(conn):
    return {r["rel"].rsplit("/", 1)[1]: dict(r) for r in conn.execute("SELECT * FROM files")}


def until_idle(monkeypatch):
    monkeypatch.setattr(scanner.time, "sleep", lambda s: setattr(scanner, "_stop", s == 60))


# ------------------------------------------------------------- module ----

def test_pixels_match_siglip_preprocessing():
    x = classifier.pixels(Image.new("RGB", (300, 100), (255, 0, 128)))
    assert x.shape == (1, 3, 224, 224)  # squashed, not cropped
    assert np.allclose(x[0, 0], 1.0) and np.allclose(x[0, 1], -1.0) and abs(x[0, 2].mean() - 0.0039) < 0.01


def test_pack_round_trips_in_a_kilobyte_and_a_half():
    v = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    blob = classifier.pack(v)
    assert len(blob) == DIM * 2 and np.allclose(classifier.unpack(blob), v, atol=1e-3)


def test_verdicts_read_every_head(tmp_path):
    clf = classifier.Classifier("unused", heads_file(tmp_path / "h.npz"), session=FakeSession())
    red = clf.verdicts(clf.embed(Image.new("RGB", (32, 32), (255, 0, 0))))
    blue = clf.verdicts(clf.embed(Image.new("RGB", (32, 32), (0, 0, 255))))
    assert red["label"] == "screenshot" and blue["label"] == "photo"
    assert blue["clutter_p"] > 0.9 and red["clutter_p"] < 0.1 and 0.3 < blue["blur_p"] < 0.5
    assert 0 < red["label_p"] <= 1


def test_no_heads_means_embeddings_only(tmp_path):
    clf = classifier.Classifier("unused", str(tmp_path / "absent.npz"), session=FakeSession())
    assert clf.heads is None and clf.verdicts(np.ones(DIM, np.float32)) is None


def test_switched_off_or_missing_model_loads_nothing(tmp_path, monkeypatch):
    classifier.reset()
    monkeypatch.setattr(config, "CLASSIFIER", "siglip")
    monkeypatch.setattr(config, "CLASSIFIER_MODEL", str(tmp_path / "nope.onnx"))
    assert classifier.enabled() is False and classifier.load() is None
    monkeypatch.setattr(config, "CLASSIFIER", "none")
    assert classifier.load() is None


def test_a_model_that_will_not_load_is_logged_once_and_left_off(tmp_path, monkeypatch):
    classifier.reset()
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr(config, "CLASSIFIER", "siglip")
    monkeypatch.setattr(config, "CLASSIFIER_MODEL", str(bad))
    assert classifier.load() is None and classifier._failed


# ------------------------------------------------------------ scanner ----

def test_with_no_vision_model_the_classifier_sorts_everything(lib, monkeypatch):
    monkeypatch.setattr(config, "LABELLING", False)
    until_idle(monkeypatch)
    scanner.run()
    got = by_name(db.connect(config.DB_PATH))
    assert got["red.jpg"]["category"] == "screenshot" and got["red.jpg"]["status"] == "done"
    assert got["blue.jpg"]["category"] == "photo" and got["green.jpg"]["category"] == "photo"
    assert all(r["label_src"] == "classifier" and r["reason"] == "image classifier" for r in got.values())
    # a photo's clutter and blur answers too, whatever their confidence, with nobody else to ask
    assert got["blue.jpg"]["clutter"] == 1 and got["blue.jpg"]["clutter_src"] == "classifier"
    assert got["green.jpg"]["clutter"] == 0 and got["green.jpg"]["blurry"] == 0
    assert got["red.jpg"]["clutter"] is None  # not a photo: no photo questions
    assert all(len(r["embedding"]) == DIM * 2 and r["dhash"] for r in got.values())


def test_with_a_vision_model_only_sure_calls_are_final(lib, monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER_TRUST", 1.01)  # nothing is that sure
    asked = []

    def model(_h, _u, _t, _m, b64, **_k):
        asked.append(b64)
        return ("meme", 80, "the model")
    monkeypatch.setattr(classify, "classify", model)
    monkeypatch.setattr(classify, "judge_blur", lambda *a, **k: (False, 90))
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: (False, 90, "x"))
    until_idle(monkeypatch)
    scanner.run()
    got = by_name(db.connect(config.DB_PATH))
    assert len(asked) == 3  # all three went to the model
    assert all(r["category"] == "meme" and r["label_src"] == "model" for r in got.values())
    assert all(r["cpu_label"] is not None and r["cpu_label_p"] is not None for r in got.values())


def test_sure_calls_skip_the_vision_model_and_its_photo_questions(lib, monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER_TRUST", 0.9)
    asked = {"label": 0, "clutter": 0, "blur": 0}
    monkeypatch.setattr(classify, "classify", lambda *a, **k: asked.__setitem__("label", asked["label"] + 1)
                        or ("photo", 90, "x"))
    monkeypatch.setattr(classify, "judge_blur", lambda *a, **k: asked.__setitem__("blur", asked["blur"] + 1)
                        or (False, 90))
    monkeypatch.setattr(classify, "judge_clutter", lambda *a, **k: asked.__setitem__("clutter", asked["clutter"] + 1)
                        or (False, 90, "x"))
    until_idle(monkeypatch)
    scanner.run()
    got = by_name(db.connect(config.DB_PATH))
    p = {k: v["cpu_label_p"] for k, v in got.items()}
    assert all(v > 0.9 for v in p.values()), p  # the fake heads are sure of all three
    assert asked["label"] == 0
    assert got["blue.jpg"]["clutter"] == 1 and got["blue.jpg"]["clutter_src"] == "classifier"
    assert asked["clutter"] == 0  # sure either way for both photos
    # blur is never sure in the fake heads, so the model is asked; not of blue,
    # which is clutter already (next_blur skips those)
    assert asked["blur"] == 1 and got["green.jpg"]["blur_src"] == "model"


def test_a_model_label_takes_the_classifiers_sure_photo_answers(lib, monkeypatch):
    conn = db.connect(config.DB_PATH)
    scanner.walk(conn)
    row = conn.execute("SELECT id, rel, attempts, dhash FROM files WHERE rel LIKE '%blue%'").fetchone()
    monkeypatch.setattr(config, "CLASSIFIER_TRUST", 0.9)
    conn.execute("UPDATE files SET dhash='x', cpu_clutter_p=0.97, cpu_blur_p=0.5 WHERE id=?", (row["id"],))
    os.makedirs(os.path.dirname(scanner.thumb_path(row["id"])), exist_ok=True)
    Image.new("RGB", (32, 32)).save(scanner.thumb_path(row["id"]))
    monkeypatch.setattr(classify, "classify", lambda *a, **k: ("photo", 90, "model says photo"))
    row = conn.execute("SELECT id, rel, attempts, dhash FROM files WHERE id=?", (row["id"],)).fetchone()
    scanner.label_one(conn, None, None, row)
    r = conn.execute("SELECT * FROM files WHERE id=?", (row["id"],)).fetchone()
    assert (r["category"], r["label_src"]) == ("photo", "model")
    assert (r["clutter"], r["clutter_src"], r["clutter_conf"]) == (1, "classifier", 97)
    assert r["blurry"] is None  # 0.5 is not sure: the model will be asked


def test_backfill_embeds_old_rows_and_sorts_only_the_pending(lib, monkeypatch):
    monkeypatch.setattr(config, "LABELLING", False)
    conn = db.connect(config.DB_PATH)
    scanner.walk(conn)
    ids = {r["rel"].rsplit("/", 1)[1]: r["id"] for r in conn.execute("SELECT id, rel FROM files")}
    colours = {"red.jpg": (250, 10, 10), "blue.jpg": (10, 10, 250), "green.jpg": (10, 250, 10)}
    for name, i in ids.items():
        os.makedirs(os.path.dirname(scanner.thumb_path(i)), exist_ok=True)
        Image.new("RGB", (32, 32), colours[name]).save(scanner.thumb_path(i))
    # red was labelled by the model long ago; blue was deleted; green is fingerprinted, unsorted
    conn.execute("UPDATE files SET status='done', category='meme', label_src=NULL, dhash='a' WHERE id=?", (ids["red.jpg"],))
    conn.execute("UPDATE files SET status='deleted', category='photo', dhash='b' WHERE id=?", (ids["blue.jpg"],))
    conn.execute("UPDATE files SET dhash='c' WHERE id=?", (ids["green.jpg"],))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        assert scanner.embed_backfill_batch(conn, pool) == 3
        assert scanner.embed_backfill_batch(conn, pool) == 0  # each row once
    got = by_name(conn)
    assert got["red.jpg"]["category"] == "meme" and got["red.jpg"]["cpu_label"] == "screenshot"  # the label stands
    assert got["blue.jpg"]["status"] == "deleted" and len(got["blue.jpg"]["embedding"]) == DIM * 2
    assert got["green.jpg"]["status"] == "done" and got["green.jpg"]["label_src"] == "classifier"


def test_backfill_marks_a_missing_thumbnail_tried(lib):
    conn = db.connect(config.DB_PATH)
    scanner.walk(conn)
    conn.execute("UPDATE files SET status='done', category='photo'")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as pool:
        assert scanner.embed_backfill_batch(conn, pool) == 3
        assert scanner.embed_backfill_batch(conn, pool) == 0
    assert all(r["embedding"] == b"" for r in by_name(conn).values())


# ----------------------------------------------------------- training ----

def test_train_heads_learns_separable_classes_and_honours_the_owner(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    conn = db.connect(str(path))
    rng = np.random.default_rng(3)
    centres = {c: rng.standard_normal(DIM) for c in train_heads.LABELS}
    clutter_dir = rng.standard_normal(DIM)

    def vec(c, clutter=0.0):
        v = centres[c] + 0.3 * rng.standard_normal(DIM) + clutter * clutter_dir
        return classifier.pack(v / np.linalg.norm(v))
    n = 0
    for c in train_heads.LABELS:
        for k in range(120 if c == "photo" else 60):
            n += 1
            is_clutter = c == "photo" and k % 3 == 0
            conn.execute("INSERT INTO files (rel, size, mtime, status, category, clutter, blurry, kept, embedding) "
                         "VALUES (?,?,?,?,?,?,?,?,?)",
                         (f"{n}.jpg", 1, "x", "done", c, int(is_clutter) if c == "photo" else None,
                          0 if c == "photo" else None, 0, vec(c, 2.0 if is_clutter else 0.0)))
    # the classifier's own answers are never learned from
    conn.execute("INSERT INTO files (rel, size, mtime, status, category, label_src, embedding) "
                 "VALUES ('own.jpg', 1, 'x', 'done', 'meme', 'classifier', ?)", (vec("photo"),))
    # the owner kept one the model called clutter: it trains as a keeper
    conn.execute("UPDATE files SET kept=1 WHERE rel='241.jpg'")  # the first photo, a clutter one
    rows, X = train_heads.load(str(path))
    label, clutter, blur = train_heads.sets(rows)
    assert len(label[2]) == n  # own.jpg left out
    kept_at = [j for j, i in enumerate(clutter[2]) if rows[i]["kept"]]
    assert clutter[3][kept_at[0]] == 0 and clutter[4][kept_at[0]] == train_heads.OWNER_WEIGHT
    out = tmp_path / "heads.npz"
    train_heads.main(["--db", str(path), "--out", str(out)])
    h = classifier.Heads(str(out))
    assert set(h.q) == {"label", "clutter"}  # blur skipped: one class only
    ok = 0
    for c in train_heads.LABELS:
        v = classifier.unpack(vec(c))
        ok += h.verdicts(v)["label"] == c
    assert ok == 5


def test_train_heads_refuses_an_empty_library(tmp_path):
    db.connect(str(tmp_path / "e.db"))
    with pytest.raises(SystemExit):
        train_heads.main(["--db", str(tmp_path / "e.db"), "--out", str(tmp_path / "h.npz"), "--dry-run"])
