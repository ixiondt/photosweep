"""Blurry's exposure filter: the measurement on drawn pictures, the filter and
its counts through the API, and the backfill for pictures from before it."""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from photosweep import db, exposure, imaging

OP = "owner@example.com"


def test_exposure_reads_dark_bright_and_ordinary_pictures():
    black = Image.new("RGB", (512, 384), (5, 5, 5))
    mean, dark, bright = imaging.exposure(black)
    assert mean < 10 and dark == 1.0 and bright == 0.0
    white = Image.new("RGB", (512, 384), (252, 252, 252))
    mean, dark, bright = imaging.exposure(white)
    assert mean > 245 and dark == 0.0 and bright == 1.0
    # half black, half mid grey: the average alone says dim, the shares say why
    half = Image.new("RGB", (512, 384), (128, 128, 128))
    ImageDraw.Draw(half).rectangle((0, 0, 255, 383), fill=(0, 0, 0))
    mean, dark, bright = imaging.exposure(half)
    assert 55 < mean < 75 and 0.45 < dark < 0.55 and bright == 0.0
    assert imaging.exposure(Image.new("L", (0, 0))) is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PICS_ROOT", str(tmp_path / "pics"))
    monkeypatch.setenv("OPERATORS", OP)
    monkeypatch.setenv("AUTH_MODE", "header")
    for name in [m for m in sys.modules if m.startswith("photosweep")]:
        del sys.modules[name]
    web = importlib.import_module("photosweep.web")
    from photosweep import db as fresh_db
    c = fresh_db.connect(web.config.DB_PATH)
    # all in Blurry (the model said blurry); the defaults cut at 50 / 0.5 dark, 190 / 0.25 bright
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, blurry, blur_conf, camera, "
        "luma, luma_dark, luma_bright) VALUES (?,1,'x','done','photo',95,1,90,?,?,?,?)",
        [("night.jpg", "Apple", 30.0, 0.7, 0.0),       # dark by both measures
         ("cellar.jpg", None, 70.0, 0.55, 0.0),        # dark: over half crushed
         ("lamp.jpg", "Apple", 40.0, 0.6, 0.3),        # both: dark wins
         ("snow.jpg", "Apple", 205.0, 0.0, 0.2),       # bright: high average
         ("window.jpg", None, 150.0, 0.0, 0.3),        # bright: a quarter blown
         ("park.jpg", "Apple", 110.0, 0.05, 0.02),     # well exposed
         ("unmeasured.jpg", "Apple", None, None, None)])  # counts as well exposed
    return TestClient(web.app)


def items(client, **q):
    query = "&".join(f"{k}={v}" for k, v in {"tab": "blurry", **q}.items())
    r = client.get(f"/api/items?{query}", headers={"Tailscale-User-Login": OP})
    assert r.status_code == 200, r.text
    return r.json()


def names(body):
    return {i["rel"] for i in body["items"]}


def test_each_photo_falls_in_exactly_one_exposure_and_the_counts_say_so(client):
    groups = {k: names(items(client, exposure=k)) for k in ("dark", "bright", "normal")}
    assert groups == {"dark": {"night.jpg", "cellar.jpg", "lamp.jpg"},
                      "bright": {"snow.jpg", "window.jpg"},
                      "normal": {"park.jpg", "unmeasured.jpg"}}
    body = items(client)
    assert len(body["items"]) == 7 and body["exposures"] == {"all": 7, "dark": 3, "bright": 2, "normal": 2}


def test_exposure_and_source_narrow_together_and_count_each_other(client):
    body = items(client, exposure="dark", source="camera")
    assert names(body) == {"night.jpg", "lamp.jpg"} and body["total"] == 2
    # the source labels count within the exposure, and the exposure labels within the source
    assert body["sources"] == {"all": 3, "camera": 2, "nocamera": 1, "messaging": 0}
    assert body["exposures"] == {"all": 5, "dark": 2, "bright": 1, "normal": 2}


def test_an_unknown_exposure_is_refused(client):
    r = client.get("/api/items?tab=blurry&exposure=x%27%20OR%201=1--", headers={"Tailscale-User-Login": OP})
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


def test_backfill_measures_old_rows_from_their_thumbnails(tmp_path):
    conn = db.connect(str(tmp_path / "b.db"))
    conn.executemany("INSERT INTO files (id, rel, size, mtime, status, dhash) VALUES (?, ?, 1, 'x', ?, ?)",
                     [(1, "a.jpg", "done", "00"), (2, "b.jpg", "pending", "00"), (3, "c.jpg", "done", "00"),
                      (4, "d.jpg", "pending", None), (5, "e.jpg", "deleted", "00")])
    thumbs = tmp_path / "t"
    thumbs.mkdir()
    Image.new("RGB", (64, 48), (5, 5, 5)).save(thumbs / "1.jpg")
    Image.new("RGB", (64, 48), (250, 250, 250)).save(thumbs / "2.jpg")
    for i in (4, 5):
        Image.new("RGB", (64, 48), (120, 120, 120)).save(thumbs / f"{i}.jpg")
    # 3 has no thumbnail: skipped, left NULL; 4 is not fingerprinted; 5 is deleted
    assert exposure.backfill(conn, lambda i: str(thumbs / f"{i}.jpg"), lambda: False, batch=1) == 2
    got = {r["id"]: r["luma"] for r in conn.execute("SELECT id, luma FROM files")}
    assert got[1] < 15 and got[2] > 240 and got[3] is None and got[4] is None and got[5] is None
    assert exposure.backfill(conn, lambda i: str(thumbs / f"{i}.jpg"), lambda: False) == 0  # nothing left
