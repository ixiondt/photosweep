"""The year filter: where a year comes from, the filter and its counts through
the API, and the backfill for pictures from before it."""
import datetime
import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photosweep import db, years

OP = "owner@example.com"


@pytest.mark.parametrize("rel, want", [
    ("2024/IMG_0001.jpg", 2024),                       # a year folder
    ("Photos/2019/July/a.jpg", 2019),                  # at any depth
    ("2018/IMG_20240304_101112.jpg", 2018),            # the folder wins over the name
    ("Camera/IMG_20240304_101112.jpg", 2024),          # a date in the name
    ("Camera/PXL_20230101123456789.jpg", 2023),        # digits run straight on
    ("Screenshots/Screenshot 2022-03-04 at 10.11.jpg", 2022),
    ("WhatsApp/IMG-20210709-WA0001.jpg", 2021),
    ("misc/IMG_1234.jpg", None),                       # a counter, not a date
    # a folder named for the day it was exported or backed up says nothing of when
    # its photos were taken; only a folder that is exactly a year counts
    ("Backup 2024-01-01/IMG_1234.jpg", None),
    ("Export 20240101/IMG_1234.jpg", None),
    ("misc/20241399.jpg", None),                       # month 13 is no date
    ("2400/a.jpg", None),                              # not a year a photo is from
    ("misc/scan_1899_12_01.jpg", None),
])
def test_year_from_the_path(rel, want):
    assert years.of_path(rel) == want


def _jpeg_with_date(path, when):
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = when
    img.save(path, exif=exif)


def test_year_from_exif_only_when_the_path_has_none(tmp_path):
    _jpeg_with_date(tmp_path / "a.jpg", "2016:05:04 10:11:12")
    Image.new("RGB", (8, 8)).save(tmp_path / "plain.jpg")
    (tmp_path / "broken.jpg").write_bytes(b"not a picture")
    assert years.of_picture("misc/a.jpg", str(tmp_path / "a.jpg")) == 2016
    assert years.of_picture("2020/a.jpg", str(tmp_path / "a.jpg")) == 2020  # the folder first
    assert years.of_picture("misc/plain.jpg", str(tmp_path / "plain.jpg")) == 0
    assert years.of_picture("misc/broken.jpg", str(tmp_path / "broken.jpg")) == 0
    assert years.of_picture("misc/gone.jpg", str(tmp_path / "gone.jpg")) == 0
    _jpeg_with_date(tmp_path / "zero.jpg", "0000:00:00 00:00:00")  # cameras with no clock set
    assert years.of_file(str(tmp_path / "zero.jpg")) is None
    _jpeg_with_date(tmp_path / "late.jpg", f"{datetime.date.today().year + 5}:01:01 00:00:00")
    assert years.of_file(str(tmp_path / "late.jpg")) is None


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
    c.executemany(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, camera, year) "
        "VALUES (?,1,'x','done','screenshot',95,?,?)",
        [("2024/a.png", "Apple", 2024), ("2024/b.png", None, 2024), ("2023/c.png", "Apple", 2023),
         ("misc/d.png", None, 0), ("misc/e.png", "Apple", None)])  # e: not looked at yet
    return TestClient(web.app)


def items(client, **q):
    query = "&".join(f"{k}={v}" for k, v in {"tab": "screenshot", **q}.items())
    r = client.get(f"/api/items?{query}", headers={"Tailscale-User-Login": OP})
    assert r.status_code == 200, r.text
    return r.json()


def names(body):
    return {i["rel"] for i in body["items"]}


def test_each_picture_falls_in_one_year_and_the_counts_say_so(client):
    body = items(client)
    assert len(body["items"]) == 5 and body["years"] == {"2024": 2, "2023": 1, "none": 2}
    assert names(items(client, year="2024")) == {"2024/a.png", "2024/b.png"}
    assert names(items(client, year="none")) == {"misc/d.png", "misc/e.png"}  # 0 and NULL alike
    assert items(client, year="2019")["items"] == []


def test_year_and_source_narrow_together_and_count_each_other(client):
    body = items(client, year="2024", source="camera")
    assert names(body) == {"2024/a.png"} and body["total"] == 1
    assert body["sources"] == {"all": 2, "camera": 1, "nocamera": 1, "messaging": 0}  # within 2024
    assert body["years"] == {"2024": 1, "2023": 1, "none": 1}                          # within Camera


@pytest.mark.parametrize("bad", ["x%27%20OR%201=1--", "24", "20245", "-2024"])
def test_an_unknown_year_is_refused(client, bad):
    r = client.get(f"/api/items?tab=screenshot&year={bad}", headers={"Tailscale-User-Login": OP})
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


def test_backfill_dates_old_rows_from_the_path_then_exif(tmp_path):
    root = tmp_path / "pics"
    (root / "misc").mkdir(parents=True)
    _jpeg_with_date(root / "misc" / "b.jpg", "2015:01:02 03:04:05")
    Image.new("RGB", (8, 8)).save(root / "misc" / "c.jpg")
    _jpeg_with_date(tmp_path / "outside.jpg", "2011:01:01 00:00:00")
    conn = db.connect(str(tmp_path / "b.db"))
    conn.executemany("INSERT INTO files (id, rel, size, mtime, status, dhash) VALUES (?, ?, 1, 'x', ?, ?)",
                     [(1, "2024/a.jpg", "done", "00"),      # from the folder, never opened
                      (2, "misc/b.jpg", "pending", "00"),   # from EXIF
                      (3, "misc/c.jpg", "done", "00"),      # nothing: 0, so it is not looked at again
                      (4, "2020/d.jpg", "pending", None),   # not fingerprinted: prepare() dates it
                      (5, "2020/e.jpg", "deleted", "00"),   # gone from the sweep
                      (6, "../outside.jpg", "done", "00")])  # outside the library: never opened
    assert years.backfill(conn, str(root), lambda: False, batch=2) == 4
    got = {r["id"]: r["year"] for r in conn.execute("SELECT id, year FROM files")}
    assert got == {1: 2024, 2: 2015, 3: 0, 4: None, 5: None, 6: 0}
    assert years.backfill(conn, str(root), lambda: False) == 0  # nothing left
    assert years.backfill(conn, str(root), lambda: True) == 0   # and it stops when asked
