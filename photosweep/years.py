"""The year filter: which year each picture belongs to, and the SQL that
narrows a tab to one.

The year comes from the path first: a folder named exactly for a year
(2024/...), else a date in the file name (IMG_20240304_..., Screenshot 2024-03-04 ...), and
only then the EXIF capture date, which costs opening the file. Path before
EXIF because the path is how the owner arranged the library and what Nextcloud
shows them, and because it is free: on a real 23,735-picture library every
picture sat under a year folder, and that folder agreed with the EXIF capture
year on 516 of 517 sampled pictures that had one, where the file's modified
time agreed on only 382 (a bulk upload had stamped thousands with its own date).
A date in any other folder name is ignored: "Backup 2024-01-01" is when the
folder was made, not when its photos were taken.

Stored in `year`: the year, 0 when nothing gave one (so the backfill does not
look again), NULL when not yet looked at. The filter treats 0 and NULL alike,
as No date.
"""
import datetime
import logging
import os
import re

from PIL import Image

from . import db

log = logging.getLogger("photosweep.years")

_FOLDER = re.compile(r"(?:19|20)\d\d")
# a date at the start of a run of digits: 20240304, 2024-03-04, 2024_03_04, 2024.03.04
_NAME = re.compile(r"(?<!\d)((?:19|20)\d\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])")
_EXIF_TAKEN = 36867      # DateTimeOriginal, in the Exif IFD
_EXIF_IFD = 0x8769
_EXIF_DATETIME = 306     # DateTime, in IFD0: the fallback, often the edit time


def _sane(y):
    """A year a photo could be from: not before 1900, not after next year."""
    return y if y and 1900 <= y <= datetime.date.today().year + 1 else None


def of_path(rel):
    """The year a relative path names, or None: a folder first, then the name."""
    *folders, name = rel.split("/")
    for seg in folders:
        if _FOLDER.fullmatch(seg) and _sane(int(seg)):
            return int(seg)
    m = _NAME.search(name)
    return _sane(int(m.group(1))) if m else None


def of_exif(exif):
    """The capture year in an Image.Exif, or None. Never raises."""
    if not exif:
        return None
    try:
        raw = exif.get_ifd(_EXIF_IFD).get(_EXIF_TAKEN) or exif.get(_EXIF_DATETIME)
    except Exception:  # corrupt EXIF is this file's problem, not the scan's
        return None
    s = str(raw or "").strip("\x00 ")
    return _sane(int(s[:4])) if s[:4].isdigit() else None


def of_file(full):
    """The capture year in the file at `full`, reading only its header; None
    when it has none or cannot be opened."""
    try:
        with Image.open(full) as img:
            return of_exif(img.getexif())
    except Exception:
        return None


def of_picture(rel, full):
    """The year to store for one picture: 0 when nothing gives one."""
    return of_path(rel) or of_file(full) or 0


def where(year):
    """(SQL, args) narrowing to one choice: "all", "none", or a year as text.
    None for anything else, which the route refuses."""
    if year == "all":
        return "1", []
    if year == "none":
        return "coalesce(year, 0) = 0", []
    if year.isdigit() and len(year) == 4:
        return "year = ?", [int(year)]
    return None


def counts(conn, where_sql, args):
    """{year as text: count} over the rows `where_sql` selects, No date as "none"."""
    out = {}
    for y, n in conn.execute(f"SELECT coalesce(year, 0), count(*) FROM files WHERE {where_sql} "
                             "GROUP BY 1", args):
        out["none" if not y else str(y)] = n
    return out


def backfill(conn, root, stopping, batch=500):
    """Give a year to every live picture from before this column. Most come
    from the path alone; the rest open their header for EXIF. Returns how many
    rows were set."""
    done = 0
    real_root = os.path.realpath(root)
    while not stopping():
        rows = conn.execute("SELECT id, rel FROM files WHERE year IS NULL "
                            "AND status IN ('pending','done') AND dhash IS NOT NULL LIMIT ?",
                            (batch,)).fetchall()
        if not rows:
            break
        sets = []
        for r in rows:
            full = os.path.realpath(os.path.join(root, r["rel"]))
            inside = full.startswith(real_root + os.sep)
            sets.append((of_path(r["rel"]) or (of_file(full) if inside else None) or 0, r["id"]))
        with db.immediate(conn):
            conn.executemany("UPDATE files SET year=? WHERE id=? AND year IS NULL", sets)
        done += len(sets)
    return done
