"""The read routes: progress, a tab of images, duplicate groups, and the
two image renditions. Every one needs an operator (gate.operator)."""
import calendar
import os
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from . import classifier, config, db, dupes, imaging, remote
from .gate import conn, operator
from .scanner import thumb_path

router = APIRouter()
TABS = ("screenshot", "meme", "document", "junk", "photo", "clutter", "blurry", "kept", "error")
# Where an image came from, as the filename and EXIF tell it: a filter on any
# tab. Most of what the label model calls a photo but is not one has no camera
# in its EXIF (chat forwards, saves, downloads).
SOURCES = {
    "all": "1=1",
    "camera": "camera IS NOT NULL",
    "nocamera": "camera IS NULL",
    "messaging": "name_hint = 'messaging'",
}
PAGE_MAX = 200
GROUP_PAGE_MAX = 60
ITEM_COLS = ("id, rel, category, confidence, reason, camera, width, height, name_hint, mtime, error, "
             "sharpness, blurry, blur_conf, blur_recheck, clutter, clutter_conf, clutter_reason, clutter_recheck, "
             "size, kept, label_src, clutter_src, blur_src")


# ---------------------------------------------------------------- read ----

# A photo is Blurry when the model said so, or, where it has not answered
# (NULL: not asked yet; -1: no usable answer), when the pixel score is under
# BLUR_MAX. A model "sharp" overrides a low pixel score: smooth skin, sky and
# plain walls score low without being blurred.
# Both take BLUR_MAX as their one parameter. The test is NULL-proof on
# purpose: NOT of a NULL comparison is NULL, which would drop an unscored photo
# from Photos and Blurry both.
_PHOTOS = "status='done' AND kept=0 AND category='photo'"
_IS_BLURRY = ("(coalesce(blurry, -1) = 1 OR "
              "(coalesce(blurry, -1) = -1 AND coalesce(sharpness, 1e9) < ?))")
# Clutter (the third question, classify.judge_clutter) outranks Blurry: a
# blurred photo of a floor is clutter first. Each photo is in exactly one of
# Clutter, Blurry and Photos. coalesce keeps NOT from meeting a NULL.
_IS_CLUTTER = "coalesce(clutter, -1) = 1"
CLUTTER_WHERE = f"{_PHOTOS} AND {_IS_CLUTTER}"
BLURRY_WHERE = f"{_PHOTOS} AND NOT {_IS_CLUTTER} AND {_IS_BLURRY}"
PHOTO_WHERE = f"{_PHOTOS} AND NOT {_IS_CLUTTER} AND NOT {_IS_BLURRY}"
# the model's surest calls first, then the not-yet-asked by pixel score
# (coalesce, not a bare `blurry = 1`, which is NULL for the not-yet-asked and
# would sort them below the -1 rows regardless of their pixel score)
BLURRY_ORDER = "(coalesce(blurry, -1) = 1) DESC, blur_conf DESC, sharpness ASC, rel"


@router.get("/api/summary")
def summary(request: Request):
    who = operator(request)
    c = conn()
    counts = {t: 0 for t in TABS}
    for r in c.execute("SELECT category, count(*) n FROM files WHERE status='done' AND kept=0 GROUP BY category"):
        counts[r["category"]] = r["n"]
    # Blurry is carved out of Photos: the label model files a blurred photo as
    # "photo", and a second question (or, until it is asked, the pixel score)
    # moves it here. A photo appears in exactly one of the two.
    counts["photo"] = c.execute(f"SELECT count(*) FROM files WHERE {PHOTO_WHERE}",
                                (config.BLUR_MAX,)).fetchone()[0]
    counts["blurry"] = c.execute(f"SELECT count(*) FROM files WHERE {BLURRY_WHERE}",
                                 (config.BLUR_MAX,)).fetchone()[0]
    counts["clutter"] = c.execute(f"SELECT count(*) FROM files WHERE {CLUTTER_WHERE}").fetchone()[0]
    # What is left of each second question, as the scanner sees it (next_blur,
    # next_clutter): a photo waiting for a re-ask is not checked yet, and a
    # photo in Clutter is never asked about blur, so it is not in that total.
    blur_left, blur_total, clutter_left, clutter_rechecks, photo_total = c.execute(
        "SELECT coalesce(sum((blurry IS NULL OR blur_recheck = 1) "
        "                    AND (coalesce(clutter, -1) != 1 OR blur_recheck = 1)), 0), "
        "       coalesce(sum(coalesce(clutter, -1) != 1 OR blur_recheck = 1), 0), "
        "       coalesce(sum(clutter IS NULL OR clutter_recheck = 1), 0), "
        "       coalesce(sum(clutter_recheck = 1), 0), count(*) FROM files "
        "WHERE status='done' AND kept=0 AND category='photo'").fetchone()
    counts["kept"] = c.execute(f"SELECT count(*) FROM files WHERE {db.LIVE} AND kept=1").fetchone()[0]
    counts["error"] = c.execute("SELECT count(*) FROM files WHERE status='error'").fetchone()[0]
    dup = dupes.counts(dupes.groups(c))
    counts["dupes"] = dup["extras"]  # the files beyond each group's best copy
    status = {r["status"]: r["n"] for r in c.execute("SELECT status, count(*) n FROM files GROUP BY status")}
    total = sum(status.get(s, 0) for s in ("pending", "done", "error"))
    labelled = status.get("done", 0) + status.get("error", 0)
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 1800))
    recent = c.execute("SELECT count(*) FROM files WHERE scanned_at >= ?", (since,)).fetchone()[0]
    per_hour = recent * 2
    # answers an hour over the half hour, or over the time since the first
    # answer in it when that is shorter (just after a restart, or the first
    # answers of a re-ask), so the time left is not inflated while it warms up
    n, first = c.execute("SELECT count(*), min(asked_at) FROM files WHERE asked_at >= ?", (since,)).fetchone()
    asked = 0
    if n >= 5:
        span = time.time() - calendar.timegm(time.strptime(first, "%Y-%m-%dT%H:%M:%S"))
        asked = round(n * 3600 / max(span, 120))
    side_left = blur_left + clutter_left
    pending = status.get("pending", 0)
    # the model-free first step: a pending row is fingerprinted once it has a
    # hash; a labelled or failed one is past that step either way (a labelled
    # row whose thumbnail was lost has no hash and never will)
    printed = c.execute("SELECT count(*) FROM files WHERE status='pending' AND dhash IS NOT NULL").fetchone()[0]
    return {
        "operator": who,
        "counts": counts,
        "progress": {"total": total, "labelled": labelled, "pending": pending,
                     "fingerprinted": printed + labelled,
                     "deleted": status.get("deleted", 0), "perHour": per_hour,
                     "etaHours": round(pending / per_hour, 1) if per_hour > 0 and pending else None},
        "paused": db.get_setting(c, "paused", "0") == "1",
        "nextcloudReady": config.nextcloud_configured() and config.read_secret(config.NC_SECRET_FILE) is not None,
        "nextcloudSetUp": config.nextcloud_configured(),
        "signOut": config.AUTH_MODE == "password",
        "model": config.MODEL,
        # the sorting tabs need something to sort: the vision model, the
        # image classifier on the CPU, or both
        "labelling": config.LABELLING or classifier.sorting(),
        "visionModel": config.LABELLING,
        "classifier": classifier.sorting(),
        # with no model nothing is sent anywhere, whatever the address says
        "modelRemote": remote.status() if config.LABELLING else {"remote": False, "host": "", "allowed": True},
        "blurMax": config.BLUR_MAX,
        "blurCheck": {"checked": blur_total - blur_left, "total": blur_total},
        "clutterCheck": {"checked": photo_total - clutter_left, "total": photo_total},
        # the second questions still to ask, re-asks included, and their pace
        "questions": {"left": side_left, "rechecks": clutter_rechecks, "perHour": asked,
                      "etaHours": round(side_left / asked, 1) if asked and side_left else None},
        "dupGroups": {t: dup[t] for t in dupes.TIERS},
        "dupExtras": dup["deletable"],
    }


@router.get("/api/items")
def items(request: Request, tab: str = "screenshot", min_conf: int = 0, offset: int = 0, limit: int = 120,
          source: str = "all"):
    operator(request)
    if tab not in TABS:
        raise HTTPException(status_code=400, detail="unknown tab")
    if source not in SOURCES:
        raise HTTPException(status_code=400, detail="unknown source")
    limit = max(1, min(int(limit), PAGE_MAX))
    offset = max(0, int(offset))
    min_conf = max(0, min(int(min_conf), 100))
    if tab == "kept":
        where, args = f"{db.LIVE} AND kept=1", []
    elif tab == "error":
        where, args = "status='error'", []
    elif tab == "blurry":
        where, args = BLURRY_WHERE, [config.BLUR_MAX]
    elif tab == "clutter":
        where, args = CLUTTER_WHERE, []
    elif tab == "photo":
        where, args = f"{PHOTO_WHERE} AND confidence >= ?", [config.BLUR_MAX, min_conf]
    else:
        where, args = "status='done' AND kept=0 AND category=? AND confidence >= ?", [tab, min_conf]
    # Photos run least certain first: those are the ones the model may have got
    # wrong, and the confident end of that tab is the library you are keeping.
    order = {"kept": "rel", "error": "rel", "photo": "confidence ASC, rel", "blurry": BLURRY_ORDER,
             "clutter": "clutter_conf DESC, rel"}.get(tab, "confidence DESC, rel")
    # how many of this tab each source filter would leave, for the filter's labels
    n = conn().execute(
        f"SELECT count(*), count(camera), sum(name_hint = 'messaging') FROM files WHERE {where}", args).fetchone()
    sources = {"all": n[0], "camera": n[1], "nocamera": n[0] - n[1], "messaging": n[2] or 0}
    where = f"{where} AND {SOURCES[source]}"
    rows = conn().execute(f"SELECT {ITEM_COLS} FROM files WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                          [*args, limit, offset]).fetchall()
    return {"items": [dict(r) for r in rows], "total": sources[source], "offset": offset, "sources": sources}


@router.get("/api/dupes")
def dupe_groups(request: Request, tier: str = "same", offset: int = 0, limit: int = 40):
    """Duplicate groups of one tier, best copy first in each (dupes.groups).
    Paged by group, not by file, so a group never splits across pages."""
    operator(request)
    if tier not in dupes.TIERS:
        raise HTTPException(status_code=400, detail="unknown tier")
    return dupes.page(conn(), tier, max(0, int(offset)), max(1, min(int(limit), GROUP_PAGE_MAX)), ITEM_COLS)


@router.get("/api/dupes/extras")
def dupe_extras(request: Request, tier: str = "same"):
    """The ids Select all extras takes: every extra of every Same group,
    loaded on the page or not. Same only: a Similar group is a judgement per
    group, and bulk-selecting them deleted 886 burst frames on 2026-11-21."""
    operator(request)
    if tier != "same":
        raise HTTPException(status_code=400, detail="only Same groups can be selected in bulk")
    ids = dupes.extras(dupes.groups(conn()), tier)
    return {"ids": ids, "count": len(ids)}


def _row(file_id):
    r = conn().execute("SELECT id, rel, status FROM files WHERE id=?", (file_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="no such image")
    return r


@router.get("/thumb/{file_id}")
def thumb(request: Request, file_id: int):
    operator(request)
    _row(file_id)
    path = thumb_path(file_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no thumbnail yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/view/{file_id}")
def view(request: Request, file_id: int):
    """A larger rendition from the original, made on demand and not stored."""
    operator(request)
    r = _row(file_id)
    root = os.path.realpath(config.PICS_ROOT)
    full = os.path.realpath(os.path.join(root, r["rel"]))
    if not full.startswith(root + os.sep):
        raise HTTPException(status_code=404, detail="no such image")
    try:
        img, _, _, _ = imaging.open_image(full, config.VIEW_PX)
        data = imaging.jpeg_bytes(img, config.VIEW_PX, quality=85)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="the original is no longer there")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
