"""The write routes: keep, recheck, pause, the Nextcloud password, delete.
Every one passes gate.mutation (operator, same origin, custom header, JSON)
and writes an audit row in the same transaction as its change."""
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from . import config, db, dupes, nextcloud
from .gate import api_error, conn, ids_from, mutation

router = APIRouter()
DELETE_MAX = 500
log = logging.getLogger("photosweep.web")


# --------------------------------------------------------------- write ----

@router.post("/api/keep")
async def keep(request: Request):
    who, body = await mutation(request)
    ids = ids_from(body, DELETE_MAX)
    flag = 1 if body.get("keep", True) else 0
    c = conn()
    marks = ",".join("?" * len(ids))
    with db.immediate(c):
        # a fingerprinted file not yet labelled (or never, with MODEL=none) can be kept too
        n = c.execute(f"UPDATE files SET kept=? WHERE {db.LIVE} AND id IN ({marks})", [flag, *ids]).rowcount
        db.audit(c, who, "keep" if flag else "unkeep", {"ids": ids, "n": n})
    return {"updated": n}


@router.post("/api/recheck")
async def recheck(request: Request):
    """Ask the model again. An Errors row goes back to pending with fresh
    attempts (relabelled in library order). A photo (in Photos, Clutter or
    Blurry) is re-asked both the clutter and the blur question ahead of the
    queue, from the original at RECHECK_PX, since asking again from the same
    thumbnail would only repeat the answer; its current verdicts stand until
    the new ones land, so it does not hop tabs meanwhile."""
    who, body = await mutation(request)
    ids = ids_from(body, DELETE_MAX)
    c = conn()
    marks = ",".join("?" * len(ids))
    with db.immediate(c):
        relabel = c.execute(
            f"UPDATE files SET status='pending', attempts=0, error=NULL WHERE status='error' AND id IN ({marks})",
            ids).rowcount
        reblur = c.execute(
            "UPDATE files SET blur_recheck=1, clutter_recheck=1 "
            f"WHERE status='done' AND kept=0 AND category='photo' AND id IN ({marks})", ids).rowcount
        db.audit(c, who, "recheck", {"ids": ids, "relabel": relabel, "blur": reblur})
    return {"relabel": relabel, "blur": reblur}


@router.post("/api/pause")
async def pause(request: Request):
    who, body = await mutation(request)
    paused = bool(body.get("paused"))
    c = conn()
    with db.immediate(c):
        db.set_setting(c, "paused", "1" if paused else "0")
        db.audit(c, who, "pause" if paused else "resume", {})
    return {"paused": paused}


@router.post("/api/nextcloud")
async def set_nextcloud(request: Request):
    """Save the app password after proving it works. Logged by length, never value."""
    who, body = await mutation(request)
    if not config.nextcloud_configured():
        return api_error(409, "NEXTCLOUD_NOT_SET_UP", "Set NC_BASE and NC_USER, then restart.")
    password = body.get("password")
    if not isinstance(password, str) or not (20 <= len(password.strip()) <= 200):
        raise HTTPException(status_code=400, detail="paste the whole app password")
    password = password.strip()
    try:
        ok = await run_in_threadpool(nextcloud.check, config.NC_BASE, config.NC_USER, config.NC_PREFIX, password)
    except Exception as exc:
        log.warning("nextcloud check failed err=%s", type(exc).__name__)
        return api_error(502, "NEXTCLOUD_UNREACHABLE", "Could not reach Nextcloud.", retryable=True)
    if not ok:
        return api_error(400, "NEXTCLOUD_REJECTED", "Nextcloud did not accept that app password.")
    tmp = config.NC_SECRET_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(password + "\n")
    os.replace(tmp, config.NC_SECRET_FILE)
    c = conn()
    with db.immediate(c):
        db.audit(c, who, "nextcloud-password-set", {"length": len(password)})
    return {"ok": True}


@router.post("/api/delete")
async def delete(request: Request):
    """Send each file to the Nextcloud trash bin, one at a time, and record
    each outcome. Rows are claimed (status -> 'deleting') before the first call
    so a double-click cannot delete twice; anything not finished is put back."""
    who, body = await mutation(request)
    ids = ids_from(body, DELETE_MAX)
    password = config.read_secret(config.NC_SECRET_FILE)
    if not password or not config.nextcloud_configured():
        return api_error(409, "NEXTCLOUD_NOT_CONFIGURED", "Save a Nextcloud app password first.")
    # Up to DELETE_MAX sequential HTTP calls: off the event loop, or the page
    # (thumbnails included) stalls for the whole batch.
    return await run_in_threadpool(_delete_batch, who, ids, password)


@router.post("/api/dupes/delete-extras")
async def delete_extras(request: Request):
    """Delete for Select all extras. The server, not the page, decides what is
    an extra: the groups are rebuilt now, and any id that is not currently an
    extra of a Same group (a best copy, something kept, a Similar frame, a file
    the page saw earlier but that has since become a group's best copy) is
    skipped and reported, never deleted. A Same group therefore cannot lose its
    last copy through this route, whatever the page sends. With kind
    "mirrored" (Select all mirrored copies) the allowed set is instead the
    copies captioned mirrored in Similar (dupes.mirrored_extras), rebuilt the
    same way."""
    who, body = await mutation(request)
    ids = ids_from(body, DELETE_MAX)
    password = config.read_secret(config.NC_SECRET_FILE)
    if not password or not config.nextcloud_configured():
        return api_error(409, "NEXTCLOUD_NOT_CONFIGURED", "Save a Nextcloud app password first.")
    kind = body.get("kind", "same") if isinstance(body, dict) else "same"
    if kind not in ("same", "mirrored"):
        return api_error(400, "BAD_REQUEST", "kind must be same or mirrored.")
    groups = dupes.groups(conn())
    allowed = set(dupes.mirrored_extras(groups) if kind == "mirrored" else dupes.extras(groups, "same"))
    todo = [i for i in ids if i in allowed]
    skipped = [i for i in ids if i not in allowed]
    c = conn()
    with db.immediate(c):
        db.audit(c, who, "delete-extras", {"kind": kind, "asked": len(ids), "skipped": skipped})
    res = (await run_in_threadpool(_delete_batch, who, todo, password)) if todo else         {"deleted": 0, "gone": 0, "failed": 0, "failedIds": []}
    return {**res, "skipped": len(skipped), "skippedIds": skipped}


def _delete_batch(who, ids, password):
    c = conn()
    marks = ",".join("?" * len(ids))
    with db.immediate(c):
        before = {r["id"]: r["status"] for r in c.execute(
            f"SELECT id, status FROM files WHERE id IN ({marks})", ids)}
        claimed = c.execute(
            # Errors rows too: an empty (0-byte) or garbage file can only be cleared this way.
            # Pending too: a duplicate is found before it is labelled, or with no model at all.
            f"UPDATE files SET status='deleting' WHERE status IN ('pending','done','error') AND id IN ({marks}) "
            "RETURNING id, rel",
            ids).fetchall()
    results = {"deleted": 0, "gone": 0, "failed": 0}
    failed = []
    try:
        with nextcloud.client(config.NC_USER, password) as nc:
            for r in claimed:
                try:
                    outcome = nextcloud.delete(nc, config.NC_BASE, config.NC_USER, config.NC_PREFIX, r["rel"])
                except Exception as exc:
                    outcome = f"error:{type(exc).__name__}"
                if outcome in ("deleted", "gone"):
                    with db.immediate(c):
                        c.execute("UPDATE files SET status='deleted', deleted_at=? WHERE id=? AND status='deleting'",
                                  (db.now(), r["id"]))
                        db.audit(c, who, "delete", {"id": r["id"], "rel": r["rel"], "outcome": outcome})
                    results[outcome] += 1
                else:
                    failed.append({"id": r["id"], "outcome": outcome})
                    results["failed"] += 1
                    if outcome in ("error:401", "error:403"):
                        break  # a credential problem will not fix itself on the next file
    finally:
        with db.immediate(c):
            # back to exactly where it came from: labelled, Errors, or still pending
            c.executemany("UPDATE files SET status=? WHERE id=? AND status='deleting'",
                          [(before[r["id"]], r["id"]) for r in claimed])
    if failed:
        log.warning("delete failures n=%d first=%s", len(failed), failed[0])
    return {**results, "failedIds": [f["id"] for f in failed]}
