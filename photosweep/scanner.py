"""The scan: walk the library, then label one image at a time, alternating
with a second question on the photos already labelled: is it clutter
(classify.judge_clutter), is it blurry (classify.judge_blur).

    python -m photosweep.scanner              # the service: walk, label, idle, re-walk
    python -m photosweep.scanner --limit 50   # the timing run: label 50 and print a report

Resumable by construction. A row is `pending` until it has a verdict, so a
restart, a reboot or a pause picks up exactly where it stopped. The pause flag
is a row in `settings` that the web page flips; the loop checks it before every
image.
"""
import argparse
import base64
import io
import logging
import os
import random
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
from PIL import Image

from . import classify, config, db, dupes, imaging

log = logging.getLogger("photosweep.scanner")
_stop = False


def _on_term(_sig, _frame):
    global _stop
    _stop = True


def _done_future(value):
    fut = Future()
    fut.set_result(value)
    return fut


def thumb_path(file_id):
    return os.path.join(config.THUMB_DIR, f"{file_id % 256:02x}", f"{file_id}.jpg")


def walk(conn):
    """Register every image under PICS_ROOT; mark rows whose file vanished as
    `gone`. Returns (seen, added, gone). New rows go in 500 to a transaction."""
    root = config.PICS_ROOT
    if not os.path.isdir(root) or not os.listdir(root):
        # An empty or missing mount must not be read as "every photo was deleted".
        raise RuntimeError(f"{root} is missing or empty; is the bind mount up?")
    seen = set()
    batch = []

    def flush():
        with db.immediate(conn):
            conn.executemany(
                "INSERT INTO files (rel, size, mtime) VALUES (?, ?, ?) ON CONFLICT (rel) DO NOTHING",
                batch)
        batch.clear()

    before = conn.execute("SELECT count(*) FROM files").fetchone()[0]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in config.IMAGE_EXTS:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                st = os.stat(full)
            except OSError as exc:
                log.warning("stat failed rel=%s err=%s", rel, exc)
                continue
            seen.add(rel)
            mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")
            batch.append((rel, st.st_size, mtime))
            if len(batch) >= 500:
                flush()
    if batch:
        flush()
    after = conn.execute("SELECT count(*) FROM files").fetchone()[0]

    gone = 0
    live = conn.execute("SELECT id, rel FROM files WHERE status IN ('pending','done','error')").fetchall()
    missing = [r["id"] for r in live if r["rel"] not in seen]
    for i in range(0, len(missing), 500):
        chunk = missing[i:i + 500]
        with db.immediate(conn):
            conn.execute(f"UPDATE files SET status='gone' WHERE id IN ({','.join('?' * len(chunk))})", chunk)
        gone += len(chunk)
    return len(seen), after - before, gone


def next_pending(conn, sample=False, match=None, exclude=None):
    """The next row to label: library order for the service, or a random pick
    (optionally filtered by a path substring) for a timing or accuracy sample.
    `exclude` is the row already in hand, so the prefetch picks the one after."""
    sql = "SELECT id, rel, attempts FROM files WHERE status='pending' AND attempts < ?"
    args = [config.MAX_ATTEMPTS]
    if exclude is not None:
        sql += " AND id != ?"
        args.append(exclude)
    if match:
        sql += " AND rel LIKE ? ESCAPE '\\'"
        args.append("%" + match.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    sql += " ORDER BY random() LIMIT 1" if sample else " ORDER BY id LIMIT 1"
    return conn.execute(sql, args).fetchone()


def fail(conn, file_id, message):
    """A failure that is the file's own (unreadable, undecodable, or the model
    gave nonsense for it): count it, and give up after MAX_ATTEMPTS."""
    with db.immediate(conn):
        conn.execute(
            "UPDATE files SET attempts = attempts + 1, error = ?, "
            "status = CASE WHEN attempts + 1 >= ? THEN 'error' ELSE status END WHERE id = ?",
            (message[:300], config.MAX_ATTEMPTS, file_id))


def prepare(file_id, rel):
    """Decode one file and write its thumbnail. Touches no database, so it is
    safe on the prefetch thread while the model works on the previous image.
    Returns ("ok", thumb, camera, width, height, sharpness, dhash), ("gone",)
    or ("bad", message)."""
    full = os.path.realpath(os.path.join(config.PICS_ROOT, rel))
    if not full.startswith(os.path.realpath(config.PICS_ROOT) + os.sep):
        return ("bad", "path escapes the library")
    try:
        img, camera, width, height = imaging.open_image(full, config.THUMB_PX)
        thumb = imaging.jpeg_bytes(img, config.THUMB_PX)
        # from the JPEG as stored, not from `img`, so the backfills agree with it
        with Image.open(io.BytesIO(thumb)) as stored:
            sharp = imaging.sharpness(stored)
            dh = imaging.dhash(stored)
    except FileNotFoundError:
        return ("gone",)
    except Exception as exc:  # any decoder failure is this file's problem
        return ("bad", f"decode: {type(exc).__name__}: {exc}")
    path = thumb_path(file_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(thumb)
    os.replace(tmp, path)
    return ("ok", thumb, camera, width, height, sharp, dh)


def label_one(conn, http, token, row, prepared=None):
    """Label one row, from `prepared` when the prefetch already decoded it.
    Raises classify.RouterUnavailable for the caller's backoff."""
    if prepared is None:
        prepared = prepare(row["id"], row["rel"])
    if prepared[0] == "gone":
        with db.immediate(conn):
            conn.execute("UPDATE files SET status='gone' WHERE id=?", (row["id"],))
        return None
    if prepared[0] == "bad":
        fail(conn, row["id"], prepared[1])
        return None
    _, thumb, camera, width, height, sharp, dh = prepared

    t0 = time.monotonic()
    try:
        # a retry samples: at temperature 0 it would repeat the same loop exactly
        category, confidence, reason = classify.classify(
            http, config.ROUTER_URL, token, config.MODEL, base64.b64encode(thumb).decode("ascii"),
            sample=row["attempts"] > 0)
    except classify.BadVerdict as exc:
        fail(conn, row["id"], f"model: {exc}")
        return None
    ms = int((time.monotonic() - t0) * 1000)
    with db.immediate(conn):
        conn.execute(
            "UPDATE files SET status='done', category=?, confidence=?, reason=?, camera=?, width=?, "
            "height=?, name_hint=?, model_ms=?, scanned_at=?, sharpness=?, dhash=?, error=NULL "
            "WHERE id=? AND status='pending'",
            (category, confidence, reason, camera, width, height,
             imaging.name_hint(row["rel"]), ms, db.now(), sharp, dh, row["id"]))
    return category, confidence, ms


def next_blur(conn):
    """The next photo to ask the blur question of. Likeliest-blurry first, so
    the top of the Blurry tab is settled within hours rather than at the end:
    those whose label reason already says blurred, then lowest pixel score."""
    # A photo already called clutter is not asked: it is in Clutter whatever the
    # blur answer, so the question would buy nothing. An operator's Recheck is
    # still honoured.
    return conn.execute(
        "SELECT id, rel, blur_recheck FROM files WHERE status='done' AND kept=0 AND category='photo' "
        "AND (blurry IS NULL OR blur_recheck = 1) "
        "AND (coalesce(clutter, -1) != 1 OR blur_recheck = 1) ORDER BY blur_recheck DESC, "
        "(reason LIKE '%blur%' OR reason LIKE '%out of focus%') DESC, "
        "sharpness IS NULL, sharpness, id LIMIT 1").fetchone()


def _blur_image(row):
    return _question_image(row, row["blur_recheck"])


def _question_image(row, recheck):
    """Base64 JPEG for a second question. A recheck reads the original at
    RECHECK_PX, where blur and small detail are far easier to see than in the
    512 px thumbnail; if the original cannot be read it falls back to the
    thumbnail. Raises OSError when there is no image at all."""
    if recheck:
        full = os.path.realpath(os.path.join(config.PICS_ROOT, row["rel"]))
        if full.startswith(os.path.realpath(config.PICS_ROOT) + os.sep):
            try:
                img, *_ = imaging.open_image(full, config.RECHECK_PX)
                return base64.b64encode(imaging.jpeg_bytes(img, config.RECHECK_PX)).decode("ascii")
            except Exception as exc:  # any decoder failure: the thumbnail still serves
                log.warning("recheck: original unreadable id=%d err=%s", row["id"], type(exc).__name__)
    with open(thumb_path(row["id"]), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def judge_one(conn, http, token, row):
    """Ask the blur question of one labelled photo. Raises
    classify.RouterUnavailable for the caller's backoff. An unusable answer gets
    one sampled retry (a greedy loop repeats itself exactly); a second one, or
    no image at all, records -1 and the tab falls back to the pixel score."""
    blurry, conf = -1, None
    try:
        b64 = _blur_image(row)
    except OSError as exc:
        log.warning("blur check: no image id=%d err=%s", row["id"], type(exc).__name__)
    else:
        for sample in (False, True):
            try:
                verdict, conf = classify.judge_blur(http, config.ROUTER_URL, token, config.MODEL, b64,
                                                    sample=sample)
                blurry = int(verdict)
                break
            except classify.BadVerdict as exc:
                log.warning("blur check: bad verdict id=%d sampled=%s err=%s", row["id"], sample, exc)
    with db.immediate(conn):
        conn.execute("UPDATE files SET blurry=?, blur_conf=?, blur_recheck=0 "
                     "WHERE id=? AND (blurry IS NULL OR blur_recheck = 1)",
                     (blurry, conf, row["id"]))
    return blurry


def next_clutter(conn):
    """The next photo to ask the clutter question of. Photos with no camera in
    their EXIF first (chat forwards, saves, downloads: where the clutter is
    thickest, about 8,500 of 33,000 on 2026-11-19), and photos already in
    Blurry last, since those are in front of the owner already."""
    return conn.execute(
        "SELECT id, rel, clutter_recheck FROM files WHERE status='done' AND kept=0 AND category='photo' "
        "AND (clutter IS NULL OR clutter_recheck = 1) ORDER BY clutter_recheck DESC, "
        "(camera IS NULL) DESC, (coalesce(blurry, -1) = 1), id LIMIT 1").fetchone()


def next_side(conn):
    """The next second-question job, as (kind, row) or None. An operator's
    Recheck goes first, since someone is waiting on it (clutter, then blur);
    then the clutter question, which is what the Photos tab most needs; then
    the blur backlog."""
    blur = next_blur(conn)
    clutter = next_clutter(conn)
    if clutter is not None and clutter["clutter_recheck"]:
        return "clutter", clutter
    if blur is not None and blur["blur_recheck"]:
        return "blur", blur
    if clutter is not None:
        return "clutter", clutter
    return ("blur", blur) if blur is not None else None


def clutter_one(conn, http, token, row):
    """Ask the clutter question of one labelled photo, from its thumbnail, or
    from the original at RECHECK_PX on an operator's Recheck. Same contract as
    judge_one: RouterUnavailable goes to the caller, one sampled retry on an
    unusable answer, then -1 (it stays in Photos)."""
    verdict, conf, reason = -1, None, None
    try:
        b64 = _question_image(row, row["clutter_recheck"])
    except OSError as exc:
        log.warning("clutter check: no image id=%d err=%s", row["id"], type(exc).__name__)
    else:
        for sample in (False, True):
            try:
                answer, conf, reason = classify.judge_clutter(http, config.ROUTER_URL, token, config.MODEL,
                                                              b64, sample=sample)
                verdict = int(answer)
                break
            except classify.BadVerdict as exc:
                log.warning("clutter check: bad verdict id=%d sampled=%s err=%s", row["id"], sample, exc)
    with db.immediate(conn):
        conn.execute("UPDATE files SET clutter=?, clutter_conf=?, clutter_reason=?, clutter_recheck=0 "
                     "WHERE id=? AND (clutter IS NULL OR clutter_recheck = 1)",
                     (verdict, conf, reason, row["id"]))
    return verdict


def backfill_sharpness(conn, batch=200):
    """Score labelled rows that predate the sharpness column, from their stored
    thumbnails (about 3 ms each). A row whose thumbnail is missing or unreadable
    stays NULL, which keeps it out of Blurry rather than guessing. Returns the
    number scored."""
    todo = [r["id"] for r in conn.execute(
        "SELECT id FROM files WHERE status='done' AND sharpness IS NULL ORDER BY id")]
    scored, pending = 0, []

    def flush():
        with db.immediate(conn):
            conn.executemany("UPDATE files SET sharpness=? WHERE id=? AND sharpness IS NULL", pending)
        pending.clear()

    for file_id in todo:
        if _stop:
            break
        try:
            with Image.open(thumb_path(file_id)) as im:
                sharp = imaging.sharpness(im)
        except (OSError, ValueError) as exc:
            log.warning("sharpness backfill skipped id=%d err=%s", file_id, type(exc).__name__)
            continue
        if sharp is None:
            continue
        pending.append((sharp, file_id))
        scored += 1
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return scored


def _dupes(conn, backfill=False):
    """Hash what has no hash yet (start-up only) and link what is unlinked.
    Never fatal: duplicates are a side view, and a failure here must not stop
    the labelling."""
    try:
        if backfill:
            hashed = dupes.backfill_hashes(conn, thumb_path, lambda: _stop)
            if hashed:
                log.info("dhash backfill hashed=%d", hashed)
        linked, pairs = dupes.link(conn, lambda: _stop)
        if linked:
            log.info("duplicates linked=%d pairs=%d", linked, pairs)
    except Exception as exc:
        log.error("duplicate linking failed err=%s", f"{type(exc).__name__}: {exc}")


def _back_off(exc, backoff):
    """Background job: exponential backoff with full jitter, honouring
    Retry-After when the router sends one. The file keeps its attempts.
    Sleeps, and returns the next backoff ceiling."""
    wait = exc.retry_after if exc.retry_after is not None else random.uniform(0, backoff)
    log.warning("router unavailable (%s); sleeping %.0fs", exc, wait)
    time.sleep(min(wait, 600))
    return min(backoff * 2, 300.0)


def _token():
    """(usable, token). No token file configured means plain Ollama: no token,
    and that is fine. A file configured but missing or empty means wait."""
    if not config.ROUTER_TOKEN_FILE:
        return True, None
    token = config.read_secret(config.ROUTER_TOKEN_FILE)
    return token is not None, token


def run(limit=None, sample=False, match=None):
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    os.makedirs(config.THUMB_DIR, exist_ok=True)
    conn = db.connect(config.DB_PATH)
    seen, added, gone = walk(conn)
    log.info("walk seen=%d added=%d gone=%d", seen, added, gone)
    last_walk = time.monotonic()
    scored = backfill_sharpness(conn)
    if scored:
        log.info("sharpness backfill scored=%d", scored)
    if limit is None:  # the service only; a timing run measures labelling
        _dupes(conn, backfill=True)

    done, timings, backoff = 0, [], 5.0
    # The service alternates one label with one second-question turn (a blur
    # or clutter check, next_side) while both have work, and does those alone
    # once labelling is done. Timing runs (--limit) measure labelling only and
    # never take a side turn.
    side_due = False
    started = time.monotonic()
    # One image is decoded ahead while the model labels the current one: the
    # GPU otherwise sat idle ~0.5 s an image (2.77 s wall against a 2.27 s
    # model median, measured 2026-11-17). Still one request in flight.
    ahead = None  # (row, future of prepare())
    with httpx.Client(timeout=httpx.Timeout(config.MODEL_TIMEOUT_S, connect=10.0)) as http, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as pool:
        while not _stop and (limit is None or done < limit):
            if db.get_setting(conn, "paused", "0") == "1":
                ahead = None  # its thumbnail is harmless; it is re-prepared on resume
                time.sleep(5)
                continue
            if limit is None and time.monotonic() - last_walk > config.REWALK_S:
                seen, added, gone = walk(conn)
                log.info("rewalk seen=%d added=%d gone=%d", seen, added, gone)
                last_walk = time.monotonic()
            if side_due:
                side_due = False
                job = next_side(conn)
                if job is not None:
                    ok, token = _token()
                    if not ok:
                        log.error("no model token at %s; waiting", config.ROUTER_TOKEN_FILE)
                        side_due = True
                        time.sleep(60)
                        continue
                    kind, srow = job
                    try:
                        (judge_one if kind == "blur" else clutter_one)(conn, http, token, srow)
                    except classify.RouterUnavailable as exc:
                        side_due = True  # the same photo, next time round
                        backoff = _back_off(exc, backoff)
                        continue
                    backoff = 5.0
                    continue
            if ahead is not None:
                (row, fut), ahead = ahead, None
            else:
                row = next_pending(conn, sample, match)
                if row is None:
                    if limit is not None:
                        break
                    _dupes(conn)  # files labelled since the last pass; a no-op when there are none
                    if next_side(conn) is not None:
                        side_due = True
                        continue
                    time.sleep(60)  # idle until the next re-walk finds something
                    continue
                fut = pool.submit(prepare, row["id"], row["rel"])
            ok, token = _token()
            if not ok:
                log.error("no model token at %s; waiting", config.ROUTER_TOKEN_FILE)
                ahead = (row, fut)
                time.sleep(60)
                continue
            prepared = fut.result()
            if limit is None or done + 1 < limit:
                nxt = next_pending(conn, sample, match, exclude=row["id"])
                if nxt is not None:
                    ahead = (nxt, pool.submit(prepare, nxt["id"], nxt["rel"]))
            try:
                result = label_one(conn, http, token, row, prepared)
            except classify.RouterUnavailable as exc:
                ahead = (row, _done_future(prepared))  # the same image, next time round
                backoff = _back_off(exc, backoff)
                continue
            backoff = 5.0
            side_due = limit is None
            if result:
                done += 1
                timings.append(result[2])
                if limit is not None:
                    print(f"{done:>4}  {result[0]:<10} {result[1]:>3}%  {result[2]:>6} ms  {row['rel']}")
    if limit is not None and timings:
        elapsed = time.monotonic() - started
        timings.sort()
        print(f"\nlabelled {done} in {elapsed:.0f}s wall: {elapsed / done:.2f}s per image overall; "
              f"model median {timings[len(timings) // 2]} ms, p90 {timings[int(len(timings) * 0.9)]} ms")


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format='{"t":"%(asctime)s","lvl":"%(levelname)s","src":"%(name)s","msg":"%(message)s"}')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="label this many, print a timing report, and exit")
    ap.add_argument("--sample", action="store_true", help="with --limit: pick at random, not in order")
    ap.add_argument("--match", help="with --limit: only paths containing this text")
    args = ap.parse_args()
    if (args.sample or args.match) and args.limit is None:
        ap.error("--sample and --match need --limit")
    run(args.limit, args.sample, args.match)


if __name__ == "__main__":
    main()
