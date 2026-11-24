"""Automatic retraining of the classifier's heads, from the scanner's idle loop.

When the scanner has nothing to do, at most once per RETRAIN_EVERY_S, and
only once RETRAIN_MIN_NEW new answers have arrived since the heads in use were
trained (the retrain mark), it runs train_heads in a child process at low
priority, with heads_gate judging the result against the heads in use on
exactly those new answers. New heads that pass replace the old in
CLASSIFIER_HEADS, the old are kept as <heads>.prev.npz, and the mark moves to
when the new heads read the library. New heads that fail are thrown away and
the mark stays, so the next try judges on more.

The first time it runs on a library it only sets the mark: the heads in use
may have been trained on everything already there, and judging them on their
own training answers would favour them for no reason.

Rolling back by hand: copy <heads>.prev.npz over <heads> and restart the
scanner (or delete <heads> to go back to the bundled heads).

Settings it keeps: retrain_mark (UTC), retrain_tried_at (UTC), retrain_last
(JSON: when, outcome, reasons and figures, counts only).
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from . import classifier, config, db

log = logging.getLogger("photosweep.retrain")

MARK, TRIED, LAST = "retrain_mark", "retrain_tried_at", "retrain_last"
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _age_s(stamp):
    try:
        then = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


def new_answers(conn, since):
    """How many answers the heads can learn from arrived after `since`: the
    owner's keeps, deletes of something called clutter, the vision model's
    labels and second-question answers. A trigger, not the gate's count."""
    s = (since or "")[:19]
    return conn.execute(
        "SELECT (SELECT count(*) FROM files WHERE substr(kept_at, 1, 19) > ?)"
        " + (SELECT count(*) FROM files WHERE status='deleted' AND clutter=1 AND substr(deleted_at, 1, 19) > ?)"
        " + (SELECT count(*) FROM files WHERE substr(clutter_at, 1, 19) > ?)"
        " + (SELECT count(*) FROM files WHERE substr(blur_at, 1, 19) > ?)"
        " + (SELECT count(*) FROM files WHERE status IN ('done','deleted') AND label_src IS NOT 'classifier'"
        "    AND substr(scanned_at, 1, 19) > ?)", (s, s, s, s, s)).fetchone()[0]


def due(conn):
    """Whether to retrain now. Sets the first mark, and otherwise only reads."""
    if not (config.AUTO_RETRAIN and config.CLASSIFIER_HEADS and classifier.sorting()):
        return False
    mark = db.get_setting(conn, MARK)
    if mark is None:
        db.set_setting(conn, MARK, db.now())
        log.info("retrain mark set; answers from now on are what new heads are judged on")
        return False
    if _age_s(db.get_setting(conn, TRIED)) < config.RETRAIN_EVERY_S:
        return False
    return new_answers(conn, mark) >= config.RETRAIN_MIN_NEW


def _record(conn, outcome, **extra):
    db.set_setting(conn, LAST, json.dumps({"at": db.now(), "outcome": outcome, **extra}, sort_keys=True))


def _read_verdict(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
    except (OSError, ValueError) as exc:
        log.error("retrain verdict unreadable err=%s", type(exc).__name__)
        return None
    ok = (isinstance(v, dict) and isinstance(v.get("swap"), bool) and isinstance(v.get("reasons"), list)
          and isinstance(v.get("questions"), dict) and isinstance(v.get("read_at"), str))
    if not ok:
        log.error("retrain verdict has the wrong shape")
    return v if ok else None


def _remove(*paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def run(conn, stopping):
    """Train, judge, and replace the heads if the new ones pass. Returns True
    when the heads file changed (the caller reloads and rescores). Never
    raises for a failed training: it is logged and recorded."""
    db.set_setting(conn, TRIED, db.now())
    own = config.CLASSIFIER_HEADS
    current = classifier.heads_path()
    cand, verdict_path = own + ".candidate.npz", own + ".verdict.json"
    _remove(cand, verdict_path)
    cmd = [sys.executable, "-m", "photosweep.train_heads", "--db", config.DB_PATH, "--out", cand,
           "--gate", current, "--fresh-since", db.get_setting(conn, MARK), "--trust", str(config.CLASSIFIER_TRUST),
           "--verdict", verdict_path]
    nice = shutil.which("nice")
    if nice:
        cmd = [nice, "-n", "10", *cmd]
    log.info("retrain started heads_in_use=%s", current)
    started = time.monotonic()
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, cwd=_APP_DIR, stdout=subprocess.DEVNULL, stderr=err)
        while proc.poll() is None:
            if stopping() or time.monotonic() - started > config.RETRAIN_TIMEOUT_S:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                _remove(cand, verdict_path)
                why = "stopping" if stopping() else "timed out"
                log.warning("retrain abandoned (%s); the heads in use stay", why)
                _record(conn, "abandoned", reasons=[why])
                return False
            time.sleep(2)
        err.seek(0)
        tail = err.read()[-400:].decode("utf-8", "replace").strip().replace("\n", " | ")
    secs = round(time.monotonic() - started)
    if proc.returncode != 0:
        _remove(cand, verdict_path)
        log.error("retrain failed exit=%s secs=%s err=%s", proc.returncode, secs, tail)
        _record(conn, "failed", reasons=[f"exit {proc.returncode}"])
        return False
    v = _read_verdict(verdict_path)
    _remove(verdict_path)
    if v is None:
        _remove(cand)
        _record(conn, "failed", reasons=["verdict unreadable"])
        return False
    if not v["swap"]:
        _remove(cand)
        log.info("retrain kept the heads in use secs=%s reasons=%s", secs, "; ".join(v["reasons"]))
        _record(conn, "kept", reasons=v["reasons"], questions=v["questions"])
        return False
    try:
        classifier.Heads(cand)  # never swap in a file that will not load
    except (OSError, ValueError, KeyError) as exc:
        _remove(cand)
        log.error("retrain wrote heads that will not load err=%s", type(exc).__name__)
        _record(conn, "failed", reasons=["new heads will not load"])
        return False
    if os.path.isfile(own):
        shutil.copy2(own, own + ".prev.npz")
    os.replace(cand, own)
    db.set_setting(conn, MARK, v["read_at"])
    log.info("retrain replaced the heads secs=%s questions=%s", secs, json.dumps(v["questions"], sort_keys=True))
    _record(conn, "replaced", reasons=[], questions=v["questions"])
    return True
