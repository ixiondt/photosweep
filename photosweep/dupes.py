"""Duplicates: one picture held in more than one file.

A filename or byte hash misses the common case: a WhatsApp forward is
re-compressed, a HEIC was exported as JPEG, a re-import added `_1` and dropped
the EXIF. The bytes differ, the picture does not. So every labelled file gets a
perceptual hash of its thumbnail (imaging.dhash), every file is compared once
against every other (link), and the pairs within DUP_SIMILAR bits are stored.
Groups are built from those pairs at read time (groups), over the files still
on disk, so a delete or a Keep reshapes them with nothing to maintain.

Two tiers, from the 2026-11-21 measurement of the whole library:
  same     every member joined by pairs within DUP_SAME bits: copies of one
           picture (original and forward, HEIC and JPEG, `x.jpg` and `x_1.jpg`)
  similar  joined only through looser pairs: burst frames and near-repeats,
           where picking the best shot is a judgement, not a cleanup

No model is involved. Linking the 32,000-file library from scratch is about
20 s of pure Python for 32,000 photos (measured); after that only new files are
compared, each against all.
"""
import logging
import os
import re
from collections import defaultdict

from PIL import Image

from . import config, db, imaging

log = logging.getLogger("photosweep.dupes")
TIERS = ("same", "similar")

# Names that say "a copy of something": `x (1).jpg`, `x_1.jpg`, `x copy.jpg`, a
# WhatsApp or Messenger save. Short `_N` only: `IMG_4575.JPG` is an original.
_COPY_NAME = re.compile(r"\(\d+\)|_\d{1,2}\.\w+$|\bcopy\b|-WA\d{4}|^received_|^FB_IMG_", re.I)

# Every column keeper_key() reads
_RANK_COLS = "id, rel, size, width, height, camera, kept, blurry, sharpness, mtime"


def distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def keeper_key(r):
    """Sort key: the copy worth keeping sorts first. Anything already kept;
    most pixels (an original beats its forward); a camera in the EXIF; not
    called blurry; the larger file; a name that does not say copy; the sharper;
    the older; then id, so the choice is stable.

    The blur verdict ranks below pixels and camera on purpose. Copies of one
    picture are equally blurred, but the model does not always say so alike:
    on 2026-11-21 it called a 1.4 MB camera original blurry and its 212 KB
    forward sharp, and 30 groups picked a smaller file as best. Burst frames
    tie on pixels and camera, so that is where the verdict still decides."""
    return (-(r["kept"] or 0), -((r["width"] or 0) * (r["height"] or 0)), r["camera"] is None,
            r["blurry"] == 1, -(r["size"] or 0), bool(_COPY_NAME.search(os.path.basename(r["rel"]))),
            -(r["sharpness"] or 0), r["mtime"] or "", r["id"])


# ------------------------------------------------------------ scanner ----

def backfill_hashes(conn, thumb_path, should_stop, batch=200):
    """Hash labelled rows that have no dhash yet, from their stored thumbnails
    (about 2 ms each). A missing or unreadable thumbnail stays NULL and out of
    every group. Returns the number hashed."""
    todo = [r["id"] for r in conn.execute(
        "SELECT id FROM files WHERE status='done' AND dhash IS NULL ORDER BY id")]
    hashed, pending = 0, []

    def flush():
        with db.immediate(conn):
            conn.executemany("UPDATE files SET dhash=? WHERE id=? AND dhash IS NULL", pending)
        pending.clear()

    for file_id in todo:
        if should_stop():
            break
        try:
            with Image.open(thumb_path(file_id)) as im:
                h = imaging.dhash(im)
        except (OSError, ValueError) as exc:
            log.warning("dhash backfill skipped id=%d err=%s", file_id, type(exc).__name__)
            continue
        pending.append((h, file_id))
        hashed += 1
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return hashed


def _reset_if_cutoff_changed(conn):
    """Pairs are stored only within DUP_SIMILAR, so raising it needs them all
    recomputed; lowering it does too, or the old looser pairs would stay.
    Read first: the scanner calls this on every idle pass, and the write lock
    is only worth taking when something changed."""
    if db.get_setting(conn, "dup_link_max") == str(config.DUP_SIMILAR):
        return False
    with db.immediate(conn):
        if db.get_setting(conn, "dup_link_max") == str(config.DUP_SIMILAR):
            return False
        conn.execute("DELETE FROM dup_pairs")
        conn.execute("UPDATE files SET dup_linked=0 WHERE dup_linked=1")
        db.set_setting(conn, "dup_link_max", config.DUP_SIMILAR)
    log.info("duplicate cutoff is now %d bits; relinking everything", config.DUP_SIMILAR)
    return True


def link(conn, should_stop, batch=500):
    """Compare each hashed-but-unlinked file with every hashed file before it
    (all linked ones, and unlinked ones earlier in this run), and store the
    pairs within DUP_SIMILAR. Each pair is found exactly once. Commits every
    `batch` files, so a stop loses at most a batch, which is redone next time.
    Returns (files linked, pairs stored)."""
    _reset_if_cutoff_changed(conn)
    if conn.execute("SELECT 1 FROM files WHERE dhash IS NOT NULL AND dup_linked=0 LIMIT 1").fetchone() is None:
        return 0, 0
    pool, todo = [], []
    for r in conn.execute("SELECT id, dhash, dup_linked FROM files WHERE dhash IS NOT NULL ORDER BY id"):
        (pool if r["dup_linked"] else todo).append((r["id"], int(r["dhash"], 16)))
    cap = config.DUP_SIMILAR
    linked = stored = 0
    edges, ids = [], []

    def flush():
        nonlocal linked, stored
        marks = ",".join("?" * len(ids))
        with db.immediate(conn):
            conn.executemany("INSERT INTO dup_pairs (a, b, dist) VALUES (?, ?, ?) "
                             "ON CONFLICT (a, b) DO UPDATE SET dist = excluded.dist", edges)
            conn.execute(f"UPDATE files SET dup_linked=1 WHERE id IN ({marks})", ids)
        linked += len(ids)
        stored += len(edges)
        edges.clear()
        ids.clear()

    for file_id, h in todo:
        if should_stop():
            break
        for other, d in [(o, d) for o, oh in pool if (d := (h ^ oh).bit_count()) <= cap]:
            edges.append((min(file_id, other), max(file_id, other), d))
        pool.append((file_id, h))
        ids.append(file_id)
        if len(ids) >= batch:
            flush()
    if ids:
        flush()
    return linked, stored


# --------------------------------------------------------------- read ----

class _Sets:
    """Union-find over file ids."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        parent = self.parent
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def groups(conn):
    """Every live duplicate group, as dicts {tier, ids, dist}: `ids` keeper
    first, `dist` each member's bits from the keeper where they were directly
    compared (None for the keeper, and for a member joined only through
    another). Only files still on disk count, and a group where every member
    is already kept is settled and left out. Newest keeper first."""
    pairs = conn.execute(
        "SELECT p.a, p.b, p.dist FROM dup_pairs p "
        "JOIN files fa ON fa.id = p.a AND fa.status = 'done' "
        "JOIN files fb ON fb.id = p.b AND fb.status = 'done' WHERE p.dist <= ?",
        (config.DUP_SIMILAR,)).fetchall()
    if not pairs:
        return []
    loose, tight, dist = _Sets(), _Sets(), {}
    for a, b, d in pairs:
        loose.union(a, b)
        if d <= config.DUP_SAME:
            tight.union(a, b)
        dist[(a, b)] = d
    members = defaultdict(list)
    for x in list(loose.parent):
        members[loose.find(x)].append(x)
    info = {r["id"]: r for r in conn.execute(
        f"SELECT {_RANK_COLS} FROM files WHERE status='done' AND id IN "
        "(SELECT a FROM dup_pairs UNION SELECT b FROM dup_pairs)")}
    out = []
    for ids in members.values():
        rows = [info[i] for i in ids if i in info]
        if len(rows) < 2 or all(r["kept"] for r in rows):
            continue
        rows.sort(key=keeper_key)
        keep = rows[0]["id"]
        out.append({
            "tier": "same" if len({tight.find(r["id"]) for r in rows}) == 1 else "similar",
            "ids": [r["id"] for r in rows],
            "kept": [bool(r["kept"]) for r in rows],
            "dist": [None] + [dist.get((min(keep, r["id"]), max(keep, r["id"]))) for r in rows[1:]],
            "_sort": rows[0]["rel"],
        })
    out.sort(key=lambda g: g["_sort"], reverse=True)
    for g in out:
        del g["_sort"]
    return out


def counts(all_groups):
    """Groups per tier, the files beyond the keeper in all of them, and per
    tier how many extras a bulk delete could take (extras() of that tier)."""
    c = {t: 0 for t in TIERS}
    for g in all_groups:
        c[g["tier"]] += 1
    return {**c, "extras": sum(len(g["ids"]) - 1 for g in all_groups),
            "deletable": {t: len(extras(all_groups, t)) for t in TIERS}}


def extras(all_groups, tier):
    """Every file of a tier's groups except each best copy and anything kept:
    what Select all extras takes, and all that a bulk delete may remove."""
    return [i for g in all_groups if g["tier"] == tier
            for i, kept in zip(g["ids"][1:], g["kept"][1:]) if not kept]


def page(conn, tier, offset, limit, cols):
    """One page of a tier's groups, each member as the full item row the grid
    shows (`cols`), plus `keeper` and `dist`."""
    all_groups = groups(conn)
    chosen = [g for g in all_groups if g["tier"] == tier]
    window = chosen[offset:offset + limit]
    want = [i for g in window for i in g["ids"]]
    rows = {}
    if want:
        marks = ",".join("?" * len(want))
        rows = {r["id"]: dict(r) for r in conn.execute(f"SELECT {cols} FROM files WHERE id IN ({marks})", want)}
    out = []
    for g in window:
        items = []
        for k, (i, d) in enumerate(zip(g["ids"], g["dist"])):
            if i in rows:
                items.append({**rows[i], "keeper": k == 0, "dist": d})
        out.append({"tier": g["tier"], "items": items})
    return {"groups": out, "total": len(chosen), "offset": offset, "counts": counts(all_groups)}
