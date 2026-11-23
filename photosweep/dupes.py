"""Duplicates: one picture held in more than one file.

A filename or byte hash misses the common case: a WhatsApp forward is
re-compressed, a HEIC was exported as JPEG, a re-import added `_1` and dropped
the EXIF. The bytes differ, the picture does not. So every labelled file gets a
perceptual hash of its thumbnail (imaging.dhash), every file is compared once
against every other (link), and the pairs within DUP_SIMILAR bits are stored.
Groups are built from those pairs at read time (groups), over the files still
on disk, so a delete or a Keep reshapes them with nothing to maintain.

The classifier's embeddings add pairs the hash misses (crops, rotations,
filters, bursts with movement): neardupes.py. Those join the Similar tier
only, never Same, and every group is capped at DUP_GROUP_MAX. Among them are
mirrored copies (a selfie and its flipped forward), which the hash alone can
never pair; each is marked against the members ranked above it (mirrored())
so Similar can show them on their own.

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

from . import config, db, imaging, neardupes

log = logging.getLogger("photosweep.dupes")
TIERS = ("same", "similar")

# Names that say "a copy of something": `x (1).jpg`, `x_1.jpg`, `x copy.jpg`, a
# WhatsApp or Messenger save. Short `_N` only: `IMG_4575.JPG` is an original.
_COPY_NAME = re.compile(r"\(\d+\)|_\d{1,2}\.\w+$|\bcopy\b|-WA\d{4}|^received_|^FB_IMG_", re.I)

# Every column keeper_key() reads
_RANK_COLS = "id, rel, size, width, height, camera, kept, blurry, sharpness, mtime"


def distance(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def mirrored(a_hash, a_flip, b_hash):
    """Whether b is picture a flipped left to right: b's hash near a's
    flipped hash (imaging.dhash_flipped), and nearer than to a as it is."""
    if not a_hash or not a_flip or not b_hash:
        return False
    d = distance(a_flip, b_hash)
    return d <= config.DUP_MIRROR and d < distance(a_hash, b_hash)


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


def backfill_flips(conn, thumb_path, should_stop, batch=200):
    """Store the flipped hash (imaging.dhash_flipped) of every hashed row that
    has none, from its stored thumbnail, so Similar can tell a mirrored copy.
    A thumbnail that cannot be read gets '' and is not tried again. Cheap when
    there is nothing to do (an index on the unflipped rows), so the scanner
    calls it on every idle pass. Returns the number done."""
    todo = [r["id"] for r in conn.execute(
        "SELECT id FROM files WHERE dhash IS NOT NULL AND dhash_flip IS NULL ORDER BY id")]
    done, pending = 0, []

    def flush():
        with db.immediate(conn):
            conn.executemany("UPDATE files SET dhash_flip=? WHERE id=? AND dhash_flip IS NULL", pending)
        pending.clear()

    for file_id in todo:
        if should_stop():
            break
        try:
            with Image.open(thumb_path(file_id)) as im:
                h = imaging.dhash_flipped(im)
        except (OSError, ValueError) as exc:
            log.warning("flipped hash skipped id=%d err=%s", file_id, type(exc).__name__)
            h = ""
        pending.append((h, file_id))
        done += 1
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return done


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
        f"JOIN files fa ON fa.id = p.a AND fa.{db.LIVE} "
        f"JOIN files fb ON fb.id = p.b AND fb.{db.LIVE} WHERE p.dist <= ?",
        (config.DUP_SIMILAR,)).fetchall()
    near = neardupes.live_pairs(conn)
    if not pairs and not near:
        return []
    loose, tight, dist = _Sets(), _Sets(), {}
    for a, b, d in pairs:
        loose.union(a, b)
        if d <= config.DUP_SAME:
            tight.union(a, b)
        dist[(a, b)] = d
    # A look-alike never touches a Same group: attached, it would turn the
    # copies into Similar and take away Select all extras (the smoke test's
    # copy pair, 2026-11-23). So pairs with an end in one are left out.
    roots = defaultdict(set)
    for x in list(loose.parent):
        roots[loose.find(x)].add(tight.find(x))
    in_same = {x for x in list(loose.parent) if len(roots[loose.find(x)]) == 1}
    # hash pairs weigh 1.0 and are never cut; embedding pairs weigh their
    # similarity, so a group over the cap sheds its loosest look-alikes first
    edges = [(a, b, 1.0) for a, b, _ in pairs] + [
        (a, b, s) for a, b, s in near if (a, b) not in dist and a not in in_same and b not in in_same]
    members = neardupes.split(edges, config.DUP_GROUP_MAX)
    want = sorted({x for m in members for x in m})
    info = {}
    for i in range(0, len(want), 900):
        chunk = want[i:i + 900]
        info.update({r["id"]: r for r in conn.execute(
            f"SELECT {_RANK_COLS}, dhash, dhash_flip FROM files WHERE {db.LIVE} AND id IN ({','.join('?' * len(chunk))})",
            chunk)})
    out = []
    for ids in members:
        rows = [info[i] for i in ids if i in info]
        if len(rows) < 2 or all(r["kept"] for r in rows):
            continue
        rows.sort(key=keeper_key)
        keep = rows[0]["id"]
        tier = "same" if len({tight.find(r["id"]) for r in rows}) == 1 else "similar"
        # a member is marked when it is a flipped copy of one ranked above it,
        # so of a mirrored pair the better copy stays unmarked
        mirror = [k > 0 and tier == "similar" and any(
            mirrored(rows[j]["dhash"], rows[j]["dhash_flip"], rows[k]["dhash"]) for j in range(k))
            for k in range(len(rows))]
        out.append({
            "tier": tier,
            "mirror": mirror,
            "mirrored": any(mirror),
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
    c["mirrored"] = 0
    for g in all_groups:
        c[g["tier"]] += 1
        c["mirrored"] += g.get("mirrored", False)
    return {**c, "extras": sum(len(g["ids"]) - 1 for g in all_groups),
            "deletable": {t: len(extras(all_groups, t)) for t in TIERS}}


def extras(all_groups, tier):
    """Every file of a tier's groups except each best copy and anything kept:
    what Select all extras takes, and all that a bulk delete may remove."""
    return [i for g in all_groups if g["tier"] == tier
            for i, kept in zip(g["ids"][1:], g["kept"][1:]) if not kept]


def page(conn, tier, offset, limit, cols, only_mirrored=False):
    """One page of a tier's groups, each member as the full item row the grid
    shows (`cols`), plus `keeper`, `dist` and `mirror`. With only_mirrored,
    just the groups holding a mirrored copy."""
    all_groups = groups(conn)
    chosen = [g for g in all_groups if g["tier"] == tier and (g["mirrored"] or not only_mirrored)]
    window = chosen[offset:offset + limit]
    want = [i for g in window for i in g["ids"]]
    rows = {}
    if want:
        marks = ",".join("?" * len(want))
        rows = {r["id"]: dict(r) for r in conn.execute(f"SELECT {cols} FROM files WHERE id IN ({marks})", want)}
    out = []
    for g in window:
        items = []
        for k, (i, d, m) in enumerate(zip(g["ids"], g["dist"], g["mirror"])):
            if i in rows:
                items.append({**rows[i], "keeper": k == 0, "dist": d, "mirror": m})
        out.append({"tier": g["tier"], "items": items})
    return {"groups": out, "total": len(chosen), "offset": offset, "counts": counts(all_groups)}
