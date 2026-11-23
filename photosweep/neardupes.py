"""Near-duplicates the perceptual hash misses, from the classifier's embeddings.

The hash (dupes.py) is a picture of the thumbnail's brightness edges, so a
crop, a rotation, a filter or a burst frame with a little movement lands far
from its twin. The classifier already stores a 768-number embedding of every
picture (classifier.py), in which pictures that look alike sit close together.
link() stores every pair of pictures whose embeddings are at least DUP_EMBED
alike (cosine), and dupes.groups() joins them into the **Similar** tier only:
an embedding pair never makes a Same group, so it can never feed Select all
extras, and choosing the copy to keep stays with pixels, camera and size.

Measured 2026-11-23 on a 26,551-picture live library (counts only): at 0.96
the embeddings found 2,547 pairs the hash had not, 81% taken within an hour of
each other and 97% in one folder, and all of them photo pairs; 98% of the
hash's own Same pairs were that alike. Below 0.95 groups chained (146, then 529
pictures in one), the failure that ruled out a looser hash. So groups are also
capped (split()): one over DUP_GROUP_MAX is re-cut at a higher similarity
until no group is larger, which left 1,571 groups, 96% of them in one folder.

Linking is numpy: about 5 s for the whole library on two cores, then only new
pictures, each against all.
"""
import logging

import numpy as np

from . import classifier, config, db

log = logging.getLogger("photosweep.neardupes")

BLOCK = 256    # unlinked pictures compared at once: 256 x 40,000 floats is 40 MB
STEP = 0.005   # how far split() raises the cut-off each time
# Pairs this alike are never cut by the cap: the hash's own Same pairs have a
# median of 0.987, so these are as sure as copies. Without it a long burst of
# near-identical frames was re-cut until nothing was left of it.
SURE = 0.99


def enabled():
    return config.DUP_EMBED > 0


def _reset_if_cutoff_changed(conn):
    """Pairs are stored only at DUP_EMBED or above, so a new cut-off needs them
    all recomputed. Read first; the write lock only when something changed."""
    want = repr(config.DUP_EMBED)
    if db.get_setting(conn, "embed_link_min") == want:
        return False
    with db.immediate(conn):
        if db.get_setting(conn, "embed_link_min") == want:
            return False
        conn.execute("DELETE FROM embed_pairs")
        conn.execute("UPDATE files SET embed_linked=0 WHERE embed_linked=1")
        db.set_setting(conn, "embed_link_min", want)
    log.info("near-duplicate cutoff is now %s; relinking everything", want)
    return True


def link(conn, should_stop):
    """Compare each embedded-but-unlinked picture with every embedded picture
    before it (all linked ones, and unlinked ones with a smaller id), and store
    the pairs at DUP_EMBED or above. Each pair is found once. Commits a block at
    a time, so a stop loses at most a block. Returns (pictures linked, pairs)."""
    if not enabled():
        return 0, 0
    _reset_if_cutoff_changed(conn)
    size = classifier.DIM * 2
    if conn.execute("SELECT 1 FROM files WHERE embed_linked=0 AND length(embedding)=? LIMIT 1",
                    (size,)).fetchone() is None:
        return 0, 0
    rows = conn.execute("SELECT id, embed_linked, embedding FROM files WHERE length(embedding)=? ORDER BY id",
                        (size,)).fetchall()
    ids = np.array([r["id"] for r in rows])
    done = np.array([bool(r["embed_linked"]) for r in rows])
    X = np.stack([classifier.unpack(r["embedding"]) for r in rows]).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    del rows
    todo = np.nonzero(~done)[0]
    cut = config.DUP_EMBED
    linked = stored = 0
    for s in range(0, len(todo), BLOCK):
        if should_stop():
            break
        block = todo[s:s + BLOCK]
        sims = X[block] @ X.T
        # a pair is found by its later unlinked member: skip unlinked j at or after i
        later = ~done[None, :] & (np.arange(len(ids))[None, :] >= block[:, None])
        sims[later] = -1.0
        bi, j = np.nonzero(sims >= cut)
        a, b = ids[block[bi]], ids[j]
        edges = [(int(min(x, y)), int(max(x, y)), round(float(v), 4))
                 for x, y, v in zip(a, b, sims[bi, j])]
        marks = ",".join("?" * len(block))
        with db.immediate(conn):
            conn.executemany("INSERT INTO embed_pairs (a, b, sim) VALUES (?, ?, ?) "
                             "ON CONFLICT (a, b) DO UPDATE SET sim = excluded.sim", edges)
            conn.execute(f"UPDATE files SET embed_linked=1 WHERE id IN ({marks})", [int(i) for i in ids[block]])
        done[block] = True
        linked += len(block)
        stored += len(edges)
    return linked, stored


def live_pairs(conn):
    """[(a, b, sim)] between pictures still on disk, at the current cut-off."""
    if not enabled():
        return []
    return conn.execute(
        "SELECT p.a, p.b, p.sim FROM embed_pairs p "
        f"JOIN files fa ON fa.id = p.a AND fa.{db.LIVE} "
        f"JOIN files fb ON fb.id = p.b AND fb.{db.LIVE} WHERE p.sim >= ?",
        (config.DUP_EMBED,)).fetchall()


def split(edges, cap):
    """Groups (sets of ids) from weighted edges [(a, b, weight)], none larger
    than `cap` unless held together by sure edges: a component over the cap is
    cut again keeping only its edges above a higher weight, until it falls
    apart into small enough pieces. Edges of weight SURE or more are never cut,
    so the hash's groups (weight 1.0) and tight bursts stay whole whatever
    their size; the cap breaks only chains of looser look-alikes."""
    out = []
    work = [(edges, min((w for _, _, w in edges), default=1.0))]
    while work:
        es, floor = work.pop()
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b, _ in es:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        comps = {}
        for e in es:
            comps.setdefault(find(e[0]), []).append(e)
        for ce in comps.values():
            members = {x for a, b, _ in ce for x in (a, b)}
            if len(members) <= cap or floor >= SURE:
                out.append(members)
                continue
            # only edges strictly above the next cut survive; anything left
            # with no edge at all drops out of every group
            nxt = floor + STEP
            work.append(([e for e in ce if e[2] >= nxt], nxt))
    return out
