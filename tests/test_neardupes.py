"""Near-duplicates from embeddings: linking finds each pair once and only new
pictures after, a changed cut-off relinks, the pairs join the Similar tier only,
groups are capped without cutting hash pairs, and only live files count."""
import numpy as np
import pytest

from photosweep import classifier, config, db, dupes, neardupes

DIM = classifier.DIM


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(config, "DUP_EMBED", 0.96)
    monkeypatch.setattr(config, "DUP_GROUP_MAX", 20)
    return db.connect(config.DB_PATH)


RNG = np.random.default_rng(7)
BASES = {k: RNG.standard_normal(DIM) for k in range(50)}


def vec(base, noise=0.0, seed=0):
    v = BASES[base] + noise * np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)


def add(conn, rel, emb=None, dhash=None, status="done", kept=0, w=1000, h=750, size=100_000):
    cur = conn.execute(
        "INSERT INTO files (rel, size, mtime, status, category, confidence, kept, width, height, dhash, embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (rel, size, "2020-01-01", status, "photo", 90, kept, w, h, dhash,
         classifier.pack(emb) if emb is not None else None))
    return cur.lastrowid


def never():
    return False


def pairs(conn):
    return {(r["a"], r["b"]) for r in conn.execute("SELECT a, b FROM embed_pairs")}


def test_link_finds_each_pair_once_then_only_new_pictures(conn):
    a = add(conn, "a.jpg", vec(0))
    b = add(conn, "b.jpg", vec(0, 0.05, 1))
    add(conn, "c.jpg", vec(1))
    add(conn, "none.jpg")                         # no embedding: never compared
    conn.execute("INSERT INTO files (rel, size, mtime, embedding) VALUES ('bad.jpg', 1, 'x', x'')")  # unembeddable
    assert neardupes.link(conn, never) == (3, 1)
    assert pairs(conn) == {(a, b)}
    sim = conn.execute("SELECT sim FROM embed_pairs").fetchone()[0]
    assert 0.96 <= sim <= 1.0
    assert neardupes.link(conn, never) == (0, 0)
    d = add(conn, "d.jpg", vec(0, 0.05, 2))
    assert neardupes.link(conn, never) == (1, 2)  # the newcomer against everyone, once
    assert pairs(conn) == {(a, b), (a, d), (b, d)}


def test_link_stops_between_blocks_and_resumes(conn, monkeypatch):
    monkeypatch.setattr(neardupes, "BLOCK", 2)
    ids = [add(conn, f"{k}.jpg", vec(0, 0.05, k)) for k in range(5)]
    calls = iter([False, True])
    assert neardupes.link(conn, lambda: next(calls, True))[0] == 2
    assert neardupes.link(conn, never)[0] == 3
    assert len(pairs(conn)) == 10 and min(ids) in {a for a, _ in pairs(conn)}


def test_changing_the_cutoff_relinks_everything(conn, monkeypatch):
    add(conn, "a.jpg", vec(0))
    add(conn, "b.jpg", vec(0, 0.5, 1))  # alike, about 0.89, not 0.96
    neardupes.link(conn, never)
    assert pairs(conn) == set()
    monkeypatch.setattr(config, "DUP_EMBED", 0.5)
    assert neardupes.link(conn, never) == (2, 1)
    monkeypatch.setattr(config, "DUP_EMBED", 0.0)  # off: nothing linked, nothing read
    assert neardupes.link(conn, never) == (0, 0) and neardupes.live_pairs(conn) == []


def test_look_alikes_join_similar_never_same(conn):
    a = add(conn, "a.jpg", vec(0), w=2000, h=1500)
    b = add(conn, "b.jpg", vec(0, 0.05, 1))
    neardupes.link(conn, never)
    g = dupes.groups(conn)
    assert [(x["tier"], x["ids"]) for x in g] == [("similar", [a, b])]
    assert g[0]["dist"] == [None, None]  # no hash distance to show
    assert dupes.extras(g, "same") == []  # Select all extras can never reach them


def test_a_look_alike_never_touches_a_same_group(conn):
    h = "f" * 64
    a = add(conn, "a.jpg", vec(0), dhash=h, w=2000, h=1500)
    b = add(conn, "b.jpg", vec(0, 0.01, 1), dhash=h)
    add(conn, "c.jpg", vec(0, 0.05, 2))           # looks like both, but no hash pair
    d = add(conn, "d.jpg", vec(3))
    e = add(conn, "e.jpg", vec(3, 0.05, 4))
    dupes.link(conn, never)
    neardupes.link(conn, never)
    g = {x["tier"]: x for x in dupes.groups(conn)}
    # the copies stay Same, Select all extras still takes the smaller one
    assert set(g) == {"same", "similar"} and set(g["same"]["ids"]) == {a, b}
    assert dupes.extras(list(g.values()), "same") == [b]
    # look-alikes away from any Same group still group
    assert set(g["similar"]["ids"]) == {d, e}


def test_only_live_pictures_count(conn):
    add(conn, "a.jpg", vec(0))
    add(conn, "b.jpg", vec(0, 0.05, 1), status="deleted")
    neardupes.link(conn, never)
    assert len(pairs(conn)) == 1 and dupes.groups(conn) == []


def test_split_caps_a_chain_and_never_cuts_hash_pairs():
    # a chain of 30 look-alikes, the middle link the weakest
    chain = [(k, k + 1, 0.99 if k != 14 else 0.965) for k in range(29)]
    groups = neardupes.split(chain, 20)
    assert sorted(len(g) for g in groups) == [15, 15]
    # 25 held together by hash pairs alone stay one group whatever the cap
    hashed = [(100 + k, 101 + k, 1.0) for k in range(24)]
    assert [len(g) for g in neardupes.split(hashed, 20)] == [25]
    # a look-alike hanging on a big hash group is shed, the group kept
    mixed = hashed + [(124, 200, 0.97)]
    assert sorted(len(g) for g in neardupes.split(mixed, 20)) == [25]
    # a burst of near-identical frames is sure: kept whole over the cap
    burst = [(300 + i, 300 + j, 0.995) for i in range(30) for j in range(i + 1, 30)]
    assert [len(g) for g in neardupes.split(burst, 20)] == [30]


def test_groups_respect_the_cap_but_keep_a_tight_burst(conn, monkeypatch):
    monkeypatch.setattr(config, "DUP_GROUP_MAX", 3)
    burst = [add(conn, f"b{k}.jpg", vec(0, 0.02, k)) for k in range(6)]   # about 0.9997 alike
    neardupes.link(conn, never)
    assert [sorted(x["ids"]) for x in dupes.groups(conn)] == [sorted(burst)]
    # two tight trios bridged by one looser look-alike: the bridge is cut
    conn.execute("DELETE FROM files")
    conn.execute("DELETE FROM embed_pairs")
    mid = vec(1) + vec(2)
    trio_a = [add(conn, f"a{k}.jpg", vec(1, 0.02, k)) for k in range(3)]
    trio_b = [add(conn, f"c{k}.jpg", vec(2, 0.02, k)) for k in range(3)]
    monkeypatch.setattr(config, "DUP_EMBED", 0.7)
    add(conn, "bridge.jpg", mid / np.linalg.norm(mid))
    neardupes.link(conn, never)
    got = sorted(sorted(x["ids"]) for x in dupes.groups(conn))
    assert got == sorted([sorted(trio_a), sorted(trio_b)])
