"""SQLite store shared by the scanner and the web app (two processes).

WAL plus a busy timeout lets them interleave. Every transaction that reads and
then writes opens with BEGIN IMMEDIATE: a deferred one would take a snapshot at
the read and fail with SQLITE_BUSY_SNAPSHOT if the other process committed in
between, which no busy handler retries.

Timestamps are ISO-8601 UTC text everywhere. Paths (`rel`) are relative to
PICS_ROOT with forward slashes, which is also their WebDAV path under NC_PREFIX.
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    rel         TEXT NOT NULL UNIQUE,
    size        INTEGER NOT NULL,
    mtime       TEXT NOT NULL,
    -- pending | done | error | gone (vanished from disk) | deleted (by us)
    status      TEXT NOT NULL DEFAULT 'pending',
    category    TEXT,
    confidence  INTEGER,
    reason      TEXT,
    camera      TEXT,
    width       INTEGER,
    height      INTEGER,
    name_hint   TEXT,
    kept        INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    model_ms    INTEGER,
    scanned_at  TEXT,
    deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS files_review ON files (status, category, kept, confidence);
CREATE INDEX IF NOT EXISTS files_scanned ON files (scanned_at);

-- Pairs of files whose pictures are within DUP_SIMILAR bits (dupes.link).
-- Kept for files later deleted or gone; every read joins files for status.
CREATE TABLE IF NOT EXISTS dup_pairs (
    a     INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    b     INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    dist  INTEGER NOT NULL,
    PRIMARY KEY (a, b),
    CHECK (a < b)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS dup_pairs_b ON dup_pairs (b);

-- Pairs whose classifier embeddings are at least DUP_EMBED alike (neardupes.link):
-- the near-duplicates the hash misses. Read like dup_pairs, Similar tier only.
CREATE TABLE IF NOT EXISTS embed_pairs (
    a     INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    b     INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
    sim   REAL NOT NULL,
    PRIMARY KEY (a, b),
    CHECK (a < b)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS embed_pairs_b ON embed_pairs (b);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Append-only and hash-chained: each row's hash covers the previous row's.
CREATE TABLE IF NOT EXISTS audit (
    id      INTEGER PRIMARY KEY,
    at      TEXT NOT NULL,
    who     TEXT NOT NULL,
    action  TEXT NOT NULL,
    detail  TEXT NOT NULL,
    prev    TEXT NOT NULL,
    hash    TEXT NOT NULL UNIQUE
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
    BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
    BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""

GENESIS = "0" * 64


# A file still in the library, as Duplicates and Keep see it: labelled, or
# waiting for a label (with MODEL=none, waiting for good). A pending row only
# ever has a dhash, and so a duplicate pair, once it has been fingerprinted.
LIVE = "status IN ('pending','done')"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    conn = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Columns added after the first deploy. CREATE TABLE IF NOT EXISTS leaves an
# existing table as it was, so each one is added here if it is missing.
_ADDED = [
    ("sharpness", "REAL"),   # imaging.sharpness() of the thumbnail; NULL = not measured
    # the model's blur verdict on a photo: NULL = not asked yet, 1 blurry,
    # 0 sharp, -1 asked but no usable answer (falls back to the pixel score)
    ("blurry", "INTEGER"),
    ("blur_conf", "INTEGER"),
    # 1 = the operator pressed Recheck: ask again ahead of the queue, from the
    # original at RECHECK_PX rather than the 512 px thumbnail
    ("blur_recheck", "INTEGER NOT NULL DEFAULT 0"),
    # the model's clutter verdict on a photo (classify.judge_clutter), same
    # encoding as `blurry`: NULL not asked, 1 clutter, 0 keeper, -1 no answer
    ("clutter", "INTEGER"),
    ("clutter_conf", "INTEGER"),
    ("clutter_reason", "TEXT"),
    # 1 = the operator pressed Recheck: ask the clutter question again, ahead
    # of the queue and from the original at RECHECK_PX, as blur_recheck does
    ("clutter_recheck", "INTEGER NOT NULL DEFAULT 0"),
    # imaging.dhash() of the thumbnail, 64 hex chars; NULL = not hashed yet
    ("dhash", "TEXT"),
    # 1 = compared against every other hashed file and its pairs stored
    ("dup_linked", "INTEGER NOT NULL DEFAULT 0"),
    # the image classifier (classifier.py): the thumbnail's embedding, float16,
    # and what the heads made of it, kept whether or not it decided anything
    ("embedding", "BLOB"),
    ("cpu_label", "TEXT"),
    ("cpu_label_p", "REAL"),
    ("cpu_clutter_p", "REAL"),
    ("cpu_blur_p", "REAL"),
    # who settled each answer: 'model' (the vision model) or 'classifier'.
    # NULL on rows from before the classifier, which were all the model's.
    # train_heads learns only from the model's answers and the owner's actions.
    ("label_src", "TEXT"),
    ("clutter_src", "TEXT"),
    ("blur_src", "TEXT"),
    # 1 = embedding compared against every other one and its pairs stored
    ("embed_linked", "INTEGER NOT NULL DEFAULT 0"),
]


def _migrate(conn):
    have = {r["name"] for r in conn.execute("PRAGMA table_info(files)")}
    for name, decl in _ADDED:
        if name in have:
            continue
        try:
            conn.execute(f"ALTER TABLE files ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as exc:
            # the scanner and the web app start together; the other one won
            if "duplicate column" not in str(exc):
                raise
    conn.execute("CREATE INDEX IF NOT EXISTS files_blur ON files (status, kept, category, sharpness)")
    conn.execute("CREATE INDEX IF NOT EXISTS files_blur_verdict ON files (status, kept, category, blurry)")
    conn.execute("CREATE INDEX IF NOT EXISTS files_clutter ON files (status, kept, category, clutter)")
    conn.execute("CREATE INDEX IF NOT EXISTS files_unlinked ON files (dup_linked) WHERE dhash IS NOT NULL")
    # the fingerprint stage's queue: on disk, not yet decoded
    conn.execute("CREATE INDEX IF NOT EXISTS files_unprinted ON files (id) WHERE status='pending' AND dhash IS NULL")
    # the embedding backfill's queue: rows from before the classifier
    conn.execute("CREATE INDEX IF NOT EXISTS files_unembedded ON files (id) WHERE embedding IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS files_embed_unlinked ON files (id) WHERE embed_linked=0")


@contextmanager
def immediate(conn):
    """A write transaction that takes the write lock up front."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn, key, value):
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value=excluded.value", (key, str(value)))


def chain_hash(prev, at, who, action, detail):
    payload = json.dumps([prev, at, who, action, detail], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit(conn, who, action, detail):
    """Append one audit row. Must be called inside an immediate() transaction so
    two writers cannot read the same tail and fork the chain."""
    detail_s = json.dumps(detail, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    row = conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS
    at = now()
    h = chain_hash(prev, at, who, action, detail_s)
    conn.execute("INSERT INTO audit (at, who, action, detail, prev, hash) VALUES (?,?,?,?,?,?)",
                 (at, who, action, detail_s, prev, h))
    return h


def verify_audit(conn):
    """(ok, first_bad_id). Recomputes the whole chain."""
    prev = GENESIS
    for r in conn.execute("SELECT * FROM audit ORDER BY id"):
        if r["prev"] != prev or chain_hash(prev, r["at"], r["who"], r["action"], r["detail"]) != r["hash"]:
            return False, r["id"]
        prev = r["hash"]
    return True, None
