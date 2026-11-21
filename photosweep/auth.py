"""Password sign-in: one password for the one person whose photos these are.

The hash (scrypt) and the session key live in DATA_DIR as files. A session is a
signed cookie carrying its expiry and a fingerprint of the password hash, so
setting a new password signs every browser out. Failed attempts are limited
per client address and in total; the total is what holds when the address is
a proxy's.
"""
import base64
import collections
import hashlib
import hmac
import os
import secrets
import threading
import time

from . import config

COOKIE = "photosweep_session"
_N, _R, _P = 2 ** 14, 8, 1          # scrypt cost: ~16 MiB, ~50 ms
PER_CLIENT = (5, 60.0)              # failures, seconds
OVERALL = (30, 60.0)
_fails = collections.defaultdict(collections.deque)
_lock = threading.Lock()


# ----------------------------------------------------------- password ----

def hash_password(password):
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, maxmem=64 * 1024 * 1024)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(digest)}"


def check_password(password, stored):
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        want = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=64 * 1024 * 1024, dklen=len(want))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, want)


def password_problem(password):
    """Why a new password is refused, or None. Length is the rule (NIST
    SP 800-63B: 15 for a single factor); no composition rules."""
    if len(password) < config.PASSWORD_MIN:
        return f"use at least {config.PASSWORD_MIN} characters; a few ordinary words make a good one"
    if len(password) > 256:
        return "use at most 256 characters"
    if len(set(password)) < 4:
        return "that repeats too few characters to be a password"
    return None


def stored_hash():
    return config.read_secret(config.PASSWORD_FILE)


def _write_private(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(tmp, path)


def set_password(password):
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    _write_private(config.PASSWORD_FILE, hash_password(password))


# ------------------------------------------------------------ session ----

def _key():
    key = config.read_secret(config.SESSION_KEY_FILE)
    if key:
        return key.encode()
    try:
        fd = os.open(config.SESSION_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another process made it first
        return config.read_secret(config.SESSION_KEY_FILE).encode()
    key = secrets.token_hex(32)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    return key.encode()


def _fingerprint(stored):
    return hashlib.sha256(stored.encode()).hexdigest()[:16]


def _sign(exp, stored):
    return hmac.new(_key(), f"{exp}.{_fingerprint(stored)}".encode(), hashlib.sha256).hexdigest()


def new_session():
    stored = stored_hash()
    exp = int(time.time() + config.SESSION_HOURS * 3600)
    return f"v1.{exp}.{_sign(exp, stored)}", exp


def valid_session(token):
    stored = stored_hash()
    if not token or not stored:
        return False
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1" or not parts[1].isdigit():
        return False
    exp = int(parts[1])
    if exp < time.time():
        return False
    return hmac.compare_digest(parts[2], _sign(exp, stored))


# --------------------------------------------------------- rate limit ----

def _trim(q, window, now):
    while q and now - q[0] > window:
        q.popleft()


def retry_after(client):
    """Seconds until this client may try again, or 0."""
    now = time.monotonic()
    with _lock:
        waits = []
        for key, (limit, window) in ((client, PER_CLIENT), ("*", OVERALL)):
            q = _fails[key]
            _trim(q, window, now)
            if len(q) >= limit:
                waits.append(window - (now - q[0]))
        return int(max(waits)) + 1 if waits else 0


def record_failure(client):
    now = time.monotonic()
    with _lock:
        for key in (client, "*"):
            _fails[key].append(now)
        if len(_fails) > 10_000:  # bound the map: drop clients with nothing recent
            for k in [k for k, q in _fails.items() if k != "*" and (not q or now - q[-1] > PER_CLIENT[1])]:
                del _fails[k]


def reset_limits():
    with _lock:
        _fails.clear()
