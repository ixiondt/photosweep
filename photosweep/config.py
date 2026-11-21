"""Every setting in one place, read from the environment.

Secrets are never in the environment: the model token, the Nextcloud app
password, the sign-in password hash and the session key are files, read on
use, so rotating any of them needs no restart.
"""
import os


def _env(name, default):
    return os.environ.get(name, default)


PICS_ROOT = _env("PICS_ROOT", "/photos")            # read-only view of the Nextcloud folder
DATA_DIR = _env("DATA_DIR", "/data")                # sqlite, thumbnails, secrets
DB_PATH = os.path.join(DATA_DIR, "photosweep.db")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
THUMB_PX = int(_env("THUMB_PX", "512"))             # what the model sees and the grid shows
RECHECK_PX = int(_env("RECHECK_PX", "1024"))       # what a Recheck shows the blur question
VIEW_PX = int(_env("VIEW_PX", "1600"))              # the larger view in the review dialog

# Any server speaking the Ollama chat API: Ollama itself, or a proxy in front
# of one. The token file is optional; plain Ollama takes no token.
ROUTER_URL = _env("OLLAMA_URL", _env("ROUTER_URL", "http://ollama:11434")).rstrip("/")
ROUTER_TOKEN_FILE = _env("OLLAMA_TOKEN_FILE", _env("ROUTER_TOKEN_FILE", ""))
MODEL = _env("MODEL", "qwen3-vl:8b")
MODEL_TIMEOUT_S = float(_env("MODEL_TIMEOUT_S", "180"))

NC_BASE = _env("NC_BASE", "")                       # https://cloud.example.com
NC_USER = _env("NC_USER", "")                       # the Nextcloud user whose files these are
NC_PREFIX = _env("NC_PREFIX", "Photos")             # the folder PICS_ROOT mirrors, under that user's files
NC_SECRET_FILE = os.path.join(DATA_DIR, "nextcloud.secret")

# Who may use the page.
#   password: one password, set with `python -m photosweep.passwd`; a signed cookie after.
#   header:   a reverse proxy that has already signed the user in passes their
#             name in AUTH_HEADER (Tailscale serve, Authelia, Authentik...), and
#             it must be one of OPERATORS. Only safe when nothing but that
#             proxy can reach the port.
AUTH_MODE = _env("AUTH_MODE", "password")
AUTH_HEADER = _env("AUTH_HEADER", "Tailscale-User-Login").lower()
OPERATORS = {e.strip() for e in _env("OPERATORS", "").split(",") if e.strip()}
PASSWORD_FILE = os.path.join(DATA_DIR, "password.hash")
SESSION_KEY_FILE = os.path.join(DATA_DIR, "session.key")
SESSION_HOURS = float(_env("SESSION_HOURS", "12"))
PASSWORD_MIN = 15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".bmp"}
BLUR_MAX = float(_env("BLUR_MAX", "18"))  # photos scoring under this are listed as Blurry until asked
# Duplicates (dupes.py), in bits of the 256-bit imaging.dhash(). Measured on a
# 32,000-photo library: copies cluster at 0-12 bits, burst shots at ~15-40; at
# 50 unrelated photos began chaining into one group.
DUP_SAME = int(_env("DUP_SAME", "12"))
DUP_SIMILAR = int(_env("DUP_SIMILAR", "40"))
MAX_ATTEMPTS = 3         # per file, for failures that are the file's fault
REWALK_S = int(_env("REWALK_S", str(6 * 3600)))  # pick up photos added while the sweep is live


def read_secret(path):
    """A secret file's first line, or None. Missing is a normal state (not yet
    configured); anything else is logged by the caller."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            value = f.readline().strip()
    except FileNotFoundError:
        return None
    return value or None


def nextcloud_configured():
    return bool(NC_BASE and NC_USER)
