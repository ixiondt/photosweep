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

# The model server. MODEL_API says which protocol it speaks:
#   ollama  the Ollama chat API: Ollama itself, or a proxy in front of one (default)
#   openai  an OpenAI-compatible chat completions API: LM Studio, vLLM, or a
#           cloud service. MODEL_URL is the base that `/chat/completions`
#           hangs off, e.g. https://api.example.com/v1
# The token file is optional; plain Ollama takes no token. A cloud service's
# API key goes in it.
MODEL_API = _env("MODEL_API", "ollama").strip().lower()
ROUTER_URL = _env("MODEL_URL", _env("OLLAMA_URL", _env("ROUTER_URL", "http://ollama:11434"))).rstrip("/")
ROUTER_TOKEN_FILE = _env("MODEL_KEY_FILE", _env("OLLAMA_TOKEN_FILE", _env("ROUTER_TOKEN_FILE", "")))
MODEL = _env("MODEL", "qwen3-vl:8b")
# MODEL=none: no model at all. Every picture is still fingerprinted, so
# Duplicates works in full; the label tabs, Blurry and Clutter stay off.
LABELLING = MODEL.strip().lower() not in ("", "none")
# Pictures decoded at once while fingerprinting (thumbnail, sharpness, hash).
# Decoding is the whole cost, and Pillow releases the GIL while it decodes.
FINGERPRINT_WORKERS = max(1, int(_env("FINGERPRINT_WORKERS", str(min(4, os.cpu_count() or 1)))))
LINK_EVERY = 2000  # fingerprints between duplicate-linking passes during the first sweep
# Stage 2, the image classifier on the CPU (classifier.py). "none" turns it off.
# With MODEL=none it sorts everything itself; with a vision model as well, it
# settles what it is sure of (CLASSIFIER_TRUST) and leaves the rest to the model.
CLASSIFIER = _env("CLASSIFIER", "siglip").strip().lower()
CLASSIFIER_MODEL = _env("CLASSIFIER_MODEL", "/app/models/siglip-b16-int8.onnx")
CLASSIFIER_TRUST = float(_env("CLASSIFIER_TRUST", "0.95"))
# Heads trained on this library (python -m photosweep.train_heads writes here);
# used when present, else the heads bundled with the code.
CLASSIFIER_HEADS = _env("CLASSIFIER_HEADS", os.path.join(DATA_DIR, "heads.npz"))
# Blurry's exposure filter (exposure.py): too dark is an average grey under
# EXPOSE_DARK_MEAN (0-255) or at least EXPOSE_DARK_SHARE of the picture crushed
# to black; too bright, over EXPOSE_BRIGHT_MEAN or EXPOSE_BRIGHT_SHARE blown.
EXPOSE_DARK_MEAN = float(_env("EXPOSE_DARK_MEAN", "50"))
EXPOSE_DARK_SHARE = float(_env("EXPOSE_DARK_SHARE", "0.5"))
EXPOSE_BRIGHT_MEAN = float(_env("EXPOSE_BRIGHT_MEAN", "190"))
EXPOSE_BRIGHT_SHARE = float(_env("EXPOSE_BRIGHT_SHARE", "0.25"))
# Automatic retraining (retrain.py): when the scanner is idle, at most once per
# RETRAIN_EVERY_S, and only once RETRAIN_MIN_NEW new answers (the owner's keeps
# and deletes, the vision model's labels) have arrived since the heads in use
# were trained. New heads replace the old only if they beat them on those new
# answers; the old are kept beside them as <heads>.prev.npz. "0" turns it off.
AUTO_RETRAIN = _env("AUTO_RETRAIN", "1").strip() != "0"
RETRAIN_MIN_NEW = int(_env("RETRAIN_MIN_NEW", "200"))
RETRAIN_EVERY_S = int(_env("RETRAIN_EVERY_S", str(24 * 3600)))
RETRAIN_TIMEOUT_S = int(_env("RETRAIN_TIMEOUT_S", "3600"))
MODEL_TIMEOUT_S = float(_env("MODEL_TIMEOUT_S", "180"))
# openai only: how the JSON answer is asked for. "schema" (structured output),
# "object" (JSON mode) or "none" (the prompt alone), for services that refuse one.
MODEL_JSON = _env("MODEL_JSON", "schema").strip().lower()
# Seconds between model requests at least. Free cloud tiers cap requests per
# minute: 15 a minute is 4. 0 leaves pacing to the server's 429s.
MODEL_MIN_INTERVAL_S = float(_env("MODEL_MIN_INTERVAL_S", "0"))
# A model server outside the local network sees every photo it labels. The
# scanner refuses one until this is exactly "yes" (remote.py).
def _is_yes(value):
    """Only the word yes is consent: not true, 1, y or a typo of yes."""
    return (value or "").strip().lower() == "yes"


PHOTOS_LEAVE_THIS_SERVER = _is_yes(_env("PHOTOS_LEAVE_THIS_SERVER", ""))

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
# A Similar member is a mirrored copy of its group's best copy when its hash is
# within this many bits of the best copy's flipped hash (imaging.dhash_flipped)
# and closer that way than as it is.
DUP_MIRROR = int(_env("DUP_MIRROR", "24"))
# Near-duplicates from the classifier's embeddings (neardupes.py): pairs at
# least this alike (cosine) join the Similar tier. 0 turns it off.
DUP_EMBED = float(_env("DUP_EMBED", "0.96"))
# No group larger than this: a bigger one is re-cut at a higher similarity.
DUP_GROUP_MAX = int(_env("DUP_GROUP_MAX", "20"))
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
