"""Who may ask, and in what shape an answer or refusal goes back: the
operator gate, the same-origin write gate, the error envelope, and the
per-thread database connection every route shares.

Who: in password mode, a signed session cookie (auth.py); in header mode, the
name a signing-in proxy puts in AUTH_HEADER, which must be in OPERATORS (empty
denies everyone). Anything else denies everyone. A cookie or a proxy header
rides along on cross-site requests too, so every POST must also carry a
same-host Origin and the X-Photosweep header, which a cross-site form cannot
send without a preflight this app never answers.
"""
import json
import threading

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from . import auth, config, db

OWNER = "owner"  # who the audit log names in password mode
_local = threading.local()


def conn():
    c = getattr(_local, "conn", None)
    if c is None:
        c = _local.conn = db.connect(config.DB_PATH)
    return c


# ------------------------------------------------------------ envelope ----

def api_error(status, code, message, retryable=False, trace=None, headers=None):
    body = {"code": code, "message": message, "retryable": retryable}
    if trace:
        body["traceId"] = trace
    return JSONResponse(body, status_code=status, headers=headers)


# ---------------------------------------------------------------- auth ----

def signed_in(request: Request):
    """Who is asking, or None."""
    if config.AUTH_MODE == "header":
        who = request.headers.get(config.AUTH_HEADER, "")
        return who if who and who in config.OPERATORS else None
    if config.AUTH_MODE == "password":
        return OWNER if auth.valid_session(request.cookies.get(auth.COOKIE)) else None
    return None  # an unknown mode is a typo: fail closed


def operator(request: Request) -> str:
    who = signed_in(request)
    if who is None:
        raise HTTPException(status_code=401, detail="sign in first")
    return who


async def mutation(request: Request):
    """Operator, same origin, custom header, JSON body. Returns (who, body)."""
    who = operator(request)
    return who, await same_origin_json(request)


async def same_origin_json(request: Request):
    """The write gate without the operator check (sign-in itself uses it)."""
    origin = request.headers.get("origin", "")
    if not origin or origin.split("://", 1)[-1] != request.headers.get("host", ""):
        raise HTTPException(status_code=403, detail="cross-origin request refused")
    if request.headers.get("x-photosweep") != "1":
        raise HTTPException(status_code=403, detail="missing photosweep header")
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(status_code=415, detail="send application/json")
    try:
        body = json.loads(await request.body())
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    return body


def ids_from(body, cap):
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise HTTPException(status_code=400, detail="ids must be a non-empty list of integers")
    if len(ids) > cap:
        raise HTTPException(status_code=400, detail=f"at most {cap} at a time")
    return sorted(set(ids))
