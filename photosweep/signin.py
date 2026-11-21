"""Sign-in and sign-out for AUTH_MODE=password. In header mode the proxy
signs people in, and these routes refuse."""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import auth, config, db
from .gate import api_error, conn, same_origin_json, signed_in

log = logging.getLogger("photosweep.signin")
router = APIRouter()


@router.get("/api/auth")
def auth_state(request: Request):
    """What the sign-in page needs to know. Safe to answer anyone."""
    return {"mode": config.AUTH_MODE,
            "passwordSet": config.AUTH_MODE == "password" and auth.stored_hash() is not None,
            "signedIn": signed_in(request) is not None}


@router.post("/api/login")
async def login(request: Request):
    if config.AUTH_MODE != "password":
        return api_error(409, "PASSWORD_SIGNIN_OFF", "Sign-in is handled in front of this app.")
    body = await same_origin_json(request)
    client = request.client.host if request.client else "?"
    wait = auth.retry_after(client)
    if wait:
        return api_error(429, "TOO_MANY_ATTEMPTS", f"Too many tries. Wait {wait} s.", retryable=True,
                         headers={"Retry-After": str(wait)})
    stored = auth.stored_hash()
    if stored is None:
        return api_error(409, "PASSWORD_NOT_SET",
                         "No password yet. Run: docker compose exec web python -m photosweep.passwd")
    password = body.get("password")
    if not isinstance(password, str) or not auth.check_password(password, stored):
        auth.record_failure(client)
        log.warning("sign-in failed client=%s", client)
        return api_error(401, "WRONG_PASSWORD", "That password is not right.")
    token, exp = auth.new_session()
    c = conn()
    with db.immediate(c):
        db.audit(c, "owner", "sign-in", {"client": client})
    resp = JSONResponse({"ok": True})
    resp.set_cookie(auth.COOKIE, token, max_age=int(config.SESSION_HOURS * 3600), httponly=True,
                    samesite="strict", secure=request.url.scheme == "https", path="/")
    return resp


@router.post("/api/logout")
async def logout(request: Request):
    await same_origin_json(request)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp
