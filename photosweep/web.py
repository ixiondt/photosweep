"""The review page: the app, its headers and error handlers, and the static
page. The routes are in reads.py (GET) and writes.py (POST); the gates they
share are in gate.py.
"""
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config, reads, signin, writes  # noqa: F401  (config: tests read web.config)
from .gate import api_error, signed_in

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

log = logging.getLogger("photosweep.web")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format='{"t":"%(asctime)s","lvl":"%(levelname)s","src":"%(name)s","msg":"%(message)s"}')

app = FastAPI(title="Photo sweep", docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(HTTPException)
def _http_exc(request, exc):
    code = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND",
            409: "CONFLICT", 415: "UNSUPPORTED_MEDIA_TYPE", 429: "TOO_MANY_REQUESTS"}.get(exc.status_code, "ERROR")
    return api_error(exc.status_code, code, str(exc.detail), trace=getattr(request.state, "trace", None))


@app.exception_handler(Exception)
def _unhandled(request, exc):
    trace = getattr(request.state, "trace", None)
    log.error("unhandled trace=%s path=%s err=%s", trace, request.url.path, f"{type(exc).__name__}: {exc}")
    return api_error(500, "INTERNAL", "Something went wrong.", retryable=True, trace=trace)


@app.middleware("http")
async def _headers(request, call_next):
    request.state.trace = request.headers.get("x-trace-id") or str(uuid.uuid4())
    resp = await call_next(request)
    resp.headers.update({
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "same-origin",
        "Content-Security-Policy": (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
            "connect-src 'self'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'"),
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "X-Trace-Id": request.state.trace,
    })
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


# --------------------------------------------------------------- pages ----

@app.get("/")
def index(request: Request):
    if signed_in(request) is None:
        if config.AUTH_MODE == "password":
            return RedirectResponse("/login", status_code=303)
        raise HTTPException(status_code=401, detail="sign in first")
    return FileResponse(os.path.join(STATIC, "index.html"), media_type="text/html")


@app.get("/login")
def login_page(request: Request):
    if config.AUTH_MODE != "password" or signed_in(request) is not None:
        return RedirectResponse("/", status_code=303)
    return FileResponse(os.path.join(STATIC, "login.html"), media_type="text/html")


@app.get("/healthz")
def healthz():
    """Liveness for the container healthcheck: the process answers. No data."""
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


app.include_router(signin.router)
app.include_router(reads.router)
app.include_router(writes.router)
