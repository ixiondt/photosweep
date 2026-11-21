"""Sign-in: password mode, header mode, and every way in that must stay shut."""
import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

WRITE = {"Origin": "http://testserver", "X-Photosweep": "1", "Content-Type": "application/json"}
GOOD = "correct horse battery staple"


def _app(tmp_path, monkeypatch, mode, **env):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PICS_ROOT", str(tmp_path / "pics"))
    monkeypatch.setenv("AUTH_MODE", mode)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for name in [m for m in sys.modules if m.startswith("photosweep")]:
        del sys.modules[name]
    web = importlib.import_module("photosweep.web")
    return web, TestClient(web.app)


@pytest.fixture
def pw(tmp_path, monkeypatch):
    web, client = _app(tmp_path, monkeypatch, "password")
    from photosweep import auth
    auth.reset_limits()
    return client, auth


def login(client, password, headers=WRITE):
    return client.post("/api/login", headers=headers, content=json.dumps({"password": password}))


def test_signed_out_sees_only_the_sign_in_page(pw):
    client, _ = pw
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/login").status_code == 200
    assert client.get("/api/summary").status_code == 401
    assert client.get("/api/items?tab=photo").status_code == 401
    assert client.get("/healthz").json() == {"ok": True}


def test_no_password_yet_says_how_to_set_one(pw):
    client, _ = pw
    assert client.get("/api/auth").json() == {"mode": "password", "passwordSet": False, "signedIn": False}
    r = login(client, GOOD)
    assert r.status_code == 409 and r.json()["code"] == "PASSWORD_NOT_SET"


def test_sign_in_then_out(pw):
    client, auth = pw
    auth.set_password(GOOD)
    assert login(client, "wrong password entirely").status_code == 401
    assert client.get("/api/summary").status_code == 401
    r = login(client, GOOD)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    s = client.get("/api/summary").json()
    assert s["signOut"] is True and s["operator"] == "owner"
    assert client.post("/api/logout", headers=WRITE, content="{}").status_code == 200
    assert client.get("/api/summary").status_code == 401


def test_a_new_password_signs_everyone_out(pw):
    client, auth = pw
    auth.set_password(GOOD)
    login(client, GOOD)
    assert client.get("/api/summary").status_code == 200
    auth.set_password("an entirely different passphrase")
    assert client.get("/api/summary").status_code == 401


def test_forged_and_expired_sessions_are_refused(pw, monkeypatch):
    client, auth = pw
    auth.set_password(GOOD)
    token, exp = auth.new_session()
    assert auth.valid_session(token)
    v, e, sig = token.split(".")
    assert not auth.valid_session(f"{v}.{int(e) + 999999}.{sig}")  # a longer life, same signature
    assert not auth.valid_session(f"{v}.{e}.{'0' * len(sig)}")
    assert not auth.valid_session("garbage")
    monkeypatch.setattr(auth.time, "time", lambda: exp + 1)
    assert not auth.valid_session(token)


def test_guessing_is_rate_limited(pw):
    client, auth = pw
    auth.set_password(GOOD)
    for _ in range(auth.PER_CLIENT[0]):
        assert login(client, "not the password at all").status_code == 401
    r = login(client, GOOD)  # even the right one, until the window passes
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert r.json()["code"] == "TOO_MANY_ATTEMPTS"


def test_sign_in_refuses_cross_site_and_wrong_shapes(pw):
    client, auth = pw
    auth.set_password(GOOD)
    assert login(client, GOOD, {**WRITE, "Origin": "https://evil.example"}).status_code == 403
    assert login(client, GOOD, {k: v for k, v in WRITE.items() if k != "X-Photosweep"}).status_code == 403
    assert client.post("/api/login", headers=WRITE, content="[]").status_code == 400
    assert login(client, 12345).status_code == 401


def test_a_proxy_header_means_nothing_in_password_mode(pw):
    client, auth = pw
    auth.set_password(GOOD)
    h = {"Tailscale-User-Login": "owner", "Remote-User": "owner"}
    assert client.get("/api/summary", headers=h).status_code == 401


def test_short_passwords_are_refused(pw):
    _, auth = pw
    for bad in ("short", "aaaaaaaaaaaaaaaaaaaa", "x" * 300):
        with pytest.raises(ValueError):
            auth.set_password(bad)
    assert auth.stored_hash() is None


def test_header_mode_uses_the_named_header_and_list(tmp_path, monkeypatch):
    _, client = _app(tmp_path, monkeypatch, "header", AUTH_HEADER="Remote-User", OPERATORS="alice")
    assert client.get("/api/summary", headers={"Remote-User": "alice"}).status_code == 200
    assert client.get("/api/summary", headers={"Remote-User": "mallory"}).status_code == 401
    assert client.get("/api/summary", headers={"Tailscale-User-Login": "alice"}).status_code == 401
    assert client.get("/", headers={"Remote-User": "alice"}).status_code == 200
    assert login(client, GOOD).json()["code"] == "PASSWORD_SIGNIN_OFF"


def test_an_unknown_mode_lets_nobody_in(tmp_path, monkeypatch):
    _, client = _app(tmp_path, monkeypatch, "pasword", OPERATORS="alice")  # the typo
    assert client.get("/api/summary", headers={"Tailscale-User-Login": "alice"}).status_code == 401
    assert client.get("/", follow_redirects=False).status_code == 401


def test_plain_ollama_gets_no_token_header():
    from photosweep import classify

    class Fake:
        def post(self, url, json, headers):
            self.headers = headers
            raise classify.httpx.ConnectError("no")

    for token, want in ((None, {}), ("t0k", {"Authorization": "Bearer t0k"})):
        f = Fake()
        with pytest.raises(classify.RouterUnavailable):
            classify._ask(f, "http://ollama:11434", token, {})
        assert f.headers == want


def test_the_model_token_is_optional(tmp_path, monkeypatch):
    from photosweep import config, scanner
    monkeypatch.setattr(config, "ROUTER_TOKEN_FILE", "")
    assert scanner._token() == (True, None)            # plain Ollama: carry on
    monkeypatch.setattr(config, "ROUTER_TOKEN_FILE", str(tmp_path / "absent"))
    assert scanner._token() == (False, None)           # configured but missing: wait
    (tmp_path / "absent").write_text("t0k\n")
    assert scanner._token() == (True, "t0k")
