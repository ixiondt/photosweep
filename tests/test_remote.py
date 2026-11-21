"""A model server outside the local network: refused until the owner says yes,
and spoken to over the OpenAI-compatible API when MODEL_API=openai."""
import json

import pytest

from photosweep import classify, config, remote, scanner


def _public(monkeypatch, addrs=("142.250.1.1",)):
    monkeypatch.setattr(remote, "_resolve", lambda host: set(addrs))
    remote._host_is_local.cache_clear()


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _Recorder:
    """A client that records what would have been sent."""

    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        return self.resp


OLLAMA_OK = {"message": {"content": '{"category":"photo","confidence":90,"reason":"x"}'}}


def _openai_ok(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ------------------------------------------------------------ where is it ---

@pytest.mark.parametrize("host", ["127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.20",
                                  "100.100.100.100", "169.254.1.1", "::1", "fd00::5"])
def test_local_addresses_are_local(host):
    assert not remote.is_remote(f"http://[{host}]:11434" if ":" in host else f"http://{host}:11434")


@pytest.mark.parametrize("host", ["8.8.8.8", "142.250.1.1", "2001:4860:4860::8888"])
def test_public_addresses_are_remote(host):
    assert remote.is_remote(f"http://[{host}]/v1" if ":" in host else f"https://{host}/v1")


def test_a_name_is_remote_if_any_address_is_public(monkeypatch):
    _public(monkeypatch, ("10.0.0.5", "142.250.1.1"))
    assert remote.is_remote("https://api.example.com/v1")


def test_a_name_that_does_not_resolve_is_remote(monkeypatch):
    def fail(host):
        raise OSError("no such host")
    monkeypatch.setattr(remote, "_resolve", fail)
    remote._host_is_local.cache_clear()
    assert remote.is_remote("https://nowhere.invalid/v1")


def test_status_reports_host_and_consent(monkeypatch):
    _public(monkeypatch)
    monkeypatch.setattr(config, "ROUTER_URL", "https://api.example.com/v1")
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", False)
    assert remote.status() == {"remote": True, "host": "api.example.com", "allowed": False}
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", True)
    assert remote.status()["allowed"] is True


def test_a_local_model_needs_no_consent(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_URL", "http://ollama:11434")
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", False)
    assert remote.status() == {"remote": False, "host": "ollama", "allowed": True}


# ------------------------------------------------------------ the gate -----

def test_a_remote_model_is_never_sent_a_photo_without_consent(monkeypatch):
    _public(monkeypatch)
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", False)
    client = _Recorder(_Resp(200, OLLAMA_OK))
    with pytest.raises(classify.RouterUnavailable) as exc:
        classify.classify(client, "https://api.example.com", "k", "m", "QUJD")
    assert client.calls == [], "the request must not leave at all"
    assert "PHOTOS_LEAVE_THIS_SERVER" in str(exc.value) and exc.value.retry_after == 600


def test_consent_lets_it_through(monkeypatch):
    _public(monkeypatch)
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", True)
    client = _Recorder(_Resp(200, OLLAMA_OK))
    assert classify.classify(client, "https://api.example.com", "k", "m", "QUJD") == ("photo", 90, "x")
    assert len(client.calls) == 1


def test_scanner_labels_nothing_while_refused(tmp_path, monkeypatch):
    pics = tmp_path / "pics"
    pics.mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64), (200, 30, 30)).save(pics / "a.jpg")
    monkeypatch.setattr(config, "PICS_ROOT", str(pics))
    monkeypatch.setattr(config, "THUMB_DIR", str(tmp_path / "thumbs"))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(config, "ROUTER_TOKEN_FILE", "")
    monkeypatch.setattr(config, "ROUTER_URL", "https://api.example.com/v1")
    monkeypatch.setattr(config, "PHOTOS_LEAVE_THIS_SERVER", False)
    _public(monkeypatch)
    called = []
    monkeypatch.setattr(classify, "classify", lambda *a, **k: called.append(a) or ("photo", 90, "x"))
    scanner.run(limit=1)
    assert called == []
    from photosweep import db
    conn = db.connect(config.DB_PATH)
    assert conn.execute("SELECT status FROM files").fetchone()[0] == "pending", "walked, not labelled"


# ------------------------------------------------------------ openai API ---

@pytest.fixture
def openai_mode(monkeypatch):
    monkeypatch.setattr(config, "MODEL_API", "openai")
    monkeypatch.setattr(config, "MODEL_JSON", "schema")
    monkeypatch.setattr(config, "MODEL_MIN_INTERVAL_S", 0.0)


def test_openai_request_shape(openai_mode, monkeypatch):
    monkeypatch.delenv("MODEL_THINK", raising=False)
    client = _Recorder(_Resp(200, _openai_ok('{"category":"meme","confidence":80,"reason":"caption"}')))
    assert classify.classify(client, "http://10.0.0.9:1234/v1", "key", "vision-model", "QUJD") == (
        "meme", 80, "caption")
    url, body, headers = client.calls[0]
    assert url == "http://10.0.0.9:1234/v1/chat/completions"
    assert headers == {"Authorization": "Bearer key"}
    assert body["model"] == "vision-model" and body["temperature"] == 0 and body["max_tokens"] == 4096
    text, image = body["messages"][0]["content"]
    assert text["type"] == "text" and "Classify this ONE image" in text["text"]
    assert image == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["required"] == ["category", "confidence", "reason"]
    for ollama_only in ("options", "format", "keep_alive", "stream", "think", "top_k"):
        assert ollama_only not in body


def test_openai_retry_samples_without_ollama_only_fields(openai_mode):
    body = classify.openai_body(classify.build_request("m", "QUJD", sample=True))
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95 and "top_k" not in body


@pytest.mark.parametrize("mode,expected", [("object", {"type": "json_object"}), ("none", None)])
def test_openai_json_mode_can_be_relaxed(openai_mode, monkeypatch, mode, expected):
    monkeypatch.setattr(config, "MODEL_JSON", mode)
    body = classify.openai_body(classify.build_blur_request("m", "QUJD"))
    assert body.get("response_format") == expected


@pytest.mark.parametrize("content", [
    '```json\n{"blurry": true, "confidence": 71}\n```',
    '```\n{"blurry": true, "confidence": 71}\n```',
    '{"blurry": true, "confidence": 71}',
])
def test_openai_fenced_json_is_read(openai_mode, content):
    client = _Recorder(_Resp(200, _openai_ok(content)))
    assert classify.judge_blur(client, "http://10.0.0.9/v1", None, "m", "QUJD") == (True, 71)
    assert client.calls[0][2] == {}, "no token, no Authorization header"


@pytest.mark.parametrize("body", [{"choices": []}, {"choices": [{"message": {}}]}, {"error": "x"}])
def test_openai_body_without_content_is_a_bad_verdict(openai_mode, body):
    with pytest.raises(classify.BadVerdict):
        classify.classify(_Recorder(_Resp(200, body)), "http://10.0.0.9/v1", None, "m", "QUJD")


def test_cloud_429_reads_the_wait_from_the_body(openai_mode):
    body = {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                 "retryDelay": "31s"}]}}
    with pytest.raises(classify.RouterUnavailable) as exc:
        classify.classify(_Recorder(_Resp(429, body)), "http://10.0.0.9/v1", "k", "m", "QUJD")
    assert exc.value.retry_after == 31.0


def test_requests_are_paced(openai_mode, monkeypatch):
    monkeypatch.setattr(config, "MODEL_MIN_INTERVAL_S", 4.0)
    clock = [1000.0]
    slept = []
    monkeypatch.setattr(classify.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(classify.time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(classify, "_last_request", 0.0)
    client = _Recorder(_Resp(200, _openai_ok('{"category":"photo","confidence":90,"reason":"x"}')))
    classify.classify(client, "http://10.0.0.9/v1", None, "m", "QUJD")
    clock[0] += 1.5
    classify.classify(client, "http://10.0.0.9/v1", None, "m", "QUJD")
    assert slept == [2.5]



@pytest.mark.parametrize("value,consent", [
    (None, False), ("", False), ("no", False), ("true", False), ("1", False), ("y", False), ("yess", False),
    ("yes", True), ("YES", True), (" yes ", True),
])
def test_only_an_explicit_yes_is_consent(value, consent):
    assert config._is_yes(value) is consent
