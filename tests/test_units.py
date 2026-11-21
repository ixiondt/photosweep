import json
import sqlite3

import pytest
from PIL import Image

from photosweep import classify, db, imaging, nextcloud


# ------------------------------------------------------------ classify ----

def test_parse_verdict_good():
    assert classify.parse_verdict('{"category":"meme","confidence":91,"reason":" caption over cat "}') == (
        "meme", 91, "caption over cat")


@pytest.mark.parametrize("raw", [
    "", "not json", "[]", "null", '"meme"',
    '{"category":"selfie","confidence":50,"reason":"x"}',
    '{"category":"meme","confidence":"high","reason":"x"}',
    '{"category":"meme","confidence":true,"reason":"x"}',
    '{"category":"meme","confidence":101,"reason":"x"}',
    '{"category":"meme","confidence":-1,"reason":"x"}',
    '{"category":"meme","reason":"x"}',
])
def test_parse_verdict_rejects(raw):
    with pytest.raises(classify.BadVerdict):
        classify.parse_verdict(raw)


def test_parse_verdict_rounds_float_and_tolerates_missing_reason():
    assert classify.parse_verdict('{"category":"photo","confidence":72.6}') == ("photo", 73, "")


def test_build_request_names_model_and_keeps_one_image(monkeypatch):
    monkeypatch.delenv("MODEL_THINK", raising=False)
    body = classify.build_request("qwen3-vl:8b", "QUJD")
    assert body["model"] == "qwen3-vl:8b" and body["stream"] is False
    assert body["messages"][0]["images"] == ["QUJD"]
    assert "think" not in body
    monkeypatch.setenv("MODEL_THINK", "false")
    assert classify.build_request("m", "x")["think"] is False


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Client:
    def __init__(self, resp):
        self.resp = resp

    def post(self, *_a, **_k):
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def test_classify_router_busy_is_retryable_and_reads_retry_after():
    with pytest.raises(classify.RouterUnavailable) as exc:
        classify.classify(_Client(_Resp(429, {}, {"retry-after": "7"})), "http://r", "t", "m", "x")
    assert exc.value.retry_after == 7.0


def test_classify_network_error_is_retryable():
    import httpx
    with pytest.raises(classify.RouterUnavailable):
        classify.classify(_Client(httpx.ConnectError("boom")), "http://r", "t", "m", "x")


def test_classify_bad_body_is_bad_verdict():
    with pytest.raises(classify.BadVerdict):
        classify.classify(_Client(_Resp(200, {"nope": 1})), "http://r", "t", "m", "x")
    with pytest.raises(classify.BadVerdict):
        classify.classify(_Client(_Resp(200, ValueError("not json"))), "http://r", "t", "m", "x")


def test_classify_ok():
    body = {"message": {"content": json.dumps({"category": "junk", "confidence": 88, "reason": "black frame"})}}
    assert classify.classify(_Client(_Resp(200, body)), "http://r", "t", "m", "x") == ("junk", 88, "black frame")


def test_length_ceiling_leaves_room_for_a_long_think():
    # 1024 cut a photographed sketch off mid-thought, empty content, three times
    assert classify.build_request("m", "x")["options"]["num_predict"] >= 4096


def test_first_try_is_greedy_and_a_retry_samples():
    assert classify.build_request("m", "x")["options"]["temperature"] == 0
    sampled = classify.build_request("m", "x", sample=True)["options"]
    assert sampled["temperature"] > 0 and sampled["top_k"] == 20 and sampled["num_predict"] >= 4096
    assert classify.build_blur_request("m", "x", sample=True)["options"]["temperature"] > 0


def test_blur_request_asks_the_blur_question_with_its_own_schema():
    body = classify.build_blur_request("m", "QUJD")
    assert body["messages"][0]["content"] == classify.BLUR_PROMPT
    assert body["messages"][0]["images"] == ["QUJD"]
    assert body["format"] == classify.BLUR_SCHEMA
    assert classify.build_request("m", "x")["format"] == classify.SCHEMA  # the label call is untouched


def test_parse_blur():
    assert classify.parse_blur('{"blurry": true, "confidence": 84.6}') == (True, 85)
    assert classify.parse_blur('{"blurry": false, "confidence": 95}') == (False, 95)


@pytest.mark.parametrize("raw", ['', 'nope', '[]', '{"blurry": "yes", "confidence": 90}',
                                 '{"blurry": 1, "confidence": 90}', '{"blurry": true}',
                                 '{"blurry": true, "confidence": true}', '{"blurry": true, "confidence": 101}',
                                 '{"blurry": true, "confidence": NaN}'])
def test_parse_blur_rejects(raw):
    with pytest.raises(classify.BadVerdict):
        classify.parse_blur(raw)


def test_judge_blur_shares_the_router_failure_contract():
    with pytest.raises(classify.RouterUnavailable):
        classify.judge_blur(_Client(_Resp(503, {})), "http://r", "t", "m", "x")
    body = {"message": {"content": '{"blurry": true, "confidence": 70}'}}
    assert classify.judge_blur(_Client(_Resp(200, body)), "http://r", "t", "m", "x") == (True, 70)


# ----------------------------------------------------------- nextcloud ----

def test_dav_url_encodes_each_segment():
    url = nextcloud.dav_url("https://nc.example/", "jas", "Pics", "2020/07/a b#1?.jpg")
    assert url == "https://nc.example/remote.php/dav/files/jas/Pics/2020/07/a%20b%231%3F.jpg"


def test_dav_url_unicode_and_percent():
    assert nextcloud.dav_url("https://n", "u", "Pics", "foto del niño 100%.jpg").endswith(
        "/Pics/foto%20del%20ni%C3%B1o%20100%25.jpg")


@pytest.mark.parametrize("rel", ["../etc/passwd", "a/../../b.jpg", "a//b.jpg", "./a.jpg", ""])
def test_dav_url_refuses_traversal(rel):
    with pytest.raises(ValueError):
        nextcloud.dav_url("https://n", "u", "Pics", rel)


def test_client_requires_password():
    with pytest.raises(nextcloud.NotConfigured):
        nextcloud.client("u", None)


# ------------------------------------------------------------- imaging ----

@pytest.mark.parametrize("name,hint", [
    ("Screenshot_20231225-123456.png", "screenshot"),
    ("Screen Shot 2019-01-01 at 10.00.00.png", "screenshot"),
    ("FB_IMG_1600000000000.jpg", "facebook"),
    ("IMG-20231225-WA0001.jpg", "messaging"),
    ("signal-2023-12-25-123456.jpg", "messaging"),
    ("ReactNativeBlobUtilTmp_abc.jpg", "app-download"),
    ("image0.jpeg", "ios-saved"),
    ("IMG_1234.HEIC", None),
    ("PXL_20240101_000000.jpg", None),
])
def test_name_hint(name, hint):
    assert imaging.name_hint("2020/07/" + name) == hint


def test_camera_of():
    assert imaging.camera_of(None) is None
    assert imaging.camera_of({}) is None
    assert imaging.camera_of({0x010F: "Apple", 0x0110: "iPhone 12"}) == "Apple iPhone 12"
    assert imaging.camera_of({0x010F: "Google", 0x0110: "Google Pixel 7"}) == "Google Pixel 7"
    assert imaging.camera_of({0x010F: "samsung\x00"}) == "samsung"


def test_open_and_thumb_png_with_alpha(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (1200, 2400), (10, 20, 30, 0)).save(path)
    img, camera, w, h = imaging.open_image(str(path), 512)
    assert (w, h) == (1200, 2400) and camera is None and img.mode == "RGB"
    thumb = Image.open(__import__("io").BytesIO(imaging.jpeg_bytes(img, 512)))
    assert max(thumb.size) == 512


def _checker(size=256, cell=8):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    px = img.load()
    for y in range(size):
        for x in range(size):
            if (x // cell + y // cell) % 2:
                px[x, y] = (0, 0, 0)
    return img


def test_sharpness_ranks_blur_below_sharp():
    from PIL import ImageFilter
    sharp = _checker()
    blurred = sharp.filter(ImageFilter.GaussianBlur(6))
    assert imaging.sharpness(blurred) < 18 < imaging.sharpness(sharp)


def test_sharpness_ignores_overall_brightness():
    """A dim but in-focus frame must not read as blurred: the score is divided
    by the frame's own contrast."""
    sharp = _checker()
    dim = sharp.point(lambda v: v // 6)
    assert imaging.sharpness(dim) > 18
    # without the contrast division its score falls with the square of the
    # dimming and it would sit at the top of Blurry


def test_sharpness_sharp_subject_on_soft_background_is_sharp():
    from PIL import ImageFilter
    img = _checker().filter(ImageFilter.GaussianBlur(6))
    img.paste(_checker(64), (96, 96))
    assert imaging.sharpness(img) > 18


def test_sharpness_flat_and_tiny_images_do_not_crash():
    assert imaging.sharpness(Image.new("RGB", (300, 200), (90, 90, 90))) == 0.0
    assert imaging.sharpness(Image.new("RGB", (4, 4))) is None


def test_cut_off_jpeg_decodes_as_far_as_it_goes(tmp_path):
    import io
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (200, 40, 40)).save(buf, format="JPEG", quality=90)
    path = tmp_path / "cut.jpg"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])  # an interrupted copy
    img, _cam, w, h = imaging.open_image(str(path), 512)
    assert (w, h) == (400, 300) and imaging.jpeg_bytes(img, 512)[:2] == bytes([0xFF, 0xD8])


def test_open_corrupt_raises(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(Exception):
        imaging.open_image(str(path), 512)


# ------------------------------------------------------------------ db ----

def test_audit_chain_verifies_and_is_append_only(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    with db.immediate(conn):
        db.audit(conn, "a@x", "delete", {"id": 1})
    with db.immediate(conn):
        db.audit(conn, "a@x", "keep", {"ids": [2, 3]})
    assert db.verify_audit(conn) == (True, None)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit SET who='b' WHERE id=1")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM audit")


def test_audit_chain_detects_tampering(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    with db.immediate(conn):
        db.audit(conn, "a@x", "delete", {"id": 1})
        db.audit(conn, "a@x", "delete", {"id": 2})
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("UPDATE audit SET detail='{\"id\":9}' WHERE id=1")
    assert db.verify_audit(conn) == (False, 1)


def test_immediate_rolls_back(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    with pytest.raises(RuntimeError):
        with db.immediate(conn):
            db.set_setting(conn, "paused", "1")
            raise RuntimeError("boom")
    assert db.get_setting(conn, "paused") is None


def test_clutter_request_and_parse():
    body = classify.build_clutter_request("m", "QUJD", sample=True)
    assert body["messages"][0]["content"] == classify.CLUTTER_PROMPT and body["format"] == classify.CLUTTER_SCHEMA
    assert body["messages"][0]["images"] == ["QUJD"] and body["options"]["temperature"] > 0
    assert classify.parse_clutter('{"clutter": true, "confidence": 84.6, "reason": " a floor "}') == (True, 85, "a floor")
    assert classify.parse_clutter('{"clutter": false, "confidence": 90}') == (False, 90, "")


@pytest.mark.parametrize("raw", ['', '[]', '{"clutter": "yes", "confidence": 90}', '{"clutter": 1, "confidence": 9}',
                                 '{"clutter": true}', '{"clutter": true, "confidence": 101}',
                                 '{"clutter": true, "confidence": NaN}'])
def test_parse_clutter_rejects(raw):
    with pytest.raises(classify.BadVerdict):
        classify.parse_clutter(raw)


def test_judge_clutter_shares_the_router_failure_contract():
    with pytest.raises(classify.RouterUnavailable):
        classify.judge_clutter(_Client(_Resp(429, {})), "http://r", "t", "m", "x")
    body = {"message": {"content": '{"clutter": true, "confidence": 70, "reason": "wall"}'}}
    assert classify.judge_clutter(_Client(_Resp(200, body)), "http://r", "t", "m", "x") == (True, 70, "wall")
