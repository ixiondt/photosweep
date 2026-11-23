"""Ask the vision model, over the Ollama chat API, what one image is.

The server may be Ollama itself or a proxy in front of several. Either way
the scanner keeps exactly one request in flight and never more: a GPU serves
one image at a time, and a second request only queues behind the first.

The model is named outright. A thinking model is more accurate and far too
slow for a library of tens of thousands.
"""
import json
import os
import re
import time

import httpx

from . import config, remote

CATEGORIES = ("photo", "screenshot", "meme", "document", "junk")

PROMPT = """You are sorting a personal photo library so the owner can delete clutter.
Classify this ONE image into exactly one category:

- photo: a real camera photo of people, pets, places, events, food or things. Imperfect but meaningful photos are still "photo".
- screenshot: a capture of a phone or computer screen: chats, text messages, apps, web pages, maps, settings, games, social media posts with their app interface visible.
- meme: a graphic made to be shared: caption or joke text over an image, reaction images, quotes, greetings, viral or forwarded images, cartoons, stickers.
- document: a photo or scan of paper or a label for reference: receipts, forms, letters, ID cards, whiteboards, price tags, a monitor photographed to keep information.
- junk: an accidental or failed shot: black or blank frame, extremely blurry, pocket or finger shot, nothing recognizable.

If unsure between photo and anything else, prefer photo. Answer with JSON only:
{"category": one of the five, "confidence": integer 0-100 for how sure you are, "reason": at most 12 words}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["category", "confidence", "reason"],
}


class RouterUnavailable(Exception):
    """Retryable: the router or every backend is busy, down or slow. The file
    is not at fault, so this never counts against its attempts."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class BadVerdict(ValueError):
    """The model answered, but not with something usable."""


# Qwen's published thinking-mode sampling. Greedy decoding (temperature 0)
# is what Qwen warns causes endless repetition in thinking mode, and six photos
# showed it: still thinking, and repeating, at 12,288 tokens, the same way on
# every try. With these settings all six answered in 5-50 s (2026-11-18). The
# first try stays at 0 so a verdict is stable; retries sample.
SAMPLED = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def build_request(model, image_b64, sample=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
        "stream": False,
        "format": SCHEMA,
        # qwen3-vl:8b on this fleet THINKS whatever `think` says (~600 chars,
        # ~150 tokens, measured 2026-11-17), and thinking counts against
        # num_predict. At 160 a long think cut the JSON mid-key, the same way
        # on every retry at temperature 0. The verdict itself is ~30 tokens.
        # 1024 was still short for ambiguous images: a photographed pencil
        # sketch deliberated 1,505 tokens over photo-vs-document and came back
        # with empty content (done_reason=length) on all three tries; 34 files
        # were filed as errors that way (2026-11-18). A clear image still
        # stops at ~150, so the ceiling costs only the hard ones their time.
        "options": {"temperature": 0, "num_predict": 4096},
        "keep_alive": "15m",
    }
    if sample:
        body["options"].update(SAMPLED)
    think = os.environ.get("MODEL_THINK", "").strip().lower()
    if think in ("true", "false"):
        body["think"] = think == "true"
    return body


def parse_verdict(content):
    """(category, confidence, reason) from the model's message content. Parsing
    is not enough; the shape is checked too, because a 200 with the wrong body
    is not a verdict."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise BadVerdict(f"not JSON: {str(content)[:120]!r}") from exc
    if not isinstance(data, dict):
        raise BadVerdict("not an object")
    category = data.get("category")
    if category not in CATEGORIES:
        raise BadVerdict(f"unknown category {category!r}")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise BadVerdict(f"confidence is not a number: {confidence!r}")
    if not (confidence == confidence) or confidence < 0 or confidence > 100:  # NaN fails ==
        raise BadVerdict(f"confidence out of range: {confidence!r}")
    reason = data.get("reason")
    reason = reason.strip()[:200] if isinstance(reason, str) else ""
    return category, int(round(confidence)), reason


# A second, narrower question, asked only of photos. The pixel score could not
# tell a smooth subject from a smeared one: in a trial of 90 (2026-11-18) it
# had filed sunsets, plain selfies and a mugshot as blurry, and scored night
# shots and motion-smeared kids as sharp, since noise and point lights read as
# detail. This prompt cleared 13 of 30 pixel-blurry photos and caught the night
# and motion ones, so it decides Blurry wherever it has answered.
BLUR_PROMPT = """Judge ONLY whether this photo is blurry. Ignore what it shows and whether it is a good photo.

- blurry: camera shake, motion blur or missed focus has smeared the main subject, so its edges and details are soft or doubled.
- sharp: the main subject is in focus, even if the picture is dark, grainy, low-resolution, compressed, plain, or a smooth close-up of skin, a wall or the sky.

Answer with JSON only: {"blurry": true or false, "confidence": integer 0-100}"""

BLUR_SCHEMA = {
    "type": "object",
    "properties": {
        "blurry": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["blurry", "confidence"],
}


def build_blur_request(model, image_b64, sample=False):
    body = build_request(model, image_b64, sample)
    body["messages"][0]["content"] = BLUR_PROMPT
    body["format"] = BLUR_SCHEMA
    return body


def parse_blur(content):
    """(blurry, confidence) from the model's message content, shape-checked."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise BadVerdict(f"not JSON: {str(content)[:120]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("blurry"), bool):
        raise BadVerdict("no boolean 'blurry'")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise BadVerdict(f"confidence is not a number: {confidence!r}")
    if not (confidence == confidence) or confidence < 0 or confidence > 100:
        raise BadVerdict(f"confidence out of range: {confidence!r}")
    return data["blurry"], int(round(confidence))


# A third question, also asked only of photos. The label prompt sets the bar
# for "junk" at "nothing recognizable" and prefers "photo" when unsure, so a
# floor, a parking lot at night or a scrap of paper is a photo by its rules:
# after the first full pass 33,237 were photos and 51 junk (2026-11-19). That
# bias is right for the five-way label, where a wrong call files a real photo
# as a screenshot. This question asks the narrower thing the owner wants
# answered, and its answer only moves a photo into a tab to be looked at.
#
# Two themes moved from clutter to keeper on 2026-11-23, from what the owner
# actually did on the Clutter tab rather than what was guessed up front: of
# 2,133 reviewed, 11.5% kept overall, but about half of the bathroom, wall,
# damage and renovation shots and a quarter of children's drawings, crafts and
# builds. Objects, products, food and screens were kept 3-4% of the time and
# stay clutter.
# The first wording took every room as a keeper; probed against 60 deleted
# photos it spared 17, ten of them plain rooms (kept only ~27% of the time), so
# the house rule is now its records: damage, repairs, renovation, the outside.
CLUTTER_PROMPT = """This image is in a personal photo library. Decide ONLY whether it is clutter the owner would not miss, or a keeper.

- clutter: nothing worth remembering. A patch of floor, carpet, ground or sky photographed by accident; an accidental or pocket shot; a frame too dark or washed out to show much; a random object, product, part, cable, light or appliance photographed for a moment's use; food or fridge contents on their own; any screen or display photographed, whatever it shows (TV, monitor, laptop, phone, projector, car dashboard or stereo, baby monitor); an engraved, printed or decorated object; a sticky note, handwritten note, to-do list, scrap of paper, card, label, barcode, receipt or packaging; a bought toy, puzzle, game or printed picture on its own; an animal that is not a pet, seen in passing; a plain, empty or everyday room, kitchen, garage or corner with nothing being done to it; a car park, road, shop or sports field with nothing happening. A hand, finger or foot at the edge of such a shot does not make it a keeper: a person must be what the photo is of.
- keeper: people, pets, family or friends; a place visited, an event or occasion, a shared meal; nature and scenery (a hike, a view, a beach, a sunset, flowers); a record of the owner's house: damage, a leak, a repair, construction or renovation in progress or just finished (a wall, ceiling, floor, bathroom or any room), the house from outside, a decorated room, a garden or yard; something a child made or built (a drawing, painting, craft, schoolwork, a Lego or model build). A dark, blurred or imperfect photo of people or pets is still a keeper.

The owner wants clutter gone, but keeps records of work on the house and of the children's work. An ordinary room is not a record. If no person or pet is the subject and it is not clearly an occasion, a place visited, nature, work or damage on the house, or something a child made, answer clutter.

Answer with JSON only: {"clutter": true or false, "confidence": integer 0-100, "reason": at most 10 words}"""

CLUTTER_SCHEMA = {
    "type": "object",
    "properties": {
        "clutter": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["clutter", "confidence", "reason"],
}


def build_clutter_request(model, image_b64, sample=False):
    body = build_request(model, image_b64, sample)
    body["messages"][0]["content"] = CLUTTER_PROMPT
    body["format"] = CLUTTER_SCHEMA
    return body


def parse_clutter(content):
    """(clutter, confidence, reason) from the model's message content, shape-checked."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise BadVerdict(f"not JSON: {str(content)[:120]!r}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("clutter"), bool):
        raise BadVerdict("no boolean 'clutter'")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise BadVerdict(f"confidence is not a number: {confidence!r}")
    if not (confidence == confidence) or confidence < 0 or confidence > 100:
        raise BadVerdict(f"confidence out of range: {confidence!r}")
    reason = data.get("reason")
    reason = reason.strip()[:200] if isinstance(reason, str) else ""
    return data["clutter"], int(round(confidence)), reason


def judge_clutter(client, router_url, token, model, image_b64, sample=False):
    """(clutter, confidence, reason) for one photo. Same failure contract as classify()."""
    return parse_clutter(_ask(client, router_url, token, build_clutter_request(model, image_b64, sample)))


def _retry_after(resp):
    value = resp.headers.get("retry-after")
    try:
        if value:
            return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP-date; the caller's own backoff applies
    # Some cloud services put the wait in the error body instead
    # ({"error": {"details": [{"retryDelay": "31s"}]}}).
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text[:4096])
    return float(match.group(1)) if match else None


def openai_body(body):
    """The same question as an OpenAI-compatible chat completion: the prompt
    and the image as content parts, the JSON shape as `response_format` (or
    not, per MODEL_JSON), and only the sampling fields that API defines."""
    msg, opts = body["messages"][0], body["options"]
    out = {
        "model": body["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": msg["content"]},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + msg["images"][0]}},
        ]}],
        "temperature": opts["temperature"],
        "max_tokens": opts["num_predict"],
    }
    if "top_p" in opts:
        out["top_p"] = opts["top_p"]
    if config.MODEL_JSON == "schema":
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": body["format"]}}
    elif config.MODEL_JSON == "object":
        out["response_format"] = {"type": "json_object"}
    return out


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _unfence(content):
    """A cloud model asked for JSON by the prompt alone often wraps it in a
    Markdown code fence; the parsers then see the JSON inside."""
    if not isinstance(content, str):
        return content
    match = _FENCE.match(content)
    return match.group(1) if match else content


_last_request = 0.0


def _pace():
    """Hold to MODEL_MIN_INTERVAL_S between requests. The scanner keeps one
    request in flight, so a single timestamp is the whole state."""
    global _last_request
    gap = config.MODEL_MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if gap > 0:
        time.sleep(gap)
    _last_request = time.monotonic()


def classify(client, router_url, token, model, image_b64, sample=False):
    """One verdict. Raises RouterUnavailable (retry later) or BadVerdict."""
    return parse_verdict(_ask(client, router_url, token, build_request(model, image_b64, sample)))


def judge_blur(client, router_url, token, model, image_b64, sample=False):
    """(blurry, confidence) for one photo. Same failure contract as classify()."""
    return parse_blur(_ask(client, router_url, token, build_blur_request(model, image_b64, sample)))


def _ask(client, router_url, token, body):
    """The message content of one chat call, or RouterUnavailable / BadVerdict."""
    # The scanner checks this too; checking here as well means no code path
    # can send a photo outside the network without the owner's yes.
    if not remote.allowed(router_url):
        raise RouterUnavailable(f"model server {remote.host(router_url)} is outside this network and "
                                "PHOTOS_LEAVE_THIS_SERVER is not yes; no photo was sent", 600)
    openai = config.MODEL_API == "openai"
    url = f"{router_url}/chat/completions" if openai else f"{router_url}/api/chat"
    _pace()
    try:
        resp = client.post(url, json=openai_body(body) if openai else body,
                           headers={"Authorization": f"Bearer {token}"} if token else {})
    except httpx.HTTPError as exc:
        raise RouterUnavailable(f"router unreachable: {type(exc).__name__}") from exc
    if resp.status_code in (408, 429) or resp.status_code >= 500:
        raise RouterUnavailable(f"router {resp.status_code}", _retry_after(resp))
    if resp.status_code != 200:
        # 401/403/404: a configuration fault. Retrying cannot fix it, but neither
        # is it this file's fault, so it still must not burn the file's attempts.
        raise RouterUnavailable(f"router refused: {resp.status_code} {resp.text[:160]}", 300)
    try:
        if openai:
            return _unfence(resp.json()["choices"][0]["message"]["content"])
        return resp.json()["message"]["content"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise BadVerdict("model body has no message content") from exc
