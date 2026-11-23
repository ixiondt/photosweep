"""Stage 2: a small image classifier that runs on the CPU.

SigLIP B/16 (int8, through onnxruntime) turns a thumbnail into a 768-number
embedding; three small trained heads (heads/siglip-b16.npz) read the label,
the clutter question and the blur question off it: a linear layer for the
label, a one-hidden-layer network for each yes/no question, every probability
calibrated on held-out pictures. Measured figures are in the README. About
0.1-0.3 s a picture on one CPU core, all three questions together.

The scanner stores every embedding, so the heads can be retrained from a
library's own labels and from what its owner kept and deleted
(python -m photosweep.train_heads). Nothing here touches the network.
"""
import logging
import os
import threading

import numpy as np
from PIL import Image

from . import config

log = logging.getLogger("photosweep.classifier")

HEADS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heads", "siglip-b16.npz")
DIM = 768
SIZE = 224
QUESTIONS = ("label", "clutter", "blur")

_lock = threading.Lock()
_loaded = None   # the Classifier, once loaded
_failed = False  # loading failed once: logged, not retried until restart


def pixels(img):
    """SigLIP's preprocessing (preprocessor_config.json): RGB, squashed to
    224x224 bicubic, scaled to [-1, 1], channels first, batch of one."""
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) * (2.0 / 255.0) - 1.0
    return a.transpose(2, 0, 1)[None]


def pack(vec):
    """An embedding as stored: float16, 1.5 KB."""
    return np.asarray(vec, dtype=np.float16).tobytes()


def unpack(blob):
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def heads_path():
    """This library's own heads when they exist (CLASSIFIER_HEADS, in the data
    volume, so they survive an image update), else the bundled ones."""
    own = config.CLASSIFIER_HEADS
    return own if own and os.path.isfile(own) else HEADS_FILE


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def hidden(emb, W1, b1, w2, b2, gain):
    """The yes/no network's raw score: one ReLU layer, then a single output."""
    return np.maximum(emb @ W1 * gain + b1, 0.0) @ w2 + b2


class Heads:
    """The trained heads. Each question is one of two shapes:
    linear:  {q}_W (DIM x k), {q}_b, {q}_scale, softmax over k classes;
    network: {q}_W1, {q}_b1, {q}_w2, {q}_b2, {q}_gain, then the calibration
             p(yes) = sigmoid({q}_a * score + {q}_c); exactly two classes.
    Both carry {q}_classes."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.q = {}
            for name in QUESTIONS:
                if f"{name}_W1" in z:
                    W1, b1, w2 = (z[f"{name}_{k}"].astype(np.float32) for k in ("W1", "b1", "w2"))
                    self.q[name] = ("net", (W1, b1, w2, float(z[f"{name}_b2"]), float(z[f"{name}_gain"])),
                                    (float(z[f"{name}_a"]), float(z[f"{name}_c"])),
                                    [str(c) for c in z[f"{name}_classes"]])
                elif f"{name}_W" in z:
                    self.q[name] = ("linear", (z[f"{name}_W"].astype(np.float32), z[f"{name}_b"].astype(np.float32)),
                                    float(z[f"{name}_scale"]), [str(c) for c in z[f"{name}_classes"]])
        if "label" not in self.q:
            raise ValueError("heads file has no label head")
        for name, (kind, _, _, classes) in self.q.items():
            if kind == "net" and len(classes) != 2:
                raise ValueError(f"{name}: a network head answers yes or no")

    def probs(self, name, emb):
        kind, weights, cal, classes = self.q[name]
        if kind == "linear":
            W, b = weights
            return classes, softmax((emb @ W) * cal + b)
        a, c = cal
        p = float(sigmoid(a * hidden(emb, *weights) + c))
        return classes, np.array([1.0 - p, p], np.float32)

    def batch(self, X):
        """verdicts() for many embeddings at once (rows of X): a dict of
        arrays, label as class names, the probabilities NaN where the heads
        file carries no such head. Seconds for a whole library."""
        X = np.asarray(X, np.float32)
        kind, weights, cal, classes = self.q["label"]
        W, b = weights
        P = softmax((X @ W) * cal + b)
        i = P.argmax(1)
        out = {"label": np.array(classes)[i], "label_p": P[np.arange(len(X)), i]}
        for name, yes in (("clutter", "clutter"), ("blur", "blurry")):
            p = np.full(len(X), np.nan, np.float32)
            if name in self.q:
                kind, weights, cal, classes = self.q[name]
                if kind == "net":
                    p = sigmoid(cal[0] * hidden(X, *weights) + cal[1])
                else:
                    p = softmax((X @ weights[0]) * cal + weights[1])[:, classes.index(yes)]
            out[f"{name}_p"] = p
        return out

    def verdicts(self, emb):
        """{label, label_p, clutter_p, blur_p}: the likeliest label and its
        probability, and the probability of clutter and of blur (None when
        the heads file carries no such head)."""
        classes, p = self.probs("label", emb)
        i = int(p.argmax())
        out = {"label": classes[i], "label_p": float(p[i]), "clutter_p": None, "blur_p": None}
        for name, yes in (("clutter", "clutter"), ("blur", "blurry")):
            if name in self.q:
                classes, p = self.probs(name, emb)
                out[f"{name}_p"] = float(p[classes.index(yes)])
        return out


class Classifier:
    def __init__(self, model_path, heads=None, session=None):
        if session is None:
            import onnxruntime as ort  # imported here: the web process never needs it
            so = ort.SessionOptions()
            # one thread a run: the scanner already decodes FINGERPRINT_WORKERS pictures at once
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.session = session
        outs = [o.name for o in session.get_outputs()]
        self._out = outs.index("pooler_output") if "pooler_output" in outs else 0
        heads = heads or heads_path()
        self.heads_file = heads if os.path.isfile(heads) else None
        self.heads = Heads(heads) if self.heads_file else None

    def embed(self, img):
        """A unit-length embedding of one picture. Thread-safe: onnxruntime
        runs one session from several threads."""
        v = self.session.run(None, {"pixel_values": pixels(img)})[self._out][0].astype(np.float32)
        n = float(np.linalg.norm(v))
        if not n > 0:
            raise ValueError("empty embedding")
        return v / n

    def verdicts(self, emb):
        return self.heads.verdicts(emb) if self.heads else None


def enabled():
    """The classifier is configured and its model file is there. Cheap: the
    web page asks this on every summary and never loads the model."""
    return config.CLASSIFIER != "none" and os.path.isfile(config.CLASSIFIER_MODEL)


def sorting():
    """The classifier can sort (model and heads both present)."""
    return enabled() and os.path.isfile(heads_path())


def load():
    """The shared Classifier, or None when it is switched off, its model file
    is missing, or it failed to load (logged once)."""
    global _loaded, _failed
    if _loaded is not None or _failed or not enabled():
        return _loaded
    with _lock:
        if _loaded is None and not _failed:
            try:
                _loaded = Classifier(config.CLASSIFIER_MODEL)
                log.info("image classifier loaded model=%s heads=%s", config.CLASSIFIER_MODEL,
                         _loaded.heads_file or "none (embeddings only)")
            except Exception as exc:
                _failed = True
                log.error("image classifier failed to load err=%s", f"{type(exc).__name__}: {exc}")
    return _loaded


def reset():
    """Forget the loaded classifier (tests)."""
    global _loaded, _failed
    _loaded, _failed = None, False
