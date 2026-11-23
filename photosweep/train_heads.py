"""Train the image classifier's heads from a library's own labels.

    python -m photosweep.train_heads              # report, and write the heads
    python -m photosweep.train_heads --dry-run    # report only

Learns only from what a vision model answered (label_src, clutter_src and
blur_src not 'classifier') and from what the owner did: a photo they kept is
a keeper whatever the model said, which is how the 241 photos the owner
rescued from Clutter teach it where the model was wrong. Every number it
prints is out-of-fold: each picture scored by heads that never saw it. Prints
counts only, never a path or a picture. numpy only.
"""
import argparse
import os
import sqlite3

import numpy as np

from . import classifier, config

LABELS = ["screenshot", "meme", "document", "junk", "photo"]
SCALE = 30.0      # logits = (x @ W) * scale + b, on unit-length x
OWNER_WEIGHT = 3  # an owner's keep counts this many times a model's answer
FOLDS = 5


def fit(X, y, k, w, l2=1e-3, steps=400, lr=0.5):
    """Weighted multinomial logistic regression by full-batch gradient
    descent. Classes are balanced, so rare ones (junk, memes) are not ignored."""
    n, d = X.shape
    W = np.zeros((d, k), np.float32)
    b = np.zeros(k, np.float32)
    Y = np.eye(k, dtype=np.float32)[y]
    counts = np.bincount(y, weights=w, minlength=k).clip(1e-6)
    sw = (w * (w.sum() / (k * counts))[y]).astype(np.float32)[:, None]
    sw /= sw.sum()
    for _ in range(steps):
        p = classifier.softmax(X @ W * SCALE + b)
        g = (p - Y) * sw
        W -= lr * (SCALE * X.T @ g + l2 * W)
        b -= lr * g.sum(0)
    return W, b


def out_of_fold(X, y, k, w, seed=0):
    """Logits for every row from heads trained without it."""
    fold = np.random.default_rng(seed).integers(0, FOLDS, len(y))
    Z = np.zeros((len(y), k), np.float32)
    for f in range(FOLDS):
        W, b = fit(X[fold != f], y[fold != f], k, w[fold != f])
        Z[fold == f] = X[fold == f] @ W * SCALE + b
    return Z


def temperature(Z, y, w):
    """The multiplier on the logits that makes the out-of-fold probabilities
    honest (least weighted log-loss), so a p of 0.95 means about 95%."""
    best, best_t = None, 1.0
    for t in np.exp(np.linspace(np.log(0.05), np.log(3.0), 80)):
        p = classifier.softmax(Z * t)[np.arange(len(y)), y]
        loss = -(w * np.log(np.clip(p, 1e-9, 1))).sum()
        if best is None or loss < best:
            best, best_t = loss, float(t)
    return best_t


def load(db_path):
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT status, category, clutter, blurry, kept, label_src, clutter_src, blur_src, embedding FROM files "
        "WHERE embedding IS NOT NULL AND length(embedding) = ? AND status IN ('done','deleted') "
        "AND category IS NOT NULL", (classifier.DIM * 2,)).fetchall()
    X = np.stack([classifier.unpack(r["embedding"]) for r in rows]) if rows else np.zeros((0, classifier.DIM))
    return rows, X.astype(np.float32)


def sets(rows):
    """(name, classes, row index, target, weight) for each head."""
    lab_i, lab_y, cl_i, cl_y, cl_w, bl_i, bl_y = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        by_model = r["label_src"] in (None, "model")
        if by_model:
            lab_i.append(i)
            lab_y.append(LABELS.index(r["category"]) if r["category"] in LABELS else -1)
        if r["category"] != "photo":
            continue
        if r["kept"]:
            cl_i.append(i); cl_y.append(0); cl_w.append(OWNER_WEIGHT)  # the owner said keeper
        elif r["clutter"] in (0, 1) and r["clutter_src"] in (None, "model"):
            cl_i.append(i); cl_y.append(r["clutter"]); cl_w.append(1)
        if r["blurry"] in (0, 1) and r["blur_src"] in (None, "model"):
            bl_i.append(i); bl_y.append(r["blurry"])
    keep = [j for j, y in enumerate(lab_y) if y >= 0]
    return [
        ("label", LABELS, np.array([lab_i[j] for j in keep], int), np.array([lab_y[j] for j in keep], int),
         np.ones(len(keep), np.float32)),
        ("clutter", ["keeper", "clutter"], np.array(cl_i, int), np.array(cl_y, int), np.array(cl_w, np.float32)),
        ("blur", ["sharp", "blurry"], np.array(bl_i, int), np.array(bl_y, int), np.ones(len(bl_i), np.float32)),
    ]


def report(name, classes, rows, idx, y, P):
    pred, conf = P.argmax(1), P.max(1)
    print(f"\n{name}: {len(y)} pictures, {(pred == y).mean() * 100:.1f}% agree out of fold")
    if name == "label":
        for j, c in enumerate(classes):
            m = y == j
            if m.any():
                print(f"  {c:<10} {m.sum():>6}  {(pred[m] == j).mean() * 100:5.1f}% agree")
        ph = y == classes.index("photo")
        for t in (0.9, 0.95, 0.98):
            bad = ph & (pred != y) & (conf >= t)
            print(f"  p>={t}: {(conf >= t).mean() * 100:4.0f}% decided, {(pred == y)[conf >= t].mean() * 100:5.1f}% "
                  f"agree; photos called something else {bad.sum()} ({bad.sum() / max(ph.sum(), 1) * 100:.2f}%)")
        return
    yes = 1
    for t in (0.5, 0.9, 0.95):
        flag = P[:, yes] >= t
        tp = (flag & (y == yes)).sum()
        sure = (P[:, yes] >= t) | (P[:, yes] <= 1 - t)
        print(f"  p>={t}: flags {flag.sum()}, precision {tp / max(flag.sum(), 1) * 100:.0f}%, "
              f"recall {tp / max((y == yes).sum(), 1) * 100:.0f}%; settled without the model {sure.mean() * 100:.0f}%")
    if name == "clutter":
        kept = np.array([rows[i]["kept"] for i in idx]) == 1
        deleted = np.array([rows[i]["status"] == "deleted" for i in idx]) & (y == 1)
        for t in (0.5, 0.9, 0.95):
            print(f"  owner's view at p>={t}: kept photos flagged {(kept & (P[:, 1] >= t)).sum()} of {kept.sum()}, "
                  f"deleted clutter flagged {(deleted & (P[:, 1] >= t)).sum()} of {deleted.sum()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--out", default=classifier.HEADS_FILE)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)
    rows, X = load(args.db)
    print(f"{len(rows)} labelled pictures with an embedding")
    out = {}
    for name, classes, idx, y, w in sets(rows):
        if len(y) < 50 or len(set(y.tolist())) < len(classes):
            print(f"\n{name}: too few examples ({len(y)}), skipped")
            continue
        Xs = X[idx]
        Z = out_of_fold(Xs, y, len(classes), w)
        t = temperature(Z, y, w)
        report(name, classes, rows, idx, y, classifier.softmax(Z * t))
        W, b = fit(Xs, y, len(classes), w)
        out.update({f"{name}_W": W, f"{name}_b": (b * t).astype(np.float32),
                    f"{name}_scale": np.float32(SCALE * t), f"{name}_classes": np.array(classes)})
        print(f"  calibrated scale {SCALE * t:.1f}")
    if "label_W" not in out:
        raise SystemExit("not enough labelled pictures to train the label head")
    if args.dry_run:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
