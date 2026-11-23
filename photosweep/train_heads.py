"""Train the image classifier's heads from a library's own labels.

    python -m photosweep.train_heads              # report, and write the heads
    python -m photosweep.train_heads --dry-run    # report only

Writes CLASSIFIER_HEADS (default <data>/heads.npz), which the classifier then
prefers over the bundled heads; restart the scanner to load them.

Learns only from what a vision model answered (label_src, clutter_src and
blur_src not 'classifier') and from what the owner did: a photo they kept is
a keeper whatever the model said, which is how the 241 photos the owner
rescued from Clutter teach it where the model was wrong. Every number it
prints is out-of-fold: each picture scored by heads that never saw it, with
every duplicate group (dup_pairs) kept inside one fold, so a copy in training
cannot vouch for its twin in testing. Prints counts only, never a path or a
picture. numpy only.

The label head is linear (softmax, temperature-calibrated). The clutter and
blur heads are one-hidden-layer networks with Platt calibration: on one
37,000-picture library the network lifted clutter's AUC from 0.968 to 0.988
over a linear head, with duplicate-grouped folds.
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
HIDDEN = 256      # the yes/no networks' hidden units
GAIN = 8.0        # input scale for the networks, on unit-length x
EPOCHS = 25


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


def fit_net(X, y, w, seed=1, lr=1e-3, l2=1e-4, epochs=EPOCHS):
    """A one-hidden-layer yes/no network by Adam on minibatches, weighted
    log-loss. Not class-balanced: Platt calibration follows."""
    n, d = X.shape
    r = np.random.default_rng(seed)
    P = {"W1": r.normal(0, (2 / d) ** .5, (d, HIDDEN)).astype(np.float32),
         "b1": np.zeros(HIDDEN, np.float32),
         "w2": r.normal(0, (1 / HIDDEN) ** .5, HIDDEN).astype(np.float32),
         "b2": np.zeros((), np.float32)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    ww = (w / w.mean()).astype(np.float32)
    yy = y.astype(np.float32)
    t = 0
    for _ in range(epochs):
        for idx in np.array_split(r.permutation(n), max(1, n // 256)):
            x = X[idx]
            H = np.maximum(x @ P["W1"] * GAIN + P["b1"], 0)
            g = (classifier.sigmoid(H @ P["w2"] + P["b2"]) - yy[idx]) * ww[idx] / len(idx)
            dH = np.outer(g, P["w2"]) * (H > 0)
            G = {"W1": GAIN * x.T @ dH + l2 * P["W1"], "b1": dH.sum(0),
                 "w2": H.T @ g + l2 * P["w2"], "b2": g.sum()}
            t += 1
            for k in P:
                m[k] = .9 * m[k] + .1 * G[k]
                v[k] = .999 * v[k] + .001 * G[k] ** 2
                step = lr * (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
                P[k] = (P[k] - step).astype(np.float32)
    return P


def net_score(P, X):
    return classifier.hidden(X, P["W1"], P["b1"], P["w2"], float(P["b2"]), GAIN)


def platt(z, y, w):
    """(a, c) for p = sigmoid(a*z + c), by damped Newton on weighted log-loss."""
    z, y, w = z.astype(np.float64), y.astype(np.float64), w.astype(np.float64)
    mu, sd = z.mean(), z.std() + 1e-9
    u = (z - mu) / sd

    def loss(a, c):
        s = a * u + c
        return (w * (np.logaddexp(0, s) - y * s)).sum()

    a, c = 1.0, 0.0
    for _ in range(100):
        p = classifier.sigmoid(a * u + c)
        g, h = w * (p - y), w * p * (1 - p) + 1e-12
        G = np.array([(g * u).sum(), g.sum()])
        H = np.array([[(h * u * u).sum(), (h * u).sum()], [(h * u).sum(), h.sum()]]) + 1e-9 * np.eye(2)
        step, k, before = np.linalg.solve(H, G), 1.0, loss(a, c)
        while loss(a - k * step[0], c - k * step[1]) > before and k > 1e-6:
            k /= 2
        a, c = a - k * step[0], c - k * step[1]
    return float(a / sd), float(c - a * mu / sd)


def folds(conn, ids):
    """A fold for each row, with every duplicate group in one fold."""
    parent = {}

    def root(i):
        while parent.get(i, i) != i:
            parent[i] = parent.get(parent[i], parent[i])
            i = parent[i]
        return i
    for a, b in conn.execute("SELECT a, b FROM dup_pairs"):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) * 2654435761 % 2 ** 32 % FOLDS for i in ids], int)


def out_of_fold(X, y, k, w, fold):
    """Logits for every row from linear heads trained without it."""
    Z = np.zeros((len(y), k), np.float32)
    for f in range(FOLDS):
        W, b = fit(X[fold != f], y[fold != f], k, w[fold != f])
        Z[fold == f] = X[fold == f] @ W * SCALE + b
    return Z


def net_out_of_fold(X, y, w, fold):
    """The network's raw score for every row from networks trained without it."""
    z = np.zeros(len(y), np.float32)
    for f in range(FOLDS):
        if (fold == f).any():
            z[fold == f] = net_score(fit_net(X[fold != f], y[fold != f], w[fold != f]), X[fold == f])
    return z


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
    """(rows, embeddings, fold per row), the rows read in one statement so a
    running scanner cannot change the set halfway."""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT id, status, category, clutter, blurry, kept, label_src, clutter_src, blur_src, embedding FROM files "
        "WHERE embedding IS NOT NULL AND length(embedding) = ? AND status IN ('done','deleted') "
        "AND category IS NOT NULL", (classifier.DIM * 2,)).fetchall()
    X = np.stack([classifier.unpack(r["embedding"]) for r in rows]) if rows else np.zeros((0, classifier.DIM))
    fold = folds(c, [r["id"] for r in rows])
    c.close()
    return rows, X.astype(np.float32), fold


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


def auc(z, y):
    """The chance a random yes outscores a random no."""
    order = np.argsort(z)
    rank = np.empty(len(z))
    rank[order] = np.arange(1, len(z) + 1)
    pos = y == 1
    return (rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1)


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
    no = P[:, yes] <= 0.05
    print(f"  p<=0.05: {no.mean() * 100:.0f}% settled no, {(no & (y == yes)).sum()} of them a yes")
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


def train(rows, X, fold):
    """{array name: array} for the heads file, printing the out-of-fold report."""
    out = {}
    for name, classes, idx, y, w in sets(rows):
        if len(y) < 50 or len(set(y.tolist())) < len(classes):
            print(f"\n{name}: too few examples ({len(y)}), skipped")
            continue
        Xs, fs = X[idx], fold[idx]
        if name == "label":
            Z = out_of_fold(Xs, y, len(classes), w, fs)
            t = temperature(Z, y, w)
            report(name, classes, rows, idx, y, classifier.softmax(Z * t))
            W, b = fit(Xs, y, len(classes), w)
            out.update({f"{name}_W": W, f"{name}_b": (b * t).astype(np.float32),
                        f"{name}_scale": np.float32(SCALE * t), f"{name}_classes": np.array(classes)})
            print(f"  calibrated scale {SCALE * t:.1f}")
            continue
        z = net_out_of_fold(Xs, y, w, fs)
        a, c = platt(z, y, w)
        p = classifier.sigmoid(a * z + c)
        report(name, classes, rows, idx, y, np.stack([1 - p, p], 1))
        print(f"  AUC {auc(z, y):.3f}")
        P = fit_net(Xs, y, w)
        out.update({f"{name}_W1": P["W1"], f"{name}_b1": P["b1"], f"{name}_w2": P["w2"],
                    f"{name}_b2": np.float32(P["b2"]), f"{name}_gain": np.float32(GAIN),
                    f"{name}_a": np.float32(a), f"{name}_c": np.float32(c),
                    f"{name}_classes": np.array(classes)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--out", default=config.CLASSIFIER_HEADS)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)
    rows, X, fold = load(args.db)
    print(f"{len(rows)} labelled pictures with an embedding; folds keep duplicate groups together")
    out = train(rows, X, fold)
    if "label_W" not in out:
        raise SystemExit("not enough labelled pictures to train the label head")
    if args.dry_run:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
