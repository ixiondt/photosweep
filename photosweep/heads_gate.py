"""The automatic retrain's test: do freshly trained heads beat the heads in
use, on answers the heads in use have never seen?

Fresh answers are the ones given after the heads in use were trained (the
retrain mark): the owner's keeps and deletes and the vision model's answers,
each dated by when it was given. Both sets of heads are scored on exactly
those rows: the heads in use directly (they cannot have learned them), the new
heads out of fold (each row by heads trained without it, so they cannot vouch
for it either). Out-of-fold heads saw a fifth less data than the ones shipped,
so the comparison leans toward keeping what is in use, which is the safe way
to lean.

A question passes when its weighted log-loss on its fresh rows is no more than
TOLERANCE worse, and, for clutter, when the new heads would not file more of
the owner's fresh keeps as clutter at the trust level than the old would. A
question with fewer than MIN_FRESH fresh answers, or only one kind, is not
judged. The new heads replace the old only when at least one question was
judged and none failed. numpy only; counts and scores only, never a path.
"""
import numpy as np

from . import classifier

MIN_FRESH = 100   # fewer fresh answers than this and a question is not judged
TOLERANCE = 0.02  # log-loss noise allowed on a judged question


def _t(stamp):
    """Timestamps compare as text to the second: the columns carry +00:00 or
    not depending on who wrote them, and both are UTC."""
    return (stamp or "")[:19]


def decided_at(row, name):
    """When the answer a head learns from this row was given, or None.
    Mirrors train_heads.sets(): a keep is the owner's word, then a delete of
    something called clutter, then the vision model's answer."""
    if name == "label":
        return row["scanned_at"]
    if name == "clutter":
        if row["kept"]:
            return row["kept_at"]
        if row["status"] == "deleted" and row["clutter"] == 1:
            return row["deleted_at"]
        return row["clutter_at"]
    # Each question's own date, never asked_at: that one moves on either
    # question, and on a real library 3,891 clutter re-asks made 2,587 old blur
    # answers look new, which the heads in use had learned. Judged on them, old
    # heads won on blur (0.16 log-loss against 0.30) and blocked a clutter head
    # three times better.
    return row["blur_at"]


def logloss(p_true, w):
    return float(-(w * np.log(np.clip(p_true, 1e-6, 1.0))).sum() / max(float(w.sum()), 1e-9))


def judge(rows, X, oof, current_path, since, trust):
    """{"swap": bool, "reasons": [...], "questions": {name: figures}}.
    oof maps a question to (row index, target, weight, classes, out-of-fold P)."""
    cur = classifier.Heads(current_path)
    since = _t(since)
    questions, reasons, judged = {}, [], 0
    for name in classifier.QUESTIONS:
        if name not in oof:
            if name in cur.q:
                questions[name] = {"verdict": "missing"}
                reasons.append(f"{name}: the new heads have no answer for it and the heads in use do")
            continue
        idx, y, w, classes, P = oof[name]
        fresh = np.array([_t(decided_at(rows[i], name)) > since for i in idx], bool)
        n = int(fresh.sum())
        q = questions[name] = {"fresh": n}
        if n < MIN_FRESH or len(set(y[fresh].tolist())) < 2 or name not in cur.q:
            q["verdict"] = "not judged"
            continue
        cur_classes, Pc = cur.full(name, X[idx[fresh]])
        try:
            Pc = Pc[:, [cur_classes.index(c) for c in classes]]
        except ValueError:
            q["verdict"] = "classes differ"
            reasons.append(f"{name}: the heads in use answer with other classes")
            continue
        judged += 1
        yf, wf, Pf = y[fresh], w[fresh], P[fresh]
        rows_n = np.arange(n)
        new, old = logloss(Pf[rows_n, yf], wf), logloss(Pc[rows_n, yf], wf)
        q.update(new_loss=round(new, 4), old_loss=round(old, 4))
        ok = new <= old * (1 + TOLERANCE)
        if not ok:
            reasons.append(f"{name}: log-loss {new:.4f} on {n} new answers, the heads in use {old:.4f}")
        if name == "clutter":
            kept = np.array([bool(rows[i]["kept"]) for i in idx], bool)[fresh]
            if kept.any():
                nf, of = int((kept & (Pf[:, 1] >= trust)).sum()), int((kept & (Pc[:, 1] >= trust)).sum())
                q.update(kept=int(kept.sum()), kept_flagged_new=nf, kept_flagged_old=of)
                if nf > of:
                    ok = False
                    reasons.append(f"clutter: would file {nf} of {int(kept.sum())} new keeps as clutter "
                                   f"at p>={trust}, the heads in use {of}")
        q["verdict"] = ("better" if new < old else "pass") if ok else "worse"
    if not judged:
        reasons.append(f"too few new answers to judge any question (need {MIN_FRESH})")
    return {"swap": judged > 0 and not reasons, "reasons": reasons, "questions": questions}
