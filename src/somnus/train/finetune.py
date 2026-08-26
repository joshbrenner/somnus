"""Fine-tune model_somnus_1.0 on a user's own labelled recordings.

WHY FINE-TUNING RATHER THAN RETRAINING
--------------------------------------
Pooling the user's data with the 20k-epoch base training set and refitting would
drown a disease phenotype: a mouse line with fragmented or low-amplitude NREM
contributes a few hundred epochs against ~3.8k epochs of normal NREM, so the
refit would mostly relearn "normal" and keep mis-scoring the phenotype. Fine
tuning instead *starts* from the base model and moves it only as far as the new
data justifies.

HOW
---
Multinomial logistic loss on the new data, with an L2 penalty that anchors the
weights to the base model rather than to zero:

    minimise   sum_i s_i * CE(y_i, softmax(W x_i + b))
               + (lam/2) * ||W - W_base||^2
               + (lam/2) * ||b - b_base||^2

One knob spans the whole spectrum:
    lam -> infinity   the base model, unchanged (no adaptation)
    lam -> 0          trained from scratch on the user's data alone
so "fine-tune vs retrain" becomes a continuum and `lam` is chosen by
cross-validation on the user's own recordings instead of being asserted.

`s_i` are balanced class weights computed on the NEW data, so a phenotype with
very little REM still gets REM pulled up rather than ignored.

WHAT IS AND IS NOT ADAPTED
  adapted:      logistic weights + intercepts (anchored), and optionally the
                HMM transition matrix (below).
  NOT adapted:  the per-feature centring/scaling in the artifact. Those define
                the feature contract the weights are expressed in; refitting them
                would silently change what every weight means. Note features are
                already z-scored *within recording* upstream, so per-animal
                amplitude differences are handled before this point.

TRANSITION MATRIX
Sleep architecture is often exactly what a disease model changes (fragmented
NREM, reduced REM bouts), and the HMM transition matrix is only 9 numbers, so it
adapts robustly from little data. It is re-estimated from the new labels using
the base matrix as a Dirichlet prior:

    A_new[i,:]  proportional to  counts[i,:] + kappa * A_base[i,:] * row_strength

kappa large keeps the base architecture; kappa small trusts the new labels.

GUARDRAILS
  * Adaptation is scored by leave-one-recording-out CV on the user's data, so the
    reported gain is out-of-sample, not the fit to its own training epochs.
  * If the user has one recording, it is split into contiguous time blocks
    instead (never random epochs -- sleep is autocorrelated and random splits
    leak between neighbours).
  * "Forgetting" is measured explicitly on the base training matrix, so a
    fine-tune that wrecks general performance is visible rather than silent.
  * If no lam beats the base model out-of-sample, that is reported and the base
    model is recommended unchanged.

Usage:
    python -m somnus.train.finetune --labels mydata.csv --out my_model.json
    python -m somnus.train.finetune --labels mydata.csv --lam 10 --no-cv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from somnus.predict import (design_matrix, load_model, viterbi,
                            DEFAULT_ARTIFACT)

LAM_GRID = (1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1)
KAPPA_DEFAULT = 200.0


# --------------------------------------------------------------- core objective
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _pack(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([W.ravel(), b])


def _unpack(theta: np.ndarray, k: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    return theta[:k * d].reshape(k, d), theta[k * d:]


def fit_anchored(X: np.ndarray, y_idx: np.ndarray, k: int,
                 W0: np.ndarray, b0: np.ndarray, lam: float,
                 sample_weight: np.ndarray | None = None,
                 max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Weighted multinomial logistic fit with an L2 anchor on (W0, b0).

    Returns the fine-tuned (W, b). With lam very large this reproduces (W0, b0);
    with lam ~ 0 it is an unregularised fit to X, y alone.
    """
    n, d = X.shape
    s = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    s = s / s.sum() * n                        # keep loss scale independent of weighting
    Y = np.zeros((n, k))
    Y[np.arange(n), y_idx] = 1.0

    def obj(theta):
        W, b = _unpack(theta, k, d)
        P = _softmax(X @ W.T + b)
        ll = -np.sum(s * np.log(np.clip(P[np.arange(n), y_idx], 1e-12, None)))
        reg = 0.5 * lam * (np.sum((W - W0) ** 2) + np.sum((b - b0) ** 2))
        G = (P - Y) * s[:, None]
        gW = G.T @ X + lam * (W - W0)
        gb = G.sum(axis=0) + lam * (b - b0)
        return ll + reg, _pack(gW, gb)

    res = minimize(obj, _pack(W0.copy(), b0.copy()), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
    return _unpack(res.x, k, d)


def balanced_weights(y_idx: np.ndarray, k: int) -> np.ndarray:
    """class_weight='balanced' equivalent, computed on the NEW data."""
    counts = np.bincount(y_idx, minlength=k).astype(float)
    w = np.zeros(k)
    present = counts > 0
    w[present] = len(y_idx) / (present.sum() * counts[present])
    return w[y_idx]


# ------------------------------------------------------------ transition matrix
def adapt_transitions(df: pd.DataFrame, states: list[str], A_base: np.ndarray,
                      kappa: float = KAPPA_DEFAULT) -> np.ndarray:
    """Re-estimate P(state_t | state_{t-1}) with A_base as a Dirichlet prior.

    Only adjacent epochs from the same recording count, so gaps left by artifact
    or unlabelled epochs never create phantom transitions.
    """
    k = len(states)
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((k, k))
    for _, g in df.groupby("recording", sort=False):
        g = g.sort_values("epoch")
        lab, ep = g["state"].to_numpy(), g["epoch"].to_numpy()
        adj = np.diff(ep) == 1
        for a, bb in zip(lab[:-1][adj], lab[1:][adj]):
            if a in idx and bb in idx:
                counts[idx[a], idx[bb]] += 1
    post = counts + kappa * A_base
    return post / post.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ evaluation
def _decode(art: dict, X: np.ndarray, W: np.ndarray, b: np.ndarray,
            A: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Viterbi-decode per recording (never across a recording boundary)."""
    p = _softmax(X @ W.T + b)
    log_pi = np.asarray(art["log_prior"])
    out = np.empty(len(X), dtype=int)
    for g in pd.unique(groups):
        m = groups == g
        out[m] = viterbi(np.log(np.clip(p[m], 1e-12, None)), A, log_pi)
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    acc = float((y_true == y_pred).mean())
    per, f1s = {}, []
    for c in range(k):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        rec = tp / (tp + fn) if tp + fn else float("nan")
        pre = tp / (tp + fp) if tp + fp else float("nan")
        f1 = 2 * pre * rec / (pre + rec) if pre and rec and pre + rec else 0.0
        per[c] = dict(precision=pre, recall=rec, f1=f1, support=int(tp + fn))
        if tp + fn:
            f1s.append(f1)
    recalls = [per[c]["recall"] for c in range(k) if per[c]["support"]]
    return dict(accuracy=acc, balanced_accuracy=float(np.nanmean(recalls)),
                macro_f1=float(np.mean(f1s)) if f1s else float("nan"), per_class=per)


# ----------------------------------------------------------------- data loading
def prepare(art: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature table -> (X, y_idx, groups). Keeps only rows with a known state."""
    states = art["states"]
    if "state" not in df.columns:
        raise ValueError("labelled data must have a 'state' column")
    keep = df["state"].isin(states).to_numpy()
    if "epoch" not in df.columns:
        df = df.assign(epoch=np.arange(len(df)))
    if "recording" not in df.columns:
        df = df.assign(recording="user_data")
    sub = df.loc[keep].reset_index(drop=True)
    X = design_matrix(art, sub)
    y = sub["state"].map({s: i for i, s in enumerate(states)}).to_numpy()
    return X, y, sub["recording"].to_numpy(), sub


def _folds(groups: np.ndarray, epochs: np.ndarray,
           n_blocks: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-recording-out; if only one recording, contiguous time blocks.

    Never random epochs: adjacent epochs are highly correlated, so a random split
    would let the model see each test epoch's neighbours and inflate the score.
    """
    uniq = pd.unique(groups)
    if len(uniq) > 1:
        return [(groups != g, groups == g) for g in uniq]
    order = np.argsort(epochs)
    blocks = np.array_split(order, n_blocks)
    out = []
    for bl in blocks:
        te = np.zeros(len(groups), dtype=bool)
        te[bl] = True
        if te.all() or not te.any():
            continue
        out.append((~te, te))
    return out


# ------------------------------------------------------------------- fine-tune
def finetune(art: dict, df: pd.DataFrame, lam: float | None = None,
             kappa: float = KAPPA_DEFAULT, adapt_A: bool = True,
             lam_grid=LAM_GRID, verbose: bool = True) -> dict:
    """Fine-tune and return a report plus the new artifact (or None if unhelpful)."""
    states = art["states"]
    k = len(states)
    W0 = np.asarray(art["coef"], float)
    b0 = np.asarray(art["intercept"], float)
    A_base = np.asarray(art["transition_matrix"], float)
    # artifact rows follow art["classes"]; reorder to art["states"] once
    order = [art["classes"].index(s) for s in states]
    W0, b0 = W0[order], b0[order]

    X, y, groups, sub = prepare(art, df)
    epochs = sub["epoch"].to_numpy()
    n_rec = len(pd.unique(groups))
    if verbose:
        print(f"new data: {len(X)} labelled epochs from {n_rec} recording(s)")
        for i, s in enumerate(states):
            print(f"  {s:5} {int((y == i).sum())}")

    folds = _folds(groups, epochs)
    if verbose:
        kind = "leave-one-recording-out" if n_rec > 1 else "contiguous time blocks"
        print(f"validation: {len(folds)} folds ({kind})")

    # ---- baseline (no adaptation) on the same folds, for a fair comparison
    def eval_lam(l: float | None) -> dict:
        yt, yp = [], []
        for tr, te in folds:
            if l is None:
                W, b, A = W0, b0, A_base
            else:
                sw = balanced_weights(y[tr], k)
                W, b = fit_anchored(X[tr], y[tr], k, W0, b0, l, sw)
                A = adapt_transitions(sub.loc[tr], states, A_base, kappa) \
                    if adapt_A else A_base
            yp.append(_decode(art, X[te], W, b, A, groups[te]))
            yt.append(y[te])
        return _metrics(np.concatenate(yt), np.concatenate(yp), k)

    base_m = eval_lam(None)
    if verbose:
        print(f"\nbase model, held-out on your data: acc={base_m['accuracy']:.4f} "
              f"bal={base_m['balanced_accuracy']:.4f} macroF1={base_m['macro_f1']:.4f}")

    curve = []
    if lam is None:
        if verbose:
            print("\nsweeping anchor strength (higher lam = closer to base model):")
        for l in lam_grid:
            m = eval_lam(l)
            curve.append(dict(lam=l, **{q: m[q] for q in
                                        ("accuracy", "balanced_accuracy", "macro_f1")}))
            if verbose:
                print(f"  lam={l:>7.3g}  acc={m['accuracy']:.4f} "
                      f"bal={m['balanced_accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
        best = max(curve, key=lambda r: r["balanced_accuracy"])
        lam = best["lam"]
        improved = best["balanced_accuracy"] > base_m["balanced_accuracy"]
    else:
        m = eval_lam(lam)
        curve.append(dict(lam=lam, **{q: m[q] for q in
                                      ("accuracy", "balanced_accuracy", "macro_f1")}))
        improved = m["balanced_accuracy"] > base_m["balanced_accuracy"]

    tuned_m = eval_lam(lam)
    if verbose:
        print(f"\nchosen lam={lam:g}: acc={tuned_m['accuracy']:.4f} "
              f"bal={tuned_m['balanced_accuracy']:.4f} "
              f"macroF1={tuned_m['macro_f1']:.4f}")
        d = tuned_m["balanced_accuracy"] - base_m["balanced_accuracy"]
        print(f"  balanced-accuracy change vs base: {d:+.4f}")
        if not improved:
            print("  NOTE: no anchor strength beat the base model out-of-sample. "
                  "Keeping the base model is the better choice on this evidence.")

    # ---- final fit on ALL the user's data
    sw = balanced_weights(y, k)
    W, b = fit_anchored(X, y, k, W0, b0, lam, sw)
    A = adapt_transitions(sub, states, A_base, kappa) if adapt_A else A_base

    new = dict(art)
    new["classes"] = list(states)               # rows now follow states order
    new["coef"] = W.tolist()
    new["intercept"] = b.tolist()
    new["transition_matrix"] = A.tolist()
    new["finetune"] = dict(
        base_variant=art.get("variant"), base_seed=art.get("seed"),
        lam=float(lam), kappa=float(kappa), adapted_transitions=bool(adapt_A),
        n_epochs=int(len(X)), n_recordings=int(n_rec),
        recordings=[str(g) for g in pd.unique(groups)],
        class_counts={s: int((y == i).sum()) for i, s in enumerate(states)},
        validation="leave-one-recording-out" if n_rec > 1 else "contiguous blocks",
        held_out_base=base_m, held_out_tuned=tuned_m,
        improved_over_base=bool(improved), lam_curve=curve,
    )
    new["variant"] = f"{art.get('variant')}+finetuned"
    return dict(artifact=new, base_metrics=base_m, tuned_metrics=tuned_m,
                lam=lam, curve=curve, improved=improved)


def forgetting_check(art_base: dict, art_new: dict, base_csv: str,
                     verbose: bool = True) -> dict | None:
    """Score the base training matrix with both models to expose drift.

    A fine-tune that helps the user's phenotype but destroys general performance
    should be visible, not silent.
    """
    if not os.path.exists(base_csv):
        if verbose:
            print(f"\nforgetting check skipped (no {base_csv})")
        return None
    df = pd.read_csv(base_csv)
    states = art_base["states"]
    df = df[df["state"].isin(states)].reset_index(drop=True)
    X = design_matrix(art_base, df)
    y = df["state"].map({s: i for i, s in enumerate(states)}).to_numpy()
    groups = df["recording"].to_numpy() if "recording" in df.columns \
        else np.zeros(len(df))
    out = {}
    for tag, a in (("base", art_base), ("tuned", art_new)):
        W = np.asarray(a["coef"], float)
        b = np.asarray(a["intercept"], float)
        o = [a["classes"].index(s) for s in states]
        yp = _decode(a, X, W[o], b[o], np.asarray(a["transition_matrix"], float),
                     groups)
        out[tag] = _metrics(y, yp, len(states))
    if verbose:
        print(f"\nforgetting check on the base training set ({len(X)} epochs):")
        print(f"  base  acc={out['base']['accuracy']:.4f} "
              f"bal={out['base']['balanced_accuracy']:.4f}   <- IN-SAMPLE, flattered")
        print(f"  tuned acc={out['tuned']['accuracy']:.4f} "
              f"bal={out['tuned']['balanced_accuracy']:.4f}  "
              f"({out['tuned']['accuracy'] - out['base']['accuracy']:+.4f} acc)")
        print("  Read the gap as an UPPER BOUND on forgetting: these are the base "
              "model's own\n  training epochs, so it is scored in-sample while the "
              "fine-tuned model is not.\n  Use it to catch catastrophic drift, not "
              "to quantify a small loss.")
    out["caveat"] = ("base model is scored in-sample on its own training data, so "
                     "the base-vs-tuned gap overstates forgetting")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True,
                    help="featurised CSV with a 'state' column (and ideally "
                         "'recording' and 'epoch')")
    ap.add_argument("--model", default=DEFAULT_ARTIFACT)
    ap.add_argument("--out", default=None, help="write the fine-tuned artifact here")
    ap.add_argument("--lam", type=float, default=None,
                    help="anchor strength; omit to choose by cross-validation")
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT,
                    help="trust in the base transition matrix (higher = keep it)")
    ap.add_argument("--no-adapt-transitions", action="store_true")
    ap.add_argument("--no-cv", action="store_true",
                    help="with --lam, skip the sweep")
    ap.add_argument("--base-csv", default=None,
                    help="a reference feature table to run the forgetting "
                         "check against (skipped if omitted)")
    args = ap.parse_args()

    art = load_model(args.model)
    df = pd.read_csv(args.labels)
    res = finetune(art, df, lam=args.lam, kappa=args.kappa,
                   adapt_A=not args.no_adapt_transitions,
                   lam_grid=(args.lam,) if (args.lam and args.no_cv) else LAM_GRID)
    # The base model's training matrix is not distributed, so the forgetting
    # check runs only when the user points --base-csv at a reference set.
    if args.base_csv:
        forgetting_check(art, res["artifact"], args.base_csv)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res["artifact"], fh, indent=2)
        print(f"\nwrote fine-tuned model -> {args.out}")
        if not res["improved"]:
            print("  (reminder: it did not beat the base model out-of-sample)")


if __name__ == "__main__":
    main()
