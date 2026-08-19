"""Export the headline model as a PORTABLE artifact, and score a recording with it.

Why: `joblib.dump` of the training objects pickles the custom `Model` /
`AvailabilityScaler` classes, so reloading requires `somnus.train.train` to be
importable and the same sklearn version present. That is fragile for sharing or
publication. This writes a plain-JSON artifact -- column names, per-feature
centring statistics, logistic coefficients, transition matrix -- plus a
dependency-free `predict()` that reproduces the pipeline with numpy only.

Usage:
    python -m somnus.train.export --seed 0                     # write the artifact
    python -m somnus.train.export --seed 0 --score <recording> # and score one file
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from somnus.data import datasets as B
from somnus.train import train as T

STATES = T.STATES
HEADLINE = "unified_z_noind"


def export(seed: int, variant: str = HEADLINE) -> dict:
    csv = os.path.join(B.DATA_DIR, f"train_generalized_seed{seed}.csv.gz")
    if not os.path.exists(csv):
        B.build_training(seed)
    train = pd.read_csv(csv)
    train = train[train["state"].isin(STATES)].reset_index(drop=True)
    y = train["state"].to_numpy()

    guards = T.build_guards(B.model_columns(train)
                            + B.model_columns(train, zscored=True))
    cols = T.feature_set(train, variant)
    model = T.Model(variant, cols, guards).fit(train, y)
    A = T.estimate_transitions(train)
    prior = (pd.Series(y).value_counts(normalize=True)
             .reindex(STATES).fillna(1e-6).to_numpy())

    art = {
        "format_version": 1,
        "variant": variant,
        "seed": seed,
        "states": STATES,
        "classes": list(model.classes_),
        "columns": cols,
        # a feature is "present" where its guard column is 1; None = always
        "guards": {c: guards.get(c) for c in cols},
        "center": [model.scaler.mu_[c] for c in cols],
        "scale": [model.scaler.sd_[c] for c in cols],
        "coef": model.lr.coef_.tolist(),          # (n_classes, n_features)
        "intercept": model.lr.intercept_.tolist(),
        "transition_matrix": A.tolist(),
        "log_prior": np.log(prior / prior.sum()).tolist(),
        "epoch_sec": 4.0,
        "feature_backend": B.FEATURE_BACKEND,
        "notes": (f"Features must be produced by {B.FEATURE_BACKEND} via "
                  "somnus.data.datasets.featurize(). The scipy and openseize "
                  "backends are verified numerically identical by "
                  "tools/verify_openseize_port.py, so either reproduces this "
                  "model. Missing features are set to 0 after centring, i.e. "
                  "they contribute nothing."),
    }
    out = os.path.abspath(f"model_portable_seed{seed}.json")
    with open(out, "w") as fh:
        json.dump(art, fh, indent=2)
    print(f"Wrote portable artifact -> {out}")
    print(f"  variant={variant}  n_features={len(cols)}  classes={art['classes']}")
    return art


def predict(art: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score a featurised recording using only the artifact and numpy.

    Returns (viterbi_labels, class_probabilities). Mirrors AvailabilityScaler:
    standardise where present, zero where absent.
    """
    cols = art["columns"]
    X = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        g = art["guards"].get(c)
        if g and g in df.columns:
            m &= df[g].to_numpy(dtype=float) > 0.5
        z = (v - art["center"][j]) / art["scale"][j]
        X[m, j] = np.nan_to_num(z[m], nan=0.0, posinf=0.0, neginf=0.0)

    logits = X @ np.asarray(art["coef"]).T + np.asarray(art["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)

    # reorder to canonical STATES
    order = [art["classes"].index(s) for s in art["states"]]
    p = p[:, order]

    path = T.viterbi(np.log(np.clip(p, 1e-12, None)),
                     np.asarray(art["transition_matrix"]),
                     np.asarray(art["log_prior"]))
    return np.array(art["states"])[path], p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default=HEADLINE)
    ap.add_argument("--score", default=None,
                    help="recording id from either dataset, to score as a check")
    args = ap.parse_args()

    art = export(args.seed, args.variant)

    if args.score:
        entries = B.discover_local() + B.discover_bids()
        match = [e for e in entries if e["recording"] == args.score]
        if not match:
            print(f"No recording named {args.score}. Examples: "
                  f"{[e['recording'] for e in entries[:3]]}")
            return
        df = B.featurize(match[0])
        labels, p = predict(art, df)
        m = df["state"].isin(STATES).to_numpy()
        print(f"\nScored {args.score}: {len(df)} epochs "
              f"({int(m.sum())} labelled)")
        if m.sum():
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            yt = df.loc[m, "state"].to_numpy()
            print(f"  accuracy={accuracy_score(yt, labels[m]):.4f}  "
                  f"balanced={balanced_accuracy_score(yt, labels[m]):.4f}")
        print("  predicted state counts:",
              {s: int((labels == s).sum()) for s in STATES})


if __name__ == "__main__":
    main()
