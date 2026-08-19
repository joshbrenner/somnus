"""Train the cross-dataset sleep scorer and evaluate on held-out subjects.

MODEL
-----
Multinomial logistic regression, balanced class weights, plus an HMM/Viterbi
decode for state inertia (transition matrix estimated from training labels only).

HOW MISSING FEATURES ARE HANDLED (the "learn to deal with NaN" requirement)
--------------------------------------------------------------------------
Each optional feature is guarded by an availability indicator. In training and at
prediction the feature is standardised using statistics from the rows where it is
PRESENT, and rows where it is absent are set to 0 -- i.e. exactly the centred
mean, so they contribute nothing to the linear predictor. The indicator column
then absorbs the baseline shift for that condition.

    contribution = beta_x * z(x) * present + beta_ind * present

So a feature supplies its own slope when measured and cleanly drops out when not,
with no imputed phantom value and no need to retrain per data type. This is the
linear-model equivalent of "handling NaN", and it is why one model can score
128 Hz EEG-only data and 5000 Hz EEG+EMG+video data.

FEATURE SETS COMPARED
  universal : tier-1 spectral + low-band EMG only. Computable on literally any
              recording, including 25 Hz-lowpass-filtered EEG.
  no_video  : everything the EEG/EMG can give, with tier indicators. No video.
  unified   : no_video + velocity. The full model.

Usage:
    python -m somnus.train.train --seed 0
    python -m somnus.train.train --seed 0 --max-self-transition 0.95
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             classification_report, f1_score)

from somnus.data import datasets as B

STATES = ["Wake", "NREM", "REM"]
STATE_COLORS = {"Wake": "#999999", "NREM": "#9470DB", "REM": "#DB3333"}

VIDEO_COLS_ROOT = ["log_velocity_z"]


# ---------------------------------------------------- availability-aware scaler
class AvailabilityScaler:
    """Standardise using present-rows only; set absent entries to 0.

    `guards` maps a feature column -> indicator column. A feature is 'present'
    where its guard is 1 and the value is finite. Ungurded columns fall back to
    finiteness only.
    """

    def __init__(self, columns: list[str], guards: dict[str, str]):
        self.columns = columns
        self.guards = guards
        self.mu_: dict[str, float] = {}
        self.sd_: dict[str, float] = {}

    def _mask(self, df: pd.DataFrame, c: str) -> np.ndarray:
        m = np.isfinite(df[c].to_numpy(dtype=float))
        g = self.guards.get(c)
        if g and g in df.columns:
            m &= df[g].to_numpy(dtype=float) > 0.5
        return m

    def fit(self, df: pd.DataFrame) -> "AvailabilityScaler":
        for c in self.columns:
            if c not in df.columns:
                self.mu_[c], self.sd_[c] = 0.0, 1.0
                continue
            v = df[c].to_numpy(dtype=float)
            m = self._mask(df, c)
            if m.sum() < 10:
                self.mu_[c], self.sd_[c] = 0.0, 1.0
            else:
                self.mu_[c] = float(np.mean(v[m]))
                sd = float(np.std(v[m]))
                self.sd_[c] = sd if sd > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.columns)), dtype=float)
        for j, c in enumerate(self.columns):
            if c not in df.columns:
                continue
            v = df[c].to_numpy(dtype=float)
            z = (v - self.mu_[c]) / self.sd_[c]
            m = self._mask(df, c)
            out[m, j] = np.nan_to_num(z[m], nan=0.0, posinf=0.0, neginf=0.0)
        return out


def build_guards(columns: list[str]) -> dict[str, str]:
    """Map each optional feature (and its context variants) to its indicator."""
    guards = {}
    for _, (cols, ind) in B.OPTIONAL.items():
        for root in cols:
            for c in columns:
                if c == root or c.startswith(root + "_mean") \
                        or c.startswith(root + "_std"):
                    guards[c] = ind
    return guards


# Variants compared. Each is (use z-scored columns, include optional tiers,
# include availability indicators, include video).
VARIANTS = {
    # raw-feature baselines (what the first evaluation ran)
    "universal":        dict(z=False, optional=False, indicators=False, video=False),
    "unified":          dict(z=False, optional=True,  indicators=True,  video=True),
    # site-offset removed by within-recording z-scoring
    "universal_z":      dict(z=True,  optional=False, indicators=False, video=False),
    "unified_z":        dict(z=True,  optional=True,  indicators=True,  video=True),
    # z-scored AND indicators dropped: tests whether the indicators themselves
    # were the leak (they correlate with lab identity)
    "unified_z_noind":  dict(z=True,  optional=True,  indicators=False, video=True),
    "novideo_z_noind":  dict(z=True,  optional=True,  indicators=False, video=False),
}


def feature_set(df: pd.DataFrame, which: str) -> list[str]:
    """Columns for a named variant, restricted to what exists in df."""
    cfg = VARIANTS[which]
    all_cols = B.model_columns(df, zscored=cfg["z"])
    inds = {ind for _, ind in B.OPTIONAL.values()}
    univ = list(B.UNIVERSAL)
    if cfg["z"]:
        univ = [c if c.endswith("_z") else f"{c}_z" for c in univ]

    def is_universal(c: str) -> bool:
        for r in univ:
            root = r[:-2] if r.endswith("_z") else r
            suf = "_z" if r.endswith("_z") else ""
            if c == r or c.startswith(f"{root}_mean") or c.startswith(f"{root}_std"):
                if not suf or c.endswith("_z"):
                    return True
        return False

    def is_video(c: str) -> bool:
        return any(c == r or c.startswith(r[:-2] if r.endswith("_z") else r)
                   for r in ["log_velocity_z", "log_velocity"]) \
            and "velocity" in c

    keep = []
    for c in all_cols:
        if c in inds:
            if cfg["indicators"]:
                keep.append(c)
            continue
        if is_video(c):
            if cfg["video"]:
                keep.append(c)
            continue
        if is_universal(c):
            keep.append(c)
        elif cfg["optional"]:
            keep.append(c)
    return [c for c in dict.fromkeys(keep) if c in df.columns]


# --------------------------------------------------------------------- inertia
def estimate_transitions(df: pd.DataFrame, laplace: float = 1.0,
                         max_self: float | None = None) -> np.ndarray:
    """P(state_t | state_{t-1}) from adjacent labelled epoch pairs.

    Only pairs from the same recording with an epoch gap of exactly 1 count, so
    gaps left by artifact/mixed epochs never create phantom transitions.
    """
    k = len(STATES)
    idx = {s: i for i, s in enumerate(STATES)}
    c = np.full((k, k), laplace, dtype=float)
    for _, g in df.groupby("recording"):
        g = g.sort_values("epoch")
        lab, ep = g["state"].to_numpy(), g["epoch"].to_numpy()
        adj = np.diff(ep) == 1
        for a, b, ok in zip(lab[:-1], lab[1:], adj):
            if ok and isinstance(a, str) and isinstance(b, str):
                c[idx[a], idx[b]] += 1
    A = c / c.sum(axis=1, keepdims=True)
    if max_self is not None:
        for i in range(k):
            if A[i, i] > max_self:
                off = A[i].copy()
                off[i] = 0.0
                s = off.sum()
                A[i] = off / s * (1 - max_self) if s > 0 else (1 - max_self) / (k - 1)
                A[i, i] = max_self
    return A


def viterbi(log_em: np.ndarray, A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    T, K = log_em.shape
    logA = np.log(np.clip(A, 1e-300, None))
    d = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)
    d[0] = log_pi + log_em[0]
    for t in range(1, T):
        sc = d[t - 1][:, None] + logA
        psi[t] = np.argmax(sc, axis=0)
        d[t] = sc[psi[t], np.arange(K)] + log_em[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ---------------------------------------------------------------------- model
class Model:
    def __init__(self, name: str, columns: list[str], guards: dict[str, str]):
        self.name = name
        self.columns = columns
        self.scaler = AvailabilityScaler(columns, guards)
        # tol=1e-8 rather than the 1e-4 default: at 1e-4 L-BFGS stops well short
        # of the optimum, so coefficients are not reproducible across runs whose
        # features differ only by floating-point noise (measured: coef drift
        # 8.3e-2 at 1e-4 vs 4.3e-5 at 1e-8, with zero label changes at <=1e-6).
        # Reproducible coefficients matter here because they are reported.
        self.lr = LogisticRegression(max_iter=20000, class_weight="balanced",
                                     C=1.0, solver="lbfgs", tol=1e-8)
        self.classes_: list[str] = []

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "Model":
        X = self.scaler.fit(df).transform(df)
        self.lr.fit(X, y)
        self.classes_ = list(self.lr.classes_)
        return self

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        """Class probabilities, columns reordered to STATES."""
        p = self.lr.predict_proba(self.scaler.transform(df))
        out = np.zeros((len(df), len(STATES)))
        for j, s in enumerate(STATES):
            if s in self.classes_:
                out[:, j] = p[:, self.classes_.index(s)]
        rs = out.sum(axis=1, keepdims=True)
        return out / np.clip(rs, 1e-12, None)


def metrics_of(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=STATES,
                                   average="macro", zero_division=0)),
        "f1_REM": float(f1_score(y_true, y_pred, labels=["REM"],
                                 average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train and evaluate the cross-dataset sleep scorer.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-self-transition", type=float, default=None)
    ap.add_argument("--limit-test", type=int, default=None,
                    help="evaluate only the first N held-out recordings")
    args = ap.parse_args()

    train_csv = os.path.join(B.DATA_DIR, f"train_generalized_seed{args.seed}.csv.gz")
    man_path = os.path.join(B.DATA_DIR, f"manifest_generalized_seed{args.seed}.json")
    # Results go under the invoking directory, not inside the package.
    outdir = os.path.abspath(f"results_generalized_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    with open(man_path) as fh:
        manifest = json.load(fh)

    train = pd.read_csv(train_csv)
    train = train[train["state"].isin(STATES)].reset_index(drop=True)
    y = train["state"].to_numpy()
    print(f"Training epochs: {len(train)}")
    print(pd.crosstab(train["dataset"], train["state"]).to_string())

    guards = build_guards(B.model_columns(train) + B.model_columns(train, zscored=True))
    sets = {name: feature_set(train, name) for name in VARIANTS}
    for n, c in sets.items():
        print(f"  feature set '{n}': {len(c)} columns")

    models = {}
    for n, cols in sets.items():
        models[n] = Model(n, cols, guards).fit(train, y)
        print(f"  fitted {n}")

    A = estimate_transitions(train, max_self=args.max_self_transition)
    prior = (pd.Series(y).value_counts(normalize=True)
             .reindex(STATES).fillna(1e-6).to_numpy())
    log_pi = np.log(prior / prior.sum())
    print("\nTransition matrix (from training labels):")
    print(pd.DataFrame(A, index=STATES, columns=STATES)
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # ----------------------- stream over the held-out test set --------------
    train_subs = set(manifest["bids_train_subjects"])
    print(f"\nEvaluating on the held-out test set: "
          f"{manifest['n_bids_test_subjects']} subjects "
          f"(train mice excluded: {sorted(train_subs)})")

    per_rec = []
    pooled_true = {n: [] for n in sets}
    pooled_pred = {n: [] for n in sets}
    pooled_pred_greedy = {n: [] for n in sets}
    n_done = 0

    for meta, df in B.iter_test_recordings(train_subs):
        lab_mask = df["state"].isin(STATES).to_numpy()
        if lab_mask.sum() < 50:
            continue
        yt = df.loc[lab_mask, "state"].to_numpy()
        row = {"recording": meta["recording"], "subject": meta["subject"],
               "lab": meta["group"], "n_labelled": int(lab_mask.sum()),
               "eeg_edge_hz": meta["eeg_edge_hz"], "tiers": str(meta["tiers"])}
        for n, m in models.items():
            p = m.proba(df)                                # all epochs
            greedy = np.array(STATES)[np.argmax(p, axis=1)]
            vit = np.array(STATES)[viterbi(np.log(np.clip(p, 1e-12, None)),
                                           A, log_pi)]
            row[f"acc_{n}"] = accuracy_score(yt, vit[lab_mask])
            row[f"bacc_{n}"] = balanced_accuracy_score(yt, vit[lab_mask])
            row[f"kappa_{n}"] = cohen_kappa_score(yt, vit[lab_mask])
            row[f"accg_{n}"] = accuracy_score(yt, greedy[lab_mask])
            pooled_true[n].append(yt)
            pooled_pred[n].append(vit[lab_mask])
            pooled_pred_greedy[n].append(greedy[lab_mask])
        per_rec.append(row)
        n_done += 1
        if n_done % 10 == 0:
            print(f"  ...{n_done} recordings evaluated", flush=True)
        if args.limit_test and n_done >= args.limit_test:
            break

    rec_df = pd.DataFrame(per_rec)
    rec_df.to_csv(os.path.join(outdir, "per_recording_metrics.csv"), index=False)

    # ----------------------------- reporting -------------------------------
    print("\n" + "=" * 74)
    print("POOLED RESULTS on held-out mice (Viterbi decode)")
    print("=" * 74)
    summary = {}
    for n in sets:
        yt = np.concatenate(pooled_true[n])
        yp = np.concatenate(pooled_pred[n])
        yg = np.concatenate(pooled_pred_greedy[n])
        summary[n] = {"viterbi": metrics_of(yt, yp), "greedy": metrics_of(yt, yg),
                      "report": classification_report(yt, yp, labels=STATES,
                                                      output_dict=True,
                                                      zero_division=0)}
        v = summary[n]["viterbi"]
        print(f"\n--- {n} ---  n={v['n']}")
        print(f"accuracy={v['accuracy']:.4f}  balanced={v['balanced_accuracy']:.4f}"
              f"  kappa={v['kappa']:.4f}  macroF1={v['f1_macro']:.4f}"
              f"  REM_F1={v['f1_REM']:.4f}   (greedy acc "
              f"{summary[n]['greedy']['accuracy']:.4f})")
        print(classification_report(yt, yp, labels=STATES, zero_division=0))

    # pick the best variant by pooled balanced accuracy for the headline outputs
    best = max(sets, key=lambda n: summary[n]["viterbi"]["balanced_accuracy"])
    print(f"\nBest variant by pooled balanced accuracy: {best}")
    yt = np.concatenate(pooled_true[best])
    yp = np.concatenate(pooled_pred[best])
    cm = confusion_matrix(yt, yp, labels=STATES)
    print(f"Confusion matrix ({best}, rows=true):")
    print(pd.DataFrame(cm, index=STATES, columns=STATES).to_string())

    print("\n" + "=" * 74)
    print("PER-LAB BREAKDOWN (epoch-weighted accuracy, Viterbi)")
    print("=" * 74)

    def lab_row(g: pd.DataFrame) -> pd.Series:
        d = {"n_recordings": len(g), "n_epochs": int(g["n_labelled"].sum())}
        for n in sets:
            d[f"acc_{n}"] = np.average(g[f"acc_{n}"], weights=g["n_labelled"])
        d[f"bacc_{best}"] = np.average(g[f"bacc_{best}"], weights=g["n_labelled"])
        d["median_eeg_edge"] = g["eeg_edge_hz"].median()
        return pd.Series(d)

    lab_tab = rec_df.groupby("lab").apply(lab_row, include_groups=False).round(4)
    print(lab_tab.to_string())
    lab_tab.to_csv(os.path.join(outdir, "per_lab_metrics.csv"))

    print(f"\nPER-MOUSE distribution of accuracy ({best}, Viterbi):")
    pm = rec_df.groupby("subject").apply(
        lambda g: np.average(g[f"acc_{best}"], weights=g["n_labelled"]),
        include_groups=False)
    print(f"  n_mice={len(pm)}  min={pm.min():.3f}  q05={pm.quantile(.05):.3f}  "
          f"median={pm.median():.3f}  q95={pm.quantile(.95):.3f}  max={pm.max():.3f}")
    worst = pm.sort_values().head(5)
    print("  worst 5 mice: " + ", ".join(f"{k}={v:.3f}" for k, v in worst.items()))

    # ------------------- secondary: leave-one-out within the local corpus ---
    print("\n" + "=" * 74)
    print("SECONDARY: leave-one-recording-out within the local corpus")
    print("(all local data is in the main training set, so this refits to give an "
          "honest out-of-sample number and to isolate the video contribution)")
    print("=" * 74)
    local = train[train["dataset"] == "local"]
    local_rows = []
    for rec in sorted(local["recording"].unique()):
        te = train["recording"] == rec
        tr = ~te
        r = {"recording": rec, "n": int((te & train["state"].isin(STATES)).sum())}
        for n, cols in sets.items():
            m = Model(n, cols, guards).fit(train[tr], train.loc[tr, "state"].to_numpy())
            p = m.proba(train[te])
            vit = np.array(STATES)[viterbi(np.log(np.clip(p, 1e-12, None)),
                                           estimate_transitions(
                                               train[tr],
                                               max_self=args.max_self_transition),
                                           log_pi)]
            yt2 = train.loc[te, "state"].to_numpy()
            r[f"acc_{n}"] = accuracy_score(yt2, vit)
            r[f"bacc_{n}"] = balanced_accuracy_score(yt2, vit)
        local_rows.append(r)
        print(f"  {rec}: " + "  ".join(
            f"{n} acc={r[f'acc_{n}']:.3f}/bacc={r[f'bacc_{n}']:.3f}" for n in sets))
    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(os.path.join(outdir, "local_loro_metrics.csv"), index=False)
    print("\n  weighted mean over local recordings:")
    for n in sets:
        print(f"    {n}: acc={np.average(local_df[f'acc_{n}'], weights=local_df['n']):.4f}"
              f"  bacc={np.average(local_df[f'bacc_{n}'], weights=local_df['n']):.4f}")

    # ------------------------------- save ---------------------------------
    joblib.dump({"models": {n: models[n] for n in models},
                 "transition_matrix": A.tolist(), "log_pi": log_pi.tolist(),
                 "states": STATES, "feature_sets": sets, "guards": guards,
                 "best_variant": best, "seed": args.seed},
                os.path.join(outdir, "model.joblib"))
    coef = pd.DataFrame(models[best].lr.coef_, index=models[best].classes_,
                        columns=models[best].columns).T
    coef.to_csv(os.path.join(outdir, f"coefficients_{best}.csv"))
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump({"seed": args.seed, "pooled_heldout": summary,
                   "per_lab": lab_tab.reset_index().to_dict("records"),
                   "per_mouse_accuracy_quantiles": {
                       "min": float(pm.min()), "q05": float(pm.quantile(.05)),
                       "median": float(pm.median()),
                       "q95": float(pm.quantile(.95)), "max": float(pm.max())},
                   "local_loro": local_rows,
                   "transition_matrix": A.tolist(),
                   "feature_set_sizes": {n: len(c) for n, c in sets.items()},
                   "n_test_recordings": int(len(rec_df))}, fh, indent=2)

    _plots(rec_df, lab_tab, pm, cm, coef, A, summary, sets, best, outdir)
    print(f"\nWrote results -> {outdir}")


def _plots(rec_df, lab_tab, pm, cm, coef, A, summary, sets, best, outdir):
    # per-lab accuracy by variant
    names = list(sets)
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(names)), 4.6))
    labs = list(lab_tab.index)
    x = np.arange(len(labs))
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(x + (k - (len(names) - 1) / 2) * w, lab_tab[f"acc_{n}"], w, label=n)
    ax.set_xticks(x, labs)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("epoch accuracy (Viterbi)")
    ax.set_title("Held-out accuracy by lab and feature set")
    ax.legend()
    for i, lab in enumerate(labs):
        ax.text(i, 0.52, f"n={int(lab_tab.loc[lab,'n_epochs'])}",
                ha="center", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "per_lab_accuracy.png"), dpi=150)
    plt.close(fig)

    # per-mouse distribution
    fig, ax = plt.subplots(figsize=(7, 4.6))
    data = [rec_df[rec_df.lab == l][f"acc_{best}"].values
            for l in sorted(rec_df.lab.unique())]
    ax.boxplot(data, tick_labels=sorted(rec_df.lab.unique()), showfliers=True)
    for i, d in enumerate(data, start=1):        # overlay individual recordings
        if len(d):
            jitter = (np.random.RandomState(i).rand(len(d)) - 0.5) * 0.18
            ax.scatter(i + jitter, d, s=10, alpha=0.5, color="k", zorder=3)
    ax.set_ylabel("per-recording accuracy")
    ax.set_title(f"Distribution of per-recording accuracy ({best})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "per_mouse_distribution.png"), dpi=150)
    plt.close(fig)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), STATES); ax.set_yticks(range(3), STATES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Held-out test set ({best})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.2f}\n{cm[i,j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # top coefficients
    fig, ax = plt.subplots(figsize=(9, 7))
    top = coef.abs().max(axis=1).sort_values(ascending=False).head(22).index[::-1]
    yv = np.arange(len(top))
    for k, s in enumerate(coef.columns):
        ax.barh(yv + (k - 1) * 0.27, coef.loc[top, s], 0.27, label=s,
                color=STATE_COLORS.get(s))
    ax.set_yticks(yv, top, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardised coefficient")
    ax.set_title(f"{best}: strongest features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "coefficients.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
