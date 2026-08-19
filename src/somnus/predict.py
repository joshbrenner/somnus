"""Standalone scorer for the released Somnus model -- numpy/pandas only.

Unlike `somnus.train.export`, this module imports nothing from the training
code: given the JSON artifact and a featurised epoch table, it reproduces the
published predictions. This is the module the GUI and any batch-scoring tool
should call.

The feature table must be produced by `somnus.data.datasets.featurize()`, so
that column names and units match the artifact. That module selects its PSD
backend via `SOMNUS_FEATURE_BACKEND` (default `openseize_backend`, the
implementation the released model was trained with; `scipy_backend` is the
scipy equivalent, verified numerically identical by
`tools/verify_openseize_port.py`). The artifact records which one produced it
in its `feature_backend` field.

Usage as a library:
    from somnus import load_model, predict
    art = load_model()                 # the packaged v1.0 artifact
    labels, proba = predict(art, feature_df)

Usage as a CLI (scores one recording from either configured dataset and, if the
recording carries manual labels, reports agreement):
    python -m somnus.predict --score <recording_id>
    python -m somnus.predict --score <recording_id> --out scored.csv
"""
from __future__ import annotations

import argparse
import json
from importlib.resources import files as _resource_files

import numpy as np
import pandas as pd

# The released weights ship inside the package; wheels install unpacked, so
# this resolves to a real file path usable as a CLI default and GUI display.
DEFAULT_ARTIFACT = str(_resource_files("somnus.models")
                       .joinpath("model_somnus_1.0.json"))


def load_model(path: str | None = None) -> dict:
    """Load a portable JSON artifact; None loads the packaged released model."""
    with open(path or DEFAULT_ARTIFACT) as fh:
        art = json.load(fh)
    if art.get("format_version") != 1:
        raise ValueError(f"unsupported format_version {art.get('format_version')}")
    return art


def design_matrix(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Centre/scale each feature where it is available, zero where it is not.

    Mirrors the training-time AvailabilityScaler: a feature that is absent (or
    non-finite, or whose guard column says its tier was unavailable in this
    recording) is set to exactly 0 *after* centring, so it contributes nothing
    to the logit rather than biasing it.
    """
    cols = art["columns"]
    X = np.zeros((len(df), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        g = art["guards"].get(c)
        if g and g in df.columns:
            ok &= df[g].to_numpy(dtype=float) > 0.5
        z = (v - art["center"][j]) / art["scale"][j]
        X[ok, j] = np.nan_to_num(z[ok], nan=0.0, posinf=0.0, neginf=0.0)
    return X


def probabilities(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Per-epoch class probabilities, columns ordered as art['states']."""
    X = design_matrix(art, df)
    logits = X @ np.asarray(art["coef"]).T + np.asarray(art["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    order = [art["classes"].index(s) for s in art["states"]]
    return p[:, order]


def viterbi(log_em: np.ndarray, A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Maximum-likelihood state path (same implementation as training)."""
    n, k = log_em.shape
    logA = np.log(np.clip(A, 1e-300, None))
    d = np.full((n, k), -np.inf)
    psi = np.zeros((n, k), dtype=int)
    d[0] = log_pi + log_em[0]
    for t in range(1, n):
        sc = d[t - 1][:, None] + logA
        psi[t] = np.argmax(sc, axis=0)
        d[t] = sc[psi[t], np.arange(k)] + log_em[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def scale_transitions(A: np.ndarray, stickiness: float = 1.0) -> np.ndarray:
    """Raise the transition matrix to a power and renormalise rows.

    `stickiness` is a single knob for how hard the decode resists state changes,
    because taking A**g multiplies every transition log-cost by g:

        g = 0    all transitions equally likely -> no temporal inertia at all,
                 so the decode collapses to the per-epoch argmax (up to the
                 state prior). Equivalent to turning smoothing off.
        g = 1    the matrix as estimated from scored data (default).
        g > 1    off-diagonal costs amplified -> longer, cleaner bouts, at the
                 risk of swallowing genuine brief events (short REM bouts are
                 the first casualty).
        0<g<1    weaker inertia than the data suggests; more flicker.

    Rows are renormalised, so the result is always a valid transition matrix.
    """
    if stickiness < 0:
        raise ValueError("stickiness must be >= 0")
    A = np.asarray(A, dtype=float)
    out = np.power(np.clip(A, 1e-300, None), float(stickiness))
    return out / out.sum(axis=1, keepdims=True)


def predict(art: dict, df: pd.DataFrame, decode: bool = True,
            stickiness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Score a featurised recording.

    Args:
        art: artifact from load_model().
        df: per-epoch feature table (rows must be in time order).
        decode: apply the HMM/Viterbi temporal decode. False returns the raw
            per-epoch (memoryless) argmax -- useful for seeing what the model
            believes before any smoothing, and for diagnosing whether an error is
            the classifier's or the decode's.
        stickiness: resistance to state changes, see scale_transitions().
            1.0 uses the transition matrix as estimated; 0 removes inertia
            entirely; >1 enforces longer bouts.

    Returns:
        (labels, proba) -- labels is an array of state strings. `proba` is always
        the raw per-epoch probability, unaffected by the decode, so callers can
        display model confidence alongside a smoothed label.
    """
    p = probabilities(art, df)
    states = np.array(art["states"])
    if not decode:
        return states[p.argmax(axis=1)], p
    A = scale_transitions(np.asarray(art["transition_matrix"]), stickiness)
    path = viterbi(np.log(np.clip(p, 1e-12, None)), A,
                   np.asarray(art["log_prior"]))
    return states[path], p


def to_scored_csv(labels: np.ndarray, t_start: np.ndarray,
                  epoch_sec: float = 4.0) -> pd.DataFrame:
    """Convert predictions to the one-hot layout used by the Somnus scorer UI."""
    return pd.DataFrame({
        "Time_sec": t_start,
        "Wake": (labels == "Wake").astype(int),
        "NREM": (labels == "NREM").astype(int),
        "REM": (labels == "REM").astype(int),
        "Artifact": 0,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_ARTIFACT)
    ap.add_argument("--score", required=True, help="recording id to score")
    ap.add_argument("--out", default=None, help="write predictions to this CSV")
    ap.add_argument("--no-decode", action="store_true",
                    help="skip the HMM decode (raw per-epoch argmax only)")
    ap.add_argument("--stickiness", type=float, default=1.0,
                    help="resistance to state changes: 0 = none (same as "
                         "--no-decode), 1 = as estimated from data, >1 = longer "
                         "bouts")
    args = ap.parse_args()

    # Featurising needs the dataset adapters; import lazily so that library
    # use of this module stays dependency-light (numpy/pandas only).
    from somnus.data import datasets as B

    art = load_model(args.model)
    entries = B.discover_local() + B.discover_bids()
    match = [e for e in entries if e["recording"] == args.score]
    if not match:
        print(f"No recording '{args.score}'. First few available: "
              f"{[e['recording'] for e in entries[:5]]}")
        return

    df = B.featurize(match[0])
    labels, p = predict(art, df, decode=not args.no_decode,
                        stickiness=args.stickiness)
    print(f"Scored {args.score}: {len(df)} epochs")
    print("  predicted:", {s: int((labels == s).sum()) for s in art["states"]})

    if "state" in df.columns:
        m = df["state"].isin(art["states"]).to_numpy()
        if m.sum():
            agree = float((labels[m] == df.loc[m, "state"].to_numpy()).mean())
            print(f"  {int(m.sum())} manually labelled epochs, "
                  f"agreement = {agree:.4f}")

    if args.out:
        to_scored_csv(labels, df["t_start"].to_numpy(),
                      art["epoch_sec"]).to_csv(args.out, index=False)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
