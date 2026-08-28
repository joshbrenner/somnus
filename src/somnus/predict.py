"""Score a recording: assign Wake, NREM or REM to each epoch.
Where the actual scoring happens. Give it the trained
model and a table of features (one row per epoch, produced by
`somnus.data.datasets.featurize`) and it returns a sleep state for every row,
along with how confident it was.

Scoring happens in two passes. First each epoch is judged from its features,
then the sequence is smoothed over time: a lone epoch of REM in the middle of Wake 
is far more likely to be a mistake than a real event.

Using it from Python:
    from somnus import load_model
    from somnus.predict import predict
    art = load_model()                 # the model that ships with Somnus
    labels, proba = predict(art, feature_df)

Using it from the command line:
    python -m somnus.predict --score myrec.edf --eeg 0 1 2 --emg 3
    python -m somnus.predict --score myrec.edf --eeg 0 1 2 --emg 3 --out scored.csv
"""
from __future__ import annotations

import argparse
import json
import os
from importlib.resources import files as _resource_files

import numpy as np
import pandas as pd

DEFAULT_ARTIFACT = str(_resource_files("somnus.models")
                       .joinpath("model_somnus_1.0.json"))


def load_model(path: str | None = None) -> dict:
    """Load a trained model from a JSON file.

    With no path, loads the model that ships with Somnus.
    """
    with open(path or DEFAULT_ARTIFACT) as fh:
        art = json.load(fh)
    if art.get("format_version") != 1:
        raise ValueError(f"unsupported format_version {art.get('format_version')}")
    return art


def design_matrix(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Lay the recording's features out in the order the model expects them.

    Each feature is put on a common scale, so that recordings made on different
    equipment can be compared. Anything this recording could not measure is left
    at zero, which means it neither pushes the answer one way nor the other.
    That covers a feature that is missing outright, a gap in the data, and a
    frequency band the recording equipment never reached.

    This is what lets one model score both a 128 Hz EEG-only recording and a
    5 kHz recording with EMG and video, without retraining for either.
    """
    cols = art["columns"]
    X = np.zeros((len(df), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        # Each feature may have a companion column recording whether this
        # recording could measure it at all. Zero there means "not measured".
        g = art["guards"].get(c)
        if g and g in df.columns:
            ok &= df[g].to_numpy(dtype=float) > 0.5
        z = (v - art["center"][j]) / art["scale"][j]
        X[ok, j] = np.nan_to_num(z[ok], nan=0.0, posinf=0.0, neginf=0.0)
    return X


def probabilities(art: dict, df: pd.DataFrame) -> np.ndarray:
    """How likely each sleep state is, for every epoch.

    Returns one row per epoch and one column per state, in the order given by
    `art["states"]`. Each row adds up to 1.
    """
    X = design_matrix(art, df)
    logits = X @ np.asarray(art["coef"]).T + np.asarray(art["intercept"])
    # Shift each row down by its largest value before exponentiating. This
    # changes none of the resulting probabilities and keeps the numbers small
    # enough that they cannot overflow.
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    order = [art["classes"].index(s) for s in art["states"]]
    return p[:, order]


def viterbi(log_em: np.ndarray, A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """Choose the single best run of states for the whole recording at once.

    Every transition carries a cost (see scale_transitions): staying in
    the same state is nearly free, while a move like Wake straight into REM
    is essentially impossible. This function finds the sequence with the lowest total 
    cost over the entire recording, so a transition is drawn only where the evidence for 
    it outweighs the cost of making it. Optional in GUI.

    """
    n, k = log_em.shape
    logA = np.log(np.clip(A, 1e-300, None))
    d = np.full((n, k), -np.inf)
    psi = np.zeros((n, k), dtype=int)
    d[0] = log_pi + log_em[0]
    # Sweep forward, recording for each epoch the best way to arrive in each
    # state, and which state it came from.
    for t in range(1, n):
        sc = d[t - 1][:, None] + logA
        psi[t] = np.argmax(sc, axis=0)
        d[t] = sc[psi[t], np.arange(k)] + log_em[t]
    # Then sweep back from the best final state to read off the winning route.
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(d[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def scale_transitions(A: np.ndarray, stickiness: float = 1.0) -> np.ndarray:
    """Adjust how strongly the scoring resists changing state.

    state `stickiness` is controlled on a single dial :

        0     ignore timing altogether; every epoch is scored on its own evidence
        1     use the preloaded transition rates found in training data (default)
        > 1   hold each state longer, giving fewer and longer bouts
        < 1   switch more readily, giving more and shorter bouts

    Turning it up produces a tidier hypnogram, but brief real events get
    absorbed into their neighbours, and short REM bouts are the first to go.
    Turn it down if fragmented sleep is part of what you are studying.
    """
    if stickiness < 0:
        raise ValueError("stickiness must be >= 0")
    A = np.asarray(A, dtype=float)
    out = np.power(np.clip(A, 1e-300, None), float(stickiness))
    # Rescale each row to sum to 1, so the result is still a set of
    # probabilities after the adjustment above.
    return out / out.sum(axis=1, keepdims=True)


def predict(art: dict, df: pd.DataFrame, decode: bool = True,
            stickiness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Score a recording, returning one sleep state per epoch.

    Args:
        art: a model from load_model().
        df: the recording's feature table, rows in time order.
        decode: smooth the states over time, which is the default. Turn it off
            to see what the model makes of each epoch on its own -- useful for
            telling whether a mistake came from the model or from the smoothing.
        stickiness: how strongly to resist changing state, see
            scale_transitions().

    Returns:
        (labels, proba). `labels` is the sleep state for each epoch. `proba` is
        always the unsmoothed confidence for each epoch, so you can show how
        sure the model was alongside the final answer.
    """
    p = probabilities(art, df)
    states = np.array(art["states"])
    if not decode or not len(p):
        return states[p.argmax(axis=1)], p
    A = scale_transitions(np.asarray(art["transition_matrix"]), stickiness)
    path = viterbi(np.log(np.clip(p, 1e-12, None)), A,
                   np.asarray(art["log_prior"]) * float(stickiness))
    return states[path], p


def to_scored_csv(labels: np.ndarray, t_start: np.ndarray) -> pd.DataFrame:
    """Put the scored states into the table layout the Somnus scorer reads.

    There is one column per state and a 1 in whichever column applies to each
    epoch, so the predictions can be opened in the scorer and corrected by hand.
    """
    return pd.DataFrame({
        "Time_sec": t_start,
        "Wake": (labels == "Wake").astype(int),
        "NREM": (labels == "NREM").astype(int),
        "REM": (labels == "REM").astype(int),
        "Artifact": 0,
    })


def main() -> None:
    """Score one recording from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_ARTIFACT)
    ap.add_argument("--score", required=True,
                    help="path to the EDF recording to score")
    ap.add_argument("--scored", default=None,
                    help="optional one-hot scoring CSV, to report agreement")
    ap.add_argument("--coords", default=None,
                    help="optional video tracking coordinates (.pkl)")
    ap.add_argument("--eeg", nargs="+", default=None, metavar="CHAN",
                    type=lambda v: int(v) if v.isdigit() else v,
                    help="the EEG channel(s), as names or numbers; the "
                         "cleanest is used")
    ap.add_argument("--emg", default=None, metavar="CHAN",
                    type=lambda v: int(v) if v.isdigit() else v,
                    help="the EMG channel, as a name or number")
    ap.add_argument("--assume-frame-times", action="store_true",
                    help="allow tracking whose frame times can only be "
                         "assumed, not measured")
    ap.add_argument("--out", default=None, help="write predictions to this CSV")
    ap.add_argument("--no-decode", action="store_true",
                    help="skip the HMM decode (raw per-epoch argmax only)")
    ap.add_argument("--stickiness", type=float, default=1.0,
                    help="resistance to state changes: 0 = none (same as "
                         "--no-decode), 1 = as estimated from data, >1 = longer "
                         "bouts")
    args = ap.parse_args()

    from somnus.data import datasets as B

    art = load_model(args.model)
    if not os.path.exists(args.score):
        raise SystemExit(f"No such recording: {args.score}")
    name = os.path.splitext(os.path.basename(args.score))[0]

    df = B.featurize({"recording": name, "edf": args.score, "dataset": "user",
                      "group": "user", "subject": name.split("_")[0],
                      "scored": args.scored, "pkl": args.coords},
                     eeg_chan=args.eeg, emg_chan=args.emg,
                     assume_frame_times=args.assume_frame_times)
    labels, p = predict(art, df, decode=not args.no_decode,
                        stickiness=args.stickiness)
    print(f"Scored {name}: {len(df)} epochs")
    print("  predicted:", {s: int((labels == s).sum()) for s in art["states"]})

    # If the recording came with manual scoring, say how far the two agree.
    if "state" in df.columns:
        m = df["state"].isin(art["states"]).to_numpy()
        if m.sum():
            agree = float((labels[m] == df.loc[m, "state"].to_numpy()).mean())
            print(f"  {int(m.sum())} manually labeled epochs, "
                  f"agreement = {agree:.4f}")

    if args.out:
        to_scored_csv(labels, df["t_start"].to_numpy()).to_csv(
            args.out, index=False)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
