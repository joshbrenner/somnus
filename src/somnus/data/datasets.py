"""Load one recording and turn it into the per-epoch feature table.

`featurize()` is the entry point: give it an EDF (plus, optionally, a scoring
file and video tracking) and it returns the table `somnus.predict.predict()`
consumes. Bandwidth is measured from the data itself -- separately for EEG and
EMG -- and tiers the recording cannot support are emitted as NaN with their
availability indicator set to 0, so one model scores recordings of very
different quality without retraining.

An `entry` is a plain dict:

    {"recording": "myrec",            # name used in the output table
     "edf":       "path/to/myrec.edf",
     "dataset":   "user",             # "user" | "bids"
     "group":     "user",             # free-form grouping label
     "subject":   "m1",
     "scored":    None,               # optional one-hot scoring CSV
     "pkl":       None}               # optional video tracking coordinates

For `dataset="user"` the first up-to-three EDF channels are treated as EEG and
the last as EMG. For `dataset="bids"` add `"channels"` (a BIDS `channels.tsv`,
used to identify EEG/EMG) and optionally `"events"` (a stage-scored
`events.tsv`).

Labels are OPTIONAL throughout: scoring an unlabelled recording is the normal
case. All source data is opened READ-ONLY.
"""
from __future__ import annotations

import csv
import glob
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd
import mne

# Feature computation: band definitions, harmonisation tiers, PSD.
from somnus import features as H

mne.set_log_level("ERROR")

BIDS_STAGE_MAP = {"1": "Wake", "2": "NREM", "3": "REM"}  # 4 = Artifact -> None
STATES = ["Wake", "NREM", "REM"]


# ---- model feature layout ---------------------------------------------------
# NOTE: `delta_index` is deliberately NOT a model input. By construction
#   delta_index = (delta - (t1 - delta)) / t1 = 2 * delta_rel - 1,
# an exact affine transform of delta_rel (verified to 4.4e-16). For a linear
# model with an intercept the two are linearly dependent, and after
# within-recording z-scoring they become the *identical* column -- so including
# both duplicated 5 columns (the feature and its 4 rolling-context variants) and
# left the design matrix rank-deficient (condition number 6.1e15). L-BFGS then
# stopped anywhere along the resulting flat ridge, so fitted coefficients were
# not reproducible (they moved by 0.08 between runs whose features differed only
# at 1e-10) even though predictions barely changed (0.06% of labels).
# Dropping it takes the condition number to 1.0e2 with no loss of information.
# It is still computed and stored in the epoch table for plotting and continuity
# with SleepScorer.py -- it is simply not fed to the classifier.
UNIVERSAL = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel",
             "theta_delta_log", "t1_power_log_z", "emg_low_log_z"]
# optional block -> (columns, indicator name)
OPTIONAL = {
    "tier2":    (["gamma1_ratio_log"], "has_tier2"),
    "tier3":    (["gamma2_ratio_log"], "has_tier3"),
    "tier4":    (["gamma3_ratio_log"], "has_tier4"),
    "emg_mid":  (["emg_mid_log_z", "emg_ratio_hi_lo"], "has_emg_mid"),
    "emg_high": (["emg_high_log_z"], "has_emg_high"),
    "video":    (["log_velocity_z"], "has_video"),
}
CONTEXT_WINDOWS = (3, 15)


# ------------------------------------------------------------ pickle loading
def _pathlib_shim() -> None:
    if "pathlib._local" in sys.modules:
        return
    import pathlib
    m = types.ModuleType("pathlib._local")
    for n in ("Path", "PosixPath", "WindowsPath", "PurePath",
              "PurePosixPath", "PureWindowsPath"):
        setattr(m, n, getattr(pathlib, n))
    sys.modules["pathlib._local"] = m


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type("_M", (), {"__init__": lambda s, *a, **k: None,
                                   "__setstate__": lambda s, st: None})


def load_coordinates(path: str) -> tuple[np.ndarray, dict]:
    """MouseFinder pickle; handles dict and bare-array layouts."""
    _pathlib_shim()
    with open(path, "rb") as fh:
        obj = _SafeUnpickler(fh).load()
    if isinstance(obj, dict):
        return (np.asarray(obj["coordinates"], float),
                {k: v for k, v in obj.items() if k != "coordinates"})
    return np.asarray(obj, float), {}


def resolve_frame_times(base: str, n_frames: int, duration: float,
                        meta: dict, search_dir: str) -> tuple[np.ndarray, str]:
    """True per-frame times for a coordinate array. Raises if unavailable.

    Frames were dropped during acquisition, so timing is irregular; assuming a
    constant fps misplaces frames by minutes. A timestamp file is only accepted
    if its length matches the coordinate array EXACTLY, or if the tracking
    metadata declares a constant rate consistent with the recording duration
    (the `*_cropped_ds` re-encodes).

    `search_dir` is searched and nothing else. No parent/sibling lookup: a stale
    timestamps file from another folder would pair positions with the wrong times
    and corrupt velocity silently. If the coordinates are here but their
    timestamps are not, that is an incomplete dataset and an error, not something
    to work around.
    """
    cands = sorted(glob.glob(os.path.join(search_dir, base + "*timestamps.npy")))
    lengths = []
    for cand in cands:
        t = np.load(cand, allow_pickle=True).ravel().astype(float)
        if len(t) == n_frames:
            return t, os.path.basename(cand)
        lengths.append((os.path.basename(cand), len(t)))

    sr = meta.get("sample_rate")
    if sr and duration and abs(n_frames / duration - float(sr)) < 0.05 * float(sr):
        return np.arange(n_frames, dtype=float) / float(sr), f"constant {sr:g} fps"

    if lengths:
        detail = ", ".join(f"{n} has {k} frames" for n, k in lengths)
        raise FileNotFoundError(
            f"{base}: tracking has {n_frames} frames but no timestamps file in "
            f"{search_dir} matches that length ({detail}). Frame timing cannot be "
            f"trusted, so velocity is refused rather than guessed.")
    raise FileNotFoundError(
        f"{base}: found tracking coordinates but no '*timestamps.npy' in "
        f"{search_dir}. Put the matching timestamps file alongside the "
        f"coordinates, or remove the coordinates to score without velocity.")


# ------------------------------------------------------------- EEG selection
def pick_best_eeg(raw, eeg_idx: list[int], sfreq: float) -> int:
    """Highest 0.5-30 / >30 Hz power ratio over the first 5 min (as in
    SleepScorer.py's channel chooser)."""
    if len(eeg_idx) == 1:
        return eeg_idx[0]
    stop = min(int(300 * sfreq), raw.n_times)
    best, best_snr = eeg_idx[0], -np.inf
    for i in eeg_idx:
        x = raw.get_data(picks=[i], start=0, stop=stop)[0]
        # route through the feature module so the whole pipeline shares one
        # PSD implementation
        f, p = H._welch(x, sfreq, int(2 * sfreq))
        snr = p[(f >= 0.5) & (f <= 30)].sum() / (p[f > 30].sum() + 1e-12)
        if snr > best_snr:
            best, best_snr = i, snr
    return best


# ------------------------------------------------------------- featurisation
def featurize(entry: dict, probe_seconds: float = 900.0) -> pd.DataFrame:
    """Load one recording and return its per-epoch feature table.

    Bandwidth is measured from the data itself (separately for EEG and EMG), and
    unsupported tiers are emitted as NaN with their indicator set to 0.
    """
    raw = mne.io.read_raw_edf(entry["edf"], preload=False)
    sfreq = float(raw.info["sfreq"])
    names = raw.ch_names

    if entry["dataset"] != "bids":       # plain EDF: EEG first, EMG last
        eeg_idx = list(range(min(3, len(names))))
        emg_i = len(names) - 1
    else:
        types_map = {}
        with open(entry["channels"]) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                types_map[row["name"]] = row.get("type", "").upper()
        eeg_names = [c for c in names if types_map.get(c) == "EEG"] or names[:-1]
        emg_names = [c for c in names if types_map.get(c) == "EMG"] or [names[-1]]
        eeg_idx = [names.index(c) for c in eeg_names]
        emg_i = names.index(emg_names[0])

    best = pick_best_eeg(raw, eeg_idx, sfreq)
    eeg = raw.get_data(picks=[best])[0]
    emg = raw.get_data(picks=[emg_i])[0]
    n_ep = H.n_epochs_for(raw.n_times, sfreq)

    probe = int(min(probe_seconds * sfreq, len(eeg)))
    eeg_edge = H.detect_bandwidth(eeg[:probe], sfreq)
    emg_edge = H.detect_bandwidth(emg[:probe], sfreq)
    tiers = H.available_tiers(eeg_edge)
    emg_bands = H.available_emg_bands(emg_edge)

    df = pd.concat([H.eeg_features(eeg, sfreq, n_ep, tiers),
                    H.emg_features(emg, sfreq, n_ep, emg_bands)], axis=1)

    # --- velocity (only if the recording has video tracking) ---
    vel_src = "none"
    if entry.get("pkl"):
        coords, meta = load_coordinates(entry["pkl"])
        # search only the folder the coordinates came from
        t, vel_src = resolve_frame_times(entry["recording"], len(coords),
                                         raw.n_times / sfreq, meta,
                                         os.path.dirname(entry["pkl"]))
        df = pd.concat([df, H.velocity_features(coords, t, n_ep)], axis=1)
    else:
        df = pd.concat([df, H.velocity_features(None, None, n_ep)], axis=1)

    # --- labels ---
    # Labels are OPTIONAL: inference runs on unscored recordings, which is the
    # normal case for a user scoring new data. An absent scoring file yields an
    # all-None state column rather than an error.
    if not entry.get("scored") and not entry.get("events"):
        labels = np.full(n_ep, None, dtype=object)
    elif entry["dataset"] != "bids":     # one-hot scoring CSV
        labels = H.labels_from_onehot(pd.read_csv(entry["scored"]), n_ep)
    else:
        ev = pd.read_csv(entry["events"], sep="\t")
        labels = H.labels_from_stage_events(ev["onset"].values,
                                            ev["stage"].values, n_ep,
                                            BIDS_STAGE_MAP)

    df.insert(0, "epoch", np.arange(n_ep))
    df.insert(1, "t_start", H.epoch_times(n_ep))
    df.insert(2, "recording", entry["recording"])
    df.insert(3, "subject", entry["subject"])
    df.insert(4, "dataset", entry["dataset"])
    df.insert(5, "group", entry["group"])
    df["state"] = labels[:n_ep]

    # temporal context on the raw (pre-z-score) signal features
    base_cols = [c for c in df.columns
                 if c not in ("epoch", "t_start", "recording", "subject",
                              "dataset", "group", "state")]
    df = H.add_temporal_context(df, base_cols, windows=CONTEXT_WINDOWS)

    # Within-recording z-scoring. Applied to amplitude AND ratio features: both
    # were measured to carry large per-site offsets, and z-scoring is computed
    # here on the recording's FULL epoch set (before any sampling) so the
    # statistics reflect the recording's natural state mix.
    df = H.zscore_within(df, H.zscore_target_columns(df), group_col="recording")

    # availability indicators
    df["has_tier2"] = int(2 in tiers)
    df["has_tier3"] = int(3 in tiers)
    df["has_tier4"] = int(4 in tiers)
    df["has_emg_mid"] = int("emg_mid" in emg_bands)
    df["has_emg_high"] = int("emg_high" in emg_bands)
    df["has_video"] = int(df["log_velocity"].notna().any())

    df.attrs["meta"] = {
        "recording": entry["recording"], "subject": entry["subject"],
        "dataset": entry["dataset"], "group": entry["group"],
        "sfreq": sfreq, "eeg_edge_hz": round(eeg_edge, 1),
        "emg_edge_hz": round(emg_edge, 1), "tiers": sorted(tiers),
        "emg_bands": sorted(emg_bands), "velocity_source": vel_src,
        "eeg_channel": names[best], "emg_channel": names[emg_i],
        "n_epochs": int(n_ep),
    }
    return df


def model_columns(df: pd.DataFrame, zscored: bool = False) -> list[str]:
    """Feature columns the model consumes, including context and indicators.

    zscored=True swaps every feature for its within-recording z-scored variant
    (site-offset removed), which is the fix for the cross-lab fingerprinting
    problem measured in the first evaluation.
    """
    base = list(UNIVERSAL)
    for cols, _ in OPTIONAL.values():
        base += cols
    if zscored:
        base = [c if c.endswith("_z") else f"{c}_z" for c in base]
    ctx = []
    for c in base:
        root = c[:-2] if c.endswith("_z") else c
        suf = "_z" if c.endswith("_z") else ""
        for w in CONTEXT_WINDOWS:
            ctx += [f"{root}_mean{w}{suf}", f"{root}_std{w}{suf}"]
    inds = [ind for _, ind in OPTIONAL.values()]
    wanted = base + ctx + inds
    return [c for c in wanted if c in df.columns]


