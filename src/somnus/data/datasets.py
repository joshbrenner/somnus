"""Assemble the cross-dataset training set, and provide a test-set iterator.

DESIGN
------
Training set = ALL usable recordings from the local corpus + a lab-stratified
sample from the public BIDS corpus, matched PER LAB: the local corpus is
treated as one lab among many, and every lab (each public lab, and the local
one) contributes the same per-state share -- the share being what the local
corpus has labelled. With N public labs the training set therefore holds
(N+1) x local_count epochs of each state, so no single lab dominates any
class, and minority states (REM) get N+1 times the epochs a corpus-level
match would allow. Natural state proportions are preserved (Wake >> REM); the
classifier handles that with balanced class weights rather than by discarding
data. `--rem-per-lab` can raise the REM share beyond what the local corpus
supplies, for feeding the minority class from REM-rich public labs.

Public labs start from ONE MOUSE PER LAB (recruiting more only to fill the
share), so training sees every filtering regime and mains condition in the
corpus -- including a lab whose EEG is lowpass-filtered at ~25 Hz and
therefore exercises the missing-tier path.

Every remaining public-corpus subject is reserved for test. Test features are never
materialised into one giant table: `iter_test_recordings()` yields one recording
at a time so evaluation streams (the full corpus would otherwise be ~10 GB).

All source data is opened READ-ONLY.

Usage:
    python -m somnus.data.datasets                 # build training set
    python -m somnus.data.datasets --seed 1        # different draw
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd
import mne

# ---- feature backend -------------------------------------------------------
# The shipped implementation is the openseize port (`openseize_backend`), which
# is what the released model must be trained with. `scipy_backend` is the
# original scipy implementation, kept as a cross-check: the two are verified
# numerically identical by `tools/verify_openseize_port.py`, so either
# reproduces the same model. Override with SOMNUS_FEATURE_BACKEND=scipy|openseize.
from somnus.features import get_backend

H = get_backend()
FEATURE_BACKEND = H.__name__

mne.set_log_level("ERROR")

HERE = os.path.dirname(os.path.abspath(__file__))

# Training matrices and manifests live in a training_data/ folder so there is
# exactly one canonical copy. Resolution order: SOMNUS_DATA_DIR, the repository
# checkout's training_data/ (when running from a git clone), else the current
# working directory (for installed-package use).
_REPO_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                          "training_data"))
DATA_DIR = os.environ.get("SOMNUS_DATA_DIR") or \
    (_REPO_DATA if os.path.isdir(_REPO_DATA) else os.getcwd())

# Source-recording locations for the two training corpora. Deliberately no
# built-in defaults: point these at your local copies to reproduce training.
#   SOMNUS_LOCAL_DIR -> flat folder of EDFs with <base>_scored.csv one-hot
#                       scoring and optional tracking (+ timestamps) files
#   SOMNUS_BIDS_DIR  -> a BIDS-layout corpus with *_events.tsv stage scoring
# When unset, discover_local()/discover_bids() simply return no recordings.
LOCAL_DIR = os.environ.get("SOMNUS_LOCAL_DIR", "")
BIDS_DIR = os.environ.get("SOMNUS_BIDS_DIR", "")

# Subjects to exclude from either corpus (comma-separated env vars), e.g. for
# recordings known to be corrupted or modified.
EXCLUDE_LOCAL_SUBJECTS = set(filter(None, os.environ.get(
    "SOMNUS_EXCLUDE_LOCAL", "").split(",")))
EXCLUDE_BIDS_SUBJECTS = set(filter(None, os.environ.get(
    "SOMNUS_EXCLUDE_BIDS", "").split(",")))
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
        # route through the active feature backend so the whole pipeline uses
        # one PSD implementation (scipy or openseize) rather than mixing them
        f, p = H._welch(x, sfreq, int(2 * sfreq))
        snr = p[(f >= 0.5) & (f <= 30)].sum() / (p[f > 30].sum() + 1e-12)
        if snr > best_snr:
            best, best_snr = i, snr
    return best


# ------------------------------------------------------------------ discovery
def discover_local() -> list[dict]:
    out = []
    if not LOCAL_DIR or not os.path.isdir(LOCAL_DIR):
        return out
    for edf in sorted(glob.glob(os.path.join(LOCAL_DIR, "*.edf"))):
        base = os.path.basename(edf)[:-4]
        mouse = base.split("_")[0]
        if mouse in EXCLUDE_LOCAL_SUBJECTS:
            continue
        scored = os.path.join(LOCAL_DIR, base + "_scored.csv")
        if not os.path.exists(scored):
            alt = os.path.join(LOCAL_DIR, base + "_scored_man.csv")
            if not os.path.exists(alt):
                continue
            scored = alt
        pkl = sorted(glob.glob(os.path.join(LOCAL_DIR, base + "*_coordinates.pkl")))
        out.append({"dataset": "local", "recording": base, "subject": mouse,
                    "group": "local", "edf": edf, "scored": scored,
                    "pkl": pkl[0] if pkl else None})
    return out


def discover_bids() -> list[dict]:
    labs = {}
    if not BIDS_DIR or not os.path.isdir(BIDS_DIR):
        return []
    with open(os.path.join(BIDS_DIR, "participants.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            labs[row["participant_id"]] = row.get("lab", "?")
    out = []
    for edf in sorted(glob.glob(os.path.join(BIDS_DIR, "sub-*/eeg/*_eeg.edf"))):
        b = os.path.basename(edf)
        subject = b.split("_")[0]
        if subject in EXCLUDE_BIDS_SUBJECTS:
            continue
        events = edf.replace("_eeg.edf", "_events.tsv")
        chans = glob.glob(os.path.join(os.path.dirname(edf), "*_channels.tsv"))
        if not os.path.exists(events) or not chans:
            continue
        run = next((p for p in b.split("_") if p.startswith("run-")), "run-1")
        out.append({"dataset": "bids", "recording": f"{subject}_{run}",
                    "subject": subject, "group": labs.get(subject, "?"),
                    "edf": edf, "events": events, "channels": chans[0]})
    return out


# ------------------------------------------------------------- featurisation
def featurize(entry: dict, probe_seconds: float = 900.0) -> pd.DataFrame:
    """Load one recording and return its per-epoch feature table.

    Bandwidth is measured from the data itself (separately for EEG and EMG), and
    unsupported tiers are emitted as NaN with their indicator set to 0.
    """
    raw = mne.io.read_raw_edf(entry["edf"], preload=False)
    sfreq = float(raw.info["sfreq"])
    names = raw.ch_names

    if entry["dataset"] in ("local", "user"):   # same on-disk format
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

    # --- velocity (only the local corpus has video/tracking) ---
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
    elif entry["dataset"] in ("local", "user"):   # same on-disk format
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


# --------------------------------------------------------------- assembly
def build_training(seed: int = 0, rem_per_lab: int | None = None) -> None:
    rng = np.random.RandomState(seed)

    local = discover_local()
    print(f"Local recordings: {len(local)}")
    local_tables, metas = [], []
    for e in local:
        df = featurize(e)
        metas.append(df.attrs["meta"])
        local_tables.append(df)
        m = df.attrs["meta"]
        print(f"  {m['recording']}: tiers={m['tiers']} emg={m['emg_bands']} "
              f"video={m['velocity_source']}")
    local_all = pd.concat(local_tables, ignore_index=True)
    local_lab = local_all[local_all["state"].notna()]
    target = {s: int((local_lab["state"] == s).sum()) for s in STATES}
    print(f"\nLocal labelled epochs (the per-lab share): {target}")

    # ---- lab-stratified public-corpus training pool ----
    # The local corpus counts as one lab; EVERY lab contributes the same
    # per-state share (what the local corpus has labelled), so with N public
    # labs the training set holds (N+1) x share epochs of each state. Labs
    # whose recordings are too short to cover the share recruit additional
    # mice from the SAME lab until it is met. Every recruited mouse is then
    # excluded from the test set.
    on = discover_bids()
    by_lab: dict[str, list[str]] = {}
    for e in on:
        by_lab.setdefault(e["group"], []).append(e["subject"])
    labs_sorted = sorted(by_lab)
    per_lab_target = dict(target)
    if rem_per_lab is not None:
        per_lab_target["REM"] = int(rem_per_lab)
    print(f"\nPer-lab share target (each of {len(labs_sorted)} public labs): "
          f"{per_lab_target}")

    train_subjects: list[str] = []
    picks: list[pd.DataFrame] = []
    leftovers: list[pd.DataFrame] = []          # unused epochs, for top-up
    lab_report = {}

    for lab in labs_sorted:
        subs = sorted(set(by_lab[lab]))
        order = list(rng.permutation(subs))
        got = {s: 0 for s in STATES}
        used = []
        for sub in order:
            if all(got[s] >= per_lab_target[s] for s in STATES):
                break
            sub_tables = []
            for e in [x for x in on if x["subject"] == sub]:
                df = featurize(e)
                metas.append(df.attrs["meta"])
                sub_tables.append(df)
                m = df.attrs["meta"]
                print(f"  {m['recording']} ({m['group']}): tiers={m['tiers']} "
                      f"emg={m['emg_bands']} eeg_edge={m['eeg_edge_hz']}Hz")
            if not sub_tables:
                continue
            sdf = pd.concat(sub_tables, ignore_index=True)
            sdf = sdf[sdf["state"].notna()]
            used.append(str(sub))
            train_subjects.append(str(sub))
            for s in STATES:
                need = per_lab_target[s] - got[s]
                pool = sdf[sdf["state"] == s]
                if len(pool) == 0:
                    continue
                take = int(min(max(need, 0), len(pool)))
                sel = rng.choice(len(pool), take, replace=False) if take else []
                if take:
                    picks.append(pool.iloc[sel])
                    got[s] += take
                rest = pool.drop(pool.index[sel]) if take else pool
                if len(rest):
                    leftovers.append(rest)
        lab_report[lab] = {"mice_used": used, "drawn": got}
        short = {s: per_lab_target[s] - got[s] for s in STATES
                 if got[s] < per_lab_target[s]}
        msg = f"  [{lab}] mice={used} drawn={got}"
        if short:
            msg += f"  STILL SHORT {short} (lab exhausted)"
        print(msg)

    on_matched = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    # ---- exact-target reconciliation ----
    # The public-corpus total per state is per_lab_target x n_labs. A lab that
    # ran out of epochs is topped up from other labs' unused epochs, trading a
    # little lab-balance for the class count.
    public_target = {s: per_lab_target[s] * len(labs_sorted) for s in STATES}
    spare = pd.concat(leftovers, ignore_index=True) if leftovers else pd.DataFrame()
    final = []
    for s in STATES:
        cur = on_matched[on_matched["state"] == s]
        need = public_target[s] - len(cur)
        if need > 0 and len(spare):
            pool = spare[spare["state"] == s]
            take = int(min(need, len(pool)))
            if take:
                cur = pd.concat([cur, pool.iloc[rng.choice(len(pool), take,
                                                           replace=False)]])
                print(f"  topped up {s} with {take} epochs from already-used mice")
        if len(cur) > public_target[s]:
            cur = cur.iloc[rng.choice(len(cur), public_target[s], replace=False)]
        if len(cur) < public_target[s]:
            print(f"  WARNING: {s} still short "
                  f"({len(cur)}/{public_target[s]}) after exhausting the pool")
        final.append(cur)
        print(f"  matched {s}: {len(cur)}/{public_target[s]} "
              f"from {cur['subject'].nunique()} mice")
    on_matched = pd.concat(final, ignore_index=True)

    train = pd.concat([local_lab, on_matched], ignore_index=True)
    out_csv = os.path.join(DATA_DIR, f"train_generalized_seed{seed}.csv.gz")
    train.to_csv(out_csv, index=False, compression="gzip")

    # Every mouse opened for training is excluded from the test set, including
    # those recruited only to top a lab up.
    test_subjects = sorted({e["subject"] for e in on
                            if e["subject"] not in set(train_subjects)})
    manifest = {
        "seed": seed,
        "matching": "per-lab: local corpus counts as one lab; every lab "
                    "contributes the same per-state share",
        "local_recordings": [e["recording"] for e in local],
        "bids_train_subjects": sorted(set(train_subjects)),
        "bids_test_subjects": test_subjects,
        "n_bids_test_subjects": len(test_subjects),
        "per_lab_share_target": per_lab_target,
        "public_target": public_target,
        "per_lab_report": lab_report,
        "local_share": target,
        "train_counts_by_dataset": {
            d: {s: int(((train["dataset"] == d) & (train["state"] == s)).sum())
                for s in STATES} for d in ("local", "bids")},
        "model_columns": model_columns(train),
        "recording_meta": metas,
    }
    with open(os.path.join(DATA_DIR, f"manifest_generalized_seed{seed}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nTraining set: {len(train)} epochs -> {out_csv}")
    print(pd.crosstab(train["dataset"], train["state"]).to_string())
    print(f"Model feature columns: {len(manifest['model_columns'])}")
    print(f"Held-out test subjects: {len(test_subjects)}")


def iter_test_recordings(train_subjects: set[str]):
    """Yield (meta, featurized DataFrame) for each held-out public-corpus recording.

    Streams so the full test corpus never has to be held in memory at once.
    """
    for e in discover_bids():
        if e["subject"] in train_subjects:
            continue
        df = featurize(e)
        yield df.attrs["meta"], df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cross-dataset training set.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rem-per-lab", type=int, default=None,
                    help="REM epochs each lab contributes (default: as many "
                         "as the local corpus has labelled)")
    args = ap.parse_args()
    build_training(args.seed, rem_per_lab=args.rem_per_lab)


if __name__ == "__main__":
    main()
