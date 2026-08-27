"""Load one recording and turn it into the per-epoch feature table.

`featurize()` is the entry point: give it an EDF, plus optionally a scoring file
and video tracking, and it returns the table `somnus.predict.predict()` reads.
What the recording can actually measure is worked out from the data itself, and
anything beyond that is left blank, so one model handles recordings of very
different quality.

An `entry` is a plain dict:

    {"recording": "myrec",            # name used in the output table
     "edf":       "path/to/myrec.edf",
     "dataset":   "user",             # "user" | "bids"
     "group":     "user",             # free-form grouping label
     "subject":   "m1",
     "scored":    None,               # optional scoring CSV
     "pkl":       None}               # optional video tracking coordinates

Name the channels with `eeg_chan` and `emg_chan`, either as channel names or as
numbers. Leave them out and Somnus works out which is which from the data and
says so. EEG and EMG are never the same channel.

For `dataset="bids"` add `"channels"` (a BIDS `channels.tsv`, used to identify
EEG/EMG) and optionally `"events"` (a stage-scored `events.tsv`).

Labels are OPTIONAL throughout: scoring an unlabeled recording is the normal
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

# Feature computation: band definitions, harmonization tiers, PSD.
from somnus import features as H

mne.set_log_level("ERROR")

BIDS_STAGE_MAP = {"1": "Wake", "2": "NREM", "3": "REM"}  # 4 = Artifact -> None
STATES = ["Wake", "NREM", "REM"]


# ---- model feature layout ---------------------------------------------------
# `delta_index` is deliberately NOT given to the model. It is just delta_rel
# rescaled, so the two say exactly the same thing, and feeding both makes the
# fitted weights arbitrary. It is still computed for plotting.
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
    """Let pickles written by older Python versions still load.

    Some tracking files were saved when file paths lived elsewhere in the
    standard library. This puts them back where the pickle expects to find them.
    """
    if "pathlib._local" in sys.modules:
        return
    import pathlib
    m = types.ModuleType("pathlib._local")
    for n in ("Path", "PosixPath", "WindowsPath", "PurePath",
              "PurePosixPath", "PureWindowsPath"):
        setattr(m, n, getattr(pathlib, n))
    sys.modules["pathlib._local"] = m


class _SafeUnpickler(pickle.Unpickler):
    """Reads a tracking pickle even when it mentions software you do not have.

    Files often carry references to the tool that wrote them. Anything that
    cannot be found is replaced by a harmless empty stand-in, so the
    coordinates still load.
    """

    def find_class(self, module, name):
        """Return the requested class, or a placeholder if it is unavailable."""
        try:
            return super().find_class(module, name)
        except Exception:
            return type("_M", (), {"__init__": lambda s, *a, **k: None,
                                   "__setstate__": lambda s, st: None})


CENTROID_BODYPART = "mouse_center"


def _xy_from_table(df: pd.DataFrame, path: str) -> np.ndarray:
    """Pull one x/y position per frame out of a tracking table.

    Understands a DeepLabCut export, which carries many bodyparts, and a plain
    table with `x` and `y` columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        names = list(df.columns.names)
        lvl = names.index("bodyparts") if "bodyparts" in names else -2
        parts = list(dict.fromkeys(df.columns.get_level_values(lvl)))
        if CENTROID_BODYPART in parts:
            want = CENTROID_BODYPART
        elif len(parts) == 1:
            want = parts[0]
        else:
            raise ValueError(
                f"{os.path.basename(path)} tracks {len(parts)} bodyparts and "
                f"none is called {CENTROID_BODYPART!r}, so there is no single "
                f"point to measure movement from. Either export a "
                f"{CENTROID_BODYPART!r} bodypart, or supply a table with one "
                f"x/y pair per frame. Found: {', '.join(map(str, parts))}")
        sub = df.xs(want, axis=1, level=lvl)
        cols = {str(c).lower(): c for c in sub.columns.get_level_values(-1)}
        sub.columns = sub.columns.get_level_values(-1)
        if "x" not in cols or "y" not in cols:
            raise ValueError(f"{os.path.basename(path)}: bodypart {want!r} has "
                             f"no x/y columns")
        return sub[[cols["x"], cols["y"]]].to_numpy(dtype=float)

    cols = {str(c).lower(): c for c in df.columns}
    if "x" in cols and "y" in cols:
        return df[[cols["x"], cols["y"]]].to_numpy(dtype=float)
    if df.shape[1] == 2:
        return df.to_numpy(dtype=float)
    raise ValueError(
        f"{os.path.basename(path)}: could not find the centroid. Supply either "
        f"columns named 'x' and 'y' (any other columns are ignored) or a table "
        f"with exactly two columns, x then y. Found: "
        f"{', '.join(map(str, df.columns))}")


def load_coordinates(path: str) -> tuple[np.ndarray, dict]:
    """Per-frame position of the animal, as (n_frames, 2) x/y and metadata.

    One row per video frame, in frame order. Accepted formats:

    * ``.csv`` -- a DeepLabCut export, or any table with `x` and `y` columns.
    * ``.h5`` / ``.hdf5`` -- a DeepLabCut export (needs the `tables` package).
    * ``.pkl`` / ``.pickle`` -- an (n, 2) array of x/y, or a dict with a
      `coordinates` key holding one; anything else in the dict is returned as
      metadata.

    A DeepLabCut file usually tracks many bodyparts. The one named
    ``mouse_center`` is used; if the file has exactly one bodypart that is used
    instead. Otherwise this raises, rather than guessing which point represents
    the animal.

    Low-likelihood points are NOT filtered here -- a poorly tracked frame is
    kept as-is and will read as real movement. Filter before passing the file
    in, replacing rejected points with NaN (velocity ignores non-finite steps).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".csv", ".h5", ".hdf5"):
        if ext == ".csv":
            with open(path) as fh:
                first = fh.readline().split(",", 1)[0].strip().lower()
            # a DeepLabCut export starts its header block with "scorer"
            hdr = [0, 1, 2] if first == "scorer" else [0]
            df = pd.read_csv(path, header=hdr, index_col=0 if hdr != [0] else None)
        else:
            try:
                df = pd.read_hdf(path)
            except ImportError as e:
                raise ImportError(
                    f"Reading a DeepLabCut .h5 needs the `tables` package "
                    f"(pip install tables). Alternatively point Somnus at the "
                    f"matching .csv export.") from e
        return _xy_from_table(df, path), {}

    _pathlib_shim()
    with open(path, "rb") as fh:
        obj = _SafeUnpickler(fh).load()

    if isinstance(obj, dict):
        if "coordinates" in obj:
            return (np.asarray(obj["coordinates"], float),
                    {k: v for k, v in obj.items() if k != "coordinates"})
        if any(str(k).startswith("frame") for k in obj):
            raise ValueError(
                f"{os.path.basename(path)} looks like a DeepLabCut "
                f"'_full.pickle': raw per-frame detections, before they are "
                f"assembled into tracks. Use the .h5 or .csv export written "
                f"alongside it instead.")
        raise ValueError(f"{os.path.basename(path)}: pickled dict has no "
                         f"'coordinates' key. Keys: "
                         f"{', '.join(map(str, list(obj)[:8]))}")
    return np.asarray(obj, float), {}


def resolve_frame_times(base: str, n_frames: int, duration: float,
                        meta: dict, search_dir: str,
                        fps: float | None = None) -> tuple[np.ndarray, str]:
    """Find the true time of every video frame. Raises if it cannot be trusted.

    Cameras drop frames, so the gaps between them are uneven and assuming a
    fixed rate would misplace positions by minutes. A timestamps file is
    accepted only if it has exactly one entry per tracked frame.

    Only the folder the coordinates came from is searched. Picking up a
    timestamps file from anywhere else risks pairing positions with the wrong
    times, which would corrupt movement silently.

    `fps` is the caller stating that the video really does run at a fixed rate.
    It is used only when no timestamps file exists at all. A file that exists
    but has the wrong number of entries means the coordinates and the times came
    from different videos, and stays an error.
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
        # A timestamps file IS here and disagrees: that is a real mismatch (for
        # instance tracking run on a re-encoded copy of the video), not a
        # missing file, so a declared fps must not paper over it.
        detail = ", ".join(f"{n} has {k} frames" for n, k in lengths)
        raise FileNotFoundError(
            f"{base}: tracking has {n_frames} frames but no timestamps file in "
            f"{search_dir} matches that length ({detail}). Frame timing cannot be "
            f"trusted, so velocity is refused rather than guessed.")

    if fps:
        return np.arange(n_frames, dtype=float) / float(fps), f"declared {fps:g} fps"

    raise FileNotFoundError(
        f"{base}: found tracking coordinates but no '*timestamps.npy' in "
        f"{search_dir}. Frame timing cannot be trusted, so velocity is refused "
        f"rather than guessed. Put the matching timestamps file alongside the "
        f"coordinates, pass fps=<frames per second> to featurize() to accept a "
        f"constant frame rate, or omit the coordinates to score without "
        f"velocity.")


# ------------------------------------------------------------- EEG selection
def _as_index(ch, names: list[str], what: str) -> int:
    """Turn a channel name or number into an index, or say why it cannot."""
    if isinstance(ch, str):
        if ch not in names:
            raise ValueError(f"{what}: no channel named {ch!r} in this "
                             f"recording. It has: {', '.join(names)}")
        return names.index(ch)
    i = int(ch)
    if not 0 <= i < len(names):
        raise ValueError(f"{what}: channel {i} is out of range. This recording "
                         f"has {len(names)} channels (0-{len(names) - 1}).")
    return i


def resolve_channels(raw, eeg_chan, emg_chan) -> tuple[list[int], int]:
    """Work out which channels are EEG and which one is EMG.

    Both must be given, as channel names or numbers. Somnus will not guess: a
    file says nothing reliable about which electrode is which, and picking the
    wrong one costs you the muscle tone that identifies REM, silently. Several
    EEG channels may be named -- the cleanest is chosen from among them -- but
    exactly one EMG, and it can never be one of the EEG channels.
    """
    names = raw.ch_names
    if eeg_chan is None or emg_chan is None:
        raise ValueError(
            f"this recording has {len(names)} channels and Somnus will not "
            f"guess which is which. Say so with eeg_chan and emg_chan, as "
            f"names or numbers -- for example eeg_chan=[0, 1, 2], emg_chan=3. "
            f"The channels are: {', '.join(names)}")

    eeg = [_as_index(c, names, "eeg_chan")
           for c in (eeg_chan if isinstance(eeg_chan, (list, tuple)) else [eeg_chan])]
    emg_list = emg_chan if isinstance(emg_chan, (list, tuple)) else [emg_chan]
    if len(emg_list) != 1:
        raise ValueError(f"emg_chan must name exactly one channel, got "
                         f"{len(emg_list)}")
    emg = _as_index(emg_list[0], names, "emg_chan")

    if not eeg:
        raise ValueError("eeg_chan must name at least one channel")
    if emg in eeg:
        raise ValueError(f"channel {names[emg]!r} is listed as both EEG and "
                         f"EMG. They must be different channels.")
    if len(set(eeg)) != len(eeg):
        raise ValueError("eeg_chan lists the same channel more than once")
    return eeg, emg


def pick_best_eeg(raw, eeg_idx: list[int], sfreq: float) -> int:
    """Pick the cleanest EEG channel out of the several a recording may hold.

    Chooses whichever has the most of its power in the frequencies sleep scoring
    depends on, rather than in high-frequency noise, judged over the first five
    minutes.
    """
    if len(eeg_idx) == 1:
        return eeg_idx[0]
    stop = min(int(300 * sfreq), raw.n_times)
    best, best_snr = eeg_idx[0], -np.inf
    for i in eeg_idx:
        x = raw.get_data(picks=[i], start=0, stop=stop)[0]
        # use the same frequency measurement as the rest of the pipeline
        f, p = H._welch(x, sfreq, int(2 * sfreq))
        snr = p[(f >= 0.5) & (f <= 30)].sum() / (p[f > 30].sum() + 1e-12)
        if snr > best_snr:
            best, best_snr = i, snr
    return best


# ------------------------------------------------------------- featurization
def featurize(entry: dict, probe_seconds: float = 900.0,
              fps: float | None = None,
              mm_per_px: float | None = None,
              eeg_chan=None, emg_chan=None) -> pd.DataFrame:
    """Read one recording and return its per-epoch feature table.

    Opens the EDF, picks the best EEG channel, works out what the recording can
    measure, and computes the EEG, EMG and movement features for every epoch.
    Manual scoring and video tracking are used if given and skipped if not.

    `fps` declares a constant video frame rate, used only when the tracking has
    no timestamps file at all -- see resolve_frame_times(). `mm_per_px` converts
    tracked positions to millimetres, so `velocity` is reported in mm/s instead
    of px/s; it does not change what the model sees, because velocity is
    z-scored within recording and a constant scale cancels exactly. Both also
    accept an `entry` key of the same name.
    """
    fps = fps if fps is not None else entry.get("fps")
    mm_per_px = mm_per_px if mm_per_px is not None else entry.get("mm_per_px")
    eeg_chan = eeg_chan if eeg_chan is not None else entry.get("eeg_chan")
    emg_chan = emg_chan if emg_chan is not None else entry.get("emg_chan")
    raw = mne.io.read_raw_edf(entry["edf"], preload=False)
    sfreq = float(raw.info["sfreq"])
    names = raw.ch_names

    if entry["dataset"] != "bids":
        eeg_idx, emg_i = resolve_channels(raw, eeg_chan, emg_chan)
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
    vel_unit = "px/s"
    if entry.get("pkl"):
        coords, meta = load_coordinates(entry["pkl"])
        # search only the folder the coordinates came from
        t, vel_src = resolve_frame_times(entry["recording"], len(coords),
                                         raw.n_times / sfreq, meta,
                                         os.path.dirname(entry["pkl"]), fps=fps)
        vel = H.velocity_features(coords, t, n_ep)
        if mm_per_px:
            # Convert after the feature, not before it. `log_velocity` is
            # log10(v + eps) with a fixed eps, so rescaling the coordinates
            # would move it by log10(k) at high speed but by much less near
            # zero -- exactly where a sleeping animal sits. Shifting the log by
            # log10(k) instead is the same as scaling eps with the units, which
            # keeps the change an exact constant offset and therefore invisible
            # to the within-recording z-score the model actually reads.
            k = float(mm_per_px)
            vel["velocity"] = vel["velocity"] * k
            vel["log_velocity"] = vel["log_velocity"] + np.log10(k)
            vel_unit = "mm/s"
        df = pd.concat([df, vel], axis=1)
    else:
        df = pd.concat([df, H.velocity_features(None, None, n_ep)], axis=1)

    # --- labels ---
    # Scoring is optional. Most recordings arrive unscored -- that is the whole
    # point -- so a missing file leaves the state column empty, not an error.
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

    # add each feature's local average and variability over nearby epochs
    base_cols = [c for c in df.columns
                 if c not in ("epoch", "t_start", "recording", "subject",
                              "dataset", "group", "state")]
    df = H.add_temporal_context(df, base_cols, windows=CONTEXT_WINDOWS)

    # Rescale features against this recording's own average and spread. Done
    # over every epoch, so the reference reflects the animal's real mix of
    # sleep and wake rather than whichever epochs get used later.
    df = H.zscore_within(df, H.zscore_target_columns(df), group_col="recording")

    # record which features this recording was actually able to supply
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
        "velocity_unit": vel_unit,
        "eeg_channel": names[best], "emg_channel": names[emg_i],
        "n_epochs": int(n_ep),
    }
    return df

