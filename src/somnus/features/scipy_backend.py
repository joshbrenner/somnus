"""Tiered, bandwidth-aware feature extraction for a generalizable sleep scorer.

GOAL
----
One model that runs on whatever data you give it: it uses high-frequency EEG and
video when they exist, and degrades gracefully (not silently wrongly) when they
do not. Achieved by splitting features into TIERS by frequency, measuring each
recording's true usable bandwidth, and emitting NaN + an availability indicator
for tiers a recording cannot support.

WHY TIERS (measured, not assumed)
---------------------------------
Bandwidth and mains contamination vary drastically across sources:
  * A 5 kHz in-house rig: effective edge ~1700-2500 Hz. 60 Hz mains +3 to
    +13 dB, plus a 120 Hz harmonic. No 50 Hz.
  * Most labs in a public multi-lab corpus (128 Hz): edge ~64 Hz, with 50 Hz
    mains up to +21 dB in some.
  * One contributing lab: EEG lowpass-filtered with a corner near 25-26 Hz --
    it tracks other labs to 25 Hz then falls off a cliff (-13 dB by 28 Hz).
    Its EMG is NOT filtered (edge 64 Hz).
So the only frequency range every source genuinely measures is <= 25 Hz.

TIERS
  1  <= 25 Hz   delta .5-4, theta 5-10, alpha 10-15, beta 15-25   (universal)
  2  25-45 Hz   high-beta / low-gamma
  3  45-63 Hz   gamma (fits inside a 128 Hz recording's Nyquist)
  4  63-150 Hz  high gamma + wideband EMG              (high-rate rigs only)

Tier-1 relative powers are normalised to the TIER-1 SUM only, so their meaning is
identical no matter which higher tiers exist. Higher tiers are expressed as
ratios to that same tier-1 total, so they are comparable too.

MAINS
Rather than time-domain notch filtering (expensive on 75M samples and prone to
edge transients), mains bins are interpolated across in the PSD before band
integration. Exact, cheap, and applied identically everywhere.

Everything here is dataset-agnostic: arrays in, features out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

# Band integration method, kept here so this module and its openseize twin
# (`openseize_backend.py`) expose the same knob. Only "trapezoid" is
# implemented on the scipy path -- it is what the published model was fitted
# with; the Simpson alternative lives in the openseize port.
INTEGRATION = "trapezoid"


def _welch(x: np.ndarray, sfreq: float, nperseg: int,
           axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD, one independent estimate per row for 2-D input.

    Exists so that callers outside this module (e.g. the dataset adapter's
    channel chooser) can obtain a PSD from whichever backend is active, rather
    than importing scipy directly and thereby bypassing the port.
    `openseize_backend._welch` has an identical signature and returns values
    equal to this one to floating-point noise.
    """
    arr = np.asarray(x, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    f, p = welch(arr, fs=sfreq, nperseg=nperseg, axis=-1)
    return f, (p[0] if squeeze else p)

EPOCH_SEC = 4.0

# ---- tier definitions -------------------------------------------------------
TIER1_BANDS = {"delta": (0.5, 4.0), "theta": (5.0, 10.0),
               "alpha": (10.0, 15.0), "beta": (15.0, 25.0)}
TIER1_ORDER = ["delta", "theta", "alpha", "beta"]
TIER1_TOP = 25.0

TIER_BANDS = {           # tier -> (name, lo, hi)
    2: ("gamma1", 25.0, 45.0),
    3: ("gamma2", 45.0, 63.0),
    4: ("gamma3", 63.0, 150.0),
}
TIER_TOP = {1: TIER1_TOP, 2: 45.0, 3: 63.0, 4: 150.0}

# EMG bands: low/mid are universal (every source reaches 64 Hz on EMG),
# high requires a wideband recording.
EMG_BANDS = {"emg_low": (5.0, 25.0), "emg_mid": (30.0, 63.0),
             "emg_high": (63.0, 300.0)}
EMG_TIER = {"emg_low": 1, "emg_mid": 3, "emg_high": 4}

MAINS_HZ = (50.0, 60.0, 100.0, 120.0, 150.0, 180.0)
MAINS_HALFWIDTH = 1.5

# Features that are amplitude-like and therefore z-scored within recording.
AMPLITUDE_FEATURES = ["t1_power_log", "emg_low_log", "emg_mid_log",
                      "emg_high_log", "log_velocity"]

# Ratio/shape features. In principle self-normalising, but measurement shows
# they still carry large per-recording offsets across labs (delta_rel differs by
# ~1 SD between sources, because highpass corners range from none at all to
# -26 dB at 0.5 Hz). z-scoring these within recording as well removes the
# site offset while preserving the within-recording modulation that identifies
# state. Both raw and _z versions are emitted so the two can be compared.
RATIO_FEATURES = ([f"{b}_rel" for b in TIER1_ORDER]
                  + ["theta_delta_log", "delta_index", "emg_ratio_hi_lo"]
                  + [f"{TIER_BANDS[t][0]}_ratio_log" for t in (2, 3, 4)])

# Optional feature blocks -> the availability indicator that guards them.
# Used by the model to know when a value is genuinely absent.
OPTIONAL_BLOCKS = {
    "tier2": ["gamma1_ratio_log"],
    "tier3": ["gamma2_ratio_log", "emg_mid_log"],
    "tier4": ["gamma3_ratio_log", "emg_high_log"],
    "video": ["log_velocity"],
}


# ---------------------------------------------------------------- bandwidth
def detect_bandwidth(signal: np.ndarray, sfreq: float,
                     ref: tuple[float, float] = (8.0, 20.0),
                     slope_db_per_hz: float = -2.0,
                     floor_db: float = -45.0,
                     start_hz: float = 20.0,
                     max_seconds: float = 900.0) -> float:
    """Effective upper edge (Hz) of genuine signal, i.e. the filter corner.

    EEG power falls with frequency even with no filter (1/f), so a fixed dB
    threshold would mistake natural decay for filtering: lab_1 is ~-18 dB at
    63 Hz yet is entirely real. What distinguishes an anti-alias/lowpass filter
    is a sudden *change of slope* -- a cliff.

    Detection: smooth the mains-suppressed spectrum, then scan upward from
    `start_hz` for the first frequency where the local slope is steeper than
    `slope_db_per_hz` and stays steep in the following window (sustained cliff),
    or where power falls below `floor_db` relative to the 8-20 Hz reference.

    Calibrated against measured data: lab_3 falls ~-3.4 dB/Hz across 25-30 Hz
    (correctly flagged, corner ~26 Hz) whereas lab_1 falls ~-0.6 dB/Hz across
    25-40 Hz (correctly passed).
    """
    n = min(len(signal), int(max_seconds * sfreq))
    x = np.asarray(signal[:n], dtype=float)
    nper = int(4 * sfreq)
    if len(x) < nper:
        return float(sfreq / 2)
    f, p = welch(x, fs=sfreq, nperseg=nper)
    p = suppress_mains(f, p)
    refl = np.median(p[(f >= ref[0]) & (f <= ref[1])])
    if not np.isfinite(refl) or refl <= 0:
        return float(sfreq / 2)

    db = 10 * np.log10(np.clip(p, 1e-300, None) / refl)
    res = f[1] - f[0]
    k = max(1, int(round(2.0 / res)))                 # ~2 Hz smoothing
    dbs = pd.Series(db).rolling(k, center=True, min_periods=1).median().values

    step = max(1, int(round(3.0 / res)))              # slope over ~3 Hz
    i0 = int(np.searchsorted(f, start_hz))
    fmax = float(f[-1])
    for i in range(i0, len(f) - 2 * step):
        if dbs[i] <= floor_db:
            return float(f[i])
        s1 = (dbs[i + step] - dbs[i]) / (f[i + step] - f[i])
        if s1 < slope_db_per_hz:
            s2 = (dbs[i + 2 * step] - dbs[i + step]) / (f[i + 2 * step] - f[i + step])
            # sustained cliff, and already meaningfully below the reference
            if s2 < slope_db_per_hz / 2 and dbs[i + step] < -8.0:
                return float(f[i])
    return fmax


def available_tiers(edge_hz: float) -> set[int]:
    """Which spectral tiers a recording with this EEG edge can support.

    Tier 1 (<=25 Hz) is unconditional: it is the universal baseline, and a
    recording that could not support it would be unusable for sleep scoring at
    all. This also absorbs the fact that the cliff detector fires at the *onset*
    of a rolloff, so lab_3 measures ~24 Hz even though its 25 Hz content is
    demonstrably as strong as every other lab's (-3.4 to -3.9 dB vs -4.1 to
    -8.5 dB relative to 8-20 Hz).
    """
    return {1} | {t for t, top in TIER_TOP.items()
                  if t != 1 and top <= edge_hz + 1e-9}


def available_emg_bands(edge_hz: float) -> set[str]:
    """Which EMG bands a recording supports, gated by the EMG channel's own
    bandwidth. Must be measured separately from EEG: lab_3 lowpass-filtered its
    EEG at ~25 Hz but left its EMG intact to 64 Hz."""
    ok = {"emg_low"}                     # 5-25 Hz: always available
    for name, (lo, hi) in EMG_BANDS.items():
        if name == "emg_low":
            continue
        if hi <= edge_hz + 1e-9:
            ok.add(name)
    return ok


# ------------------------------------------------------------------- mains
def suppress_mains(freqs: np.ndarray, psd: np.ndarray,
                   mains: tuple[float, ...] = MAINS_HZ,
                   halfwidth: float = MAINS_HALFWIDTH) -> np.ndarray:
    """Linearly interpolate the PSD across mains bins (and harmonics).

    psd may be 1D (n_freq) or 2D (n_epochs, n_freq); returns a copy.
    """
    out = np.array(psd, dtype=float, copy=True)
    nyq = freqs[-1]
    for f0 in mains:
        if f0 - halfwidth <= freqs[0] or f0 + halfwidth >= nyq:
            continue
        kill = (freqs >= f0 - halfwidth) & (freqs <= f0 + halfwidth)
        if not kill.any():
            continue
        lo_i = np.flatnonzero(~kill & (freqs < f0))
        hi_i = np.flatnonzero(~kill & (freqs > f0))
        if len(lo_i) == 0 or len(hi_i) == 0:
            continue
        a, b = lo_i[-1], hi_i[0]
        w = (freqs[kill] - freqs[a]) / (freqs[b] - freqs[a])
        if out.ndim == 1:
            out[kill] = out[a] * (1 - w) + out[b] * w
        else:
            out[:, kill] = (out[:, [a]] * (1 - w)[None, :]
                            + out[:, [b]] * w[None, :])
    return out


def _band_power(freqs: np.ndarray, psd: np.ndarray,
                lo: float, hi: float) -> np.ndarray | float:
    hi = min(hi, freqs[-1])
    if hi <= lo:
        return np.full(psd.shape[0], np.nan) if psd.ndim == 2 else np.nan
    idx = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if len(idx) < 2:
        return np.full(psd.shape[0], np.nan) if psd.ndim == 2 else np.nan
    if psd.ndim == 2:
        return np.trapezoid(psd[:, idx], freqs[idx], axis=1)
    return float(np.trapezoid(psd[idx], freqs[idx]))


# ------------------------------------------------------------ epoch layout
def n_epochs_for(n_samples: int, sfreq: float,
                 epoch_sec: float = EPOCH_SEC) -> int:
    return int(np.floor(n_samples / sfreq / epoch_sec))


def epoch_times(n_epochs: int, epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    return np.arange(n_epochs, dtype=float) * epoch_sec


def _iter_windows(sig: np.ndarray, win: int, n_epochs: int,
                  block: int = 400):
    """Yield (start_epoch, (n,win) array) blocks to bound peak memory."""
    n_valid = min(n_epochs, len(sig) // win)
    for s in range(0, n_valid, block):
        e = min(s + block, n_valid)
        yield s, sig[s * win:e * win].reshape(e - s, win)


# -------------------------------------------------------- spectral features
def eeg_features(eeg: np.ndarray, sfreq: float, n_epochs: int,
                 tiers: set[int], epoch_sec: float = EPOCH_SEC
                 ) -> pd.DataFrame:
    """Tier-aware EEG features. Unsupported tiers are left as NaN.

    Tier-1 relative powers use the tier-1 sum as denominator, so they mean the
    same thing regardless of which higher tiers exist.
    """
    win = int(round(epoch_sec * sfreq))
    cols = ([f"{b}_rel" for b in TIER1_ORDER]
            + ["theta_delta_log", "delta_index", "t1_power_log"]
            + [f"{TIER_BANDS[t][0]}_ratio_log" for t in (2, 3, 4)])
    out = pd.DataFrame(index=np.arange(n_epochs), columns=cols, dtype=float)
    if eeg is None:
        return out

    for s0, w in _iter_windows(np.asarray(eeg, dtype=float), win, n_epochs):
        f, psd = welch(w, fs=sfreq, nperseg=win, axis=1)
        psd = suppress_mains(f, psd)
        idx = np.arange(s0, s0 + len(w))

        bp = {b: _band_power(f, psd, *TIER1_BANDS[b]) for b in TIER1_ORDER}
        t1 = sum(bp[b] for b in TIER1_ORDER) + 1e-30
        for b in TIER1_ORDER:
            out.loc[idx, f"{b}_rel"] = bp[b] / t1
        d = bp["delta"] + 1e-30
        th = bp["theta"] + 1e-30
        out.loc[idx, "theta_delta_log"] = np.log10(th / d)
        # delta index re-based on the tier-1 total (comparable across sources)
        out.loc[idx, "delta_index"] = (d - (t1 - d)) / t1
        out.loc[idx, "t1_power_log"] = np.log10(t1)

        for t in (2, 3, 4):
            name, lo, hi = TIER_BANDS[t]
            if t in tiers:
                p = _band_power(f, psd, lo, hi)
                out.loc[idx, f"{name}_ratio_log"] = np.log10(
                    np.clip(p, 1e-30, None) / t1)
    return out


def emg_features(emg: np.ndarray, sfreq: float, n_epochs: int,
                 bands: set[str], epoch_sec: float = EPOCH_SEC
                 ) -> pd.DataFrame:
    """EMG band powers (log). Bands beyond the EMG channel's edge stay NaN.

    `bands` comes from available_emg_bands() on the EMG channel's OWN measured
    bandwidth, not the EEG's.

    EMG spectral shape varies hugely between recordings and labs -- one animal
    can put ~50% of EMG power below 25 Hz while another puts ~75% above 90 Hz
    and a third 99% above 45 Hz. So no single band is comparable in
    absolute terms; several are emitted and each is z-scored within recording
    downstream. What carries the atonia signal is modulation over time within a
    recording, not absolute level.
    """
    win = int(round(epoch_sec * sfreq))
    cols = [f"{k}_log" for k in EMG_BANDS] + ["emg_ratio_hi_lo"]
    out = pd.DataFrame(index=np.arange(n_epochs), columns=cols, dtype=float)
    if emg is None:
        return out

    for s0, w in _iter_windows(np.asarray(emg, dtype=float), win, n_epochs):
        f, psd = welch(w, fs=sfreq, nperseg=win, axis=1)
        psd = suppress_mains(f, psd)
        idx = np.arange(s0, s0 + len(w))
        vals = {}
        for k, (lo, hi) in EMG_BANDS.items():
            if k not in bands:
                continue
            p = _band_power(f, psd, lo, hi)
            vals[k] = p
            out.loc[idx, f"{k}_log"] = np.log10(np.clip(p, 1e-30, None))
        # scale-free shape descriptor, available whenever both bands are
        if "emg_mid" in vals and "emg_low" in vals:
            out.loc[idx, "emg_ratio_hi_lo"] = np.log10(
                np.clip(vals["emg_mid"], 1e-30, None)
                / np.clip(vals["emg_low"], 1e-30, None))
    return out


# -------------------------------------------------------- velocity features
def velocity_features(coords: np.ndarray | None,
                      frame_times: np.ndarray | None,
                      n_epochs: int,
                      epoch_sec: float = EPOCH_SEC) -> pd.DataFrame:
    """Mean speed per epoch = path length / elapsed time within the epoch.

    Robust to the irregular frame intervals caused by dropped frames. Requires
    TRUE per-frame times; a constant-fps assumption misplaces frames by minutes.
    """
    out = pd.DataFrame(index=np.arange(n_epochs),
                       columns=["velocity", "log_velocity"], dtype=float)
    if coords is None or frame_times is None:
        return out
    xy = np.asarray(coords, dtype=float)
    t = np.asarray(frame_times, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or len(t) != len(xy) or len(t) < 2:
        return out

    step = np.sqrt(np.sum(np.diff(xy[:, :2], axis=0) ** 2, axis=1))
    dt = np.diff(t)
    mid = 0.5 * (t[:-1] + t[1:])
    idx = np.floor(mid / epoch_sec).astype(int)
    ok = ((idx >= 0) & (idx < n_epochs) & (dt > 0)
          & np.isfinite(step) & np.isfinite(dt))
    if not ok.any():
        return out
    path = np.bincount(idx[ok], weights=step[ok], minlength=n_epochs)[:n_epochs]
    el = np.bincount(idx[ok], weights=dt[ok], minlength=n_epochs)[:n_epochs]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(el > 0, path / el, np.nan)
    out["velocity"] = v
    out["log_velocity"] = np.log10(v + 1e-3)
    return out


# ------------------------------------------------------------------ context
def add_temporal_context(df: pd.DataFrame, columns: list[str],
                         windows: tuple[int, ...] = (3, 15)) -> pd.DataFrame:
    """Centred rolling mean/std per feature: label-free temporal context.

    NaN-safe: a column that is entirely NaN (unsupported tier) yields NaN
    context, which the missing-indicator machinery then handles.
    """
    out = df.copy()
    cols = [c for c in columns if c in df.columns]
    for w in windows:
        r = df[cols].rolling(window=w, center=True, min_periods=1)
        m, s = r.mean(), r.std()
        for c in cols:
            out[f"{c}_mean{w}"] = m[c]
            out[f"{c}_std{w}"] = s[c]
    return out


# ------------------------------------------------------------ normalisation
def zscore_within(df: pd.DataFrame, columns: list[str],
                  group_col: str = "recording",
                  suffix: str = "_z") -> pd.DataFrame:
    """Within-recording z-score. Unsupervised, so no label leakage.

    Removes per-recording amplitude offsets (electrode impedance, gain,
    filtering), which is what lets EMG/velocity/absolute power transfer across
    animals and labs.
    """
    out = df.copy()
    cols = [c for c in columns if c in df.columns]
    g = df.groupby(group_col)[cols]
    mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
    for c in cols:
        out[f"{c}{suffix}"] = (df[c] - mu[c]) / sd[c]
    return out


def amplitude_columns(df: pd.DataFrame) -> list[str]:
    """Amplitude-like columns (incl. their context variants) to z-score."""
    return [c for c in df.columns
            if any(c == a or c.startswith(a + "_") for a in AMPLITUDE_FEATURES)]


def zscore_target_columns(df: pd.DataFrame) -> list[str]:
    """All columns to z-score within recording: amplitude AND ratio features.

    Ratio features are included because measurement showed they still carry ~1 SD
    site offsets (differing highpass corners and montages), which a model can
    latch onto as a site fingerprint instead of learning state.
    """
    roots = AMPLITUDE_FEATURES + RATIO_FEATURES
    return [c for c in df.columns
            if any(c == r or c.startswith(r + "_") for r in roots)
            and not c.endswith("_z")]


# ------------------------------------------------------------------- labels
def labels_from_onehot(df: pd.DataFrame, n_epochs: int, bin_sec: float = 0.5,
                       epoch_sec: float = EPOCH_SEC, purity: float = 0.75,
                       state_cols: tuple[str, ...] = ("Wake", "NREM", "REM"),
                       exclude_cols: tuple[str, ...] = ("Artifact", "Unclear"),
                       ) -> np.ndarray:
    """Majority label per epoch from sub-epoch one-hot bins.

    Epochs containing excluded or unlabelled bins, or failing the purity
    threshold, become None: they are dropped from training (no label noise) but
    can still be predicted, keeping the time series contiguous.
    """
    per = int(round(epoch_sec / bin_sec))
    have = [c for c in state_cols if c in df.columns]
    excl = [c for c in exclude_cols if c in df.columns]
    V = df[have].to_numpy()
    E = df[excl].to_numpy() if excl else np.zeros((len(df), 1))
    out = np.full(n_epochs, None, dtype=object)
    for e in range(min(n_epochs, len(df) // per)):
        sl = slice(e * per, (e + 1) * per)
        if E[sl].sum() > 0:
            continue
        c = V[sl].sum(axis=0)
        if c.sum() < per:
            continue
        k = int(np.argmax(c))
        if c[k] >= per * purity:
            out[e] = have[k]
    return out


def labels_from_stage_events(onsets, stages, n_epochs: int, mapping: dict,
                             epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """Labels from BIDS-style one-row-per-epoch events (onset + stage code)."""
    out = np.full(n_epochs, None, dtype=object)
    for o, s in zip(np.asarray(onsets, dtype=float), np.asarray(stages)):
        e = int(round(o / epoch_sec))
        if 0 <= e < n_epochs:
            out[e] = mapping.get(str(s), mapping.get(s))
    return out
