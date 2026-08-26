"""Turn raw EEG, EMG and video into the numbers the scorer reads.

One row per 4-second epoch: how much power sits in each frequency band, how
active the muscles are, and how fast the animal is moving.

Recording equipment varies enormously in what it can actually measure. A 5 kHz
rig captures frequencies into the thousands; a 128 Hz one stops at 64; some
labs filter their EEG at 25 Hz. So features are grouped into TIERS by frequency,
each recording's real usable range is measured from the data itself, and any
tier it cannot support is left blank rather than filled with a wrong number.
Every recording supports tier 1, which is why one model can score all of them.

    tier 1   up to 25 Hz    delta, theta, alpha, beta   (every recording)
    tier 2   25-45 Hz       high beta / low gamma
    tier 3   45-63 Hz       gamma
    tier 4   63-150 Hz      high gamma and wideband EMG (high-rate rigs only)

Band powers are given relative to the tier-1 total, so they mean the same thing
whether or not the higher tiers exist.

Nothing here knows about files or datasets: arrays in, features out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from openseize.spectra.estimators import psd as _os_psd


def _welch(x: np.ndarray, sfreq: float, nperseg: int,
           axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Measure how much power the signal carries at each frequency.

    Takes one stretch of signal, or a stack of them (one row per epoch), and
    returns the frequencies and the power at each. Rows are measured
    independently, so one epoch never influences another.
    """
    arr = np.asarray(x, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    _, freqs, p = _os_psd(arr, fs=sfreq, axis=-1,
                          resolution=float(sfreq) / float(nperseg))
    p = np.asarray(p, dtype=float)
    return np.asarray(freqs, dtype=float), (p[0] if squeeze else p)

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

# EMG bands. Low and mid are available on any recording; high needs one that
# samples fast enough to reach 300 Hz.
EMG_BANDS = {"emg_low": (5.0, 25.0), "emg_mid": (30.0, 63.0),
             "emg_high": (63.0, 300.0)}
EMG_TIER = {"emg_low": 1, "emg_mid": 3, "emg_high": 4}

MAINS_HZ = (50.0, 60.0, 100.0, 120.0, 150.0, 180.0)
MAINS_HALFWIDTH = 1.5

# Features whose absolute size depends on the equipment, so they are rescaled
# against their own recording.
AMPLITUDE_FEATURES = ["t1_power_log", "emg_low_log", "emg_mid_log",
                      "emg_high_log", "log_velocity"]

# Band ratios and shape descriptors. These are rescaled against their own
# recording too -- see zscore_target_columns() for why. Both the raw and the
# rescaled version are kept.
RATIO_FEATURES = ([f"{b}_rel" for b in TIER1_ORDER]
                  + ["theta_delta_log", "delta_index", "emg_ratio_hi_lo"]
                  + [f"{TIER_BANDS[t][0]}_ratio_log" for t in (2, 3, 4)])

# Features that not every recording can supply, each paired with the column
# that records whether this one could. Lets the model tell "no movement" from
# "no camera".
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
    """Find the highest frequency this recording genuinely measured.

    Above some point every recording is showing filter rolloff rather than
    brain activity, and features computed up there would be noise. EEG power
    naturally falls off with frequency, so a simple power threshold cannot tell
    real signal from a filter. What gives a filter away is the spectrum falling
    off a cliff, so this scans upward for a sudden and sustained steepening.

    Calibrated on real recordings: a filtered one drops ~3.4 dB per Hz at its
    corner, an unfiltered one only ~0.6 dB per Hz.
    """
    n = min(len(signal), int(max_seconds * sfreq))
    x = np.asarray(signal[:n], dtype=float)
    nper = int(4 * sfreq)
    if len(x) < nper:
        return float(sfreq / 2)
    f, p = _welch(x, sfreq, nper)
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
    """Which frequency tiers this recording can support, given its upper edge.

    Tier 1 always counts. It is the baseline every recording shares, and one
    that could not manage it would be unusable for sleep scoring anyway.
    """
    return {1} | {t for t, top in TIER_TOP.items()
                  if t != 1 and top <= edge_hz + 1e-9}


def available_emg_bands(edge_hz: float) -> set[str]:
    """Which EMG bands this recording can support.

    Measured from the EMG channel itself, not the EEG: a lab may filter its EEG
    heavily and leave the EMG untouched.
    """
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
    """Remove mains hum from the spectrum.

    Electrical noise at 50 or 60 Hz and its harmonics can be far stronger than
    the brain activity around it. Those frequencies are replaced by a straight
    line drawn between their neighbours, which is cheaper than filtering the
    signal itself and cannot introduce edge artifacts.
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
    """Total power between two frequencies, or blank if the band is out of range."""
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
    """How many whole epochs fit in a recording of this length."""
    return int(np.floor(n_samples / sfreq / epoch_sec))


def epoch_times(n_epochs: int, epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """The start time in seconds of every epoch."""
    return np.arange(n_epochs, dtype=float) * epoch_sec


def _iter_windows(sig: np.ndarray, win: int, n_epochs: int,
                  block: int = 400):
    """Hand out the signal a few hundred epochs at a time.

    A long recording at a high sample rate will not fit in memory all at once.
    """
    n_valid = min(n_epochs, len(sig) // win)
    for s in range(0, n_valid, block):
        e = min(s + block, n_valid)
        yield s, sig[s * win:e * win].reshape(e - s, win)


# -------------------------------------------------------- spectral features
def eeg_features(eeg: np.ndarray, sfreq: float, n_epochs: int,
                 tiers: set[int], epoch_sec: float = EPOCH_SEC
                 ) -> pd.DataFrame:
    """Measure the EEG frequency bands for every epoch.

    Each band is reported as a share of the tier-1 total, so the numbers mean
    the same thing whether or not the recording reaches the higher tiers. Bands
    the recording cannot support are left blank.
    """
    win = int(round(epoch_sec * sfreq))
    cols = ([f"{b}_rel" for b in TIER1_ORDER]
            + ["theta_delta_log", "delta_index", "t1_power_log"]
            + [f"{TIER_BANDS[t][0]}_ratio_log" for t in (2, 3, 4)])
    out = pd.DataFrame(index=np.arange(n_epochs), columns=cols, dtype=float)
    if eeg is None:
        return out

    for s0, w in _iter_windows(np.asarray(eeg, dtype=float), win, n_epochs):
        f, psd = _welch(w, sfreq, win)
        psd = suppress_mains(f, psd)
        idx = np.arange(s0, s0 + len(w))

        bp = {b: _band_power(f, psd, *TIER1_BANDS[b]) for b in TIER1_ORDER}
        t1 = sum(bp[b] for b in TIER1_ORDER) + 1e-30
        for b in TIER1_ORDER:
            out.loc[idx, f"{b}_rel"] = bp[b] / t1
        d = bp["delta"] + 1e-30
        th = bp["theta"] + 1e-30
        out.loc[idx, "theta_delta_log"] = np.log10(th / d)
        # how far delta dominates the other bands, on a -1 to 1 scale
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
    """Measure muscle activity in several frequency bands, for every epoch.

    Where EMG power sits varies wildly between animals and electrode placements,
    so no single band is comparable across recordings. Several are measured and
    each is later compared against that recording's own average. What identifies
    REM is muscle tone dropping relative to the rest of the recording, not any
    absolute level.
    """
    win = int(round(epoch_sec * sfreq))
    cols = [f"{k}_log" for k in EMG_BANDS] + ["emg_ratio_hi_lo"]
    out = pd.DataFrame(index=np.arange(n_epochs), columns=cols, dtype=float)
    if emg is None:
        return out

    for s0, w in _iter_windows(np.asarray(emg, dtype=float), win, n_epochs):
        f, psd = _welch(w, sfreq, win)
        psd = suppress_mains(f, psd)
        idx = np.arange(s0, s0 + len(w))
        vals = {}
        for k, (lo, hi) in EMG_BANDS.items():
            if k not in bands:
                continue
            p = _band_power(f, psd, lo, hi)
            vals[k] = p
            out.loc[idx, f"{k}_log"] = np.log10(np.clip(p, 1e-30, None))
        # where EMG power sits, high versus low, independent of overall size
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
    """How fast the animal moved, averaged over each epoch.

    Distance travelled divided by time elapsed, using the true time of every
    video frame. Cameras drop frames, so the real intervals are uneven and
    assuming a fixed frame rate would misplace positions by minutes.
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
    """Add each feature's local average and variability over nearby epochs.

    Sleep states are defined partly by what surrounds them, so every feature
    also gets a short-window and a long-window summary of its neighbourhood.
    This uses no labels, only the signal.
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


# ------------------------------------------------------------ normalization
def zscore_within(df: pd.DataFrame, columns: list[str],
                  group_col: str = "recording",
                  suffix: str = "_z") -> pd.DataFrame:
    """Rescale each feature against that recording's own average and spread.

    Electrode impedance, amplifier gain and filtering shift every recording by
    a different amount, so raw values are not comparable between animals. What
    is comparable is how a feature moves relative to the rest of its own
    recording, which is what this measures.
    """
    out = df.copy()
    cols = [c for c in columns if c in df.columns]
    g = df.groupby(group_col)[cols]
    mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
    for c in cols:
        out[f"{c}{suffix}"] = (df[c] - mu[c]) / sd[c]
    return out


def amplitude_columns(df: pd.DataFrame) -> list[str]:
    """The features whose absolute size depends on the equipment used."""
    return [c for c in df.columns
            if any(c == a or c.startswith(a + "_") for a in AMPLITUDE_FEATURES)]


def zscore_target_columns(df: pd.DataFrame) -> list[str]:
    """Every feature that gets rescaled against its own recording.

    Band ratios are included as well as raw amplitudes. In principle a ratio
    should already be comparable between labs, but in practice differences in
    filtering and electrode placement shift them enough that a model could learn
    to recognise the lab instead of the sleep state.
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
    """Read manual scoring, collapsing its fine bins onto whole epochs.

    Scoring files mark shorter stretches than one epoch, so an epoch takes the
    state that most of it agrees on. Epochs that are mixed, unscored, or marked
    as artifact get no label at all: they are still scored by the model, but
    never used to train it.
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
    """Read manual scoring from a BIDS events table, one row per epoch."""
    out = np.full(n_epochs, None, dtype=object)
    for o, s in zip(np.asarray(onsets, dtype=float), np.asarray(stages)):
        e = int(round(o / epoch_sec))
        if 0 <= e < n_epochs:
            out[e] = mapping.get(str(s), mapping.get(s))
    return out
