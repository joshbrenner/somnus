"""Verify that somnus.features.openseize_backend reproduces scipy_backend exactly.

The openseize port swaps the PSD backend (scipy.signal.welch ->
openseize.spectra.estimators.psd). Both are Welch estimators, so with matched
parameters the features should be identical to floating-point noise. This script
proves that on synthetic and (optionally) real data, and separately quantifies
what the alternative Simpson band integration would change.

Must run in an environment with BOTH scipy and openseize:
    python tools/verify_openseize_port.py
    ... --real <recording_id>                # also check one real recording
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

from somnus.features import scipy_backend as OLD
from somnus.features import openseize_backend as NEW


def compare(a: pd.DataFrame, b: pd.DataFrame, label: str,
            tol: float = 1e-9) -> bool:
    """Print the worst disagreement between two feature frames.

    Error is normalised by each column's RANGE, not by |x|. Many features here
    are z-scores that legitimately pass through zero, and dividing by |x| near
    zero inflates a numerically meaningless difference into a huge ratio -- on
    real data that reported 3.9e-06 for an absolute difference of 5.7e-11 on a
    feature spanning 7.19. Range-normalisation measures what actually matters:
    the error relative to the spread the model sees.
    """
    assert list(a.columns) == list(b.columns), "column mismatch"
    worst, worst_col, worst_abs = 0.0, "", 0.0
    for c in a.columns:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            print(f"  {label}: NaN pattern differs in {c}")
            return False
        m = ~np.isnan(x)
        if not m.any():
            continue
        d = float(np.abs(x[m] - y[m]).max())
        rng = float(np.ptp(x[m]))
        rel = d / max(rng, 1e-12)
        if rel > worst:
            worst, worst_col, worst_abs = rel, c, d
    ok = worst < tol
    print(f"  {label}: max diff / column range = {worst:.3e} "
          f"(abs {worst_abs:.3e}, {worst_col or 'n/a'})"
          f"  -> {'IDENTICAL' if ok else 'DIFFERS'}")
    return ok


def synthetic(sfreq: float, n_epochs: int, seed: int = 0):
    """EEG-like 1/f signal with delta/theta peaks and mains, plus EMG-like noise."""
    rng = np.random.default_rng(seed)
    n = int(n_epochs * NEW.EPOCH_SEC * sfreq)
    t = np.arange(n) / sfreq
    # pink-ish background
    x = np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    x += 2.0 * np.sin(2 * np.pi * 2.0 * t)      # delta
    x += 1.0 * np.sin(2 * np.pi * 7.0 * t)      # theta
    x += 0.4 * np.sin(2 * np.pi * 60.0 * t)     # mains
    emg = rng.standard_normal(n) * (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t))
    return x, emg


def run_case(sfreq: float, n_epochs: int, tag: str) -> bool:
    eeg, emg = synthetic(sfreq, n_epochs)
    print(f"\n[{tag}] sfreq={sfreq:g} n_epochs={n_epochs}")

    e_old = OLD.detect_bandwidth(eeg, sfreq)
    e_new = NEW.detect_bandwidth(eeg, sfreq)
    print(f"  detect_bandwidth: old={e_old:.3f} Hz  new={e_new:.3f} Hz  "
          f"-> {'same' if abs(e_old - e_new) < 1e-9 else 'DIFFER'}")

    tiers = OLD.available_tiers(e_old)
    bands = OLD.available_emg_bands(e_old)
    ok = abs(e_old - e_new) < 1e-9
    ok &= compare(OLD.eeg_features(eeg, sfreq, n_epochs, tiers),
                  NEW.eeg_features(eeg, sfreq, n_epochs, tiers), "eeg_features")
    ok &= compare(OLD.emg_features(emg, sfreq, n_epochs, bands),
                  NEW.emg_features(emg, sfreq, n_epochs, bands), "emg_features")
    return ok


def simpson_impact(sfreq: float = 128.0, n_epochs: int = 300) -> None:
    """How much the openseize-native Simpson integration would move features."""
    eeg, emg = synthetic(sfreq, n_epochs)
    tiers = NEW.available_tiers(NEW.detect_bandwidth(eeg, sfreq))
    base = NEW.eeg_features(eeg, sfreq, n_epochs, tiers)
    NEW.INTEGRATION = "simpson"
    importlib.reload  # no-op; INTEGRATION is read at call time
    simp = NEW.eeg_features(eeg, sfreq, n_epochs, tiers)
    NEW.INTEGRATION = "trapezoid"

    print("\n[simpson vs trapezoid] median |relative change| per feature")
    rows = []
    for c in base.columns:
        x, y = base[c].to_numpy(float), simp[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            continue
        rel = np.abs(y[m] - x[m]) / np.maximum(np.abs(x[m]), 1e-12)
        rows.append((c, float(np.median(rel)), float(np.max(rel))))
    for c, med, mx in sorted(rows, key=lambda r: -r[1]):
        print(f"  {c:22} median {med:8.2%}   max {mx:8.2%}")


def real(recording: str) -> bool:
    """Featurise one real recording through both paths via the dataset adapter."""
    from somnus.data import datasets as B
    entries = B.discover_local() + B.discover_bids()
    match = [e for e in entries if e["recording"] == recording]
    if not match:
        print(f"\n[real] no recording '{recording}' -- skipped")
        return True
    print(f"\n[real] {recording}")
    B.H = OLD
    a = B.featurize(match[0])
    B.H = NEW
    b = B.featurize(match[0])
    B.H = OLD                                   # leave the module as we found it
    shared = [c for c in a.columns if c in b.columns
              and pd.api.types.is_numeric_dtype(a[c])]
    return compare(a[shared], b[shared], "featurize (all numeric cols)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    args = ap.parse_args()

    print("Verifying openseize port against the scipy implementation")
    print(f"  openseize psd backend, INTEGRATION={NEW.INTEGRATION!r}")

    ok = True
    ok &= run_case(5000.0, 40, "wideband 5000 Hz")
    ok &= run_case(128.0, 400, "low-rate 128 Hz")

    simpson_impact()

    if args.real:
        ok &= real(args.real)

    print("\n" + ("PORT VERIFIED: features are numerically identical."
                  if ok else "PORT MISMATCH -- see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
