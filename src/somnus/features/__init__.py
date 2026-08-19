"""Feature computation backends.

Two numerically identical implementations of the tiered, bandwidth-aware
feature set (verified by ``tools/verify_openseize_port.py``):

* :mod:`somnus.features.openseize_backend` — the shipped implementation
  (openseize PSD); what the released model was trained with.
* :mod:`somnus.features.scipy_backend` — the original scipy implementation,
  kept as a cross-check and automatic fallback.

Select with ``SOMNUS_FEATURE_BACKEND=openseize|scipy`` (default openseize,
falling back to scipy if openseize is not importable).
"""
from __future__ import annotations

import os


def get_backend():
    """Return the active feature-backend module, honouring SOMNUS_FEATURE_BACKEND."""
    choice = os.environ.get("SOMNUS_FEATURE_BACKEND", "openseize").lower()
    if choice == "scipy":
        from somnus.features import scipy_backend as H
        return H
    try:
        from somnus.features import openseize_backend as H
    except ImportError as e:  # openseize not installed
        from somnus.features import scipy_backend as H
        print(f"[features] openseize unavailable ({e}); "
              f"falling back to the scipy backend (numerically equivalent).")
    return H
