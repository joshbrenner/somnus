"""Somnus — sleep-state scoring (Wake / NREM / REM) for mouse EEG/EMG(+video).

Quickstart::

    from somnus import load_model
    from somnus.predict import predict
    art = load_model()                       # the packaged v1.0 model
    labels, proba = predict(art, feature_df) # features from somnus.data.datasets.featurize()

The desktop application: ``somnus-gui`` (or ``python -m somnus.gui``).
"""
__version__ = "1.0.0"

# NOTE: only load_model is re-exported here. Re-exporting the predict()
# function as well would shadow the somnus.predict submodule attribute.
from somnus.predict import load_model  # noqa: F401
