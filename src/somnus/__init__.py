"""Somnus — sleep-state scoring (Wake / NREM / REM) for mouse EEG/EMG(+video).

Quickstart::

    from somnus import load_model
    from somnus.predict import predict
    art = load_model()                       # the packaged v1.0 model
    labels, proba = predict(art, feature_df) # features from somnus.data.datasets.featurize()

The desktop application: ``somnus-gui`` (or ``python -m somnus.gui``).
"""
__version__ = "1.0.1"


def __getattr__(name):
    if name == "load_model":
        from somnus.predict import load_model
        return load_model
    raise AttributeError(f"module 'somnus' has no attribute {name!r}")
