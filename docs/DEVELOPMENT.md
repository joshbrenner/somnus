# Somnus — developer reference (v1.0)

Logistic-regression scorer for **Wake / NREM / REM** on 4 s epochs of mouse
EEG/EMG (+ optional video-derived velocity), with an HMM/Viterbi model for temporal smoothing.

Developed on Python 3.13 (openseize, mne, pandas, scipy, matplotlib,
PySide6, opencv-python, av). Requires Python 3.12+, the floor set by openseize.

## Layout

| Path | Contents |
|---|---|
| `src/somnus/` | The installable package (`pip install -e .`). |
| `src/somnus/predict.py` | Self-contained scorer — numpy/pandas only. What the GUI and batch tools call. |
| `src/somnus/models/` | The released JSON artifact, shipped as package data. |
| `src/somnus/features/` | Feature computation: PSD, band definitions, harmonization tiers. Pure — arrays in, features out. |
| `src/somnus/data/datasets.py` | `featurize()`: one recording (EDF + optional scoring and video tracking) → per-epoch feature table. |
| `src/somnus/train/finetune.py` | Anchored fine-tuning of the released model on your own labels. |
| `src/somnus/gui/` | The 5-tab PySide6 app; `core.py` is Qt-free and testable headless. |
| `src/somnus/scorer/` | The OpenCV signal/video scorer (`python -m somnus.scorer`). |

## Pipeline

Scoring one recording, from anywhere (installed or `pip install -e .`):

```bash
python -m somnus.predict --score myrec.edf --eeg 1 2 3 --emg 4 --out scored.csv
```

| Module | Role |
|---|---|
| `somnus.features` | **Feature computation.** Pure: arrays in → features out. Welch PSD via openseize; holds the band definitions and cross-site harmonization tiers. |
| `somnus.data.datasets` | `featurize()`: EDF (+ optional scoring, video tracking) → per-epoch feature table, with measured-bandwidth tier gating. |
| `somnus.predict` | Design matrix, logistic probabilities, HMM/Viterbi decode. Reads the JSON artifact; numpy/pandas only. |
| `somnus.train.finetune` | Anchored fine-tuning: adapt the released weights to your labels. |

### Spectral method

PSD is a Welch estimate from `openseize.spectra.estimators.psd`, called with
`resolution = sfreq / nperseg` on a window whose length equals nperseg — one
segment per epoch, Hann window, `detrend='constant'`, `scaling='density'`.

Bands are integrated with `np.trapezoid`. Do not substitute openseize's
`metrics.power`: it integrates by Simpson's rule, which shifts
`t1_power_log` / `beta_rel` / `alpha_rel` by 11–13% and invalidates the released
weights.

## Model

| File | What it is |
|---|---|
| `src/somnus/models/model_somnus_1.0.json` | **The release artifact.** Plain JSON: columns, centering stats, coefficients, transition matrix, priors. ~15 KB, no pickles. Ships in the wheel. |
| `src/somnus/predict.py` | **Self-contained scorer** |

```python
from somnus import load_model
from somnus.predict import predict
art = load_model()                         # packaged artifact
labels, proba = predict(art, feature_df)   # feature_df from somnus.data.datasets.featurize()
```

## Headline performance

Released variant `unified_z_noind`, 70 features, fitted on 61,164 epochs drawn
from six labs. Tested on held-out public-corpus mice (74 subjects, 1,515,059
labeled epochs):

| metric | value |
|---|---|
| accuracy | 0.9143 |
| balanced accuracy | 0.8582 |
| κ | 0.8447 |
| REM F1 | 0.7284 |

In-house data, leave-one-recording-out: accuracy 0.9749, balanced 0.9543.

These numbers describe the *base* model, which is a starting point rather than
the product: see the fine-tuning section of the README for why adapting it to
your own recordings is the intended workflow. A full write-up of the corpus
assembly, training recipe, and evaluation accompanies the forthcoming preprint.
