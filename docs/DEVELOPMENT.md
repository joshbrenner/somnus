# Somnus — developer / pipeline reference (v1.0)

Logistic-regression scorer for **Wake / NREM / REM** on 4 s epochs of mouse
EEG/EMG (+ optional video-derived velocity), with an HMM/Viterbi temporal decode.

An invariant enforced throughout: no source recording (EDF, scored CSV, video,
tracking pickle) is ever modified or moved — all output goes to the invoking
or project directory.

Developed on Python 3.13 (openseize, mne, scikit-learn, pandas, scipy,
matplotlib, joblib, PySide6, opencv-python, av); also runs on 3.10 without
openseize via the scipy feature backend.

## Layout

| Path | Contents |
|---|---|
| `src/somnus/` | The installable package (`pip install -e .`). |
| `src/somnus/predict.py` | Self-contained scorer — numpy/pandas only. What the GUI and batch tools call. |
| `src/somnus/models/` | The released JSON artifact, shipped as package data. |
| `src/somnus/features/` | Feature computation: `openseize_backend` (shipped) and `scipy_backend` (verified-identical fallback). Band definitions and harmonisation tiers live here. |
| `src/somnus/data/datasets.py` | Per-dataset adapters (file discovery, EDF/scoring/tracking parsing) → per-epoch table, manifest, CV folds. Add a new dataset here and nothing else changes. |
| `src/somnus/train/` | `train` (logistic + HMM, LORO CV, metrics, plots), `export` (portable JSON artifact), `finetune` (anchored fine-tuning). |
| `src/somnus/gui/` | The 5-tab PySide6 app; `core.py` is Qt-free and testable headless. |
| `src/somnus/scorer/` | The OpenCV signal/video scorer (`python -m somnus.scorer`). |
| `tools/` | `verify_openseize_port.py`: proves the two feature backends agree. |

Seed 0 throughout.

## Pipeline

Source-data locations are configured by environment variable (no defaults):

```bash
export SOMNUS_LOCAL_DIR=/path/to/local/recordings   # flat EDF + scored-CSV corpus
export SOMNUS_BIDS_DIR=/path/to/bids/corpus          # BIDS layout, events.tsv scoring
# optional: SOMNUS_DATA_DIR (training-matrix location),
#           SOMNUS_EXCLUDE_LOCAL / SOMNUS_EXCLUDE_BIDS (subjects to skip)
```

Then, from anywhere (installed or `pip install -e .`):

```bash
python -m somnus.data.datasets --seed 0 --rem-per-lab 20   # released recipe
python -m somnus.train.train   --seed 0    # -> ./results_generalized_seed0/
python -m somnus.train.export  --seed 0    # -> portable JSON artifact
python tools/verify_openseize_port.py      # checks the two feature backends agree
```

| Module / script | Role |
|---|---|
| `somnus.features.openseize_backend` | **Shipped feature computation.** Pure: arrays in → features out. openseize PSD backend; holds the band definitions and cross-site harmonisation tiers. |
| `somnus.features.scipy_backend` | The original scipy implementation, kept as a cross-check and as the fallback when openseize is unavailable. Verified numerically identical. |
| `somnus.data.datasets` | Dataset adapters → per-epoch table, manifest, CV folds. |
| `somnus.train.train` | Logistic regression + HMM decode, leave-one-recording-out CV, metrics, plots. |
| `somnus.train.export` | Writes the dependency-free JSON artifact. |
| `somnus.train.finetune` | Anchored fine-tuning. |
| `tools/verify_openseize_port.py` | Proves the openseize and scipy backends produce the same features. |

### Feature backend

`SOMNUS_FEATURE_BACKEND=openseize` (default) or `scipy`. If openseize is missing
the pipeline falls back to scipy automatically and says so. The two agree to
floating-point noise, so either reproduces the released model; the artifact
records which one produced it in its `feature_backend` field.

Band integration uses `np.trapezoid`. openseize's `metrics.power` integrates by
Simpson's rule instead, which shifts `t1_power_log`/`beta_rel`/`alpha_rel` by
12–13% — available via `INTEGRATION = "simpson"`, but not the default.

## Model

| File | What it is |
|---|---|
| `src/somnus/models/model_somnus_1.0.json` | **The release artifact.** Plain JSON: columns, centring stats, coefficients, transition matrix, priors. ~15 KB, no pickles. Ships in the wheel. |
| `src/somnus/predict.py` | **Self-contained scorer** — numpy/pandas only, imports nothing from the training code. |

```python
from somnus import load_model
from somnus.predict import predict
art = load_model()                         # packaged artifact
labels, proba = predict(art, feature_df)   # feature_df from somnus.data.datasets.featurize()
```

Verified: `somnus.predict` reproduces the trained sklearn pipeline to
floating-point noise (max |Δprob| = 3.3e-16, argmax agreement 1.000000), and the
packaged restructure reproduces the pre-restructure code bit-for-bit (labels and
probabilities identical on the full training matrix; every feature column
identical in both backends).

**The JSON artifact is deliberately pickle-free**: reloading it needs nothing
but numpy/pandas, with no coupling to a training module or sklearn version.

## Headline performance

Released variant `unified_z_noind`, 70 features. Training set: 61,164 epochs —
every lab (five public labs + the local corpus) contributes an equal per-state
share, except REM, where each public lab contributes 20 epochs (the
`--rem-per-lab 20` recipe: REM scoring criteria differ between labs, and a
REM class anchored on one consistent standard measurably outperforms a pooled
one). Tested on held-out public-corpus mice (74 subjects, 1,515,059 labelled
epochs):

| metric | value |
|---|---|
| accuracy | 0.9136 |
| balanced accuracy | 0.8559 |
| κ | 0.8434 |
| REM F1 | 0.7229 |

In-house data, leave-one-recording-out: accuracy 0.9749, balanced 0.9543.

Numbers are from a single training seed. REM is the bottleneck; a full
write-up accompanies the forthcoming preprint.
