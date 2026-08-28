<h1 align="center">
    <img src="https://github.com/joshbrenner/somnus/raw/main/imgs/logo.png"
    style="width:500px;height:auto;" alt="Somnus"/>
</h1>

### Sleep-state scoring for mouse EEG/EMG (+ video)

Somnus assigns **Wake / NREM / REM** to every 4-second epoch of a mouse
polysomnography recording, using EEG, EMG, and — when available —
video-derived locomotion. It includes a trained logistic model, a batch scorer, 
a review interface for correcting the model's output, and a fine-tuning step that
adapts the model to your dataset.

The design goal is not maximum accuracy on one rig. It is a system that **runs
on whatever data you have**, and that can be readily adapted to disease models
with degraded sleep architecture where heavier commercial scorers may fail altogether.

Features are computed in frequency tiers gated by each recording's measured bandwidth. 
Every feature is then z-scored within its own recording. The temporal structure of sleep comes from an 
HMM/Viterbi model with a "transition resistance" knob which allows you to adjust the 
frequency of state transitions. 

## Install

Install somnus in a virtual environment using conda, with a python version > 3.12:
```bash
conda create -n somnus-eeg python=3.13 -y
conda activate somnus-eeg
pip install somnus-eeg
```
## The desktop GUI

To start the GUI, use:

```bash
somnus-gui
```

The GUI includes a five-tab workflow: **Project** (point it at a folder of recordings) →
**Score** (batch scoring with smoothing controls) → **Review & Relabel**
(whole-recording hypnogram with a confidence trace, plus a signal/video scorer
pre-loaded with the model's labels, so you correct rather than score from
scratch) → **Fine-tune** (adapt the model to your corrections) → **Evaluate**
(compare against the base model; export sleep-architecture statistics).

Two notes: **source data is never written to**, and
**the model never trains on its own output** — every epoch records where its
label came from, and only manually sourced labels can become training targets.

## Alternative -- Score a recording in Python

```python
from somnus import load_model
from somnus.predict import predict
from somnus.data.datasets import featurize

art = load_model()                      # the packaged v1.0 model
df = featurize({"recording": "myrec", "edf": "path/to/myrec.edf",
                "dataset": "user", "group": "user", "subject": "m1",
                "scored": None, "pkl": None},
               eeg_chan=[0, 1, 2], emg_chan=3)
labels, proba = predict(art, df)        # per-epoch states + probabilities
```

`eeg_chan` and `emg_chan` set which channels are which, by number or by name.
When several EEG channels are named, the best snr is used.

`predict(..., stickiness=...)` controls the temporal decode: `0` disables
smoothing, `1` uses the transition matrix as estimated, `>1` enforces longer
bouts.

## Video tracking (optional)

If a recording has video with the animal's position tracked, Somnus adds a
locomotion feature. Pass the tracking file as `"pkl"` in the `featurize()`
entry, or put it beside the EDF and the GUI will find it.

One row per video frame, in frame order. Accepted formats:

| File | Contents |
|---|---|
| `.csv` | A DeepLabCut export, or any table with `x` and `y` columns |
| `.h5` | A DeepLabCut export (needs `pip install tables`) |
| `.pkl` | An `(n_frames, 2)` array of x/y, or a dict with a `coordinates` key |

A DeepLabCut file usually tracks many bodyparts. Somnus uses the one named
**`mouse_center`**; if the file tracks exactly one bodypart, it uses that.
Otherwise it stops rather than guess which point represents the animal — export
a `mouse_center` bodypart, or hand it a plain two-column `x,y` table instead.

**Filter low-confidence points yourself.** Somnus does not drop them, so a badly
tracked frame reads as real movement. Replace rejected points with `NaN` before
passing the file.

**Frame times are strongly suggested.** Somnus looks for a
`*_timestamps.npy` beside the tracking file with exactly one timestamp per row.
Cameras drop frames, so assuming a constant frame rate can misplace positions by
minutes; if no timestamps file is found, Somnus requires confirmation.

## Fine-tuning

The performance loss on labeled epochs is minimized with a penalty that
pulls the weights toward the existing model. We use a parameter λ,
where "keep the shipped model" (λ→∞), and "train on my data
alone" (λ→0). The default λ value is chosen by cross-validation on your recordings,
so fine-tune versus retrain is decided by evidence. 

## Performance

Tested on held-out mice from a public multi-lab validation corpus —
74 subjects, ~1.52 million labeled epochs — after training on 61,164 epochs
from six labs:

| metric | value |
|---|---|
| accuracy | 0.914 |
| balanced accuracy | 0.856 |
| Cohen's κ | 0.843 |

On in-house 5 kHz recordings (leave-one-recording-out): accuracy 0.975.

## Limitations to keep in mind

- **REM scoring criteria differ between labs** so a general model necessarily compromises;
  (F1 REM 0.723 vs >0.90 for Wake/NREM). Fine-tuning to your own data is the intended remedy!

A full write-up of the evaluation, caveats, and design rationale will
accompany the forthcoming preprint.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the module reference.

## License

BSD 3-Clause. The trained model was fitted in part on a publicly available
multi-lab mouse polysomnography dataset. PSD estimation uses
[openseize](https://github.com/mscaudill/openseize).

## Citation

A preprint is in preparation; until then, please cite this repository.
