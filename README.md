<h1 align="center">
    <img src="https://github.com/joshbrenner/somnus/raw/master/imgs/logo.png"
    style="width:500px;height:auto;" alt="Somnus"/>
</h1>

### Sleep-state scoring for mouse EEG/EMG (+ video)

Somnus assigns **Wake / NREM / REM** to every 4-second epoch of a mouse
polysomnography recording, using EEG, EMG, and — when available —
video-derived locomotion. It ships a trained model, a batch scorer, a review
interface for correcting the model's output, and a fine-tuning step that
adapts the model to your animals without discarding what it already knows.

The design goal is not maximum accuracy on one rig. It is a model that **runs
on whatever data you have** and **degrades honestly** when something is
missing: features are computed in frequency tiers gated by each recording's
*measured* bandwidth, every feature is z-scored within its own recording, and
temporal structure comes from an HMM/Viterbi decode with a user-facing
"transition resistance" knob. One model scores both a 128 Hz EEG-only
recording and a 5 kHz EEG+EMG+video recording.

## Install

```bash
pip install somnus-eeg
```

## Quickstart — score a recording in Python

```python
from somnus import load_model
from somnus.predict import predict
from somnus.data.datasets import featurize

art = load_model()                      # the packaged v1.0 model
df = featurize({"recording": "myrec", "edf": "path/to/myrec.edf",
                "dataset": "user", "group": "user", "subject": "m1",
                "scored": None, "pkl": None})
labels, proba = predict(art, df)        # per-epoch states + probabilities
```

`predict(..., stickiness=...)` controls the temporal decode: `0` disables
smoothing, `1` uses the transition matrix as estimated, `>1` enforces longer
bouts.

## The desktop application

```bash
somnus-gui
```

A five-tab workflow: **Project** (point it at a folder of recordings) →
**Score** (batch scoring with smoothing controls) → **Review & Relabel**
(whole-recording hypnogram with a confidence trace, plus a signal/video scorer
pre-loaded with the model's labels so you correct rather than score from
scratch) → **Fine-tune** (adapt the model to your corrections) → **Evaluate**
(compare against the base model; export sleep-architecture statistics).

Two invariants are enforced in code: **source data is never written to**, and
**the model never trains on its own output** — every epoch records where its
label came from, and only human-sourced labels can become training targets.

## Fine-tuning, not retraining

The logistic loss on *your* labelled epochs is minimised with a penalty that
pulls the weights toward the existing model. A single strength parameter λ
spans the spectrum from "keep the shipped model" (λ→∞) to "train on my data
alone" (λ→0), and cross-validation on your recordings picks λ — so fine-tune
versus retrain is decided by evidence. On the three subjects the base model
handles worst, balanced accuracy went 0.706 → 0.783; training on those
recordings alone scored 0.642, *worse than not adapting at all*.

## Performance

Tested on held-out mice from a public multi-lab validation corpus —
81 subjects, ~1.52 million labelled epochs — after training on 20,388 epochs:

| metric | value |
|---|---|
| accuracy | 0.906 |
| balanced accuracy | 0.872 |
| Cohen's κ | 0.832 |
| REM F1 | 0.687 |

On in-house 5 kHz recordings (leave-one-recording-out): accuracy 0.975.

## Honest limitations

- **REM is the bottleneck** (F1 0.687 vs >0.90 for Wake/NREM); it needs more
  scored REM epochs, not better features.
- **The default smoothing over-smooths.** If sleep fragmentation is your
  phenotype, lower the transition resistance and check the hypnogram.
- Reported numbers are one training seed; across 5 seeds the same pipeline
  spans 0.884–0.906 (mean ≈0.893).

A full write-up of the evaluation, caveats, and design rationale will
accompany the forthcoming preprint.

## Reproducing / retraining

The training pipeline ships in `somnus.data` / `somnus.train`. Point
`SOMNUS_LOCAL_DIR` / `SOMNUS_BIDS_DIR` at your source corpora and see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the pipeline reference.

## License

BSD 3-Clause. The trained model was fitted in part on a publicly available
multi-lab mouse polysomnography dataset. PSD estimation uses
[openseize](https://github.com/mscaudill/openseize).

## Citation

A preprint is in preparation; until then, please cite this repository.
