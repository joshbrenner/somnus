<h1 align="center">
    <img src="https://github.com/joshbrenner/somnus/raw/master/imgs/logo.png"
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

Features are computed in frequency tiers gated by each recording's *measured* bandwidth. 
Every feature is then z-scored within its own recording. The temporal structure of sleep comes from an 
HMM/Viterbi model with a "transition resistance" knob which allows you to adjust the 
frequency of state transitions. Hence, a single logistic model scores both a 128 Hz EEG-only 
recording and a 5 kHz EEG+EMG+labeled video recording.

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
pre-loaded with the model's labels, so you correct rather than score from
scratch) → **Fine-tune** (adapt the model to your corrections) → **Evaluate**
(compare against the base model; export sleep-architecture statistics).

Two invariants are enforced: **source data is never written to**, and
**the model never trains on its own output** — every epoch records where its
label came from, and only human-sourced labels can become training targets.

## Fine-tuning

**This is the point of the project.** Somnus is not trying to be the best sleep
scorer in the world — commercial scorers beat the base model on clean wild-type
data. It is trying to be the one you can actually adapt when your animals do not
look like anyone else's: a disease model whose sleep architecture is degraded
enough that a fixed commercial scorer fails outright.

The performance loss on labelled epochs is minimized with a penalty that
pulls the weights toward the existing model. We use a parameter λ,
where "keep the shipped model" (λ→∞), and "train on my data
alone" (λ→0). The default λ value is chosen by cross-validation on your recordings,
so fine-tune versus retrain is decided by evidence. Adjust it at your risk -
in fine-tuning tests on three subjects that the base model handled poorly, accuracy with our method went from
0.706 → 0.783, while training on those recordings alone (λ→0) scored 0.642, worse than not adapting at all.

## Performance

Tested on held-out mice from a public multi-lab validation corpus —
74 subjects, ~1.52 million labelled epochs — after training on 61,164 epochs
from six labs:

| metric | value |
|---|---|
| accuracy | 0.914 |
| balanced accuracy | 0.856 |
| Cohen's κ | 0.843 |

On in-house 5 kHz recordings (leave-one-recording-out): accuracy 0.975.

## Limitations to keep in mind

- **REM scoring criteria differ between labs** so a general model necessarily compromises;
  (F1 REM 0.723 vs >0.90 for Wake/NREM). Fine-tuning to your own scoring is the intended remedy.

A full write-up of the evaluation, caveats, and design rationale will
accompany the forthcoming preprint.

## What ships, and what doesn't

The package is the scorer and the tools to adapt it: feature extraction, the
trained model, batch scoring, the review GUI, and fine-tuning. The code that
assembled the original multi-lab training corpus and fitted the released
weights is deliberately **not** included — it encodes choices specific to our
data, and the base model is a starting point rather than the product. The
corpus assembly and training recipe are documented in the forthcoming preprint.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the module reference.

## License

BSD 3-Clause. The trained model was fitted in part on a publicly available
multi-lab mouse polysomnography dataset. PSD estimation uses
[openseize](https://github.com/mscaudill/openseize).

## Citation

A preprint is in preparation; until then, please cite this repository.
