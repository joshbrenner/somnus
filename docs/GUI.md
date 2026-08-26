# Somnus GUI

Tabbed desktop app (PySide6) for scoring recordings with `model_somnus_1.0`,
reviewing the result, and fine-tuning on manually corrected labels.

```bash
somnus-gui            # or: python -m somnus.gui
```

Five tabs: **Project → Score → Review & Relabel → Fine-tune → Evaluate**.

## Files

| File | Role |
|---|---|
| `somnus/gui/core.py` | Everything non-GUI: project state, the provenance-aware label store, featurizing, scoring, uncertainty ranking, QC flags, architecture export. Qt-free, so it can be scripted and tested headlessly. |
| `somnus/gui/app.py` | The Qt window: five tabs plus two embedded matplotlib canvases. |

## Tabs

**Project** — name the project on creation, add recordings from a folder, then
tick the ones you want to work with. The Score tab processes exactly what is ticked, 
and fine-tuning uses the ticked recordings that carry manual labels. Selection persists in
`project.json`. Each row shows the paths found for labels, video and tracking,
plus whether velocity is available.

**Score** 

- *Apply HMM temporal smoothing* — on, the decode finds the most likely state
  sequence, so isolated one-epoch flickers get absorbed. Off, you get the raw
  per-epoch model output.
- *Transition resistance* — how hard the decode resists state changes. `0` is
  identical to smoothing off, `1` uses the default transition matrix estimated from
  scored data, `>1` forces longer bouts. If fragmentation is part of your phenotype,
  keep it low!

**Review & Relabel** — a whole-recording hypnogram (x axis in hours) with a
confidence panel underneath: raw per-epoch confidence, an adjustable
rolling-median overlaid (default 1 min), the
low-certainty threshold as a dashed red line, and red ticks marking the epochs
that fail the threshold. Green ticks on the ribbon mark manually reviewed epochs. 
Relabeling happens in the scorer (see below).

**Evaluate** 

- *Compare models* scores each recording with the base model and with a chosen
  fine-tuned model, measuring each against your manual labels only.
- *Architecture breakdown* gives % time, minutes, bout counts, mean/median bout
  durations, transition counts and latencies per recording. Manual labels replace
  the model's wherever they exist, and a `manual_epochs` column says how many.

Both export to CSV.

**Fine-tune** — trains on manual labels only, from the ticked recordings.
λ is either fixed or chosen by cross-validation. Reports held-out before/after,
and only offers to activate the new model if it actually beat the base model out
of sample. Each run writes a new
`models/finetuned_NN.json` so earlier models stay comparable in the Evaluate tab.

## The scorer hand-off

The Review tab writes two files inside the project and launches `python -m somnus.scorer` on them:

- `labels/<rec>_scored.csv` — pre-filled with the model's labels so you correct rather
  than score from scratch. 
- `labels/<rec>_review_meta.csv` — per-epoch `uncertainty`, `confidence` and
  `hmm_smoothed`.

The scorer gains three things from that metadata:

- a **MODEL BELIEF panel** showing mean per-state probability over the visible window,
  mean confidence, and how many visible epochs fall below threshold,
- a **Next low certainty** button (**`u`** forward, **`U`** back, **`[`**/**`]`**
  to adjust the threshold), walking temporally through epochs the model is not sure about,
- a distinct color (`HMM_Smoothed`) for epochs whose label the HMM smoothing
  changed, hatched along the top of the timeline
- a **Confirm brush** (key **6**) that affirms the label
  already present without changing it. Confirmed bins render more opaque in the
  same state color; **Erase** (key **7**) clears the confirmation. This is how
  the scorer records "the model got this right", which is also used in fine tuning.

Smoothed epochs are **excluded from the uncertain queue** by design: smoothing
changing a label is not the same as the model being unsure. They are flagged by
color so you can still scan them.

**A label is marked manual only if it CHANGED, or was confirmed with the Confirm
brush.** The hand-off CSV starts as the model's own predictions, so an untouched 
epoch returns identical — counting those as "confirmed" would turn the model's 
entire output into training targets, a feedback loop that raises apparent confidence while adding no
information. Epochs painted **Artifact** or **Unclear** are recorded as
`manual_excluded` and kept out of fine-tuning entirely.

## Also in `core.py`

- `qc_flags()` — REM not preceded by NREM, sub-8 s REM bouts, very short bouts.
  Hints, never auto-corrections: "unusual" is exactly what a disease model may
  legitimately produce. Reported in the Score tab's summary; no dedicated
  export.
- `architecture()` — % time and minutes per state, bout counts, mean/median bout
  duration, transition counts, latency to first NREM/REM. The paper numbers.
  Exported from the Evaluate tab's *Architecture breakdown*.

## Known gaps

- The Score tab batches the ticked recordings, but there is no resume if it dies
  part-way.
- There is no UI for restoring a snapshot from `labels/history/` — copy the file
  over `labels/<rec>_scored.csv` by hand.
- Corrections are pulled back by pressing **Reload corrections**; the GUI does
  not detect the scorer closing on its own.
