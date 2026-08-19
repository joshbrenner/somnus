# Somnus GUI

Tabbed desktop app (PySide6) for scoring recordings with `model_somnus_1.0`,
reviewing the result, and fine-tuning on corrected labels.

```bash
somnus-gui            # or: python -m somnus.gui
```

Five tabs: **Project → Score → Review & Relabel → Fine-tune → Evaluate**.

## Files

| File | Role |
|---|---|
| `somnus/gui/core.py` | Everything non-GUI: project state, the provenance-aware label store, featurising, scoring, uncertainty ranking, QC flags, architecture export. Qt-free, so it can be scripted and tested headlessly. |
| `somnus/gui/app.py` | The Qt window: four tabs plus two embedded matplotlib canvases. |

## Tabs

**Project** — name the project on creation, add recordings from a folder, then
**tick the ones you want to work with**. The tick-list drives everything
downstream: the Score tab processes exactly what is ticked, and fine-tuning uses
the ticked recordings that carry human labels. Selection persists in
`project.json`. Each row shows the paths found for labels, video and tracking,
plus whether velocity is available.

**Data files are only ever looked for in the folder you point at.** No
parent/sibling/recursive search: a stale timestamps file from elsewhere would
pair positions with the wrong times and, because the cameras dropped frames,
misalign tracking from the EEG by minutes — silently. If coordinates are present
but their timestamps are not, scoring **raises an error** rather than guessing. A
recording with no tracking at all is fine; velocity is simply absent.

**Score** — two controls, both of which matter:

- *Apply HMM temporal smoothing* — on, the decode finds the most likely state
  *sequence*, so isolated one-epoch flickers get absorbed. Off, you get the raw
  per-epoch model output.
- *Transition resistance* — how hard the decode resists state changes. `0` is
  identical to smoothing off, `1` uses the transition matrix as estimated from
  scored data, `>1` forces longer bouts.

  Measured on one recording (2688 epochs): resistance 0 → 366 bouts,
  1 → 49 bouts, 8 → 14 bouts, against 136 in the manual scoring. **The default
  over-smooths that recording**, and accuracy is nearly flat from 0.5 to 4, so
  this knob mostly trades fragmentation against bout preservation. If
  fragmentation is your phenotype, keep it low and check the hypnogram.

**Review & Relabel** — a whole-recording hypnogram (x axis in **hours**) with a
confidence panel underneath: raw per-epoch confidence faint, an adjustable
rolling-median smoothing over it (default 1 min, display only), the
low-certainty threshold as a dashed red line, and red ticks marking the epochs
the scorer's jump will actually visit. Moving the threshold redraws all of it, so
you can see what the setting includes before opening the scorer. Green ticks on
the ribbon mark human-reviewed epochs. Relabelling happens in **your own
scorer**, opened by a button — see below.

**Evaluate** — two reports over the ticked recordings:

- *Compare models* scores each recording with the base model and with a chosen
  fine-tuned model, measuring both **against your human labels only**. Rows are
  flagged `in_sample` when the compared model was fine-tuned on that recording,
  because those rows measure fit rather than generalisation — the distinction is
  easy to miss and it flatters the fine-tuned model.
- *Architecture breakdown* gives % time, minutes, bout counts, mean/median bout
  durations, transition counts and latencies per recording. Human labels replace
  the model's wherever they exist, and a `human_epochs` column says how many.

Both export to CSV.

**Fine-tune** — trains on human-sourced labels only, from the ticked recordings.
λ is either fixed or chosen by cross-validation. Reports held-out before/after
and a forgetting check, and only offers to activate the new model if it actually
beat the base model out of sample. Each run writes a new
`models/finetuned_NN.json` so earlier models stay comparable in the Evaluate tab.

## The scorer hand-off

The Review tab writes two files **inside the project** and launches
`python -m somnus.scorer` on them:

- `labels/<rec>_scored.csv` — the one-hot layout the scorer already reads,
  pre-filled with the model's labels so you correct rather than score from
  scratch. Pointing the scorer here matters: its `save_csv()` overwrites whatever
  path it is given, and the default would be beside your EDF.
- `labels/<rec>_review_meta.csv` — per-epoch `uncertainty`, `confidence` and
  `hmm_smoothed`.

The scorer gains three things from that metadata (and runs exactly as before
without it):

- a **MODEL BELIEF panel** (replacing the old spectral-index panel) showing mean
  per-state probability over the visible window, mean confidence, and how many
  visible epochs fall below threshold,
- a **Next low certainty** button (**`u`** forward, **`U`** back, **`[`**/**`]`**
  to adjust the threshold), walking **temporally**,
- a distinct colour (`HMM_Smoothed`) for epochs whose label the HMM smoothing
  changed, hatched along the top of the timeline,
- a **Confirm brush** (key **6**; Erase is now **7**) that affirms the label
  already present without changing it. Confirmed bins render more opaque in the
  same state colour; **Erase** clears the confirmation. This is the only way a
  human can record "the model got this right" — an untouched epoch is identical to
  a checked one once written to a CSV, so without it that judgement is lost.

Smoothed epochs are **excluded from the uncertain queue** by design: smoothing
changing a label is not the same as the model being unsure. They are flagged by
colour so you can still scan them.

**A label becomes human only if it CHANGED, or was confirmed with the Confirm
brush.** The hand-off CSV
starts as the model's own predictions, so an untouched epoch returns identical —
counting those as "confirmed" would turn the model's entire output into training
targets, a feedback loop that raises apparent confidence while adding no
information. Epochs painted **Artifact** or **Unclear** are recorded as
`human_excluded` and kept out of fine-tuning entirely.

**Changing a label and then changing it back revokes the human flag.** Each row
keeps a `model_state` column recording what the classifier originally predicted,
so an edit that ends up matching it again is reverted to `source=model`: the
epoch stops showing as human-reviewed and stops being a fine-tuning target.
Without that the row would stay green forever and would feed the model its own
prediction as ground truth.

**Scoring is versioned.** The scorer overwrites its CSV in place on save, so
`write_viewer_bundle` snapshots the previous one into `labels/history/` first,
keeping the most recent `MAX_SCORING_VERSIONS` (5) and pruning older ones. The
live file is always the newest; the snapshots are there for when a mis-paint or a
bad drag needs undoing.

## Two invariants the code enforces

**1. Source data is read-only.** Every write goes under the project directory
(`labels/`, `cache/`, `models/`, `project.json`). Nothing is written beside a
recording.

**2. A model write never overwrites a human label.** Every epoch carries a
`source`: `model`, `human_confirmed`, `human_corrected`, `human_imported`, or
`human_excluded`. Re-scoring skips all of those; fine-tuning trains only on the
human ones, never on `human_excluded`.

The second one is not bookkeeping for its own sake — training on the model's own
predictions is a feedback loop that raises apparent confidence while adding no
information, and it is very easy to do by accident once scoring and relabelling
live in the same tool.

## Also in `core.py`, not yet surfaced in the UI

- `qc_flags()` — REM not preceded by NREM, sub-8 s REM bouts, very short bouts.
  Hints, never auto-corrections: "unusual" is exactly what a disease model may
  legitimately produce.
- `architecture()` — % time and minutes per state, bout counts, mean/median bout
  duration, transition counts, latency to first NREM/REM. The paper numbers.

Both are computed and reported in the Score tab's summary; a dedicated export
button is still to come.

## Known gaps

- The Score tab batches the ticked recordings, but there is no resume if it dies
  part-way.
- There is no UI for restoring a snapshot from `labels/history/` — copy the file
  over `labels/<rec>_scored.csv` by hand.
- The scorer opens as a separate window rather than embedded. Embedding would
  mean replacing its `cv2.imshow`/`waitKey` loop with a Qt timer and event
  forwarding; `render_composite` returns a plain numpy image so it is feasible,
  but it rewrites the one part of that file that cannot survive embedding.
- Corrections are pulled back by pressing **Reload corrections**; the GUI does
  not yet detect the scorer closing on its own.
- No Models tab: the active model is set from the fine-tune dialog, and the
  Evaluate tab's dropdown is the only place to browse `models/`.
- Fine-tuning is implemented in `somnus.train.finetune`.
