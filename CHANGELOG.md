# Changelog

## Unreleased

### Base model 1.1

- Retrained without the 60-s (15-epoch) context features. Each feature keeps
  its per-epoch value and its 12-s (3-epoch) context. Same training data,
  recipe and seed; 42 features instead of 70. `model_somnus_1.1.json`
  replaces `model_somnus_1.0.json`.
- Held-out public corpus (74 mice): κ 0.845 → 0.854, accuracy 0.914 → 0.920,
  REM F1 0.728 → 0.731 (precision 0.74 → 0.77, recall 0.72 → 0.69), balanced
  accuracy 0.858 → 0.855. κ was higher in each of 10 training seeds.
- After fine-tuning on mice with short REM bouts, REM agreement is higher and
  less REM is missed at every label budget tested; false REM calls are fewer
  from 20 to 2,000 labels.
- Models fine-tuned from 1.0 use the removed features; fine-tune again from 1.1.
- A project whose saved model file no longer exists uses the current base model.

### Building a fine-tuning set

- **Build dataset for fine-tuning** (Review tab) sets out 1-minute segments
  from every ticked, scored recording and plays them in the scorer one after
  another. The first round (15 segments) is spread evenly over the model's
  Wake, NREM and REM calls and over the recordings. After each round the
  labels are checked against fixed targets (every recording labelled; at
  least 20 epochs of each state; 3 REM bouts; at least a quarter confirmed or
  imported rather than corrected), shown beside the button as epochs and
  bouts per state.
- Later rounds share their segments between the states in proportion to what
  each still lacks, counting every manual label already in the project,
  imported ones included.
- When REM falls short, Somnus says so and offers to search again for
  REM-like stretches (model REM probability and theta/delta, outside likely
  wake) or to let you go through the recordings yourself.
- In the scorer, the side panel holds Previous and Next for the segments.
  Next confirms every label left in the segment, so agreeing with the model
  needs no Confirm brush, and reads Close on the last segment. Undo clears
  the confirmation again.
- Labels from the scorer come into the project as soon as it closes; the Load
  true labels button is gone. Newer manual labels replace older ones.

### Project tab

- A new project gets its own folder, named after it, inside the folder you
  choose.
- The use, labels and tracking columns each have a tick box per recording and
  one in the header for the whole column. Everything found is used by
  default: recordings are ticked, scoring files found beside them are
  imported as manual labels (untick to take them out again; they can also be
  changed in the scorer), and tracking is used for movement.

### Other changes

- Video frame times recorded as wall-clock (Unix) time are aligned to the EDF
  start, in the features and in the scorer's video. They were previously
  ignored without warning, which left those recordings without movement
  features. Somnus now warns when under 90% of epochs get a movement value.
- The scorer's video seeks by the video's own frame times and matches them to
  the EEG by capture time, so it stays aligned when the two clocks differ.
- Reviewed bins in the scorer are drawn darker, between the previous colour
  and the model's.
- A recording's video is found in any common container (.mp4, .mkv, .mov,
  .avi and others, in any letter case), by the recording's name or else its
  date-time stamp, skipping overlays and clips made from it. The project and
  the scorer use the same rule.
