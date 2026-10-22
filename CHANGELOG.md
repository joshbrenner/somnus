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
