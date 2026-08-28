"""The OpenCV signal/video scorer. Run with ``python -m somnus.scorer <edf>
[scored.csv] [review_meta.csv] [certainty_threshold] [eeg_channels]`` — this is
how the GUI's Review tab launches it. ``eeg_channels`` is a comma-separated
list of channel numbers (e.g. ``0,1,2``); every other channel shows as EMG."""
