import os
import sys
import time
import re
import cv2
import numpy as np
import pandas as pd
import mne
from mne.filter import filter_data
import tkinter as tk
from tkinter import filedialog

from somnus.scorer.video_handler import VideoHandler
from somnus.scorer.ui_rendering import render_composite, WINDOW_NAME, MENU_HEADERS, MENU_ITEMS
from somnus.scorer.signal_utils import compute_psd

def find_associated_video(edf_path):
    """Look beside the recording for the video that belongs to it."""
    directory = os.path.dirname(os.path.abspath(edf_path))
    basename = os.path.basename(edf_path)
    candidates = []
    
    # 1. Search by timestamp match
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', basename)
    if match:
        timestamp = match.group(0)
        for file in os.listdir(directory):
            if file.lower().endswith(('.mp4', '.avi', '.mkv')) and timestamp in file:
                candidates.append(os.path.join(directory, file))
                
    # 2. Search by base filename match if no timestamps match
    if not candidates:
        base_no_ext = os.path.splitext(basename)[0]
        for file in os.listdir(directory):
            if file.lower().endswith(('.mp4', '.avi', '.mkv')) and base_no_ext in file:
                candidates.append(os.path.join(directory, file))
                
    if not candidates:
        return None
        
    # 3. Pick the largest file
    best_video = None
    max_size = -1
    
    for candidate_path in candidates:
        try:
            size = os.path.getsize(candidate_path)
            if size > max_size:
                max_size = size
                best_video = candidate_path
        except OSError:
            pass
            
    return best_video

class SleepReviewState:
    """Everything the scorer window is currently showing and holding."""
    def __init__(self, edf_path, csv_path, sfreq, ch_names, n_times, screen_w, screen_h,
                 review_meta_path=None, certainty_threshold=0.80):
        """Load the recording, its scoring, and the model output if present."""
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.csv_path = csv_path
        self.n_times = n_times
        
        # --- DYNAMIC LAYOUT MATH (HEIGHT & WIDTH SAFEGUARDS) ---
        self.video_w = 450
        self.side_w = 200
        
        # Leave a small 20px horizontal margin to prevent edge-clipping
        self.eeg_w = max(1000, screen_w - 20)
        
        # Calculate dynamic height based on available physical screen space.
        # Top Menu = 30px, Bottom Row ~ 250px (SHRUNK). 
        # Leave ~150px for the Mac Menu Bar, Window Title, and Dock.
        available_height = screen_h - 30 - 250 - 150
        self.eeg_h = max(300, available_height)
        
        self.current_time_sec = 0.0
        self.window_width_sec = 10.0
        self.playback_offset_sec = 0.0 
        self.paused = False
        self.pending_jump_time = None 
        
        self.active_menu = None
        self.active_brush = None
        self.paint_mode = 'Paint'
        self.is_mouse_down = False
        self.range_start_sec = None
        
        self.update_counter = 0 
        
        self.df = pd.read_csv(csv_path)
        t = self.df['Time_sec'].to_numpy(dtype=float)
        self.bin_step = float(t[1] - t[0]) if len(t) > 1 else 0.5
        
        # Default Brush Size
        self.brush_size_sec = max(self.bin_step, 1.0)
        
        # Older scoring files predate these two columns, so add them empty
        # rather than fail. 'Unclear' marks an epoch as ambiguous, 'Confirmed'
        # marks one the user has explicitly agreed with.
        if 'Unclear' not in self.df.columns:
            self.df['Unclear'] = 0
        # Manual affirmation of the label already present. Round-trips through
        # save_csv() for free, because that only drops 'State' and 'Bout_ID'.
        if 'Confirmed' not in self.df.columns:
            self.df['Confirmed'] = 0
        self.df['Confirmed'] = self.df['Confirmed'].fillna(0).astype(int)

        # How sure the model was about each epoch, written by the Somnus app.
        # Absent when the scorer is opened on its own, in which case the
        # model-review parts of the display are simply not drawn.
        self.review_meta = None
        self.uncertain_btn = None
        # An epoch counts as low certainty below this. Adjustable with [ and ].
        self.certainty_threshold = float(certainty_threshold)
        self.epoch_sec = 4.0
        self._load_review_meta(review_meta_path)

        self._calculate_bouts()

    # ------------------------------------------------------ model review extras
    def _load_review_meta(self, path):
        """Read how sure the model was about each epoch, if the app supplied it."""
        if not path or not os.path.exists(path):
            return
        try:
            m = pd.read_csv(path)
        except Exception as e:
            print(f"[review] could not read {path}: {e}")
            return
        need = {'epoch', 'uncertainty', 'confidence', 'hmm_smoothed'}
        if not need.issubset(m.columns):
            print(f"[review] {os.path.basename(path)} missing {need - set(m.columns)}")
            return
        m = m.sort_values('epoch').reset_index(drop=True)
        if 't_start' in m.columns and len(m) > 1:
            self.epoch_sec = float(m['t_start'].iloc[1] - m['t_start'].iloc[0]) or 4.0
        self.review_meta = m
        self._unc = m['uncertainty'].to_numpy(dtype=float)
        self._conf = m['confidence'].to_numpy(dtype=float)
        self._sm = m['hmm_smoothed'].to_numpy(dtype=int).astype(bool)
        self._reviewed = (m['reviewed'].to_numpy(dtype=int).astype(bool)
                          if 'reviewed' in m.columns
                          else np.zeros(len(m), dtype=bool))
        n_flag = int(self._sm.sum())
        n_low = int(self._low_mask().sum())
        print(f"[review] loaded metadata for {len(m)} epochs: {n_flag} "
              f"HMM-smoothed, {n_low} below the {self.certainty_threshold:.2f} "
              f"certainty threshold")

    def _epoch_of(self, t_sec):
        """Which epoch covers this moment in the recording."""
        if self.review_meta is None:
            return None
        i = int(t_sec // self.epoch_sec)
        return i if 0 <= i < len(self.review_meta) else None

    def _low_mask(self):
        """Which epochs the 'Next low certainty' button will visit.

        Epochs the model was unsure about. Ones the smoothing changed are left
        out -- that is not the same as the model being unsure, and they already
        have their own colour in the timeline.
        """
        return ((self._conf < self.certainty_threshold) & ~self._sm
                & ~self._reviewed)

    def n_low_certainty(self):
        """How many epochs the button would visit, counted the way it picks them."""
        if self.review_meta is None:
            return 0
        return int(self._low_mask().sum())

    def window_belief(self, t0, t1):
        """What the model believes about the stretch currently on screen.

        Averages each state's probability over the visible epochs, and counts
        how many of them fall below the certainty threshold.
        """
        if self.review_meta is None:
            return {}, 0.0, 0, 0
        lo = max(0, int(t0 // self.epoch_sec))
        hi = min(len(self.review_meta), int(np.ceil(t1 / self.epoch_sec)))
        if hi <= lo:
            return {}, 0.0, 0, 0
        probs = {}
        for s in ('Wake', 'NREM', 'REM'):
            col = f'p_{s}'
            probs[s] = (float(np.nanmean(self.review_meta[col].to_numpy()[lo:hi]))
                        if col in self.review_meta.columns else 0.0)
        conf = float(np.nanmean(self._conf[lo:hi]))
        n_low = int(self._low_mask()[lo:hi].sum())
        return probs, conf, n_low, hi - lo

    def n_smoothed_in(self, t0, t1):
        """How many epochs in this stretch had their label changed by smoothing."""
        if self.review_meta is None:
            return 0
        lo = max(0, int(t0 // self.epoch_sec))
        hi = min(len(self._sm), int(np.ceil(t1 / self.epoch_sec)))
        return int(self._sm[lo:hi].sum()) if hi > lo else 0

    def nudge_threshold(self, delta):
        """Move the certainty threshold up or down."""
        self.certainty_threshold = float(np.clip(
            self.certainty_threshold + delta, 0.0, 1.0))
        self.update_counter += 1          # force the panels to redraw
        print(f"certainty threshold -> {self.certainty_threshold:.2f} "
              f"({self.n_low_certainty()} epochs below)")

    def smoothed_spans(self, t0, t1):
        """Which stretches on screen had their label changed by the smoothing."""
        if self.review_meta is None:
            return []
        lo = max(0, int(t0 // self.epoch_sec))
        hi = min(len(self._sm), int(t1 // self.epoch_sec) + 1)
        return [(i * self.epoch_sec, (i + 1) * self.epoch_sec)
                for i in range(lo, hi) if self._sm[i]]

    def mark_reviewed(self, t_sec):
        """Remember that the user has visited this epoch, so it stops coming up."""
        i = self._epoch_of(t_sec)
        if i is not None:
            self._reviewed[i] = True

    def jump_to_next_uncertain(self, direction=1):
        """Jump to the next epoch the model was unsure about.

        Moves forward through the recording in time rather than jumping to the
        single worst epoch, so reviewing follows the recording the way scoring
        does. Wraps round at the end.
        """
        if self.review_meta is None:
            print("No model review metadata loaded.")
            return
        cand = np.flatnonzero(self._low_mask())
        if cand.size == 0:
            print(f"No epochs below the {self.certainty_threshold:.2f} "
                  f"certainty threshold.")
            return

        # Anchor on the CENTER of the visible window, not its left edge. 
        if abs(self.playback_offset_sec) > 1e-9:
            anchor = self.current_time_sec + self.playback_offset_sec
        else:
            anchor = self.current_time_sec + self.window_width_sec / 2.0
        tol = self.epoch_sec / 2.0

        times = cand * self.epoch_sec
        if direction >= 0:
            nxt = cand[times > anchor + tol]
            target = int(nxt[0]) if nxt.size else int(cand[0])   # wrap to start
            if not nxt.size:
                print("(wrapped to the first low-certainty epoch)")
        else:
            prev = cand[times < anchor - tol]
            target = int(prev[-1]) if prev.size else int(cand[-1])
            if not prev.size:
                print("(wrapped to the last low-certainty epoch)")
        t = target * self.epoch_sec
        self.current_time_sec = max(0.0, t - self.window_width_sec / 2.0)
        self.pending_jump_time = t
        self.mark_reviewed(t)
        pos = int(np.searchsorted(cand, target)) + 1
        print(f"-> epoch {target} (t={t:.0f}s) confidence="
              f"{self._conf[target]:.2f}  [{pos}/{cand.size} below "
              f"{self.certainty_threshold:.2f}]")

    def _calculate_bouts(self):
        """Group runs of the same state into bouts, for the navigation buttons."""
        # One painted column per state, so read them back into a single label.
        # 'Unknown' is not a brush and has no column: it is what a bin gets when
        # nothing has been painted on it at all.
        conditions = [
            self.df['Artifact'] == 1,
            self.df['REM'] == 1,
            self.df['NREM'] == 1,
            self.df['Wake'] == 1,
            self.df['Unclear'] == 1
        ]
        choices = ['Artifact', 'REM', 'NREM', 'Wake', 'Unclear']
        self.df['State'] = np.select(conditions, choices, default='Unknown')
        self.df['Bout_ID'] = (self.df['State'] != self.df['State'].shift()).cumsum()
        
        self.bouts = self.df.groupby('Bout_ID').agg(
            State=('State', 'first'),
            Start_Time=('Time_sec', 'min'),
            End_Time=('Time_sec', 'max')
        ).reset_index()

    def jump_to_next(self, target_state):
        """Move to the next or previous bout of a given state."""
        exact_time = self.current_time_sec + self.playback_offset_sec
        future_bouts = self.bouts[(self.bouts['Start_Time'] > exact_time) & 
                                  (self.bouts['State'] == target_state)]
        if not future_bouts.empty:
            target_boundary = future_bouts.iloc[0]['Start_Time']
            half_window = self.window_width_sec / 2.0
            self.current_time_sec = max(0.0, target_boundary - half_window)
            self.pending_jump_time = target_boundary
        else:
            print(f"No future {target_state} bouts found.")

    def jump_to_adjacent_epoch(self, direction='next'):
        """Step to the next or previous bout boundary.
        """
        exact_time = self.current_time_sec + self.playback_offset_sec
        current_bout_mask = self.bouts['Start_Time'] <= exact_time
        
        if current_bout_mask.any():
            current_bout_start = self.bouts[current_bout_mask].iloc[-1]['Start_Time']
        else:
            current_bout_start = 0.0

        if direction == 'next':
            future_bouts = self.bouts[self.bouts['Start_Time'] > current_bout_start + 0.1]
            if not future_bouts.empty:
                target_boundary = future_bouts.iloc[0]['Start_Time']
            else:
                return 
        else:
            past_bouts = self.bouts[self.bouts['Start_Time'] < current_bout_start - 0.1]
            if not past_bouts.empty:
                target_boundary = past_bouts.iloc[-1]['Start_Time']
            else:
                target_boundary = 0.0

        half_window = self.window_width_sec / 2.0
        self.current_time_sec = max(0.0, target_boundary - half_window)
        self.pending_jump_time = target_boundary

    def paint_state(self, start_sec, end_sec, save=True):
        """Apply the active brush to every bin in this stretch of time."""
        if not self.active_brush: return

        # Safely capture all overlapping bins
        mask = (self.df['Time_sec'] + self.bin_step > start_sec) & (self.df['Time_sec'] < end_sec)
        if not mask.any(): return

        if self.active_brush == 'Confirm':
            # Affirms the label already there WITHOUT changing it. This is the
            # only way the user can say "the model got this right", because a
            # label left untouched is indistinguishable from one never looked at
            # once it has been written to a CSV -- so an unconfirmed epoch is
            # never treated as reviewed, and never becomes a training target.
            self.df.loc[mask, 'Confirmed'] = 1
        else:
            # The paintable states are mutually exclusive, so clear them all
            # before setting the new one. Erase clears and sets nothing.
            for state_name in ['Artifact', 'Wake', 'NREM', 'REM', 'Unclear']:
                self.df.loc[mask, state_name] = 0

            # If the brush is anything other than Erase, assign the 1
            if self.active_brush != 'Erase':
                self.df.loc[mask, self.active_brush] = 1

            # Any state change (and Erase) drops a prior confirmation: that
            # confirmation referred to the label being replaced.
            self.df.loc[mask, 'Confirmed'] = 0

        self._calculate_bouts()
        self.update_counter += 1

        if save:
            self.save_csv()

    def n_confirmed(self):
        """How many epochs the user has explicitly confirmed."""
        return int(self.df['Confirmed'].sum()) if 'Confirmed' in self.df else 0

    def save_csv(self):
        """Write the scoring back to its file."""
        export_df = self.df.drop(columns=['State', 'Bout_ID'])
        export_df.to_csv(self.csv_path, index=False)
        print(f"Successfully saved updated scoring to {self.csv_path}")

def on_mouse(event, x, y, flags, param):
    """Handle every click and drag in the window."""
    state = param
    
    # --- 0. DROPDOWN MENU INTERCEPT ---
    if state.active_menu is not None:
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start = MENU_HEADERS[state.active_menu][0]
            items = MENU_ITEMS[state.active_menu]
            dd_w = 150
            dd_h = len(items) * 30
            
            if x_start <= x <= x_start + dd_w and 30 <= y <= 30 + dd_h:
                idx = (y - 30) // 30
                if 0 <= idx < len(items):
                    item = items[idx]
                    if item == 'Save CSV': state.save_csv()
                    elif item in ['Artifact', 'Wake', 'NREM', 'REM', 'Unclear', 'Confirm', 'Erase']: state.active_brush = item
                    elif item == '4 sec': state.window_width_sec = 4.0
                    elif item == '10 sec': state.window_width_sec = 10.0
                    elif item == '30 sec': state.window_width_sec = 30.0
                    elif item == '1 min': state.window_width_sec = 60.0
                    elif item == 'Paint Mode': state.paint_mode = 'Paint'
                    elif item == 'Range Mode': state.paint_mode = 'Range'
                    elif item == 'Min Size': state.brush_size_sec = max(state.bin_step, 1.0)
                    elif item == 'Decrease (-)': state.brush_size_sec = max(state.bin_step, state.brush_size_sec - state.bin_step)
                    elif item == 'Increase (+)': state.brush_size_sec += state.bin_step
            
            state.active_menu = None
        return 

    if event == cv2.EVENT_LBUTTONDOWN:
        # --- 1. TOP MENU BAR (y <= 30) ---
        if y <= 30: 
            for menu_name, (x1, x2) in MENU_HEADERS.items():
                if x1 <= x <= x2:
                    state.active_menu = menu_name
                    break
            return
                
        # --- 2. EEG PAINTING AREA ---
        elif 30 < y <= 30 + state.eeg_h:
            if state.active_brush is not None:
                x_eeg = x  
                click_time = state.current_time_sec + (x_eeg / state.eeg_w) * state.window_width_sec
                
                if state.paint_mode == 'Paint':
                    state.is_mouse_down = True
                    state.paint_state(click_time, click_time + state.brush_size_sec, save=False)
                    
                elif state.paint_mode == 'Range':
                    if state.range_start_sec is None:
                        state.range_start_sec = click_time
                    else:
                        start_t = min(state.range_start_sec, click_time)
                        end_t = max(state.range_start_sec, click_time)
                        state.paint_state(start_t, end_t, save=True)
                        state.range_start_sec = None

        # --- 3. SIDE NAVIGATION PANEL (UPDATED FOR 2-COLUMN GRID) ---
        elif y > 30 + state.eeg_h and x >= state.eeg_w - state.side_w:
            y_adj = y - (30 + state.eeg_h) 
            x_adj = x - (state.eeg_w - state.side_w)
            
            btn = getattr(state, 'uncertain_btn', None)
            if btn is not None:
                bx1, by1, bx2, by2 = btn
                if bx1 <= x_adj <= bx2 and by1 <= y_adj <= by2:
                    state.jump_to_next_uncertain()
                    return

            valid_labels = ['Wake', 'NREM', 'REM']
            for idx, label in enumerate(valid_labels):
                row = idx // 2
                col = idx % 2
                
                x_start = 10 + col * 92
                x_end = x_start + 85
                y_start = 80 + row * 50
                y_end = y_start + 40
                
                if x_start <= x_adj <= x_end and y_start <= y_adj <= y_end:
                    state.jump_to_next(label)
                    break

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.is_mouse_down and state.paint_mode == 'Paint' and state.active_brush is not None:
            if 30 < y <= 30 + state.eeg_h:
                x_eeg = x
                click_time = state.current_time_sec + (x_eeg / state.eeg_w) * state.window_width_sec
                state.paint_state(click_time, click_time + state.brush_size_sec, save=False)

    elif event == cv2.EVENT_LBUTTONUP:
        if state.is_mouse_down:
            state.is_mouse_down = False
            state.save_csv()


def review_sleep(edf_path, csv_path, video_path, screen_w, screen_h,
                 review_meta_path=None, certainty_threshold=0.80,
                 eeg_idx=None):
    """Run the scorer: open the recording and loop until the user quits."""
    print(f"Lazy Loading {os.path.basename(edf_path)}...")
    mne.set_log_level('ERROR')

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    sfreq, ch_names, n_times = raw.info['sfreq'], raw.ch_names, raw.n_times

    if eeg_idx:
        eeg_idx = [i for i in eeg_idx if 0 <= i < len(ch_names)]
    num_eeg = 3
    if eeg_idx:
        eeg_ch_names = [ch_names[i] for i in eeg_idx]
        emg_ch_names = [ch for i, ch in enumerate(ch_names) if i not in eeg_idx]
        raw.reorder_channels(eeg_ch_names + emg_ch_names)
        ch_names = raw.ch_names
        num_eeg = len(eeg_ch_names)
        print(f"Channels from the project: EEG {eeg_ch_names}, "
              f"EMG {emg_ch_names}")
    elif len(ch_names) != 4:
        num_eeg = max(1, len(ch_names) - 1)
        interactive = sys.stdin is not None and sys.stdin.isatty()
        if not interactive:
            print(f"{len(ch_names)} channels; showing the first {num_eeg} as "
                  f"EEG and {ch_names[-1]!r} as EMG. Start the scorer from a "
                  f"terminal to choose the channels yourself.")
        else:
            print(f"\nFound {len(ch_names)} channels in this recording.")
            for i, ch in enumerate(ch_names):
                print(f"[{i+1}] {ch}")

            while True:
                try:
                    eeg_input = input("\nEnter the numbers corresponding to the EEG channels (comma-separated, e.g., 1, 2): ")
                    eeg_indices = [int(x.strip()) - 1 for x in eeg_input.split(',')]
                    if any(i < 0 or i >= len(ch_names) for i in eeg_indices):
                        print("Invalid selection. Please choose numbers from the list above.")
                        continue

                    eeg_ch_names = [ch_names[i] for i in eeg_indices]
                    emg_ch_names = [ch for i, ch in enumerate(ch_names) if i not in eeg_indices]

                    print(f"\nAssigned EEG: {eeg_ch_names}")
                    print(f"Assigned EMG: {emg_ch_names}\n")

                    ordered_ch_names = eeg_ch_names + emg_ch_names
                    raw.reorder_channels(ordered_ch_names)
                    ch_names = raw.ch_names
                    num_eeg = len(eeg_ch_names)
                    break
                except EOFError:
                    print(f"No answer; showing the first {num_eeg} channels as "
                          f"EEG and {ch_names[-1]!r} as EMG.")
                    break
                except ValueError:
                    print("Please enter valid comma-separated numbers.")

    state = SleepReviewState(edf_path, csv_path, sfreq, ch_names, n_times,
                             screen_w, screen_h, review_meta_path=review_meta_path,
                             certainty_threshold=certainty_threshold)
    
    print("Calculating initial voltage scale and EMG baseline...")
    init_samples = min(int(10 * sfreq), n_times)
    init_data = raw[0:num_eeg, 0:init_samples][0]
    if init_data.size > 0:
        v_range = np.max(init_data) - np.min(init_data)
        if v_range == 0: v_range = 1.0
    else:
        v_range = 1.0
        
    # Calculate the initial EMG log baseline for relative plotting
    if len(ch_names) > num_eeg:
        init_emg_data = raw[num_eeg:, 0:init_samples][0]
        if init_emg_data.shape[1] > sfreq:
            _, init_psd_emg = compute_psd(init_emg_data, sfreq)
            init_emg_log = np.mean(init_psd_emg) / 10.0 if init_psd_emg is not None else -10.0
        else:
            init_emg_log = -10.0
    else:
        init_emg_log = -10.0
    
    # Only apply a 200 Hz low-pass if the sampling rate captures frequencies above 200 Hz
    safe_h_freq = 200.0 if (sfreq / 2.0) > 200.0 else None
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)    
    # Force coordinates to 0, 40 to stay clear of the macOS menu bar
    cv2.moveWindow(WINDOW_NAME, 0, 40)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, param=state)

    # When the Somnus app launches this, the scoring CSV it hands us lives in
    # <project>/labels/, so the project's cache is its sibling. Opened on its
    # own there is no project, and nothing is written to disk.
    labels_dir = os.path.dirname(os.path.abspath(csv_path))
    cache_dir = os.path.join(os.path.dirname(labels_dir), "cache") \
        if os.path.basename(labels_dir) == "labels" else None
    video_engine = VideoHandler(video_path, cache_dir=cache_dir)
    
    eeg_gain = 1.0
    emg_gain = 1.0
    eeg_offset = 0.0
    first_run = True
    last_wall_time = time.time()

    print("\nControls:")
    print(" CLICK (Menu) : Access File, Brush, View, Mode, and Size options")
    print(" DRAG (EEG)  : Continuous 'Paint' mode (Updates CSV on mouse release)")
    print(" CLICK (EEG) : Point-A to Point-B 'Range' mode")
    print(" 1-7         : Select brush (6 = Confirm, 7 = Erase)")
    print(" ENTER       : Save scoring to CSV")
    print(" u / U       : Next / previous low-certainty epoch")
    print(" [ / ]       : Lower / raise the certainty threshold")
    print(" SPACE       : Pause / Play video loop")
    print(" L/R ARROW   : Skip and center to next / previous bout boundary")
    print(" A / D       : Smooth step timeline backward / forward (scales with view)")
    print(" W / S       : Adjust EEG Y-Offset")
    print(" = / -       : Adjust EEG Gain")
    print(" SHIFT + =/- : Adjust EMG Gain")
    print(" ESC         : Quit")

    while True:
        # Real elapsed time, capped at 0.1 s to prevent huge jumps
        current_wall_time = time.time()
        dt = min(current_wall_time - last_wall_time, 0.1)
        last_wall_time = current_wall_time
        
        if state.pending_jump_time is not None:
            state.playback_offset_sec = (state.pending_jump_time
                                         - state.current_time_sec)
            state.pending_jump_time = None

        if not state.paused:
            state.playback_offset_sec += dt
            if state.playback_offset_sec >= state.window_width_sec:
                state.playback_offset_sec = 0.0

        exact_time = state.current_time_sec + state.playback_offset_sec
        frame = video_engine.get_frame_at_time(exact_time)

        start_idx = int(state.current_time_sec * sfreq)
        end_idx = int((state.current_time_sec + state.window_width_sec) * sfreq)
        
        pad_sec = 2.0
        pad_samples = int(pad_sec * sfreq)
        
        fetch_start = max(0, start_idx - pad_samples)
        fetch_end = min(n_times, end_idx + pad_samples)
        
        padded_slice = raw[:, fetch_start:fetch_end][0]
        
        if padded_slice.shape[1] == 0:
            state.current_time_sec = max(0.0, state.current_time_sec - state.window_width_sec)
            continue
            
        padded_slice = filter_data(padded_slice, sfreq, l_freq=0.5, h_freq=safe_h_freq, verbose=False)

        trim_left = start_idx - fetch_start
        trim_right = trim_left + (end_idx - start_idx)
        eeg_slice = padded_slice[:, trim_left:trim_right]

        composite_img = render_composite(
            state=state, frame=frame, eeg_slice=eeg_slice, ch_names=ch_names, 
            sfreq=sfreq, window_start_sec=state.current_time_sec, 
            playback_offset_sec=state.playback_offset_sec, window_dur=state.window_width_sec,
            eeg_gain=eeg_gain, emg_gain=emg_gain, offset=eeg_offset, v_range=v_range,
            active_brush=state.active_brush, num_eeg=num_eeg, init_emg_log=init_emg_log
        )
        
        if state.paused:
            cv2.putText(composite_img, "PAUSED", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
        cv2.imshow(WINDOW_NAME, composite_img)
        
        if first_run:
            h_curr, w_curr = composite_img.shape[:2]
            # Force the move command again after the OS maps the window
            cv2.resizeWindow(WINDOW_NAME, w_curr, h_curr)
            cv2.moveWindow(WINDOW_NAME, 0, 40)
            first_run = False

        key = cv2.waitKey(10) & 0xFF
        
        if key == 27: break
        elif key == ord(' '): state.paused = not state.paused
        elif key == 13: state.save_csv() 
        elif key == ord('1'): state.active_brush = 'Artifact'
        elif key == ord('2'): state.active_brush = 'Wake'
        elif key == ord('3'): state.active_brush = 'NREM'
        elif key == ord('4'): state.active_brush = 'REM'
        elif key == ord('5'): state.active_brush = 'Unclear'
        elif key == ord('6'): state.active_brush = 'Confirm'
        elif key == ord('7'): state.active_brush = 'Erase'
        elif key in [81, 2]: state.jump_to_adjacent_epoch('prev')
        elif key in [83, 3]: state.jump_to_adjacent_epoch('next')
            
        elif key == ord('u'): state.jump_to_next_uncertain(1)
        elif key == ord('U'): state.jump_to_next_uncertain(-1)
        elif key == ord('['): state.nudge_threshold(-0.05)
        elif key == ord(']'): state.nudge_threshold(+0.05)
        elif key == ord('d'): 
            step = {4.0: 0.5, 10.0: 2.0, 30.0: 10.0, 60.0: 30.0}.get(state.window_width_sec, 4.0)
            max_time = max(0.0, (n_times / sfreq) - state.window_width_sec)
            state.current_time_sec = min(max_time, state.current_time_sec + step)
            state.playback_offset_sec = 0.0 
            
        elif key == ord('a'): 
            step = {4.0: 0.5, 10.0: 2.0, 30.0: 10.0, 60.0: 30.0}.get(state.window_width_sec, 4.0)
            state.current_time_sec = max(0.0, state.current_time_sec - step)
            state.playback_offset_sec = 0.0 
            
        elif key == ord('w'): eeg_offset += 0.1
        elif key == ord('s'): eeg_offset -= 0.1
        elif key in [0, 82, 126, ord('=')]: eeg_gain *= 1.2
        elif key in [1, 84, 125, ord('-')]: eeg_gain /= 1.2
        elif key == ord('+'): emg_gain *= 1.2
        elif key == ord('_'): emg_gain /= 1.2

    video_engine.release()
    cv2.destroyAllWindows()


def main():
    """Start the scorer from the command line."""
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass 

    # 1. Initialize Tkinter FIRST to securely grab macOS display dimensions
    root = tk.Tk()
    root.withdraw()
    
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # 2. Bypass Tkinter dialog if a file is provided via terminal
    if len(sys.argv) > 1:
        edf_path = os.path.abspath(sys.argv[1])
    else:
        # 3. Fall back to Tkinter dialog if no file is provided
        print("Waiting for file selection...")
        edf_path = filedialog.askopenfilename(title="Select EDF File", filetypes=[("EDF Files", "*.edf")])
        if not edf_path: return
        edf_path = os.path.abspath(edf_path)
    
    # Optional extra arguments, used when the Somnus GUI launches this viewer:
    #   argv[2] = scoring CSV to edit   (kept inside the project, so the source
    #             folder is never written to -- save_csv() writes back here)
    #   argv[3] = review metadata CSV   (per-epoch uncertainty / HMM flags)
    csv_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 \
        else os.path.splitext(edf_path)[0] + "_scored.csv"
    review_meta_path = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
    certainty_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.80
    eeg_idx = [int(x) - 1 for x in sys.argv[5].split(",")] \
        if len(sys.argv) > 5 and sys.argv[5].strip() else None

    if not os.path.exists(csv_path):
        print(f"Error: Could not find matching {os.path.basename(csv_path)}")
        return

    video_path = find_associated_video(edf_path)
    if video_path:
        print(f"Auto-detected associated video: {os.path.basename(video_path)}")
    else:
        print("Note: No associated video found. Proceeding with EEG only.")

    review_sleep(edf_path, csv_path, video_path, screen_w, screen_h,
                 review_meta_path=review_meta_path,
                 certainty_threshold=certainty_threshold,
                 eeg_idx=eeg_idx)

if __name__ == "__main__":
    main()
