import os
import sys
import time
import json
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import mne
from mne.filter import filter_data
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QVBoxLayout, QWidget,
)

from somnus.data.datasets import find_video
from somnus.scorer.video_handler import VideoHandler
from somnus.scorer.ui_rendering import (render_composite, clear_render_cache,
                                        WINDOW_NAME, PAINTABLE, LABELS)

# Which state a bin is shown as when it carries more than one.
PRECEDENCE = ('Artifact', 'REM', 'NREM', 'Wake', 'Unclear')
BRUSHES = PAINTABLE + ('Confirm', 'Erase')
VIEW_WIDTHS = ((4.0, '4 sec'), (10.0, '10 sec'), (30.0, '30 sec'), (60.0, '1 min'))
STEP_SEC = {4.0: 0.5, 10.0: 2.0, 30.0: 10.0, 60.0: 30.0}
AUTOSAVE_MS = 1500

# One opened recording: everything the window draws from.
View = namedtuple("View", "state raw video num_eeg v_range")


def _hms(t_sec):
    m, s = divmod(int(t_sec), 60)
    return f"{m // 60:02d}:{m % 60:02d}:{s:02d}"


class Playlist:
    """Segments of several recordings to review in turn.

    Read from a JSON file the Somnus app writes
    """
    def __init__(self, path):
        self.path = path
        with open(path) as fh:
            self.data = json.load(fh)
        self.recordings = self.data["recordings"]
        self.segments = self.data["segments"]
        self.index = next((i for i, seg in enumerate(self.segments)
                           if not seg.get("done")), 0)

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2)
        os.replace(tmp, self.path)


class SleepReviewState:
    """Everything the scorer window is currently showing and holding."""
    def __init__(self, csv_path, sfreq, ch_names, n_times,
                 review_meta_path=None, certainty_threshold=0.80):
        """Load the recording, its scoring, and the model output if present."""
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.csv_path = csv_path
        self.n_times = n_times
        self.duration_sec = n_times / sfreq
        
        # Fixed panes; the EEG pane takes whatever the window leaves.
        self.video_w = 450
        self.side_w = 200
        self.bottom_h = 340
        self.eeg_w = 1000
        self.eeg_h = 300
        
        self.current_time_sec = 0.0
        self.window_width_sec = 10.0
        self.playback_offset_sec = 0.0 
        self.paused = False
        
        self.active_brush = None
        self.paint_mode = 'Paint'
        self.range_start_sec = None
        self.segment = None                # (start, end) s while a playlist segment is shown
        self.segment_nav = None            # (index, count) of that segment
        self.nav_buttons = []
        self.message = print
        
        self.update_counter = 0 
        
        df = pd.read_csv(csv_path)
        # 'Unclear' marks an epoch as ambiguous, 'Confirmed' marks
        # one the user has explicitly agreed with.
        for col in PAINTABLE + ('Confirmed',):
            if col not in df.columns:
                df[col] = 0
        # Written back on every save with only the label columns replaced.
        self._base = df
        self.t = df['Time_sec'].to_numpy(dtype=float)
        self.bin_step = float(self.t[1] - self.t[0]) if len(self.t) > 1 else 0.5
        
        # Default Brush Size
        self.brush_size_sec = max(self.bin_step, 1.0)
        
        self.onehot = (df[list(PAINTABLE)] == 1).to_numpy().astype(np.uint8)
        self.confirmed = (df['Confirmed'] == 1).to_numpy().astype(np.uint8)
        self.code = self._codes(self.onehot)
        self.model_code = (
            df['Model_State'].map({n: i for i, n in enumerate(LABELS)})
            .fillna(0).to_numpy(dtype=np.int8)
            if 'Model_State' in df.columns else None)

        self._initial = (self.onehot.copy(), self.confirmed.copy())
        self._stroke = None
        self._undo, self._redo = [], []
        # Every edit bumps `version`; the saver records which one is on disk.
        self.version = 0
        self.saved_version = 0
        self.save_error = None
        self._saver = ThreadPoolExecutor(max_workers=1)

        # How sure the model was about each epoch, written by the Somnus app.
        # Absent when the scorer is opened on its own, in which case the
        # model-review parts of the display are simply not drawn.
        self.review_meta = None
        # An epoch counts as low certainty below this. Adjustable with [ and ].
        self.certainty_threshold = float(certainty_threshold)
        self.epoch_sec = 4.0
        self._load_review_meta(review_meta_path)

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
        self.message(f"certainty threshold -> {self.certainty_threshold:.2f} "
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
            self.message("No model review metadata loaded.")
            return
        cand = np.flatnonzero(self._low_mask())
        cand = cand[self._inside(cand * self.epoch_sec)]
        if cand.size == 0:
            self.message(f"No epochs below the {self.certainty_threshold:.2f} "
                         f"certainty threshold.")
            return

        # Anchor on the CENTER of the visible window, not its left edge.
        if abs(self.playback_offset_sec) > 1e-9:
            anchor = self.current_time_sec + self.playback_offset_sec
        else:
            anchor = self.current_time_sec + self.window_width_sec / 2.0
        cur_ep = int(anchor // self.epoch_sec)

        if direction >= 0:
            hit = cand[cand > cur_ep]
            target = int(hit[0]) if hit.size else int(cand[0])   # wrap to start
        else:
            hit = cand[cand < cur_ep]
            target = int(hit[-1]) if hit.size else int(cand[-1])
        t = target * self.epoch_sec
        # center the MIDDLE of the epoch, so the whole block sits mid-window
        self._center_on(t + self.epoch_sec / 2.0)
        self.mark_reviewed(t)
        pos = int(np.searchsorted(cand, target)) + 1
        self.message(f"epoch {target} (t={t:.0f}s) confidence="
                     f"{self._conf[target]:.2f}  [{pos}/{cand.size} below "
                     f"{self.certainty_threshold:.2f}]"
                     + ("" if hit.size else "  (wrapped)"))

    # --------------------------------------------------------------- navigation
    def bounds(self):

        return self.segment or (0.0, self.duration_sec)

    def view_range(self):

        lo, hi = self.bounds()
        if self.segment:
            pad = self.window_width_sec / 2.0
            lo, hi = max(0.0, lo - pad), min(self.duration_sec, hi + pad)
        return lo, hi

    def _inside(self, t_sec):
        lo, hi = self.bounds()
        return (t_sec >= lo) & (t_sec < hi)

    def max_start(self):
        """The latest the window can start and still be filled."""
        lo, hi = self.view_range()
        return max(lo, hi - self.window_width_sec)
        
    def scroll_to(self, start_sec):
        """Show the window starting here, playing from its left edge."""
        self.current_time_sec = float(np.clip(start_sec, self.view_range()[0], self.max_start()))
        self.playback_offset_sec = 0.0

    def _center_on(self, t_sec):
        """Put this moment mid-window and play from it."""
        self.current_time_sec = float(np.clip(
            t_sec - self.window_width_sec / 2.0, self.view_range()[0], self.max_start()))
        self.playback_offset_sec = t_sec - self.current_time_sec

    def _playhead(self):
        """Where the playhead is, nudged past float error from the last jump."""
        return self.current_time_sec + self.playback_offset_sec + 1e-6

    def _bouts(self):
        """Start time and label code of every run of one label."""
        starts = np.flatnonzero(np.r_[True, self.code[1:] != self.code[:-1]])
        return self.t[starts], self.code[starts]

    def jump_to_bout(self, target_state, direction=1):
        """Move to the next or previous bout of a given state.

        Previous means before the bout under the playhead, so repeated presses
        walk back through bouts rather than return to the start of this one.
        """
        starts, codes = self._bouts()
        here = self._playhead()
        mine = (codes == LABELS.index(target_state)) & self._inside(starts)
        if direction > 0:
            hit = starts[mine & (starts > here)][:1]
        else:
            cur = starts[max(0, np.searchsorted(starts, here, side='right') - 1)]
            hit = starts[mine & (starts < cur)][-1:]
        if not hit.size:
            self.message(f"No {'later' if direction > 0 else 'earlier'} "
                         f"{target_state} bout.")
            return
        self._center_on(float(hit[0]))

    def jump_to_boundary(self, direction=1):
        """Step to the next or previous bout boundary."""
        starts, _ = self._bouts()
        here = self._playhead()
        i = np.searchsorted(starts, here, side='right') - 1
        cur = starts[i] if i >= 0 else 0.0
        inside = self._inside(starts)
        if direction > 0:
            later = starts[(starts > cur + 0.1) & inside]
            if not later.size:
                return
            target = later[0]
        else:
            earlier = starts[(starts < cur - 0.1) & inside]
            target = earlier[-1] if earlier.size else self.bounds()[0]
        self._center_on(float(target))

    # ----------------------------------------------------------------- labels
    @staticmethod
    def _codes(onehot):
        """Each bin's label code, from the first state in PRECEDENCE it carries."""
        code = np.zeros(len(onehot), dtype=np.int8)
        for name in reversed(PRECEDENCE):
            k = PAINTABLE.index(name)
            code[onehot[:, k] == 1] = k + 1
        return code

    def is_strong(self, idx):
        """Whether a person has taken a position on these bins: confirmed them,
        or painted something other than what the model said."""
        code = self.code[idx]
        changed = code != 0
        if self.model_code is not None:
            changed &= code != self.model_code[idx]
        return (self.confirmed[idx] == 1) | changed

    def _bins(self, start_sec, end_sec):
        """Index range of the bins overlapping this stretch of time."""
        lo = int(np.searchsorted(self.t, start_sec - self.bin_step, side='right'))
        hi = int(np.searchsorted(self.t, end_sec, side='left'))
        return lo, hi

    def paint_state(self, start_sec, end_sec):
        """Apply the active brush to every bin in this stretch of time.

        Part of a stroke, which end_stroke() closes. A state brush over bins
        that carried that state when the stroke began confirms them, exactly
        as the Confirm brush does.
        """
        if not self.active_brush:
            return
        if self.segment:
            start_sec = max(start_sec, self.segment[0])
            end_sec = min(end_sec, self.segment[1])
        lo, hi = self._bins(start_sec, end_sec)
        if hi <= lo:
            return
        if self._stroke is None:
            self._stroke = (self.onehot.copy(), self.confirmed.copy())

        if self.active_brush == 'Confirm':
            # Affirms the label already there WITHOUT changing it. This is the
            # only way the user can say "the model got this right", because a
            # label left untouched is indistinguishable from one never looked at
            # once it has been written to a CSV -- so an unconfirmed epoch is
            # never treated as reviewed, and never becomes a training target.
            self.confirmed[lo:hi] = 1
        elif self.active_brush == 'Erase':
            self.onehot[lo:hi] = 0
            self.confirmed[lo:hi] = 0
        else:
            k = PAINTABLE.index(self.active_brush)
            before = self._stroke[0][lo:hi]
            same = self._codes(before) == k + 1
            new = np.zeros_like(before)
            new[:, k] = 1
            self.onehot[lo:hi] = np.where(same[:, None], before, new)
            # A state change drops a prior confirmation: that confirmation
            # referred to the label being replaced.
            self.confirmed[lo:hi] = same
        self.code[lo:hi] = self._codes(self.onehot[lo:hi])
        self.update_counter += 1

    def confirm_segment(self, tag=None):
        """Confirm every Wake, NREM and REM bin in the segment, as one
        step."""
        lo, hi = self._bins(*self.segment)
        self.end_stroke()
        self._stroke = (self.onehot.copy(), self.confirmed.copy())
        sleep = np.isin(self.code[lo:hi], [LABELS.index(n) for n in ('Wake', 'NREM', 'REM')])
        self.confirmed[lo:hi][sleep] = 1
        self.update_counter += 1
        return self.end_stroke(tag)

    def end_stroke(self, tag=None):
        """Close the current stroke"""
        if self._stroke is None:
            return False
        onehot, confirmed = self._stroke
        self._stroke = None
        diff = np.flatnonzero((onehot != self.onehot).any(axis=1)
                              | (confirmed != self.confirmed))
        if not diff.size:
            return False
        lo, hi = int(diff[0]), int(diff[-1]) + 1
        self._undo.append((lo, onehot[lo:hi].copy(), confirmed[lo:hi].copy(),
                           self.onehot[lo:hi].copy(), self.confirmed[lo:hi].copy(),
                           tag))
        self._redo.clear()
        self.version += 1
        return True

    def _apply(self, lo, onehot, confirmed):
        hi = lo + len(onehot)
        self.onehot[lo:hi] = onehot
        self.confirmed[lo:hi] = confirmed
        self.code[lo:hi] = self._codes(onehot)
        self.update_counter += 1
        self.version += 1

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def undo(self):
        """Take back the last stroke, and return it (None if there was none)."""
        self.end_stroke()
        if not self._undo:
            return None
        step = self._undo.pop()
        self._apply(step[0], step[1], step[2])
        self._redo.append(step)
        return step

    def redo(self):
        """Put back the last stroke undone, and return it (None if there was none)."""
        self.end_stroke()
        if not self._redo:
            return None
        step = self._redo.pop()
        self._apply(step[0], step[3], step[4])
        self._undo.append(step)
        return step

    def revert(self):
        """Return to the scoring as it was when the scorer opened, history and all."""
        self._stroke = None
        self.onehot, self.confirmed = (a.copy() for a in self._initial)
        self.code = self._codes(self.onehot)
        self._undo.clear()
        self._redo.clear()
        self.update_counter += 1
        self.version += 1

    # ------------------------------------------------------------------ saving
    @property
    def dirty(self):
        return self.version != self.saved_version

    def save_async(self):
        """Write the scoring in the background, so painting never waits on the disk."""
        if self.dirty:
            self._saver.submit(self._write, self.version,
                               self.onehot.copy(), self.confirmed.copy())

    def flush(self):
        """Wait until the latest edit is on disk. True if it got there."""
        self.save_async()
        self._saver.submit(lambda: None).result()
        return not self.dirty

    def shutdown(self):
        self._saver.shutdown(wait=True)
    
    def _write(self, version, onehot, confirmed):
        out = self._base.copy()
        out[list(PAINTABLE)] = onehot
        out['Confirmed'] = confirmed
        tmp = self.csv_path + '.tmp'
        try:
            out.to_csv(tmp, index=False)
            # one atomic step, so a reader never sees a half-written file
            os.replace(tmp, self.csv_path)
        except Exception as e:
            self.save_error = str(e)
            print(f"Could not save {self.csv_path}: {e}")
            return
        self.save_error = None
        self.saved_version = version
        print(f"Saved scoring to {self.csv_path}")
            
            
class Canvas(QWidget):
    """Shows the rendered frame, and reports the mouse in its pixel coordinates."""
    pressed = Signal(int, int)
    dragged = Signal(int, int)
    released = Signal()
    resized = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1000, 640)
        self._frame = None
        self._image = None

    def show_frame(self, frame):
        # QImage borrows the buffer, so the array is kept alongside it
        self._frame = np.ascontiguousarray(frame)
        h, w = self._frame.shape[:2]
        self._image = QImage(self._frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._image is not None:
            p.drawImage(0, 0, self._image)

    def resizeEvent(self, ev):
        self.resized.emit(ev.size().width(), ev.size().height())

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.pressed.emit(int(ev.position().x()), int(ev.position().y()))

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self.dragged.emit(int(ev.position().x()), int(ev.position().y()))

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.released.emit()


class TimelineBar(QWidget):
    """A scroll bar over the whole recording, or the segment being reviewed.

    The thumb never shrinks below MIN_THUMB pixels, so it stays easy to grab
    however long the recording is.
    """
    MIN_THUMB = 64
    scrolled = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(18)
        self._view = (0.0, 1.0, 1.0)          # start, width, total (s)
        self._grab = None

    def set_view(self, start, width, total):
        if (start, width, total) != self._view:
            self._view = (start, width, total)
            self.update()

    def _thumb(self):
        """Left edge and width of the thumb, in pixels."""
        start, width, total = self._view
        w = self.width()
        tw = min(w, max(self.MIN_THUMB, w * width / max(total, 1e-9)))
        span = total - width
        return ((w - tw) * start / span if span > 0 else 0.0), tw

    def _drag_to(self, x):
        _, width, total = self._view
        free = self.width() - self._thumb()[1]
        if free > 0:
            frac = min(max((x - self._grab) / free, 0.0), 1.0)
            self.scrolled.emit(frac * max(total - width, 0.0))

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        x = ev.position().x()
        x0, tw = self._thumb()
        # a click on the track centres the thumb there, then drags it
        self._grab = x - x0 if x0 <= x <= x0 + tw else tw / 2.0
        self._drag_to(x)
        self.update()
                
    def mouseMoveEvent(self, ev):
        if self._grab is not None:
            self._drag_to(ev.position().x())
                
    def mouseReleaseEvent(self, ev):
        self._grab = None
        self.update()
                    
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        x, tw = self._thumb()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(170, 170, 170) if self._grab is not None
                   else QColor(115, 115, 115))
        p.drawRoundedRect(QRectF(x + 1, 3, tw - 2, self.height() - 6), 5, 5)


class ScorerWindow(QMainWindow):
    """The scorer window: traces, video, menus and a scroll bar, saving as it goes."""

    def __init__(self, view=None, playlist=None):
        """Show one recording (`view`), or step through a playlist's segments."""
        super().__init__()
        self.playlist = playlist
        self.views = {}                    # recording name -> View, kept open
        self.state = None
        self.eeg_gain = 1.0
        self.emg_gain = 1.0
        self.eeg_offset = 0.0
        self._painting = False
        self._last_paint_sec = 0.0
        self._status = None
        self._help = None

        self.setFocusPolicy(Qt.StrongFocus)
        self.canvas = Canvas()
        self.scroll = TimelineBar()
        self.banner = self._build_banner() if playlist else None
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if self.banner:
            lay.addWidget(self.banner)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.scroll)
        self.setCentralWidget(central)
        self.canvas.pressed.connect(self._on_press)
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.released.connect(self._on_release)
        self.canvas.resized.connect(self._on_resize)
        self.scroll.scrolled.connect(
            lambda v: self.state.scroll_to(self.state.view_range()[0] + v))

        self.lbl_mode = QLabel()
        self.lbl_save = QLabel()
        self.statusBar().addPermanentWidget(self.lbl_mode)
        self.statusBar().addPermanentWidget(self.lbl_save)

        if playlist:
            self._initial_done = [seg.get("done", False) for seg in playlist.segments]
            view = self._open(playlist.segments[playlist.index]["recording"])
        self._show(view)
        self._build_menus()
        if playlist:
            self._go_to(playlist.index)

        # Saves at most AUTOSAVE_MS after the first unsaved edit.
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(AUTOSAVE_MS)
        self._autosave.timeout.connect(lambda: self.state.save_async())

        self._last_tick = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(15)

    def _show(self, view):
        """Put a recording on screen, keeping the brush and view settings."""
        old = self.state
        self.view = view
        self.state, self.raw, self.video, self.num_eeg, self.v_range = view
        s = self.state
        if old is not None and old is not s:
            for k in ('active_brush', 'paint_mode', 'window_width_sec',
                      'brush_size_sec', 'paused'):
                setattr(s, k, getattr(old, k))
        s.message = self._message
        s.eeg_w, s.eeg_h = self.canvas.width(), self.canvas.height() - s.bottom_h
        self._eeg_key = None
        self._eeg_slice = None
        # Only apply a 200 Hz low-pass if the sampling rate captures frequencies above 200 Hz
        self._h_freq = 200.0 if (s.sfreq / 2.0) > 200.0 else None
        self._status = None
        clear_render_cache()
        self.setWindowTitle(f"{WINDOW_NAME} — {os.path.basename(s.csv_path)}")

    # -------------------------------------------------------------- playlist
    def _open(self, name):
        """The View for a playlist recording, opening it the first time."""
        if name not in self.views:
            self.statusBar().showMessage(f"Opening {name} …")
            QApplication.processEvents()
            self.views[name] = open_playlist_recording(self.playlist, name)
        return self.views[name]

    def _build_banner(self):
        bar = QWidget()
        bar.setStyleSheet("background:#2b2b2b; color:white;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        self.lbl_segment = QLabel()
        row.addWidget(self.lbl_segment, 1)
        return bar

    def _refresh_banner(self):
        pl = self.playlist
        seg = pl.segments[pl.index]
        n_done = sum(bool(x.get("done")) for x in pl.segments)
        self.lbl_segment.setText(
            f"<b>Segment {pl.index + 1} of {len(pl.segments)}</b> &nbsp; "
            f"{seg['recording']} &nbsp; {_hms(seg['start'])} – {_hms(seg['end'])} "
            f"({(seg['end'] - seg['start']) / 60:.1f} min)"
            + (" &nbsp; <b style='color:#6fdc8c'>✓ done</b>" if seg.get("done") else "")
            + f" &nbsp;&nbsp; <span style='color:#aaa'>{n_done} of "
              f"{len(pl.segments)} done</span>")

    def _go_to(self, i):
        """Show segment i of the playlist."""
        pl = self.playlist
        i = min(max(i, 0), len(pl.segments) - 1)
        seg = pl.segments[i]
        self._painting = False
        self.state.end_stroke()
        self.state.save_async()
        view = self._open(seg["recording"])
        moved = view is not self.view
        if moved:
            self._show(view)
        pl.index = i
        s = self.state
        s.segment = (float(seg["start"]), float(seg["end"]))
        s.segment_nav = (i, len(pl.segments))
        s.range_start_sec = None
        s.scroll_to(s.segment[0])
        s.update_counter += 1
        self._refresh_banner()
        self._message(f"Segment {i + 1} of {len(pl.segments)}"
                      + (f": now in {seg['recording']}" if moved else ""))

    def _next_segment(self):
        """Confirm the labels left in this segment, then show the next one, or
        close after the last."""
        pl = self.playlist
        self._painting = False
        self._edited(self.state.confirm_segment(tag=pl.index))
        self._set_done(pl.index, True)
        if pl.index + 1 < len(pl.segments):
            self._go_to(pl.index + 1)
        else:
            self.close()

    def _set_done(self, i, done):
        self.playlist.segments[i]["done"] = done
        self.playlist.save()
        self._refresh_banner()

    # ------------------------------------------------------------------ menus
    def _action(self, menu, text, slot, keys=None):
        act = QAction(text, self)
        if keys is not None:
            act.setShortcuts(keys)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _choices(self, menu, items, pick, checked=None):
        group = QActionGroup(self)
        for value, label, key in items:
            act = QAction(label, self, checkable=True)
            act.setChecked(value == checked)
            if key:
                act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _=False, v=value: pick(v))
            group.addAction(act)
            menu.addAction(act)

    def _build_menus(self):
        s = self.state
        mb = self.menuBar()
        m = mb.addMenu("File")
        self._action(m, "Save Now", lambda: self.state.save_async(), QKeySequence.Save)
        self._action(m, "Revert to Last Version…", self._revert)
        m.addSeparator()
        self._action(m, "Close", self.close, QKeySequence.Close)

        m = mb.addMenu("Edit")
        self.act_undo = self._action(m, "Undo", self._undo, QKeySequence.Undo)
        redo = QKeySequence.keyBindings(QKeySequence.Redo)
        if QKeySequence("Ctrl+Y") not in redo:
            redo.append(QKeySequence("Ctrl+Y"))
        self.act_redo = self._action(m, "Redo", self._redo, redo)

        self._choices(mb.addMenu("Brush"),
                      [(b, b, str(i)) for i, b in enumerate(BRUSHES, 1)],
                      self._set_brush)
        self._choices(mb.addMenu("View"),
                      [(w, label, None) for w, label in VIEW_WIDTHS],
                      self._set_view_width, checked=s.window_width_sec)
        self._choices(mb.addMenu("Mode"),
                      [('Paint', 'Paint Mode', None), ('Range', 'Range Mode', None)],
                      self._set_mode, checked=s.paint_mode)
        m = mb.addMenu("Brush Size")
        self._action(m, "Minimum", lambda: self._set_brush_size(None))
        self._action(m, "Decrease", lambda: self._set_brush_size(-1))
        self._action(m, "Increase", lambda: self._set_brush_size(+1))

        if self.playlist:
            m = mb.addMenu("Segment")
            self._action(m, "Previous Segment", lambda: self._go_to(self.playlist.index - 1),
                         [QKeySequence("Ctrl+Left")])
            self._action(m, "Next Segment (or Close After the Last)", self._next_segment,
                         [QKeySequence("Ctrl+Right")])

        m = mb.addMenu("Help")
        self._action(m, "Keyboard Controls", self._show_help, QKeySequence.HelpContents)

    def _set_brush(self, brush):
        self.state.active_brush = brush

    def _set_view_width(self, width):
        s = self.state
        s.window_width_sec = width
        s.scroll_to(s.current_time_sec)

    def _set_mode(self, mode):
        self.state.paint_mode = mode
        self.state.range_start_sec = None

    def _set_brush_size(self, steps):
        s = self.state
        if steps is None:
            s.brush_size_sec = max(s.bin_step, 1.0)
        else:
            s.brush_size_sec = max(s.bin_step, s.brush_size_sec + steps * s.bin_step)

    def _controls(self):
        """(keys, what they do) for the Help menu."""
        def native(k):
            return QKeySequence(k).toString(QKeySequence.NativeText)
        return [
            ("Drag on the traces", "Paint with the active brush (Paint mode)"),
            ("Click, then click again", "Paint between the two clicks (Range mode)"),
            ("◀ / ▶ in the side panel", "Previous / next bout of that state, or "
                                        "low-certainty epoch"),
            ("Scroll bar", "Drag, or click along it, to move through the recording "
                           "or segment"),
            ("1 – 7", "Brush: Wake, NREM, REM, Artifact, Unclear, Confirm, Erase"),
            ("", "Painting a state over the same state confirms it, as Confirm does"),
            ("Space", "Pause / play"),
            ("← / →", "Previous / next bout boundary"),
            ("A / D", "Step back / forward (the step scales with the view)"),
            ("↑ / ↓  or  = / -", "EEG gain"),
            ("+ / _", "EMG gain"),
            ("W / S", "EEG vertical offset"),
            ("U / Shift+U", "Next / previous low-certainty epoch"),
            ("[ / ]", "Lower / raise the certainty threshold"),
            (native("Ctrl+Z"), "Undo"),
            (f"{native('Ctrl+Shift+Z')}  or  {native('Ctrl+Y')}", "Redo"),
            (native("Ctrl+S"), "Save now (changes also save automatically)"),
            (f"Esc  or  {native('Ctrl+W')}", "Close"),
        ] + ([
            (native("Ctrl+Left"), "Previous segment"),
            (native("Ctrl+Right"), "Next segment, confirming the labels left in "
                                   "this one; after the last, close"),
        ] if self.playlist else [])

    def _show_help(self):
        if self._help is None:
            rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                           for k, v in self._controls())
            self._help = QDialog(self)
            self._help.setWindowTitle("Keyboard Controls")
            lay = QVBoxLayout(self._help)
            lab = QLabel(f"<table cellspacing='6'>{rows}</table>")
            lab.setTextFormat(Qt.RichText)
            lay.addWidget(lab)
            bb = QDialogButtonBox(QDialogButtonBox.Close)
            bb.rejected.connect(self._help.close)
            lay.addWidget(bb)
        self._help.show()
        self._help.raise_()

    # ---------------------------------------------------------------- editing
    def _edited(self, changed=True):
        if changed and not self._autosave.isActive():
            self._autosave.start()
        self._refresh_status()

    def _undo(self):
        self._painting = False
        step = self.state.undo()
        self._step_tag(step, done=False)
        self._edited(step is not None)

    def _redo(self):
        self._painting = False
        step = self.state.redo()
        self._step_tag(step, done=True)
        self._edited(step is not None)

    def _step_tag(self, step, done):
        """Undoing or redoing a "Segment done" also clears or sets its mark."""
        if self.playlist and step is not None and step[5] is not None:
            self._set_done(step[5], done)

    def _all_views(self):
        return list(self.views.values()) if self.playlist else [self.view]

    def _revert(self):
        if QMessageBox.question(
                self, "Revert to last version",
                "You will lose all changes from this session. Are you sure?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) == QMessageBox.Yes:
            self._painting = False
            for v in self._all_views():
                v.state.revert()
                v.state.save_async()
            if self.playlist:
                for seg, done in zip(self.playlist.segments, self._initial_done):
                    seg["done"] = done
                self.playlist.save()
                self._refresh_banner()
            self._refresh_status()

    def _message(self, text):
        print(text)
        self.statusBar().showMessage(text, 5000)

    # ------------------------------------------------------------------ mouse
    def _time_at(self, x):
        s = self.state
        return s.current_time_sec + (min(max(x, 0), s.eeg_w) / s.eeg_w) * s.window_width_sec

    def _on_press(self, x, y):
        s = self.state
        if y < s.eeg_h:
            if s.active_brush is None:
                return
            t = self._time_at(x)
            if s.paint_mode == 'Paint':
                self._painting = True
                self._last_paint_sec = t
                s.paint_state(t, t + s.brush_size_sec)
            elif s.range_start_sec is None:
                s.range_start_sec = t
            else:
                s.paint_state(min(s.range_start_sec, t), max(s.range_start_sec, t))
                s.range_start_sec = None
                self._edited(s.end_stroke())
        elif x >= s.eeg_w - s.side_w:
            bx, by = x - (s.eeg_w - s.side_w), y - s.eeg_h
            for (x1, y1, x2, y2), target, direction in s.nav_buttons:
                if x1 <= bx <= x2 and y1 <= by <= y2:
                    if target == 'segment':
                        if direction < 0:
                            self._go_to(self.playlist.index - 1)
                        else:
                            self._next_segment()
                    elif target == 'uncertain':
                        s.jump_to_next_uncertain(direction)
                    else:
                        s.jump_to_bout(target, direction)
                    break

    def _on_drag(self, x, y):
        s = self.state
        if self._painting and 0 <= y < s.eeg_h:
            t = self._time_at(x)
            # cover the ground between events, so a fast drag leaves no gaps
            s.paint_state(min(t, self._last_paint_sec),
                          max(t, self._last_paint_sec) + s.brush_size_sec)
            self._last_paint_sec = t

    def _on_release(self):
        if self._painting:
            self._painting = False
            self._edited(self.state.end_stroke())

    def _on_resize(self, w, h):
        self.state.eeg_w = w
        self.state.eeg_h = h - self.state.bottom_h

    # --------------------------------------------------------------- keyboard
    def keyPressEvent(self, ev):
        s = self.state
        key, text = ev.key(), ev.text()
        if ev.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            super().keyPressEvent(ev)
        elif key == Qt.Key_Escape: self.close()
        elif key == Qt.Key_Space: s.paused = not s.paused
        elif key == Qt.Key_Left: s.jump_to_boundary(-1)
        elif key == Qt.Key_Right: s.jump_to_boundary(+1)
        elif key == Qt.Key_Up or text == '=': self.eeg_gain *= 1.2
        elif key == Qt.Key_Down or text == '-': self.eeg_gain /= 1.2
        elif text == '+': self.emg_gain *= 1.2
        elif text == '_': self.emg_gain /= 1.2
        elif text == 'w': self.eeg_offset += 0.1
        elif text == 's': self.eeg_offset -= 0.1
        elif text in ('a', 'd'):
            step = STEP_SEC.get(s.window_width_sec, 4.0)
            s.scroll_to(s.current_time_sec + (step if text == 'd' else -step))
        elif text == 'u': s.jump_to_next_uncertain(1)
        elif text == 'U': s.jump_to_next_uncertain(-1)
        elif text == '[': s.nudge_threshold(-0.05)
        elif text == ']': s.nudge_threshold(+0.05)
        else:
            super().keyPressEvent(ev)

    # ---------------------------------------------------------------- drawing
    def _eeg_window(self):
        """The filtered traces in view, read again only when the view moves."""
        s = self.state
        start = int(s.current_time_sec * s.sfreq)
        end = int((s.current_time_sec + s.window_width_sec) * s.sfreq)
        if self._eeg_key != (start, end):
            # filter with 2 s of margin so the window's edges are not distorted
            pad = int(2.0 * s.sfreq)
            a, b = max(0, start - pad), min(s.n_times, end + pad)
            padded = filter_data(self.raw[:, a:b][0], s.sfreq, l_freq=0.5,
                                 h_freq=self._h_freq, verbose=False)
            self._eeg_slice = padded[:, start - a:end - a]
            self._eeg_key = (start, end)
        return self._eeg_slice

    def _tick(self):
        s = self.state
        # Real elapsed time, capped at 0.1 s to prevent huge jumps
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        if not s.paused:
            s.playback_offset_sec += dt
            if s.playback_offset_sec >= s.window_width_sec:
                s.playback_offset_sec = 0.0

        frame = self.video.get_frame_at_time(s.current_time_sec + s.playback_offset_sec)
        img = render_composite(
            state=s, frame=frame, eeg_slice=self._eeg_window(), ch_names=s.ch_names,
            sfreq=s.sfreq, window_start_sec=s.current_time_sec,
            playback_offset_sec=s.playback_offset_sec, window_dur=s.window_width_sec,
            eeg_gain=self.eeg_gain, emg_gain=self.emg_gain, offset=self.eeg_offset,
            v_range=self.v_range, num_eeg=self.num_eeg)
        if s.paused:
            cv2.putText(img, "PAUSED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        self.canvas.show_frame(img)
        lo, hi = s.view_range()
        self.scroll.set_view(s.current_time_sec - lo, s.window_width_sec, hi - lo)
        self._refresh_status()

    def _refresh_status(self):
        s = self.state
        mode = (f"Brush: {s.active_brush or 'none'}   View: {s.window_width_sec:g} s   "
                f"{s.paint_mode}   Size: {s.brush_size_sec:.1f} s")
        if s.save_error:
            saved = f"Save failed: {s.save_error}"
        elif s.dirty:
            saved = "Saving…"
        else:
            saved = "All changes saved"
        status = (mode, saved, s.can_undo, s.can_redo)
        if status != self._status:
            self._status = status
            self.lbl_mode.setText(mode)
            self.lbl_save.setText(saved)
            self.act_undo.setEnabled(s.can_undo)
            self.act_redo.setEnabled(s.can_redo)

    def closeEvent(self, ev):
        self._timer.stop()
        self._autosave.stop()
        self._painting = False
        self.state.end_stroke()
        failed = [v.state for v in self._all_views() if not v.state.flush()]
        if failed and QMessageBox.question(
                self, "Could not save",
                "The scoring could not be saved:\n"
                + "\n".join(f"{os.path.basename(f.csv_path)}: {f.save_error}" for f in failed)
                + "\n\nClose anyway and lose the unsaved changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            self._timer.start()
            ev.ignore()
            return
        for v in self._all_views():
            v.state.shutdown()
            v.video.release()
        ev.accept()


def open_recording(edf_path, csv_path, video_path,
                   review_meta_path=None, certainty_threshold=0.80,
                   eeg_idx=None):
    """Open a recording, its scoring and its video, ready for the window."""
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

    state = SleepReviewState(csv_path, sfreq, ch_names, n_times,
                             review_meta_path=review_meta_path,
                             certainty_threshold=certainty_threshold)
    
    print("Calculating initial voltage scale...")
    init_samples = min(int(10 * sfreq), n_times)
    init_data = raw[0:num_eeg, 0:init_samples][0]
    if init_data.size > 0:
        v_range = np.max(init_data) - np.min(init_data)
        if v_range == 0: v_range = 1.0
    else:
        v_range = 1.0

    # When the Somnus app launches this, the scoring CSV it hands us lives in
    # <project>/labels/, so the project's cache is its sibling. Opened on its
    # own there is no project, and nothing is written to disk.
    labels_dir = os.path.dirname(os.path.abspath(csv_path))
    cache_dir = os.path.join(os.path.dirname(labels_dir), "cache") \
        if os.path.basename(labels_dir) == "labels" else None
    start = raw.info['meas_date']
    video_engine = VideoHandler(video_path, cache_dir=cache_dir,
                                duration=n_times / sfreq,
                                start_unix=start.timestamp() if start else None)
    return View(state, raw, video_engine, num_eeg, v_range)
    

def open_playlist_recording(playlist, name):
    """Open one of a playlist's recordings, with the channels the project set."""
    rec = playlist.recordings[name]
    chans = playlist.data.get("eeg_channels") or []
    return open_recording(rec["edf"], rec["scored"],
                          rec.get("video") or find_video(rec["edf"]),
                          review_meta_path=rec.get("meta"),
                          certainty_threshold=playlist.data.get("certainty_threshold", 0.80),
                          eeg_idx=[int(c) - 1 for c in chans] or None)


def _run(window):
    app = QApplication.instance() or QApplication(sys.argv)
    window.showMaximized()
    # the Somnus app waits for this line to know the window is up
    print("Scorer ready.", flush=True)
    app.exec()


def review_sleep(edf_path, csv_path, video_path,
                 review_meta_path=None, certainty_threshold=0.80,
                 eeg_idx=None):
    """Run the scorer: open the recording and show it until the window closes."""
    QApplication.instance() or QApplication(sys.argv)
    _run(ScorerWindow(open_recording(edf_path, csv_path, video_path,
                                     review_meta_path, certainty_threshold, eeg_idx)))


def review_playlist(path):
    """Run the scorer over the segments listed in a playlist file."""
    QApplication.instance() or QApplication(sys.argv)
    _run(ScorerWindow(playlist=Playlist(path)))


def main():
    """Start the scorer from the command line."""
    app = QApplication(sys.argv)

    if len(sys.argv) > 2 and sys.argv[1] == "--playlist":
        review_playlist(os.path.abspath(sys.argv[2]))
        return
    if len(sys.argv) > 1:
        edf_path = os.path.abspath(sys.argv[1])
    else:
        print("Waiting for file selection...")
        edf_path, _ = QFileDialog.getOpenFileName(None, "Select EDF File", "",
                                                  "EDF Files (*.edf)")
        if not edf_path: return
        edf_path = os.path.abspath(edf_path)
    
    # Optional extra arguments, used when the Somnus GUI launches this viewer:
    #   argv[2] = scoring CSV to edit   (kept inside the project, so the source
    #             folder is never written to -- saves go back here)
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

    video_path = find_video(edf_path)
    if video_path:
        print(f"Auto-detected associated video: {os.path.basename(video_path)}")
    else:
        print("Note: No associated video found. Proceeding with EEG only.")

    review_sleep(edf_path, csv_path, video_path,
                 review_meta_path=review_meta_path,
                 certainty_threshold=certainty_threshold,
                 eeg_idx=eeg_idx)

if __name__ == "__main__":
    main()
