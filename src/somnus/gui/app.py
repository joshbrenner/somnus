"""Somnus GUI -- tabbed desktop app for scoring, reviewing and fine-tuning.

    Project  ->  Score  ->  Review & Relabel  ->  Fine-tune  ->  Evaluate

Layout follows DeepLabCut's tabbed workflow. All heavy work (featurizing,
scoring, fine-tuning) runs on a worker thread so the window stays responsive.

Nothing is ever written outside the project directory; source recordings are
opened read-only.

Run:
    somnus-gui            (or: python -m somnus.gui)
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from somnus.gui import core

STATE_COLORS = {"Wake": "#999999", "NREM": "#9470DB", "REM": "#DB3333"}
STATE_Y = {"Wake": 2, "NREM": 1, "REM": 0}


# ----------------------------------------------------------------- worker glue
class Worker(QObject):
    """Runs one callable off the GUI thread and reports back."""
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn, self.a, self.kw = fn, a, kw

    @Slot()
    def run(self):
        try:
            self.kw["log"] = self.progress.emit
            self.finished.emit(self.fn(*self.a, **self.kw))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TaskRunner:
    """Keeps a QThread + Worker alive for the duration of one task."""

    def __init__(self, parent):
        self.parent = parent
        self.thread: QThread | None = None
        self.worker: Worker | None = None

    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start(self, fn, on_done, on_log, *a, **kw) -> bool:
        if self.busy():
            return False
        self.thread = QThread()
        self.worker = Worker(fn, *a, **kw)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(on_log)
        self.worker.failed.connect(on_log)
        self.worker.finished.connect(on_done)
        for sig in (self.worker.finished, self.worker.failed):
            sig.connect(self.thread.quit)
        self.thread.start()
        return True


# ------------------------------------------------------------------- hypnogram
class HypnogramCanvas(FigureCanvas):
    """Whole-recording state ribbon + confidence trace, click to seek.

    The x axis is elapsed time (hours), not epoch index -- epoch numbers are an
    implementation detail, whereas time is what you compare against the video and
    the light cycle. Clicks are converted back to an epoch index for the caller.
    """
    seeked = Signal(int)

    def __init__(self):
        self.fig = Figure(figsize=(10, 2.6), constrained_layout=True)
        super().__init__(self.fig)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        self.ax = self.fig.add_subplot(gs[0])
        self.axc = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.cursor = None
        self.cursor_c = None
        self._n = 0
        self._epoch_sec = core.EPOCH_SEC
        self.mpl_connect("button_press_event", self._click)

    def _hours(self, n):
        return np.arange(n) * self._epoch_sec / 3600.0

    def _click(self, ev):
        if ev.inaxes in (self.ax, self.axc) and ev.xdata is not None and self._n:
            epoch = ev.xdata * 3600.0 / self._epoch_sec
            self.seeked.emit(int(np.clip(round(epoch), 0, self._n - 1)))

    def draw_hypnogram(self, labels, confidence=None, manual=None,
                       threshold=None, conf_raw=None, eligible=None):
        """Draw the ribbon and the confidence panel.

        `confidence` is the smoothed trace (legibility); `conf_raw` is the
        unsmoothed per-epoch confidence the threshold actually applies to. Both
        are shown, because a threshold line drawn only against a smoothed trace
        would imply the wrong set of epochs -- smoothing lifts isolated dips above
        the line. `eligible` marks the epochs the scorer's jump will actually
        visit, so the answer to "what gets included" is shown directly rather
        than inferred from where a line crosses a curve.
        """
        self.ax.clear(); self.axc.clear()
        self._n = n = len(labels)
        if n == 0:
            self.draw_idle(); return

        x = self._hours(n)
        y = np.array([STATE_Y.get(s, np.nan) for s in labels], dtype=float)
        self.ax.step(x, y, where="post", lw=0.8, color="#333333")
        for s, yy in STATE_Y.items():
            m = np.array([l == s for l in labels])
            if m.any():
                self.ax.fill_between(x, yy - 0.4, yy + 0.4, where=m,
                                     step="post", color=STATE_COLORS[s],
                                     alpha=0.75, lw=0)
        if manual is not None and np.any(manual):
            self.ax.plot(x[np.flatnonzero(manual)],
                         np.full(int(np.sum(manual)), 2.65), "|",
                         color="#1a7f37", ms=6, mew=1.2)
            # as a right-aligned title, not an in-axes label: at 4 s per epoch a
            # long recording fills the tick row and an overlay would sit on top
            self.ax.set_title(f"green ticks = manually reviewed "
                              f"({int(np.sum(manual))} epochs)",
                              loc="right", fontsize=7, color="#1a7f37", pad=2)
        self.ax.set_yticks(list(STATE_Y.values()))
        self.ax.set_yticklabels(list(STATE_Y.keys()), fontsize=8)
        self.ax.set_ylim(-0.7, 2.8)
        self.ax.set_xlim(0, max(x[-1], self._epoch_sec / 3600.0))
        self.ax.tick_params(labelbottom=False, labelsize=8)
        for sp in ("top", "right"):
            self.ax.spines[sp].set_visible(False)

        if conf_raw is not None and len(conf_raw) == n:
            # raw per-epoch confidence, faint: this is what the threshold tests
            self.axc.fill_between(x, 0, conf_raw, step="post",
                                  color="#4C78A8", alpha=0.22, lw=0)
        if confidence is not None and len(confidence) == n:
            self.axc.plot(x, confidence, color="#2b5d8a", lw=0.9)
        if threshold is not None:
            self.axc.axhline(float(threshold), color="#cc4444", lw=1.0, ls="--")
            self.axc.text(0.004, float(threshold) + 0.03, f"{float(threshold):.2f}",
                          transform=self.axc.get_yaxis_transform(),
                          color="#cc4444", fontsize=7, va="bottom")
        if eligible is not None and np.any(eligible):
            # the epochs the jump will actually visit
            self.axc.plot(x[np.flatnonzero(eligible)],
                          np.full(int(np.sum(eligible)), 0.03), "|",
                          color="#cc4444", ms=5, mew=1.0)
        self.axc.set_ylim(0, 1.02)
        self.axc.set_ylabel("conf.", fontsize=8)
        self.axc.set_xlabel("time (hours)", fontsize=8)
        self.axc.tick_params(labelsize=8)
        for sp in ("top", "right"):
            self.axc.spines[sp].set_visible(False)

        self.cursor = self.ax.axvline(0, color="red", lw=1.2, alpha=0.9)
        self.cursor_c = self.axc.axvline(0, color="red", lw=1.2, alpha=0.9)
        self.draw_idle()

    def set_cursor(self, epoch: int):
        if self.cursor is not None:
            t = epoch * self._epoch_sec / 3600.0
            self.cursor.set_xdata([t]); self.cursor_c.set_xdata([t])
            self.draw_idle()


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Somnus — sleep scoring")
        self.resize(1180, 820)
        self.project: core.Project | None = None
        self.runner = TaskRunner(self)

        # per-recording working state
        self.rec_name: str | None = None
        self.feat: pd.DataFrame | None = None
        self.labels: np.ndarray | None = None
        self.raw: np.ndarray | None = None
        self.proba: np.ndarray | None = None
        self.store: core.LabelStore | None = None
        self.scored: dict = {}
        self._eval_models: dict = {}
        self.queue: np.ndarray = np.array([], dtype=int)
        self.cur_epoch: int = 0

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_project(), "Project")
        self.tabs.addTab(self._tab_score(), "Score")
        self.tabs.addTab(self._tab_review(), "Review && Relabel")
        self.tabs.addTab(self._tab_finetune(), "Fine-tune")
        self.tabs.addTab(self._tab_evaluate(), "Evaluate")
        self.setCentralWidget(self.tabs)

        self.status = self.statusBar()
        self.status.setSizeGripEnabled(True)
        self.status.showMessage("Create or open a project to begin.")
        self._set_enabled(False)

    # ------------------------------------------------------------- log helper
    def log(self, msg: str) -> None:
        for box in (self.p_log, self.s_log, self.f_log):
            box.appendPlainText(msg.rstrip())
        self.status.showMessage(msg.strip().splitlines()[-1][:140])

    def _set_enabled(self, on: bool) -> None:
        for i in (1, 2, 3, 4):
            self.tabs.setTabEnabled(i, on)

    # ------------------------------------------------------------ Project tab
    def _tab_project(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        row = QHBoxLayout()
        b_new = QPushButton("New project…"); b_new.clicked.connect(self.on_new)
        b_open = QPushButton("Open project…"); b_open.clicked.connect(self.on_open)
        b_add = QPushButton("Add recordings from folder…")
        b_add.clicked.connect(self.on_add_folder)
        self.lbl_proj = QLabel("<i>no project</i>")
        for b in (b_new, b_open, b_add):
            row.addWidget(b)
        row.addStretch(1); row.addWidget(self.lbl_proj)
        lay.addLayout(row)

        tick_note = QLabel(
            "<b>Tick the recordings you want to work with.</b> The Score tab "
            "processes everything ticked; fine-tuning uses the ticked "
            "recordings that have manual labels.")
        tick_note.setWordWrap(True)
        lay.addWidget(tick_note)

        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["use", "recording", "labels", "video", "tracking",
             "velocity", "manual epochs"])
        hh = self.tbl.horizontalHeader()
        # Only the recording name stretches; the rest are fixed-ish. Long
        # filenames are elided with the full path on hover, so the table's size
        # hint cannot grow with the longest filename and drag the window wider
        # than the screen.
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setStretchLastSection(False)
        for col, wdt in ((0, 40), (2, 90), (3, 120), (4, 120), (5, 100), (6, 95)):
            self.tbl.setColumnWidth(col, wdt)
        self.tbl.setTextElideMode(Qt.ElideMiddle)
        self.tbl.setWordWrap(False)
        self.tbl.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.tbl.setMinimumWidth(520)          # small hint, so the window resizes
        self.tbl.setSizeAdjustPolicy(QTableWidget.AdjustIgnored)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.itemChanged.connect(self.on_tbl_item_changed)
        self.tbl.itemSelectionChanged.connect(self.on_pick_recording)
        lay.addWidget(self.tbl, 3)

        sp = QHBoxLayout()
        b_all = QPushButton("Select all"); b_all.clicked.connect(
            lambda: self.on_check_all(True))
        b_none = QPushButton("Select none"); b_none.clicked.connect(
            lambda: self.on_check_all(False))
        sp.addWidget(b_all); sp.addWidget(b_none)
        self.lbl_nsel = QLabel(""); sp.addWidget(self.lbl_nsel)
        sp.addStretch(1)
        lay.addLayout(sp)

        note = QLabel(
            "Source recordings are opened <b>read-only</b>. Every output "
            "(labels, cache, models) is written inside the project folder. "
            "<i>velocity</i> needs both a tracking file and its frame "
            "timestamps — without the timestamps the feature is withheld rather "
            "than guessed, because dropped frames make a constant frame rate "
            "wrong by minutes.")
        note.setWordWrap(True); note.setStyleSheet("color:#555;")
        lay.addWidget(note)

        self.p_log = QPlainTextEdit(); self.p_log.setReadOnly(True)
        self.p_log.setMaximumHeight(120); lay.addWidget(self.p_log, 1)
        return w

    # -------------------------------------------------------------- Score tab
    def _tab_score(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        box = QGroupBox("Scoring options"); form = QFormLayout(box)
        self.cb_decode = QCheckBox("Apply HMM temporal smoothing")
        self.cb_decode.setChecked(True)
        self.cb_decode.setToolTip(
            "On: find the most likely state *sequence*, so isolated one-epoch "
            "flickers get absorbed.\nOff: take the raw per-epoch model output.")
        self.cb_decode.toggled.connect(
            lambda on: self.sp_stick.setEnabled(on))
        form.addRow(self.cb_decode)

        self.sp_stick = QDoubleSpinBox()
        self.sp_stick.setRange(0.0, 20.0); self.sp_stick.setSingleStep(0.25)
        self.sp_stick.setValue(1.0); self.sp_stick.setDecimals(2)
        self.sp_stick.setToolTip(
            "Resistance to state changes.\n"
            "0 = no inertia (same as smoothing off)\n"
            "1 = transition matrix as estimated from scored data\n"
            ">1 = longer, cleaner bouts, but short REM bouts may be swallowed")
        form.addRow("Transition resistance", self.sp_stick)
        hint = QLabel("Higher values give fewer, longer bouts. If you study "
                      "fragmentation, keep this low and check the hypnogram.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#555;")
        form.addRow(hint)
        lay.addWidget(box)

        qlab = QLabel("<b>Queue</b> — the recordings ticked on the Project tab")
        qlab.setWordWrap(True)
        lay.addWidget(qlab)
        self.lst_score = QListWidget()
        self.lst_score.setMaximumHeight(150)
        lay.addWidget(self.lst_score)

        row = QHBoxLayout()
        self.b_score = QPushButton("Score queue")
        self.b_score.clicked.connect(self.on_score)
        row.addWidget(self.b_score)
        self.lbl_score = QLabel("")
        row.addWidget(self.lbl_score); row.addStretch(1)
        lay.addLayout(row)

        self.bar = QProgressBar(); self.bar.setRange(0, 0); self.bar.hide()
        lay.addWidget(self.bar)

        self.s_summary = QPlainTextEdit(); self.s_summary.setReadOnly(True)
        self.s_summary.setMaximumHeight(190)
        f = QFont("Menlo"); f.setPointSize(11); self.s_summary.setFont(f)
        lay.addWidget(QLabel("Result")); lay.addWidget(self.s_summary)

        self.s_log = QPlainTextEdit(); self.s_log.setReadOnly(True)
        lay.addWidget(QLabel("Log")); lay.addWidget(self.s_log, 1)
        return w

    # ------------------------------------------------------------- Review tab
    def _tab_review(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel("Recording:"))
        self.cmb_review = QComboBox()
        self.cmb_review.setMinimumWidth(200)
        self.cmb_review.currentTextChanged.connect(self.load_for_review)
        top.addWidget(self.cmb_review)
        top.addWidget(QLabel("  smoothing:"))
        self.sp_smooth = QDoubleSpinBox()
        self.sp_smooth.setRange(1, 301); self.sp_smooth.setSingleStep(2)
        self.sp_smooth.setDecimals(0); self.sp_smooth.setValue(15)
        self.sp_smooth.setSuffix(" epochs")
        self.sp_smooth.setToolTip(
            "Rolling median applied to the confidence trace for legibility "
            "(15 epochs = 1 min). Display only — never fed back into scoring.")
        self.sp_smooth.valueChanged.connect(lambda _: self.rebuild_review())
        top.addWidget(self.sp_smooth)
        top.addWidget(QLabel("  low-certainty thr:"))
        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(0.0, 1.0); self.sp_thr.setSingleStep(0.05)
        self.sp_thr.setDecimals(2); self.sp_thr.setValue(0.80)
        self.sp_thr.setToolTip(
            "An epoch counts as low-certainty when the model's top class "
            "probability is below this. Used by the scorer's 'Next low "
            "certainty' button; adjustable in the scorer with [ and ].")
        self.sp_thr.valueChanged.connect(lambda _: self.rebuild_review())
        top.addWidget(self.sp_thr)
        top.addStretch(1)
        lay.addLayout(top)

        self.hyp = HypnogramCanvas()
        self.hyp.seeked.connect(self.goto_epoch)
        lay.addWidget(self.hyp, 3)

        self.lbl_epoch = QLabel("—")
        self.lbl_epoch.setStyleSheet("font-size:13px;")
        lay.addWidget(self.lbl_epoch)

        box = QGroupBox("Relabel in the Somnus scorer")
        v = QVBoxLayout(box)
        blurb = QLabel(
            "Opens the full scorer on this recording, pre-loaded with the "
            "model's labels so you correct rather than score from scratch. Its "
            "side panel shows the <b>model's belief averaged over the window on "
            "screen</b>, and the bout navigator gains a <b>Next low certainty</b> "
            "button (<b>u</b> forward, <b>U</b> back; <b>[</b>/<b>]</b> adjust "
            "the threshold). Epochs the HMM smoothing changed are marked in gold "
            "and skipped by that walk. Use the <b>Confirm</b> brush (key <b>6</b>) to affirm a label the model got right without changing it — it renders more opaque, and <b>Erase</b> clears it. Corrections and confirmations are read back when you close it.")
        blurb.setWordWrap(True)
        v.addWidget(blurb)
        rowb = QHBoxLayout()
        self.b_launch = QPushButton("Open scorer for this recording")
        self.b_launch.clicked.connect(self.on_launch_scorer)
        rowb.addWidget(self.b_launch)
        self.b_reload = QPushButton("Reload corrections")
        self.b_reload.clicked.connect(self.on_reload_corrections)
        rowb.addWidget(self.b_reload)
        rowb.addStretch(1)
        v.addLayout(rowb)
        self.lbl_launch = QLabel(""); self.lbl_launch.setWordWrap(True)
        self.lbl_launch.setStyleSheet("color:#555;")
        v.addWidget(self.lbl_launch)
        lay.addWidget(box)

        self.lbl_qinfo = QLabel(""); self.lbl_qinfo.setWordWrap(True)
        self.lbl_qinfo.setStyleSheet("color:#555;")
        lay.addWidget(self.lbl_qinfo)
        return w

    # ----------------------------------------------------------- Fine-tune tab
    def _tab_finetune(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        info = QLabel(
            "Fine-tuning adapts the model to <b>your</b> recordings without "
            "retraining from scratch, so a phenotype is not diluted by the "
            "normal-sleep training set. It trains only on <b>manually sourced</b> "
            "labels; model predictions are never used as training targets.")
        info.setWordWrap(True); lay.addWidget(info)

        box = QGroupBox("Settings"); form = QFormLayout(box)
        self.cmb_lam = QComboBox()
        self.cmb_lam.addItems(["auto (cross-validated)", "1000", "300", "100",
                               "30", "10", "3", "1"])
        self.cmb_lam.setToolTip(
            "Anchor strength. Large = stay close to the base model; small = "
            "trust your data more.\n'auto' picks it by leave-one-recording-out "
            "cross-validation on your labels.")
        form.addRow("Adaptation strength (λ)", self.cmb_lam)
        self.cb_adaptA = QCheckBox("Also adapt the transition matrix "
                                   "(sleep architecture)")
        self.cb_adaptA.setChecked(True)
        form.addRow(self.cb_adaptA)
        lay.addWidget(box)

        row = QHBoxLayout()
        self.b_ft = QPushButton("Fine-tune on manually labeled epochs")
        self.b_ft.clicked.connect(self.on_finetune)
        row.addWidget(self.b_ft)
        self.lbl_ft = QLabel(""); row.addWidget(self.lbl_ft); row.addStretch(1)
        lay.addLayout(row)

        self.bar_ft = QProgressBar(); self.bar_ft.setRange(0, 0); self.bar_ft.hide()
        lay.addWidget(self.bar_ft)

        self.f_log = QPlainTextEdit(); self.f_log.setReadOnly(True)
        f = QFont("Menlo"); f.setPointSize(11); self.f_log.setFont(f)
        lay.addWidget(self.f_log, 1)
        return w

    # ------------------------------------------------------------ Evaluate tab
    def _tab_evaluate(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        info = QLabel(
            "Runs on the recordings <b>ticked on the Project tab</b>. "
            "<b>Compare models</b> scores each one with the base model and with "
            "another model, and measures both against your manual labels. "
            "<b>Architecture</b> summarizes sleep structure per recording — the "
            "numbers that go in a paper.")
        info.setWordWrap(True); lay.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("compare against:"))
        self.cmb_evalmodel = QComboBox(); self.cmb_evalmodel.setMinimumWidth(200)
        row.addWidget(self.cmb_evalmodel)
        b_ref = QPushButton("↻"); b_ref.setFixedWidth(30)
        b_ref.setToolTip("rescan the project's models/ folder")
        b_ref.clicked.connect(self.refresh_eval_models)
        row.addWidget(b_ref)
        self.b_cmp = QPushButton("Compare models")
        self.b_cmp.clicked.connect(self.on_compare)
        row.addWidget(self.b_cmp)
        self.b_arch = QPushButton("Architecture breakdown")
        self.b_arch.clicked.connect(self.on_architecture)
        row.addWidget(self.b_arch)
        self.b_exp = QPushButton("Export table…")
        self.b_exp.clicked.connect(self.on_export_table)
        row.addWidget(self.b_exp)
        row.addStretch(1)
        lay.addLayout(row)

        self.bar_ev = QProgressBar(); self.bar_ev.setRange(0, 0); self.bar_ev.hide()
        lay.addWidget(self.bar_ev)

        self.tbl_ev = QTableWidget(0, 0)
        self.tbl_ev.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.tbl_ev.setMinimumWidth(400)
        self.tbl_ev.setSizeAdjustPolicy(QTableWidget.AdjustIgnored)
        lay.addWidget(self.tbl_ev, 3)

        self.lbl_ev = QLabel(""); self.lbl_ev.setWordWrap(True)
        self.lbl_ev.setStyleSheet("color:#555;")
        lay.addWidget(self.lbl_ev)
        self._ev_table: pd.DataFrame | None = None
        return w

    def refresh_eval_models(self):
        if not self.project:
            return
        import glob
        cur = self.cmb_evalmodel.currentText()
        found = sorted(glob.glob(os.path.join(self.project.models_dir, "*.json")))
        self.cmb_evalmodel.clear()
        self.cmb_evalmodel.addItems([os.path.basename(p) for p in found]
                                    or ["(no fine-tuned models yet)"])
        if cur:
            i = self.cmb_evalmodel.findText(cur)
            if i >= 0:
                self.cmb_evalmodel.setCurrentIndex(i)
        self._eval_models = {os.path.basename(p): p for p in found}

    def _show_table(self, df: pd.DataFrame):
        self._ev_table = df
        self.tbl_ev.clear()
        self.tbl_ev.setRowCount(len(df)); self.tbl_ev.setColumnCount(len(df.columns))
        self.tbl_ev.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i in range(len(df)):
            for j, c in enumerate(df.columns):
                v = df.iloc[i][c]
                txt = f"{v:.4f}" if isinstance(v, float) else str(v)
                it = QTableWidgetItem(txt)
                if not isinstance(v, str):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_ev.setItem(i, j, it)

    def on_compare(self):
        queue = self.checked()
        if not queue or self.runner.busy():
            QMessageBox.information(self, "Nothing selected",
                                    "Tick recordings on the Project tab first.")
            return
        name = self.cmb_evalmodel.currentText()
        other = getattr(self, "_eval_models", {}).get(name)
        if not other:
            QMessageBox.information(
                self, "No model to compare",
                "Fine-tune a model first — it will appear in this list.")
            return
        base = core.DEFAULT_ARTIFACT
        proj = self.project
        decode = self.cb_decode.isChecked()
        stick = float(self.sp_stick.value())

        def job(log):
            import json
            from somnus.predict import load_model
            # which recordings did this model train on? evaluating on those is
            # in-sample and must not be read as generalization
            try:
                trained_on = set(json.load(open(other))
                                 .get("finetune", {}).get("recordings", []))
            except Exception:
                trained_on = set()
            rows = []
            for i, r in enumerate(queue, 1):
                st = core.LabelStore(proj, r.name)
                hu = st.df[st.manual_mask()]
                if not len(hu):
                    log(f"[{i}/{len(queue)}] {r.name}: no manual labels, skipped")
                    continue
                log(f"[{i}/{len(queue)}] {r.name}: {len(hu)} manual epochs …")
                cache = os.path.join(proj.cache_dir, r.name + "_features.csv")
                feat = core.featurize(r, cache=cache)
                truth = hu.set_index("epoch")["state"]
                idx = truth.index.to_numpy()
                idx = idx[(idx >= 0) & (idx < len(feat))]
                y = truth.loc[idx].to_numpy()
                rec_row = {"recording": r.name, "manual_epochs": len(idx)}
                for tag, path in (("base", base), ("compare", other)):
                    lab, _ = core.score(feat, path, decode=decode,
                                        stickiness=stick)
                    p = lab[idx]
                    rec_row[f"{tag}_acc"] = float((p == y).mean())
                    recs = []
                    for s in core.STATES:
                        m = y == s
                        if m.any():
                            recs.append(float((p[m] == s).mean()))
                    rec_row[f"{tag}_balanced"] = float(np.mean(recs)) if recs else float("nan")
                rec_row["delta_acc"] = rec_row["compare_acc"] - rec_row["base_acc"]
                rec_row["in_sample"] = "YES" if r.name in trained_on else ""
                rows.append(rec_row)
            if not rows:
                raise RuntimeError(
                    "No ticked recording has manual labels. Relabel some epochs "
                    "in the Review tab first — comparison needs ground truth.")
            return dict(kind="compare", df=pd.DataFrame(rows),
                        model=name, trained_on=trained_on)

        self.bar_ev.show(); self.b_cmp.setEnabled(False)
        self.runner.start(job, self.after_eval, self.log)

    def on_architecture(self):
        queue = self.checked()
        if not queue or self.runner.busy():
            QMessageBox.information(self, "Nothing selected",
                                    "Tick recordings on the Project tab first.")
            return
        proj = self.project
        model = proj.model
        decode = self.cb_decode.isChecked()
        stick = float(self.sp_stick.value())

        def job(log):
            rows = []
            for i, r in enumerate(queue, 1):
                log(f"[{i}/{len(queue)}] {r.name} …")
                cache = os.path.join(proj.cache_dir, r.name + "_features.csv")
                feat = core.featurize(r, cache=cache)
                lab, _ = core.score(feat, model, decode=decode, stickiness=stick)
                # prefer manual labels wherever they exist
                st = core.LabelStore(proj, r.name)
                n_hu = 0
                if len(st.df):
                    hu = st.df[st.manual_mask()]
                    for e, s in zip(hu["epoch"].to_numpy(), hu["state"].to_numpy()):
                        if 0 <= int(e) < len(lab):
                            lab[int(e)] = s; n_hu += 1
                a = core.architecture(lab)
                rows.append({"recording": r.name, "manual_epochs": n_hu, **a})
            return dict(kind="architecture", df=pd.DataFrame(rows))

        self.bar_ev.show(); self.b_arch.setEnabled(False)
        self.runner.start(job, self.after_eval, self.log)

    @Slot(object)
    def after_eval(self, res):
        self.bar_ev.hide()
        self.b_cmp.setEnabled(True); self.b_arch.setEnabled(True)
        if not isinstance(res, dict):
            return
        df = res["df"]
        self._show_table(df)
        if res["kind"] == "compare":
            n_in = int((df["in_sample"] == "YES").sum())
            better = int((df["delta_acc"] > 0).sum())
            mean_d = float(df["delta_acc"].mean())
            msg = (f"{res['model']} vs base on {len(df)} recording(s), scored "
                   f"against manual labels only. Mean accuracy change "
                   f"<b>{mean_d:+.4f}</b>; better on {better}/{len(df)}.")
            if n_in:
                msg += (f" <b>{n_in} recording(s) are marked in_sample</b> — the "
                        f"compared model was fine-tuned on them, so those rows "
                        f"measure fit, not generalization. Judge the model on "
                        f"the rows without that flag.")
            else:
                msg += (" No recording here was used to fine-tune that model, so "
                        "these are out-of-sample.")
            self.lbl_ev.setText(msg)
        else:
            tot = df["recording_hours"].sum() if "recording_hours" in df else 0
            self.lbl_ev.setText(
                f"{len(df)} recording(s), {tot:.1f} h total. Where you have "
                f"manual labels they replace the model's, so the numbers reflect "
                f"your best current scoring (the <i>manual_epochs</i> column says "
                f"how many). Use <b>Export table…</b> for the CSV.")
        self.log(f"evaluation done ({res['kind']})")

    def on_export_table(self):
        if self._ev_table is None or not len(self._ev_table):
            QMessageBox.information(self, "Nothing to export",
                                    "Run a comparison or breakdown first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save table", os.path.join(self.project.path, "evaluation.csv"),
            "CSV (*.csv)")
        if path:
            self._ev_table.to_csv(path, index=False)
            self.log(f"wrote {path}")

    # ------------------------------------------------------------- project ops
    def on_new(self):
        d = QFileDialog.getExistingDirectory(self, "Choose an empty folder for "
                                                  "the new project")
        if not d:
            return
        name, ok = QInputDialog.getText(
            self, "Name this project", "Project name:",
            text=os.path.basename(d.rstrip("/")) or "somnus_project")
        if not ok:
            return
        name = name.strip() or os.path.basename(d.rstrip("/"))
        self.project = core.Project.create(d, name=name)
        self.refresh_project(); self.refresh_score_list()
        self.refresh_eval_models()
        self.log(f"created project '{name}' at {d}")

    def on_open(self):
        d = QFileDialog.getExistingDirectory(self, "Open project folder")
        if not d:
            return
        try:
            self.project = core.Project.load(d)
        except Exception as e:
            QMessageBox.critical(self, "Cannot open", str(e)); return
        self.refresh_project(); self.refresh_score_list()
        self.refresh_eval_models()
        self.log(f"opened project {d} ({len(self.project.recordings)} recordings)")

    def on_add_folder(self):
        if not self.project:
            return
        d = QFileDialog.getExistingDirectory(self, "Folder containing EDF "
                                                  "recordings (read-only)")
        if not d:
            return
        found = core.discover_recordings(d)
        have = {r.name for r in self.project.recordings}
        new = [r for r in found if r.name not in have]
        self.project.recordings.extend(new)
        self.project.save()
        self.refresh_project(); self.refresh_score_list()
        self.log(f"added {len(new)} recording(s) from {d}")

    def refresh_project(self):
        p = self.project
        self.lbl_proj.setText(f"<b>{p.name}</b> — {p.path}")
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(0)
        for r in p.recordings:
            store = core.LabelStore(p, r.name)
            row = self.tbl.rowCount(); self.tbl.insertRow(row)

            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
                         | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if r.selected else Qt.Unchecked)
            self.tbl.setItem(row, 0, chk)

            def path_cell(path):
                it = QTableWidgetItem(os.path.basename(path) if path else "—")
                if path:
                    it.setToolTip(path)
                else:
                    it.setForeground(Qt.gray)
                return it

            self.tbl.setItem(row, 1, QTableWidgetItem(r.name))
            self.tbl.setItem(row, 2, path_cell(r.scored))
            self.tbl.setItem(row, 3, path_cell(r.video))
            self.tbl.setItem(row, 4, path_cell(r.coords))
            vel = QTableWidgetItem("yes" if r.has_velocity else "no")
            vel.setTextAlignment(Qt.AlignCenter)
            if r.coords and not r.timestamps:
                vel.setText("no timestamps")
                vel.setToolTip(
                    "Tracking found, but no *_timestamps.npy in this folder.\n\n"
                    "Only this folder is searched — a stale timestamps file from "
                    "elsewhere would pair positions with the wrong times, and "
                    "because the cameras dropped frames that misaligns tracking "
                    "from the EEG by minutes, silently.\n\n"
                    "Scoring this recording will ERROR rather than guess. Either "
                    "put the matching timestamps file beside the coordinates, or "
                    "remove the coordinates to score without velocity.")
            self.tbl.setItem(row, 5, vel)
            nh = QTableWidgetItem(str(store.n_manual()))
            nh.setTextAlignment(Qt.AlignCenter)
            self.tbl.setItem(row, 6, nh)
        self.tbl.blockSignals(False)
        self._update_nsel()
        self._set_enabled(bool(p.recordings))

    def on_tbl_item_changed(self, item: QTableWidgetItem):
        if item.column() != 0 or not self.project:
            return
        row = item.row()
        if row < len(self.project.recordings):
            self.project.recordings[row].selected = \
                item.checkState() == Qt.Checked
            self.project.save()
            self._update_nsel()
            self.refresh_score_list()

    def on_check_all(self, on: bool):
        if not self.project:
            return
        for r in self.project.recordings:
            r.selected = on
        self.project.save(); self.refresh_project(); self.refresh_score_list()

    def _update_nsel(self):
        n = len(self.checked())
        self.lbl_nsel.setText(f"<b>{n}</b> of {len(self.project.recordings)} "
                              f"ticked" if self.project else "")

    def checked(self) -> list[core.Recording]:
        return [r for r in self.project.recordings if r.selected] \
            if self.project else []

    def _selected(self) -> core.Recording | None:
        """The row the cursor is on (used for Review, which is one at a time)."""
        rows = {i.row() for i in self.tbl.selectedIndexes()}
        if not rows or not self.project:
            return None
        return self.project.recordings[sorted(rows)[0]]

    def on_pick_recording(self):
        r = self._selected()
        if r:
            self.lbl_score.setText(f"cursor on: <b>{r.name}</b>")

    # ---------------------------------------------------------------- scoring
    def refresh_score_list(self):
        self.lst_score.clear()
        for r in self.checked():
            bits = [r.name]
            if not r.has_velocity:
                bits.append("(no velocity)")
            self.lst_score.addItem("   ".join(bits))
        n = self.lst_score.count()
        self.b_score.setText(f"Score queue ({n})" if n else "Score queue")
        self.lbl_score.setText("" if n else
                               "<i>nothing ticked on the Project tab</i>")

    def on_score(self):
        queue = self.checked()
        if not queue:
            QMessageBox.information(
                self, "Nothing to score",
                "Tick one or more recordings on the Project tab first.")
            return
        if self.runner.busy():
            return
        decode = self.cb_decode.isChecked()
        stick = float(self.sp_stick.value())
        proj, model = self.project, self.project.model

        def job(log):
            results = []
            for i, r in enumerate(queue, 1):
                log(f"[{i}/{len(queue)}] {r.name}: featurizing …")
                cache = os.path.join(proj.cache_dir, r.name + "_features.csv")
                try:
                    feat = core.featurize(r, cache=cache)
                except Exception as e:
                    log(f"    SKIPPED — {type(e).__name__}: {e}")
                    continue
                log(f"    {len(feat)} epochs; scoring "
                    f"(smoothing={'on' if decode else 'off'}, "
                    f"resistance={stick:g}) …")
                lab, proba = core.score(feat, model, decode=decode,
                                       stickiness=stick)
                raw, _ = core.score(feat, model, decode=False)

                store = core.LabelStore(proj, r.name)
                n_new = store.set_model_labels(
                    feat["epoch"].to_numpy(), feat["t_start"].to_numpy(),
                    lab, proba.max(axis=1), model)
                if r.scored and store.n_manual() == 0:
                    k = store.import_manual(r.scored, len(feat))
                    if k:
                        log(f"    imported {k} existing manual labels")
                store.save()
                log(f"    wrote {n_new} model rows "
                    f"({store.n_manual()} manual rows preserved)")
                results.append(dict(rec=r, feat=feat, labels=lab, raw=raw,
                                    proba=proba, store=store))
            if not results:
                raise RuntimeError("nothing scored — see the log above")
            return results

        self.bar.show(); self.b_score.setEnabled(False)
        self.runner.start(job, self.after_score, self.log)

    @Slot(object)
    def after_score(self, res):
        self.bar.hide(); self.b_score.setEnabled(True)
        if not isinstance(res, list) or not res:
            return
        self.scored = {d["rec"].name: d for d in res}

        lines = []
        for d in res:
            r, lab, raw = d["rec"], d["labels"], d["raw"]
            arch = core.architecture(lab)
            over = int((lab != raw).sum())
            # Report velocity from the features actually computed, not from what
            # discovery found: the feature pipeline runs its own timestamp lookup
            # (validated by exact array-length match), so it can succeed where the
            # folder-only scan in the Project tab found nothing.
            vel = ("log_velocity" in d["feat"].columns
                   and bool(np.isfinite(d["feat"]["log_velocity"]
                                        .to_numpy(dtype=float)).any()))
            lines += [
                f"{r.name}: {len(lab)} epochs ({arch['recording_hours']:.2f} h)"
                f"{'   [velocity used]' if vel else '   [no velocity]'}",
                f"   Wake {arch['pct_Wake']:5.1f}%  NREM {arch['pct_NREM']:5.1f}%"
                f"  REM {arch['pct_REM']:5.1f}%",
                f"   bouts  W {arch['bouts_Wake']}  N {arch['bouts_NREM']}  "
                f"R {arch['bouts_REM']}   mean REM bout "
                f"{arch['mean_bout_s_REM']:.0f}s",
                f"   smoothing changed {over} epochs "
                f"({100*over/max(len(lab),1):.2f}%)   "
                f"QC flags {len(core.qc_flags(lab))}",
            ]
            truth = d["feat"]["state"].to_numpy() \
                if "state" in d["feat"] else None
            if truth is not None:
                m = np.isin(truth, core.STATES)
                if m.any():
                    lines.append(
                        f"   agreement with existing labels "
                        f"{(lab[m]==truth[m]).mean():.4f} on {int(m.sum())} epochs")
            lines.append("")
        self.s_summary.setPlainText("\n".join(lines))
        self.log(f"scored {len(res)} recording(s)")

        # load the first result into the Review tab
        self.load_for_review(res[0]["rec"].name)
        self.refresh_project()
        self.tabs.setCurrentIndex(2)

    def load_for_review(self, name: str):
        d = getattr(self, "scored", {}).get(name)
        if not d:
            return
        self.rec_name = name
        self.feat, self.labels = d["feat"], d["labels"]
        self.raw, self.proba = d["raw"], d["proba"]
        self.store = d["store"]
        self.cmb_review.blockSignals(True)
        self.cmb_review.clear()
        self.cmb_review.addItems(list(self.scored.keys()))
        self.cmb_review.setCurrentText(name)
        self.cmb_review.blockSignals(False)
        self.rebuild_review()

    # ----------------------------------------------------------------- review
    def _manual_mask(self) -> np.ndarray:
        manual = np.zeros(len(self.labels), dtype=bool)
        if self.store is not None and len(self.store.df):
            h = self.store.df.loc[self.store.manual_mask(), "epoch"].to_numpy()
            h = h[(h >= 0) & (h < len(manual))]
            manual[h.astype(int)] = True
        return manual

    def rebuild_review(self):
        if self.labels is None:
            return
        manual = self._manual_mask()
        conf_raw = self.proba.max(axis=1)
        conf = core.smooth_trace(conf_raw, int(self.sp_smooth.value()))
        thr = float(self.sp_thr.value())
        # same criteria as the scorer's jump, so the ticks match what it visits
        smoothed = self.labels != self.raw
        eligible = (conf_raw < thr) & ~smoothed
        self.hyp.draw_hypnogram(self.labels, conf, manual, threshold=thr,
                                conf_raw=conf_raw, eligible=eligible)
        n_over = int(smoothed.sum())
        n_low = int(eligible.sum())
        self.lbl_qinfo.setText(
            f"{len(self.labels)} epochs. <b>{n_low}</b> fall below the "
            f"{thr:.2f} certainty threshold — the scorer's <i>Next low "
            f"certainty</i> button walks these in time order. <b>{n_over}</b> "
            f"were changed by the HMM smoothing; those get their own color in "
            f"the scorer and are deliberately <i>excluded</i> from that walk, "
            f"since smoothing changing a label is not the same as the model "
            f"being unsure. {int(manual.sum())} epochs are manually labeled.")
        self.goto_epoch(self.cur_epoch if self.cur_epoch < len(self.labels) else 0)

    def goto_epoch(self, epoch: int):
        if self.labels is None or not len(self.labels):
            return
        self.cur_epoch = int(np.clip(epoch, 0, len(self.labels) - 1))
        self.hyp.set_cursor(self.cur_epoch)
        e = self.cur_epoch
        src = "—"
        if self.store is not None:
            row = self.store.df[self.store.df["epoch"] == e]
            if len(row):
                src = str(row["source"].iloc[0])
        t = e * core.EPOCH_SEC
        sm = " | <b style='color:#c8a000'>HMM-smoothed</b>" \
            if self.labels[e] != self.raw[e] else ""
        self.lbl_epoch.setText(
            f"epoch <b>{e}</b>  |  t = {t:.0f}s ({t/3600:.2f} h)  |  "
            f"label <b style='color:{STATE_COLORS.get(self.labels[e],'#000')}'>"
            f"{self.labels[e]}</b>  |  raw {self.raw[e]}  |  "
            f"confidence {self.proba[e].max():.3f}  |  source {src}{sm}")

    def on_launch_scorer(self):
        """Write the scorer's inputs into the project, then launch it."""
        if not self.rec_name or self.project is None:
            return
        r = self.project.get(self.rec_name)
        if r is None:
            return
        scored, meta = core.write_viewer_bundle(
            self.project, r, self.labels, self.proba, self.raw, self.store)

        # The scorer is part of this package; launch it as a module so the
        # hand-off works identically from a checkout and an installed wheel.
        import subprocess
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "somnus.scorer", r.edf, scored, meta,
                 f"{float(self.sp_thr.value()):.4f}"])
        except Exception as e:
            QMessageBox.critical(self, "Could not launch scorer", str(e)); return
        self._scorer_proc = proc
        self.lbl_launch.setText(
            f"Scorer launched (pid {proc.pid}). It is editing "
            f"<code>{os.path.basename(scored)}</code> <b>inside the project</b>, "
            f"not your source folder. Press Enter in the scorer to save, close "
            f"it, then click <b>Reload corrections</b>.")
        self.log(f"launched scorer on {r.name} (pid {proc.pid})")

    def on_reload_corrections(self):
        if not self.rec_name or self.project is None or self.store is None:
            return
        r = self.project.get(self.rec_name)
        res = core.read_viewer_labels(self.project, r, self.store,
                                      len(self.labels))
        self.store.save()
        # reflect the corrections in the hypnogram
        if len(self.store.df):
            h = self.store.df[self.store.manual_mask()]
            for e, s in zip(h["epoch"].to_numpy(), h["state"].to_numpy()):
                if 0 <= int(e) < len(self.labels):
                    self.labels[int(e)] = s
        self.rebuild_review(); self.refresh_project()
        msg = (f"pulled in {res['corrected']} corrected and "
               f"{res.get('confirmed', 0)} confirmed epoch(s), "
               f"{res['excluded']} marked unscorable, "
               f"{res.get('reverted', 0)} reverted to the model "
               f"(edited then changed back). "
               f"{self.store.n_manual()} labels now usable for fine-tuning, "
               f"{self.store.n_excluded()} excluded. "
               f"Only epochs whose label CHANGED are counted — an untouched "
               f"epoch comes back identical to the model's own prediction, and "
               f"treating those as manual labels would train the model on itself.")
        self.lbl_launch.setText(msg)
        self.log(f"reloaded: {res['corrected']} corrected, "
                 f"{res.get('confirmed', 0)} confirmed, "
                 f"{res['excluded']} excluded, "
                 f"{res.get('reverted', 0)} reverted")

    # -------------------------------------------------------------- fine-tune
    def on_finetune(self):
        if not self.project or self.runner.busy():
            return
        proj = self.project
        model = proj.model
        sel = self.cmb_lam.currentText()
        lam = None if sel.startswith("auto") else float(sel)
        adapt = self.cb_adaptA.isChecked()

        def job(log):
            from somnus.train import finetune as F
            from somnus.predict import load_model
            frames = []
            comp = []
            picked = [r for r in proj.recordings if r.selected] or proj.recordings
            for r in picked:
                st = core.LabelStore(proj, r.name)
                if st.n_manual() == 0:
                    continue
                cache = os.path.join(proj.cache_dir, r.name + "_features.csv")
                if not os.path.exists(cache):
                    log(f"featurizing {r.name} …")
                feat = core.featurize(r, cache=cache)
                hu = st.df[st.manual_mask()][["epoch", "state"]]
                m = feat.merge(hu, on="epoch", suffixes=("_old", ""))
                if "state_old" in m.columns:
                    m = m.drop(columns=["state_old"])
                m["recording"] = r.name
                frames.append(m)
                comp.append((r.name, st.n_corrected(), st.n_confirmed(),
                             st.n_imported()))
                log(f"  {r.name}: {len(m)} manual epochs "
                    f"({st.n_corrected()} corrected, {st.n_confirmed()} confirmed, "
                    f"{st.n_imported()} imported)")
            if not frames:
                raise RuntimeError(
                    "No manually labeled epochs found. Review some epochs first "
                    "(Review tab), or import existing manual scoring.")
            df = pd.concat(frames, ignore_index=True)
            tc = sum(c for _, c, _, _ in comp)
            tf = sum(f for _, _, f, _ in comp)
            ti = sum(i for _, _, _, i in comp)
            log(f"\ntraining-set composition: {tc} corrected, {tf} confirmed, "
                f"{ti} imported  (total {tc + tf + ti})")
            if ti > 2 * (tc + tf) and ti:
                log("  NOTE: imported pre-existing scoring dominates. Those are "
                    "manual labels, but they are not what you reviewed in this "
                    "session, so gains mostly reflect training on that scoring.")
            if tf and not tc:
                log("  NOTE: confirmations only, no corrections. Confirming an "
                    "epoch the model was already sure about adds little gradient; "
                    "the value is in confirming LOW-confidence epochs.")
            art = load_model(model)
            log("")
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = F.finetune(art, df, lam=lam, adapt_A=adapt, verbose=True)
            log(buf.getvalue())
            n_existing = len([f for f in os.listdir(proj.models_dir)
                              if f.startswith("finetuned") and f.endswith(".json")])
            out = os.path.join(proj.models_dir,
                               f"finetuned_{n_existing + 1:02d}.json")
            import json
            with open(out, "w") as fh:
                json.dump(res["artifact"], fh, indent=2)
            return dict(path=out, improved=res["improved"], lam=res["lam"])

        self.bar_ft.show(); self.b_ft.setEnabled(False)
        self.f_log.clear()
        self.runner.start(job, self.after_finetune, self.log)

    @Slot(object)
    def after_finetune(self, res):
        self.bar_ft.hide(); self.b_ft.setEnabled(True)
        if not isinstance(res, dict):
            return
        self.refresh_eval_models()
        msg = (f"Fine-tuned model written to:\n{res['path']}\n\n"
               f"chosen λ = {res['lam']:g}\n")
        if res["improved"]:
            msg += ("It beat the base model on held-out recordings. You can set "
                    "it as the active model.")
            if QMessageBox.question(
                    self, "Use the fine-tuned model?",
                    msg + "\n\nMake it active now?") == QMessageBox.Yes:
                self.project.model = res["path"]; self.project.save()
                self.log(f"active model -> {res['path']}")
        else:
            msg += ("It did NOT beat the base model out-of-sample, so the base "
                    "model is the better choice on current evidence. Collect "
                    "more corrected epochs and try again.")
            QMessageBox.information(self, "Fine-tune finished", msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
