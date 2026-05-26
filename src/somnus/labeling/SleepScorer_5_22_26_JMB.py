import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import mne
from scipy.signal import welch
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons, CheckButtons, Slider

try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# --- Config ---
DEFAULT_WAKE_PERSISTENCE = 5.0
DEFAULT_SLEEP_PERSISTENCE = 5.0
DEFAULT_REM_DELAY = 30.0  

# --- Tunable Frequency Bands ---
DEFAULT_DELTA_BAND = [0.5, 4]
DEFAULT_THETA_BAND = [5, 10]

class InteractiveSleepScorer:
    def __init__(self, best_eeg, all_eegs, emg_data, sfreq, edf_path, video_path=None, amp_thresh=2e-3, sync_thresh=0.90):
        self.eeg = best_eeg
        self.all_eegs = all_eegs  # Shape: (n_channels, n_samples)
        self.emg = emg_data
        self.sfreq = sfreq
        self.edf_path = edf_path
        self.video_path = video_path
        
        # --- Artifact Thresholds ---
        self.amp_thresh = amp_thresh  # Default 2 mV
        self.sync_thresh = sync_thresh 
        self.exclude_artifacts = True # Default state
        
        # --- Safely Check Video Path ---
        self.valid_video = False
        if self.video_path and os.path.exists(self.video_path):
            tmp_cap = cv2.VideoCapture(self.video_path)
            if tmp_cap.isOpened():
                self.valid_video = True
            tmp_cap.release()
        
        # --- Frequency Bands ---
        self.DELTA_BAND = DEFAULT_DELTA_BAND   
        self.THETA_BAND = DEFAULT_THETA_BAND   
        self.EMG_BAND =[30, min(300, sfreq/2 - 1)] 
        
        # --- Timing Params ---
        self.STEP_SEC = 0.5  
        self.SMOOTH_WIN_SEC = 2.5
        self.wake_persistence = DEFAULT_WAKE_PERSISTENCE
        self.sleep_persistence = DEFAULT_SLEEP_PERSISTENCE
        self.rem_delay = DEFAULT_REM_DELAY
        
        # --- State Logic ---
        self.metric_states = {'Delta': False, 'EMG': True}
        self.combinator_state = 'Union'
        self.zoom_target = 'Delta' 
        self.final_states = None
        
        # --- Fills for Plotting ---
        self.fills_eeg = []
        self.fills_theta = []
        self.fills_emg = []
        self.fills_zoom = []
        self.fills_artifact = []
        
        # --- Execution ---
        self.calc_power()
        self.set_active_data(init=True) # Initialize thresholds only once
        
    def calc_power(self):
        print("[Sleep Scorer] Calculating Features & Detecting Artifacts...")
        win_size = int(2.0 * self.sfreq)
        step = int(self.STEP_SEC * self.sfreq)
        starts = range(0, len(self.eeg) - win_size, step)
        
        raw_delta_p, raw_theta_p, raw_total_p, raw_emg_p = [], [], [], []
        self.time_bins = []
        self.artifacts = []
        
        for s in starts:
            epoch_eeg = self.eeg[s : s + win_size]
            epoch_all_eeg = self.all_eegs[:, s : s + win_size]
            epoch_emg = self.emg[s : s + win_size]
            
            # --- ARTIFACT DETECTION ---
            is_artifact = False
            if np.max(np.abs(epoch_eeg)) > self.amp_thresh:
                is_artifact = True
            elif self.all_eegs.shape[0] > 1:
                corr_matrix = np.corrcoef(epoch_all_eeg)
                upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                if np.mean(np.abs(upper_tri)) > self.sync_thresh:
                    is_artifact = True
                    
            self.artifacts.append(is_artifact)

            # --- POWER CALCULATIONS ---
            freqs_eeg, psd_eeg = welch(epoch_eeg, fs=self.sfreq, nperseg=win_size)
            total_p = np.trapezoid(psd_eeg, freqs_eeg)
            
            idx_delta = np.where((freqs_eeg >= self.DELTA_BAND[0]) & (freqs_eeg <= self.DELTA_BAND[1]))
            delta_p = np.trapezoid(psd_eeg[idx_delta], freqs_eeg[idx_delta])
            
            idx_theta = np.where((freqs_eeg >= self.THETA_BAND[0]) & (freqs_eeg <= self.THETA_BAND[1]))
            theta_p = np.trapezoid(psd_eeg[idx_theta], freqs_eeg[idx_theta])
            
            freqs_emg, psd_emg = welch(epoch_emg, fs=self.sfreq, nperseg=win_size)
            idx_emg = np.where((freqs_emg >= self.EMG_BAND[0]) & (freqs_emg <= self.EMG_BAND[1]))
            emg_p = np.trapezoid(psd_emg[idx_emg], freqs_emg[idx_emg])
            
            raw_delta_p.append(delta_p)
            raw_theta_p.append(theta_p)
            raw_total_p.append(total_p)
            raw_emg_p.append(emg_p)
            self.time_bins.append(s / self.sfreq)
            
        total = np.array(raw_total_p) + 1e-20
        delta = np.array(raw_delta_p) + 1e-20
        theta = np.array(raw_theta_p) + 1e-20
        self.artifacts = np.array(self.artifacts)
        self.time_bins = np.array(self.time_bins)
        
        # 1. Base indices
        delta_val_raw = (delta - (total - delta)) / total
        theta_val_raw = np.log10(theta / delta)
        emg_val_raw = np.log10(np.array(raw_emg_p) + 1e-12) 
        
        # 2. Masked indices for clean generation
        delta_val_clean = delta_val_raw.copy()
        theta_val_clean = theta_val_raw.copy()
        emg_val_clean = emg_val_raw.copy()
        
        delta_val_clean[self.artifacts] = np.nan
        theta_val_clean[self.artifacts] = np.nan
        emg_val_clean[self.artifacts] = np.nan
        
        bins_smooth = int(self.SMOOTH_WIN_SEC / self.STEP_SEC)
        
        # 3. Roll Raw Data
        self.delta_log_raw = pd.Series(delta_val_raw).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        self.theta_log_raw = pd.Series(theta_val_raw).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        self.emg_log_raw = pd.Series(emg_val_raw).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        
        # 4. Roll Clean Data and re-enforce NaNs to break the plotted lines
        self.delta_log_clean = pd.Series(delta_val_clean).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        self.theta_log_clean = pd.Series(theta_val_clean).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        self.emg_log_clean = pd.Series(emg_val_clean).rolling(window=bins_smooth, center=True, min_periods=1).median().values
        
        self.delta_log_clean[self.artifacts] = np.nan
        self.theta_log_clean[self.artifacts] = np.nan
        self.emg_log_clean[self.artifacts] = np.nan

        # Cache fixed ranges based on raw data to prevent axes from jumping on toggle
        self.theta_range = (np.nanmin(self.theta_log_raw), np.nanmax(self.theta_log_raw))
        self.emg_range = (np.nanmin(self.emg_log_raw), np.nanmax(self.emg_log_raw))

    def set_active_data(self, init=False):
        """ Switches pointer between raw and clean data streams based on toggle state.
            Only recalculates thresholds if init=True. """
        if self.exclude_artifacts:
            self.delta_active = self.delta_log_clean
            self.theta_active = self.theta_log_clean
            self.emg_active = self.emg_log_clean
        else:
            self.delta_active = self.delta_log_raw
            self.theta_active = self.theta_log_raw
            self.emg_active = self.emg_log_raw
            
        if init:
            self.delta_thresh = self.calculate_threshold(self.delta_active)
            self.theta_thresh = self.calculate_threshold(self.theta_active)
            self.emg_thresh = self.calculate_threshold(self.emg_active)

    def calculate_threshold(self, X):
        X_clean = X[~np.isnan(X)]
        if len(X_clean) == 0: return 0 
        
        if HAS_SKLEARN:
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
                gmm.fit(X_clean.reshape(-1, 1))
                means = gmm.means_.flatten()
                idx_wake, idx_sleep = (0, 1) if means[0] < means[1] else (1, 0)
                
                p1, p99 = np.percentile(X_clean, 1), np.percentile(X_clean, 99)
                x_axis = np.linspace(p1, p99, 1000).reshape(-1, 1)
                
                probs = gmm.predict_proba(x_axis)
                return x_axis[np.argmin(np.abs(probs[:, idx_wake] - probs[:, idx_sleep]))][0]
            except: pass
            
        v_min, v_max = np.min(X_clean), np.max(X_clean)
        if v_max - v_min == 0: return v_min
        X_norm = ((X_clean - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(X_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (otsu_val / 255.0 * (v_max - v_min)) + v_min

    def apply_persistence(self, raw_mask):
        req_sleep = int(self.sleep_persistence / self.STEP_SEC)
        req_wake = int(self.wake_persistence / self.STEP_SEC)
        
        clean = np.zeros_like(raw_mask)
        if len(raw_mask) == 0: return clean
        
        curr = raw_mask[0]
        pers = 0
        
        for i, val in enumerate(raw_mask):
            if val != curr:
                pers += 1
                bins_req = req_wake if val else req_sleep
                if pers >= bins_req:
                    curr = val
                    clean[i - pers + 1 : i + 1] = curr
                    pers = 0
            else: 
                pers = 0
            clean[i] = curr
            
        return clean

    def get_sleep_mask(self):
        with np.errstate(invalid='ignore'):
            mask_delta = self.delta_active > self.delta_thresh
            mask_emg = self.emg_active < self.emg_thresh
            
        mask_delta = self.apply_persistence(mask_delta)
        mask_emg = self.apply_persistence(mask_emg)
        active_metrics = [k for k, v in self.metric_states.items() if v]
        
        if self.combinator_state == 'None' or not active_metrics:
            return np.zeros_like(mask_delta) 
            
        if self.combinator_state == 'Union':
            combined = np.zeros_like(mask_delta)
            if 'Delta' in active_metrics: combined |= mask_delta
            if 'EMG' in active_metrics: combined |= mask_emg
            return combined
            
        elif self.combinator_state == 'Intersect':
            combined = np.ones_like(mask_delta)
            if 'Delta' in active_metrics: combined &= mask_delta
            if 'EMG' in active_metrics: combined &= mask_emg
            return combined

    def get_current_states(self):
        sleep_mask = self.get_sleep_mask()
        with np.errstate(invalid='ignore'):
            mask_theta = self.apply_persistence(self.theta_active > self.theta_thresh)
        
        req_sleep_epochs = int(self.rem_delay / self.STEP_SEC)
        valid_for_rem = np.zeros_like(sleep_mask, dtype=bool)
        streak = 0
        
        for i in range(len(sleep_mask)):
            if sleep_mask[i]: streak += 1
            else: streak = 0
            if streak >= req_sleep_epochs: valid_for_rem[i] = True

        raw_states = np.zeros(len(self.time_bins), dtype=int)
        rem_mask = sleep_mask & mask_theta & valid_for_rem
        nrem_mask = sleep_mask & ~rem_mask
        
        raw_states[nrem_mask] = 1
        raw_states[rem_mask] = 2
        
        # If artifacts are fully excluded from the logic, ensure they stay as Wake
        if self.exclude_artifacts:
            raw_states[self.artifacts] = 0 
            
        return raw_states

    def remove_spines(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def run(self):
        if self.valid_video:
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        else:
            self.cap = None
            self.fps = 10.0
        
        self.fig = plt.figure(figsize=(13, 8.5))
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.28, wspace=0.15, hspace=0.45)
        
        gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1, 4])
        
        self.ax_vid = self.fig.add_subplot(gs[0, 0])
        self.ax_vid.set_xticks([]); self.ax_vid.set_yticks([])
        for spine in self.ax_vid.spines.values(): spine.set_visible(False)
        self.ax_vid.set_facecolor('lightgray')
        
        self.ax_zoom = self.fig.add_subplot(gs[0, 1])
        self.ax_hist_delta = self.fig.add_subplot(gs[1, 0])
        self.ax_eeg = self.fig.add_subplot(gs[1, 1])
        self.ax_hist_emg = self.fig.add_subplot(gs[2, 0])
        self.ax_emg = self.fig.add_subplot(gs[2, 1], sharex=self.ax_eeg)
        self.ax_hist_theta = self.fig.add_subplot(gs[3, 0])
        self.ax_theta = self.fig.add_subplot(gs[3, 1], sharex=self.ax_eeg)
        
        for ax in [self.ax_hist_delta, self.ax_eeg, self.ax_hist_emg, self.ax_emg, 
                   self.ax_hist_theta, self.ax_theta, self.ax_zoom]:
            self.remove_spines(ax)

        # --- Control Board ---
        ax_control_bg = self.fig.add_axes([0.02, 0.02, 0.96, 0.23])
        ax_control_bg.set_xticks([]); ax_control_bg.set_yticks([])
        for spine in ax_control_bg.spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(1.5)
        ax_control_bg.set_facecolor('#f9f9f9')
        
        # Sliders
        ax_wake_slider = self.fig.add_axes([0.05, 0.16, 0.35, 0.02])
        ax_sleep_slider = self.fig.add_axes([0.05, 0.10, 0.35, 0.02])
        ax_rem_delay_slider = self.fig.add_axes([0.05, 0.04, 0.35, 0.02])
        
        self.slider_wake = Slider(ax_wake_slider, '', 0.0, 30.0, valinit=self.wake_persistence, valstep=1.0)
        ax_wake_slider.set_title('Wake Persistence (s)', fontsize=10)
        
        self.slider_sleep = Slider(ax_sleep_slider, '', 0.0, 30.0, valinit=self.sleep_persistence, valstep=1.0)
        ax_sleep_slider.set_title('Sleep Persistence (s)', fontsize=10)
        
        self.slider_rem_delay = Slider(ax_rem_delay_slider, '', 0.0, 180.0, valinit=self.rem_delay, valstep=5.0)
        ax_rem_delay_slider.set_title('Minimum sleep before REM (s)', fontsize=10)
        
        self.slider_wake.on_changed(self.update_sliders)
        self.slider_sleep.on_changed(self.update_sliders)
        self.slider_rem_delay.on_changed(self.update_sliders)

        # Metrics (Checkboxes)
        ax_metric_ui = self.fig.add_axes([0.48, 0.04, 0.20, 0.16], facecolor='#f9f9f9')
        ax_metric_ui.set_title("Sleep Metrics", fontweight='bold', fontsize=10)
        for spine in ax_metric_ui.spines.values(): spine.set_visible(False)
        
        self.check_metrics = CheckButtons(ax_metric_ui, list(self.metric_states.keys()), actives=list(self.metric_states.values()))
        self.check_metrics.on_clicked(self.set_metric)
        
        # Combinator (Radio Buttons)
        ax_comb_ui = self.fig.add_axes([0.72, 0.04, 0.20, 0.16], facecolor='#f9f9f9')
        ax_comb_ui.set_title("Combinator", fontweight='bold', fontsize=10)
        for spine in ax_comb_ui.spines.values(): spine.set_visible(False)
        self.radio_comb = RadioButtons(ax_comb_ui, ('None', 'Union', 'Intersect'), active=1)
        self.radio_comb.on_clicked(self.set_combinator)
        
        # Artifact Toggle
        ax_exclude_ui = self.fig.add_axes([0.48, 0.20, 0.44, 0.04], facecolor='#f9f9f9')
        for spine in ax_exclude_ui.spines.values(): spine.set_visible(False)
        self.check_exclude = CheckButtons(ax_exclude_ui, ['Exclude Artifacts'], actives=[self.exclude_artifacts])
        self.check_exclude.on_clicked(self.toggle_artifacts)
        for label in self.check_exclude.labels: 
            label.set_fontsize(10)
            label.set_fontweight('bold')

        for label in self.radio_comb.labels + self.check_metrics.labels: label.set_fontsize(9)

        # --- Initial Draws ---
        self.draw_static_elements()
        self.update_dynamic_elements()
        self.update_video(0)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.show(block=True)
        if self.cap is not None:
            self.cap.release()

    def toggle_artifacts(self, label):
        self.exclude_artifacts = not self.exclude_artifacts
        self.set_active_data(init=False)  # We do NOT want to recalculate thresholds here
        self.update_line_data()
        self.update_dynamic_elements()
        self.fig.canvas.draw_idle()

    def set_metric(self, label):
        self.metric_states[label] = not self.metric_states[label]
        self.update_dynamic_elements()
        
    def set_combinator(self, label):
        self.combinator_state = label
        self.update_dynamic_elements()

    def update_sliders(self, val):
        self.wake_persistence = self.slider_wake.val
        self.sleep_persistence = self.slider_sleep.val
        self.rem_delay = self.slider_rem_delay.val
        self.update_dynamic_elements()

    def draw_static_elements(self):
        self.line_eeg, = self.ax_eeg.plot(self.time_bins, self.delta_active, 'k-', lw=0.5, alpha=0.8)
        self.ax_eeg.set_ylabel("Delta Idx")
        self.ax_eeg.set_ylim(-1.1, 1.1)
        self.thresh_line_eeg = self.ax_eeg.axhline(self.delta_thresh, color='red', linestyle='--')
        
        self.line_emg, = self.ax_emg.plot(self.time_bins, self.emg_active, 'k-', lw=0.5, alpha=0.8)
        self.ax_emg.set_ylabel("log(EMG)")
        self.ax_emg.set_ylim(self.emg_range[0] - 0.5, self.emg_range[1] + 0.5)
        self.thresh_line_emg = self.ax_emg.axhline(self.emg_thresh, color='red', linestyle='--')

        self.line_theta, = self.ax_theta.plot(self.time_bins, self.theta_active, 'k-', lw=0.5, alpha=0.8)
        self.ax_theta.set_ylabel("Log(T/D)")
        self.ax_theta.set_ylim(self.theta_range[0] - 0.2, self.theta_range[1] + 0.2)
        self.thresh_line_theta = self.ax_theta.axhline(self.theta_thresh, color='red', linestyle='--')
        
        self.line_zoom, = self.ax_zoom.plot(self.time_bins, self.delta_active, 'k.-', lw=1)
        self.ax_zoom.set_ylabel("Zoom (Delta)")
        self.ax_zoom.set_ylim(-1.1, 1.1)
        self.thresh_line_zoom = self.ax_zoom.axhline(self.delta_thresh, color='red', linestyle='--')
        
        # Dedicated scatter for rendering artifacts as open circles in the zoom panel
        self.scatter_zoom_art = self.ax_zoom.scatter([], [], facecolors='none', edgecolors='k', s=40, zorder=5)
        
        self.marker_eeg = self.ax_eeg.axvline(0, color='red', alpha=0.5)
        self.marker_emg = self.ax_emg.axvline(0, color='red', alpha=0.5)
        self.marker_theta = self.ax_theta.axvline(0, color='red', alpha=0.5)
        self.marker_zoom = self.ax_zoom.axvline(0, color='red', alpha=0.5)

        label_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='none')
        self.ax_eeg.text(0.01, 0.92, "DELTA", transform=self.ax_eeg.transAxes, fontsize=11, fontweight='bold', bbox=label_props, zorder=10)
        self.ax_emg.text(0.01, 0.92, "EMG", transform=self.ax_emg.transAxes, fontsize=11, fontweight='bold', bbox=label_props, zorder=10)
        self.ax_theta.text(0.01, 0.92, "THETA / DELTA", transform=self.ax_theta.transAxes, fontsize=11, fontweight='bold', bbox=label_props, zorder=10)
        self.zoom_label = self.ax_zoom.text(0.01, 0.92, "ZOOM: DELTA", transform=self.ax_zoom.transAxes, fontsize=11, fontweight='bold', bbox=label_props, zorder=10)
        
        self.update_line_data()

    def update_line_data(self):
        self.line_eeg.set_ydata(self.delta_active)
        self.line_emg.set_ydata(self.emg_active)
        self.line_theta.set_ydata(self.theta_active)

        if self.zoom_target == 'Delta':
            zoom_active, zoom_raw = self.delta_active, self.delta_log_raw
        elif self.zoom_target == 'EMG':
            zoom_active, zoom_raw = self.emg_active, self.emg_log_raw
        else:
            zoom_active, zoom_raw = self.theta_active, self.theta_log_raw
            
        self.line_zoom.set_ydata(zoom_active)
        
        # Manage open circles for artifacts
        if self.exclude_artifacts and np.sum(self.artifacts) > 0:
            art_x = self.time_bins[self.artifacts]
            art_y = zoom_raw[self.artifacts]
            self.scatter_zoom_art.set_offsets(np.column_stack((art_x, art_y)))
            self.scatter_zoom_art.set_visible(True)
        else:
            self.scatter_zoom_art.set_visible(False)

    def set_zoom_target(self, target):
        self.zoom_target = target
        self.update_line_data()
        
        if target == 'Delta':
            self.ax_zoom.set_ylabel("Zoom (Delta)")
            self.ax_zoom.set_ylim(-1.1, 1.1)
            self.zoom_label.set_text("ZOOM: DELTA")
        elif target == 'EMG':
            self.ax_zoom.set_ylabel("Zoom (EMG)")
            self.ax_zoom.set_ylim(self.emg_range[0] - 0.5, self.emg_range[1] + 0.5)
            self.zoom_label.set_text("ZOOM: EMG")
        else:
            self.ax_zoom.set_ylabel("Zoom (Log T/D)")
            self.ax_zoom.set_ylim(self.theta_range[0] - 0.2, self.theta_range[1] + 0.2)
            self.zoom_label.set_text("ZOOM: THETA/DELTA")
            
        self.update_dynamic_elements()

    def update_dynamic_elements(self):
        sleep_mask = self.get_sleep_mask()
        states = self.get_current_states()
        nrem_mask = (states == 1)
        rem_mask = (states == 2)

        bbox_props = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)

        # --- Dynamic Histograms ---
        self.ax_hist_delta.cla()
        delta_valid = self.delta_active[~np.isnan(self.delta_active)]
        if len(delta_valid) > 0:
            self.ax_hist_delta.hist(delta_valid, bins=50, density=True, color='gray', alpha=0.4)
        self.ax_hist_delta.axvline(self.delta_thresh, color='red', lw=2)
        self.ax_hist_delta.text(0.95, 0.95, f"{self.delta_thresh:.2f}", transform=self.ax_hist_delta.transAxes, ha='right', va='top', fontsize=10, bbox=bbox_props)
        self.ax_hist_delta.set_ylabel("Density", fontsize=9)
        self.remove_spines(self.ax_hist_delta)

        self.ax_hist_emg.cla()
        emg_valid = self.emg_active[~np.isnan(self.emg_active)]
        if len(emg_valid) > 0:
            p1, p99 = np.percentile(emg_valid, 1), np.percentile(emg_valid, 99)
            self.ax_hist_emg.hist(emg_valid, bins=50, range=(p1, p99), density=True, color='gray', alpha=0.4)
            self.ax_hist_emg.set_xlim(p1, p99)
        self.ax_hist_emg.axvline(self.emg_thresh, color='red', lw=2)
        self.ax_hist_emg.text(0.95, 0.95, f"{self.emg_thresh:.2f}", transform=self.ax_hist_emg.transAxes, ha='right', va='top', fontsize=10, bbox=bbox_props)
        self.ax_hist_emg.set_ylabel("Density", fontsize=9)
        self.remove_spines(self.ax_hist_emg)
        
        self.ax_hist_theta.cla()
        theta_sleep = self.theta_active[sleep_mask & ~np.isnan(self.theta_active)]
        bins_theta = np.linspace(self.theta_range[0], self.theta_range[1], 50)
        if len(theta_sleep) > 0:
            self.ax_hist_theta.hist(theta_sleep, bins=bins_theta, density=True, color='mediumpurple', alpha=0.7)
            
        self.ax_hist_theta.axvline(self.theta_thresh, color='red', lw=2)
        self.ax_hist_theta.set_xlim(self.theta_range)
        self.ax_hist_theta.text(0.95, 0.95, f"{self.theta_thresh:.2f}", transform=self.ax_hist_theta.transAxes, ha='right', va='top', fontsize=10, bbox=bbox_props)
        self.ax_hist_theta.set_ylabel("Density", fontsize=9)
        self.remove_spines(self.ax_hist_theta)

        # Update Threshold Lines
        self.thresh_line_eeg.set_ydata([self.delta_thresh])
        self.thresh_line_emg.set_ydata([self.emg_thresh])
        self.thresh_line_theta.set_ydata([self.theta_thresh])

        if self.zoom_target == 'Delta': val_z = self.delta_thresh
        elif self.zoom_target == 'EMG': val_z = self.emg_thresh
        else: val_z = self.theta_thresh
        self.thresh_line_zoom.set_ydata([val_z])
        
        for f in self.fills_eeg + self.fills_theta + self.fills_emg + self.fills_zoom + self.fills_artifact:
            try: f.remove()
            except: pass
        self.fills_eeg, self.fills_theta, self.fills_emg, self.fills_zoom, self.fills_artifact = [], [], [], [], []
        
        def apply_fill(ax_target, target_list, mask, color, alpha, y_bounds):
            f = ax_target.fill_between(self.time_bins, y_bounds[0], y_bounds[1], where=mask, 
                                       color=color, alpha=alpha, linewidth=0, step='post')
            target_list.append(f)

        bounds_index = (-1.1, 1.1)
        bounds_emg = self.ax_emg.get_ylim()
        bounds_theta_fill = self.ax_theta.get_ylim()
        if self.zoom_target == 'Delta': bounds_zoom = bounds_index
        elif self.zoom_target == 'EMG': bounds_zoom = bounds_emg
        else: bounds_zoom = bounds_theta_fill
        
        # 1. Base Sleep Masks
        apply_fill(self.ax_eeg, self.fills_eeg, sleep_mask, 'mediumpurple', 0.4, bounds_index)
        apply_fill(self.ax_emg, self.fills_emg, sleep_mask, 'mediumpurple', 0.4, bounds_emg)
        apply_fill(self.ax_theta, self.fills_theta, nrem_mask, 'mediumpurple', 0.4, bounds_theta_fill)
        apply_fill(self.ax_theta, self.fills_theta, rem_mask, 'red', 0.5, bounds_theta_fill)

        if self.zoom_target in ['Delta', 'EMG']:
            apply_fill(self.ax_zoom, self.fills_zoom, sleep_mask, 'mediumpurple', 0.4, bounds_zoom)
        else:
            apply_fill(self.ax_zoom, self.fills_zoom, nrem_mask, 'mediumpurple', 0.4, bounds_zoom)
            apply_fill(self.ax_zoom, self.fills_zoom, rem_mask, 'red', 0.5, bounds_zoom)

        # 2. Render yellow artifact warnings ONLY if we are actively highlighting/including them 
        if not self.exclude_artifacts:
            apply_fill(self.ax_eeg, self.fills_artifact, self.artifacts, '#ffcc00', 0.6, bounds_index)
            apply_fill(self.ax_emg, self.fills_artifact, self.artifacts, '#ffcc00', 0.6, bounds_emg)
            apply_fill(self.ax_theta, self.fills_artifact, self.artifacts, '#ffcc00', 0.6, bounds_theta_fill)
            apply_fill(self.ax_zoom, self.fills_artifact, self.artifacts, '#ffcc00', 0.6, bounds_zoom)

        # 3. Dynamic Status Update
        wake_min = (np.sum(states == 0) * self.STEP_SEC) / 60.0
        nrem_min = (np.sum(states == 1) * self.STEP_SEC) / 60.0
        rem_min = (np.sum(states == 2) * self.STEP_SEC) / 60.0
        art_count = np.sum(self.artifacts)
        
        status_text = f"Wake: {wake_min:.1f} m  |  NREM: {nrem_min:.1f} m  |  REM: {rem_min:.1f} m  |  Artifacts: {art_count} bins"
        status_text += " (Excluded)" if self.exclude_artifacts else " (Included)"
            
        self.fig.suptitle(status_text, fontsize=14, fontweight='bold')
        self.fig.canvas.draw_idle()

    def update_video(self, t):
        if t is None: return
        ret = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * self.fps))
            ret, frame = self.cap.read()
            
        idx = min(int(t / self.STEP_SEC), len(self.time_bins) - 1)
        states = self.get_current_states()
        state_val = states[idx]
        
        if self.artifacts[idx] and self.exclude_artifacts: state_text = "ARTIFACT (Excluded)"
        elif state_val == 1: state_text = "NREM"
        elif state_val == 2: state_text = "REM"
        else: state_text = "WAKE"
            
        self.ax_vid.set_title(f"T: {t:.1f}s | State: {state_text}", fontsize=12, fontweight='bold')
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not hasattr(self, 'vid_im'):
                self.vid_im = self.ax_vid.imshow(frame)
                self.ax_vid.set_aspect('equal', adjustable='datalim')
            else:
                self.vid_im.set_data(frame)
                self.vid_im.set_visible(True)
            for txt in self.ax_vid.texts: txt.set_visible(False)
        else:
            if hasattr(self, 'vid_im'): self.vid_im.set_visible(False)
            has_text = any(txt.get_text() == "No Video Available" and txt.get_visible() for txt in self.ax_vid.texts)
            if not has_text:
                self.ax_vid.text(0.5, 0.5, "No Video Available", ha='center', va='center', fontsize=12, transform=self.ax_vid.transAxes)
            
        self.marker_eeg.set_xdata([t])
        self.marker_emg.set_xdata([t])
        self.marker_theta.set_xdata([t])
        self.marker_zoom.set_xdata([t])
        
        zoom_window = 30
        self.ax_zoom.set_xlim(t - zoom_window, t + zoom_window)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.xdata is None: return
        
        if event.inaxes == self.ax_hist_delta:
            self.delta_thresh = event.xdata
            self.update_dynamic_elements()
        elif event.inaxes == self.ax_hist_emg:
            self.emg_thresh = event.xdata
            self.update_dynamic_elements()
        elif event.inaxes == self.ax_hist_theta:
            self.theta_thresh = event.xdata
            self.update_dynamic_elements()
        elif event.inaxes == self.ax_eeg:
            self.set_zoom_target('Delta')
            self.update_video(event.xdata)
        elif event.inaxes == self.ax_emg:
            self.set_zoom_target('EMG')
            self.update_video(event.xdata)
        elif event.inaxes == self.ax_theta:
            self.set_zoom_target('Theta')
            self.update_video(event.xdata)
        elif event.inaxes == self.ax_zoom:
            self.update_video(event.xdata)

    def on_key(self, event):
        if event.key == 'enter':
            print("\n[Sleep Scorer] Finalizing mask and generating CSV...")
            self.final_states = self.get_current_states()
            
            df = pd.DataFrame({
                'Time_sec': self.time_bins,
                'Wake': (self.final_states == 0).astype(int),
                'NREM': (self.final_states == 1).astype(int),
                'REM': (self.final_states == 2).astype(int),
                'Artifact': self.artifacts.astype(int) 
            })
            
            out_path = os.path.splitext(self.edf_path)[0] + "_scored.csv"
            df.to_csv(out_path, index=False)
            print(f"Success! Saved scored data to: {out_path}\n")
            plt.close(self.fig)

# --- Helpers ---
def select_best_eeg_channel(raw, emg_ch_name):
    eeg_candidates = [ch for ch in raw.ch_names if ch != emg_ch_name and 'EEG' in ch]
    if not eeg_candidates:
        print("Warning: No distinct EEG channels found. Defaulting to first channel.")
        return raw.ch_names[0]
        
    best_ch = None
    best_snr = -np.inf
    print("Evaluating EEG channels for best Signal-to-Noise Ratio...")
    for ch in eeg_candidates:
        data = raw.get_data(picks=ch)[0]
        freqs, psd = welch(data, fs=raw.info['sfreq'], nperseg=int(2 * raw.info['sfreq']))
        signal_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 30)])
        noise_power = np.sum(psd[freqs > 30])
        snr = signal_power / (noise_power + 1e-12)
        print(f" - {ch}: SNR = {snr:.2f}")
        
        if snr > best_snr:
            best_snr = snr; best_ch = ch
            
    print(f"Selected best EEG channel: {best_ch} (SNR = {best_snr:.2f})\n")
    return best_ch

def find_associated_video(edf_path):
    """
    Attempts to locate a video file matching the timestamp in the EDF filename.
    If multiple are found, it opens each to count frames and returns the longest one.
    """
    directory = os.path.dirname(os.path.abspath(edf_path))
    basename = os.path.basename(edf_path)
    candidates = []
    
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', basename)
    if match:
        timestamp = match.group(0)
        for file in os.listdir(directory):
            if file.lower().endswith(('.mp4', '.avi', '.mkv')) and timestamp in file:
                candidates.append(os.path.join(directory, file))
                
    if not candidates:
        base_no_ext = os.path.splitext(basename)[0]
        for file in os.listdir(directory):
            if file.lower().endswith(('.mp4', '.avi', '.mkv')) and base_no_ext in file:
                candidates.append(os.path.join(directory, file))
                
    if not candidates:
        return None
        
    best_video = None
    max_frames = -1
    
    for candidate_path in candidates:
        cap = cv2.VideoCapture(candidate_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > max_frames:
                max_frames = frame_count
                best_video = candidate_path
        cap.release()
        
    return best_video

def main():
    print("=== Interactive EDF Sleep Scorer (Wake/NREM/REM) ===")
    
    parser = argparse.ArgumentParser(description="Interactive EDF Sleep Scorer")
    parser.add_argument("edf_path", help="Path to the input .edf file")
    parser.add_argument("--video", dest="video_path", default=None)
    parser.add_argument("--emg-ch", dest="emg_ch_name", default="EEG EEG_4_SA-B")
    parser.add_argument("--amp-thresh", type=float, default=2e-3, help="Max amplitude threshold (Volts) - Default 2 mV")
    parser.add_argument("--sync-thresh", type=float, default=0.90, help="Max Pearson correlation across channels")
    args = parser.parse_args()
    
    mne.set_log_level('ERROR')
    try: raw = mne.io.read_raw_edf(args.edf_path, preload=True)
    except Exception as e: print(f"Failed to load EDF: {e}"); return
        
    try: emg_data = raw.get_data(picks=args.emg_ch_name)[0]
    except ValueError: print(f"Error: EMG channel '{args.emg_ch_name}' does not exist in this EDF."); return

    # Video Auto-detection Logic restored
    if args.video_path is None:
        video_path = find_associated_video(args.edf_path)
        if video_path:
            print(f"Auto-detected associated video: {os.path.basename(video_path)}")
        else:
            print("No associated video found automatically. Proceeding without video.")
            video_path = None
    else:
        video_path = args.video_path

    best_eeg_ch_name = select_best_eeg_channel(raw, args.emg_ch_name)
    best_eeg_data = raw.get_data(picks=best_eeg_ch_name)[0]
    
    eeg_channels = [ch for ch in raw.ch_names if ch != args.emg_ch_name and 'EEG' in ch]
    all_eeg_data = raw.get_data(picks=eeg_channels)

    print("Launching visualizer... (Press 'Enter' on the graph window when finished to save CSV)")
    scorer = InteractiveSleepScorer(
        best_eeg=best_eeg_data, 
        all_eegs=all_eeg_data,
        emg_data=emg_data, 
        sfreq=raw.info['sfreq'], 
        edf_path=args.edf_path, 
        video_path=video_path,
        amp_thresh=args.amp_thresh,
        sync_thresh=args.sync_thresh
    )
    scorer.run()

if __name__ == "__main__":
    main()