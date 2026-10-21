import cv2
import numpy as np
from somnus.scorer.signal_utils import get_nice_number

WINDOW_NAME = "Sleep Scorer Validator"

COLORS = {
    # --- paintable: these are brushes, and each is a column in the scored CSV
    'Wake': (50, 200, 50),
    'NREM': (200, 100, 50),
    'REM': (50, 50, 200),
    'Artifact': (0, 200, 200),      # bad signal: exclude this epoch
    'Unclear': (200, 50, 200),      # ambiguous physiology: exclude this epoch (or just leave unscored)
    # --- display only: worked out while drawing, never painted and never stored
    'Unknown': (50, 50, 50),        # nothing scored here yet
    'HMM_Smoothed': (230, 200, 60),  # label came from smoothing, not the model
}

NAV_LABELS = ['Wake', 'NREM', 'REM']

# The paintable states, in their CSV column order. A bin's label code is 0 for
# Unknown, else 1 + its index here.
PAINTABLE = ('Wake', 'NREM', 'REM', 'Artifact', 'Unclear')
LABELS = ('Unknown',) + PAINTABLE

# Model labels are drawn at this fraction of their color over the black
# background. Bins a person confirmed or changed keep the hue but gain OKLab
# lightness and chroma, so they read as the same color, stronger.
MODEL_ALPHA = 0.20
LIGHTNESS_GAIN = 0.10
CHROMA_GAIN = 2.5

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _stronger(bgr):
    """The model tint of a color, with OKLab lightness and chroma raised."""
    c = np.asarray(bgr[::-1], dtype=float) * MODEL_ALPHA / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lab = _M2 @ np.cbrt(_M1 @ lin)
    # largest chroma gain that stays inside the sRGB gamut
    for k in np.linspace(CHROMA_GAIN, 1.0, 64):
        lin = np.linalg.solve(_M1, np.linalg.solve(
            _M2, np.r_[lab[0] + LIGHTNESS_GAIN, lab[1:] * k]) ** 3)
        if lin.min() >= 0.0 and lin.max() <= 1.0:
            break
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(v)) for v in 255.0 * c[::-1])


STRONG_COLORS = {name: _stronger(COLORS[name]) for name in LABELS}
_MODEL_FILL = np.array([np.round(np.array(COLORS[n]) * MODEL_ALPHA)
                        for n in LABELS], dtype=np.uint8)
_STRONG_FILL = np.array([STRONG_COLORS[n] for n in LABELS], dtype=np.uint8)

RENDER_CACHE = {
    'start_sec': -1,
    'size': None,
    'eeg_gain': -1.0,
    'emg_gain': -1.0,
    'offset': -1.0,
    'window_dur': -1.0, 
    'update_counter': -1,
    'psd_img': None,
    'eeg_base': None,
    'side_img': None,
}

def draw_state_probability_panel(state, w, h, window_start_sec, window_dur):
    """Draw what the model believes about the stretch on screen.

    The average probability of each state, how confident it was, and how many
    visible epochs it was unsure about. 
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, h), (20, 20, 20), -1)

    if getattr(state, 'review_meta', None) is None:
        cv2.putText(img, "MODEL BELIEF", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        cv2.putText(img, "no model metadata", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(img, "(open from the Somnus GUI)", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        return img

    probs, conf, n_low, n_vis = state.window_belief(window_start_sec,
                                                    window_start_sec + window_dur)
    cv2.putText(img, "MODEL BELIEF (in view)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.line(img, (10, 35), (w - 10, 35), (100, 100, 100), 1)

    if n_vis == 0:
        cv2.putText(img, "no epochs in view", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        return img

    bar_x0 = 95
    bar_w_max = w - bar_x0 - 45
    y = 62
    for name in NAV_LABELS:
        pv = float(probs.get(name, 0.0))
        col = COLORS[name]
        cv2.putText(img, name, (10, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)
        cv2.rectangle(img, (bar_x0, y), (bar_x0 + bar_w_max, y + 18),
                      (45, 45, 45), -1)
        bw = int(max(0.0, min(1.0, pv)) * bar_w_max)
        if bw > 0:
            cv2.rectangle(img, (bar_x0, y), (bar_x0 + bw, y + 18), col, -1)
        cv2.putText(img, f"{pv:.2f}", (bar_x0 + bar_w_max + 6, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        y += 26

    y += 6
    cconf = (80, 220, 80) if conf >= 0.85 else \
            (60, 200, 240) if conf >= 0.60 else (80, 80, 240)
    cv2.putText(img, f"mean confidence {conf:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cconf, 1)
    y += 22
    thr = float(getattr(state, 'certainty_threshold', 0.80))
    cv2.putText(img, f"{n_low}/{n_vis} epochs below {thr:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)
    y += 20
    sm = state.n_smoothed_in(window_start_sec, window_start_sec + window_dur)
    if sm:
        cv2.putText(img, f"{sm} HMM-smoothed in view", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS['HMM_Smoothed'], 1)
    return img


def draw_eeg_side_panel(state, data_slice, ch_names, window_start_sec, window_duration, sfreq, w, h, eeg_gain, emg_gain, v_range, num_eeg, offset=0.0):
    """Draw the EEG and EMG traces with the scored timeline beneath them."""
    n_ch, n_samples = data_slice.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if n_samples == 0: return img
    
    ch_h = h // n_ch
    t_scale = w / (window_duration * sfreq)
    end_time_sec = window_start_sec + window_duration
    
    # One bin per pixel column: strong where a person confirmed or changed it,
    # the model tint otherwise.
    cols = window_start_sec + (np.arange(w) + 0.5) * (window_duration / w)
    idx = np.searchsorted(state.t, cols, side='right') - 1
    inside = idx >= 0
    idx = np.clip(idx, 0, len(state.t) - 1)
    inside &= cols < state.t[idx] + state.bin_step
    codes = state.code[idx]
    strip = np.where(state.is_strong(idx)[:, None],
                     _STRONG_FILL[codes], _MODEL_FILL[codes])
    strip[~inside] = 0
    img[:] = strip[None, :, :]

    # Mark epochs the smoothing changed with a thin hatch along the top
    if getattr(state, 'review_meta', None) is not None:
        for t0, t1 in state.smoothed_spans(window_start_sec, end_time_sec):
            x1 = int(((t0 - window_start_sec) / window_duration) * w)
            x2 = int(((t1 - window_start_sec) / window_duration) * w)
            cv2.rectangle(img, (max(0, x1), 0), (min(w, x2), 5),
                          COLORS['HMM_Smoothed'], -1)

    for i in range(n_ch):
        y_offset = i * ch_h + (ch_h // 2)
        
        ch_data = data_slice[i]
        ch_mean = np.mean(ch_data) if len(ch_data) > 0 else 0.0
        
        current_gain = eeg_gain if i < num_eeg else emg_gain
        
        norm_basic = (ch_data - ch_mean) / v_range
        centered_val = (norm_basic * current_gain) - offset
        y_points = (y_offset - (centered_val * ch_h)).astype(int)
        y_points = np.clip(y_points, i * ch_h + 2, (i + 1) * ch_h - 2)
        
        x_points = np.arange(n_samples) * t_scale
        pts = np.column_stack((x_points, y_points)).astype(np.int32)
        
        cv2.line(img, (0, y_offset), (w, y_offset), (50, 50, 50), 1)
        cv2.polylines(img, [pts], False, (200, 200, 200), 1, cv2.LINE_AA)
        
        label = f"{ch_names[i]} (x{current_gain:.1f})"
        cv2.putText(img, label, (5, i * ch_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # a voltage bar on the last EEG row (tracks the EEG gain) and on the
        # EMG row (tracks the EMG gain); the time bar only on the bottom row
        if i in (num_eeg - 1, n_ch - 1):
            bar_x = w - 20; bar_y = (i + 1) * ch_h - 15

            effective_v_range = v_range / current_gain
            volts_per_pixel = effective_v_range / ch_h
            target_v_size = effective_v_range * 0.25

            if target_v_size < 1e-6: unit, mult = "nV", 1e9
            elif target_v_size < 1e-3: unit, mult = "uV", 1e6
            else: unit, mult = "mV", 1e3

            nice_val_units = get_nice_number(target_v_size * mult, round_up=True)
            nice_val_volts = nice_val_units / mult
            pixel_height = int(nice_val_volts / volts_per_pixel)

            color = (240, 240, 240)
            cv2.line(img, (bar_x, bar_y), (bar_x, bar_y - pixel_height), color, 2)
            cv2.putText(img, f"{int(nice_val_units)} {unit}", (bar_x - 55, bar_y - pixel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if i == n_ch - 1:
                seconds_per_pixel = window_duration / w
                target_t_size = 20 * seconds_per_pixel
                nice_t_val = get_nice_number(target_t_size, round_up=True)
                pixel_width = int(nice_t_val / seconds_per_pixel)
                cv2.line(img, (bar_x, bar_y), (bar_x - pixel_width, bar_y), color, 2)

                if nice_t_val >= 1.0: t_label = f"{nice_t_val:.1f}s".replace(".0s", "s")
                elif nice_t_val >= 0.1: t_label = f"{nice_t_val:.2f}s"
                elif nice_t_val >= 0.001: t_label = f"{int(nice_t_val * 1000)}ms"
                else: t_label = f"{nice_t_val * 1e6:.0f}us"

                cv2.putText(img, t_label, (bar_x - pixel_width - 5, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img

def _nav_pair(img, state, y, w_side, label, fill, border, target):
    """Draw one row of previous / next buttons and register where they are."""
    half = (w_side - 24) // 2
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    yc = y + 15
    for i, direction in enumerate((-1, 1)):
        x1 = 10 + i * (half + 4)
        x2 = x1 + half
        cv2.rectangle(img, (x1, y), (x2, y + 30), fill, -1)
        if border is not None:
            cv2.rectangle(img, (x1, y), (x2, y + 30), border, 1)
        if direction < 0:
            tip, base, tx = x1 + 8, x1 + 15, x1 + 21
        else:
            tip, base, tx = x2 - 8, x2 - 15, x2 - 21 - tw
        tri = np.array([[tip, yc], [base, yc - 5], [base, yc + 5]], dtype=np.int32)
        cv2.fillPoly(img, [tri], (255, 255, 255), cv2.LINE_AA)
        cv2.putText(img, label, (tx, yc + th // 2), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
        state.nav_buttons.append(((x1, y, x2, y + 30), target, direction))


def draw_sidebar(state, h, w_side):
    """Draw the side panel: previous / next buttons per state and for low certainty."""
    img = np.zeros((h, w_side, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w_side, h), (30, 30, 30), -1)
    cv2.line(img, (0, 0), (0, h), (100, 100, 100), 1)
    
    cv2.putText(img, "BOUT NAVIGATION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(img, (10, 35), (w_side-10, 35), (100, 100, 100), 1)
    
    state.nav_buttons = []
    y = 46
    for label in NAV_LABELS:
        _nav_pair(img, state, y, w_side, label, STRONG_COLORS[label], None, label)
        y += 38

    # --- model review controls (only when the GUI supplied review metadata) ---
    if getattr(state, 'review_meta', None) is not None and y + 76 <= h:
        cv2.line(img, (10, y), (w_side - 10, y), (100, 100, 100), 1)
        cv2.putText(img, "LOW CERTAINTY", (10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        _nav_pair(img, state, y + 32, w_side, "Epoch", (150, 110, 20),
                  (220, 180, 60), 'uncertain')
        thr = float(getattr(state, 'certainty_threshold', 0.80))
        cv2.putText(img, f"thr {thr:.2f}  u/U  [ / ]", (10, y + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)
    return img

def render_composite(state, frame, eeg_slice, ch_names, sfreq, window_start_sec, playback_offset_sec, window_dur, eeg_gain, emg_gain, offset, v_range, num_eeg):
    """Assemble one complete frame of the scorer window."""
    
    VIDEO_W = state.video_w
    EEG_W = state.eeg_w
    EEG_H = state.eeg_h
    SIDE_W = state.side_w
    PSD_W = EEG_W - VIDEO_W - SIDE_W
    BOTTOM_H = state.bottom_h

    video_disp = np.zeros((BOTTOM_H, VIDEO_W, 3), dtype=np.uint8)
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        scale = min(VIDEO_W / frame.shape[1], BOTTOM_H / frame.shape[0])
        vw = max(1, int(frame.shape[1] * scale))
        vh = max(1, int(frame.shape[0] * scale))
        x0 = (VIDEO_W - vw) // 2
        y0 = (BOTTOM_H - vh) // 2
        video_disp[y0:y0 + vh, x0:x0 + vw] = cv2.resize(frame, (vw, vh))

    exact_time = window_start_sec + playback_offset_sec
    m, s = divmod(int(exact_time), 60)
    h_time, m = divmod(m, 60)
    cv2.putText(video_disp, f"{h_time:02d}:{m:02d}:{s:02d}", (VIDEO_W - 120, BOTTOM_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if (RENDER_CACHE['start_sec'] != window_start_sec or 
        RENDER_CACHE['size'] != (EEG_W, EEG_H) or
        RENDER_CACHE['eeg_gain'] != eeg_gain or 
        RENDER_CACHE['emg_gain'] != emg_gain or 
        RENDER_CACHE['offset'] != offset or 
        RENDER_CACHE['window_dur'] != window_dur or 
        RENDER_CACHE['update_counter'] != state.update_counter or
        RENDER_CACHE['psd_img'] is None):
        
        RENDER_CACHE['psd_img'] = draw_state_probability_panel(
            state, PSD_W, BOTTOM_H, window_start_sec, window_dur)
        
        RENDER_CACHE['eeg_base'] = draw_eeg_side_panel(
            state, eeg_slice, ch_names, window_start_sec, window_dur, sfreq, EEG_W, EEG_H, eeg_gain, emg_gain, v_range, num_eeg, offset
        )
        
        RENDER_CACHE['side_img'] = draw_sidebar(state, BOTTOM_H, SIDE_W)
        
        RENDER_CACHE['start_sec'] = window_start_sec
        RENDER_CACHE['size'] = (EEG_W, EEG_H)
        RENDER_CACHE['eeg_gain'] = eeg_gain
        RENDER_CACHE['emg_gain'] = emg_gain
        RENDER_CACHE['offset'] = offset
        RENDER_CACHE['window_dur'] = window_dur
        RENDER_CACHE['update_counter'] = state.update_counter

    bottom_row = np.hstack((video_disp, RENDER_CACHE['psd_img'], RENDER_CACHE['side_img']))
    
    eeg_img = RENDER_CACHE['eeg_base'].copy()
    playhead_x = int((playback_offset_sec / window_dur) * EEG_W)
    cv2.line(eeg_img, (playhead_x, 0), (playhead_x, EEG_H), (255, 255, 255), 2)
    
    if state.paint_mode == 'Range' and state.range_start_sec is not None:
        if window_start_sec <= state.range_start_sec <= window_start_sec + window_dur:
            rx = int(((state.range_start_sec - window_start_sec) / window_dur) * EEG_W)
            cv2.line(eeg_img, (rx, 0), (rx, EEG_H), (0, 0, 255), 2)
    
    return np.vstack((eeg_img, bottom_row))
