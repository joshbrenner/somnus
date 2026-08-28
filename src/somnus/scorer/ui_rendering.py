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

MENU_HEADERS = {
    'File': (0, 60),
    'Brush': (60, 130),
    'View': (130, 200),
    'Mode': (200, 270),
    'Brush Size': (270, 370)
}

MENU_ITEMS = {
    'File': ['Save CSV'],
    'Brush': ['Wake', 'NREM', 'REM', 'Artifact', 'Unclear', 'Confirm', 'Erase'],
    'View': ['4 sec', '10 sec', '30 sec', '1 min'],
    'Mode': ['Paint Mode', 'Range Mode'],
    'Brush Size': ['Min Size', 'Decrease (-)', 'Increase (+)']
}

RENDER_CACHE = {
    'start_sec': -1,
    'eeg_gain': -1.0,
    'emg_gain': -1.0,
    'offset': -1.0,
    'window_dur': -1.0, 
    'update_counter': -1,
    'init_emg_log': None,
    'psd_img': None,
    'eeg_base': None,
    'side_img': None,
    'total_h': 0
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
    
    overlay = img.copy()
    end_time_sec = window_start_sec + window_duration
    
    view_df = state.df[(state.df['Time_sec'] >= window_start_sec) & (state.df['Time_sec'] < end_time_sec)]

    # A bin the user has taken a position on -- confirmed, or painted to
    # something other than what the model said -- is drawn near-opaque; a bin
    # still carrying the model's own label is a faint wash of the same color.
    has_model = 'Model_State' in view_df.columns
    manual_spans = []
    for _, row in view_df.iterrows():
        bin_start = row['Time_sec']
        bin_state = row['State']

        x1 = int(((bin_start - window_start_sec) / window_duration) * w)
        x2 = int(((bin_start + state.bin_step - window_start_sec) / window_duration) * w)

        color = COLORS.get(bin_state, (50, 50, 50))
        manual = bool(row.get('Confirmed', 0)) or (
            bin_state != 'Unknown'
            and (row['Model_State'] != bin_state if has_model else True))
        if manual:
            manual_spans.append((x1, x2, color))
            continue
        cv2.rectangle(overlay, (max(0, x1), 0), (min(w, x2), h), color, -1)

    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    if manual_spans:
        ov2 = img.copy()
        for x1, x2, color in manual_spans:
            cv2.rectangle(ov2, (max(0, x1), 0), (min(w, x2), h), color, -1)
        cv2.addWeighted(ov2, 0.85, img, 0.15, 0, img)

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

        if i == n_ch - 1: 
            bar_x = w - 15; bar_y = (i + 1) * ch_h - 15
            
            effective_v_range = v_range / current_gain
            volts_per_pixel = effective_v_range / ch_h
            target_v_size = effective_v_range * 0.25 
            
            if target_v_size < 1e-6: unit, mult = "nV", 1e9
            elif target_v_size < 1e-3: unit, mult = "uV", 1e6
            else: unit, mult = "mV", 1e3
            
            nice_val_units = get_nice_number(target_v_size * mult, round_up=True)
            nice_val_volts = nice_val_units / mult
            pixel_height = int(nice_val_volts / volts_per_pixel)
            
            seconds_per_pixel = window_duration / w
            target_t_size = 20 * seconds_per_pixel 
            nice_t_val = get_nice_number(target_t_size, round_up=True)
            pixel_width = int(nice_t_val / seconds_per_pixel)
            
            color = (180, 180, 180)
            cv2.line(img, (bar_x, bar_y), (bar_x, bar_y - pixel_height), color, 2)
            cv2.line(img, (bar_x, bar_y), (bar_x - pixel_width, bar_y), color, 2)
            cv2.putText(img, f"{int(nice_val_units)} {unit}", (bar_x - 50, bar_y - pixel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if nice_t_val >= 1.0: t_label = f"{nice_t_val:.1f}s".replace(".0s", "s")
            elif nice_t_val >= 0.1: t_label = f"{nice_t_val:.2f}s" 
            elif nice_t_val >= 0.001: t_label = f"{int(nice_t_val * 1000)}ms"
            else: t_label = f"{nice_t_val * 1e6:.0f}us"
                
            cv2.putText(img, t_label, (bar_x - pixel_width - 5, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img

def draw_menubar(img, w, h, state):
    """Draw the menu strip across the top of the window."""
    cv2.rectangle(img, (0, 0), (w, h), (40, 40, 40), -1)
    cv2.line(img, (0, h-1), (w, h-1), (100, 100, 100), 1)

    for menu_name, (x1, x2) in MENU_HEADERS.items():
        is_active = (state.active_menu == menu_name)
        bg_color = (80, 80, 80) if is_active else (40, 40, 40)
        text_color = (255, 255, 255)

        cv2.rectangle(img, (x1, 0), (x2, h), bg_color, -1)
        text_sz, _ = cv2.getTextSize(menu_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        txt_x = x1 + (x2 - x1 - text_sz[0]) // 2
        txt_y = (h + text_sz[1]) // 2
        cv2.putText(img, menu_name, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    status_text = (f"Active: {state.active_brush or 'None'} | "
                   f"View: {state.window_width_sec}s | {state.paint_mode} | "
                   f"Size: {state.brush_size_sec:.1f}s | "
                   f"ENTER saves, ESC closes")
    text_sz, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, status_text, (w - text_sz[0] - 20, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img

def draw_dropdown(img, state):
    """Draw the open menu's list of options."""
    if not state.active_menu: return

    x_start = MENU_HEADERS[state.active_menu][0]
    items = MENU_ITEMS[state.active_menu]
    dd_w = 150
    dd_h = len(items) * 30
    y_start = 30

    cv2.rectangle(img, (x_start, y_start), (x_start + dd_w, y_start + dd_h), (50, 50, 50), -1)
    cv2.rectangle(img, (x_start, y_start), (x_start + dd_w, y_start + dd_h), (200, 200, 200), 1)

    for i, item in enumerate(items):
        y_item = y_start + i * 30
        
        is_active = False
        if state.active_menu == 'Brush' and item == state.active_brush: is_active = True
        if state.active_menu == 'View' and str(int(state.window_width_sec)) in item: is_active = True
        if state.active_menu == 'Mode' and state.paint_mode in item: is_active = True

        if is_active:
            cv2.putText(img, ">", (x_start + 10, y_item + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(img, item, (x_start + 30, y_item + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_sidebar(state, h, w_side):
    """Draw the side panel: bout navigation and the model-review controls."""
    img = np.zeros((h, w_side, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w_side, h), (30, 30, 30), -1)
    cv2.line(img, (0, 0), (0, h), (100, 100, 100), 1)
    
    cv2.putText(img, "BOUT NAVIGATION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(img, (10, 35), (w_side-10, 35), (100, 100, 100), 1)
    cv2.putText(img, "Click below to Jump:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Only the three sleep states get jump buttons -- see NAV_LABELS.
    valid_labels = NAV_LABELS

    for idx, label in enumerate(valid_labels):
        row = idx // 2
        col = idx % 2
        
        x_start = 10 + col * 92
        x_end = x_start + 85
        y_start = 80 + row * 50
        y_end = y_start + 40
        
        color = COLORS[label]
        ui_color = (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7))
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), ui_color, -1)
        
        font_scale = 0.45
        thickness = 1
        display_text = label
        text_sz, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        if text_sz[0] > 75:
            font_scale = 0.4
            text_sz, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        txt_x = x_start + (85 - text_sz[0]) // 2
        txt_y = y_start + 20 + text_sz[1] // 2

        cv2.putText(img, display_text, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # --- model review controls (only when the GUI supplied review metadata) ---
    # Rows used by the buttons above: ceil(len(NAV_LABELS)/2) at 50 px each.
    rows = (len(valid_labels) + 1) // 2
    y = 80 + rows * 50 + 4
    state.uncertain_btn = None
    if getattr(state, 'review_meta', None) is not None:
        # There may be as little as 250 px of height here, so each control is
        # drawn only if it fits.
        if y + 46 <= h:
            cv2.line(img, (10, y), (w_side - 10, y), (100, 100, 100), 1)
            y += 6

            # Jumps forward through the recording to the next epoch the model
            # was unsure about. 
            bx1, bx2 = 10, w_side - 10
            by1, by2 = y, y + 34
            state.uncertain_btn = (bx1, by1, bx2, by2)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (150, 110, 20), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (220, 180, 60), 1)
            cv2.putText(img, "Next low certainty", (bx1 + 8, by1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            y = by2 + 15

            thr = float(getattr(state, 'certainty_threshold', 0.80))
            if y + 4 <= h:
                cv2.putText(img, f"thr {thr:.2f}  u/U  [ / ]", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)
    return img

def render_composite(state, frame, eeg_slice, ch_names, sfreq, window_start_sec, playback_offset_sec, window_dur, eeg_gain, emg_gain, offset, v_range, active_brush, num_eeg, init_emg_log):
    """Assemble one complete frame of the scorer window."""
    
    VIDEO_W = state.video_w
    EEG_W = state.eeg_w
    EEG_H = state.eeg_h
    SIDE_W = state.side_w
    PSD_W = EEG_W - VIDEO_W - SIDE_W
    BAR_H = 30
    BOTTOM_H = getattr(state, 'bottom_h', 340)

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
        RENDER_CACHE['eeg_gain'] != eeg_gain or 
        RENDER_CACHE['emg_gain'] != emg_gain or 
        RENDER_CACHE['offset'] != offset or 
        RENDER_CACHE['window_dur'] != window_dur or 
        RENDER_CACHE['update_counter'] != state.update_counter or
        RENDER_CACHE['init_emg_log'] != init_emg_log or
        RENDER_CACHE['psd_img'] is None):
        
        RENDER_CACHE['psd_img'] = draw_state_probability_panel(
            state, PSD_W, BOTTOM_H, window_start_sec, window_dur)
        
        RENDER_CACHE['eeg_base'] = draw_eeg_side_panel(
            state, eeg_slice, ch_names, window_start_sec, window_dur, sfreq, EEG_W, EEG_H, eeg_gain, emg_gain, v_range, num_eeg, offset
        )
        
        RENDER_CACHE['side_img'] = draw_sidebar(state, BOTTOM_H, SIDE_W)
        
        RENDER_CACHE['start_sec'] = window_start_sec
        RENDER_CACHE['eeg_gain'] = eeg_gain
        RENDER_CACHE['emg_gain'] = emg_gain
        RENDER_CACHE['offset'] = offset
        RENDER_CACHE['window_dur'] = window_dur
        RENDER_CACHE['update_counter'] = state.update_counter
        RENDER_CACHE['init_emg_log'] = init_emg_log
        RENDER_CACHE['total_h'] = EEG_H 

    bottom_row = np.hstack((video_disp, RENDER_CACHE['psd_img'], RENDER_CACHE['side_img']))
    
    eeg_img = RENDER_CACHE['eeg_base'].copy()
    playhead_x = int((playback_offset_sec / window_dur) * EEG_W)
    cv2.line(eeg_img, (playhead_x, 0), (playhead_x, EEG_H), (255, 255, 255), 2)
    
    if state.paint_mode == 'Range' and state.range_start_sec is not None:
        if window_start_sec <= state.range_start_sec <= window_start_sec + window_dur:
            rx = int(((state.range_start_sec - window_start_sec) / window_dur) * EEG_W)
            cv2.line(eeg_img, (rx, 0), (rx, EEG_H), (0, 0, 255), 2)
    
    main_layout = np.vstack((eeg_img, bottom_row))
    
    toolbar = np.zeros((BAR_H, EEG_W, 3), dtype=np.uint8)
    toolbar = draw_menubar(toolbar, EEG_W, BAR_H, state)
    
    final_img = np.vstack((toolbar, main_layout))
    
    if state.active_menu:
        draw_dropdown(final_img, state)
        
    return final_img
