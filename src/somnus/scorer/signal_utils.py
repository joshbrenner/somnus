# signal_utils.py

import numpy as np
from scipy.signal import welch

def compute_psd(data, sfreq, fmin=0, fmax=200):
    """How much power the signal carries at each frequency, for display."""
    if data.shape[1] < sfreq: return None, None
    n_ch_use = min(3, data.shape[0])
    data_use = data[:n_ch_use, :]
    nperseg = min(int(sfreq * 2), data_use.shape[1]) 
    freqs, psd = welch(data_use, fs=sfreq, nperseg=nperseg)
    avg_psd = np.mean(psd, axis=0)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    return freqs[idx], 10 * np.log10(avg_psd[idx] + 1e-12)

def get_nice_number(val, round_up=False):
    """Round a value to a neat 1, 2 or 5 so axis labels read well."""
    if val == 0: return 1
    exponent = np.floor(np.log10(val))
    fraction = val / (10**exponent)
    if round_up:
        if fraction <= 1: nice_frac = 1
        elif fraction <= 2: nice_frac = 2
        elif fraction <= 5: nice_frac = 5
        else: nice_frac = 10
    else:
        if fraction >= 5: nice_frac = 5
        elif fraction >= 2: nice_frac = 2
        else: nice_frac = 1
    return nice_frac * (10**exponent)