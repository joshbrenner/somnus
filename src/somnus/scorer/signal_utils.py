# signal_utils.py

import numpy as np
from scipy.signal import welch

def compute_psd(data, sfreq, fmin=0, fmax=200):
    if data.shape[1] < sfreq: return None, None
    n_ch_use = min(3, data.shape[0])
    data_use = data[:n_ch_use, :]
    nperseg = min(int(sfreq * 2), data_use.shape[1]) 
    freqs, psd = welch(data_use, fs=sfreq, nperseg=nperseg)
    avg_psd = np.mean(psd, axis=0)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    return freqs[idx], 10 * np.log10(avg_psd[idx] + 1e-12)

def get_band_power(data_snippet, sfreq, target_freq, bandwidth=2.0):
    if len(data_snippet) < int(sfreq * 0.1): return -100.0 
    
    nperseg = len(data_snippet)
    target_res = 0.2
    nfft = int(sfreq / target_res)
    nfft = max(nfft, nperseg)

    freqs, psd = welch(data_snippet, fs=sfreq, nperseg=nperseg, nfft=nfft)
    
    f_min = max(0, target_freq - bandwidth/2)
    f_max = target_freq + bandwidth/2
    idx = np.where((freqs >= f_min) & (freqs <= f_max))
    
    if len(idx[0]) == 0: return -100.0
    
    peak_power = np.max(psd[idx])
    return 10 * np.log10(peak_power + 1e-12)

def get_nice_number(val, round_up=False):
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