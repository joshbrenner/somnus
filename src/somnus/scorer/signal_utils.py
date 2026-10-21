# signal_utils.py

import numpy as np

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