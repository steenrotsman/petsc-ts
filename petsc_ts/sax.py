"""Implementation of SAX using numpy, assuming window%w==0."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm, zscore


def sax(ts, window, stride, w, alpha):
    # Create sliding windows and z-normalise each sliding window
    windows = sliding_window_view(ts, window)[::stride]
    windows = zscore(windows, axis=1)

    # PAA
    paa = windows.reshape((len(windows), w, -1)).mean(axis=2)

    # SAX
    sax = np.digitize(paa, np.array(get_breakpoints(alpha)))
    sax = [[chr(ord("a") + x) for x in row] for row in sax]
    return sax


def get_breakpoints(alpha):
    return norm.ppf(np.arange(1, alpha) / alpha, loc=0)
