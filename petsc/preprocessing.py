import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm, zscore


class SAX:
    def __init__(self, window, stride, w, alpha):
        self.window = window
        self.stride = stride
        self.w = w
        self.alpha = alpha
        self.breakpoints = norm.ppf(np.arange(1, alpha) / alpha, loc=0)

    def transform(self, X):
        discrete = []
        labels = []
        for label, ts in enumerate(X):
            d = self.discretise(ts)
            discrete.append(d)
            labels += [label] * d.shape[0]
        return np.vstack(discrete), labels

    def discretise(self, row):
        # Create sliding windows and z-normalise each sliding window
        windows = sliding_window_view(row, self.window)[:: self.stride]
        windows = zscore(windows, axis=1)

        # Unequal length series are padded with nan, remove resulting windows
        windows = windows[~np.isnan(windows).any(axis=1)]

        # PAA
        indices = np.arange(windows.shape[1] * self.w) // self.w
        indices = indices.reshape(self.w, -1)
        paa = windows[:, indices].sum(axis=2)

        # SAX
        sax = np.digitize(paa, self.breakpoints)

        return sax
