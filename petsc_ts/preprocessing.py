import numpy as np
from sax_ts import sax as general_sax

from .sax import sax as efficient_sax


class SAX:
    def __init__(self, window, stride, w, alpha):
        self.window = window
        self.stride = stride
        self.w = w
        self.alpha = alpha

    def transform(self, X):
        sax = efficient_sax if self.window % self.w == 0 else general_sax
        discrete = []
        labels = []
        for label, ts in enumerate(X):
            d = sax(ts, self.window, self.stride, self.w, self.alpha)
            discrete.append(d)
            labels += [label] * len(d)
        return np.vstack(discrete), labels
