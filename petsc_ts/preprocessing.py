import numpy as np
from sax_ts import sax


class SAX:
    def __init__(self, window, stride, w, alpha):
        self.window = window
        self.stride = stride
        self.w = w
        self.alpha = alpha

    def transform(self, X):
        discrete = []
        labels = []
        for label, ts in enumerate(X):
            d = sax(ts, self.window, self.stride, self.w, self.alpha)
            discrete.append(d)
            labels += [label] * len(d)
        return np.vstack(discrete), labels
