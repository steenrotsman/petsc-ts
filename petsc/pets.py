import numpy as np

from .miner import PatternMiner
from .preprocessing import SAX


class PetsTransformer:
    def __init__(
        self,
        window,
        stride,
        w,
        alpha,
        min_size,
        max_size,
        duration,
        k,
        sort_alpha,
        multiresolution,
        soft,
        tau,
    ):
        self.min_size = min_size
        self.w = w
        max_size = max_size if max_size is not None else w
        self.max_size = max_size
        self.multiresolution = multiresolution
        self.soft = soft
        self.tau = tau if tau is not None else 1 / (2 * alpha)

        self.sax = SAX(window, stride, w, alpha)
        self.miner = PatternMiner(alpha, min_size, max_size, duration, k, sort_alpha)

        self.patterns_ = []
        self.windows = []

    # Technically, fit_transform would be faster than .fit().transform(), because the
    # separate calls require separate discretisation calls. However, discretisation
    # is very efficient and separate methods are more maintainable
    def fit(self, X, y=None):
        # Start with window at min ts length and reduce window each iteration while >w
        if self.multiresolution:
            self.sax.window = np.min(np.sum(~np.isnan(X), axis=2))

        # Mine each multivariate signal separately
        for signal, ts in enumerate(np.moveaxis(X, 1, 0)):
            while True:
                discrete, _ = self.sax.transform(ts)
                patterns = self.miner.mine(discrete)
                for pattern in patterns:
                    pattern.signal = signal
                    pattern.window = self.sax.window
                self.patterns_.append(patterns)
                self.windows.append(self.sax.window)
                self.sax.window //= 2
                if not self.multiresolution or self.sax.window < self.w:
                    break
        return self

    def transform(self, X):
        embedding = np.zeros((X.shape[0], 0), dtype=int)
        it = zip(np.moveaxis(X, 1, 0), self.windows, self.patterns_)
        for ts, window, patterns in it:
            self.sax.window = window
            discrete, labels = self.sax.transform(ts)
            if self.soft:
                embedding = np.hstack(
                    (embedding, self._embed_soft(discrete, labels, patterns))
                )
            else:
                embedding = np.hstack((embedding, self._embed(labels, patterns)))
        return embedding

    def _embed(self, labels, patterns):
        embedding = np.zeros((len(set(labels)), len(patterns)), dtype=int)
        for pat_idx, pattern in enumerate(patterns):
            for window_idx in pattern.projection.keys():
                ts_idx = labels[window_idx]
                embedding[ts_idx][pat_idx] += 1
        return embedding

    def _embed_soft(self, windows, labels, patterns):
        embedding = np.zeros((len(set(labels)), len(patterns)), dtype=int)
        for pat_idx, pattern in enumerate(patterns):
            for window_idx, window in enumerate(windows):
                ts_idx = labels[window_idx]
                if window_idx in pattern.projection:
                    embedding[ts_idx][pat_idx] += 1
                else:
                    embedding[ts_idx][pat_idx] += self._find(window, pattern.pattern)
        return embedding

    def _find(self, window, pattern):
        max_dist = (self.tau * len(pattern)) ** 2
        for i in range(len(window) - len(pattern) + 1):
            dist = 0
            for j in range(len(pattern)):
                dist += (window[i + j] - pattern[j]) ** 2
                if dist > max_dist:
                    break
            if dist < max_dist:
                return 1
        return 0
