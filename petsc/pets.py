from itertools import cycle

import numpy as np
from petsc_miner import Pattern, PatternMiner

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
        self.windows_ = []
        self.data_ = []

    def fit_transform(self, X, y=None):
        self.fit(X)

        embedding = np.zeros((len(X), 0), dtype=int)

        it = zip(self.data_, self.windows_, self.patterns_)
        for (discrete, labels), window, patterns in it:
            col = np.zeros((len(set(labels)), len(patterns)), dtype=int)
            for pat_idx, pattern in enumerate(patterns):
                for window_idx, window in enumerate(discrete):
                    ts_idx = labels[window_idx]
                    if window_idx in pattern.projection:
                        col[ts_idx][pat_idx] += 1
                    elif self.soft:
                        col[ts_idx][pat_idx] += self._find(window, pattern.pattern)
            embedding = np.hstack((embedding, col))

        return embedding

    def fit(self, X, y=None):
        # Mine each multivariate signal separately
        n_signals = X[0].shape[0]
        signals = [[x[signal] for x in X] for signal in range(n_signals)]
        for signal, ts in enumerate(signals):
            # Start with window at min ts length and reduce window each iteration
            if self.multiresolution:
                self.sax.window = min(row.shape[1] for row in X)
            while True:
                discrete, labels = self.sax.transform(ts)
                self.data_.append((discrete, labels))
                patterns = self.miner.mine(discrete)
                self.patterns_.append(patterns)
                self.windows_.append(self.sax.window)
                if self.multiresolution:
                    self.sax.window //= 2
                    if self.sax.window < self.w:
                        break
                else:
                    break
        return self

    def transform(self, X):
        embedding = np.zeros((len(X), 0), dtype=int)
        if self.soft:
            embed = self._embed_soft
        else:
            embed = self._embed

        n_signals = X[0].shape[0]
        signals = cycle([[x[signal] for x in X] for signal in range(n_signals)])
        it = zip(signals, self.windows_, self.patterns_)
        for ts, window, patterns in it:
            self.sax.window = window
            discrete, labels = self.sax.transform(ts)
            embedding = np.hstack((embedding, embed(discrete, labels, patterns)))

        return embedding

    def _embed(self, windows, labels, patterns):
        embedding = np.zeros((len(set(labels)), len(patterns)), dtype=int)
        for pat_idx, pattern in enumerate(patterns):
            projection = self._project(windows, pattern)
            for window_idx in projection.keys():
                ts_idx = labels[window_idx]
                embedding[ts_idx][pat_idx] += 1
        return embedding

    def _project(self, ts, pattern):
        item = pattern.pattern[0]
        projection = self.miner.compute_projection_singleton(ts, item)
        candidates = self.miner.get_candidates(ts, projection, [item])
        current_pattern = Pattern([item], projection, candidates)

        for item in pattern.pattern[1:]:
            p = self.miner.compute_projection_incremental(ts, current_pattern, item)
            current_pattern.projection = p
            current_pattern.pattern += [item]

        return current_pattern.projection

    def _embed_soft(self, windows, labels, patterns):
        embedding = np.zeros((len(set(labels)), len(patterns)), dtype=int)
        for pat_idx, pattern in enumerate(patterns):
            for window_idx, window in enumerate(windows):
                ts_idx = labels[window_idx]
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
