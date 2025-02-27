from itertools import cycle

import numpy as np
from aeon.transformations.base import BaseTransformer
from petsc_miner import PatternMiner

from .preprocessing import SAX


class PetsTransformer(BaseTransformer):
    """PETS: Create Pattern-based Embeddings from Time Series."""

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "y_inner_type": "numpy1D",
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "capability:predict_proba": True,
        "output_data_type": "Tabular",
        "fit_is_empty": False,
        "requires_y": False,
    }

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
        self._data = []

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self._transform()

    def fit(self, X, y=None):
        # Mine each multivariate signal separately
        for ts in self._get_signals(X):
            # Start with window at min ts length and reduce window each iteration
            if self.multiresolution:
                self.sax.window = min(row.shape[1] for row in X)

            while True:
                discrete, labels = self.sax.transform(ts)
                patterns = self.miner.mine(discrete)

                # Store for use in _transform
                self._data.append((discrete, labels))
                self.patterns_.append(patterns)
                self.windows_.append(self.sax.window)

                # Add more patterns with halved window while window >= SAX word size
                if self.multiresolution:
                    self.sax.window //= 2
                    if self.sax.window < self.w:
                        break
                else:
                    break
        return self

    def transform(self, X):
        return self._transform(X)

    def _transform(self, X=None):
        # Caller is fit_transform; _data is already filled, else preprocess new data
        if X is not None:
            self._data = []
            for ts in self._get_signals(X):
                for window in self.windows_:
                    self.sax.window = window
                    self._data.append(self.sax.transform(ts))

        embedding = []

        it = zip(cycle(self._data), self.windows_, self.patterns_)
        for (discrete, labels), window, patterns in it:
            # Col contains all columns that correspond to patterns mined with window
            col = np.zeros((len(set(labels)), len(patterns)), dtype=int)
            for pat_idx, pattern in enumerate(patterns):
                if X is None:
                    projection = pattern.projection
                else:
                    projection = self.miner.project(discrete, pattern)
                if len(projection) > 0:
                    # Sum the number of windows in each time series where pattern occurs
                    ts_indices = np.array([labels[idx] for idx in projection])
                    np.add.at(col, (ts_indices, pat_idx), 1)

                # PETSC-soft allows approximate matching of patterns
                if self.soft:
                    for window_idx, window in enumerate(discrete):
                        ts_idx = labels[window_idx]

                        # Only check for approximate match if there was no exact match
                        if window_idx not in projection:
                            col[ts_idx][pat_idx] += self._find(window, pattern.pattern)
            embedding.append(col)

        embedding = np.concatenate(embedding, axis=1)
        return embedding

    def _get_signals(self, X):
        """Split X into list of separate multivariate signals."""
        return [[x[signal] for x in X] for signal in range(X[0].shape[0])]

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
