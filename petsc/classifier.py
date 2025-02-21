from itertools import chain

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sax_ts import sax
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.classification._delegate import _DelegatedClassifier
from sktime.transformations.panel.padder import PaddingTransformer

from .pets import PetsTransformer


class PetsClassifier(_DelegatedClassifier):
    """PETSC: Pattern-based Embedding for Time Series Classification."""

    _tags = {
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:predict_proba": True,
    }

    def __init__(
        self,
        w=15,
        alpha=4,
        window=None,
        stride=1,
        min_size=1,
        max_size=None,
        duration=1.1,
        k=200,
        sort_alpha=False,
        multiresolution=False,
        soft=False,
        tau=None,
    ):
        if window is None and not multiresolution:
            raise ValueError("window must be set if multiresolution = False")
        if window is not None and min_size > window:
            raise ValueError("min_size cannot be larger than window.")
        if duration > 1 and soft:
            raise ValueError("soft = True requires duration = 1.0")
        super().__init__()

        self.window = window
        self.stride = stride
        self.w = w
        self.alpha = alpha
        self.min_size = min_size
        self.max_size = max_size
        self.duration = duration
        self.k = k
        self.sort_alpha = sort_alpha
        self.multiresolution = multiresolution
        self.soft = soft
        self.tau = tau

        padder = PaddingTransformer(fill_value=np.nan)
        self.pets = PetsTransformer(
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
        )

        scaler = StandardScaler()
        self.classifier = RidgeClassifierCV()
        self.estimator_ = make_pipeline(padder, self.pets, scaler, self.classifier)

    def get_attribution(self, x, reference=None):
        if self.duration > 1:
            raise ValueError("Can only get attribution of patterns if duration=1")
        if self.soft:
            raise ValueError("Can only get attribution of patterns if soft=False")

        if reference is None:
            reference = self.predict(np.expand_dims(x, 0))[0][0]
        elif reference in self.classes_:
            reference = np.nonzero(self.classes_ == reference)[0][0]
        else:
            raise ValueError(f"Class {reference} must be one of {self.classes_}")
        if self.n_classes_ == 2 and reference == self.classes_[0]:
            coefs = -self.classifier.coef_
        elif self.n_classes_ == 2 and self.classes_[1]:
            coefs = self.classifier.coef_
        else:
            coefs = self.classifier.coef_[reference]

        # Give each pattern its coefficient
        for pattern, coef in zip(chain.from_iterable(self.pets.patterns_), coefs):
            pattern.coef = coef

        attribution = np.zeros(x.shape)
        i = zip(enumerate(x), self.pets.windows, self.pets.patterns_)
        for (signal, ts), window, patterns in i:
            self.window = window
            discrete = sax(ts, self.window, self.stride, self.w, self.alpha)

            for pattern in patterns:
                win = sliding_window_view(discrete, len(pattern.pattern), axis=1)
                indexes = np.where(np.all(win == pattern.pattern, axis=2))
                for w_idx, start in zip(*indexes):
                    factor = window / self.w
                    ts_start = w_idx * self.stride + int(start * factor)
                    end = start + len(pattern.pattern)
                    ts_end = w_idx * self.stride + int(end * factor)
                    attribution[signal, ts_start:ts_end] += pattern.coef

        return attribution
