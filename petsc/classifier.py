from itertools import chain

import numpy as np
from aeon.classification import BaseClassifier
from numpy.lib.stride_tricks import sliding_window_view
from sax_ts import sax
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .pets import PetsTransformer


class PetsClassifier(BaseClassifier):
    """PETSC: Pattern-based Embedding for Time Series Classification."""

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
        "y_inner_type": "numpy1D",
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "capability:predict_proba": True,
    }

    def __init__(
        self,
        w=15,
        alpha=4,
        window=15,
        stride=1,
        min_size=5,
        max_size=None,
        duration=1.1,
        k=200,
        sort_alpha=False,
        multiresolution=False,
        soft=False,
        tau=None,
        class_weight=None,
        n_jobs=1,
        random_state=None,
    ):
        if min_size > w:
            raise ValueError("min_size cannot be larger than w.")
        if duration > 1 and soft:
            raise ValueError("soft = True requires duration = 1.0")

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

        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        """Fit PETSC to training data.

        Parameters
        ----------
        X : 3D np.ndarray (any number of channels, equal length series)
                of shape (n_cases, n_channels, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_cases], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
        y : 1D np.array, of shape [n_cases] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        self._transformer = PetsTransformer(
            self.window,
            self.stride,
            self.w,
            self.alpha,
            self.min_size,
            self.max_size,
            self.duration,
            self.k,
            self.sort_alpha,
            self.multiresolution,
            self.soft,
            self.tau,
        )

        self._scaler = StandardScaler()
        self._estimator = SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            penalty="elasticnet",
            loss="log_loss",
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self.pipeline_ = make_pipeline(
            self._transformer,
            self._scaler,
            self._estimator,
        )
        self.pipeline_.fit(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = (n_cases,)
            Predicted class labels.
        """
        return self.pipeline_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = (n_cases, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        return self.pipeline_.predict_proba(X)

    def get_attribution(self, x, reference=None) -> np.ndarray:
        """Get classification attribution of one row.

        Parameters
        ----------
        x : 2D np.ndarray of shape (n_channels, n_timepoints)
            or 1D np.ndarray of shape (n_timepoints)

        Returns
        -------
        attribution :
            Coefficient values of the patterns that each time point is part of.
        """
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
            coefs = -self._estimator.coef_[0]
        elif self.n_classes_ == 2 and self.classes_[1]:
            coefs = self._estimator.coef_[0]
        else:
            coefs = self._estimator.coef_[reference]

        # Give each pattern its coefficient
        for pattern, coef in zip(
            chain.from_iterable(self._transformer.patterns_), coefs
        ):
            pattern.coef = coef

        attribution = np.zeros(x.shape)
        i = zip(enumerate(x), self._transformer.windows_, self._transformer.patterns_)
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
