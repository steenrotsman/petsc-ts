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
    """PETSC: Pattern-based Embedding for Time Series Classification.

    Extract frequent sequential patterns from discretised time series, turn counts of
    pattern occurrences into a tabular embedding and train a linear classifier.

    Vanilla PETSC, MR-PETSC and PETSC-SOFT are supported.

    Parameters
    ----------
    w : int, default=15
        Length of SAX words.
    alpha : int, default=4
        Alphabet size of SAX words.
    window : int, default=15
        Length of sliding window for SAX transformation.
    stride : int, default=1
        Stride of sliding window for SAX transformation.
    min_size : int, default=5
        Minimum length of frequent sequential pattern, cannot be larger than w.
    max_size : int, optional
        Maximum length of frequent sequential pattern, default to w.
    duration : float, default=1.1
        Maximum relative duration of frequent sequential patterns. Frequent sequential
        patterns are allowed (duration - 1) * pattern_length gaps in occurrences.
    k : int, default=200
        Number of top frequent sequential patterns to mine.
    sort_alpha : bool, default=False
        Sort patterns alphabetically, affects embedding column ordering.
    multiresolution : bool, default=False
        Use MR-PETSC: start with window at minimum time series length and combine top k
        frequent sequential patterns of all windows longer than w.
    soft : bool, default=False
        Use PETSC-SOFT: allow some deviations in patterns when finding occurrences.
    tau : float, optional
        Control the amoung of deviation allowed for PETSC-SOFT, default to 1/(2*alpha).
    cache : bool, default=True
        Store SAX representations of training data. Requires more memory, but makes fit() more efficient.
    class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
        From sklearn documentation:
        If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        The “balanced_subsample” mode is the same as “balanced” except that weights
        are computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    """

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
        cache=True,
        class_weight=None,
        n_jobs=1,
        random_state=None,
        verbosity=0,
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
        self.verbosity = verbosity
        self.cache = cache

        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
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
            self.verbosity,
            self.cache,
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
        return self.pipeline_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
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
