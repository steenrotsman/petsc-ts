import numpy as np
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
        clsf = RidgeClassifierCV()
        self.estimator_ = make_pipeline(padder, self.pets, scaler, clsf)
