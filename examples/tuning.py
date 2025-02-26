from aeon.datasets import load_gunpoint
from sklearn.model_selection import RandomizedSearchCV

from petsc.classifier import PetsClassifier

X_train, y_train = load_gunpoint(split="TRAIN")
X_test, y_test = load_gunpoint(split="TEST")

PARAMS = [
    {
        "window": range(X_train.shape[-1] // 10, X_train.shape[-1]),
        "w": [w],
        "alpha": range(3, 12),
        "k": range(500, 2500),
        "min_size": range(1, w),
        "duration": [1.0, 1.1, 1.2, 1.5],
    }
    for w in range(5, 30)
]

clsf = RandomizedSearchCV(
    PetsClassifier(window=16, multiresolution=False),
    PARAMS,
    n_iter=64,
    n_jobs=-1,
    verbose=4,
    error_score="raise",
    random_state=31415,
)
clsf.fit(X_train, y_train)
y_pred = clsf.predict(X_test)
print(clsf.best_params_)
print((y_test == y_pred).mean())
