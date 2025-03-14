# PETSC #

Pattern-based Embedding for Time Series Classification (PETSC).

This repository contains a Python/C++ implementation of PETSC, that is compatible with [`aeon`](https://aeon-toolkit.org). Two variants of PETSC are supported as well: MR-PETSC and PETSC-SOFT.

# Installation
The easiest way to install `petsc-ts` is via pip:
```
pip install petsc-ts
```

If you want to run the tests locally (currently only an `aeon` test to check if the estimator complies with the required format), install the optional dependency:
```
pip install "petsc-ts[test]"
```

You can then import the `PetsClassifier` class as follows and use it as any other `scikit-learn`-compatible estimator:
```Python
from petsc_ts.classifier import PetsClassifier

clsf = PetsClassifier()
clsf.fit(X_train, y_train)
clsf.predict(X_test)
clsf.predict_proba(X_test)
```