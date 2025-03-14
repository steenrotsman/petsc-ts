import cProfile

from aeon.datasets import load_gunpoint
from petsc_ts.classifier import PetsClassifier

X_train, y_train = load_gunpoint(split="TRAIN")
X_test, y_test = load_gunpoint(split="TEST")

with cProfile.Profile() as pf:
    clsf = PetsClassifier(window=24, k=1000)
    clsf.fit(X_train, y_train)
    y_pred = clsf.predict(X_test)
    pf.print_stats("tottime")
