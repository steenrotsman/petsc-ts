from random import randint, seed
from time import perf_counter

import numpy as np
from aeon.datasets import load_classification

from petsc.classifier import PetsClassifier

NAMES = ["Beef", "GunPoint"]
PATH = "examples/data/UCRArchive_2018"


for name in NAMES:
    X_train, y_train = load_classification(name, extract_path=PATH, split="train")
    X_test, y_test = load_classification(name, extract_path=PATH, split="test")
    seed(123)
    X_train = [row[:, : randint(100, X_train.shape[-1])] for row in X_train]
    X_test = [row[:, : randint(100, X_test.shape[-1])] for row in X_test]
    pets = PetsClassifier(duration=1, min_size=5, k=50, multiresolution=True)
    start = perf_counter()
    pets.fit(X_train, y_train)
    end = perf_counter()
    print(end - start)
    y_pred = pets.predict(X_test)
    print(name, np.mean(y_pred == y_test))
