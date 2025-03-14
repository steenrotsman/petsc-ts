from time import perf_counter

import numpy as np
from aeon.datasets import load_from_ts_file
from petsc_ts.classifier import PetsClassifier

NAME = "ArticularyWordRecognition"
PATH = "examples/data/multivariate/ArticularyWordRecognition/"


X_train, y_train = load_from_ts_file(PATH + NAME + "_TRAIN.ts")
X_test, y_test = load_from_ts_file(PATH + NAME + "_TEST.ts")

pets = PetsClassifier(duration=1, window=48, min_size=5, stride=5, k=50)
start = perf_counter()
pets.fit(X_train, y_train)
end = perf_counter()
print(end - start)
y_pred = pets.predict(X_test)
print(np.mean(y_pred == y_test))
