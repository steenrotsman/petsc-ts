from functools import partial

import numpy as np
from sktime.datasets import load_UCR_UEA_dataset

from petsc.classifier import PetsClassifier

NAMES = ["Beef", "GunPoint"]
PATH = "data/UCRArchive_2018"


load_ucr = partial(load_UCR_UEA_dataset, return_type="np3d", extract_path=PATH)
for name in NAMES:
    X_train, y_train = load_UCR_UEA_dataset(name, split="train")
    X_test, y_test = load_UCR_UEA_dataset(name, split="test")
    pets = PetsClassifier(duration=1, multiresolution=True, soft=True)
    pets.fit(X_train, y_train)
    y_pred = pets.predict(X_test)
    print(name, np.mean(y_pred == y_test))
