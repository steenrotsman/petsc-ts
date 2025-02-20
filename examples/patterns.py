from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sktime.datasets import load_from_tsfile, load_UCR_UEA_dataset

from petsc.classifier import PetsClassifier

FOLDER = "examples/data/tetris/"
COLS = ["gaze_angle_x", "gaze_angle_y", "ear"]
TRAIN_FILES = [f"23400_{col}_0_4_TRAIN.ts" for col in COLS]
TEST_FILES = [f"23400_{col}_0_4_TEST.ts" for col in COLS]


def main():
    X_train, y_train, X_test, y_test = get_gun_point()
    pets = PetsClassifier(window=24, duration=1, min_size=5, k=10, multiresolution=True)
    pets.fit(X_train, y_train)
    y_pred = pets.predict(X_test)
    print(np.mean(y_pred == y_test))
    for ts, y_hat, y_true in zip(X_test, y_pred, y_test):
        attribution = pets.get_attribution(ts)
        fig, axs = plt.subplots(ts.shape[0], sharex=True, layout="constrained")
        if ts.shape[0] == 1:
            axs = [axs]
        xs = np.arange(ts.shape[1])
        for signal, att, ax in zip(ts, attribution, axs):
            att = zscore(att)
            for i in range(len(xs) - 1):
                x1, x2 = xs[i], xs[i + 1]
                y1, y2 = signal[i], signal[i + 1]
                weight = abs(att[i]) + 0.01
                color = "red" if att[i] < 0 else "blue" if att[i] > 0 else "black"
                ax.plot([x1, x2], [y1, y2], lw=weight, c=color)
        ax.set_xlabel(f"pred: {y_hat}, true: {y_true}")
        plt.show()
        plt.close()


def get_tetris():
    X_train = []
    y_train = []
    for train_file in TRAIN_FILES:
        x_train, y_train = load_from_tsfile(
            FOLDER + train_file, return_data_type="np2d"
        )
        X_train.append(x_train)
    X_train = np.stack(X_train).reshape((-1, 3, 23400))

    X_test = []
    y_test = []
    for test_file in TEST_FILES:
        x_test, y_test = load_from_tsfile(FOLDER + test_file, return_data_type="np2d")
        X_test.append(x_test)
    X_test = np.stack(X_test).reshape((-1, 3, 23400))

    return X_train, y_train, X_test, y_test


def get_gun_point():
    load = partial(
        load_UCR_UEA_dataset,
        "GunPoint",
        return_type="np3D",
        extract_path="examples/data/UCRArchive_2018",
    )

    X_train, y_train = load(split="train")
    X_test, y_test = load(split="test")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
