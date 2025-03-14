import matplotlib.pyplot as plt
import numpy as np
from aeon.datasets import load_gunpoint
from petsc_ts.classifier import PetsClassifier


def main():
    X_train, y_train = load_gunpoint(split="TRAIN")
    X_test, y_test = load_gunpoint(split="TEST")
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
            att = (att - np.mean(att)) / np.std(att)
            for i in range(len(xs) - 1):
                x1, x2 = xs[i], xs[i + 1]
                y1, y2 = signal[i], signal[i + 1]
                weight = abs(att[i]) + 0.01
                color = "red" if att[i] < 0 else "blue" if att[i] > 0 else "black"
                ax.plot([x1, x2], [y1, y2], lw=weight, c=color)
        ax.set_xlabel(f"pred: {y_hat}, true: {y_true}")
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
