import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from aeon.datasets import load_gunpoint
from aeon.datasets import load_from_ts_file
from petsc_ts.classifier import PetsClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
n_splits = 3
n_trials = 50

def main():
    X_train, y_train = load_gunpoint(split="TRAIN")
    X_test, y_test = load_gunpoint(split="TEST")
    # X_train, y_train = load_from_ts_file("data/ts/0_ear_0_4_TRAIN.ts")
    # X_test, y_test = load_from_ts_file("data/ts/0_ear_0_4_TEST.ts")

    # infer series length for safe window upper bound
    series_length = X_train.shape[-1]

    def objective(trial: optuna.Trial) -> float:
        w = trial.suggest_int("w", 10, 150)
        max_stride = max(1, w // 2)
        stride = trial.suggest_int("stride", 1, max_stride)
        alpha = trial.suggest_int("alpha", 2, 10)
        k = trial.suggest_int("k", 10, 500)

        # --- evaluation with cross-validation on the TRAIN split ---
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[valid_idx]
            y_tr, y_val = y_train[train_idx], y_train[valid_idx]

            # Create classifier with sampled hyperparameters
            pets = PetsClassifier(
                k=k,
                w=w,
                stride=stride,
                alpha=alpha,
                random_state=RANDOM_STATE,
            )

            pets.fit(X_tr, y_tr)
            y_pred = pets.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        # Optuna will try to maximize this
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)  # adjust n_trials as you like

    print("Number of finished trials:", len(study.trials))
    print("Best CV accuracy:", study.best_value)
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # --- train final model on full TRAIN with best params, evaluate on TEST ---
    best_params = study.best_params

    # recompute safe window bound for test training (just reusing series_length)
    w_best = best_params["w"]
    stride_best = best_params["stride"]
    alpha_best = best_params["alpha"]
    k_best = best_params["k"]

    pets_best = PetsClassifier(
        k=k_best,
        multiresolution=True,
        w=w_best,
        stride=stride_best,
        alpha=alpha_best,
        random_state=RANDOM_STATE,
    )

    pets_best.fit(X_train, y_train)
    y_pred = pets_best.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"Test accuracy with best hyperparameters: {test_acc:.4f}")

    # save logs and study metadata
    trials_df = study.trials_dataframe()
    trials_df.to_csv("tetris_optuna_trials.csv", index=False)
    with open("tetris_optuna_best_params.txt", "w") as f:
        f.write(str(study.best_params))
    with open("tetris_optuna_experiment_metadata.txt", "w") as f:
        f.write(f"test_accuracy: {test_acc:.4f}\n")
        f.write(f"n_splits: {n_splits}\n")
        f.write(f"n_trials: {n_trials}\n")
        f.write(f"random_state: {RANDOM_STATE}\n")

if __name__ == "__main__":
    main()
