#!/usr/bin/env python3
"""Quick grid search for QKRR hyperparameters on the new real-world datasets."""

import os
import sys
import itertools
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import (
    load_gaze, load_no2, load_ise,
    ensure_reproducibility,
)
from algos import QKRR


DATASETS = {
    'gaze': load_gaze,
    'no2': load_no2,
    'ise': load_ise,
}

PARAM_GRID = {
    'n_qubits': [6, 8, 10],
    'length_scale': [0.5, 2.0, 10.0],
    'beta': [0.05, 0.2],
    'lambda_reg': [1e-5, 1e-3],
}

N_FOLDS = 3


def evaluate_params(X, y, params, n_folds=5, seed=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = QKRR(pde_weight=0.0, hhl_eps=0.01, track_complexity=False, **params)
        model.fit(X_train, y_train, X_train, np.zeros(len(y_train)))
        preds = model.predict(X_val)
        rmses.append(np.sqrt(np.mean((preds - y_val) ** 2)))
    return np.mean(rmses)


def main():
    ensure_reproducibility(42)
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    for name, loader in DATASETS.items():
        print(f"\n{'='*60}\nTuning QKRR on {name}\n{'='*60}")
        X, y = loader()
        best_rmse = float('inf')
        best_params = None
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            rmse = evaluate_params(X, y, params, n_folds=N_FOLDS, seed=42)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
            print(f"  {params} -> CV RMSE {rmse:.4f}")
        print(f"\nBest for {name}: {best_params} -> CV RMSE {best_rmse:.4f}")


if __name__ == '__main__':
    main()
