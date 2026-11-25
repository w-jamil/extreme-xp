#!/usr/bin/env python3
"""Focused grid search around the best coarse QKRR hyperparameters."""

import os
import sys
import itertools
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import load_gaze, load_no2, load_ise, ensure_reproducibility
from algos import QKRR


DATASETS = {
    'gaze': (load_gaze, {
        'n_qubits': [6, 8, 10],
        'length_scale': [5.0, 10.0, 20.0, 50.0],
        'beta': [0.01, 0.03, 0.05, 0.07, 0.1],
        'lambda_reg': [1e-6, 1e-5, 1e-4],
    }),
    'no2': (load_no2, {
        'n_qubits': [4, 6, 8],
        'length_scale': [5.0, 10.0, 20.0, 50.0],
        'beta': [0.01, 0.03, 0.05, 0.07, 0.1],
        'lambda_reg': [1e-5, 1e-4, 1e-3],
    }),
    'ise': (load_ise, {
        'n_qubits': [4, 6, 8],
        'length_scale': [5.0, 10.0, 20.0, 50.0],
        'beta': [0.01, 0.03, 0.05, 0.07, 0.1],
        'lambda_reg': [1e-5, 1e-4, 1e-3],
    }),
}

N_FOLDS = 3


def evaluate_params(X, y, params, seed=42):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
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
    for name, (loader, grid) in DATASETS.items():
        print(f"\n{'='*60}\nFine tuning QKRR on {name}\n{'='*60}")
        X, y = loader()
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        best_rmse = float('inf')
        best_params = None
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            rmse = evaluate_params(X, y, params)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
            print(f"  {params} -> CV RMSE {rmse:.4f}")
        print(f"\nBest for {name}: {best_params} -> CV RMSE {best_rmse:.4f}")


if __name__ == '__main__':
    main()
