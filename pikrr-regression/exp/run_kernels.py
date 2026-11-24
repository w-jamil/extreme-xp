#!/usr/bin/env python3
"""run_kernels.py — Classical KRR / IKRR (and PI-KRR / PI-IKRR, run data-only)
on the five real-world regression benchmarks.

These use the same loaders, sample cap (max_samples=1000), seed, and
train/test split as exp_real_accuracy, so the results are directly comparable
with the existing QKRR / neural rows in results/expR1_accuracy_*.csv.

Because the real-world benchmarks have no governing PDE, the physics-informed
variants are run without collocation: PI-KRR coincides with KRR and PI-IKRR
with IKRR.  All four are reported so the table mirrors the physics-informed
study; the coincidence is stated in the manuscript.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (load_household_power, load_nslkdd, load_gaze,
                         load_no2, load_ise, ensure_reproducibility,
                         _save_table)
from algos import KRRReal, IterativeKRRReal

HERE = os.path.dirname(os.path.abspath(__file__))

LOADERS = {
    'household': lambda ms: load_household_power(max_samples=ms, pca_components=3),
    'nslkdd': lambda ms: load_nslkdd(max_samples=ms, pca_components=3),
    'gaze': lambda ms: load_gaze(max_samples=ms, pca_components=None),
    'no2': lambda ms: load_no2(max_samples=ms, pca_components=None),
    'ise': lambda ms: load_ise(max_samples=ms, pca_components=None),
}
TEMPORAL = ('household', 'ise')

LS_GRID = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
LAM_GRID = [1e-6, 1e-4, 1e-2, 1e-1]


def _split(name, X, y, seed):
    n = len(y)
    n_train = int(0.8 * n)
    if name in TEMPORAL:
        return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _tune(cls, Xtr, ytr):
    """Select (length_scale, lambda_reg) on an inner validation split."""
    n = len(ytr)
    n_val = max(4, int(0.2 * n))
    Xa, ya, Xv, yv = Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]
    best, best_err = None, np.inf
    for ls in LS_GRID:
        for lam in LAM_GRID:
            try:
                m = cls(length_scale=ls, lambda_reg=lam).fit(Xa, ya)
                err = float(np.mean((m.predict(Xv) - yv) ** 2))
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(err) and err < best_err:
                best, best_err = (ls, lam), err
    if best is None:
        raise RuntimeError('kernel tuning failed')
    return best


def exp_real_kernels(dataset, output_dir, max_samples=1000, seed=42):
    print(f"\n=== kernels on {dataset} ===")
    ensure_reproducibility(seed)
    X, y = LOADERS[dataset](max_samples)
    Xtr, ytr, Xte, yte = _split(dataset, X, y, seed)

    # Compute the two distinct algorithms; the PI- variants coincide with them
    # when no physics is available.
    fitted = {}
    for key, cls in (('KRR', KRRReal), ('IKRR', IterativeKRRReal)):
        ls, lam = _tune(cls, Xtr, ytr)
        t0 = time.time()
        m = cls(length_scale=ls, lambda_reg=lam).fit(Xtr, ytr)
        elapsed = time.time() - t0
        fitted[key] = (m, ls, lam, elapsed)

    rows = []
    for name, key in (('KRR', 'KRR'), ('IKRR', 'IKRR'),
                      ('PI-KRR', 'KRR'), ('PI-IKRR', 'IKRR')):
        m, ls, lam, elapsed = fitted[key]
        pred = m.predict(Xte)
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        r2 = float(r2_score(yte, pred))
        rows.append({'Method': name, 'Dataset': dataset, 'RMSE': rmse,
                     'R2': r2, 'Time_sec': elapsed,
                     'length_scale': ls, 'lambda_reg': lam})
        print(f"  {name:9s} (={key:5s}) ls={ls:<4} lam={lam:<7} "
              f"RMSE={rmse:.4f}  R2={r2:.4f}  ({elapsed:.3f}s)")
    _save_table(rows, os.path.join(output_dir, f'expR2_kernels_{dataset}.csv'))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    for ds in ['household', 'nslkdd', 'gaze', 'no2', 'ise']:
        exp_real_kernels(ds, out, max_samples=1000, seed=42)
    print('\nDone.')
