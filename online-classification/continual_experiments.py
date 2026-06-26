#!/usr/bin/env python3
"""
continual_experiments.py — Continual-learning evaluation of online algorithms.

Implements the three standard incremental-learning scenarios from
``book/vision/experiment.ipynb`` for every online algorithm in ``algos.py``:

  * Task-IL   — task identity known at test time; isolated per-task weights.
  * Domain-IL — fixed label space, distribution shifts across chunks.
  * Class-IL  — label space grows; single shared predictor, no task identity.

Multi-class problems (MNIST 0-9) use a one-vs-rest weight matrix, one binary
online classifier per class.  Binary problems run Domain-IL over temporal or
random chunks; Task-IL and Class-IL are not applicable and are reported as N/A.

The online updates are carried out by the same Numba JIT kernels used in
``algos.py``, ensuring that the continual-learning results are directly
comparable to the existing online benchmark.

Usage
-----
    from continual_experiments import run_continual_experiments
    run_continual_experiments(data_dir='data', output_dir='results')

Or via ``main.py --experiment continual``.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Re-use the compiled Numba kernels from algos.py
from algos import (
    _pa_kernel, _percept_kernel, _ogc_kernel,
    _arow_diag_kernel, _rda_kernel, _scw_diag_kernel, _adarda_kernel,
    warmup_jit,
)


# ═══════════════════════════════════════════════════════════════════════
#   CONSTANTS / CONFIG                                                   
# ═══════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 256, 789, 1024]

# MNIST digit pairs — identical to the notebook's PAIRS constant.
MNIST_PAIRS = [(3, 5), (4, 9), (1, 8), (2, 7), (0, 6)]

# Number of domain chunks for Domain-IL on binary datasets.
N_DOMAIN_CHUNKS = 5

# Default hyper-parameters for the online algorithms (mirrors algos.py).
ALGO_PARAMS = {
    'PA': {},
    'Perceptron': {},
    'OGC': {},
    'AROW': {'r': 0.1},
    # Use a mild sparsity threshold: lambda=1 zeros all weights on scaled data
    # in a single pass, which is catastrophic for the continual scenarios.
    'RDA': {'lambda_param': 0.0001, 'gamma_param': 1.0},
    'SCW': {'C': 1.0, 'eta': 0.9},
    'AdaRDA': {'lambda_param': 0.0001, 'eta_param': 1.0, 'delta_param': 1.0},
}

ALGO_NAMES = list(ALGO_PARAMS.keys())


def ensure_reproducibility(seed: int = 42) -> None:
    """Pin NumPy's random seed for a single run."""
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════
#   BINARY ONLINE STATE — wraps the Numba kernels                        
# ═══════════════════════════════════════════════════════════════════════

class BinaryOnlineState:
    """State and in-order chunk updates for one binary online classifier.

    Each call to ``fit_chunk`` makes predictions *before* updating on the
    chunk, exactly as the standalone Numba kernels do for a single pass.
    """

    def __init__(self, algo_name: str, n_features: int, params: Optional[Dict] = None):
        if algo_name not in ALGO_NAMES:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        self.algo_name = algo_name
        self.params = dict(params or {})
        self.w = np.zeros(n_features, dtype=np.float64)
        self.t = 0  # samples processed so far (needed by RDA/AdaRDA)

        if algo_name == 'AROW':
            self.sigma_diag = np.ones(n_features, dtype=np.float64)
        elif algo_name == 'SCW':
            self.sigma_diag = np.ones(n_features, dtype=np.float64)
            eta = self.params.get('eta', 0.9)
            self.phi = float(norm.ppf(eta))
        elif algo_name in ('RDA', 'AdaRDA'):
            self.g = np.zeros(n_features, dtype=np.float64)
            if algo_name == 'AdaRDA':
                self.g1t = np.zeros(n_features, dtype=np.float64)

    def fit_chunk(self, X: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
        """Update on a chunk and return pre-update binary predictions.

        Parameters
        ----------
        X : ndarray (n, d)
        y_binary : ndarray (n,) in {-1, +1}

        Returns
        -------
        y_pred : ndarray (n,) — predictions made before the chunk update
        """
        X_np = np.ascontiguousarray(X, dtype=np.float64)
        y_np = np.ascontiguousarray(y_binary, dtype=np.float64)
        n = X_np.shape[0]
        perm = np.arange(n, dtype=np.int64)

        if self.algo_name == 'PA':
            y_pred, self.w = _pa_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'Perceptron':
            y_pred, self.w = _percept_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'OGC':
            y_pred, self.w = _ogc_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'AROW':
            r = self.params.get('r', 0.1)
            y_pred, self.w, self.sigma_diag = _arow_diag_kernel(
                X_np, y_np, perm, self.w, self.sigma_diag, r)
        elif self.algo_name == 'RDA':
            lam = self.params.get('lambda_param', 1.0)
            gamma = self.params.get('gamma_param', 1.0)
            # Keep the global time-step continuous across chunks.
            y_pred, self.w, self.g = _rda_kernel(
                X_np, y_np, perm, self.w, self.g, lam, gamma, self.t, 1)
            self.t += n
        elif self.algo_name == 'SCW':
            C = self.params.get('C', 1.0)
            y_pred, self.w, self.sigma_diag = _scw_diag_kernel(
                X_np, y_np, perm, self.w, self.sigma_diag, C, self.phi)
        elif self.algo_name == 'AdaRDA':
            lam = self.params.get('lambda_param', 1.0)
            eta = self.params.get('eta_param', 1.0)
            delta = self.params.get('delta_param', 1.0)
            y_pred, self.w, self.g, self.g1t = _adarda_kernel(
                X_np, y_np, perm, self.w, self.g, self.g1t,
                lam, eta, delta, self.t, 1)
            self.t += n

        return y_pred

    def fit_one(self, x: np.ndarray, y_binary: float) -> float:
        """Update on a single sample and return its pre-update prediction."""
        X_np = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float64)
        y_np = np.ascontiguousarray(np.array([y_binary]), dtype=np.float64)
        perm = np.array([0], dtype=np.int64)

        if self.algo_name == 'PA':
            y_pred, self.w = _pa_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'Perceptron':
            y_pred, self.w = _percept_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'OGC':
            y_pred, self.w = _ogc_kernel(X_np, y_np, perm, self.w)
        elif self.algo_name == 'AROW':
            r = self.params.get('r', 0.1)
            y_pred, self.w, self.sigma_diag = _arow_diag_kernel(
                X_np, y_np, perm, self.w, self.sigma_diag, r)
        elif self.algo_name == 'RDA':
            lam = self.params.get('lambda_param', 1.0)
            gamma = self.params.get('gamma_param', 1.0)
            y_pred, self.w, self.g = _rda_kernel(
                X_np, y_np, perm, self.w, self.g, lam, gamma, self.t, 1)
            self.t += 1
        elif self.algo_name == 'SCW':
            C = self.params.get('C', 1.0)
            y_pred, self.w, self.sigma_diag = _scw_diag_kernel(
                X_np, y_np, perm, self.w, self.sigma_diag, C, self.phi)
        elif self.algo_name == 'AdaRDA':
            lam = self.params.get('lambda_param', 1.0)
            eta = self.params.get('eta_param', 1.0)
            delta = self.params.get('delta_param', 1.0)
            y_pred, self.w, self.g, self.g1t = _adarda_kernel(
                X_np, y_np, perm, self.w, self.g, self.g1t,
                lam, eta, delta, self.t, 1)
            self.t += 1

        return float(y_pred[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions {-1, +1} using the current weights."""
        scores = X @ self.w
        return np.where(scores > 0, 1.0, np.where(scores < 0, -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
#   ONE-VS-REST MULTI-CLASS STATE                                        
# ═══════════════════════════════════════════════════════════════════════

class OvROnlineState:
    """One-vs-rest multi-class wrapper over BinaryOnlineState.

    Maintains one binary classifier per class.  Each chunk is used to
    update the active classifiers; multi-class predictions are by argmax
    of pre-update scores over the active set.
    """

    def __init__(self, algo_name: str, n_classes: int, n_features: int,
                 params: Optional[Dict] = None):
        self.n_classes = n_classes
        self.states = [
            BinaryOnlineState(algo_name, n_features, params)
            for _ in range(n_classes)
        ]

    def fit_chunk(self, X: np.ndarray, y: np.ndarray,
                  active_classes: Optional[List[int]] = None) -> np.ndarray:
        """Update on a chunk and return multi-class predictions.

        Predictions are made before any weight update, restricted to
        ``active_classes`` when provided.
        """
        if active_classes is None:
            active_classes = list(range(self.n_classes))

        # Pre-update scores for active classes
        W_active = np.stack([self.states[c].w for c in active_classes], axis=0)
        scores = X @ W_active.T  # (n, |active|)
        preds = np.array(active_classes)[np.argmax(scores, axis=1)]

        # Update each active binary classifier
        for c in active_classes:
            y_bin = np.where(y == c, 1.0, -1.0)
            self.states[c].fit_chunk(X, y_bin)

        return preds

    def predict(self, X: np.ndarray,
                active_classes: Optional[List[int]] = None) -> np.ndarray:
        if active_classes is None:
            active_classes = list(range(self.n_classes))
        W_active = np.stack([self.states[c].w for c in active_classes], axis=0)
        scores = X @ W_active.T
        return np.array(active_classes)[np.argmax(scores, axis=1)]


# ═══════════════════════════════════════════════════════════════════════
#   EVALUATION HELPERS                                                   
# ═══════════════════════════════════════════════════════════════════════

def pm_str(vals, fmt: str = ".4f") -> str:
    """Format list of numbers as ``mean ± std``."""
    a = np.array(vals)
    return f"{a.mean():{fmt}} ± {a.std():{fmt}}"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 for integer labels."""
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


# ═══════════════════════════════════════════════════════════════════════
#   CONTINUAL SCENARIO RUNNERS                                          
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#   NOTEBOOK-STYLE PER-SAMPLE UPDATE RULES
#   Same predict-before-update loop as experiment.ipynb's run_til_ogc /
#   run_dil_ogc / run_cil_ogc, extended to all 7 algorithms.
# ═══════════════════════════════════════════════════════════════════════

def _init_state(algo_name: str, d: int, params: Dict) -> Dict:
    """Initialise algorithm state for one binary weight vector."""
    s: Dict = {'w': np.zeros(d, dtype=np.float64)}
    if algo_name == 'AROW':
        s['sigma'] = np.ones(d, dtype=np.float64)
    elif algo_name == 'SCW':
        s['sigma'] = np.ones(d, dtype=np.float64)
        s['phi'] = float(norm.ppf(params.get('eta', 0.9)))
    elif algo_name == 'RDA':
        s['g'] = np.zeros(d, dtype=np.float64)
        s['t'] = 1
    elif algo_name == 'AdaRDA':
        s['g'] = np.zeros(d, dtype=np.float64)
        s['g1t'] = np.zeros(d, dtype=np.float64)
        s['t'] = 1
    return s


def _step_binary(algo_name: str, s: Dict, x: np.ndarray,
                 y: float, params: Dict) -> float:
    """One predict-before-update step for a binary classifier.

    Mirrors the per-sample loop in the notebook's run_til_ogc.
    Returns the pre-update discrete prediction (+1 or -1).
    Mutates s in-place.
    """
    eps = 1e-12
    w = s['w']
    score = float(w @ x)
    pred = 1.0 if score >= 0.0 else -1.0   # predict BEFORE update

    if algo_name == 'OGC':
        # Normalised gradient step toward target y (notebook exact match)
        s['w'] = w + ((y - score) / (float(x @ x) + eps)) * x

    elif algo_name == 'PA':
        loss = max(0.0, 1.0 - y * pred)
        if loss > 0.0:
            l2sq = float(x @ x)
            if l2sq > 0.0:
                s['w'] = w + (loss / (l2sq * l2sq)) * y * x

    elif algo_name == 'Perceptron':
        if pred != y:
            s['w'] = w + y * x

    elif algo_name == 'AROW':
        margin = y * score
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0:
            sigma = s['sigma']
            r = params.get('r', 0.1)
            vt = float(x @ (sigma * x))
            denom = vt + r
            alpha = loss / denom
            beta = 1.0 / denom
            s['w'] = w + alpha * y * sigma * x
            s['sigma'] = np.maximum(sigma - beta * (sigma * x) ** 2, 1e-10)

    elif algo_name == 'SCW':
        sigma = s['sigma']
        phi = s['phi']
        vt = float(x @ (sigma * x))
        mt = y * score
        loss = phi * np.sqrt(max(vt, 0.0)) - mt
        if loss > 0.0:
            C = params.get('C', 1.0)
            phi_sq = phi * phi
            xi = 1.0 + phi_sq
            pa = 1.0 + phi_sq / 2.0
            st = max(0.0, mt * mt * phi_sq * phi_sq / 4.0 + vt * phi_sq * xi)
            alpha = float(np.clip((-mt * pa + np.sqrt(st)) / (vt * xi) if vt * xi > 0 else 0.0, 0.0, C))
            su = max(0.0, (alpha * vt * phi) ** 2 + 4.0 * vt)
            ut_val = ((-alpha * vt * phi + np.sqrt(su)) / 2.0) ** 2
            beta = (alpha * phi) / (np.sqrt(ut_val) + vt * alpha * phi + 1e-8)
            s['w'] = w + alpha * y * sigma * x
            s['sigma'] = np.maximum(sigma - beta * (sigma * x) ** 2, 1e-10)

    elif algo_name == 'RDA':
        lam = params.get('lambda_param', 0.0001)
        gamma = params.get('gamma_param', 1.0)
        t = s['t']
        g = s['g']
        loss = max(0.0, 1.0 - y * pred)
        s['g'] = ((t - 1) / t) * g + ((1.0 / t) * (-y * x) if loss > 0.0 else 0.0)
        g_new = s['g']
        mask = np.abs(g_new) > lam
        s['w'] = np.where(mask, -(np.sqrt(float(t)) / gamma) * (g_new - lam * np.sign(g_new)), 0.0)
        s['t'] = t + 1

    elif algo_name == 'AdaRDA':
        lam = params.get('lambda_param', 0.0001)
        eta_p = params.get('eta_param', 1.0)
        delta = params.get('delta_param', 1.0)
        t = s['t']
        g = s['g']
        g1t = s['g1t']
        loss = max(0.0, 1.0 - y * score)
        if loss > 0.0:
            gt = -y * x
            s['g'] = ((t - 1) / t) * g + (1.0 / t) * gt
            s['g1t'] = g1t + gt * gt
        else:
            s['g'] = ((t - 1) / t) * g
        g_new = s['g']
        Ht = delta + np.sqrt(s['g1t'])
        mask = np.abs(g_new) > lam
        s['w'] = np.where(mask, np.sign(-g_new) * eta_p * t / (Ht + 1e-8), 0.0)
        s['t'] = t + 1

    return pred


def _init_state_mc(algo_name: str, K: int, d: int, params: Dict) -> Dict:
    """Initialise algorithm state for a shared K×d weight matrix."""
    s: Dict = {'W': np.zeros((K, d), dtype=np.float64)}
    if algo_name in ('AROW', 'SCW'):
        s['sigma'] = np.ones((K, d), dtype=np.float64)   # per-row diagonal confidence
    if algo_name == 'SCW':
        s['phi'] = float(norm.ppf(params.get('eta', 0.9)))
    if algo_name == 'RDA':
        s['g'] = np.zeros((K, d), dtype=np.float64)
        s['t'] = 1
    if algo_name == 'AdaRDA':
        s['g'] = np.zeros((K, d), dtype=np.float64)
        s['g1t'] = np.zeros((K, d), dtype=np.float64)
        s['t'] = 1
    return s


def _step_mc(algo_name: str, s: Dict, x: np.ndarray,
             y_true: int, active: List[int], params: Dict) -> int:
    """One predict-before-update step for multi-class (shared W matrix).

    Mirrors run_dil_ogc / run_cil_ogc from the notebook, extended to all
    algorithms.  Prediction is restricted to 'active' rows; updates touch
    only the predicted (wrong) row and the correct row.

    Returns the pre-update predicted class index.
    """
    eps = 1e-12
    W = s['W']
    active_arr = np.array(active, dtype=np.int64)
    scores = W[active_arr] @ x
    yhat = int(active_arr[np.argmax(scores)])   # predict BEFORE update

    if algo_name == 'OGC':
        # Notebook exact match: push wrong row to -1, pull correct row to +1
        denom = float(x @ x) + eps
        W[yhat] += (-1.0 - float(W[yhat] @ x)) / denom * x
        W[y_true] += (1.0 - float(W[y_true] @ x)) / denom * x

    elif algo_name == 'Perceptron':
        if yhat != y_true:
            W[y_true] += x
            W[yhat] -= x

    elif algo_name == 'PA':
        margin = float(W[y_true] @ x) - float(W[yhat] @ x)
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0:
            l2sq = float(x @ x)
            l2sq_sq = l2sq * l2sq
            if l2sq_sq > 0.0:
                tau = loss / l2sq_sq
                W[y_true] += tau * x
                if yhat != y_true:
                    W[yhat] -= tau * x

    elif algo_name == 'AROW':
        r = params.get('r', 0.1)
        sigma = s['sigma']
        margin = float(W[y_true] @ x) - float(W[yhat] @ x)
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0 and yhat != y_true:
            vt = float(x @ (sigma[y_true] * x)) + float(x @ (sigma[yhat] * x))
            denom = vt + r
            alpha = loss / denom
            beta = 1.0 / denom
            W[y_true] += alpha * sigma[y_true] * x
            W[yhat] -= alpha * sigma[yhat] * x
            s['sigma'][y_true] = np.maximum(sigma[y_true] - beta * (sigma[y_true] * x) ** 2, 1e-10)
            s['sigma'][yhat] = np.maximum(sigma[yhat] - beta * (sigma[yhat] * x) ** 2, 1e-10)

    elif algo_name == 'SCW':
        sigma = s['sigma']
        phi = s['phi']
        vt = float(x @ (sigma[y_true] * x)) + float(x @ (sigma[yhat] * x))
        mt = float(W[y_true] @ x) - float(W[yhat] @ x)
        loss = phi * np.sqrt(max(vt, 0.0)) - mt
        if loss > 0.0 and yhat != y_true:
            C = params.get('C', 1.0)
            phi_sq = phi * phi
            xi = 1.0 + phi_sq
            pa = 1.0 + phi_sq / 2.0
            st = max(0.0, mt * mt * phi_sq * phi_sq / 4.0 + vt * phi_sq * xi)
            alpha = float(np.clip((-mt * pa + np.sqrt(st)) / (vt * xi) if vt * xi > 0 else 0.0, 0.0, C))
            su = max(0.0, (alpha * vt * phi) ** 2 + 4.0 * vt)
            ut_val = ((-alpha * vt * phi + np.sqrt(su)) / 2.0) ** 2
            beta = (alpha * phi) / (np.sqrt(ut_val) + vt * alpha * phi + 1e-8)
            W[y_true] += alpha * sigma[y_true] * x
            W[yhat] -= alpha * sigma[yhat] * x
            s['sigma'][y_true] = np.maximum(sigma[y_true] - beta * (sigma[y_true] * x) ** 2, 1e-10)
            s['sigma'][yhat] = np.maximum(sigma[yhat] - beta * (sigma[yhat] * x) ** 2, 1e-10)

    elif algo_name == 'RDA':
        lam = params.get('lambda_param', 0.0001)
        gamma = params.get('gamma_param', 1.0)
        t = s['t']
        g = s['g']
        margin = float(W[y_true] @ x) - float(W[yhat] @ x)
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0 and yhat != y_true:
            s['g'][y_true] = ((t - 1) / t) * g[y_true] + (1.0 / t) * (-x)
            s['g'][yhat] = ((t - 1) / t) * g[yhat] + (1.0 / t) * x
        else:
            s['g'][y_true] = ((t - 1) / t) * g[y_true]
            if yhat != y_true:
                s['g'][yhat] = ((t - 1) / t) * g[yhat]
        sqrt_t = np.sqrt(float(t))
        for c in set([y_true, yhat]):
            gc = s['g'][c]
            mask = np.abs(gc) > lam
            W[c] = np.where(mask, -(sqrt_t / gamma) * (gc - lam * np.sign(gc)), 0.0)
        s['t'] = t + 1

    elif algo_name == 'AdaRDA':
        lam = params.get('lambda_param', 0.0001)
        eta_p = params.get('eta_param', 1.0)
        delta = params.get('delta_param', 1.0)
        t = s['t']
        g = s['g']
        g1t = s['g1t']
        margin = float(W[y_true] @ x) - float(W[yhat] @ x)
        loss = max(0.0, 1.0 - margin)
        if loss > 0.0 and yhat != y_true:
            gt_y = -x;  gt_yh = x
            s['g'][y_true] = ((t - 1) / t) * g[y_true] + (1.0 / t) * gt_y
            s['g'][yhat] = ((t - 1) / t) * g[yhat] + (1.0 / t) * gt_yh
            s['g1t'][y_true] += gt_y * gt_y
            s['g1t'][yhat] += gt_yh * gt_yh
        else:
            s['g'][y_true] = ((t - 1) / t) * g[y_true]
            if yhat != y_true:
                s['g'][yhat] = ((t - 1) / t) * g[yhat]
        for c in set([y_true, yhat]):
            Ht = delta + np.sqrt(s['g1t'][c])
            gc = s['g'][c]
            mask = np.abs(gc) > lam
            W[c] = np.where(mask, np.sign(-gc) * eta_p * t / (Ht + 1e-8), 0.0)
        s['t'] = t + 1

    return yhat


# ═══════════════════════════════════════════════════════════════════════
#   SCENARIO RUNNERS  (notebook-style, all algorithms)
# ═══════════════════════════════════════════════════════════════════════

def run_task_il_mnist(X: np.ndarray, y: np.ndarray, algo_name: str,
                      params: Dict, seed: int = 0,
                      task_groups: Optional[List] = None) -> Dict:
    """Task-IL: isolated weight matrix per task group (notebook style, all datasets).

    task_groups: list of class-index lists/tuples.  Defaults to MNIST_PAIRS.
    Each task gets a fresh _init_state_mc; active classes restricted to the task.
    """
    K = int(y.max()) + 1
    d = X.shape[1]
    groups = task_groups if task_groups is not None else MNIST_PAIRS
    all_preds: List[int] = [];  all_true: List[int] = []
    tot_mk = 0;  n_tot = 0

    for task_classes in groups:
        task_mask = np.isin(y, list(task_classes))
        Xj, yj = X[task_mask], y[task_mask]
        if len(Xj) == 0:
            continue
        s = _init_state_mc(algo_name, K, d, params)   # isolated per-task weights
        for x, yi in zip(Xj, yj):
            yhat = _step_mc(algo_name, s, x, int(yi), list(task_classes), params)
            all_preds.append(yhat);  all_true.append(int(yi))
            if yhat != int(yi):
                tot_mk += 1
            n_tot += 1

    if n_tot == 0:
        return {'M_T': 0.0, 'MacroF1': 0.0}
    m_t = tot_mk / n_tot
    f1 = macro_f1(np.array(all_true), np.array(all_preds))
    return {'M_T': m_t, 'MacroF1': f1}


def run_domain_il_mnist(X: np.ndarray, y: np.ndarray, algo_name: str,
                        params: Dict, seed: int = 0,
                        n_chunks: int = N_DOMAIN_CHUNKS) -> Dict:
    """Domain-IL: shared K×d weight matrix, sequential stream chunks (notebook style)."""
    K = int(y.max()) + 1
    d = X.shape[1]
    chunks = np.array_split(np.arange(len(X)), n_chunks)
    s = _init_state_mc(algo_name, K, d, params)
    active = list(range(K))
    stream_preds: List[int] = [];  stream_true: List[int] = []
    fwd_f1s = []

    for j in range(n_chunks):
        Xc, yc = X[chunks[j]], y[chunks[j]]
        for x, yi in zip(Xc, yc):
            yhat = _step_mc(algo_name, s, x, int(yi), active, params)
            stream_preds.append(yhat);  stream_true.append(int(yi))
        # Forward-transfer probe — read-only, no weight update
        W = s['W']
        fwd_scores = [
            macro_f1(y[chunks[k]], np.argmax(X[chunks[k]] @ W.T, axis=1))
            for k in range(j + 1, n_chunks)
        ]
        fwd_f1s.append(float(np.mean(fwd_scores)) if fwd_scores else np.nan)

    m_t = float(np.mean(np.array(stream_preds) != np.array(stream_true)))
    return {'M_T': m_t, 'FwdMacroF1': float(np.nanmean(fwd_f1s)) if fwd_f1s else np.nan}


def run_class_il_mnist(X: np.ndarray, y: np.ndarray, algo_name: str,
                       params: Dict, seed: int = 0) -> Dict:
    """Class-IL: shared Kxd weight matrix, predict over ALL active classes (notebook style).

    Pairs arrive sequentially.  After each pair is introduced, prediction
    uses argmax over every class seen so far (no task identity).  Update
    touches the true-class row and the wrongly-predicted row only.
    """
    K = int(y.max()) + 1
    d = X.shape[1]
    s = _init_state_mc(algo_name, K, d, params)   # W pre-allocated, all zero
    active: List[int] = []
    stream_preds: List[int] = [];  stream_true: List[int] = []
    tot_mk = 0;  n_tot = 0

    for a, b in MNIST_PAIRS:
        for c in [a, b]:
            if c not in active:
                active.append(c)
        mask = (y == a) | (y == b)
        Xj, yj = X[mask], y[mask]
        for x, yi in zip(Xj, yj):
            # Predict over ALL active classes (no task identity — Class-IL)
            yhat = _step_mc(algo_name, s, x, int(yi), active, params)
            stream_preds.append(yhat);  stream_true.append(int(yi))
            if yhat != int(yi):
                tot_mk += 1
            n_tot += 1

    m_t = tot_mk / n_tot if n_tot > 0 else 0.0
    f1 = macro_f1(np.array(stream_true), np.array(stream_preds))
    return {'M_T': m_t, 'MacroF1': f1}


def run_domain_il_binary(X: np.ndarray, y: np.ndarray, algo_name: str,
                         params: Dict, seed: int = 0,
                         n_chunks: int = N_DOMAIN_CHUNKS) -> Dict:
    """Domain-IL on a binary dataset with labels in {-1, +1}.

    The stream is split into ``n_chunks`` sequential chunks (temporal order
    preserved).  A single binary classifier is updated across all chunks.
    Predictions are made *before* the weight update (online protocol).

    Metrics returned
    ----------------
    M_T      : cumulative mistake rate over the full stream.
    FwdF1    : mean forward-transfer F1 — after seeing chunk j, evaluate
               on all future chunks j+1 … n_chunks-1 with current weights.
    """
    n = len(X)
    chunk_indices = np.array_split(np.arange(n), n_chunks)

    s = _init_state(algo_name, X.shape[1], params)
    stream_preds: List[float] = []
    stream_true: List[float] = []
    fwd_f1s: List[float] = []

    for j, idx in enumerate(chunk_indices):
        Xc, yc = X[idx], y[idx]
        chunk_preds: List[float] = []

        for x_i, y_i in zip(Xc, yc):
            pred = _step_binary(algo_name, s, x_i, float(y_i), params)
            chunk_preds.append(pred)
            stream_preds.append(pred)
            stream_true.append(float(y_i))

        # Forward-transfer: evaluate current weights (no update) on future chunks
        fwd_scores: List[float] = []
        w_cur = s['w']
        for k in range(j + 1, n_chunks):
            fwd_idx = chunk_indices[k]
            fwd_raw = np.sign(X[fwd_idx] @ w_cur)
            fwd_raw = np.where(fwd_raw == 0, 1.0, fwd_raw)
            # Convert {-1,+1} back to {0,1} for f1_score with pos_label=1
            y_true_01 = np.where(y[fwd_idx] == 1, 1, 0)
            y_pred_01 = np.where(fwd_raw == 1, 1, 0)
            fwd_scores.append(float(f1_score(y_true_01, y_pred_01,
                                             average='binary', pos_label=1,
                                             zero_division=0)))
        fwd_f1s.append(float(np.mean(fwd_scores)) if fwd_scores else np.nan)

    y_t = np.array(stream_true)
    y_p = np.array(stream_preds)
    m_t = float(np.mean(y_t != y_p))
    return {'M_T': m_t,
            'FwdF1': float(np.nanmean(fwd_f1s)) if fwd_f1s else np.nan}


# ═══════════════════════════════════════════════════════════════════════
#   DATA LOADER  (notebook-identical)
# ═══════════════════════════════════════════════════════════════════════

def load_mnist_openml() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST via OpenML exactly as in experiment.ipynb.

    Returns L2-normalised X (70000 × 784, float64) and integer labels y.
    """
    dg = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = dg.data.astype(np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    y = dg.target.astype(int)
    return X, y


# Non-feature columns that must be dropped before scaling.
_DROP_COLS = {'entity', 'user_id', 'timestamp', 'Time', 'label', 'Class'}


def _load_single_binary_parquet(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one binary parquet file and return (X_scaled, y) with y in {-1, +1}.

    Drops known metadata columns, imputes NaN with column median, then
    standardises features.  Returns None on failure.
    """
    try:
        df = pd.read_parquet(path).dropna(how='all')
    except Exception as e:
        print(f"  WARNING: could not read {path}: {e}")
        return None

    # Determine label column
    if 'label' in df.columns:
        label_col = 'label'
    elif 'Class' in df.columns:
        label_col = 'Class'
    else:
        print(f"  WARNING: no label column in {os.path.basename(path)}")
        return None

    y_raw = df[label_col].values
    feat_cols = [c for c in df.columns if c not in _DROP_COLS
                 and np.issubdtype(df[c].dtype, np.number)]
    if not feat_cols:
        print(f"  WARNING: no numeric feature columns in {os.path.basename(path)}")
        return None

    X_df = df[feat_cols].copy()
    # Impute remaining NaN with column median
    for col in X_df.columns[X_df.isna().any()]:
        X_df[col].fillna(X_df[col].median(), inplace=True)

    X = X_df.values.astype(np.float64)
    y = np.where(y_raw == 0, -1.0, 1.0)  # {0,1} → {-1,+1}

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def load_binary_online_datasets(data_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load binary {0,1}-labelled datasets matching the online experiment scope.

    Mirrors the dataset selection in online_experiments.run_online_benchmark():
      1. Cybersecurity streams  — single parquet per dataset (Crypto, NonEnc,
         OutFlash, OutTLS, P2P, Phishing — desktop and smartphone variants).
      2. MITBIH_Arrhythmia      — train + test concatenated as one stream.

    Excluded (belong to batch_experiments): UNSW_NB15, CreditFraud_kaggle.
    Excluded (multi-class): MNIST, CIFAR100.

    Returns a dict  { dataset_name: (X, y) }  with X standardised and
    y in {-1, +1}.
    """
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── 1. Pair-parquet datasets (combine train + test sequentially) ──
    pair_datasets = [
        ('MITBIH_Arrhythmia', 'MITBIH_Arrhythmia_train.parquet', 'MITBIH_Arrhythmia_test.parquet'),
    ]
    for ds_name, train_file, test_file in pair_datasets:
        train_path = os.path.join(data_dir, train_file)
        test_path  = os.path.join(data_dir, test_file)
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"  Skipping {ds_name}: parquet pair not found.")
            continue
        try:
            df_tr = pd.read_parquet(train_path).dropna(how='all')
            df_te = pd.read_parquet(test_path).dropna(how='all')
            df    = pd.concat([df_tr, df_te], ignore_index=True)
        except Exception as e:
            print(f"  WARNING: could not load {ds_name}: {e}")
            continue

        label_col = 'label' if 'label' in df.columns else 'Class'
        feat_cols = [c for c in df.columns if c not in _DROP_COLS
                     and np.issubdtype(df[c].dtype, np.number)]
        if not feat_cols:
            continue
        X_df = df[feat_cols].copy()
        for col in X_df.columns[X_df.isna().any()]:
            X_df[col].fillna(X_df[col].median(), inplace=True)
        X = StandardScaler().fit_transform(X_df.values.astype(np.float64))
        y = np.where(df[label_col].values == 0, -1.0, 1.0)
        datasets[ds_name] = (X, y)
        print(f"  Loaded {ds_name}: {X.shape}, pos={np.mean(y==1):.3f}")

    # ── 2. Single-file cybersecurity streams (exclude batch/multi-class files) ──
    _SKIP_PREFIXES = {'MNIST', 'CIFAR', 'UNSW', 'MITBIH', 'CREDIT'}
    for fpath in sorted(glob.glob(os.path.join(data_dir, '*.parquet'))):
        fname = os.path.basename(fpath)
        if any(fname.upper().startswith(k) for k in _SKIP_PREFIXES):
            continue
        ds_name = fname.replace('.parquet', '')
        result = _load_single_binary_parquet(fpath)
        if result is None:
            continue
        X, y = result
        datasets[ds_name] = (X, y)
        print(f"  Loaded {ds_name}: {X.shape}, pos={np.mean(y==1):.3f}")

    return datasets


# ═══════════════════════════════════════════════════════════════════════
#   MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_continual_experiments(data_dir: str = '', output_dir: Optional[str] = None,
                              seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """Run continual-learning experiments on MNIST and all binary online datasets.

    MNIST (Task-IL, Domain-IL, Class-IL) is fetched via fetch_openml, exactly
    as in experiment.ipynb.

    All binary {0,1}-labelled parquet datasets in ``data_dir`` are run under
    Domain-IL only (the fixed 2-class label space makes Task-IL and Class-IL
    ill-defined for single-dataset streams).

    Returns a long-form DataFrame with columns
    Dataset, Algorithm, Scenario, Metric, Value.
    """
    if seeds is None:
        seeds = SEEDS

    warmup_jit()

    all_records = []

    # ── MNIST: Task-IL, Domain-IL, Class-IL ──────────────────────────
    print("Loading MNIST via OpenML ...")
    X_mnist, y_mnist = load_mnist_openml()
    ds_name = 'MNIST'
    print(f"  shape: {X_mnist.shape}, classes: {np.unique(y_mnist)}")

    for algo_name in ALGO_NAMES:
        params = ALGO_PARAMS[algo_name]
        print(f"\n  Algorithm: {algo_name}")

        task_f1, task_mt = [], []
        for seed in seeds:
            res = run_task_il_mnist(X_mnist, y_mnist, algo_name, params, seed)
            task_f1.append(res['MacroF1'])
            task_mt.append(res['M_T'])
        print(f"    Task-IL   Macro-F1: {pm_str(task_f1)}  M_T: {pm_str(task_mt)}")
        for v in task_f1:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Task-IL', 'Metric': 'MacroF1', 'Value': v})
        for v in task_mt:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Task-IL', 'Metric': 'M_T', 'Value': v})

        dom_f1, dom_mt = [], []
        for seed in seeds:
            res = run_domain_il_mnist(X_mnist, y_mnist, algo_name, params, seed)
            dom_f1.append(res['FwdMacroF1'])
            dom_mt.append(res['M_T'])
        print(f"    Domain-IL FwdMacroF1: {pm_str(dom_f1)}  M_T: {pm_str(dom_mt)}")
        for v in dom_f1:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Domain-IL', 'Metric': 'FwdMacroF1', 'Value': v})
        for v in dom_mt:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Domain-IL', 'Metric': 'M_T', 'Value': v})

        class_f1, class_mt = [], []
        for seed in seeds:
            res = run_class_il_mnist(X_mnist, y_mnist, algo_name, params, seed)
            class_f1.append(res['MacroF1'])
            class_mt.append(res['M_T'])
        print(f"    Class-IL  Macro-F1: {pm_str(class_f1)}  M_T: {pm_str(class_mt)}")
        for v in class_f1:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Class-IL', 'Metric': 'MacroF1', 'Value': v})
        for v in class_mt:
            all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                'Scenario': 'Class-IL', 'Metric': 'M_T', 'Value': v})

    # ── Binary online datasets: Domain-IL only ───────────────────────
    if data_dir:
        print("\n\nLoading binary online datasets ...")
        binary_datasets = load_binary_online_datasets(data_dir)

        for ds_name, (X_bin, y_bin) in binary_datasets.items():
            print(f"\nDataset: {ds_name}  shape={X_bin.shape}")
            for algo_name in ALGO_NAMES:
                params = ALGO_PARAMS[algo_name]
                print(f"  Algorithm: {algo_name}")

                dom_fwd, dom_mt = [], []
                # Domain-IL is deterministic for a fixed sequential stream;
                # repeat over seeds for consistency with MNIST reporting.
                for seed in seeds:
                    ensure_reproducibility(seed)
                    res = run_domain_il_binary(X_bin, y_bin, algo_name, params, seed)
                    dom_fwd.append(res['FwdF1'])
                    dom_mt.append(res['M_T'])
                print(f"    Domain-IL FwdF1: {pm_str(dom_fwd)}  M_T: {pm_str(dom_mt)}")
                for v in dom_fwd:
                    all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                        'Scenario': 'Domain-IL', 'Metric': 'FwdF1', 'Value': v})
                for v in dom_mt:
                    all_records.append({'Dataset': ds_name, 'Algorithm': algo_name,
                                        'Scenario': 'Domain-IL', 'Metric': 'M_T', 'Value': v})

    df = pd.DataFrame(all_records)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        #path = os.path.join(output_dir, 'continual_learning_results.csv')
        #df.to_csv(path, index=False, float_format='%.4f')
        #print(f"\nResults saved to: {path}")
        _save_continual_summary(df, output_dir)
        # _plot_continual_results(df, output_dir)
    return df


def _save_continual_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Save a wide summary table: mean ± std per Dataset/Algorithm/Scenario."""
    summary = df.groupby(['Dataset', 'Algorithm', 'Scenario', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()
    summary['Score'] = summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    wide = summary.pivot_table(
        index=['Dataset', 'Metric'],
        columns=['Scenario', 'Algorithm'],
        values='Score',
        aggfunc='first',
    )
    path = os.path.join(output_dir, 'continual_learning_summary.csv')
    wide.to_csv(path)
    print(f"Summary saved to: {path}")


# def _plot_continual_results(df: pd.DataFrame, output_dir: str) -> None:
#     """Generate a grouped bar plot of mean continual-learning scores."""
#     try:
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#     except Exception as e:
#         print(f"Skipping plot: {e}")
#         return

#     summary = df.groupby(['Dataset', 'Algorithm', 'Scenario'])['Value'].mean().reset_index()
#     # Focus on the scenarios present in the data
#     scenarios = ['Task-IL', 'Domain-IL', 'Class-IL']
#     scenarios = [s for s in scenarios if s in summary['Scenario'].unique()]

#     algo_colors = {
#         'PA': '#d62728',
#         'Perceptron': '#1f77b4',
#         'OGC': '#2ca02c',
#         'AROW': '#ff7f0e',
#         'RDA': '#9467bd',
#         'SCW': '#8c564b',
#         'AdaRDA': '#e377c2',
#     }

#     n_sc = len(scenarios)
#     if n_sc == 0:
#         return

#     fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5), sharey=True)
#     if n_sc == 1:
#         axes = [axes]

#     for ax, scenario in zip(axes, scenarios):
#         sub = summary[summary['Scenario'] == scenario].copy()
#         if sub.empty:
#             ax.set_visible(False)
#             continue
#         datasets = sorted(sub['Dataset'].unique())
#         algos = sorted(sub['Algorithm'].unique())
#         x = np.arange(len(datasets))
#         bar_w = 0.11

#         for i, algo in enumerate(algos):
#             vals = []
#             for ds in datasets:
#                 v = sub[(sub['Dataset'] == ds) & (sub['Algorithm'] == algo)]['Value']
#                 vals.append(float(v.iloc[0]) if len(v) else 0.0)
#             offset = (i - (len(algos) - 1) / 2) * bar_w
#             ax.bar(x + offset, vals, bar_w, label=algo, color=algo_colors.get(algo, '#333333'))

#         ax.set_xticks(x)
#         ax.set_xticklabels([d.replace('_', '\n') for d in datasets], rotation=0, fontsize=8)
#         ax.set_ylabel('Score' if scenario == scenarios[0] else '')
#         ax.set_title(scenario, fontweight='bold')
#         ax.set_ylim(0, 1.05)
#         ax.grid(axis='y', alpha=0.3)

#     axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
#     plt.tight_layout()
#     path = os.path.join(output_dir, 'continual_learning_plot.png')
#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Plot saved to: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run continual-learning experiments.')
    parser.add_argument('--data-dir', default='data', help='Directory containing parquet datasets.')
    parser.add_argument('--results-dir', default='results', help='Directory to save results.')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    results_dir = os.path.abspath(args.results_dir)
    run_continual_experiments(data_dir, results_dir)




