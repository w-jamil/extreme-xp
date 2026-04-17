#!/usr/bin/env python3
"""
experiments.py — Experiment runners and data utilities.

Three experiment modes
---------------------
1. run_simulation()       — 11 synthetic datasets 3 imbalance ratios
                            compares Online vs Kernel Online algorithms.
2. run_online_benchmark() — Real datasets (MNIST, Kaggle, Cybersecurity)
                            runs 7 Numba-accelerated online algorithms.
3. run_kernel_benchmark() — UCI datasets (ClevelandHeart)
                            runs 7 batch-kernel algorithms with repeated shuffles.
"""

import numpy as np
import pandas as pd
import os
import time
import glob
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
)
from scipy.stats import norm

# All algorithms live in algos.py
from algos import (
    # Online (Numba)
    PA, Perceptron, OGC, AROW, RDA, SCW, AdaRDA,
    # Kernel Online
    KPA, KPerceptron, KOGC, KAROW, KRDA, KSCW, KAdaRDA,
    # Batch Kernel
    KernelPA, KernelPerceptron, KernelGC, KernelAROW,
    KernelRDA, KernelSCW, KernelAdaRDA,
    # Utilities
    calculate_class1_metrics, predict_val, warmup_jit,
    _predict_val, rbf_kernel_matrix,
    # Numba inner kernels (for warm-start online passes)
    _pa_kernel, _percept_kernel, _ogc_kernel,
    _arow_diag_kernel, _rda_kernel, _scw_diag_kernel, _adarda_kernel,
)


# ═══════════════════════════════════════════════════════════════════════
#   REPRODUCIBILITY                                                     
# ═══════════════════════════════════════════════════════════════════════

def ensure_reproducibility(seed=42):
    """Pin all random seeds for full reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 1 — SYNTHETIC DATA GENERATORS  (11 datasets)               
# ═══════════════════════════════════════════════════════════════════════

def generate_separable_data(n_samples=2000, n_features=5, noise=0.1, seed=42):
    """Separable: 5-d data with quadratic shock at midpoint.

    y = sign(w^T x + η),  w ~ N(0,1),  η ~ N(0, noise).
    After midpoint: x → x² (breaks algorithms expecting stable distribution).
    """
    np.random.seed(seed)
    # Random true weight vector (normalised)
    w_true = np.random.randn(n_features)
    w_true = w_true / np.linalg.norm(w_true)
    # Gaussian features
    X = np.random.randn(n_samples, n_features)
    # Quadratic shock at midpoint
    shock_point = min(1000, n_samples // 2)
    X[shock_point:] = X[shock_point:] ** 2
    # Labels from the true hyperplane + noise
    y_scores = X @ w_true
    y = np.sign(y_scores + np.random.normal(0, noise, n_samples))
    y = np.where(y == 0, 1, y)
    return X, y


def generate_overlapping_gaussians(n_samples=2000, noise=0.1, seed=42):
    """Two Gaussians at distance d≈1.5 with moderate overlap."""
    np.random.seed(seed)
    n_per = n_samples // 2
    X1 = np.random.randn(n_per, 2) * 0.8 + np.array([0.75, 0.75])
    X2 = np.random.randn(n_per, 2) * 0.8 + np.array([-0.75, -0.75])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_donut(n_samples=2000, noise=0.1, seed=42):
    """Concentric annuli: inner r≈0.4 (+1), outer r≈1.2 (-1)."""
    np.random.seed(seed)
    n_per = n_samples // 2
    theta1 = np.random.uniform(0, 2 * np.pi, n_per)
    r1 = np.clip(0.4 + np.random.normal(0, noise, n_per) * 0.2, 0.2, 0.6)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    theta2 = np.random.uniform(0, 2 * np.pi, n_per)
    r2 = np.clip(1.2 + np.random.normal(0, noise, n_per) * 0.3, 0.9, 1.5)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_circles(n_samples=2000, noise=0.1, seed=42):
    """Concentric circles with sine/cosine warping at midpoint."""
    np.random.seed(seed)
    n_per = n_samples // 2
    t1 = np.random.uniform(0, 2 * np.pi, n_per)
    r1 = 0.5 + np.random.normal(0, noise, n_per)
    X1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])
    t2 = np.random.uniform(0, 2 * np.pi, n_per)
    r2 = 1.5 + np.random.normal(0, noise, n_per)
    X2 = np.column_stack([r2 * np.cos(t2), r2 * np.sin(t2)])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    # Sine/cosine warping past midpoint
    mid = n_samples // 2
    X[mid:, 0] = np.sin(2 * np.pi * X[mid:, 0])
    X[mid:, 1] = np.cos(2 * np.pi * X[mid:, 1])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_quadratic(n_samples=2000, noise=0.1, seed=42):
    """Circular boundary x₁²+x₂²=1 with uniform noise."""
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n_samples, 2))
    X += np.random.normal(0, noise, X.shape)
    radius_sq = X[:, 0] ** 2 + X[:, 1] ** 2
    y = np.where(radius_sq < 1, 1, -1).astype(float)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_high_curvature(n_samples=2000, noise=0.1, seed=42):
    """S-shaped boundary: y above/below sin(x) ± 0.5."""
    np.random.seed(seed)
    n_per = n_samples // 2
    x1 = np.random.uniform(-3, 3, n_per)
    y1 = np.sin(x1) + 0.5 + np.abs(np.random.normal(0, 0.3, n_per)) + 0.1
    X1 = np.column_stack([x1, y1])
    x2 = np.random.uniform(-3, 3, n_per)
    y2 = np.sin(x2) - 0.5 - np.abs(np.random.normal(0, 0.3, n_per)) - 0.1
    X2 = np.column_stack([x2, y2])
    X = np.vstack([X1, X2]) + np.random.normal(0, noise, (n_samples, 2))
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_moons(n_samples=2000, noise=0.1, seed=42):
    """Two crescents with exponential transform at midpoint."""
    np.random.seed(seed)
    n_per = n_samples // 2
    t = np.linspace(0, np.pi, n_per)
    X1 = np.column_stack([np.cos(t), np.sin(t)])
    X2 = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])
    X = np.vstack([X1, X2]) + np.random.normal(0, noise, (n_samples, 2))
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    # Exponential transform past midpoint
    mid = n_samples // 2
    X[mid:] = np.exp(X[mid:]) - 1
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_sinewave(n_samples=2000, noise=0.1, seed=42):
    """Boundary defined by y = sin(x) ± offset."""
    np.random.seed(seed)
    n_per = n_samples // 2
    x1 = np.random.uniform(-3, 3, n_per)
    y1 = np.sin(x1) + np.abs(np.random.normal(0, 0.5, n_per)) + 0.2
    X1 = np.column_stack([x1, y1])
    x2 = np.random.uniform(-3, 3, n_per)
    y2 = np.sin(x2) - np.abs(np.random.normal(0, 0.5, n_per)) - 0.2
    X2 = np.column_stack([x2, y2])
    X = np.vstack([X1, X2]) + np.random.normal(0, noise, (n_samples, 2))
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_spirals(n_samples=2000, noise=0.1, seed=42):
    """Two intertwined spirals (2 turns)."""
    np.random.seed(seed)
    n_per = n_samples // 2
    n_turns = 2
    t1 = np.linspace(0, n_turns * 2 * np.pi, n_per)
    r1 = t1 / (n_turns * 2 * np.pi)
    X1 = np.column_stack([
        r1 * np.cos(t1) + np.random.normal(0, noise, n_per),
        r1 * np.sin(t1) + np.random.normal(0, noise, n_per),
    ])
    t2 = np.linspace(0, n_turns * 2 * np.pi, n_per)
    r2 = t2 / (n_turns * 2 * np.pi)
    X2 = np.column_stack([
        r2 * np.cos(t2 + np.pi) + np.random.normal(0, noise, n_per),
        r2 * np.sin(t2 + np.pi) + np.random.normal(0, noise, n_per),
    ])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_swissroll(n_samples=2000, noise=0.1, seed=42):
    """3-d Swiss roll with labels from z-coordinate median split."""
    np.random.seed(seed)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y_coord = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)
    X = np.column_stack([x, y_coord, z]) + np.random.normal(0, noise, (n_samples, 3))
    y = np.where(z > np.median(z), 1, -1).astype(float)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def generate_xor(n_samples=2000, noise=0.1, seed=42):
    """4-quadrant XOR with radial-basis transform at midpoint."""
    np.random.seed(seed)
    nq = n_samples // 4
    # +1 class: bottom-left + top-right
    X1 = np.random.uniform(-2, -0.5, (nq, 2))
    X2 = np.random.uniform(0.5, 2, (nq, 2))
    X_pos = np.vstack([X1, X2])
    # -1 class: top-left + bottom-right
    X3 = np.column_stack([np.random.uniform(0.5, 2, nq), np.random.uniform(-2, -0.5, nq)])
    X4 = np.column_stack([np.random.uniform(-2, -0.5, nq), np.random.uniform(0.5, 2, nq)])
    X_neg = np.vstack([X3, X4])
    X = np.vstack([X_pos, X_neg]) + np.random.normal(0, noise, (4 * nq, 2))
    y = np.hstack([np.ones(2 * nq), -np.ones(2 * nq)])
    # Radial-basis transform past midpoint
    mid = n_samples // 2
    dsq = np.sum(X[mid:] ** 2, axis=1, keepdims=True)
    X[mid:] = np.exp(-dsq) * X[mid:]
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 2 — DATA PERTURBATION                                      
# ═══════════════════════════════════════════════════════════════════════

def apply_imbalance(X, y, pos_ratio=0.5, seed=42):
    """Re-sample to achieve target class ratio.  Preserves temporal order."""
    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    n_total = len(y)
    n_pos = int(n_total * pos_ratio)
    n_neg = n_total - n_pos
    # Sample with replacement if target exceeds available
    pos_sel = np.random.choice(pos_idx, n_pos, replace=(n_pos > len(pos_idx)))
    neg_sel = np.random.choice(neg_idx, n_neg, replace=(n_neg > len(neg_idx)))
    # Sort to preserve temporal structure
    all_idx = np.sort(np.concatenate([pos_sel, neg_sel]))
    return X[all_idx], y[all_idx]


def inject_outliers(X, y, outlier_ratio=0.05, outlier_scale=10.0, seed=42):
    """Scale-based outlier injection that varies by imbalance level.

    - Minority class gets random magnitude variation
    - Majority class gets time-varying scales (breaks running-average methods)
    - Late minority samples get a progressive boost
    - For extreme imbalance (>85 % positive): sudden magnitude shift at 75 %
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    X_mod = X.copy()
    pos_mask = (y == 1)
    neg_mask = (y == -1)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    pos_ratio = n_pos / n_samples if n_samples > 0 else 0.5
    extreme = pos_ratio > 0.85

    # Strategy 1: minority magnitude variation
    if n_pos > 0:
        scale_range = (-2, 2) if extreme else (-1, 1)
        scales = 10.0 ** np.random.uniform(*scale_range, n_pos)
        X_mod[pos_mask] *= scales[:, np.newaxis]

    # Strategy 2: majority time-varying scale
    if n_neg > 0:
        neg_indices = np.where(neg_mask)[0]
        if extreme:
            t = np.linspace(0, 2 * np.pi, len(neg_indices))
            scales = 0.05 + 0.45 * (1 + np.sin(t))
            X_mod[neg_indices] *= scales[:, np.newaxis]
        else:
            X_mod[neg_mask] *= 0.1

    # Strategy 3: late-minority progressive boost
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) > 4:
        if extreme:
            start = int(len(pos_indices) * 0.5)
            late = pos_indices[start:]
            X_mod[late] *= np.linspace(10, 200, len(late))[:, np.newaxis]
        else:
            start = int(len(pos_indices) * 0.75)
            X_mod[pos_indices[start:]] *= 50.0

    # Strategy 4 (extreme only): sudden 5× shift at 75 % of stream
    if extreme:
        X_mod[int(n_samples * 0.75):] *= 5.0

    return X_mod, y


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 3 — REAL DATA LOADERS                                      
#                                                                       
#   All data is read from local parquet files — no downloads.           
# ═══════════════════════════════════════════════════════════════════════

def load_cybersecurity_data(file_path):
    """Load a cybersecurity parquet, group by timestamp, return (X, y).

    Features are summed per time period; label is +1 if any attack, else -1.
    Returns numpy arrays ready for single-pass online learning.
    """
    df = pd.read_parquet(file_path)
    # Determine time column
    time_col = 'Time' if 'Time' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if time_col is None:
        print(f"  WARNING: no time column in {os.path.basename(file_path)}")
        return None, None
    # Drop non-feature columns
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    # Group by time
    x_df = df.groupby(time_col).sum()
    # Label column
    if 'label' in x_df.columns:
        y = x_df['label'].map(lambda v: 1 if v > 0 else -1).to_numpy()
        x_df = x_df.drop(columns=['label'])
    elif 'Class' in x_df.columns:
        y = x_df['Class'].map(lambda v: 1 if v > 0 else -1).to_numpy()
        x_df = x_df.drop(columns=['Class'])
    else:
        print(f"  ERROR: no label column in {os.path.basename(file_path)}")
        return None, None
    X = x_df.values.astype(np.float64)
    print(f"  Grouped by '{time_col}': {len(X)} periods, attack ratio: {(y==1).sum()/len(y):.4f}")
    return X, y


def load_kaggle_data(file_path):
    """Load Kaggle CreditFraud parquet for pure online learning.

    Temporal 80/20 split matching online_kaggle_fraud.py approach.
    Returns (X_train, y_train, X_test, y_test) with labels in {-1, +1}.
    """
    df = pd.read_parquet(file_path)
    
    # Drop user_id if present
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    
    # Sort by Time column for temporal ordering
    if 'Time' in df.columns:
        df = df.sort_values('Time').reset_index(drop=True)
    
    # Temporal split (80% train, 20% test)
    split_idx = int(0.8 * len(df))
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    # Feature columns (exclude time and label columns)
    drop_cols = ['Time', 'timestamp', 'user_id', 'label', 'Class']
    feat_cols = [c for c in df.columns if c not in drop_cols]
    
    X_train = train_data[feat_cols].values.astype(np.float64)
    X_test = test_data[feat_cols].values.astype(np.float64)
    
    # Handle both 'label' and 'Class' column names
    label_col = 'Class' if 'Class' in df.columns else 'label'
    y_train = np.where(train_data[label_col].values == 0, -1, 1)
    y_test = np.where(test_data[label_col].values == 0, -1, 1)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"  Train fraud ratio: {(y_train==1).sum()/len(y_train):.4f}")
    print(f"  Test fraud ratio: {(y_test==1).sum()/len(y_test):.4f}")
    
    return X_train, y_train, X_test, y_test


def load_mnist_data(file_path):
    """Load MNIST parquet, binary label (≥5 → +1), 80/20 shuffle split, scale.

    Returns (X_train, y_train, X_val, y_val).
    """
    df = pd.read_parquet(file_path)
    # Binary label: digits ≥ 5 → +1, digits < 5 → −1
    df['label_binary'] = np.where(df['label'] >= 5, 1, -1)
    # Shuffle with fixed seed, then split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    # Feature columns: all numeric except label columns
    feat_cols = [c for c in df.columns
                 if c not in ['label', 'label_binary'] and np.issubdtype(df[c].dtype, np.number)]
    X_train = train_df[feat_cols].values.astype(np.float64)
    y_train = train_df['label_binary'].values.astype(np.float64)
    X_val = val_df[feat_cols].values.astype(np.float64)
    y_val = val_df['label_binary'].values.astype(np.float64)
    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Class balance: +1={np.mean(y_train==1):.3f}, -1={np.mean(y_train==-1):.3f}")
    return X_train, y_train, X_val, y_val


def load_unsw_data(data_dir):
    """Load UNSW_NB15 from separate train/test parquets, scale features.

    Expects data_dir/kernel/UNSW_NB15_train.parquet and
    data_dir/kernel/UNSW_NB15_test.parquet.

    Returns (X_train, y_train, X_test, y_test) with labels in {-1, +1}.
    """
    train_path = os.path.join(data_dir, 'kernel', 'UNSW_NB15_train.parquet')
    test_path = os.path.join(data_dir, 'kernel', 'UNSW_NB15_test.parquet')
    df_tr = pd.read_parquet(train_path).dropna()
    df_te = pd.read_parquet(test_path).dropna()
    feat_cols = [c for c in df_tr.columns if c != 'label']
    X_train = df_tr[feat_cols].values.astype(np.float64)
    X_test = df_te[feat_cols].values.astype(np.float64)
    y_train = np.where(df_tr['label'].values == 0, -1, 1)
    y_test = np.where(df_te['label'].values == 0, -1, 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"  Train attack ratio: {(y_train==1).sum()/len(y_train):.4f}")
    print(f"  Test  attack ratio: {(y_test==1).sum()/len(y_test):.4f}")
    return X_train, y_train, X_test, y_test


def load_uci_dataset(parquet_path, random_split=True, test_ratio=0.2):
    """Load a UCI dataset parquet for the kernel benchmark.

    Returns (X_train, y_train, X_test, y_test) with labels in {-1, +1}.
    """
    df = pd.read_parquet(parquet_path).dropna()
    y = df['label'].values.astype(int)
    X = df.drop('label', axis=1).values.astype(np.float64)
    y = np.where(y == 0, -1, 1)
    n = len(X)
    split = int(n * (1 - test_ratio))
    if random_split:
        idx = np.random.permutation(n)
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]
    else:
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pos_tr = (y_train == 1).sum()
    print(f"  Train: {len(X_train)}, +1: {pos_tr} ({100*pos_tr/len(y_train):.1f}%)")
    print(f"  Test:  {len(X_test)}, +1: {(y_test==1).sum()}")
    return X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 4 — EVALUATION HELPERS                                     
# ═══════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred):
    """Full metric dict for batch kernel experiments."""
    y_t = np.where(y_true == -1, 0, 1)
    y_p = np.where(y_pred == -1, 0, 1)
    p = precision_score(y_t, y_p, zero_division=0)
    r = recall_score(y_t, y_p, zero_division=0)
    f = f1_score(y_t, y_p, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    return {
        'Precision': p, 'Recall': r, 'F1': f,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def calculate_metrics(y_true, y_pred):
    """Lightweight accuracy + F1 for simulation experiments."""
    # Coerce any 0 predictions to +1 so labels stay in {-1, +1}
    y_pred = np.where(np.asarray(y_pred) == 0, 1, y_pred)
    y_true = np.where(np.asarray(y_true) == 0, 1, y_true)
    acc = accuracy_score(y_true, y_pred)
    f = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return {'Accuracy': acc, 'F1': f}


def tune_gamma(X_train, y_train, gamma_values=None, val_ratio=0.2,
               max_sv=200, epochs=1):
    """Grid-search for best RBF gamma using validation F1.

    Uses KernelPA internally (fast and stable).
    """
    if gamma_values is None:
        gamma_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02,
                        0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    n = len(X_train)
    val_size = int(n * val_ratio)
    idx = np.random.permutation(n)
    X_tr, y_tr = X_train[idx[val_size:]], y_train[idx[val_size:]]
    X_vl, y_vl = X_train[idx[:val_size]], y_train[idx[:val_size]]
    print(f"  Tuning gamma on {len(X_tr)} train / {len(X_vl)} val ...")
    best_gamma, best_f1 = gamma_values[0], -1
    for g in gamma_values:
        algo = KernelPA(gamma=g, max_sv=max_sv, C=1.0)
        algo.fit(X_tr, y_tr, epochs=epochs)
        m = evaluate(y_vl, algo.predict(X_vl))
        if m['F1'] > best_f1:
            best_f1 = m['F1']
            best_gamma = g
        print(f"    gamma={g:.4f}: F1={m['F1']:.4f}")
    print(f"  Best gamma: {best_gamma} (F1={best_f1:.4f})")
    return best_gamma


# ═══════════════════════════════════════════════════════════════════════
#   EXPERIMENT 1 — SIMULATION  (Online vs Kernel Online)               
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(n_samples=2000, seed=42, output_dir=None):
    """Compare 7 online algorithms against their kernel-online counterparts
    on 11 synthetic datasets under 3 imbalance ratios.

    Prints a per-dataset, per-algorithm comparison table and saves a
    summary CSV to ``output_dir/comparison_simulation.csv``.
    """
    ensure_reproducibility(seed)
    print("=" * 100)
    print("SIMULATION: Online vs Kernel Online | 11 Datasets | 7 Algorithms | 3 Imbalance Ratios")
    print("=" * 100)

    # 11 data generators  (name → (type_tag, function, extra kwargs))
    generators = {
        'Separable':    ('Separable',     generate_separable_data,       {'n_features': 5, 'noise': 0.1}),
        'OverlapGauss': ('Non-Separable', generate_overlapping_gaussians, {'noise': 0.1}),
        'Donut':        ('Non-Separable', generate_donut,                {'noise': 0.1}),
        'Circles':      ('Non-Separable', generate_circles,              {'noise': 0.1}),
        'Quadratic':    ('Non-Separable', generate_quadratic,            {'noise': 0.1}),
        'HighCurve':    ('Non-Separable', generate_high_curvature,       {'noise': 0.1}),
        'Moons':        ('Non-Separable', generate_moons,                {'noise': 0.1}),
        'SineWave':     ('Non-Separable', generate_sinewave,             {'noise': 0.1}),
        'Spirals':      ('Non-Separable', generate_spirals,              {'noise': 0.1}),
        'SwissRoll':    ('Non-Separable', generate_swissroll,            {'noise': 0.1}),
        'XOR':          ('Non-Separable', generate_xor,                  {'noise': 0.1}),
    }

    # 7 online algorithms  (single-pass: max_epochs=1, no validation)
    online_algos = {
        'PA': PA, 'Perceptron': Perceptron, 'AROW': AROW,
        'OGC': OGC, 'SCW': SCW, 'RDA': RDA, 'AdaRDA': AdaRDA,
    }
    # 7 kernel-online counterparts
    kernel_algos = {
        'PA': KPA, 'Perceptron': KPerceptron, 'AROW': KAROW,
        'OGC': KOGC, 'SCW': KSCW, 'RDA': KRDA, 'AdaRDA': KAdaRDA,
    }

    imbalance_ratios = [(0.5, '50-50'), (0.7, '70-30'), (0.9, '90-10')]
    results = []

    for ds_name, (ds_type, gen_fn, kwargs) in generators.items():
        print(f"\n{'='*100}")
        print(f"Dataset: {ds_name:15s} | Type: {ds_type}")
        print(f"{'='*100}")
        X_base, y_base = gen_fn(n_samples=n_samples, **kwargs)

        for pos_ratio, imb_label in imbalance_ratios:
            X, y = apply_imbalance(X_base.copy(), y_base.copy(), pos_ratio=pos_ratio, seed=seed)
            X, y = inject_outliers(X, y, seed=seed)
            n_pos, n_neg = (y == 1).sum(), (y == -1).sum()
            print(f"\n  Imbalance: {imb_label} (Pos: {n_pos:4d}, Neg: {n_neg:4d})")
            print(f"  {'Algorithm':12s} | {'Online F1':10s} | {'Kernel F1':10s} | {'Gain':8s}")
            print(f"  {'-'*50}")

            for algo_name in online_algos:
                # --- Online ---
                res_online = online_algos[algo_name](X, y)
                yp_online = res_online[0] if isinstance(res_online, tuple) else res_online
                m_online = calculate_metrics(y, yp_online)
                # --- Kernel Online ---
                yp_kernel = kernel_algos[algo_name](X, y, gamma=1.0)
                m_kernel = calculate_metrics(y, yp_kernel)
                gain = m_kernel['F1'] - m_online['F1']
                print(f"  {algo_name:12s} | {m_online['F1']:10.4f} | {m_kernel['F1']:10.4f} | {gain:+8.4f}")

                for method, metrics in [('Online', m_online), ('Kernel', m_kernel)]:
                    results.append({
                        'Data_Type': ds_type, 'Dataset': ds_name,
                        'Imbalance': imb_label, 'Pos_Ratio': pos_ratio,
                        'Algorithm': algo_name, 'Method': method,
                        'F1': metrics['F1'], 'Accuracy': metrics['Accuracy'],
                    })

    # --- Summary ---
    df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for dtype in df['Data_Type'].unique():
        print(f"\n{dtype} Data:")
        sub = df[df['Data_Type'] == dtype]
        for imb in ['50-50', '70-30', '90-10']:
            si = sub[sub['Imbalance'] == imb]
            f_on = si[si['Method'] == 'Online']['F1'].mean()
            f_ke = si[si['Method'] == 'Kernel']['F1'].mean()
            print(f"  {imb}: Online={f_on:.4f} | Kernel={f_ke:.4f} | Gain={f_ke-f_on:+.4f}")

    # Save CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'comparison_simulation.csv')
        df.to_csv(path, index=False)
        print(f"\nResults saved to: {path}")

    # --- Generate imbalance_ratio_analysis.png ---
    if output_dir and len(df) > 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        imb_order = ['50-50', '70-30', '90-10']
        imb_labels = ['50-50\n(Balanced)', '70-30\n(Moderate)', '90-10\n(Severe)']
        algo_names = list(df['Algorithm'].unique())
        n_algos = len(algo_names)
        bar_colors = ['#b2dfb2', '#f4b6b6', '#a6cee3']  # green, salmon, light blue

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        for row, (method, method_label) in enumerate([('Online', 'STANDARD'), ('Kernel', 'KERNEL')]):
            ax_bar = axes[row, 0]
            ax_line = axes[row, 1]

            # ---- Bar chart (left column) ----
            bar_w = 0.25
            x = np.arange(n_algos)
            median_f1 = {}  # algo -> {imb -> median_f1}

            for k, imb in enumerate(imb_order):
                vals = []
                for a in algo_names:
                    sub = df[(df['Algorithm'] == a) & (df['Method'] == method) & (df['Imbalance'] == imb)]
                    med = sub['F1'].median() if len(sub) > 0 else 0.0
                    vals.append(med)
                    median_f1.setdefault(a, {})[imb] = med
                bars = ax_bar.bar(x + (k - 1) * bar_w, vals, bar_w,
                                  label=imb, color=bar_colors[k], edgecolor='grey', linewidth=0.5)
                for b, v in zip(bars, vals):
                    ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(algo_names, fontsize=9, rotation=45, ha='right')
            ax_bar.set_xlim(-0.5, n_algos - 0.5)
            ax_bar.xaxis.set_major_locator(plt.FixedLocator(x))
            ax_bar.xaxis.set_minor_locator(plt.NullLocator())
            ax_bar.set_ylabel('Median F1 Score', fontsize=11)
            ax_bar.set_title(f'{method_label}: Algorithm Performance Comparison\nAcross Imbalance Ratios',
                             fontsize=12, fontweight='bold')
            ax_bar.set_ylim(0, max(1.05, ax_bar.get_ylim()[1] + 0.02))
            ax_bar.legend(fontsize=8, loc='lower right')
            ax_bar.grid(axis='y', alpha=0.2)

            # ---- Top-3 line chart (right column) ----
            # Rank by 90-10 median F1 (hardest setting)
            ranked = sorted(algo_names, key=lambda a: median_f1[a]['90-10'], reverse=True)
            top3 = ranked[:3]
            markers = ['o', 's', '^']
            line_styles = ['-', '--', '-.']
            line_colors = ['#e41a1c', '#377eb8', '#ff7f00']

            for j, algo in enumerate(top3):
                y_vals = [median_f1[algo][imb] for imb in imb_order]
                ax_line.plot(range(3), y_vals, marker=markers[j], linestyle=line_styles[j],
                             color=line_colors[j], linewidth=2, markersize=8, label=algo)

            ax_line.set_xticks(range(3))
            ax_line.set_xticklabels(imb_labels, fontsize=9)
            ax_line.set_xlim(-0.3, 2.3)
            ax_line.xaxis.set_major_locator(plt.FixedLocator(range(3)))
            ax_line.xaxis.set_minor_locator(plt.NullLocator())
            ax_line.set_ylabel('Median F1 Score', fontsize=11)
            ax_line.set_title(f'{method_label} Top 3:\n{", ".join(top3)}',
                              fontsize=12, fontweight='bold')
            ax_line.legend(fontsize=9)
            ax_line.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'imbalance_ratio_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — ONLINE BENCHMARK  (Kaggle, Cybersecurity)    
# ═══════════════════════════════════════════════════════════════════════

def _online_pass(algo_name, X, y, w):
    """Run a single online pass on data using pre-trained weights.

    Each sample is predicted *before* the model updates on it
    (predict-then-learn), giving a true online evaluation.

    Parameters
    ----------
    algo_name : str       — algorithm name (PA, Perceptron, OGC, etc.)
    X         : ndarray   — test features (n, d)
    y         : ndarray   — test labels in {-1, +1}
    w         : ndarray   — warm-started weight vector from training

    Returns
    -------
    y_pred : ndarray — online predictions on each test sample
    w      : ndarray — weight vector after the online pass
    """
    X_np = np.ascontiguousarray(X, dtype=np.float64)
    y_np = np.ascontiguousarray(y, dtype=np.float64)
    w = w.copy()  # don't mutate caller's weights
    n = X_np.shape[0]
    perm = np.arange(n, dtype=np.int64)  # sequential order

    if algo_name == 'PA':
        y_pred, w = _pa_kernel(X_np, y_np, perm, w)
    elif algo_name == 'Perceptron':
        y_pred, w = _percept_kernel(X_np, y_np, perm, w)
    elif algo_name == 'OGC':
        y_pred, w = _ogc_kernel(X_np, y_np, perm, w)
    elif algo_name == 'AROW':
        sigma_diag = np.ones(X_np.shape[1], dtype=np.float64)
        y_pred, w, _ = _arow_diag_kernel(X_np, y_np, perm, w, sigma_diag, 0.1)
    elif algo_name == 'RDA':
        g = np.zeros(X_np.shape[1], dtype=np.float64)
        y_pred, w, _ = _rda_kernel(X_np, y_np, perm, w, g, 1.0, 1.0, 0, n)
    elif algo_name == 'SCW':
        from scipy.stats import norm as sp_norm
        phi = float(sp_norm.ppf(0.5))
        sigma_diag = np.ones(X_np.shape[1], dtype=np.float64)
        y_pred, w, _ = _scw_diag_kernel(X_np, y_np, perm, w, sigma_diag, 1.0, phi)
    elif algo_name == 'AdaRDA':
        g = np.zeros(X_np.shape[1], dtype=np.float64)
        g1t = np.zeros(X_np.shape[1], dtype=np.float64)
        y_pred, w, _, _ = _adarda_kernel(X_np, y_np, perm, w, g, g1t, 1.0, 1.0, 1.0, 0, n)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return y_pred, w


def run_online_benchmark(data_dir, output_dir=None):
    """Run 7 Numba-accelerated online algorithms on all parquet datasets
    found in ``data_dir``.

    Dataset dispatch:
      MNIST          → 80/20 split,  10 epochs, early stopping
      UNSW_NB15      → separate train/test files, 10 epochs, early stopping
      Kaggle         → temporal split, single-pass
      Cybersecurity  → grouped by timestamp, prequential evaluation
    """
    warmup_jit()

    # Single-pass algorithms (Kaggle / Cybersecurity) - matching online.py parameters
    single_pass_algos = {
        'PA':        lambda X, y: PA(X, y),
        'Perceptron':lambda X, y: Perceptron(X, y),
        'OGC':       lambda X, y: OGC(X, y),
        'AROW':      lambda X, y: AROW(X, y, r=0.1),
        'RDA':       lambda X, y: RDA(X, y, lambda_param=1, gamma_param=1),
        'SCW':       lambda X, y: SCW(X, y, C=1, eta=0.5),
        'AdaRDA':    lambda X, y: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1),
    }

    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        print(f"FATAL: no parquet files in {data_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} datasets.  Starting online benchmark ...")
    all_results = []

    for i, fpath in enumerate(files):
        ds_name = os.path.basename(fpath).replace('.parquet', '')
        print("\n" + "=" * 60)
        print(f"Dataset {i+1}/{len(files)}: {ds_name}")
        print("=" * 60)

        # ---- MNIST ----
        if 'MNIST' in ds_name.upper():
            X_tr, y_tr, X_vl, y_vl = load_mnist_data(fpath)
            mnist_algos = {
                'PA':        lambda X, y, Xv, yv: PA(X, y, max_epochs=10, X_val=Xv, y_val=yv),
                'Perceptron':lambda X, y, Xv, yv: Perceptron(X, y, max_epochs=10, X_val=Xv, y_val=yv),
                'OGC':       lambda X, y, Xv, yv: OGC(X, y, max_epochs=10, X_val=Xv, y_val=yv),
                'AROW':      lambda X, y, Xv, yv: AROW(X, y, r=0.1, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
                'RDA':       lambda X, y, Xv, yv: RDA(X, y, lambda_param=1, gamma_param=1, max_epochs=10, X_val=Xv, y_val=yv),
                'SCW':       lambda X, y, Xv, yv: SCW(X, y, C=1, eta=0.5, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
                'AdaRDA':    lambda X, y, Xv, yv: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1, max_epochs=10, X_val=Xv, y_val=yv),
            }
            for aname, afn in mnist_algos.items():
                print(f"  Running {aname} (10 epochs + online test) ...")
                try:
                    _, w = afn(X_tr, y_tr, X_vl, y_vl)
                    # Continue online on test set: predict-then-update
                    yp, _ = _online_pass(aname, X_vl, y_vl, w)
                    p, r, fnr, fpr, f1 = calculate_class1_metrics(y_vl, yp)
                    print(f"    Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")
                    all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                        'Precision': p, 'Recall': r, 'F1': f1,
                                        'FNR': fnr, 'FPR': fpr})
                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                        'Precision': np.nan, 'Recall': np.nan,
                                        'F1': np.nan, 'FNR': np.nan, 'FPR': np.nan})
            continue

        # ---- UNSW_NB15 (skip if found in top-level scan — handled below) ----
        if 'UNSW' in ds_name.upper():
            continue

        # ---- Kaggle Credit Fraud ----
        # Pure online learning: temporal split, train sequentially, test with final weights
        if 'KAGGLE' in ds_name.upper() or 'CREDITFRAUD' in ds_name.upper():
            print(f"  === KAGGLE DATASET: SIMPLE LINEAR PROCESSING ===")
            X_tr, y_tr, X_te, y_te = load_kaggle_data(fpath)
            
            for aname, afn in single_pass_algos.items():
                print(f"  Running {aname} ...")
                try:
                    # Train the algorithm (sequential online learning)
                    y_pred_train, w = afn(X_tr, y_tr)
                    
                    # Test predictions using final weights
                    test_scores = X_te.dot(w)
                    y_pred_test = np.sign(test_scores)
                    y_pred_test = np.where(y_pred_test == 0, 1, y_pred_test)
                    
                    # Calculate metrics (convert to 0/1 for metrics)
                    y_te_01 = np.where(y_te == -1, 0, 1)
                    y_pred_01 = np.where(y_pred_test == -1, 0, 1)
                    
                    p, r, fnr, fpr, f1 = calculate_class1_metrics(y_te_01, y_pred_01)
                    print(f"    Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")
                    all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                        'Precision': p, 'Recall': r, 'F1': f1,
                                        'FNR': fnr, 'FPR': fpr})
                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                        'Precision': np.nan, 'Recall': np.nan,
                                        'F1': np.nan, 'FNR': np.nan, 'FPR': np.nan})
            continue

        # ---- Cybersecurity ----
        X, y_true = load_cybersecurity_data(fpath)
        if X is None or len(X) <= 1:
            print(f"  Skipping (insufficient data)")
            continue
        print(f"  Prequential evaluation on {len(X)} periods, attack ratio: {(y_true==1).sum()/len(y_true):.4f}")
        for aname, afn in single_pass_algos.items():
            print(f"  Running {aname} ...")
            try:
                yp_stream, _ = afn(X, y_true)
                # Prequential: predict at t, evaluate against true at t+1
                p, r, fnr, fpr, f1 = calculate_class1_metrics(y_true[1:], yp_stream[:-1])
                print(f"    Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")
                all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                    'Precision': p, 'Recall': r, 'F1': f1,
                                    'FNR': fnr, 'FPR': fpr})
            except Exception as e:
                print(f"    ERROR: {e}")

    # ---- UNSW_NB15 (separate train/test parquets in kernel/) ----
    unsw_train_path = os.path.join(data_dir, 'kernel', 'UNSW_NB15_train.parquet')
    unsw_test_path = os.path.join(data_dir, 'kernel', 'UNSW_NB15_test.parquet')
    if os.path.exists(unsw_train_path) and os.path.exists(unsw_test_path):
        ds_name = 'UNSW_NB15'
        print("\n" + "=" * 60)
        print(f"Dataset: {ds_name} (train/test split)")
        print("=" * 60)
        X_tr, y_tr, X_te, y_te = load_unsw_data(data_dir)
        unsw_algos = {
            'PA':        lambda X, y, Xv, yv: PA(X, y, max_epochs=10, X_val=Xv, y_val=yv),
            'Perceptron':lambda X, y, Xv, yv: Perceptron(X, y, max_epochs=10, X_val=Xv, y_val=yv),
            'OGC':       lambda X, y, Xv, yv: OGC(X, y, max_epochs=10, X_val=Xv, y_val=yv),
            'AROW':      lambda X, y, Xv, yv: AROW(X, y, r=0.1, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
            'RDA':       lambda X, y, Xv, yv: RDA(X, y, lambda_param=1, gamma_param=1, max_epochs=10, X_val=Xv, y_val=yv),
            'SCW':       lambda X, y, Xv, yv: SCW(X, y, C=1, eta=0.5, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
            'AdaRDA':    lambda X, y, Xv, yv: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1, max_epochs=10, X_val=Xv, y_val=yv),
        }
        for aname, afn in unsw_algos.items():
            print(f"  Running {aname} (10 epochs + online test) ...")
            try:
                _, w = afn(X_tr, y_tr, X_te, y_te)
                # Continue online on test set: predict-then-update
                yp, _ = _online_pass(aname, X_te, y_te, w)
                p, r, fnr, fpr, f1 = calculate_class1_metrics(y_te, yp)
                print(f"    Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")
                all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                    'Precision': p, 'Recall': r, 'F1': f1,
                                    'FNR': fnr, 'FPR': fpr})
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                    'Precision': np.nan, 'Recall': np.nan,
                                    'F1': np.nan, 'FNR': np.nan, 'FPR': np.nan})

    # --- Save ---
    df = pd.DataFrame(all_results)
    if len(df):
        df.sort_values(['Algorithm', 'Dataset'], inplace=True)
        print("\n" + "=" * 60)
        print("ONLINE BENCHMARK RESULTS")
        print("=" * 60)
        print(df.round(4).to_string())
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'online_benchmark_results.csv')
            df.to_csv(path, index=False, float_format='%.4f')
            print(f"\nSaved to: {path}")
            # Generate plot
            plot_online_benchmark_f1(df, output_dir)
    return df


def plot_online_benchmark_f1(results_csv_or_df, output_dir=None, figsize=(14, 6)):
    """Generate a single F1 bar plot for online benchmark results.

    Creates a grouped bar chart with:
    - X-axis: Datasets
    - Y-axis: F1 Score
    - Different colored bars for each algorithm
    - F1 values displayed clearly on top of each bar

    Parameters
    ----------
    results_csv_or_df : str or DataFrame
        Path to online_benchmark_results.csv or the DataFrame directly
    output_dir : str
        Directory to save the plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load data
    if isinstance(results_csv_or_df, str):
        df = pd.read_csv(results_csv_or_df)
    else:
        df = results_csv_or_df.copy()

    if len(df) == 0:
        print("No data for online benchmark plot.")
        return

    # Get unique datasets and algorithms
    datasets = df['Dataset'].unique().tolist()
    algo_names = df['Algorithm'].unique().tolist()
    n_algos = len(algo_names)
    n_datasets = len(datasets)

    # Algorithm colors
    algo_colors = {
        'PA': '#d62728',         # red
        'Perceptron': '#1f77b4', # blue
        'OGC': '#2ca02c',        # green
        'AROW': '#ff7f0e',       # orange
        'RDA': '#9467bd',        # purple
        'SCW': '#8c564b',        # brown
        'AdaRDA': '#e377c2',     # pink
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Bar settings
    bar_width = 0.11
    x = np.arange(n_datasets)

    # Plot bars for each algorithm
    for i, algo in enumerate(algo_names):
        f1_vals = []
        for ds in datasets:
            sub = df[(df['Algorithm'] == algo) & (df['Dataset'] == ds)]
            f1_val = sub['F1'].values[0] if len(sub) > 0 else 0.0
            f1_vals.append(f1_val)

        # Position bars
        offset = (i - (n_algos - 1) / 2) * bar_width
        color = algo_colors.get(algo, '#333333')

        bars = ax.bar(x + offset, f1_vals, bar_width, label=algo,
                      color=color, edgecolor='white', linewidth=0.5)

    # Format x-axis labels (cleaner dataset names)
    dataset_labels = []
    for ds in datasets:
        label = ds.replace('_combined', '').replace('_kaggle', '')
        # Wrap long names
        if len(label) > 12:
            parts = label.split('_')
            if len(parts) > 1:
                label = '\n'.join(parts)
        dataset_labels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11, rotation=0)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.set_title('Online Benchmark: F1 Score by Dataset and Algorithm',
                 fontsize=16, fontweight='bold', pad=15)

    # Legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=n_algos, fontsize=11, frameon=True)

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'online_benchmark_f1.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Online benchmark F1 plot saved to: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#   EXPERIMENT 3 — BATCH KERNEL BENCHMARK  (UCI datasets, repeated)    
# ═══════════════════════════════════════════════════════════════════════

# Datasets in data/kernel/ that should be evaluated
# Note: HeartDisease_UCI removed (corrupted - all labels=0)
#       ClevelandHeart_UCI is the correct Cleveland Heart Disease dataset
KERNEL_DATASETS = [
    'ClevelandHeart_UCI',
]


def _get_batch_kernel_algos(gamma_val, max_sv):
    """Factory for the 7 batch-kernel algorithms with given gamma."""
    return {
        'KernelPerceptron': lambda: KernelPerceptron(gamma=gamma_val, max_sv=max_sv),
        'KernelPA':         lambda: KernelPA(gamma=gamma_val, max_sv=max_sv, C=1.0),
        'KernelAROW':       lambda: KernelAROW(gamma=gamma_val, r=1.0, max_sv=min(max_sv, 500)),
        'KernelGC':         lambda: KernelGC(gamma=gamma_val, max_sv=max_sv),
        'KernelRDA':        lambda: KernelRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
        'KernelSCW':        lambda: KernelSCW(gamma=gamma_val, C=1.0, phi=0.5, max_sv=min(max_sv, 500)),
        'KernelAdaRDA':     lambda: KernelAdaRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
    }


def run_kernel_benchmark(data_dir, output_dir=None, n_repeats=100,
                         gamma='tune', max_sv=1000, epochs=1,
                         test_ratio=0.3, datasets=None):
    """Run batch-kernel algorithms on UCI datasets with repeated random shuffles.

    For each dataset:
      1. Load raw data once.
      2. On the first repeat, tune gamma (grid search) unless a fixed value is given.
      3. Shuffle → split → scale → fit all algorithms → evaluate on test set.
      4. Repeat n_repeats times.
      5. Report mean ± std for each algorithm.

    Results and per-shuffle F1 curves are saved to output_dir.
    """
    np.random.seed(42)

    ds_list = datasets or KERNEL_DATASETS
    kernel_data_dir = os.path.join(data_dir, 'kernel')
    all_results, per_shuffle = [], []

    for ds_name in ds_list:
        parquet_path = os.path.join(kernel_data_dir, f'{ds_name}.parquet')
        if not os.path.exists(parquet_path):
            print(f"  SKIP: {parquet_path} not found")
            continue
        print("\n" + "=" * 70)
        print(f"DATASET: {ds_name} ({n_repeats} repeats, "
              f"{int((1-test_ratio)*100)}/{int(test_ratio*100)} split)")
        print("=" * 70)

        # Load raw data
        df = pd.read_parquet(parquet_path).dropna()
        X_raw = df.drop('label', axis=1).values.astype(np.float64)
        y_raw = np.where(df['label'].values == 0, -1, 1)
        print(f"  Total: {len(X_raw)}, +1: {(y_raw==1).sum()} ({100*(y_raw==1).mean():.1f}%)")

        algo_metrics = {n: {'Precision': [], 'Recall': [], 'F1': [], 'FPR': [], 'SVs': []}
                        for n in _get_batch_kernel_algos(0.1, max_sv)}

        gamma_val = None
        for rep in range(n_repeats):
            np.random.seed(rep * 42)
            idx = np.random.permutation(len(X_raw))
            split = int(len(X_raw) * (1 - test_ratio))
            X_tr = X_raw[idx[:split]]
            y_tr = y_raw[idx[:split]]
            X_te = X_raw[idx[split:]]
            y_te = y_raw[idx[split:]]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            if rep == 0:
                if gamma == 'tune':
                    gamma_val = tune_gamma(X_tr, y_tr, max_sv=min(100, max_sv), epochs=1)
                elif gamma == 'auto':
                    gamma_val = 1.0 / (X_tr.shape[1] * X_tr.var() + 1e-8)
                else:
                    gamma_val = float(gamma) if isinstance(gamma, str) else gamma
                print(f"  Using gamma = {gamma_val:.6f}")

            algos = _get_batch_kernel_algos(gamma_val, max_sv)
            for name, factory in algos.items():
                algo = factory()
                algo.fit(X_tr, y_tr, epochs=epochs)
                yp = algo.predict(X_te)
                m = evaluate(y_te, yp)
                n_sv = len(algo.support_vectors) if algo.support_vectors is not None else 0
                algo_metrics[name]['Precision'].append(m['Precision'])
                algo_metrics[name]['Recall'].append(m['Recall'])
                algo_metrics[name]['F1'].append(m['F1'])
                algo_metrics[name]['FPR'].append(m['FPR'])
                algo_metrics[name]['SVs'].append(n_sv)
                per_shuffle.append({'Dataset': ds_name, 'Algorithm': name,
                                    'Shuffle': rep + 1, 'F1': m['F1']})
            print(f"  Repeat {rep+1}/{n_repeats} done", end='\r')
        print()

        # Print averaged results
        print(f"\n  {'Algorithm':<20} {'Prec':>10} {'Recall':>10} {'F1':>10} {'SVs':>8}")
        print("  " + "-" * 60)
        for name in algo_metrics:
            pm = np.mean(algo_metrics[name]['Precision'])
            ps = np.std(algo_metrics[name]['Precision'])
            rm = np.mean(algo_metrics[name]['Recall'])
            fm = np.mean(algo_metrics[name]['F1'])
            fs = np.std(algo_metrics[name]['F1'])
            svm = np.mean(algo_metrics[name]['SVs'])
            print(f"  {name:<20} {pm:.4f}+/-{ps:.2f} {rm:.4f}     {fm:.4f}+/-{fs:.2f} {svm:>6.0f}")
            # Use same column names as online benchmark for consistency
            all_results.append({
                'Dataset': ds_name, 'Algorithm': name,
                'Precision': pm, 'Recall': rm, 'F1': fm,
                'FNR': 1.0 - rm, 'FPR': np.mean(algo_metrics[name]['FPR']),
            })

    # Save
    df_res = pd.DataFrame(all_results)
    df_per_shuffle = pd.DataFrame(per_shuffle)
    if output_dir and len(df_res):
        os.makedirs(output_dir, exist_ok=True)
        df_res.to_csv(os.path.join(output_dir, 'kernel_benchmark_results.csv'),
                       index=False, float_format='%.4f')
        df_per_shuffle.to_csv(
            os.path.join(output_dir, 'kernel_per_shuffle_f1.csv'),
            index=False, float_format='%.4f')
        print(f"\nResults saved to {output_dir}/")
    return df_res, df_per_shuffle


# ═══════════════════════════════════════════════════════════════════════
#   PLOTTING FUNCTIONS FOR KERNEL SIM                         
# ═══════════════════════════════════════════════════════════════════════



def plot_kernel_imbalance_analysis(results_csv_or_df, output_dir=None,
                                   datasets=None, figsize=(14, 11)):
    """Generate imbalance ratio analysis plot for kernel benchmark.

    Creates a 2x2 grid similar to the simulation imbalance plot:
    - Left column: Bar charts comparing all algorithms across datasets
    - Right column: Line charts for top 3 algorithms

    Parameters
    ----------
    results_csv_or_df : str or DataFrame
        Path to kernel_benchmark_results.csv or the DataFrame directly
    output_dir : str
        Directory to save the plot
    datasets : list
        List of dataset names to include (acts as "imbalance levels")
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load data
    if isinstance(results_csv_or_df, str):
        df = pd.read_csv(results_csv_or_df)
    else:
        df = results_csv_or_df.copy()

    if datasets is None:
        datasets = df['Dataset'].unique().tolist()

    # Filter to requested datasets
    df = df[df['Dataset'].isin(datasets)]
    if len(df) == 0:
        print("No matching datasets found for plotting.")
        return

    algo_names = df['Algorithm'].unique().tolist()
    n_algos = len(algo_names)
    n_datasets = len(datasets)

    # Colors for each dataset (like imbalance ratios)
    bar_colors = ['#b2dfb2', '#f4b6b6', '#a6cee3', '#fdbf6f', '#cab2d6'][:n_datasets]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ---- ROW 1: Mean F1 comparison ----
    ax_bar = axes[0, 0]
    ax_line = axes[0, 1]

    bar_w = 0.8 / n_datasets
    x = np.arange(n_algos)
    f1_by_algo = {}

    for k, ds in enumerate(datasets):
        vals = []
        for a in algo_names:
            sub = df[(df['Algorithm'] == a) & (df['Dataset'] == ds)]
            # Use 'F1' column (matching online benchmark format)
            f1_val = sub['F1'].values[0] if len(sub) > 0 else 0.0
            vals.append(f1_val)
            f1_by_algo.setdefault(a, {})[ds] = f1_val

        offset = (k - (n_datasets - 1) / 2) * bar_w
        bars = ax_bar.bar(x + offset, vals, bar_w, label=ds,
                          color=bar_colors[k], edgecolor='grey', linewidth=0.5)
        # Add value labels on bars
        for b, v in zip(bars, vals):
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([a.replace('Kernel', '') for a in algo_names], fontsize=10)
    ax_bar.set_ylabel('F1 Score', fontsize=12)
    ax_bar.set_title('KERNEL: Algorithm Performance Comparison\nAcross Datasets',
                     fontsize=13, fontweight='bold')
    ax_bar.set_ylim(0, min(1.1, ax_bar.get_ylim()[1] + 0.1))
    ax_bar.legend(fontsize=9, loc='lower right')
    ax_bar.grid(axis='y', alpha=0.3)

    # ---- Top-3 line chart (by average F1 across datasets) ----
    avg_f1 = {a: np.mean(list(f1_by_algo[a].values())) for a in algo_names}
    ranked = sorted(algo_names, key=lambda a: avg_f1[a], reverse=True)
    top3 = ranked[:3]

    markers = ['o', 's', '^']
    line_styles = ['-', '--', '-.']
    line_colors = ['#e41a1c', '#377eb8', '#ff7f00']

    for j, algo in enumerate(top3):
        y_vals = [f1_by_algo[algo].get(ds, 0) for ds in datasets]
        ax_line.plot(range(n_datasets), y_vals, marker=markers[j],
                     linestyle=line_styles[j], color=line_colors[j],
                     linewidth=2.5, markersize=10, label=algo.replace('Kernel', ''))

    ax_line.set_xticks(range(n_datasets))
    ax_line.set_xticklabels([d.replace('_UCI', '').replace('_', '\n') for d in datasets],
                           fontsize=10)
    ax_line.set_ylabel('Mean F1 Score', fontsize=12)
    ax_line.set_title(f'KERNEL Top 3:\n{", ".join([a.replace("Kernel", "") for a in top3])}',
                      fontsize=13, fontweight='bold')
    ax_line.legend(fontsize=11)
    ax_line.grid(alpha=0.3)

    # ---- ROW 2: Precision and Recall comparison ----
    ax_prec = axes[1, 0]
    ax_rec = axes[1, 1]

    # Precision bars
    for k, ds in enumerate(datasets):
        vals = []
        for a in algo_names:
            sub = df[(df['Algorithm'] == a) & (df['Dataset'] == ds)]
            # Use 'Precision' column (matching online benchmark format)
            prec = sub['Precision'].values[0] if len(sub) > 0 else 0.0
            vals.append(prec)
        offset = (k - (n_datasets - 1) / 2) * bar_w
        bars = ax_prec.bar(x + offset, vals, bar_w, label=ds,
                           color=bar_colors[k], edgecolor='grey', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax_prec.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                         f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax_prec.set_xticks(x)
    ax_prec.set_xticklabels([a.replace('Kernel', '') for a in algo_names], fontsize=10)
    ax_prec.set_ylabel('Precision', fontsize=12)
    ax_prec.set_title('Precision Comparison', fontsize=13, fontweight='bold')
    ax_prec.set_ylim(0, min(1.1, ax_prec.get_ylim()[1] + 0.1))
    ax_prec.legend(fontsize=9, loc='lower right')
    ax_prec.grid(axis='y', alpha=0.3)

    # Recall bars
    for k, ds in enumerate(datasets):
        vals = []
        for a in algo_names:
            sub = df[(df['Algorithm'] == a) & (df['Dataset'] == ds)]
            # Use 'Recall' column (matching online benchmark format)
            rec = sub['Recall'].values[0] if len(sub) > 0 else 0.0
            vals.append(rec)
        offset = (k - (n_datasets - 1) / 2) * bar_w
        bars = ax_rec.bar(x + offset, vals, bar_w, label=ds,
                          color=bar_colors[k], edgecolor='grey', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax_rec.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels([a.replace('Kernel', '') for a in algo_names], fontsize=10)
    ax_rec.set_ylabel('Recall', fontsize=12)
    ax_rec.set_title('Recall Comparison', fontsize=13, fontweight='bold')
    ax_rec.set_ylim(0, min(1.1, ax_rec.get_ylim()[1] + 0.1))
    ax_rec.legend(fontsize=9, loc='lower right')
    ax_rec.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'kernel_dataset_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Kernel dataset analysis plot saved to: {path}")
    plt.close()


def run_kernel_benchmark_with_plots(data_dir, output_dir, n_repeats=30,
                                    gamma='tune', max_sv=500, epochs=1,
                                    test_ratio=0.3, datasets=None):
    """Run kernel benchmark on specified datasets and generate all plots.

    Convenience function that runs the benchmark and automatically generates:
    1. Kernel benchmark F1 bar plot (similar to online_benchmark_f1.png)
    2. Running F1 score plots across shuffles
    3. Dataset comparison analysis plot (imbalance-style)

    Parameters
    ----------
    data_dir : str
        Directory containing data/kernel/*.parquet files
    output_dir : str
        Directory to save results and plots
    n_repeats : int
        Number of random shuffles per dataset
    datasets : list
        Datasets to benchmark (default: all 4 kernel datasets)
    """
    if datasets is None:
        datasets = KERNEL_DATASETS

    print("=" * 70)
    print("KERNEL BENCHMARK WITH PLOTS")
    print(f"Datasets: {datasets}")
    print(f"Repeats: {n_repeats}, Epochs: {epochs}, Max SVs: {max_sv}")
    print("=" * 70)

    # Run the benchmark
    df_res, df_per_shuffle = run_kernel_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        n_repeats=n_repeats,
        gamma=gamma,
        max_sv=max_sv,
        epochs=epochs,
        test_ratio=test_ratio,
        datasets=datasets,
    )

    # Generate plots
    if len(df_per_shuffle) > 0:
        print("\nGenerating plots...")
        plot_kernel_imbalance_analysis(df_res, output_dir, datasets)
        print("All plots generated successfully!")

    return df_res, df_per_shuffle
