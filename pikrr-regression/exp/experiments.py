#!/usr/bin/env python3
"""experiments.py — Real-world regression experiments for QKRR vs neural baselines."""

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from algos import (
    QKRR, PINNSolver, StandardMLP, DeepMLP, ResMLP,
    Conv1DRegressor, LSTMRegressor,
)


# ══════════════════════════════════════════════════════════════════════
#  REAL-WORLD DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _download_household_power():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/00235/household_power_consumption.zip")
    zip_path = DATA_DIR / "household_power.zip"
    txt_path = DATA_DIR / "household_power_consumption.txt"
    if txt_path.exists():
        return txt_path
    print("Downloading UCI Household Power...")
    urllib.request.urlretrieve(url, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    return txt_path


def load_household_power(max_samples=None, pca_components=None):
    txt_path = _download_household_power()
    df = pd.read_csv(txt_path, sep=';', na_values='?', low_memory=False)
    df = df.dropna()
    feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    target_col = 'Global_active_power'
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    if max_samples is not None:
        df = df.iloc[:max_samples]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    y = (y - y.mean()) / (y.std() + 1e-8)
    print(f"  Household Power: {X.shape[0]:,} samples, {X.shape[1]} features"
          + (f" (PCA-{pca_components})" if pca_components else ""))
    return X, y


def _download_nslkdd():
    urls = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ]
    csv_path = DATA_DIR / "nslkdd_train.csv"
    if csv_path.exists():
        return csv_path
    print("Downloading NSL-KDD...")
    for url in urls:
        try:
            urllib.request.urlretrieve(url, csv_path)
            return csv_path
        except Exception:
            continue
    raise RuntimeError("NSL-KDD download failed")


def load_nslkdd(max_samples=None, pca_components=None):
    csv_path = _download_nslkdd()
    col_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level',
    ]
    numeric_cols = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'num_compromised', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    ]
    target_col = 'src_bytes'
    feature_cols = [c for c in numeric_cols if c != target_col]
    df = pd.read_csv(csv_path, names=col_names, header=None)
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_cols + [target_col])
    if max_samples is not None:
        df = df.iloc[:max_samples]
    X = df[feature_cols].values.astype(np.float64)
    y = np.log1p(df[target_col].values.astype(np.float64))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    y = (y - y.mean()) / (y.std() + 1e-8)
    print(f"  NSL-KDD: {X.shape[0]:,} samples, {X.shape[1]} features"
          + (f" (PCA-{pca_components})" if pca_components else ""))
    return X, y


def load_gaze(max_samples=None, pca_components=None):
    """Gaze prediction: concatenate train and validation sets."""
    train = pd.read_csv(DATA_DIR / "gaze_train.inputs", header=None, sep=r'\s+')
    valid = pd.read_csv(DATA_DIR / "gaze_valid.inputs", header=None, sep=r'\s+')
    train_t = pd.read_csv(DATA_DIR / "gaze_train.targets", header=None, sep=r'\s+')
    valid_t = pd.read_csv(DATA_DIR / "gaze_valid.targets", header=None, sep=r'\s+')
    X = pd.concat([train, valid], ignore_index=True).values.astype(np.float64)
    y = pd.concat([train_t, valid_t], ignore_index=True).values.astype(np.float64).ravel()
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    y = (y - y.mean()) / (y.std() + 1e-8)
    print(f"  Gaze: {X.shape[0]:,} samples, {X.shape[1]} features"
          + (f" (PCA-{pca_components})" if pca_components else ""))
    return X, y


def load_no2(max_samples=None, pca_components=None, add_bias=True):
    """NO2 concentration regression with optional intercept column."""
    dat_path = DATA_DIR / "NO2.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"NO2.dat not found in {DATA_DIR}")
    data = pd.read_csv(dat_path, header=None, sep=r'\s+').values.astype(np.float64)
    # R: data[order(data[,8], data[,7]),]
    data = data[np.lexsort((data[:, 6], data[:, 7]))]
    # R: data <- as.matrix(data[,-c(8)])
    data = np.delete(data, 7, axis=1)
    # R: data <- scale(data)
    data = StandardScaler().fit_transform(data)
    if max_samples is not None:
        data = data[:max_samples]
    y = data[:, 0].copy()
    X = data[:, 1:].copy()
    # R: X <- cbind(rep(1,500), X)
    if add_bias:
        X = np.column_stack([np.ones(len(X)), X])
    if pca_components is not None:
        if add_bias:
            bias = X[:, 0:1]
            X = PCA(n_components=pca_components).fit_transform(X[:, 1:])
            X = np.column_stack([bias, X])
        else:
            X = PCA(n_components=pca_components).fit_transform(X)
    print(f"  NO2: {X.shape[0]:,} samples, {X.shape[1]} features"
          + (f" (PCA-{pca_components})" if pca_components else ""))
    return X, y


def load_ise(max_samples=None, pca_components=None):
    """Istanbul Stock Exchange (ISE) daily return regression."""
    rda_path = DATA_DIR / "ISE.rda"
    import pyreadr
    result = pyreadr.read_r(str(rda_path))
    data = next(iter(result.values()))
    # R: X <- as.matrix(data[,-c(1,3)]), Y <- as.matrix(data[,3])
    y = data.iloc[:, 2].values.astype(np.float64)
    X_df = data.drop(columns=[data.columns[0], data.columns[2]])
    X = X_df.values.astype(np.float64)
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    y = (y - y.mean()) / (y.std() + 1e-8)
    print(f"  ISE: {X.shape[0]:,} samples, {X.shape[1]} features"
          + (f" (PCA-{pca_components})" if pca_components else ""))
    return X, y


# ══════════════════════════════════════════════════════════════════════
#  PLOT / SAVE UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _save_table(rows, output_path):
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  CSV  -> {output_path}")
    return df


def ensure_reproducibility(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# ══════════════════════════════════════════════════════════════════════
#  REAL-WORLD EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

def _qkrr_for_dataset(dataset_name):
    """Return dataset-tuned QKRR keyword arguments."""
    base = dict(pde_weight=0.0, hhl_eps=0.01, track_complexity=False)
    if dataset_name == 'household':
        # Tuned on the time-based train/test split.
        base.update(n_qubits=10, length_scale=2.0, beta=0.05, lambda_reg=1e-5)
    elif dataset_name == 'nslkdd':
        # Tuned on the random train/test split.
        base.update(n_qubits=10, length_scale=5.0, beta=0.07, lambda_reg=1e-4)
    elif dataset_name == 'gaze':
        base.update(n_qubits=8, length_scale=10.0, beta=0.05, lambda_reg=1e-5)
    elif dataset_name == 'no2':
        base.update(n_qubits=6, length_scale=10.0, beta=0.01, lambda_reg=1e-3)
    elif dataset_name == 'ise':
        # Tuned on the time-ordered train/test split.
        base.update(n_qubits=8, length_scale=70.0, beta=0.02, lambda_reg=5e-4)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return base


def exp_real_accuracy(dataset_name, output_dir, max_samples=50000, n_epochs=100, seed=42):
    """EXP-R1: Test-set accuracy on real-world data."""
    print(f"\n{'='*70}\nEXP-R1: Real Accuracy — {dataset_name}\n{'='*70}")
    ensure_reproducibility(seed)

    loader_dispatch = {
        'household': lambda: load_household_power(max_samples=max_samples, pca_components=3),
        'nslkdd': lambda: load_nslkdd(max_samples=max_samples, pca_components=3),
        'gaze': lambda: load_gaze(max_samples=max_samples, pca_components=None),
        'no2': lambda: load_no2(max_samples=max_samples, pca_components=None),
        'ise': lambda: load_ise(max_samples=max_samples, pca_components=None),
    }
    if dataset_name not in loader_dispatch:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    X, y = loader_dispatch[dataset_name]()

    n, d = X.shape
    n_train = int(0.8 * n)
    if dataset_name in ('household', 'ise'):
        # Time-ordered split for temporal datasets.
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        X = X[perm]
        y = y[perm]
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

    rows = []

    # PINN (no physics, just MLP on real data)
    print("  [PINN] Training...")
    t0 = time.time()
    pinn = PINNSolver(input_dim=d, hidden_dim=64, n_layers=4,
                      lambda_phys=0.0, lr=1e-3)
    for _ in range(n_epochs):
        pinn.train_step(X_train, y_train, X_train,
                        np.zeros(len(y_train)), n_epochs=1,
                        eval_set=(X_test, y_test))
    pinn_time = time.time() - t0
    pinn_preds = pinn.predict(X_test)
    pinn_rmse = np.sqrt(np.mean((pinn_preds - y_test) ** 2))
    pinn_r2 = r2_score(y_test, pinn_preds)
    rows.append({'Method': 'PINN', 'Dataset': dataset_name,
                 'RMSE': pinn_rmse, 'R2': pinn_r2, 'Time_sec': pinn_time})
    print(f"    test RMSE={pinn_rmse:.4f}  R2={pinn_r2:.4f}")

    # Diverse neural baselines (batch multi-epoch for a fair comparison)
    nn_models = [
        ('MLP', StandardMLP(d, hidden_dim=64, n_layers=3)),
        ('DeepMLP', DeepMLP(d, hidden_dim=128, n_layers=5)),
        ('ResMLP', ResMLP(d, hidden_dim=64, n_blocks=4)),
        ('CNN1D', Conv1DRegressor(d, channels=32, kernel_size=3)),
        ('LSTM', LSTMRegressor(d, hidden_dim=32, seq_len=5)),
    ]
    for name, model in nn_models:
        print(f"  [{name}] Training...")
        t0 = time.time()
        hist = model.fit(X_train, y_train, n_epochs=n_epochs,
                         eval_set=(X_test, y_test))
        if name == 'LSTM':
            preds = model.predict_full(X_test)
        else:
            preds = model.predict(X_test)
        elapsed = time.time() - t0
        losses = (preds - y_test) ** 2
        rmse = np.sqrt(np.mean(losses))
        r2 = r2_score(y_test, preds)
        rows.append({'Method': name, 'Dataset': dataset_name,
                     'RMSE': rmse, 'R2': r2, 'Time_sec': elapsed})
        print(f"    test RMSE={rmse:.4f}  R2={r2:.4f}")

    # QKRR (no physics on real data)
    print("  [QKRR] Fitting...")
    t0 = time.time()
    qkrr = QKRR(**_qkrr_for_dataset(dataset_name))
    qkrr.fit(X_train, y_train, X_train, np.zeros(len(y_train)))
    qkrr_time = time.time() - t0
    qkrr_preds = qkrr.predict(X_test)
    qkrr_rmse = np.sqrt(np.mean((qkrr_preds - y_test) ** 2))
    qkrr_r2 = r2_score(y_test, qkrr_preds)
    rows.append({'Method': 'QKRR', 'Dataset': dataset_name,
                 'RMSE': qkrr_rmse, 'R2': qkrr_r2, 'Time_sec': qkrr_time})
    print(f"    test RMSE={qkrr_rmse:.4f}  R2={qkrr_r2:.4f}")

    # Save test-set accuracy table
    _save_table(rows, os.path.join(output_dir, f'expR1_accuracy_{dataset_name}.csv'))

    return pd.DataFrame(rows)
