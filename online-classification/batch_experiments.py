#!/usr/bin/env python3

import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from algos import (
    PA, Perceptron, OGC, AROW, RDA, SCW, AdaRDA,
    calculate_class1_metrics, warmup_jit,
    _pa_kernel, _percept_kernel, _ogc_kernel,
    _arow_diag_kernel, _rda_kernel, _scw_diag_kernel, _adarda_kernel,
)


# ═══════════════════════════════════════════════════════════════════════
#   DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_kaggle_data(file_path):
    """Load Kaggle CreditFraud parquet for pure online learning.

    Temporal 80/20 split.  Returns (X_train, y_train, X_test, y_test)
    with labels in {-1, +1}.
    """
    df = pd.read_parquet(file_path)

    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])

    if 'Time' in df.columns:
        df = df.sort_values('Time').reset_index(drop=True)

    split_idx = int(0.8 * len(df))
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()

    drop_cols = ['Time', 'timestamp', 'user_id', 'label', 'Class']
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_data[feat_cols].values.astype(np.float64)
    X_test = test_data[feat_cols].values.astype(np.float64)

    label_col = 'Class' if 'Class' in df.columns else 'label'
    y_train = np.where(train_data[label_col].values == 0, -1, 1)
    y_test = np.where(test_data[label_col].values == 0, -1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"  Train fraud ratio: {(y_train==1).sum()/len(y_train):.4f}")
    print(f"  Test fraud ratio: {(y_test==1).sum()/len(y_test):.4f}")

    return X_train, y_train, X_test, y_test


def load_online_dataset(file_path):
    """Load a cybersecurity (online) parquet with temporal 80/20 split.

    Same splitting strategy as load_kaggle_data:
      - Sort rows by timestamp column
      - First 80 % → training, last 20 % → test
      - StandardScaler fitted on training set only

    Drops non-feature columns: timestamp, entity, user_id, label.
    Returns (X_train, y_train, X_test, y_test) with labels in {-1, +1}.
    """
    df = pd.read_parquet(file_path)

    # Identify and sort by time column
    time_col = ('timestamp' if 'timestamp' in df.columns
                else 'Time' if 'Time' in df.columns else None)
    if time_col is None:
        raise ValueError(f"No time column found in {os.path.basename(file_path)}")
    df = df.sort_values(time_col).reset_index(drop=True)

    split_idx = int(0.8 * len(df))
    train_data = df.iloc[:split_idx].copy()
    test_data  = df.iloc[split_idx:].copy()

    drop_cols  = {time_col, 'entity', 'user_id', 'label', 'Class'}
    feat_cols  = [c for c in df.columns if c not in drop_cols]

    X_train = train_data[feat_cols].values.astype(np.float64)
    X_test  = test_data[feat_cols].values.astype(np.float64)

    label_col = 'Class' if 'Class' in df.columns else 'label'
    y_train = np.where(train_data[label_col].values == 0, -1, 1)
    y_test  = np.where(test_data[label_col].values  == 0, -1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    print(f"  Train attack ratio: {(y_train==1).sum()/len(y_train):.4f}")
    print(f"  Test  attack ratio: {(y_test==1).sum()/len(y_test):.4f}")

    return X_train, y_train, X_test, y_test


def load_unsw_data(data_dir):
    """Load UNSW_NB15 from separate train/test parquets, scale features.

    Expects data_dir/UNSW_NB15_train.parquet and
    data_dir/UNSW_NB15_test.parquet.

    Returns (X_train, y_train, X_test, y_test) with labels in {-1, +1}.
    """
    train_path = os.path.join(data_dir, 'UNSW_NB15_train.parquet')
    test_path = os.path.join(data_dir, 'UNSW_NB15_test.parquet')
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


# ═══════════════════════════════════════════════════════════════════════
#   ONLINE PASS HELPER
# ═══════════════════════════════════════════════════════════════════════

def _online_pass(algo_name, X, y, w):
    """Run a single online pass using pre-trained weights.

    Each sample is predicted *before* the model updates on it
    (predict-then-learn).

    Parameters
    ----------
    algo_name : str     — algorithm name (PA, Perceptron, OGC, etc.)
    X         : ndarray — features (n, d)
    y         : ndarray — labels in {-1, +1}
    w         : ndarray — warm-started weight vector

    Returns
    -------
    y_pred : ndarray — predictions on each sample
    w      : ndarray — weight vector after the pass
    """
    X_np = np.ascontiguousarray(X, dtype=np.float64)
    y_np = np.ascontiguousarray(y, dtype=np.float64)
    w = w.copy()
    n = X_np.shape[0]
    perm = np.arange(n, dtype=np.int64)

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


# ═══════════════════════════════════════════════════════════════════════
#   EXPERIMENT — BATCH BENCHMARK  (Kaggle Fraud + UNSW-NB15)
# ═══════════════════════════════════════════════════════════════════════

def run_batch_benchmark(data_dir, output_dir=None):
    """Run 7 online algorithms on Kaggle Credit Fraud and UNSW-NB15.

    Kaggle  : temporal split, single online pass, final-weight test.
    UNSW-NB15: separate train/test parquets, 10-epoch training,
               followed by an online test pass (predict-then-learn).
    """
    warmup_jit()

    single_pass_algos = {
        'PA':        lambda X, y: PA(X, y),
        'Perceptron':lambda X, y: Perceptron(X, y),
        'OGC':       lambda X, y: OGC(X, y),
        'AROW':      lambda X, y: AROW(X, y, r=0.1),
        'RDA':       lambda X, y: RDA(X, y, lambda_param=1, gamma_param=1),
        'SCW':       lambda X, y: SCW(X, y, C=1, eta=0.5),
        'AdaRDA':    lambda X, y: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1),
    }

    multi_epoch_algos = {
        'PA':        lambda X, y, Xv, yv: PA(X, y, max_epochs=10, X_val=Xv, y_val=yv),
        'Perceptron':lambda X, y, Xv, yv: Perceptron(X, y, max_epochs=10, X_val=Xv, y_val=yv),
        'OGC':       lambda X, y, Xv, yv: OGC(X, y, max_epochs=10, X_val=Xv, y_val=yv),
        'AROW':      lambda X, y, Xv, yv: AROW(X, y, r=0.1, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
        'RDA':       lambda X, y, Xv, yv: RDA(X, y, lambda_param=1, gamma_param=1, max_epochs=10, X_val=Xv, y_val=yv),
        'SCW':       lambda X, y, Xv, yv: SCW(X, y, C=1, eta=0.5, max_epochs=10, X_val=Xv, y_val=yv, diagonal_sigma=True),
        'AdaRDA':    lambda X, y, Xv, yv: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1, max_epochs=10, X_val=Xv, y_val=yv),
    }

    all_results = []

    # ---- Kaggle Credit Fraud ----
    kaggle_files = sorted(
        f for f in glob.glob(os.path.join(data_dir, '*.parquet'))
        if any(k in os.path.basename(f).upper() for k in ('KAGGLE', 'CREDITFRAUD'))
    )
    for fpath in kaggle_files:
        ds_name = os.path.basename(fpath).replace('.parquet', '')
        print("\n" + "=" * 60)
        print(f"Dataset: {ds_name} (Kaggle — temporal split, final-weight test)")
        print("=" * 60)
        X_tr, y_tr, X_te, y_te = load_kaggle_data(fpath)

        for aname, afn in single_pass_algos.items():
            print(f"  Running {aname} ...")
            try:
                y_pred_train, w = afn(X_tr, y_tr)
                test_scores = X_te.dot(w)
                y_pred_test = np.sign(test_scores)
                y_pred_test = np.where(y_pred_test == 0, 1, y_pred_test)

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

    # ---- UNSW-NB15 (separate train/test parquets) ----
    unsw_train_path = os.path.join(data_dir, 'UNSW_NB15_train.parquet')
    unsw_test_path = os.path.join(data_dir, 'UNSW_NB15_test.parquet')
    if os.path.exists(unsw_train_path) and os.path.exists(unsw_test_path):
        ds_name = 'UNSW_NB15'
        print("\n" + "=" * 60)
        print(f"Dataset: {ds_name} (train/test split, 10-epoch + online test)")
        print("=" * 60)
        X_tr, y_tr, X_te, y_te = load_unsw_data(data_dir)
        for aname, afn in multi_epoch_algos.items():
            print(f"  Running {aname} (10 epochs + online test) ...")
            try:
                _, w = afn(X_tr, y_tr, X_te, y_te)
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

    # ---- MIT-BIH Arrhythmia (separate train/test parquets, 10-epoch + online test) ----
    mitbih_train_path = os.path.join(data_dir, 'MITBIH_Arrhythmia_train.parquet')
    mitbih_test_path  = os.path.join(data_dir, 'MITBIH_Arrhythmia_test.parquet')
    if os.path.exists(mitbih_train_path) and os.path.exists(mitbih_test_path):
        ds_name = 'MITBIH_Arrhythmia'
        print("\n" + "=" * 60)
        print(f"Dataset: {ds_name} (train/test split, 10-epoch + online test)")
        print("=" * 60)
        _tr = pd.read_parquet(mitbih_train_path)
        _te = pd.read_parquet(mitbih_test_path)
        feat_cols = [c for c in _tr.columns if c != 'label']
        X_tr_m = _tr[feat_cols].values.astype(np.float64)
        y_tr_m = np.where(_tr['label'].values == 0, -1.0, 1.0)
        X_te_m = _te[feat_cols].values.astype(np.float64)
        y_te_m = np.where(_te['label'].values == 0, -1.0, 1.0)
        scaler_m = StandardScaler()
        X_tr_m = scaler_m.fit_transform(X_tr_m)
        X_te_m = scaler_m.transform(X_te_m)
        n_pos_tr = int((y_tr_m == 1).sum())
        n_neg_tr = int((y_tr_m == -1).sum())
        n_pos_te = int((y_te_m == 1).sum())
        n_neg_te = int((y_te_m == -1).sum())
        print(f"  Train: {len(y_tr_m):,} samples | Arrhythmia: {n_pos_tr:,} ({100*n_pos_tr/len(y_tr_m):.1f}%)  "
              f"Normal: {n_neg_tr:,} ({100*n_neg_tr/len(y_tr_m):.1f}%)")
        print(f"  Test:  {len(y_te_m):,} samples | Arrhythmia: {n_pos_te:,} ({100*n_pos_te/len(y_te_m):.1f}%)  "
              f"Normal: {n_neg_te:,} ({100*n_neg_te/len(y_te_m):.1f}%)")
        for aname, afn in multi_epoch_algos.items():
            print(f"  Running {aname} (10 epochs + online test) ...")
            try:
                _, w = afn(X_tr_m, y_tr_m, X_te_m, y_te_m)
                yp, _ = _online_pass(aname, X_te_m, y_te_m, w)
                p, r, fnr, fpr, f1 = calculate_class1_metrics(y_te_m, yp)
                print(f"    Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")
                all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                    'Precision': p, 'Recall': r, 'F1': f1,
                                    'FNR': fnr, 'FPR': fpr})
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({'Dataset': ds_name, 'Algorithm': aname,
                                    'Precision': np.nan, 'Recall': np.nan,
                                    'F1': np.nan, 'FNR': np.nan, 'FPR': np.nan})

    # ---- Online Cybersecurity Datasets (temporal 80/20 split, same as Kaggle) ----
    _EXCL_KEYS = ('UNSW', 'KAGGLE', 'CREDITFRAUD', 'MITBIH', 'MNIST', 'CIFAR')
    online_files = sorted(
        f for f in glob.glob(os.path.join(data_dir, '*.parquet'))
        if not any(k in os.path.basename(f).upper() for k in _EXCL_KEYS)
    )
    for fpath in online_files:
        ds_name = os.path.basename(fpath).replace('.parquet', '')
        print("\n" + "=" * 60)
        print(f"Dataset: {ds_name} (temporal 80/20 split, final-weight test)")
        print("=" * 60)
        try:
            X_tr, y_tr, X_te, y_te = load_online_dataset(fpath)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        for aname, afn in single_pass_algos.items():
            print(f"  Running {aname} ...")
            try:
                y_pred_train, w = afn(X_tr, y_tr)
                test_scores = X_te.dot(w)
                y_pred_test = np.sign(test_scores)
                y_pred_test = np.where(y_pred_test == 0, 1, y_pred_test)

                y_te_01   = np.where(y_te == -1, 0, 1)
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

    df = pd.DataFrame(all_results)
    if len(df):
        df.sort_values(['Algorithm', 'Dataset'], inplace=True)
        print("\n" + "=" * 60)
        print("BATCH BENCHMARK RESULTS")
        print("=" * 60)
        print(df.round(4).to_string())
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'batch_benchmark_results.csv')
            df.to_csv(path, index=False, float_format='%.4f')
            print(f"\nSaved to: {path}")
    return df
