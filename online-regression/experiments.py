#!/usr/bin/env python3
"""
experiments.py — Data utilities and experiment runners.

Two experiment modes
--------------------
1. run_delayed_feedback()  — UCI Household Power (variable + fixed delays)
                             All 4 algorithms: CIRR, K-CIRR, Q-CIRR, QK-CIRR
2. run_batched_feedback()  — NSL-KDD (varying pack/batch sizes)
                             All 4 algorithms
"""

import collections
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from algos import ALL_ALGORITHMS, KERNEL_ALGORITHMS, QUANTUM_ALGORITHMS, warmup_jit

# ══════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════

def ensure_reproducibility(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA UTILITIES
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
    print("Downloading UCI Household Power Consumption dataset...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    print(f"  Saved to {txt_path}")
    return txt_path


def load_household_power(max_samples=None):
    """Load UCI Household Power Consumption (chronological stream).

    Returns (X, y) float64 arrays, standardised.
    Features: reactive_power, voltage, intensity, sub_metering 1-3.
    Target:   global_active_power (standardised).
    """
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
    y = (y - y.mean()) / (y.std() + 1e-8)

    print(f"  Household Power: {X.shape[0]:,} samples, {X.shape[1]} features")
    return X, y


def _download_nslkdd():
    urls = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ]
    csv_path = DATA_DIR / "nslkdd_train.csv"
    if csv_path.exists():
        return csv_path
    print("Downloading NSL-KDD (KDDTrain+.TXT)...")
    for url in urls:
        try:
            urllib.request.urlretrieve(url, csv_path)
            print(f"  Downloaded from {url}")
            return csv_path
        except Exception as e:
            print(f"  Failed from {url}: {e}")
    raise RuntimeError("NSL-KDD download failed from all sources.")


def load_nslkdd(max_samples=None):
    """Load NSL-KDD (KDDTrain+) as a regression stream.

    Target: log1p(src_bytes) — continuous, standardised.
    Features: 24 numeric columns.
    """
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
    y = (y - y.mean()) / (y.std() + 1e-8)

    print(f"  NSL-KDD: {X.shape[0]:,} samples, {X.shape[1]} features")
    return X, y


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — PLOT UTILITIES
# ══════════════════════════════════════════════════════════════════════

_PALETTE = {
    'CIRR':    '#1f77b4',
    'K-CIRR':  '#ff7f0e',
    'Q-CIRR':  '#2ca02c',
    'QK-CIRR': '#d62728',
}
_LINESTYLE = {
    'CIRR':    '-',
    'K-CIRR':  '--',
    'Q-CIRR':  '-.',
    'QK-CIRR': ':',
}


def _plot_cumulative(curves, title, ylabel, output_path, xlabel='Iteration', log_log=False):
    """Plot cumulative curves for all algorithms on one figure and save as PNG.

    Each algorithm gets its own line (consistent colour + dash style).
    x-axis is iteration number; y-axis is the cumulative quantity (time or loss).

    Parameters
    ----------
    curves      : dict  algo_name -> 1-D ndarray of cumulative values
    title       : str
    ylabel      : str
    output_path : str   destination PNG path
    xlabel      : str   (default 'Iteration')
    log_log     : bool  if True both axes use log scale (default False)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo_name, vals in curves.items():
        iterations = np.arange(1, len(vals) + 1)
        ax.plot(
            iterations, vals,
            label=algo_name,
            color=_PALETTE.get(algo_name, None),
            linestyle=_LINESTYLE.get(algo_name, '-'),
            linewidth=1.6,
        )
    if log_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {output_path}")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — PREQUENTIAL RUNNERS
#
#  Each runner returns a dict with:
#    predictions     : (T,) array
#    squared_losses  : (T,) array
#    cumulative_loss : (T,) array
#    performance     : {'RMSE': float, 'R2': float}
# ══════════════════════════════════════════════════════════════════════

def _make_algo(algo_cls, n_features, total_samples, algo_kwargs):
    """Instantiate an algorithm, passing only supported kwargs."""
    return algo_cls(n_features=n_features, total_samples=total_samples, **algo_kwargs)


def run_immediate(algo_cls, X, y, algo_kwargs=None):
    """Prequential run with immediate feedback (baseline)."""
    if algo_kwargs is None:
        algo_kwargs = {}
    n, d = X.shape
    algo = _make_algo(algo_cls, d, n, algo_kwargs)

    predictions = np.zeros(n)
    sq_losses = np.zeros(n)

    for t in range(n):
        predictions[t] = algo.predict(X[t])
        algo.update(X[t], y[t])
        sq_losses[t] = (predictions[t] - y[t]) ** 2

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    return {
        'predictions': predictions,
        'squared_losses': sq_losses,
        'cumulative_loss': np.cumsum(sq_losses),
        'performance': {'RMSE': rmse, 'R2': r2},
    }


def run_fixed_delay(algo_cls, X, y, delay, algo_kwargs=None):
    """Prequential run with a fixed delay d: feedback from round t arrives at t+d."""
    if algo_kwargs is None:
        algo_kwargs = {}
    n, d_feat = X.shape
    algo = _make_algo(algo_cls, d_feat, n, algo_kwargs)

    predictions = np.zeros(n)
    sq_losses = np.zeros(n)
    buffer = collections.deque()   # (x, y) pairs waiting to be processed

    for t in range(n):
        # Process feedback that has arrived (delay rounds ago)
        if len(buffer) > delay:
            x_old, y_old = buffer.popleft()
            algo.update(x_old, y_old)

        predictions[t] = algo.predict(X[t])
        buffer.append((X[t], y[t]))
        sq_losses[t] = (predictions[t] - y[t]) ** 2

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    return {
        'predictions': predictions,
        'squared_losses': sq_losses,
        'cumulative_loss': np.cumsum(sq_losses),
        'performance': {'RMSE': rmse, 'R2': r2},
    }


def run_variable_delay(algo_cls, X, y, delays, algo_kwargs=None):
    """Prequential run with per-sample variable delays.

    delays[t] gives the number of extra rounds before y[t] is revealed.
    """
    if algo_kwargs is None:
        algo_kwargs = {}
    n, d_feat = X.shape
    algo = _make_algo(algo_cls, d_feat, n, algo_kwargs)

    predictions = np.zeros(n)
    sq_losses = np.zeros(n)
    pending = []   # list of (arrival_round, x, y)

    for t in range(n):
        # Process all feedback that has arrived by round t
        still_pending = []
        for (arrival, xp, yp) in pending:
            if arrival <= t:
                algo.update(xp, yp)
            else:
                still_pending.append((arrival, xp, yp))
        pending = still_pending

        predictions[t] = algo.predict(X[t])
        pending.append((t + int(delays[t]) + 1, X[t], y[t]))
        sq_losses[t] = (predictions[t] - y[t]) ** 2

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    return {
        'predictions': predictions,
        'squared_losses': sq_losses,
        'cumulative_loss': np.cumsum(sq_losses),
        'performance': {'RMSE': rmse, 'R2': r2},
    }


def run_batched(algo_cls, X, y, pack_size, algo_kwargs=None):
    """Prequential run with batched (pack) feedback.

    Model is updated only after every pack_size predictions.
    Equivalent to a structured delay of pack_size − 1 rounds.
    """
    if algo_kwargs is None:
        algo_kwargs = {}
    n, d_feat = X.shape
    algo = _make_algo(algo_cls, d_feat, n, algo_kwargs)

    predictions = np.zeros(n)
    sq_losses = np.zeros(n)
    batch_buf = []

    for t in range(n):
        predictions[t] = algo.predict(X[t])
        batch_buf.append((X[t], y[t]))

        if len(batch_buf) >= pack_size:
            for xb, yb in batch_buf:
                algo.update(xb, yb)
            batch_buf = []

        sq_losses[t] = (predictions[t] - y[t]) ** 2

    # Flush remaining buffer
    for xb, yb in batch_buf:
        algo.update(xb, yb)

    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    return {
        'predictions': predictions,
        'squared_losses': sq_losses,
        'cumulative_loss': np.cumsum(sq_losses),
        'performance': {'RMSE': rmse, 'R2': r2},
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — HIGH-LEVEL EXPERIMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _algo_kwargs(name):
    """Return default kwargs for each algorithm (excluding n_features / total_samples).
    gamma is intentionally omitted for kernel algorithms — the auto-scale
    rule 1/(2*n_features) is applied inside __init__ for standardised data.
    """
    if name in ("K-CIRR",):
        return {'n_components': 50}
    if name in ("Q-CIRR",):
        return {'epsilon': 0.01}
    if name in ("QK-CIRR",):
        return {'epsilon': 0.01, 'n_components': 50}
    return {}


def run_delayed_feedback(output_dir, max_samples=50_000):
    """EXP1: Delayed feedback on UCI Household Power.

    Compares CIRR, K-CIRR, Q-CIRR, QK-CIRR under:
      · Variable exponential delays (smart-grid simulation)
      · Fixed delays d ∈ {0, 5, 10, 20, 50}

    Records RMSE, R², and wall-clock times.
    """
    print("\n" + "=" * 70)
    print("EXP1: DELAYED FEEDBACK — UCI Household Power")
    print("=" * 70)

    X, y = load_household_power(max_samples=max_samples)
    n = len(y)

    np.random.seed(42)
    delays = np.clip(np.random.exponential(scale=5, size=n).astype(int), 0, 30)
    print(f"  Variable delay range: {delays.min()}–{delays.max()}  (mean {delays.mean():.1f})")

    rows = []
    cumloss_curves = {}   # algo_name -> cumulative squared loss, variable-delay run

    for algo_name, algo_cls in ALL_ALGORITHMS.items():
        kwargs = _algo_kwargs(algo_name)
        print(f"\n  [{algo_name}]")

        res = run_variable_delay(algo_cls, X, y, delays, algo_kwargs=kwargs)
        p = res['performance']
        print(f"    Variable delay  RMSE={p['RMSE']:.4f}  R²={p['R2']:.4f}")
        rows.append({
            'Algorithm': algo_name, 'Experiment': 'delayed',
            'Setting': 'variable', 'Delay': 'variable',
            'RMSE': p['RMSE'], 'R2': p['R2'],
        })
        cumloss_curves[algo_name] = res['cumulative_loss']

        # Fixed delays
        for d in [0, 5, 10, 20, 50]:
            res = run_fixed_delay(algo_cls, X, y, d, algo_kwargs=kwargs)
            p = res['performance']
            print(f"    Fixed delay={d:2d}  RMSE={p['RMSE']:.4f}  R²={p['R2']:.4f}")
            rows.append({
                'Algorithm': algo_name, 'Experiment': 'delayed',
                'Setting': 'fixed', 'Delay': d,
                'RMSE': p['RMSE'], 'R2': p['R2'],
            })

    _plot_cumulative(
        cumloss_curves,
        title='EXP1: Cumulative Squared Loss — Variable Delay',
        ylabel='Cumulative squared loss',
        output_path=os.path.join(output_dir, 'exp1_cumloss_variable.png'),
    )

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, 'exp1_delayed_results.csv')
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df, cumloss_curves


def run_batched_feedback(output_dir, max_samples=50_000):
    """EXP2: Batched (pack) feedback on NSL-KDD.

    Compares CIRR, K-CIRR, Q-CIRR, QK-CIRR under pack sizes ∈ {1, 5, 10, 20, 50}.
    """
    print("\n" + "=" * 70)
    print("EXP2: BATCHED FEEDBACK — NSL-KDD")
    print("=" * 70)

    X, y = load_nslkdd(max_samples=max_samples)

    rows = []
    pack_sizes = [1, 5, 10, 20, 50]
    cumloss_curves = {}   # algo_name -> cumulative squared loss, pack=1 run

    for algo_name, algo_cls in ALL_ALGORITHMS.items():
        kwargs = _algo_kwargs(algo_name)
        print(f"\n  [{algo_name}]")

        for pack_size in pack_sizes:
            res = run_batched(algo_cls, X, y, pack_size, algo_kwargs=kwargs)
            p = res['performance']
            print(f"    Pack={pack_size:2d}  RMSE={p['RMSE']:.4f}  R²={p['R2']:.4f}")
            rows.append({
                'Algorithm': algo_name, 'Experiment': 'batched',
                'Pack_size': pack_size,
                'RMSE': p['RMSE'], 'R2': p['R2'],
            })
            if pack_size == 1:
                cumloss_curves[algo_name] = res['cumulative_loss']

    _plot_cumulative(
        cumloss_curves,
        title='EXP2: Cumulative Squared Loss — Batched Feedback (pack=1)',
        ylabel='Cumulative squared loss',
        output_path=os.path.join(output_dir, 'exp2_cumloss_pack1.png'),
    )

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, 'exp2_batched_results.csv')
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df, cumloss_curves


def run_corollary_test(output_dir, n_samples=5_000, seed=42):
    """EXP3 — Corollary test: Concentrated vs Perpetual delay (same total D_T).

    Illustrates Corollary (concurrent-staleness):
      Case 1 — Concentrated delay: d_t=100 for first 500 rounds, d_t=0 after.
               Algorithm suffers initially but fully recovers.
      Case 2 — Perpetual delay: d_t=10 for all rounds.
               Algorithm is perpetually 10 steps stale; never recovers.
    Both cases have identical total D_T = 500 * 100 = 5 * n_samples = 50,000.
    """
    from algos import CIRR

    print("\n" + "=" * 70)
    print("EXP3: COROLLARY TEST — Concentrated vs Perpetual Delay")
    print("=" * 70)

    np.random.seed(seed)
    d = 5
    X = np.random.randn(n_samples, d)
    w_star = np.random.randn(d)
    y = X @ w_star + np.random.randn(n_samples) * 0.1

    scaler_x = np.std(X, axis=0)
    scaler_x[scaler_x == 0] = 1.0
    X = (X - X.mean(axis=0)) / scaler_x
    y = (y - y.mean()) / (y.std() + 1e-8)

    # Case 1: concentrated — first NUM_CONC rounds have delay CONC_D, rest 0
    NUM_CONC = 500
    CONC_D   = 100
    PERP_D   = 10
    D_T = NUM_CONC * CONC_D          # 50 000

    delays_case1 = np.zeros(n_samples, dtype=int)
    delays_case1[:NUM_CONC] = CONC_D

    delays_case2 = np.full(n_samples, PERP_D, dtype=int)

    kwargs = {}
    res1 = run_variable_delay(CIRR, X, y, delays_case1, algo_kwargs=kwargs)
    res2 = run_variable_delay(CIRR, X, y, delays_case2, algo_kwargs=kwargs)

    p1, p2 = res1['performance'], res2['performance']
    print(f"  Case 1 (concentrated, D_T={D_T:,}):  RMSE={p1['RMSE']:.4f}  R²={p1['R2']:.4f}"
          f"  final_cumloss={res1['cumulative_loss'][-1]:.2f}")
    print(f"  Case 2 (perpetual d={PERP_D}, D_T={D_T:,}): RMSE={p2['RMSE']:.4f}  R²={p2['R2']:.4f}"
          f"  final_cumloss={res2['cumulative_loss'][-1]:.2f}")

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = np.arange(1, n_samples + 1)
    ax.plot(iters, res1['cumulative_loss'],
            label=f'Case 1: Concentrated ($d_t={CONC_D}$ for $t\\leq{NUM_CONC}$, then 0)',
            color='#1f77b4', linestyle='-', linewidth=1.6)
    ax.plot(iters, res2['cumulative_loss'],
            label=f'Case 2: Perpetual ($d_t={PERP_D}$ for all $t$)',
            color='#d62728', linestyle='--', linewidth=1.6)
    ax.axvline(NUM_CONC + CONC_D, color='grey', linestyle=':', linewidth=1.0,
               label=f'Case 1 recovery (t={NUM_CONC + CONC_D})')
    ax.set_title(f'EXP3: Corollary Test — Same Total Delay $D_T = {D_T:,}$', fontsize=13)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cumulative squared loss', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'exp3_corollary.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot -> {plot_path}")

    # --- Save CSV ------------------------------------------------------------
    df = pd.DataFrame([
        {'Scenario': f'Case 1: Concentrated ({NUM_CONC} x {CONC_D})',
         'Total_D_T': D_T, 'Final_cumloss': float(res1['cumulative_loss'][-1]),
         'RMSE': p1['RMSE'], 'R2': p1['R2']},
        {'Scenario': f'Case 2: Perpetual (d_t={PERP_D} for all t)',
         'Total_D_T': D_T, 'Final_cumloss': float(res2['cumulative_loss'][-1]),
         'RMSE': p2['RMSE'], 'R2': p2['R2']},
    ])
    csv_path = os.path.join(output_dir, 'exp3_corollary_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")
    return df, res1['cumulative_loss'], res2['cumulative_loss']


def make_combined_plot(output_dir, cumloss_exp1, cumloss_exp2,
                       cor_case1, cor_case2, n_cor=5_000):
    """Build the three-panel combined figure for the paper.

    Parameters
    ----------
    cumloss_exp1 : dict  algo -> cumloss array  (EXP1 variable delay)
    cumloss_exp2 : dict  algo -> cumloss array  (EXP2 pack=1)
    cor_case1    : 1-D array  (EXP3 concentrated)
    cor_case2    : 1-D array  (EXP3 perpetual)
    n_cor        : int        (length of corollary stream)
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel 1: EXP1 -------------------------------------------------------
    ax = axes[0]
    for name, vals in cumloss_exp1.items():
        iters = np.arange(1, len(vals) + 1)
        ax.plot(iters, vals, label=name,
                color=_PALETTE.get(name), linestyle=_LINESTYLE.get(name, '-'),
                linewidth=1.5)
    ax.set_title('(a) EXP1: Delayed Feedback\n(Variable Delay, Power)', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Cumulative squared loss', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)

    # --- Panel 2: EXP2 -------------------------------------------------------
    ax = axes[1]
    for name, vals in cumloss_exp2.items():
        iters = np.arange(1, len(vals) + 1)
        ax.plot(iters, vals, label=name,
                color=_PALETTE.get(name), linestyle=_LINESTYLE.get(name, '-'),
                linewidth=1.5)
    ax.set_title('(b) EXP2: Batched Feedback\n(Pack=1, Cybersecurity)', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Cumulative squared loss', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)

    # --- Panel 3: Corollary --------------------------------------------------
    ax = axes[2]
    iters = np.arange(1, n_cor + 1)
    ax.plot(iters, cor_case1, label='Case 1: Concentrated ($d_t=100$, $t\\leq500$)',
            color='#1f77b4', linestyle='-', linewidth=1.5)
    ax.plot(iters, cor_case2, label='Case 2: Perpetual ($d_t=10$)',
            color='#d62728', linestyle='--', linewidth=1.5)
    NUM_CONC, CONC_D = 500, 100
    ax.axvline(NUM_CONC + CONC_D, color='grey', linestyle=':', linewidth=0.9,
               label=f'Case 1 recovery ($t={NUM_CONC + CONC_D}$)')
    ax.set_title('(c) EXP3: Corollary Test\n(Same Total Delay $D_T=50{,}000$)', fontsize=10)
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Cumulative squared loss', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle('Simulation of Delay Dynamics', fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(output_dir, 'plot_combined.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Combined plot -> {out}")
    return out


def print_summary(df1, df2):
    """Print formatted summary tables for EXP1 and EXP2."""
    print("\n" + "=" * 60)
    print("SUMMARY — EXP1: Delayed Feedback (variable delay)")
    print("=" * 60)
    sub = df1[df1['Setting'] == 'variable'][['Algorithm', 'RMSE', 'R2']]
    print(sub.to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY — EXP1: Delayed Feedback (fixed d=10)")
    print("=" * 60)
    sub = df1[(df1['Setting'] == 'fixed') & (df1['Delay'] == 10)][
        ['Algorithm', 'RMSE', 'R2']]
    print(sub.to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY — EXP2: Batched Feedback (pack=10)")
    print("=" * 60)
    sub = df2[df2['Pack_size'] == 10][['Algorithm', 'RMSE', 'R2']]
    print(sub.to_string(index=False))
