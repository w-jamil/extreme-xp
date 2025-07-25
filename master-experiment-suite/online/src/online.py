import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import glob
from scipy.stats import norm
# This is a placeholder for your actual data handler
from data_handler import prepare_data_from_zenodo 

# =============================================================================
# 1. ALGORITHM FUNCTIONS (WITH CORRECTIONS)
# =============================================================================

# AROW (Correct in original)
def AROW(X, y, r):
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape
    u = np.zeros(n_features)
    Sigma = np.identity(n_features)
    y_pred = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        x = X_np[i]
        y_actual = y_np[i]
        prediction_at_i = np.sign(x.dot(u))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        lt = max(0, 1 - y_actual * x.dot(u))
        vt = x.T.dot(Sigma).dot(x)
        if lt > 0:
            alpha_t = lt / (vt + r) if (vt + r) > 0 else 0.0
            beta_t = 1 / (vt + r) if (vt + r) > 0 else 0.0
            u += alpha_t * y_actual * Sigma.dot(x)
            Sigma -= beta_t * Sigma.dot(np.outer(x, x)).dot(Sigma)
        weight_history[i, :] = u
    return y_pred, weight_history

# --- CORRECTED AP ---
def AP(X, y):
    """Passive-Aggressive function that returns a stream of predictions."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = np.zeros([len(data), len(data.columns)])
    y_pred = np.zeros(len(data))
    w = np.zeros(len(data.columns))
    
    for i in range(len(data)):
        x = data.iloc[i, :]
        prediction_at_i = np.sign(x.dot(w))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        
        y_actual = y.iloc[i]
        loss = max(0, 1 - y_actual * x.dot(w))
      
        if loss > 0:
            l2_norm_sq = x.dot(x)
            if l2_norm_sq > 0:
                eta = loss / l2_norm_sq
                w += eta * y_actual * x

        weight_history[i, :] = w
        
    return y_pred, weight_history

# PERCEPT (Correct in original)
def PERCEPT(X, y):
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = np.zeros([len(data), len(data.columns)])
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))
    for i in range(len(data)):
        x = data.iloc[i, :]
        y_pred[i] = np.sign(x.dot(w))
        y_actual = y.iloc[i]
        if y_pred[i] != y_actual:
            w += y_actual * x
        weight_history[i, :] = w
    return y_pred, weight_history

# --- CORRECTED OGL (as Online Gradient Descent) ---
def OGL(X, y, eta=0.1):
    """Online Gradient Descent with Hinge Loss."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features)
    y_pred = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        x = X_np[i]
        y_actual = y_np[i]
        prediction_at_i = np.sign(x.dot(w))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        if y_actual * x.dot(w) < 1:
            gradient = -y_actual * x
            w -= eta * gradient
        weight_history[i, :] = w
    return y_pred, weight_history

# RDA (Correct in original)
def RDA(X, y, lambda_param=1, gamma_param=1):
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features)
    g = np.zeros(n_features)
    y_pred = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        t = i + 1
        x = X_np[i]
        y_actual = y_np[i]
        prediction_at_i = np.sign(x.dot(w))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        lt = max(0, 1 - y_actual * x.dot(w))
        if lt > 0:
            gt = -y_actual * x
        else:
            gt = np.zeros_like(x)
        g = ((t - 1) / t) * g + (1 / t) * gt
        update_mask = np.abs(g) > lambda_param
        w.fill(0)
        w[update_mask] = -(np.sqrt(t) / gamma_param) * \
                          (g[update_mask] - lambda_param * np.sign(g[update_mask]))
        weight_history[i, :] = w
    return y_pred, weight_history

# SCW (Unchanged from original, but be aware of its specific implementation)
def SCW(X, y, C=1, eta=0.5):
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape
    phi = norm.ppf(eta) 
    u = np.zeros(n_features)
    Sigma = np.identity(n_features)
    y_pred = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        x = X_np[i]
        y_actual = y_np[i]
        prediction_at_i = np.sign(x.dot(u))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        vt = x.T.dot(Sigma).dot(x)
        mt = y_actual * x.dot(u)
        lt = max(0, phi * np.sqrt(vt) - mt)
        if lt > 0:
            pa = 1 + (phi**2) / 2
            xi = 1 + phi**2
            sqrt_term = max(0, (mt**2 * phi**4 / 4) + (vt * phi**2 * xi))
            alpha_t = max(0, (1 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term)))
            alpha_t = min(C, alpha_t)
            sqrt_ut_term = max(0, (alpha_t**2 * vt**2 * phi**2) + (4 * vt))
            ut = 0.25 * (-alpha_t * vt * phi + np.sqrt(sqrt_ut_term))**2
            beta_t = (alpha_t * phi) / (np.sqrt(ut) + vt * alpha_t * phi + 1e-8)
            u += alpha_t * y_actual * Sigma.dot(x)
            Sigma -= beta_t * Sigma.dot(np.outer(x, x)).dot(Sigma)
        weight_history[i, :] = u
    return y_pred, weight_history

# --- CORRECTED AdaRDA ---
def AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1):
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features)
    g = np.zeros(n_features)
    g_sq_sum = np.zeros(n_features)
    y_pred = np.zeros(n_samples)
    weight_history = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        t = i + 1
        x = X_np[i]
        y_actual = y_np[i]
        prediction_at_i = np.sign(x.dot(w))
        y_pred[i] = prediction_at_i if prediction_at_i != 0 else 1
        lt = max(0, 1 - y_actual * x.dot(w))
        if lt > 0:
            gt = -y_actual * x
        else:
            gt = np.zeros_like(x)
        g = ((t - 1) / t) * g + (1 / t) * gt
        g_sq_sum += gt**2
        Ht = delta_param + np.sqrt(g_sq_sum)
        update_mask = np.abs(g) > lambda_param
        w.fill(0)
        w[update_mask] = - (eta_param * t / Ht[update_mask]) * \
                         (g[update_mask] - lambda_param * np.sign(g[update_mask]))
        weight_history[i, :] = w
    return y_pred, weight_history

# =============================================================================
# 2. HELPER & MAIN SCRIPT (Your code, unchanged)
# =============================================================================

def load_and_process_data(file_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  - ERROR: Could not read file '{file_path}'. Skipping. Reason: {e}")
        return None, None
    df = df.sort_values(['timestamp']).reset_index(drop=True)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'], axis=1)
    x_df = df.groupby(['timestamp']).sum()
    y_series = x_df["label"].map({0: -1, 1: 1, 2: 1, 3: 1, 4: 1}).fillna(-1)
    x_df = x_df.drop(columns=['label'], axis=1)
    return x_df, y_series.to_numpy()

def calculate_class1_metrics(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, fnr, fpr

if __name__ == "__main__":
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'
    OUTPUT_CSV_FILE = 'results/online_results.csv'
    
    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)

    if data_ready:
        # Note: I've updated GLC to OGL with a default eta
        algorithms_to_run = {
            "PA": lambda X, y: AP(X, y),
            "Perceptron": lambda X, y: PERCEPT(X, y),
            "GLC": lambda X, y: OGL(X, y, eta=0.1), # Renamed back to GLC for consistency with your code
            "AROW": lambda X, y: AROW(X, y, r=1.0),
            "RDA": lambda X, y: RDA(X, y, lambda_param=0.01, gamma_param=1.0),
            "SCW": lambda X, y: SCW(X, y, C=0.1, eta=0.95),
            "AdaRDA": lambda X, y: AdaRDA(X, y, lambda_param=0.01, eta_param=0.1, delta_param=1.0)
        }
        
        search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
        all_data_files = sorted(glob.glob(search_path))

        if not all_data_files:
            print(f"FATAL: No .parquet files found in '{DATA_DIRECTORY}'.")
        else:
            print(f"Found {len(all_data_files)} datasets. Starting batch evaluation...")
            all_results = []
            
            for i, file_path in enumerate(all_data_files):
                dataset_name = os.path.basename(file_path).replace('.parquet', '')
                print(f"\nProcessing Dataset {i+1}/{len(all_data_files)}: {dataset_name}")
                X, y_true = load_and_process_data(file_path)
                
                if X is None or y_true is None or len(X) <= 1:
                    print(f"  - Skipping dataset due to insufficient data.")
                    continue
                
                for algo_name, algo_func in algorithms_to_run.items():
                    print(f"  - Running algorithm: {algo_name}...")
                    try:
                        y_pred_stream, _ = algo_func(X, y_true)
                        y_true_eval = y_true[1:]
                        y_pred_eval = y_pred_stream[:-1]
                        precision, recall, fnr, fpr = calculate_class1_metrics(y_true_eval, y_pred_eval)
                        all_results.append({
                            'Dataset': dataset_name, 'Algorithm': algo_name,
                            'Precision': precision, 'Recall': recall, 'FNR': fnr, 'FPR': fpr
                        })
                    except Exception as e:
                        print(f"    ERROR running {algo_name} on {dataset_name}: {e}")
                        all_results.append({
                            'Dataset': dataset_name, 'Algorithm': algo_name,
                            'Precision': np.nan, 'Recall': np.nan, 'FNR': np.nan, 'FPR': np.nan
                        })

            if all_results:
                final_df = pd.DataFrame(all_results)
                final_df.sort_values(by=['Algorithm', 'Dataset'], inplace=True)
                print("\n" + "="*60 + "\nBATCH EVALUATION COMPLETE. FINAL COMBINED RESULTS:\n" + "="*60)
                print(final_df.round(4).to_string())
                os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
                final_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
                print(f"\nSUCCESS: All combined results saved to '{OUTPUT_CSV_FILE}'")