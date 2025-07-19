import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import glob
from data_handler import prepare_data_from_zenodo
# =============================================================================
# 1. ALGORITHM FUNCTIONS (Copied exactly as provided)
# =============================================================================

def AP(X, y):
    """Passive-Aggressive function that returns a stream of predictions."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = np.zeros([len(data), len(data.columns)])
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))
    
    for i in range(len(data)):
        x = data.iloc[i, :]
        y_pred[i] = np.sign(x.dot(w))
        y_actual = y.iloc[i]
        loss = max(0, 1 - y_actual * y_pred[i])
      
        if loss > 0:
            l2_norm_sq_sq = (x.dot(x))**2
            if l2_norm_sq_sq > 0:
                eta = loss / l2_norm_sq_sq
                w += eta * y_actual * x

        weight_history[i, :] = w
        
    return y_pred, weight_history

def PERCEPT(X, y):
    """Perceptron function that returns a stream of predictions."""
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

def OGL(X, y):
    """Online Gradient Learning function that returns a stream of predictions."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = np.zeros([len(data), len(data.columns)])
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))

    for i in range(len(data)):
        x = data.iloc[i, :]
        y_pred[i] = np.sign(x.dot(w))
        y_act = y.iloc[i]
        w = w + (y_act - np.sign(x.dot(w))) / (np.sqrt(x.dot(x)) + 1e-8) * x
        weight_history[i, :] = w
        
    return y_pred, weight_history

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_and_process_data(file_path):
    """Loads and preprocesses a single parquet file."""
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
    """Calculates Precision, TPR, and FPR for the positive class (1)."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError:
        return np.nan, np.nan, np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return precision, tpr, fpr

# =============================================================================
# 3. BATCH EXPERIMENT RUNNER
# =============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'  # This is the target folder for our data
    OUTPUT_CSV_FILE = 'results/online_results.csv'
    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)



    if data_ready:
    
        algorithms_to_run = {
            "AP": AP,
            "PERCEPT": PERCEPT,
            "OGL": OGL
        }
        
        # --- SCRIPT START ---
        search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
        all_data_files = sorted(glob.glob(search_path))

        if not all_data_files:
            print(f"FATAL: No .parquet files found in '{DATA_DIRECTORY}'. Please check the path.")
        else:
            print(f"Found {len(all_data_files)} datasets. Starting batch evaluation...")
            
            all_results = []
            
            for i, file_path in enumerate(all_data_files):
                dataset_name = os.path.basename(file_path).replace('.parquet', '')
                print("\n" + "="*60)
                print(f"Processing Dataset {i+1}/{len(all_data_files)}: {dataset_name}")
                print("="*60)
                
                X, y_true = load_and_process_data(file_path)
                
                if X is None or y_true is None or len(X) <= 1:
                    print(f"  - Skipping dataset due to insufficient data.")
                    continue
                
                for algo_name, algo_func in algorithms_to_run.items():
                    print(f"  - Running algorithm: {algo_name}...")
                    
                    # --- THIS IS THE CORE LOGIC FROM YOUR EXAMPLE ---
                    
                    # 1. Call the function to get the full stream of predictions
                    y_pred_stream, _ = algo_func(X, y_true)
                    
                    # 2. Slice both y_true and y_pred to exclude the first sample
                    y_true_eval = y_true[1:]
                    y_pred_eval = y_pred_stream[1:]
                    
                    # 3. Calculate metrics on the sliced data
                    precision, tpr, fpr = calculate_class1_metrics(y_true_eval, y_pred_eval)
                    
                    # --------------------------------------------------
                    
                    all_results.append({
                        'Dataset': dataset_name,
                        'Algorithm': algo_name,
                        'Precision': precision,
                        'TPR': tpr,
                        'FPR': fpr
                    })
                    
            if all_results:
                final_df = pd.DataFrame(all_results)
                
                print("\n" + "="*60)
                print("BATCH EVALUATION COMPLETE. FINAL COMBINED RESULTS:")
                print("="*60)
                print(final_df.round(4).to_string())
                
                try:
                    final_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
                    print(f"\nSUCCESS: All combined results saved to '{OUTPUT_CSV_FILE}'")
                except Exception as e:
                    print(f"\nERROR: Could not save results to CSV. Reason: {e}")
            else:
                print("\nNo simulations were completed successfully. No output file generated.")