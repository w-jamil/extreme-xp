import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import glob
import requests
import zipfile
import io

# =============================================================================
# 1. HELPER FUNCTIONS AND ALGORITHM CLASSES
# =============================================================================

def prepare_data_from_zenodo(zenodo_archive_url, target_dir):
    """
    Checks for data locally; if not found, downloads and unzips it from a Zenodo archive link.
    """
    if os.path.exists(target_dir) and any(f.endswith('.parquet') for f in os.listdir(target_dir)):
        print(f"--> Data found locally in '{target_dir}'. Skipping download.")
        return True

    print(f"--> Local data not found. Starting download from Zenodo...")
    try:
        response = requests.get(zenodo_archive_url, stream=True)
        response.raise_for_status()
        print("--> Download complete. Unzipping relevant files from archive...")
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for member in z.infolist():
                if member.filename.endswith('.parquet') and not member.is_dir():
                    base_filename = os.path.basename(member.filename)
                    output_path = os.path.join(target_dir, base_filename)
                    with z.open(member) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
        print(f"--> Successfully extracted all .parquet files to '{target_dir}'")
        return True
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during data preparation: {e}")
        return False

class PassiveAggressive:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x): return np.sign(x @ self.weights)
    def partial_fit(self, x, y_true):
        if y_true * (x @ self.weights) < 1:
            l2_norm_sq = x @ x
            if l2_norm_sq > 0:
                eta = (1 - y_true * (x @ self.weights)) / l2_norm_sq
                self.weights += eta * y_true * x

class Perceptron:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x): return np.sign(x @ self.weights)
    def partial_fit(self, x, y_true):
        if self.predict(x) != y_true: self.weights += y_true * x

class GradientLearning:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x): return np.sign(x @ self.weights)
    def partial_fit(self, x, y_true):
        if y_true * (x @ self.weights) < 1: self.weights += y_true * x

def load_data_for_batch(file_path):
    """Loads and processes a single parquet file for batch learning."""
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  - ERROR: Could not read file '{file_path}'. Skipping. Reason: {e}")
        return None, None
    df = df.sort_values(['timestamp']).reset_index(drop=True)
    if 'user_id' in df.columns: df = df.drop(columns=['user_id'], axis=1)
    x_df = df.groupby(['timestamp']).sum()
    y_series = x_df["label"].map({0: -1, 1: 1, 2: 1, 3: 1, 4: 1}).fillna(-1)
    x_df = x_df.drop(columns=['label'], axis=1)
    return x_df.to_numpy(), y_series.to_numpy()

def calculate_class1_metrics(y_true, y_pred):
    """Calculates Precision, TPR, and FPR for the positive class (1)."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError: return np.nan, np.nan, np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, tpr, fpr

# =============================================================================
# 2. MAIN BATCH EXPERIMENT SCRIPT
# =============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'
    OUTPUT_CSV_FILE = 'results/batch_learning_results.csv'
    TRAIN_TEST_SPLIT_RATIO = 0.6

    # --- 1. PREPARE DATA ---
    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)

    if data_ready:
        # --- 2. RUN BATCH EXPERIMENT ---
        search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
        all_data_files = sorted(glob.glob(search_path))

        if not all_data_files:
            print(f"FATAL: No .parquet files found in '{DATA_DIRECTORY}' even after download attempt.")
        else:
            print(f"Found {len(all_data_files)} datasets. Starting batch learning evaluation...")
            
            all_results = []
            
            # Loop through each dataset file
            for i, file_path in enumerate(all_data_files):
                dataset_name = os.path.basename(file_path).replace('.parquet', '')
                print("\n" + f"--- Processing Dataset {i+1}/{len(all_data_files)}: {dataset_name} ---")
                
                # Load the raw data for the current file
                X_raw, y_raw = load_data_for_batch(file_path)
                
                if X_raw is None or len(X_raw) < 20: # Min samples for a meaningful split
                    print(f"  - Skipping, not enough data.")
                    continue

                # Scale the entire dataset *before* splitting
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)

                # Split into train/test sets
                split_idx = int(TRAIN_TEST_SPLIT_RATIO * len(X_scaled))
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

                if len(X_test) == 0:
                    print(f"  - Skipping, test set is empty after split.")
                    continue

                print(f"  - Train size: {len(X_train)}, Test size: {len(X_test)}")

                # Initialize a fresh set of models for this dataset
                models_to_run = {
                    "PassiveAggressive": PassiveAggressive(n_features=X_train.shape[1]),
                    "Perceptron": Perceptron(n_features=X_train.shape[1]),
                    "GradientLearning": GradientLearning(n_features=X_train.shape[1]),
                }

                # Train and evaluate each model
                for algo_name, model in models_to_run.items():
                    # Train the model on the entire training set
                    for k in range(len(X_train)):
                        model.partial_fit(X_train[k], y_train[k])
                    
                    # Evaluate the final model on the unseen test set
                    y_preds = [model.predict(x) for x in X_test]
                    precision, tpr, fpr = calculate_class1_metrics(y_test, y_preds)
                    
                    # Store the results
                    all_results.append({
                        'Dataset': dataset_name,
                        'Algorithm': algo_name,
                        'Precision': precision,
                        'TPR': tpr,
                        'FPR': fpr
                    })
            
            # --- 3. COMPILE AND SAVE RESULTS ---
            if all_results:
                final_df = pd.DataFrame(all_results)
                print("\n" + "="*60)
                print("BATCH EVALUATION COMPLETE. FINAL COMBINED RESULTS:")
                print("="*60)
                print(final_df.round(4).to_string())
                
                try:
                    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
                    final_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
                    print(f"\nSUCCESS: All results saved to '{OUTPUT_CSV_FILE}'")
                except Exception as e:
                    print(f"\nERROR: Could not save results to CSV. Reason: {e}")
            else:
                print("\nNo simulations were completed successfully.")
    else:
        print("\nHalting experiment due to data preparation failure.")