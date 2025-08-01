import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import glob
import requests
import zipfile
import io
import os
from scipy.stats import norm
import shutil
import time
# =============================================================================
# 1. HELPER FUNCTIONS AND ALGORITHM CLASSES
# =============================================================================




def prepare_data_from_zenodo(zenodo_archive_url, target_dir):
    """
    Handles the entire data acquisition process from Zenodo with a highly robust,
    on-disk streaming download to prevent memory and network timeout issues.
    """
    if os.path.exists(target_dir) and any(f.endswith('.parquet') for f in os.listdir(target_dir)):
        print(f"--> Data found locally in '{target_dir}'. Skipping download.")
        return True

    print(f"--> Local data not found. Preparing to download from Zenodo...")
    print(f"    URL: {zenodo_archive_url}")

    # Use a temporary file to save the download stream
    temp_zip_path = 'temp_download.zip'

    try:
        # --- START OF THE ROBUST DOWNLOAD FIX ---
        
        # Download the file in chunks and save directly to disk
        with requests.get(zenodo_archive_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            
            print(f"--> Connection established. Total file size: {total_size / 1024**2:.2f} MB")
            print("--> Downloading data to temporary file...")

            with open(temp_zip_path, 'wb') as f:
                start_time = time.time()
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Optional: Print progress
                    if time.time() - start_time > 2: # Print every 2 seconds
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        print(f"    Downloaded {downloaded_size / 1024**2:.2f} / {total_size / 1024**2:.2f} MB ({progress:.1f}%)", end='\r')
                        start_time = time.time()
        
        print("\n--> Download complete. Unzipping relevant files from archive...")
        # --- END OF THE ROBUST DOWNLOAD FIX ---
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Now, open the zip file from the disk
        with zipfile.ZipFile(temp_zip_path) as z:
            for member in z.infolist():
                if member.filename.endswith('.parquet') and not member.is_dir():
                    base_filename = os.path.basename(member.filename)
                    output_path = os.path.join(target_dir, base_filename)
                    
                    with z.open(member) as source, open(output_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    print(f"    - Extracted: {base_filename}")
        
        print(f"--> Successfully extracted all .parquet files to '{target_dir}'")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: A network error occurred. Please check the URL and your internet connection. Details: {e}")
        return False
    except zipfile.BadZipFile:
        print("\nERROR: The downloaded file is not a valid zip archive. It may be incomplete.")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during data preparation: {e}")
        return False
    finally:
        # --- CLEANUP: Always remove the temporary zip file ---
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
            print("--> Temporary download file cleaned up.")


class PassiveAggressive:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if y_true * x.dot(self.weights) < 1:
            l2_norm_sq = x.dot(x)
            if l2_norm_sq > 0:
                eta = (1 - y_true * x.dot(self.weights)) / l2_norm_sq
                self.weights += eta * y_true * x

class Perceptron:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if self.predict(x) != y_true: self.weights += y_true * x

class GradientLearning:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if y_true * x.dot(self.weights) < 1: self.weights += y_true * x


class AROW:
    """
    Adaptive Regularization of Weight Vectors (AROW).
    """
    def __init__(self, n_features, r):
        """
        Args:
            n_features (int): The number of features in the dataset.
            r (float): Regularization parameter.
        """
        self.r = r
        # The mean vector 'u' is the user-facing 'weights'
        self.weights = np.zeros(n_features)
        # The covariance matrix is an internal state of the model
        self.Sigma = np.identity(n_features)

    def predict(self, x):
        """
        Makes a prediction for a single data sample.
        """
        prediction = np.sign(x.dot(self.weights))
        # Ensure prediction is always -1 or 1, not 0
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        """
        Processes a single data sample (x, y_true) to update the model's weights.
        """
        # Calculate hinge loss and confidence for the current sample
        lt = max(0, 1 - y_true * x.dot(self.weights))
        vt = x.T.dot(self.Sigma).dot(x)
        
        # Only update the model's state if there is a loss
        if lt > 0:
            # Calculate the update coefficients, beta_t and alpha_t
            beta_t = 1 / (vt + self.r) if (vt + self.r) > 0 else 0.0
            alpha_t = lt * beta_t
           
            # Update the internal state (weights and covariance matrix)
            self.weights += alpha_t * y_true * self.Sigma.dot(x)
            self.Sigma -= beta_t * self.Sigma.dot(np.outer(x, x)).dot(self.Sigma)


class RDA:
    """Regularized Dual Averaging (RDA)."""
    def __init__(self, n_features, lambda_param, gamma_param):
        self.weights = np.zeros(n_features)
        self.g = np.zeros(n_features)
        self.t = 0
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        self.t += 1
        lt = max(0, 1 - y_true * x.dot(self.weights))
        gt = -y_true * x if lt > 0 else np.zeros_like(x)
        self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
        update_mask = np.abs(self.g) > self.lambda_param
        self.weights.fill(0)
        self.weights[update_mask] = -(np.sqrt(self.t) / self.gamma_param) * \
                                   (self.g[update_mask] - self.lambda_param * np.sign(self.g[update_mask]))

class SCW:
    """Soft Confidence-Weighted (SCW)."""
    def __init__(self, n_features, C, eta):
        self.phi = norm.ppf(eta)
        self.C = C
        self.weights = np.zeros(n_features)
        self.Sigma = np.identity(n_features)

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        vt = x.T.dot(self.Sigma).dot(x)
        mt = y_true * x.dot(self.weights)
        lt = max(0, self.phi * np.sqrt(vt) - mt)
        if lt > 0:
            pa = 1 + (self.phi**2) / 2
            xi = 1 + self.phi**2
            sqrt_term = max(0, (mt**2 * self.phi**4 / 4) + (vt * self.phi**2 * xi))
            alpha_t = min(self.C, max(0, (1 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term))))
            sqrt_ut_term = max(0, (alpha_t**2 * vt**2 * self.phi**2) + (4 * vt))
            ut = 0.25 * (-alpha_t * vt * self.phi + np.sqrt(sqrt_ut_term))**2
            beta_t = (alpha_t * self.phi) / (np.sqrt(ut) + vt * alpha_t * self.phi + 1e-8)
            self.weights += alpha_t * y_true * self.Sigma.dot(x)
            self.Sigma -= beta_t * self.Sigma.dot(np.outer(x, x)).dot(self.Sigma)

class AdaRDA:
    """Adaptive Regularized Dual Averaging (AdaRDA)."""
    def __init__(self, n_features, lambda_param, eta_param, delta_param):
        self.weights = np.zeros(n_features)
        self.g, self.g1t = np.zeros(n_features), np.zeros(n_features)
        self.t = 0
        self.lambda_param, self.eta_param, self.delta_param = lambda_param, eta_param, delta_param

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        self.t += 1
        lt = max(0, 1 - y_true * x.dot(self.weights))
        gt = -y_true * x if lt > 0 else np.zeros_like(x)
        self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
        self.g1t += gt**2
        Ht = self.delta_param + np.sqrt(self.g1t)
        update_mask = np.abs(self.g) > self.lambda_param
        self.weights.fill(0)
        self.weights[update_mask] = np.sign(-self.g[update_mask]) * self.eta_param * self.t / (Ht[update_mask] + 1e-8)


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
    y_series = x_df["label"].map({0: -1, 1: 1}).fillna(1)
    x_df = x_df.drop(columns=['label'], axis=1)
    return x_df.to_numpy(), y_series.to_numpy()

def calculate_class1_metrics(y_true, y_pred):
    """Calculates Precision, TPR, and FPR for the positive class (1)."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError: return np.nan, np.nan, np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return precision, recall, fnr, fpr

# =============================================================================
# 2. MAIN BATCH EXPERIMENT SCRIPT
# =============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'
    OUTPUT_CSV_FILE = 'results/batch_learning_results.csv'
    TRAIN_TEST_SPLIT_RATIO = 0.6
    # --- NEW: ADD EPOCHS PARAMETER FOR BATCH TRAINING ---
    EPOCHS = 5

    # --- 1. PREPARE DATA ---
    data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)

    if data_ready:
        # --- 2. RUN BATCH EXPERIMENT ---
        search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
        all_data_files = sorted(glob.glob(search_path))

        if not all_data_files:
            print(f"FATAL: No .parquet files found in '{DATA_DIRECTORY}'.")
        else:
            print(f"Found {len(all_data_files)} datasets. Starting batch learning evaluation...")
            
            all_results = []
            
            for i, file_path in enumerate(all_data_files):
                dataset_name = os.path.basename(file_path).replace('.parquet', '')
                print("\n" + f"--- Processing Dataset {i+1}/{len(all_data_files)}: {dataset_name} ---")
                
                X_raw, y_raw = load_data_for_batch(file_path)
                
                if X_raw is None or len(X_raw) < 20:
                    print(f"  - Skipping, not enough data.")
                    continue

                X_scaled = StandardScaler().fit_transform(X_raw)
                split_idx = int(TRAIN_TEST_SPLIT_RATIO * len(X_scaled))
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

                if len(X_test) == 0 or len(X_train) == 0:
                    print(f"  - Skipping, train or test set is empty.")
                    continue

                print(f"  - Train size: {len(X_train)}, Test size: {len(X_test)}")

                # Initialize models for this dataset
                n_features = X_train.shape[1]
                models_to_run = {
                    "PA": PassiveAggressive(n_features=n_features),
                    "Perceptron": Perceptron(n_features=n_features),
                    "GLC": GradientLearning(n_features=n_features),
                    "AROW": AROW(n_features=n_features, r=1.0),
                    "RDA": RDA(n_features=n_features, lambda_param=0.01, gamma_param=1.0),
                    "SCW": SCW(n_features=n_features, C=0.1, eta=0.95),
                    "AdaRDA": AdaRDA(n_features=n_features, lambda_param=0.01, eta_param=0.1, delta_param=1.0)
                }

                for algo_name, model in models_to_run.items():
                    print(f"  - Training {algo_name} for {EPOCHS} epochs...")
                    
                    # --- FIX: WRAP THE TRAINING IN AN EPOCHS LOOP FOR BATCH LEARNING ---
                    for epoch in range(EPOCHS):
                        # Optional: Add shuffling here if desired for non-time-series data
                        # permutation = np.random.permutation(len(X_train))
                        # for k in permutation:
                        for k in range(len(X_train)): # Current: sequential processing per epoch
                            model.partial_fit(X_train[k], y_train[k])
                    # --- END OF FIX ---
                    
                    # Evaluate on the training set after all epochs are complete
                    y_preds_train = [model.predict(x) for x in X_train]
                    train_precision, train_tpr, train_fnr, train_fpr = calculate_class1_metrics(y_train, y_preds_train)

                    # Evaluate the final model on the unseen test set
                    y_preds_test = [model.predict(x) for x in X_test]
                    test_precision, test_tpr, test_fnr, test_fpr = calculate_class1_metrics(y_test, y_preds_test)

                    all_results.append({
                        'Dataset': dataset_name, 'Algorithm': algo_name,
                        'Train_Precision': train_precision, 'Train_Recall': train_tpr, 'Train_FNR': train_fnr, 'Train_FPR': train_fpr,
                        'Test_Precision': test_precision, 'Test_Recall': test_tpr, 'Test_FNR': test_fnr, 'Test_FPR': test_fpr
                    })
            
            # --- Compile and save results ---
            if all_results:
                final_df = pd.DataFrame(all_results)
                # A more consistent and readable order
                cols_order = [
                    # Identifiers
                    'Dataset', 
                    'Algorithm', 
                    
                    # Training Set Performance
                    'Train_Precision',
                    'Train_Recall',
                    'Train_FPR',
                    'Train_FNR',
                    
                    # Testing Set Performance (in the same order)
                    'Test_Precision',
                    'Test_Recall',
                    'Test_FPR',
                    'Test_FNR'
                ]

                final_df = final_df[cols_order]
                final_df.sort_values(by=['Algorithm', 'Dataset'], inplace=True)
                print("\n" + "="*80)
                print("BATCH EVALUATION COMPLETE. FINAL COMBINED RESULTS:")
                print("="*80)
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