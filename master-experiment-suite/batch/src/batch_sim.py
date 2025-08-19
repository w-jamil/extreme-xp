import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import os

# Import the new data loading module
import data_loader

# =============================================================================
# 1. ALGORITHM CLASSES 
# =============================================================================


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

# (Other algorithm classes like RDA, SCW, AdaRDA would be here, unchanged)

# =============================================================================
# 2. HELPER FUNCTIONS FOR EVALUATION
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculates Precision, Recall, FNR, and FPR for the positive class (1)."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    except ValueError: return np.nan, np.nan, np.nan, np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, fnr, fpr

def calculate_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

# =============================================================================
# 3. MAIN BATCH EXPERIMENT SCRIPT
# =============================================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    VALIDATION_SPLIT_RATIO = 0.20 # Hold out 20% of training data for validation
    EPOCHS = 5
    OUTPUT_CSV_FILE = 'results/batch_learning_results.csv'

    # --- 1. PREPARE DATA USING THE NEW DATA LOADER ---
    # Data is now loaded once at the beginning, already split and scaled.
    print("--> Loading and preparing all datasets...")
    (X_train_full, Y_train_full), (X_test_all, Y_test_all), _ = data_loader.rbd24(lg=True)
    data_loader.summarise_data(X_train_full, Y_train_full, X_test_all, Y_test_all)
    
    all_results = []
    
    # --- 2. LOOP THROUGH EACH DATASET (from the loaded dictionary) ---
    for dataset_name in sorted(X_train_full.keys()):
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        
        # Get the data for the current dataset
        x_train_full_ds = X_train_full[dataset_name]
        y_train_full_ds = Y_train_full[dataset_name].astype(int) # Convert bool to int
        x_test_ds = X_test_all[dataset_name]
        y_test_ds = Y_test_all[dataset_name].astype(int)
        
        # IMPORTANT: Convert labels from {0, 1} to {-1, 1} for the algorithms
        y_train_full_ds[y_train_full_ds == 0] = -1
        y_test_ds[y_test_ds == 0] = -1

        # --- 3. CREATE VALIDATION SPLIT ---
        val_split_idx = int(len(x_train_full_ds) * (1 - VALIDATION_SPLIT_RATIO))
        X_train, X_val = x_train_full_ds[:val_split_idx], x_train_full_ds[val_split_idx:]
        y_train, y_val = y_train_full_ds[:val_split_idx], y_train_full_ds[val_split_idx:]

        if any(len(arr) == 0 for arr in [X_train, X_val, x_test_ds]):
            print("  - Skipping, a data partition is empty after splitting.")
            continue
        
        print(f"  - Data Partitions -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(x_test_ds)}")
        n_features = X_train.shape[1]
        
        # --- 4. INITIALIZE AND TRAIN MODELS ---
        models_to_run = {
            "Perceptron": Perceptron(n_features=n_features),
            "PA": PassiveAggressive(n_features=n_features),
            "OGC": GradientLearning(n_features=n_features),
            "AROW": AROW(n_features=n_features, r=1.0),
            # Add other models here
        }

        for algo_name, model in models_to_run.items():
            print(f"  - Training {algo_name}...")
            best_val_score = -1.0
            best_weights = np.copy(model.weights)
            if hasattr(model, 'Sigma'): best_Sigma = np.copy(model.Sigma)

            # --- 5. STOCHASTIC TRAINING LOOP (EPOCHS) ---
            for epoch in range(EPOCHS):
                permutation = np.random.permutation(len(X_train))
                for k in permutation: model.partial_fit(X_train[k], y_train[k])
                
                y_preds_val = [model.predict(x) for x in X_val]
                current_val_accuracy = calculate_accuracy(y_val, y_preds_val)
                
                if current_val_accuracy > best_val_score:
                    best_val_score = current_val_accuracy
                    best_weights = np.copy(model.weights)
                    if hasattr(model, 'Sigma'): best_Sigma = np.copy(model.Sigma)

            model.weights = np.copy(best_weights)
            if hasattr(model, 'Sigma'): model.Sigma = np.copy(best_Sigma)

            # --- 6. FINAL EVALUATION ---
            y_preds_test = [model.predict(x) for x in x_test_ds]
            precision, recall, fnr, fpr = calculate_metrics(y_test_ds, y_preds_test)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            all_results.append({
                'Dataset': dataset_name, 'Algorithm': algo_name,
                'Precision': precision, 'Recall': recall, 'FNR': fnr, 'FPR': fpr, 'F1-Score': f1_score
            })
    
    # --- 7. COMPILE AND SAVE RESULTS ---
    if all_results:
        final_df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
        final_df.sort_values(by=['Dataset', 'Algorithm'], inplace=True)
        
        print("\n" + "="*80 + "\nBATCH EVALUATION COMPLETE. FINAL RESULTS:\n" + "="*80)
        print(final_df.round(4).to_string())
        
        final_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
        print(f"\nSUCCESS: All results saved to '{OUTPUT_CSV_FILE}'")