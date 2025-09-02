import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import glob
from scipy.stats import norm
from data_handler import prepare_data_from_zenodo
# Kaggle dataset support
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# =============================================================================
# 1. ALGORITHM FUNCTIONS WITH MULTI-EPOCH SUPPORT
# =============================================================================

def AROW(X, y, r, max_epochs=1, patience=3, X_val=None, y_val=None):
    """
    Online learning implementation of AROW with optional multi-epoch training and early stopping.
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    u = np.zeros(n_features)
    Sigma = np.identity(n_features)
    
    y_pred = np.zeros(n_samples)
    weight_history = []
    
    # Early stopping variables
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None
    
    for epoch in range(max_epochs):
        epoch_weights = []
        
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

            epoch_weights.append(u.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check for multi-epoch training
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(u)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = u.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                if best_weights is not None:
                    u = best_weights
                break
        
    return y_pred, np.array(weight_history)

def AP(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Passive-Aggressive function with optional multi-epoch training."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = []
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))
    
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None
    
    for epoch in range(max_epochs):
        epoch_weights = []
        
        for i in range(len(data)):
            x = data.iloc[i, :]
            y_pred[i] = np.sign(x.dot(w))  # Always predict, not just in epoch 0
            y_actual = y.iloc[i]
            loss = max(0, 1 - y_actual * y_pred[i])  # Use actual prediction for loss
          
            if loss > 0:
                l2_norm_sq_sq = (x.dot(x))**2
                if l2_norm_sq_sq > 0:
                    eta = loss / l2_norm_sq_sq
                    w += eta * y_actual * x

            epoch_weights.append(w.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(w)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    w = best_weights
                break
        
    return y_pred, np.array(weight_history)

def PERCEPT(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Perceptron function with optional multi-epoch training."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = []
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))
    
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None
    
    for epoch in range(max_epochs):
        epoch_weights = []
        
        for i in range(len(data)):
            x = data.iloc[i, :]
            y_pred[i] = np.sign(x.dot(w))  # Always predict, not just in epoch 0
            y_actual = y.iloc[i]

            if y_pred[i] != y_actual:
                w += y_actual * x
            
            epoch_weights.append(w.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(w)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    w = best_weights
                break
        
    return y_pred, np.array(weight_history)

def OGL(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Online Gradient Learning function with optional multi-epoch training."""
    data = pd.DataFrame(X)
    y = pd.Series(y)
    weight_history = []
    y_pred = np.ones(len(data))
    w = np.zeros(len(data.columns))

    best_val_f1 = -1
    patience_counter = 0
    best_weights = None
    
    for epoch in range(max_epochs):
        epoch_weights = []
        
        for i in range(len(data)):
            x = data.iloc[i, :]
            y_pred[i] = np.sign(x.dot(w))  # Always predict, not just epoch 0
            y_act = y.iloc[i]
            w = w + (y_act - y_pred[i]) / (np.sqrt(x.dot(x)) + 1e-8) * x  # Use actual prediction
            
            epoch_weights.append(w.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(w)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    w = best_weights
                break
        
    return y_pred, np.array(weight_history)

def RDA(X, y, lambda_param=1, gamma_param=1, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Regularized Dual Averaging for L1 regularization with optional multi-epoch training."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    w = np.zeros(n_features)
    g = np.zeros(n_features)
    
    y_pred = np.zeros(n_samples)
    weight_history = []
    
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None

    for epoch in range(max_epochs):
        epoch_weights = []
        
        for i in range(n_samples):
            t = i + 1 + epoch * n_samples
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
            w[update_mask] = -(np.sqrt(t) / gamma_param) * (g[update_mask] - lambda_param * np.sign(g[update_mask]))
            
            epoch_weights.append(w.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(w)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    w = best_weights
                break
        
    return y_pred, np.array(weight_history)

def SCW(X, y, C=1, eta=0.5, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Soft Confidence-Weighted learning with optional multi-epoch training."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    phi = norm.ppf(eta) 
    u = np.zeros(n_features)
    Sigma = np.identity(n_features)
    
    y_pred = np.zeros(n_samples)
    weight_history = []
    
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None

    for epoch in range(max_epochs):
        epoch_weights = []
        
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

            epoch_weights.append(u.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(u)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = u.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    u = best_weights
                break
        
    return y_pred, np.array(weight_history)

def AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Adaptive Regularized Dual Averaging with optional multi-epoch training."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    w = np.zeros(n_features)
    g = np.zeros(n_features)
    g1t = np.zeros(n_features)

    y_pred = np.zeros(n_samples)
    weight_history = []
    
    best_val_f1 = -1
    patience_counter = 0
    best_weights = None

    for epoch in range(max_epochs):
        epoch_weights = []
        
        for i in range(n_samples):
            t = i + 1 + epoch * n_samples
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
            g1t += gt**2
            
            Ht = delta_param + np.sqrt(g1t)
            update_mask = np.abs(g) > lambda_param
            w.fill(0)
            w[update_mask] = np.sign(-g[update_mask]) * eta_param * t / (Ht[update_mask] + 1e-8)
            
            epoch_weights.append(w.copy())
        
        weight_history.extend(epoch_weights)
        
        # Early stopping check
        if max_epochs > 1 and X_val is not None and y_val is not None:
            val_scores = X_val.dot(w)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    w = best_weights
                break
        
    return y_pred, np.array(weight_history)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_and_process_data(file_path, shuffle_mode="both", online_mode=False):
    """Loads and preprocesses a single parquet file with both shuffled and natural ordering."""
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  - ERROR: Could not read file '{file_path}'. Skipping. Reason: {e}")
        return None, None

    # Determine the correct time column name
    if 'Time' in df.columns:
        time_col = 'Time'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        print(f"  - WARNING: No time column found in {os.path.basename(file_path)}")
        time_col = None
    
    # Sort by time column if available
    if time_col:
        df = df.sort_values([time_col]).reset_index(drop=True)
        print(f"  - Sorted by '{time_col}' column")
    
    # Check if this is Kaggle Credit Fraud dataset
    dataset_name = os.path.basename(file_path).replace('.parquet', '')
    is_kaggle_fraud = 'KAGGLE' in dataset_name or 'CreditFraud' in dataset_name
    
    if is_kaggle_fraud:
        if online_mode:
            print(f"  - Pure online learning mode for Kaggle dataset (matching online_kaggle_fraud.py)")
            
            # Drop user_id if present
            if 'user_id' in df.columns:
                df = df.drop(columns=['user_id'], axis=1)
            
            # Temporal split (80% train, 20% test)
            split_idx = int(0.8 * len(df))
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
            
            # Prepare features (same as online_kaggle_fraud.py)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'Time', 'user_id', 'label', 'Class']]
            
            X_train = train_data[feature_cols].values
            # Handle both 'label' (Zenodo) and 'Class' (Kaggle) column names
            if 'label' in train_data.columns:
                y_train = train_data['label'].values
                y_test = test_data['label'].values
            elif 'Class' in train_data.columns:
                y_train = train_data['Class'].values  
                y_test = test_data['Class'].values
            else:
                raise KeyError(f"No label column found. Available columns: {list(train_data.columns)}. Expected 'label' or 'Class'.")
            X_test = test_data[feature_cols].values
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Convert labels to -1/+1 format for online algorithms
            y_train_online = np.where(y_train == 0, -1, 1)
            y_test_online = np.where(y_test == 0, -1, 1)
            
            print(f"  - Train: {len(train_data):,} samples, Test: {len(test_data):,} samples")
            print(f"  - Train fraud ratio: {(y_train == 1).sum() / len(y_train):.4f}")
            print(f"  - Test fraud ratio: {(y_test == 1).sum() / len(y_test):.4f}")
            
            return (X_train, y_train_online, X_test, y_test_online), None
        
        else:
            print(f"  - Special handling for Kaggle Credit Fraud dataset with batch training")
            
            # For Kaggle fraud: Drop user_id if present
            if 'user_id' in df.columns:
                df = df.drop(columns=['user_id'], axis=1)
            
            # Create 80/20 split for train/test first
            test_split_idx = int(0.8 * len(df))
            train_val_df = df.iloc[:test_split_idx].copy()
            test_df = df.iloc[test_split_idx:].copy()
            
            # Prepare both natural and shuffled versions for comparison
            results = {}
            
            if shuffle_mode in ["both", "natural"]:
                # NATURAL ORDERING VERSION - Pure temporal split (like online_kaggle_fraud.py)
                print(f"  - Preparing NATURAL ordering version (temporal split)")
                
                # Temporal split: 80% train, 20% test (no validation for pure online learning)
                temporal_split_idx = int(0.8 * len(df))
                train_df_natural = df.iloc[:temporal_split_idx].copy()
                test_df_natural = df.iloc[temporal_split_idx:].copy()
                
                # Extract features and labels - natural temporal ordering
                feature_cols = [col for col in train_df_natural.columns if col not in ['Time', 'timestamp', 'label', 'Class']]
                
                X_train_nat = train_df_natural[feature_cols].values
                # Handle both 'label' (Zenodo) and 'Class' (Kaggle) column names
                if 'label' in train_df_natural.columns:
                    y_train_nat = np.where(train_df_natural['label'].values == 0, -1, 1)
                    y_test_nat = np.where(test_df_natural['label'].values == 0, -1, 1)
                elif 'Class' in train_df_natural.columns:
                    y_train_nat = np.where(train_df_natural['Class'].values == 0, -1, 1)
                    y_test_nat = np.where(test_df_natural['Class'].values == 0, -1, 1)
                else:
                    raise KeyError("No label column found. Expected 'label' or 'Class'.")
                X_test_nat = test_df_natural[feature_cols].values
                
                # Apply StandardScaler
                scaler_nat = StandardScaler()
                X_train_nat_scaled = scaler_nat.fit_transform(X_train_nat)
                X_test_nat_scaled = scaler_nat.transform(X_test_nat)
                
                # For natural temporal: no validation set needed for pure online learning
                results['natural'] = (X_train_nat_scaled, y_train_nat, None, None, X_test_nat_scaled, y_test_nat)
                
                print(f"    Natural - Train fraud ratio: {(y_train_nat == 1).sum() / len(y_train_nat):.4f}")
                print(f"    Natural - Test fraud ratio: {(y_test_nat == 1).sum() / len(y_test_nat):.4f}")
                print(f"    Natural - Split: {len(train_df_natural):,} train, {len(test_df_natural):,} test (temporal)")
            
            if shuffle_mode in ["both", "shuffled"]:
                # SHUFFLED VERSION - Different split approach (train/val/test for multi-epoch)
                print(f"  - Preparing SHUFFLED version for comparison (train/val/test split)")
                
                # Use the original train_val_df and test_df for shuffled approach
                train_val_shuffled = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                val_split_idx = int(0.8 * len(train_val_shuffled))
                train_df_shuffled = train_val_shuffled.iloc[:val_split_idx].copy()
                val_df_shuffled = train_val_shuffled.iloc[val_split_idx:].copy()
                
                # Extract features and labels - shuffled
                feature_cols = [col for col in df.columns if col not in ['Time', 'timestamp', 'label', 'Class']]
                
                X_train_shuf = train_df_shuffled[feature_cols].values
                # Handle both 'label' (Zenodo) and 'Class' (Kaggle) column names
                if 'label' in train_df_shuffled.columns:
                    y_train_shuf = np.where(train_df_shuffled['label'].values == 0, -1, 1)
                    y_val_shuf = np.where(val_df_shuffled['label'].values == 0, -1, 1)
                    y_test_shuf = np.where(test_df['label'].values == 0, -1, 1)
                elif 'Class' in train_df_shuffled.columns:
                    y_train_shuf = np.where(train_df_shuffled['Class'].values == 0, -1, 1)
                    y_val_shuf = np.where(val_df_shuffled['Class'].values == 0, -1, 1)
                    y_test_shuf = np.where(test_df['Class'].values == 0, -1, 1)
                else:
                    raise KeyError("No label column found. Expected 'label' or 'Class'.")
                X_val_shuf = val_df_shuffled[feature_cols].values
                X_test_shuf = test_df[feature_cols].values
                
                # Apply StandardScaler
                scaler_shuf = StandardScaler()
                X_train_shuf_scaled = scaler_shuf.fit_transform(X_train_shuf)
                X_val_shuf_scaled = scaler_shuf.transform(X_val_shuf)
                X_test_shuf_scaled = scaler_shuf.transform(X_test_shuf)
                
                results['shuffled'] = (X_train_shuf_scaled, y_train_shuf, X_val_shuf_scaled, y_val_shuf, X_test_shuf_scaled, y_test_shuf)
                
                print(f"    Shuffled - Train fraud ratio: {(y_train_shuf == 1).sum() / len(y_train_shuf):.4f}")
                print(f"    Shuffled - Val fraud ratio: {(y_val_shuf == 1).sum() / len(y_val_shuf):.4f}")
                print(f"    Shuffled - Test fraud ratio: {(y_test_shuf == 1).sum() / len(y_test_shuf):.4f}")
                print(f"    Shuffled - Split: {len(train_df_shuffled):,} train, {len(val_df_shuffled):,} val, {len(test_df):,} test (shuffled)")
            
            print(f"  - Split comparison:")
            print(f"    Natural: Temporal split - 80% train, 20% test (like online_kaggle_fraud.py)")
            print(f"    Shuffled: Random split - 64% train, 16% val, 20% test (for early stopping)")
            
            return results, None
    
    else:
        # Original processing for cybersecurity datasets
        if 'user_id' in df.columns:
            df = df.drop(columns=['user_id'], axis=1)
            
        # Use the correct time column for grouping
        if time_col and time_col in df.columns:
            x_df = df.groupby([time_col]).sum()
            
            # Convert aggregated labels to -1/+1 format
            if 'label' in x_df.columns:
                y_series = x_df["label"].map(lambda x: 1 if x > 0 else -1)
                x_df = x_df.drop(columns=['label'], axis=1)
            elif 'Class' in x_df.columns:
                y_series = x_df["Class"].map(lambda x: 1 if x > 0 else -1)
                x_df = x_df.drop(columns=['Class'], axis=1)
            else:
                raise KeyError("No label column found. Expected 'label' or 'Class'.")
            
            # Count periods correctly
            attack_periods = (y_series == 1).sum()
            normal_periods = (y_series == -1).sum()
            
            print(f"  - Grouped by '{time_col}' - {len(x_df)} time periods")
            print(f"  - Normal periods: {normal_periods}, Attack periods: {attack_periods}")
            print(f"  - Attack ratio: {attack_periods / len(y_series):.4f}")
        else:
            print(f"  - ERROR: No valid time column found for cybersecurity dataset")
            return None, None
        
        return x_df, y_series.to_numpy()

def calculate_class1_metrics(y_true, y_pred):
    """Calculates Precision, TPR, FNR, FPR, and F1 for the positive class (1)."""
    try:
        # Get unique labels to determine the appropriate format
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        unique_labels = np.unique(np.concatenate([unique_true, unique_pred]))
        
        # Handle different label formats
        if len(unique_labels) <= 2:
            # Binary classification - use actual unique labels
            labels = sorted(unique_labels)
            
            # For confusion matrix, we need to know which is positive class
            # Assume: 1, 'fraud', 'malicious', 'attack', 'positive' are positive
            # Assume: 0, -1, 'normal', 'benign', 'negative' are negative
            positive_indicators = {1, '1', 'fraud', 'malicious', 'attack', 'positive', 'anomaly'}
            
            # Find which label is positive
            pos_label = None
            for label in labels:
                if label in positive_indicators or (isinstance(label, (int, float)) and label > 0):
                    pos_label = label
                    break
            
            # If we can't determine positive label, assume the second in sorted order
            if pos_label is None:
                pos_label = labels[1] if len(labels) > 1 else labels[0]
            
            # Create labels list with negative first, positive second (for sklearn)
            neg_label = labels[0] if labels[0] != pos_label else labels[1] if len(labels) > 1 else labels[0]
            labels = [neg_label, pos_label]
        else:
            # Multi-class - use all unique labels
            labels = sorted(unique_labels)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if cm.shape == (2, 2):
            # Binary classification
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multi-class: treat first class as negative, others as positive
            # This is a simplification for multi-class scenarios
            tn = cm[0, 0]
            fp = cm[0, 1:].sum()
            fn = cm[1:, 0].sum()
            tp = cm[1:, 1:].sum()
            
    except (ValueError, IndexError) as e:
        print(f"Error in calculate_class1_metrics: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, fnr, fpr, f1

# =============================================================================
# 3. MAIN EXPERIMENT RUNNER
# =============================================================================

if __name__ == "__main__":
    # Configuration
    DATA_DIRECTORY = '../data/'
    OUTPUT_CSV_FILE = '../results/online_results.csv'
    # Use the proper Zenodo API URL for individual file downloads
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591'
    print(f"Using data directory: {DATA_DIRECTORY}")
    # Check for required parquet files
    search_path = os.path.join(DATA_DIRECTORY, '*.parquet')
    all_data_files = sorted(glob.glob(search_path))
    kaggle_file_name = 'creditcard.csv'
    kaggle_parquet_name = 'CreditFraud_kaggle.parquet'
    kaggle_parquet_path = os.path.join(DATA_DIRECTORY, kaggle_parquet_name)

    # Always ensure we have both Zenodo and Kaggle datasets
    data_ready = True
    
    # 1. Check and download Zenodo datasets
    zenodo_files = [f for f in all_data_files if not f.endswith('CreditFraud_kaggle.parquet')]
    if len(zenodo_files) < 12:  # We expect 12 Zenodo parquet files
        print("Missing Zenodo datasets. Downloading from Zenodo...")
        zenodo_success = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)
        if not zenodo_success:
            print("WARNING: Zenodo download failed")
            data_ready = False
    else:
        print("All Zenodo datasets found locally.")
    
    # 2. Check and download Kaggle dataset
    if not os.path.exists(kaggle_parquet_path):
        print("Missing Kaggle dataset. Downloading Kaggle Credit Card Fraud dataset...")
        if KAGGLE_AVAILABLE:
            try:
                print("Downloading Kaggle Credit Card Fraud dataset using kagglehub...")
                # Use the correct dataset_download() API
                dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                
                # Find the CSV file in the downloaded dataset
                csv_path = os.path.join(dataset_path, kaggle_file_name)
                if os.path.exists(csv_path):
                    df_kaggle = pd.read_csv(csv_path)
                    print(f"Downloaded Kaggle dataset with shape: {df_kaggle.shape}")
                    # Save as parquet for consistency
                    os.makedirs(DATA_DIRECTORY, exist_ok=True)
                    df_kaggle.to_parquet(kaggle_parquet_path)
                    print(f"Saved Kaggle dataset as: {kaggle_parquet_path}")
                else:
                    print(f"ERROR: Expected CSV file not found at: {csv_path}")
                    data_ready = False
            except Exception as e:
                print(f"ERROR: Kaggle dataset download failed: {e}")
                data_ready = False
        else:
            print("ERROR: kagglehub is not installed. Please install it to enable Kaggle downloads.")
            data_ready = False
    else:
        print("Kaggle dataset found locally.")
    
    # Re-scan for all datasets after downloads
    all_data_files = sorted(glob.glob(search_path))

    if data_ready:
        # Define algorithms with their hyperparameters (matching online_kaggle_fraud.py)
        algorithms_to_run = {
            "PA": lambda X, y: AP(X, y),
            "Perceptron": lambda X, y: PERCEPT(X, y),
            "GLC": lambda X, y: OGL(X, y),
            "AROW": lambda X, y: AROW(X, y, r=0.1),
            "RDA": lambda X, y: RDA(X, y, lambda_param=1, gamma_param=1),
            "SCW": lambda X, y: SCW(X, y, C=1, eta=0.5),
            "AdaRDA": lambda X, y: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1)
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
                print("\n" + "="*60)
                print(f"Processing Dataset {i+1}/{len(all_data_files)}: {dataset_name}")
                print("="*60)
                
                # Check if this is Kaggle dataset
                is_kaggle = 'KAGGLE' in dataset_name or 'CreditFraud' in dataset_name
                
                if is_kaggle:
                    # Simple linear processing for Kaggle dataset (matching online_kaggle_fraud.py)
                    print(f"\n  === KAGGLE DATASET: SIMPLE LINEAR PROCESSING ===")
                    
                    # Load and process data using simple approach
                    online_data = load_and_process_data(file_path, online_mode=True)
                    
                    if online_data is None or (len(online_data) == 2 and online_data[0] is None):
                        print(f"  - Skipping due to data issues.")
                        continue
                    
                    X_train, y_train, X_test, y_test = online_data[0]
                    
                    for algo_name, algo_func in algorithms_to_run.items():
                        print(f"  - Running {algo_name}...")
                        
                        try:
                            # Train the algorithm (sequential online learning)
                            y_pred_train, weights_history = algo_func(X_train, y_train)
                            
                            # Get final weights for testing
                            final_weights = weights_history[-1] if len(weights_history) > 0 else np.zeros(X_train.shape[1])
                            
                            # Test predictions
                            test_scores = X_test.dot(final_weights)
                            y_pred_test = np.sign(test_scores)
                            y_pred_test = np.where(y_pred_test == 0, 1, y_pred_test)  # Handle zero case
                            
                            # Calculate metrics (convert back to 0/1 for metrics)
                            y_test_01 = np.where(y_test == -1, 0, y_test)
                            y_pred_01 = np.where(y_pred_test == -1, 0, y_pred_test)
                            
                            precision, recall, fnr, fpr, f1 = calculate_class1_metrics(y_test_01, y_pred_01)
                            
                            print(f"    Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                            
                            # Store results
                            all_results.append({
                                'Dataset': dataset_name,
                                'Algorithm': algo_name,
                                'Precision': precision,
                                'Recall': recall,
                                'FNR': fnr,
                                'FPR': fpr,
                                'F1': f1
                            })
                            
                        except Exception as e:
                            print(f"    ERROR running {algo_name}: {e}")
                            all_results.append({
                                'Dataset': dataset_name,
                                'Algorithm': algo_name,
                                'Precision': np.nan,
                                'Recall': np.nan,
                                'FNR': np.nan,
                                'FPR': np.nan,
                                'F1': np.nan
                            })
                    
                    # Continue to next dataset
                    continue
                    
                # Load and process data for cybersecurity datasets
                data_result = load_and_process_data(file_path)
                
                if data_result is None or (len(data_result) == 2 and data_result[0] is None):
                    print(f"  - Skipping dataset due to insufficient data.")
                    continue
                
                # Cybersecurity datasets use online learning
                X, y_true = data_result
                
                if len(X) <= 1:
                    print(f"  - Skipping dataset due to insufficient data.")
                    continue
                
                # Use online learning (prequential evaluation) for cybersecurity datasets
                print(f"  - Using ONLINE LEARNING mode (prequential evaluation)")
                print(f"  - Total samples: {len(X)}, Positive class ratio: {(y_true == 1).sum() / len(y_true):.4f}")
                
                for algo_name, algo_func in algorithms_to_run.items():
                    print(f"  - Running algorithm: {algo_name}...")
                    
                    try:
                        # Run algorithm in online mode
                        y_pred_stream, weights_history = algo_func(X, y_true)
                        
                        # Prequential evaluation: predict at time t using model trained up to t-1
                        y_true_eval = y_true[1:]
                        y_pred_eval = y_pred_stream[:-1]
                        
                        # Calculate metrics
                        precision, recall, fnr, fpr, f1 = calculate_class1_metrics(y_true_eval, y_pred_eval)
                        
                        print(f"    Performance - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                        
                        # Store results
                        all_results.append({
                            'Dataset': dataset_name,
                            'Algorithm': algo_name,
                            'Precision': precision,
                            'Recall': recall,
                            'FNR': fnr,
                            'FPR': fpr,
                            'F1': f1
                        })
                        
                    except Exception as e:
                        print(f"    ERROR running {algo_name} on {dataset_name}: {e}")
                        all_results.append({
                            'Dataset': dataset_name,
                            'Algorithm': algo_name,
                            'Precision': np.nan,
                            'Recall': np.nan,
                            'FNR': np.nan,
                            'FPR': np.nan,
                            'F1': np.nan
                        })

            if all_results:
                final_df = pd.DataFrame(all_results)
                final_df.sort_values(by=['Algorithm', 'Dataset'], inplace=True)
                
                print("\n" + "="*60)
                print("BATCH EVALUATION COMPLETE - BEST APPROACH SELECTED:")
                print("="*60)
                print(final_df.round(4).to_string())
                
                # Create summary of selected approaches for fraud detection
                fraud_results = final_df[final_df['Dataset'].str.contains('KAGGLE|CreditFraud', na=False)]
                if not fraud_results.empty:
                    print("\n" + "="*60)
                    print("SELECTED APPROACHES SUMMARY (Fraud Detection):")
                    print("="*60)
                    
                    approach_summary = []
                    test_results = fraud_results[fraud_results['Dataset'].str.contains('_test')]
                    
                    for _, row in test_results.iterrows():
                        if '_' in row['Algorithm']:
                            algo_base, approach = row['Algorithm'].split('_', 1)
                            approach_summary.append({
                                'Algorithm': algo_base,
                                'Selected_Approach': approach.capitalize(),
                                'Test_F1': row['F1'],
                                'Test_Precision': row['Precision'],
                                'Test_Recall': row['Recall']
                            })
                    
                    if approach_summary:
                        summary_df = pd.DataFrame(approach_summary)
                        summary_df = summary_df.sort_values('Test_F1', ascending=False)
                        print(summary_df.round(4).to_string(index=False))
                        
                        # Approach statistics
                        natural_count = (summary_df['Selected_Approach'] == 'Natural').sum()
                        shuffled_count = (summary_df['Selected_Approach'] == 'Shuffled').sum()
                        avg_f1 = summary_df['Test_F1'].mean()
                        
                        print(f"\nSUMMARY:")
                        print(f"- Natural ordering selected: {natural_count}/{len(summary_df)} algorithms")
                        print(f"- Shuffled ordering selected: {shuffled_count}/{len(summary_df)} algorithms")
                        print(f"- Average test F1 score: {avg_f1:.4f}")
                        print(f"- Best performing algorithm: {summary_df.iloc[0]['Algorithm']} ({summary_df.iloc[0]['Selected_Approach']}) - F1={summary_df.iloc[0]['Test_F1']:.4f}")
                
                try:
                    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
                    final_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.4f')
                    print(f"\nSUCCESS: Best approach results saved to '{OUTPUT_CSV_FILE}'")
                except Exception as e:
                    print(f"\nERROR: Could not save results to CSV. Reason: {e}")
            else:
                print("\nNo simulations were completed successfully.")
