import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import gc
import copy
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.stats import norm
from collections import defaultdict
import sys

# Custom Online Learning Algorithms (Exact user specifications with batch adaptation)

class PassiveAggressive:
    def __init__(self, n_features=None):
        self.weights = None
        self.n_features = n_features
        
    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            if y_i * x_i.dot(self.weights) < 1:
                l2_norm_sq = x_i.dot(x_i)
                if l2_norm_sq > 0:
                    eta = (1 - y_i * x_i.dot(self.weights)) / l2_norm_sq
                    self.weights += eta * y_i * x_i
        return self
    
    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            predictions.append(1 if prediction >= 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class Perceptron:
    def __init__(self, n_features=None):
        self.weights = None
        self.n_features = n_features
        
    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            prediction = np.sign(x_i.dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            
            if prediction != y_i:
                self.weights += y_i * x_i
        return self
    
    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class GradientLearning:
    def __init__(self, n_features=None):
        self.weights = None
        self.n_features = n_features
        
    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            if y_i * x_i.dot(self.weights) < 1:
                self.weights += y_i * x_i
        return self
    
    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class AROW:
    """Adaptive Regularization of Weight Vectors (AROW)."""
    def __init__(self, r=0.1, n_features=None):
        self.r = r
        self.weights = None
        self.Sigma = None
        self.n_features = n_features
    
    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            self.Sigma = np.identity(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            # Calculate hinge loss and confidence for the current sample
            lt = max(0, 1 - y_i * x_i.dot(self.weights))
            vt = x_i.T.dot(self.Sigma).dot(x_i)
            
            # Only update the model's state if there is a loss
            if lt > 0:
                # Calculate the update coefficients, beta_t and alpha_t
                beta_t = 1 / (vt + self.r) if (vt + self.r) > 0 else 0.0
                alpha_t = lt * beta_t
               
                # Update the internal state (weights and covariance matrix)
                self.weights += alpha_t * y_i * self.Sigma.dot(x_i)
                self.Sigma -= beta_t * self.Sigma.dot(np.outer(x_i, x_i)).dot(self.Sigma)
        return self
    
    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class RDA:
    """Regularized Dual Averaging (RDA)."""
    def __init__(self, lambda_param=0.01, gamma_param=0.1, n_features=None):
        self.weights = None
        self.g = None
        self.t = 0
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.n_features = n_features

    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            self.g = np.zeros(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            self.t += 1
            lt = max(0, 1 - y_i * x_i.dot(self.weights))
            gt = -y_i * x_i if lt > 0 else np.zeros_like(x_i)
            self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
            update_mask = np.abs(self.g) > self.lambda_param
            self.weights.fill(0)
            self.weights[update_mask] = -(np.sqrt(self.t) / self.gamma_param) * \
                                       (self.g[update_mask] - self.lambda_param * np.sign(self.g[update_mask]))
        return self

    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class SCW:
    """Soft Confidence-Weighted (SCW)."""
    def __init__(self, C=1.0, eta=0.9, n_features=None):
        self.phi = norm.ppf(eta)
        self.C = C
        self.weights = None
        self.Sigma = None
        self.n_features = n_features

    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            self.Sigma = np.identity(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            vt = x_i.T.dot(self.Sigma).dot(x_i)
            mt = y_i * x_i.dot(self.weights)
            lt = max(0, self.phi * np.sqrt(vt) - mt)
            if lt > 0:
                pa = 1 + (self.phi**2) / 2
                xi = 1 + self.phi**2
                sqrt_term = max(0, (mt**2 * self.phi**4 / 4) + (vt * self.phi**2 * xi))
                alpha_t = min(self.C, max(0, (1 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term))))
                sqrt_ut_term = max(0, (alpha_t**2 * vt**2 * self.phi**2) + (4 * vt))
                ut = 0.25 * (-alpha_t * vt * self.phi + np.sqrt(sqrt_ut_term))**2
                beta_t = (alpha_t * self.phi) / (np.sqrt(ut) + vt * alpha_t * self.phi + 1e-8)
                self.weights += alpha_t * y_i * self.Sigma.dot(x_i)
                self.Sigma -= beta_t * self.Sigma.dot(np.outer(x_i, x_i)).dot(self.Sigma)
        return self

    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

class AdaRDA:
    """Adaptive Regularized Dual Averaging (AdaRDA)."""
    def __init__(self, lambda_param=0.01, eta_param=0.1, delta_param=0.1, n_features=None):
        self.weights = None
        self.g = None
        self.g1t = None
        self.t = 0
        self.lambda_param = lambda_param
        self.eta_param = eta_param
        self.delta_param = delta_param
        self.n_features = n_features

    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            self.g = np.zeros(self.n_features)
            self.g1t = np.zeros(self.n_features)
        
        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = 2 * y[i] - 1  # Convert {0,1} to {-1,+1}
            
            self.t += 1
            lt = max(0, 1 - y_i * x_i.dot(self.weights))
            gt = -y_i * x_i if lt > 0 else np.zeros_like(x_i)
            self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
            self.g1t += gt**2
            Ht = self.delta_param + np.sqrt(self.g1t)
            update_mask = np.abs(self.g) > self.lambda_param
            self.weights.fill(0)
            self.weights[update_mask] = np.sign(-self.g[update_mask]) * self.eta_param * self.t / (Ht[update_mask] + 1e-8)
        return self

    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            prediction = prediction if prediction != 0 else 1
            predictions.append(1 if prediction > 0 else 0)  # Convert {-1,+1} to {0,1}
        return np.array(predictions)

# Advanced Feature Engineering Functions
def create_interaction_features(X, max_interactions=200):
    """Create cybersecurity-focused interaction features (memory-efficient)"""
    print("  Creating cyber-security interaction features...")
    n_features = X.shape[1]
    
    if n_features > 30:  # Limit to prevent memory explosion
        # Select top features first using variance
        feature_vars = np.var(X, axis=0)
        top_indices = np.argsort(feature_vars)[-30:]  # Top 30 most variable features
        X_selected = X[:, top_indices]
    else:
        X_selected = X
    
    # Create targeted quadratic interactions for cybersecurity patterns
    interactions = []
    n_selected = X_selected.shape[1]
    
    # High-value interactions: adjacent and spread features
    for i in range(min(15, n_selected)):
        for j in range(i+1, min(15, n_selected)):
            # Multiplicative interaction
            interactions.append(X_selected[:, i] * X_selected[:, j])
            
            # Ratio-based interaction (for rate-like features)
            denom = np.where(np.abs(X_selected[:, j]) > 1e-8, X_selected[:, j], 1e-8)
            interactions.append(X_selected[:, i] / denom)
            
            if len(interactions) >= max_interactions:
                break
        if len(interactions) >= max_interactions:
            break
    
    # Add polynomial features for non-linear patterns
    for i in range(min(10, n_selected)):
        interactions.append(X_selected[:, i] ** 2)  # Quadratic terms
        interactions.append(np.sqrt(np.abs(X_selected[:, i])))  # Square root terms
        
        if len(interactions) >= max_interactions:
            break
    
    if interactions:
        return np.column_stack([X, np.column_stack(interactions[:max_interactions])])
    return X

def create_statistical_features(X):
    """Create comprehensive statistical aggregation features for low FPR/FNR"""
    print("  Creating advanced statistical features for FPR/FNR optimization...")
    features = []
    
    # Row-wise statistics (original)
    features.append(np.mean(X, axis=1).reshape(-1, 1))      # Mean
    features.append(np.std(X, axis=1).reshape(-1, 1))       # Std
    features.append(np.min(X, axis=1).reshape(-1, 1))       # Min
    features.append(np.max(X, axis=1).reshape(-1, 1))       # Max
    features.append(np.median(X, axis=1).reshape(-1, 1))    # Median
    
    # Additional robust statistics for cybersecurity patterns
    features.append(np.percentile(X, 25, axis=1).reshape(-1, 1))  # Q1
    features.append(np.percentile(X, 75, axis=1).reshape(-1, 1))  # Q3
    features.append((np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1))  # Range
    features.append(np.sum(X > 0, axis=1).reshape(-1, 1))    # Count of positive features
    features.append(np.sum(X == 0, axis=1).reshape(-1, 1))   # Count of zero features
    
    # Entropy-like features for anomaly detection
    X_positive = np.where(X > 0, X, 1e-8)  # Avoid log(0)
    log_sum = np.sum(np.log(X_positive), axis=1)
    features.append(log_sum.reshape(-1, 1))  # Log sum for entropy approximation
    
    # Combine with original features
    return np.column_stack([X] + features)

def apply_pca_reduction(X, n_components=50):
    """Apply PCA for dimensionality reduction"""
    if X.shape[1] > n_components:
        print(f"  Applying PCA reduction: {X.shape[1]} -> {n_components} features")
        pca = PCA(n_components=n_components, random_state=1729)
        return pca.fit_transform(X), pca
    return X, None

def feature_selection(X, y, k=100):
    """Select top k features using mutual information"""
    if X.shape[1] > k:
        print(f"  Selecting top {k} features from {X.shape[1]}")
        selector = SelectKBest(mutual_info_classif, k=k)
        return selector.fit_transform(X, y), selector
    return X, None

def load_single_dataset(parquet_file, feature_engineering=True, memory_limit_mb=500):
    """
    Load and preprocess a single parquet file with memory management
    """
    print(f"\n  Loading {parquet_file.name}...")
    df = pd.read_parquet(parquet_file)
    
    # Memory check
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Memory usage: {memory_usage:.1f} MB")
    
    if memory_usage > memory_limit_mb:
        print(f"    Dataset too large, sampling to reduce memory...")
        sample_frac = memory_limit_mb / memory_usage * 0.8
        df = df.sample(frac=sample_frac, random_state=1729)
        print(f"    Sampled to {len(df)} rows")
    
    print(f"    Dataset shape: {df.shape}")
    
    # Basic preprocessing
    if 'label' in df.columns:
        y = df['label'].values.astype(bool).astype(int)
    else:
        print("    Warning: No 'label' column found, creating dummy labels")
        y = np.zeros(len(df))
    
    # Drop non-feature columns but preserve user_id for splitting
    feature_cols = [col for col in df.columns 
                   if col not in ['label', 'timestamp', 'category', 'entity']]
    
    # Handle categorical columns
    X_df = df[feature_cols].copy()
    
    # Preserve user_id for splitting, then handle categoricals
    user_id_col = None
    if 'user_id' in X_df.columns:
        user_id_col = X_df['user_id'].copy()
        categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
        if 'user_id' in categorical_cols:
            categorical_cols.remove('user_id')  # Don't encode user_id
    else:
        categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    
    # Convert categorical to dummy variables (memory-efficient)
    if len(categorical_cols) > 0:
        print(f"    Processing {len(categorical_cols)} categorical columns...")
        X_df = pd.get_dummies(X_df, columns=categorical_cols, sparse=True)
    
    # Remove user_id from features after preserving it
    if 'user_id' in X_df.columns:
        X_df = X_df.drop(['user_id'], axis=1)
    
    # Convert to numpy array
    X = X_df.values.astype(np.float32)  # Use float32 to save memory
    
    # Remove constant features
    non_constant = np.var(X, axis=0) > 1e-8
    X = X[:, non_constant]
    print(f"    After removing constant features: {X.shape[1]} features")
    
    # Clean up
    del df
    del X_df
    gc.collect()
    
    # Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"    Class distribution: {dict(zip(unique_labels, counts))}")
    
    if len(unique_labels) == 1:
        print(f"    Warning: Only one class present, skipping...")
        return None, None, None, None, None
    
    # Feature engineering
    if feature_engineering and X.shape[0] > 100:  # Only if enough samples
        print("  Applying feature engineering...")
        
        # Statistical features
        X = create_statistical_features(X)
        
        # Feature selection to control memory
        if X.shape[1] > 200:
            X, selector = feature_selection(X, y, k=200)
        else:
            selector = None
            
        # Interaction features (limited)
        if X.shape[1] <= 50:  # Only for smaller feature sets
            X = create_interaction_features(X, max_interactions=50)
            
        # PCA reduction if too many features
        if X.shape[1] > 100:
            X, pca = apply_pca_reduction(X, n_components=100)
        else:
            pca = None
            
        print(f"    Final feature shape: {X.shape}")
    else:
        selector = None
        pca = None
    
    # User-based splitting (following RBD24 logic)
    if user_id_col is not None:
        print("    Applying user-based splitting (no user overlap)...")
        uids = user_id_col.values
        unique_uids = sorted(set(uids))  # guarantee repeatability
        
        # Shuffle users deterministically with RBD24 random state
        rng = np.random.RandomState(1729)  # Same as RBD24 default
        rng.shuffle(unique_uids)
        
        # Split users (20% test users - same as RBD24 default)
        num_test_users = max(1, int(len(unique_uids) * 0.2))
        test_uids = set(unique_uids[:num_test_users])
        
        # Create train/test indices based on user membership
        test_indices = [i for i, uid in enumerate(uids) if uid in test_uids]
        train_indices = [i for i, uid in enumerate(uids) if uid not in test_uids]
        
        X_train, X_test = X[train_indices], X[test_indices] 
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"    User-based split: {len(unique_uids)} users -> {len(test_uids)} test users (20%)")
        print(f"    No user overlap: train users ∩ test users = ∅")
    else:
        print("    No user_id found, falling back to temporal splitting...")
        # Temporal splitting (80% train, 20% test - consistent with RBD24)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"    Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test, scaler

def calculate_metrics(y_true, y_pred):
    """Calculate FNR, FPR, F1, and Accuracy"""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return fnr, fpr, f1, acc
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0

def apply_kernel_features(X, algorithm_name, original_features):
    """
    Apply memory-efficient kernel feature transformations
    """
    try:
        if algorithm_name in ['PassiveAggressive', 'Perceptron', 'GradientLearning']:
            # Light polynomial features (only interactions, no higher powers)
            if original_features <= 50:
                # Only for smaller feature sets to avoid memory explosion
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_transformed = poly.fit_transform(X)
                # Limit to reasonable size
                if X_transformed.shape[1] > 200:
                    from sklearn.feature_selection import SelectKBest, f_classif
                    selector = SelectKBest(f_classif, k=200)
                    y_dummy = np.random.randint(0, 2, X.shape[0])  # Dummy labels for feature selection
                    X_transformed = selector.fit_transform(X_transformed, y_dummy)
                return X_transformed
            else:
                return X  # Too many features, use original
        
        elif algorithm_name in ['RDA', 'AdaRDA']:
            # Small RBF approximation
            n_components = min(50, X.shape[1])
            from sklearn.kernel_approximation import RBFSampler
            rbf_feature = RBFSampler(gamma=0.1, n_components=n_components, random_state=1729)
            X_transformed = rbf_feature.fit_transform(X)
            return X_transformed
        
        elif algorithm_name == 'SCW':
            # Very light transformation - just add squared features for most important features
            if X.shape[1] <= 30:
                X_squared = X**2
                X_transformed = np.column_stack([X, X_squared])
                return X_transformed
            else:
                return X
        
        else:
            # No transformation for AROW and others
            return X
            
    except Exception as e:
        print(f"      Kernel transform failed: {e}, using original features")
        return X

def evaluate_algorithm_online_to_batch(algorithm_name, algorithm_class, X_train, y_train, X_test, y_test, n_epochs=10):
    """
    OnlineToBatch Protocol Implementation
    - Algorithm-specific epoch optimization for efficiency
    - User-based train/test split with sample-based validation split
    """
    try:
        import copy
        
        start_time = time.time()
        
        # Algorithm-specific epoch optimization
        if algorithm_name == 'AROW':
            n_epochs = 3  # Expensive covariance updates
        elif algorithm_name == 'SCW':
            n_epochs = 4  # Moderate covariance complexity
        elif algorithm_name in ['RDA', 'AdaRDA']:
            n_epochs = 5  # Complex internal state
        else:
            n_epochs = 8  # Simple first-order algorithms
        
        # Split training data into train/validation for parameter selection
        from sklearn.model_selection import train_test_split
        try:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=1729, stratify=y_train
            )
        except ValueError:
            # If stratification fails (single class), do regular split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=1729
            )
        
        # (2) Initialize best score s_OPT ← -1.0
        # (3) Initialize OPT parameters θ_OPT ← θ₁
        best_algorithm = None
        best_f1_score = -1.0
        best_epoch = 0
        
        train_start = time.time()
        
        # OnlineToBatch Protocol: FOR e=1,2,...,E
        for epoch in range(n_epochs):
            # (A) Introduce stochasticity by shuffling
            indices = np.random.RandomState(epoch + 1729).permutation(len(X_train_split))
            X_shuffled = X_train_split[indices]
            y_shuffled = y_train_split[indices]
            
            # (B) Train on shuffled data
            if epoch == 0:
                # (1) Initialize parameters θ₁ from ALG
                current_algorithm = algorithm_class()
            else:
                # θ ← θ^(e-1) - Start from previous epoch's weights
                if best_algorithm is not None:
                    try:
                        current_algorithm = copy.deepcopy(best_algorithm)
                    except Exception as e:
                        print(f"      Copy failed for {algorithm_name}: {e}, reinitializing")
                        current_algorithm = algorithm_class()
                else:
                    current_algorithm = algorithm_class()
            
            # UpdateRule for each instance in D'_train
            current_algorithm.partial_fit(X_shuffled, y_shuffled)
            
            # (C) Evaluate on validation set
            y_val_pred = current_algorithm.predict(X_val)
            _, _, current_f1, _ = calculate_metrics(y_val, y_val_pred)
            
            # (D) Save the OPT: if s_current > s_OPT
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                try:
                    best_algorithm = copy.deepcopy(current_algorithm)
                except Exception as e:
                    print(f"      Best save failed for {algorithm_name}: {e}, using current")
                    best_algorithm = current_algorithm
                best_epoch = epoch + 1
        
        train_time = time.time() - train_start
        
        # Return θ_OPT: Make final predictions on test set with best algorithm
        predict_start = time.time()
        if best_algorithm is None:
            # Fallback if no valid algorithm found
            best_algorithm = algorithm_class()
            best_algorithm.partial_fit(X_train_split, y_train_split)
            
        y_pred = best_algorithm.predict(X_test)
        predict_time = time.time() - predict_start
        
        # Calculate final metrics
        fnr, fpr, f1, acc = calculate_metrics(y_test, y_pred)
        
        # Print results
        print(f"    FNR: {fnr:.4f}, FPR: {fpr:.4f}")
        print(f"    F1: {f1:.4f}, Acc: {acc:.4f}")
        print(f"    Time: {train_time:.3f}s train, {predict_time:.3f}s predict")
        print(f"    Best epoch: {best_epoch}/{n_epochs}, Val F1: {best_f1_score:.4f}")
        
        return {
            'fnr': fnr,
            'fpr': fpr,
            'f1': f1,
            'accuracy': acc,
            'train_time': train_time,
            'predict_time': predict_time,
            'total_time': time.time() - start_time,
            'best_epoch': best_epoch,
            'val_f1_score': best_f1_score,
            'n_epochs': n_epochs
        }
        
    except Exception as e:
        print(f"    Error with {algorithm_name}: {str(e)}")
        return None
        # (3) Initialize OPT parameters θ_OPT ← θ₁
        best_algorithm = None
        best_f1_score = -1.0
        best_epoch = 0
        
        train_start = time.time()
        
        # OnlineToBatch Protocol: FOR e=1,2,...,E
        for epoch in range(n_epochs):
            # (A) Introduce stochasticity by shuffling
            indices = np.random.RandomState(epoch + 1729).permutation(len(X_train_split))
            X_shuffled = X_train_split[indices]
            y_shuffled = y_train_split[indices]
            
            # (B) Train on shuffled data
            if epoch == 0:
                # (1) Initialize parameters θ₁ from ALG
                current_algorithm = algorithm_class()
            else:
                # θ ← θ^(e-1) - Start from previous epoch's weights
                if best_algorithm is not None:
                    try:
                        current_algorithm = copy.deepcopy(best_algorithm)
                    except Exception as e:
                        print(f"      Copy failed for {algorithm_name}: {e}, reinitializing")
                        current_algorithm = algorithm_class()
                else:
                    current_algorithm = algorithm_class()
            
            # UpdateRule for each instance in D'_train
            current_algorithm.partial_fit(X_shuffled, y_shuffled)
            
            # (C) Evaluate on validation set
            y_val_pred = current_algorithm.predict(X_val)
            _, _, current_f1, _ = calculate_metrics(y_val, y_val_pred)
            
            # (D) Save the OPT: if s_current > s_OPT
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                try:
                    best_algorithm = copy.deepcopy(current_algorithm)
                except Exception as e:
                    print(f"      Best save failed for {algorithm_name}: {e}, using current")
                    best_algorithm = current_algorithm
                best_epoch = epoch + 1
        
        train_time = time.time() - train_start
        
        # Return θ_OPT: Make final predictions on test set with best algorithm
        predict_start = time.time()
        if best_algorithm is None:
            # Fallback if no valid algorithm found
            best_algorithm = algorithm_class()
            best_algorithm.partial_fit(X_train_split, y_train_split)
            
        y_pred = best_algorithm.predict(X_test)
        predict_time = time.time() - predict_start
        
        # Calculate final metrics
        fnr, fpr, f1, acc = calculate_metrics(y_test, y_pred)
        
        # Print results
        print(f"    FNR: {fnr:.4f}, FPR: {fpr:.4f}")
        print(f"    F1: {f1:.4f}, Acc: {acc:.4f}")
        print(f"    Time: {train_time:.3f}s train, {predict_time:.3f}s predict")
        print(f"    Best epoch: {best_epoch}/{n_epochs}, Val F1: {best_f1_score:.4f}")
        
        return {
            'fnr': fnr,
            'fpr': fpr,
            'f1': f1,
            'accuracy': acc,
            'train_time': train_time,
            'predict_time': predict_time,
            'total_time': time.time() - start_time,
            'best_epoch': best_epoch,
            'val_f1_score': best_f1_score,
            'n_epochs': n_epochs
        }
        
    except Exception as e:
        print(f"    Error with {algorithm_name}: {str(e)}")
        return None

def main():
    print("Advanced 7-Algorithm Cybersecurity Evaluation with RBD24 Data Loading")
    print("Memory-Efficient Implementation with Feature Engineering")
    print("=" * 80)
    
    # Data directory
    data_dir = Path('/home/trader/Downloads/extreme-xp-main/master-experiment-suite/batch/cyber')
    
    # Find all parquet files
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    
    if not parquet_files:
        print(f"No .parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} datasets")
    
    # Initialize algorithm classes (not instances) for OnlineToBatch protocol
    algorithms = {
        'PassiveAggressive': PassiveAggressive,
        'Perceptron': Perceptron,
        'GradientLearning': GradientLearning,
        'AROW': lambda: AROW(r=0.1),
        'RDA': lambda: RDA(lambda_param=0.01, gamma_param=0.1),
        'SCW': lambda: SCW(C=1.0, eta=0.9),
        'AdaRDA': lambda: AdaRDA(lambda_param=0.01, eta_param=0.1, delta_param=0.1)
    }
    
    results = []
    
    print(f"\nStarting evaluation on {len(parquet_files)} datasets × {len(algorithms)} algorithms")
    print(f"Target: {len(parquet_files) * len(algorithms)} evaluations")
    
    # Process datasets with progress bar
    for file_idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing datasets")):
        try:
            # Load and preprocess dataset
            X_train, y_train, X_test, y_test, scaler = load_single_dataset(
                parquet_file, feature_engineering=True, memory_limit_mb=300
            )
            
            if X_train is None:
                print(f"  Skipping {parquet_file.name} (single class)")
                continue
            
            dataset_name = parquet_file.stem
            
            print(f"\n{'=' * 60}")
            print(f"EVALUATING: {dataset_name}")
            print(f"{'=' * 60}")
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"Features: {X_train.shape[1]}")
            
            train_pos = np.sum(y_train)
            train_neg = len(y_train) - train_pos
            test_pos = np.sum(y_test)
            test_neg = len(y_test) - test_pos
            
            print(f"Train class dist: negative={train_neg}, positive={train_pos}")
            print(f"Test class dist: negative={test_neg}, positive={test_pos}")
            
            # Evaluate each algorithm using OnlineToBatch protocol
            for alg_name, algorithm_class in algorithms.items():
                print(f"\n  Training {alg_name}...")
                
                # OnlineToBatch Protocol Implementation
                try:
                    result = evaluate_algorithm_online_to_batch(
                        alg_name, algorithm_class, X_train, y_train, X_test, y_test
                    )
                    
                    if result is not None:
                        # Store results
                        results.append({
                            'dataset': dataset_name,
                            'algorithm': alg_name,
                            'fnr': result['fnr'],
                            'fpr': result['fpr'],
                            'f1_score': result['f1'],
                            'accuracy': result['accuracy'],
                            'train_time': result['train_time'],
                            'predict_time': result['predict_time'],
                            'train_samples': len(X_train),
                            'test_samples': len(X_test),
                            'features': X_train.shape[1],
                            'best_epoch': result['best_epoch'],
                            'val_f1_score': result['val_f1_score'],
                            'n_epochs': result['n_epochs']
                        })
                    else:
                        # Algorithm failed
                        results.append({
                            'dataset': dataset_name,
                            'algorithm': alg_name,
                            'fnr': np.nan,
                            'fpr': np.nan,
                            'f1_score': np.nan,
                            'accuracy': np.nan,
                            'train_time': np.nan,
                            'predict_time': np.nan,
                            'train_samples': len(X_train),
                            'test_samples': len(X_test),
                            'features': X_train.shape[1] if X_train is not None else 0,
                            'best_epoch': np.nan,
                            'val_f1_score': np.nan,
                            'n_epochs': np.nan
                        })
                    
                except Exception as e:
                    print(f"    Error with {alg_name}: {str(e)}")
                    results.append({
                        'dataset': dataset_name,
                        'algorithm': alg_name,
                        'fnr': np.nan,
                        'fpr': np.nan,
                        'f1_score': np.nan,
                        'accuracy': np.nan,
                        'train_time': np.nan,
                        'predict_time': np.nan,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features': X_train.shape[1] if X_train is not None else 0,
                        'best_epoch': np.nan,
                        'val_f1_score': np.nan,
                        'n_epochs': np.nan
                    })
            
            # Memory cleanup
            del X_train, X_test, y_train, y_test, scaler
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {parquet_file.name}: {str(e)}")
            continue
    
    print(f"\nCompleted evaluations on {len(parquet_files)} datasets")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = 'advanced_custom_algorithms_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("ADVANCED 7-ALGORITHM EVALUATION SUMMARY")
    print("=" * 80)
    
    # Check if we have any results with required columns
    if 'fnr' not in results_df.columns or 'fpr' not in results_df.columns:
        print("ERROR: No successful evaluations completed.")
        print("All datasets failed during processing.")
        print("Check the error messages above for details.")
        return
    
    # Filter out failed runs
    valid_results = results_df.dropna(subset=['fnr', 'fpr'])
    
    if len(valid_results) == 0:
        print("No valid results to summarize.")
        return
    
    print(f"\n1. OVERALL PERFORMANCE:")
    print("─" * 40)
    print(f"Total evaluations: {len(valid_results)}")
    print(f"Average FNR: {valid_results['fnr'].mean():.4f} ± {valid_results['fnr'].std():.4f}")
    print(f"Average FPR: {valid_results['fpr'].mean():.4f} ± {valid_results['fpr'].std():.4f}")
    print(f"Average F1 Score: {valid_results['f1_score'].mean():.4f} ± {valid_results['f1_score'].std():.4f}")
    print(f"Average Accuracy: {valid_results['accuracy'].mean():.4f} ± {valid_results['accuracy'].std():.4f}")
    
    print(f"\n2. ALGORITHM PERFORMANCE COMPARISON:")
    print("─" * 50)
    
    # Group by algorithm
    algo_stats = valid_results.groupby('algorithm').agg({
        'fnr': ['mean', 'std'],
        'fpr': ['mean', 'std'], 
        'f1_score': ['mean', 'std']
    }).round(4)
    
    print("\nAlgorithm Ranking by Lowest FNR (False Negative Rate):")
    fnr_ranking = valid_results.groupby('algorithm')['fnr'].mean().sort_values()
    for i, (alg, fnr) in enumerate(fnr_ranking.items(), 1):
        fpr = valid_results[valid_results['algorithm'] == alg]['fpr'].mean()
        f1 = valid_results[valid_results['algorithm'] == alg]['f1_score'].mean()
        print(f"  {i:2d}. {alg:>18s}: FNR={fnr:.4f}, FPR={fpr:.4f}, F1={f1:.4f}")
    
    print("\nAlgorithm Ranking by Lowest FPR (False Positive Rate):")
    fpr_ranking = valid_results.groupby('algorithm')['fpr'].mean().sort_values()
    for i, (alg, fpr) in enumerate(fpr_ranking.items(), 1):
        fnr = valid_results[valid_results['algorithm'] == alg]['fnr'].mean()
        f1 = valid_results[valid_results['algorithm'] == alg]['f1_score'].mean()
        print(f"  {i:2d}. {alg:>18s}: FPR={fpr:.4f}, FNR={fnr:.4f}, F1={f1:.4f}")
    
    print(f"\n3. BEST PERFORMERS:")
    print("─" * 25)
    
    if len(valid_results) > 0:
        best_fnr = valid_results.loc[valid_results['fnr'].idxmin()]
        print(f"Lowest FNR: {best_fnr['algorithm']} on {best_fnr['dataset']}")
        print(f"  FNR={best_fnr['fnr']:.4f}, FPR={best_fnr['fpr']:.4f}, F1={best_fnr['f1_score']:.4f}")
        
        best_fpr = valid_results.loc[valid_results['fpr'].idxmin()]
        print(f"Lowest FPR: {best_fpr['algorithm']} on {best_fpr['dataset']}")
        print(f"  FPR={best_fpr['fpr']:.4f}, FNR={best_fpr['fnr']:.4f}, F1={best_fpr['f1_score']:.4f}")
        
        best_f1 = valid_results.loc[valid_results['f1_score'].idxmax()]
        print(f"Highest F1: {best_f1['algorithm']} on {best_f1['dataset']}")
        print(f"  F1={best_f1['f1_score']:.4f}, FNR={best_f1['fnr']:.4f}, FPR={best_f1['fpr']:.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Advanced evaluation with feature engineering completed")
    print("Memory-efficient processing with FNR/FPR optimization")
    print("=" * 80)

if __name__ == "__main__":
    main()
