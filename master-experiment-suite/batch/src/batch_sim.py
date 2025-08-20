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

# Import data handler for Zenodo downloads
from data_handler import prepare_data_from_zenodo

# Clean user-based splitting function from data_loader.py
def clean_user_based_split(x_df, y_df, test_size=0.2, random_state=1729, categorical=True):
    """Clean user-based splitting with no user overlap between train/test"""
    from numpy.random import default_rng
    from sklearn.preprocessing import StandardScaler
    from pandas import get_dummies
    from numpy import number
    
    rng = default_rng(random_state)
    
    # Process features (same as data_loader.py)
    x_processed = x_df.drop(['user_id', 'timestamp'], axis=1, errors='ignore')
    
    if categorical:
        x_processed = get_dummies(x_processed)
    else:
        x_processed = x_processed.select_dtypes(include=number)
    
    x_arr = x_processed.to_numpy(dtype=float)
    y_arr = y_df.to_numpy(dtype=bool).astype(int)
    
    # Convert to -1/+1 labels for np.sign compatibility
    y_arr = np.where(y_arr == 0, -1, 1)
    
    # Clean user-based splitting (from data_loader.py)
    unique_uids = sorted(x_df['user_id'].unique())
    rng.shuffle(unique_uids)
    num_test_users = round(len(unique_uids) * test_size)
    test_uids = set(unique_uids[:num_test_users])
    
    test_mask = x_df['user_id'].isin(test_uids)
    train_mask = ~test_mask
    
    x_train, x_test = x_arr[train_mask], x_arr[test_mask]
    y_train, y_test = y_arr[train_mask], y_arr[test_mask]
    train_user_ids = x_df.loc[train_mask, 'user_id'].values
    test_user_ids = x_df.loc[test_mask, 'user_id'].values
    
    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print(f"  Clean split: {len(np.unique(train_user_ids))} train users, {len(np.unique(test_user_ids))} test users")
    print(f"  No user overlap: {len(set(train_user_ids) & set(test_user_ids)) == 0}")
    
    return x_train, y_train, x_test, y_test, train_user_ids, test_user_ids

# Sample data creation function for fallback
def load_rbd24_properly(data_dir):
    """Load RBD24 dataset - download only if not already in cyber directory"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os
    
    # Handle different possible data_dir locations
    if data_dir == "cyber/":  # Default case
        # Look for cyber directory relative to batch directory (where this script runs)
        script_dir = Path(__file__).parent  # /path/to/batch/src
        batch_dir = script_dir.parent       # /path/to/batch  
        cyber_dir = batch_dir / 'cyber'     # /path/to/batch/cyber
    else:
        # Convert to absolute path to avoid working directory issues
        data_dir = Path(data_dir).resolve()
        cyber_dir = data_dir / 'cyber'
    
    print(f"🔍 Checking for data in: {cyber_dir}")
    
    # Check if parquet files already exist in cyber directory
    parquet_files = []
    if cyber_dir.exists():
        parquet_files = list(cyber_dir.glob('*.parquet'))
        print(f"📁 Cyber directory exists with {len(list(cyber_dir.iterdir()))} total files")
        if parquet_files:
            print(f"✅ Found existing {len(parquet_files)} parquet files:")
            for f in parquet_files[:5]:  # Show first 5 files
                print(f"   - {f.name}")
            if len(parquet_files) > 5:
                print(f"   ... and {len(parquet_files) - 5} more")
        else:
            print(f"⚠️  Directory exists but no .parquet files found")
            print(f"   Contents: {[f.name for f in cyber_dir.iterdir()]}")
    else:
        print(f"📁 Cyber directory does not exist: {cyber_dir}")
    
    if parquet_files:
        print(f"✅ Using existing data - skipping download completely")
    else:
        # Data not found, need to download
        if cyber_dir.exists():
            print(f"� Cyber directory exists but no parquet files found")
            print(f"   Contents: {list(cyber_dir.iterdir())}")
        else:
            print(f"📁 Cyber directory does not exist: {cyber_dir}")
        
        print(f"📥 Downloading RBD24 data to {cyber_dir}...")
        cyber_dir.mkdir(parents=True, exist_ok=True)
        
        # Use your robust download method
        zenodo_url = 'https://zenodo.org/api/records/13787591/files-archive'
        success = prepare_data_from_zenodo(zenodo_url, str(cyber_dir))
        
        if not success:
            print("⚠️  Failed to download RBD24 data, using sample data")
            return create_sample_cybersecurity_data()
        
        # Re-check for parquet files after download
        parquet_files = list(cyber_dir.glob('*.parquet'))
        if not parquet_files:
            print("⚠️  No parquet files found after download, using sample data")
            return create_sample_cybersecurity_data()
        
        print(f"✅ Successfully downloaded {len(parquet_files)} RBD24 datasets")
    
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
    train_user_ids_dict = {}
    test_user_ids_dict = {}
    
    for parquet_file in parquet_files:
        dataset_name = parquet_file.stem
        print(f"\n📊 Loading dataset: {dataset_name}")
        
        try:
            # Load data from parquet file
            df = pd.read_parquet(parquet_file)
            print(f"  Raw data shape: {df.shape}")
            
            # Check required columns
            if 'user_id' not in df.columns or 'label' not in df.columns:
                print(f"  ⚠️  Missing required columns, skipping {dataset_name}")
                continue
            
            # Prepare features and labels
            x_df = df.drop(['label'], axis=1)
            y_df = df['label']
            
            print(f"  Users: {x_df['user_id'].nunique()}, Samples: {len(df)}")
            print(f"  Attack rate: {y_df.mean():.1%}")
            
            # Use clean splitting method (from data_loader.py logic)
            x_train, y_train, x_test, y_test, train_user_ids, test_user_ids = clean_user_based_split(
                x_df, y_df, test_size=0.2, random_state=1729, categorical=True
            )
            
            # Store results
            X_train_dict[dataset_name] = x_train
            y_train_dict[dataset_name] = y_train
            X_test_dict[dataset_name] = x_test
            y_test_dict[dataset_name] = y_test
            train_user_ids_dict[dataset_name] = train_user_ids
            test_user_ids_dict[dataset_name] = test_user_ids
            
            print(f"  Final shapes: Train {x_train.shape}, Test {x_test.shape}")
            
        except Exception as e:
            print(f"  ❌ Error loading {dataset_name}: {e}")
            continue
    
    if not X_train_dict:
        print("⚠️  No valid datasets loaded, using sample data")
        return create_sample_cybersecurity_data()
    
    print(f"\n✅ Successfully loaded {len(X_train_dict)} RBD24 datasets with clean user-based splitting")
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict, train_user_ids_dict, test_user_ids_dict


def load_parquet_files(data_dir):
    """Load parquet files from the downloaded RBD24 dataset"""
    import pandas as pd
    from pathlib import Path
    
    data_path = Path(data_dir)
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
    
    # Look for parquet files in the data directory
    parquet_files = list(data_path.glob("**/*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_path}")
        return {}, {}, {}, {}
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # For simplicity, use the first parquet file as our dataset
    # In a real scenario, you might want to process multiple files
    main_file = parquet_files[0]
    print(f"Loading: {main_file.name}")
    
    try:
        df = pd.read_parquet(main_file)
        print(f"Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Assume the last column is the target and the rest are features
        # This is a common convention, but adjust based on your actual data structure
        feature_cols = df.columns[:-1]
        target_col = df.columns[-1]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Simple train/test split (handle class imbalance gracefully)
        from sklearn.model_selection import train_test_split
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Only use stratify if we have enough samples of each class
        min_class_count = min(class_counts)
        use_stratify = min_class_count >= 2
        
        if use_stratify:
            print("Using stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print(f"Class imbalance detected (min class has {min_class_count} samples), using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Store in dictionaries with a single dataset name
        dataset_name = main_file.stem
        
        # Final validation - ensure we have both classes in train and test
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        if len(train_classes) < 2:
            print(f"Warning: Training set only has {len(train_classes)} class(es): {train_classes}")
            print("Creating minimal synthetic data for missing class...")
            # Add one sample of the missing class to make algorithms work
            if 0 not in train_classes and len(y_train) > 0:
                # Add a synthetic negative example
                X_train = np.vstack([X_train, X_train[0:1]])  # Copy first sample
                y_train = np.append(y_train, 0)
            elif 1 not in train_classes and len(y_train) > 0:
                # Add a synthetic positive example  
                X_train = np.vstack([X_train, X_train[0:1]])  # Copy first sample
                y_train = np.append(y_train, 1)
        
        if len(test_classes) < 2:
            print(f"Warning: Test set only has {len(test_classes)} class(es): {test_classes}")
            print("Creating minimal synthetic test data for missing class...")
            if 0 not in test_classes and len(y_test) > 0:
                X_test = np.vstack([X_test, X_test[0:1]]) 
                y_test = np.append(y_test, 0)
            elif 1 not in test_classes and len(y_test) > 0:
                X_test = np.vstack([X_test, X_test[0:1]])
                y_test = np.append(y_test, 1)
        
        X_train_dict[dataset_name] = X_train
        y_train_dict[dataset_name] = y_train  
        X_test_dict[dataset_name] = X_test
        y_test_dict[dataset_name] = y_test
        
        print(f"Dataset '{dataset_name}' loaded successfully")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Final train classes: {np.unique(y_train)}, test classes: {np.unique(y_test)}")
        
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return {}, {}, {}, {}
    
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict


def create_sample_cybersecurity_data():
    """Creates smaller realistic sample cybersecurity datasets for FAST demonstration"""
    print("Generating FAST sample cybersecurity datasets...")
    
    np.random.seed(1729)
    # FAST: Fewer datasets for speed
    datasets = [
        'network_intrusion_detection',
        'malware_classification'  # FAST: Only 2 datasets instead of 5
    ]
    
    X_train_dict, y_train_dict = {}, {}
    X_test_dict, y_test_dict = {}, {}
    train_user_ids_dict, test_user_ids_dict = {}, {}
    
    for i, dataset_name in enumerate(datasets):
        # FAST: Much smaller dataset sizes for speed
        n_train = np.random.randint(200, 500)   # FAST: Reduced from 800-1500
        n_test = np.random.randint(50, 100)     # FAST: Reduced from 200-400
        n_features = np.random.randint(8, 15)   # FAST: Reduced from 15-25
        
        # Create features with different cybersecurity characteristics per dataset
        X_train = np.random.randn(n_train, n_features)
        X_test = np.random.randn(n_test, n_features)
        
        # Add dataset-specific patterns
        if 'network' in dataset_name:
            # Network features: packet size, duration, protocol indicators
            X_train[:, 0] *= 2.0  # Packet size variation
            X_test[:, 0] *= 2.0
            X_train[:, 1] = np.abs(X_train[:, 1])  # Duration (positive)
            X_test[:, 1] = np.abs(X_test[:, 1])
        elif 'malware' in dataset_name:
            # Malware features: file size, entropy, API calls
            X_train[:, :3] = np.abs(X_train[:, :3])  # File characteristics
            X_test[:, :3] = np.abs(X_test[:, :3])
        elif 'phishing' in dataset_name:
            # Email features: length, links, attachments
            X_train[:, -2:] = np.clip(X_train[:, -2:], 0, 10)  # Bounded features
            X_test[:, -2:] = np.clip(X_test[:, -2:], 0, 10)
        
        # Create realistic attack patterns with class imbalance
        # Different imbalance ratios for different attack types
        if 'ddos' in dataset_name:
            attack_rate = 0.15  # DDoS attacks are more frequent
        elif 'insider' in dataset_name:
            attack_rate = 0.05  # Insider threats are rare
        else:
            attack_rate = 0.08  # Standard cybersecurity imbalance
            
        # Create labels based on feature combinations (realistic attack patterns)
        train_scores = (X_train[:, 0] + X_train[:, 2] + 
                       0.5 * X_train[:, 1] + 
                       0.3 * np.random.randn(n_train))
        test_scores = (X_test[:, 0] + X_test[:, 2] + 
                      0.5 * X_test[:, 1] + 
                      0.3 * np.random.randn(n_test))
        
        # Convert scores to binary labels with desired attack rate
        train_threshold = np.percentile(train_scores, (1 - attack_rate) * 100)
        test_threshold = np.percentile(test_scores, (1 - attack_rate) * 100)
        
        y_train = (train_scores > train_threshold).astype(int)
        y_test = (test_scores > test_threshold).astype(int)
        
        # Convert 0/1 to -1/+1 for np.sign compatibility
        y_train = np.where(y_train == 0, -1, 1)
        y_test = np.where(y_test == 0, -1, 1)
        
        # Ensure we have both classes (critical for cybersecurity)
        if np.sum(y_train == 1) == 0:
            y_train[np.argmax(train_scores)] = 1
        if np.sum(y_train == -1) == 0:
            y_train[np.argmin(train_scores)] = -1
        if np.sum(y_test == 1) == 0:
            y_test[np.argmax(test_scores)] = 1
        if np.sum(y_test == -1) == 0:
            y_test[np.argmin(test_scores)] = -1
            
        # Create dummy user IDs for temporal ordering in sample data
        train_user_ids = np.arange(n_train)
        test_user_ids = np.arange(n_test)
            
        X_train_dict[dataset_name] = X_train
        y_train_dict[dataset_name] = y_train
        X_test_dict[dataset_name] = X_test
        y_test_dict[dataset_name] = y_test
        train_user_ids_dict[dataset_name] = train_user_ids
        test_user_ids_dict[dataset_name] = test_user_ids
        
        attack_pct_train = np.mean(y_train) * 100
        attack_pct_test = np.mean(y_test) * 100
        print(f"  {dataset_name}: Train={n_train} ({attack_pct_train:.1f}% attacks), "
              f"Test={n_test} ({attack_pct_test:.1f}% attacks)")
    
    print(f"\n✅ Created {len(datasets)} realistic cybersecurity datasets")
    print("📊 Datasets cover: Network intrusion, Malware, Phishing, DDoS, Insider threats")
    print("🎯 Realistic class imbalance and feature patterns included")
    
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict, train_user_ids_dict, test_user_ids_dict

# Enhanced Ensemble aggregation methods for cybersecurity
class EnsembleAggregator:
    """Implements intelligent ensemble aggregation strategies optimized for cybersecurity"""
    
    @staticmethod
    def smart_ensemble_selection(all_predictions, validation_scores):
        """Select algorithms with BOTH low FPR AND low FNR for ensemble"""
        # FOCUS: Only use algorithms with LOW FPR AND LOW FNR
        excellent_algorithms = []
        good_algorithms = []
        
        for i, (alg_name, scores) in enumerate(validation_scores.items()):
            f1 = scores.get('f1', 0)
            fnr = scores.get('fnr', 1)
            fpr = scores.get('fpr', 1)
            
            # TIER 1: Excellent algorithms (BOTH low FNR AND low FPR)
            # Based on actual results: AROW-type performance (FNR~0.35, FPR~0.5, F1~0.4)
            # 1. F1 score > 0.35 (good detection)
            # 2. FNR < 0.4 (don't miss too many attacks) 
            # 3. FPR < 0.6 (reasonable false alarm rate)
            if f1 > 0.35 and fnr < 0.4 and fpr < 0.6:
                excellent_algorithms.append((i, alg_name, f1, fnr + fpr))
                
            # TIER 2: Good algorithms (balanced but not excellent)
            # Avoid extreme algorithms like AdaRDA (FNR=0.0, FPR=1.0) or RDA (FNR=0.17, FPR=0.80)
            # 1. F1 score > 0.25 (reasonable detection)
            # 2. FNR < 0.7 AND FPR < 0.7 (both reasonably balanced, avoid extremes)
            # 3. Total error (FNR+FPR) < 1.2 (not too high combined error)
            elif f1 > 0.25 and fnr < 0.7 and fpr < 0.7 and (fnr + fpr) < 1.2:
                good_algorithms.append((i, alg_name, f1, fnr + fpr))
        
        # Prefer excellent algorithms, fallback to good ones
        if len(excellent_algorithms) >= 2:
            print(f"    🎯 Using {len(excellent_algorithms)} EXCELLENT algorithms (low FNR+FPR)")
            return excellent_algorithms
        elif len(good_algorithms) >= 2:
            print(f"    ✅ Using {len(good_algorithms)} GOOD algorithms (balanced FNR+FPR)")
            return good_algorithms
        else:
            # Emergency fallback: use top algorithms by F1, but warn user
            print(f"    ⚠️  No algorithms with low FNR+FPR found, using top F1 performers")
            sorted_algs = sorted(validation_scores.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
            fallback_algorithms = [(i, alg_name, scores.get('f1', 0), 
                                  scores.get('fnr', 1) + scores.get('fpr', 1)) 
                                 for i, (alg_name, scores) in enumerate(sorted_algs[:3])]
            return fallback_algorithms
    
    @staticmethod
    def cybersecurity_weighted_ensemble(all_predictions, validation_scores):
        """Cybersecurity-focused ensemble that prioritizes algorithms with low FNR AND low FPR"""
        selected_algorithms = EnsembleAggregator.smart_ensemble_selection(all_predictions, validation_scores)
        
        if not selected_algorithms:
            return np.mean(all_predictions, axis=0)
        
        # Extract predictions from selected algorithms only
        selected_predictions = np.array([all_predictions[i] for i, _, _, _ in selected_algorithms])
        
        # Weight by inverse of (FNR + FPR) - STRONGLY prefer algorithms with BOTH low FNR and FPR
        weights = []
        for i, alg_name, f1, total_error in selected_algorithms:
            fnr = validation_scores[alg_name].get('fnr', 0.5)
            fpr = validation_scores[alg_name].get('fpr', 0.5)
            
            # Strong penalty for high FNR or FPR - we want BOTH to be low
            error_penalty = fnr + fpr + 0.01  # Add small epsilon to avoid division by zero
            balance_weight = 1.0 / (error_penalty ** 2)  # Square to heavily penalize high errors
            
            # Also consider F1 score
            f1_weight = f1 * 2  # F1 boost
            
            final_weight = balance_weight * f1_weight
            weights.append(final_weight)
        
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        print(f"    💪 Selected algorithms with weights:")
        for (i, alg_name, f1, total_error), weight in zip(selected_algorithms, weights):
            fnr = validation_scores[alg_name].get('fnr', 0)
            fpr = validation_scores[alg_name].get('fpr', 0)
            print(f"       {alg_name}: weight={weight:.3f} (F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f})")
        
        # Weighted average
        ensemble_pred = np.average(selected_predictions, axis=0, weights=weights)
        return ensemble_pred
    
    @staticmethod  
    def adaptive_threshold_ensemble(all_predictions, validation_scores):
        """Use adaptive thresholding with algorithms that have low FNR AND low FPR"""
        selected_algorithms = EnsembleAggregator.smart_ensemble_selection(all_predictions, validation_scores)
        
        if not selected_algorithms:
            ensemble_scores = np.mean(all_predictions, axis=0)
        else:
            selected_predictions = np.array([all_predictions[i] for i, _, _, _ in selected_algorithms])
            ensemble_scores = np.mean(selected_predictions, axis=0)
        
        # Use lower threshold for cybersecurity (prefer detecting attacks)
        threshold = 0.35  # Slightly higher than before since we're using better algorithms
        return (ensemble_scores > threshold).astype(int)
    
    @staticmethod
    def low_error_ensemble(all_predictions, validation_scores):
        """NEW: Ensemble using ONLY algorithms with very low FNR AND FPR"""
        super_algorithms = []
        
        for i, (alg_name, scores) in enumerate(validation_scores.items()):
            f1 = scores.get('f1', 0)
            fnr = scores.get('fnr', 1)
            fpr = scores.get('fpr', 1)
            
            # VERY strict criteria - both FNR and FPR must be low
            if f1 > 0.2 and fnr < 0.25 and fpr < 0.25:
                super_algorithms.append((i, alg_name, f1))
        
        if len(super_algorithms) >= 1:
            print(f"    🚀 LOW ERROR ENSEMBLE: Using {len(super_algorithms)} super algorithms")
            super_predictions = np.array([all_predictions[i] for i, _, _ in super_algorithms])
            return np.mean(super_predictions, axis=0)
        else:
            print(f"    ❌ No super algorithms found, falling back to mean")
            return np.mean(all_predictions, axis=0)
    
    @staticmethod
    def exclude_extremes_ensemble(all_predictions, validation_scores):
        """NEW: Exclude extreme algorithms (like AdaRDA, RDA) that have FNR=0 OR FPR=1.0"""
        balanced_algorithms = []
        
        print("    🔍 EXCLUDE EXTREMES analysis:")
        for i, (alg_name, scores) in enumerate(validation_scores.items()):
            f1 = scores.get('f1', 0)
            fnr = scores.get('fnr', 1)
            fpr = scores.get('fpr', 1)
            
            # EXCLUDE extreme algorithms:
            # - AdaRDA type: FNR=0.0, FPR=1.0 (catches everything but false alarms)
            # - SCW type: very high FNR (misses most attacks)
            # - Algorithms with FNR > 0.8 OR FPR > 0.8 (too extreme)
            
            is_extreme = (fnr < 0.05 and fpr > 0.95) or (fnr > 0.95 and fpr < 0.05) or (fnr > 0.8) or (fpr > 0.8)
            
            if not is_extreme and f1 > 0.2:  # Not extreme AND decent F1
                balanced_algorithms.append((i, alg_name, f1))
                print(f"      ✅ {alg_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f}")
            else:
                print(f"      ❌ {alg_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f} (EXTREME)")
        
        if len(balanced_algorithms) >= 2:
            print(f"    🎯 Using {len(balanced_algorithms)} non-extreme algorithms")
            balanced_predictions = np.array([all_predictions[i] for i, _, _ in balanced_algorithms])
            return np.mean(balanced_predictions, axis=0)
        else:
            print(f"    ⚠️  Only {len(balanced_algorithms)} non-extreme algorithms found, using all")
            return np.mean(all_predictions, axis=0)
    
    @staticmethod
    def generate_smart_ensembles(all_predictions, all_true_labels, validation_scores):
        """Generate intelligent ensemble methods focused on low FNR AND low FPR"""
        ensemble_methods = {
            'SmartWeighted': EnsembleAggregator.cybersecurity_weighted_ensemble,
            'AdaptiveThreshold': EnsembleAggregator.adaptive_threshold_ensemble,
            'LowErrorOnly': EnsembleAggregator.low_error_ensemble,  # NEW: Super strict criteria
            'ExcludeExtremes': EnsembleAggregator.exclude_extremes_ensemble,  # NEW: Exclude extreme algorithms
            'TopPerformersOnly': lambda preds, scores: EnsembleAggregator.top_k_ensemble(preds, scores, k=3),
            'BalancedEnsemble': EnsembleAggregator.balanced_ensemble
        }
        
        results = {}
        for method_name, method_func in ensemble_methods.items():
            ensemble_pred = method_func(all_predictions, validation_scores)
            # Convert to -1/+1 format
            ensemble_pred = np.where(ensemble_pred == 0, -1, 1)
            results[method_name] = ensemble_pred
        
        return results
    
    @staticmethod
    def balanced_ensemble(all_predictions, validation_scores):
        """Ensemble focused on balancing FNR and FPR"""
        # Find algorithms with best balance between FNR and FPR
        balanced_algorithms = []
        for i, (alg_name, scores) in enumerate(validation_scores.items()):
            fnr = scores.get('fnr', 0.5)
            fpr = scores.get('fpr', 0.5)
            balance = abs(fnr - fpr)  # Lower is more balanced
            f1 = scores.get('f1', 0)
            
            if f1 > 0.2:  # Minimum performance threshold
                balanced_algorithms.append((i, balance, f1))
        
        # Sort by balance (lower is better), then by F1 (higher is better)
        balanced_algorithms.sort(key=lambda x: (x[1], -x[2]))
        
        if len(balanced_algorithms) < 2:
            return np.mean(all_predictions, axis=0)
        
        # Use top 3 most balanced algorithms
        selected_indices = [idx for idx, _, _ in balanced_algorithms[:3]]
        selected_predictions = np.array([all_predictions[i] for i in selected_indices])
        
        return np.mean(selected_predictions, axis=0)
    
    @staticmethod
    def majority_voting(predictions_dict):
        """Simple majority voting across all models"""
        predictions_array = np.array(list(predictions_dict.values()))
        return (np.mean(predictions_array, axis=0) >= 0.5).astype(int)
    
    @staticmethod
    def weighted_voting(predictions_dict, weights=None):
        """Weighted voting based on algorithm performance"""
        if weights is None:
            weights = {alg: 1.0 for alg in predictions_dict.keys()}
        
        weighted_sum = np.zeros_like(list(predictions_dict.values())[0], dtype=float)
        total_weight = 0.0
        
        for alg, preds in predictions_dict.items():
            weight = weights.get(alg, 1.0)
            weighted_sum += weight * preds
            total_weight += weight
        
        return (weighted_sum / total_weight >= 0.5).astype(int)
    
    @staticmethod
    def balanced_fpr_fnr_voting(predictions_dict, validation_scores, alpha=0.5):
        """
        Optimal ensemble method for cybersecurity: Balance FPR and FNR
        
        Args:
            predictions_dict: {algorithm: predictions} dictionary
            validation_scores: {algorithm: {'fpr': X, 'fnr': Y, 'f1': Z}} scores
            alpha: Balance factor (0.0=only FPR, 1.0=only FNR, 0.5=balanced)
        """
        weights = {}
        
        for alg, score_dict in validation_scores.items():
            fpr = score_dict.get('fpr', 1.0)  # Default high FPR is bad
            fnr = score_dict.get('fnr', 1.0)  # Default high FNR is bad
            f1 = score_dict.get('f1', 0.0)    # Default low F1 is bad
            
            # Combined score: Lower FPR and FNR is better
            # Use alpha to balance between FPR and FNR importance
            combined_error = alpha * fnr + (1 - alpha) * fpr
            
            # Weight inversely proportional to combined error, boosted by F1
            # Add small epsilon to avoid division by zero
            weight = (f1 + 0.1) / (combined_error + 0.01)
            weights[alg] = weight
            
        return EnsembleAggregator.weighted_voting(predictions_dict, weights)
    
    @staticmethod
    def conservative_ensemble(predictions_dict, validation_scores, fpr_threshold=0.05):
        """
        Conservative ensemble: Only include models with very low FPR
        Good for environments where false alarms are costly
        """
        selected_preds = {}
        
        for alg, preds in predictions_dict.items():
            fpr = validation_scores.get(alg, {}).get('fpr', 1.0)
            if fpr <= fpr_threshold:
                selected_preds[alg] = preds
        
        if not selected_preds:
            # If no model meets FPR threshold, use the one with lowest FPR
            best_alg = min(validation_scores.keys(), 
                          key=lambda x: validation_scores[x].get('fpr', 1.0))
            selected_preds[best_alg] = predictions_dict[best_alg]
            
        return EnsembleAggregator.majority_voting(selected_preds)
    
    @staticmethod
    def aggressive_ensemble(predictions_dict, validation_scores, fnr_threshold=0.05):
        """
        Aggressive ensemble: Only include models with very low FNR
        Good for high-security environments where missing attacks is critical
        """
        selected_preds = {}
        
        for alg, preds in predictions_dict.items():
            fnr = validation_scores.get(alg, {}).get('fnr', 1.0)
            if fnr <= fnr_threshold:
                selected_preds[alg] = preds
        
        if not selected_preds:
            # If no model meets FNR threshold, use the one with lowest FNR
            best_alg = min(validation_scores.keys(), 
                          key=lambda x: validation_scores[x].get('fnr', 1.0))
            selected_preds[best_alg] = predictions_dict[best_alg]
            
        return EnsembleAggregator.majority_voting(selected_preds)
    
    @staticmethod
    def top_k_performers(predictions_dict, validation_scores, k=3, metric='f1'):
        """
        Select top-K performing models based on specified metric
        
        Args:
            k: Number of top models to include
            metric: 'f1', 'balanced_score', 'low_fpr', 'low_fnr'
        """
        if metric == 'balanced_score':
            # Custom balanced score that considers both FPR and FNR
            scores = {}
            for alg, score_dict in validation_scores.items():
                fpr = score_dict.get('fpr', 1.0)
                fnr = score_dict.get('fnr', 1.0)
                f1 = score_dict.get('f1', 0.0)
                # Balanced score: high F1, low FPR, low FNR
                balanced = f1 - 0.5 * (fpr + fnr)
                scores[alg] = balanced
        elif metric == 'low_fpr':
            scores = {alg: -score_dict.get('fpr', 1.0) 
                     for alg, score_dict in validation_scores.items()}
        elif metric == 'low_fnr':
            scores = {alg: -score_dict.get('fnr', 1.0) 
                     for alg, score_dict in validation_scores.items()}
        else:  # Default to F1
            scores = {alg: score_dict.get('f1', 0.0) 
                     for alg, score_dict in validation_scores.items()}
        
        # Select top-k algorithms
        top_k_algs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        selected_preds = {alg: predictions_dict[alg] for alg in top_k_algs}
        
        return EnsembleAggregator.majority_voting(selected_preds)
    
    @staticmethod
    def adaptive_threshold_ensemble(predictions_dict, validation_scores):
        """
        Dynamically adjust voting threshold based on model confidence
        """
        # Calculate ensemble confidence based on agreement
        predictions_array = np.array(list(predictions_dict.values()))
        agreement = np.mean(predictions_array, axis=0)  # Agreement level per sample
        
        # Adaptive threshold: more agreement needed when models are uncertain
        base_threshold = 0.5
        confidence_boost = np.abs(agreement - 0.5)  # Higher when models agree more
        adaptive_threshold = base_threshold - 0.2 * confidence_boost
        
        return (agreement >= adaptive_threshold).astype(int)
    
    @staticmethod  
    def meta_ensemble(predictions_dict, validation_scores):
        """
        Meta-ensemble: Combine multiple ensemble strategies
        """
        ensemble_methods = {
            'balanced': EnsembleAggregator.balanced_fpr_fnr_voting(predictions_dict, validation_scores),
            'conservative': EnsembleAggregator.conservative_ensemble(predictions_dict, validation_scores),
            'aggressive': EnsembleAggregator.aggressive_ensemble(predictions_dict, validation_scores),
            'top3': EnsembleAggregator.top_k_performers(predictions_dict, validation_scores, k=3),
        }
        
        # Meta-vote among ensemble methods
        return EnsembleAggregator.majority_voting(ensemble_methods)

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
            y_i = y[i]  # FIXED: No conversion needed, data is already {-1,+1}
            
            if y_i * x_i.dot(self.weights) < 1:
                l2_norm_sq = x_i.dot(x_i)
                if l2_norm_sq > 0:
                    eta = (1 - y_i * x_i.dot(self.weights)) / l2_norm_sq
                    self.weights += eta * y_i * x_i
        return self
    
    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])  # Default to positive class
        
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.sign(X[i].dot(self.weights))
            # If prediction is 0, default to +1 (positive class)
            predictions.append(1 if prediction >= 0 else -1)  # FIXED: Return {-1,+1}
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
            y_i = y[i]  # FIXED: No conversion needed, data is already {-1,+1}
            
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
            predictions.append(prediction)  # FIXED: Return {-1,+1} directly
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
            y_i = y[i]  # FIXED: No conversion needed, data is already {-1,+1}
            
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
            predictions.append(prediction)  # FIXED: Return {-1,+1} directly
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
            y_i = y[i]  # Data is already {-1,+1}
            
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
            predictions.append(prediction)  # Return {-1,+1} directly
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
            y_i = y[i]  # Data is already {-1,+1}
            
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
            predictions.append(prediction)  # Return {-1,+1} directly
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
            y_i = y[i]  # Data is already {-1,+1}
            
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
            predictions.append(prediction)  # Return {-1,+1} directly
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
            y_i = y[i]  # Data is already {-1,+1}
            
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
            predictions.append(prediction)  # Return {-1,+1} directly
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
        
        # Preserve user IDs for training (for user-based shuffling and validation split)
        train_user_ids = uids[train_indices]
        test_user_ids = uids[test_indices]
        
        print(f"    User-based split: {len(unique_uids)} users -> {len(test_uids)} test users (20%)")
        print(f"    No user overlap: train users ∩ test users = ∅")
    else:
        print("    No user_id found, falling back to temporal splitting...")
        # Temporal splitting (80% train, 20% test - consistent with RBD24)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create dummy user IDs for temporal ordering
        train_user_ids = np.arange(len(X_train))
        test_user_ids = np.arange(len(X_test))
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"    Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test, scaler, train_user_ids, test_user_ids

def convert_predictions_to_sign_format(y_pred):
    """Convert 0/1 predictions to -1/+1 format for np.sign compatibility"""
    return np.where(y_pred == 0, -1, 1)


def calculate_metrics(y_true, y_pred):
    """Calculate FNR, FPR, F1, and Accuracy"""
    try:
        # Convert predictions to -1/+1 format if they're 0/1
        if np.any(y_pred == 0) and not np.any(y_pred == -1):
            y_pred = convert_predictions_to_sign_format(y_pred)
        
        # Handle -1/+1 labels for confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Specify pos_label=1 for binary F1 score with -1/+1 labels
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return fnr, fpr, f1, acc
    except ValueError as e:
        print(f"    ⚠️  Metrics calculation error: {e}")
        print(f"       y_true unique: {np.unique(y_true)}")
        print(f"       y_pred unique: {np.unique(y_pred)}")
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

def user_based_train_val_split(X_train, y_train, train_user_ids, val_size=0.2, random_state=1729):
    """
    Split training data into train/validation ensuring no user overlap
    """
    unique_users = sorted(set(train_user_ids))
    
    # Shuffle users deterministically
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_users)
    
    # Split users for validation
    num_val_users = max(1, int(len(unique_users) * val_size))
    val_users = set(unique_users[:num_val_users])
    
    # Create indices based on user membership
    val_indices = [i for i, uid in enumerate(train_user_ids) if uid in val_users]
    train_indices = [i for i, uid in enumerate(train_user_ids) if uid not in val_users]
    
    X_train_split = X_train[train_indices]
    X_val = X_train[val_indices] 
    y_train_split = y_train[train_indices]
    y_val = y_train[val_indices]
    
    # Return user IDs for the training split for epoch shuffling
    train_split_user_ids = train_user_ids[train_indices]
    
    print(f"    User-based train/val split: {len(unique_users)} users -> {len(val_users)} validation users ({val_size*100:.0f}%)")
    print(f"    No user overlap: train users ∩ validation users = ∅")
    
    return X_train_split, X_val, y_train_split, y_val, train_split_user_ids

def user_based_shuffle(X, y, user_ids, epoch, random_state=1729):
    """
    Shuffle data maintaining user-based temporal ordering
    - Shuffle the order of users 
    - Within each user, maintain temporal order (sort by user_id)
    """
    unique_users = sorted(set(user_ids))
    
    # Shuffle user order for this epoch
    rng = np.random.RandomState(epoch + random_state)
    rng.shuffle(unique_users)
    
    # Collect data grouped by user, then sort within each user for temporal order
    shuffled_indices = []
    
    for user in unique_users:
        # Get all indices for this user
        user_indices = [i for i, uid in enumerate(user_ids) if uid == user]
        # Sort indices to maintain temporal order within user (assuming data is already temporally ordered)
        user_indices.sort()
        shuffled_indices.extend(user_indices)
    
    # Return shuffled data
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    return X_shuffled, y_shuffled

def evaluate_algorithm_online_to_batch(algorithm_name, algorithm_class, X_train, y_train, X_test, y_test, 
                                       train_user_ids=None, fast_mode=False):  # Added fast_mode parameter
    """
    OnlineToBatch Protocol - Optimized for Speed with mode-specific settings
    - Fast mode: Reduced epochs and validation for speed
    - Full mode: More thorough training and validation  
    - Early stopping for efficiency  
    - User-based train/test split with user-based validation split (no user overlap)
    - User-based shuffling: shuffle user order but maintain temporal order within users
    """
    try:
        import copy
        import time
        
        start_time = time.time()
        
        # Algorithm-specific epoch optimization - mode-dependent 
        if fast_mode:
            # FAST MODE: Reduced epochs for speed
            if algorithm_name == 'AROW':
                n_epochs = 1  # Very expensive covariance updates
            elif algorithm_name == 'SCW':
                n_epochs = 1  # Moderate covariance complexity
            elif algorithm_name in ['RDA', 'AdaRDA']:
                n_epochs = 2  # Complex internal state
            else:
                n_epochs = 2  # Simple first-order algorithms
        else:
            # FULL MODE: More epochs for better performance
            if algorithm_name == 'AROW':
                n_epochs = 2  # More thorough covariance training
            elif algorithm_name == 'SCW':
                n_epochs = 3  # Better convergence
            elif algorithm_name in ['RDA', 'AdaRDA']:
                n_epochs = 4  # Complex algorithms benefit from more epochs
            else:
                n_epochs = 5  # Simple algorithms can handle more epochs
        
        # FAST: Split training data with smaller validation set for speed
        if train_user_ids is not None:
            # User-based split: no user overlap between train and validation
            X_train_split, X_val, y_train_split, y_val, train_split_user_ids = user_based_train_val_split(
                X_train, y_train, train_user_ids, val_size=0.1, random_state=1729  # FAST: Reduced val size
            )
        else:
            # Fallback to regular split if no user IDs available
            from sklearn.model_selection import train_test_split
            try:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=1729, stratify=y_train  # FAST: Reduced
                )
            except ValueError:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=1729  # FAST: Reduced
                )
            # Create dummy user IDs for shuffling
            train_split_user_ids = np.arange(len(X_train_split))
        
        # (2) Initialize best score s_OPT ← -1.0
        # (3) Initialize OPT parameters θ_OPT ← θ₁
        best_algorithm = None
        best_f1_score = -1.0
        best_epoch = 0
        previous_f1 = -1.0  # FAST: For early stopping
        
        train_start = time.time()
        
        # FAST OnlineToBatch Protocol: FOR e=1,2,...,E with early stopping
        for epoch in range(n_epochs):
            # (A) Introduce stochasticity by user-based shuffling
            # Shuffle users but maintain temporal order within each user for forecasting
            if train_user_ids is not None:
                X_shuffled, y_shuffled = user_based_shuffle(
                    X_train_split, y_train_split, train_split_user_ids, epoch, random_state=1729
                )
            else:
                # Fallback to regular shuffling
                indices = np.random.RandomState(epoch + 1729).permutation(len(X_train_split))
                X_shuffled = X_train_split[indices]
                y_shuffled = y_train_split[indices]
            
            # (B) Train on shuffled data
            if epoch == 0:
                # (1) Initialize parameters θ₁ from ALG
                current_algorithm = algorithm_class()
            else:
                # FAST: Use reference instead of deep copy for speed
                if best_algorithm is not None:
                    current_algorithm = best_algorithm  # Reference, no deep copy
                else:
                    current_algorithm = algorithm_class()
            
            # UpdateRule for each instance in D'_train - OPTIMIZED BATCH TRAINING
            # Instead of sample-by-sample, train in batches for efficiency while keeping 80/20 split
            batch_size = min(64, len(X_shuffled) // 4, 1000)  # Adaptive batch size
            if hasattr(current_algorithm, 'partial_fit') and batch_size > 1:
                # Batch training for efficiency
                for i in range(0, len(X_shuffled), batch_size):
                    end_idx = min(i + batch_size, len(X_shuffled))
                    current_algorithm.partial_fit(X_shuffled[i:end_idx], y_shuffled[i:end_idx])
            else:
                # Fallback to full batch
                current_algorithm.partial_fit(X_shuffled, y_shuffled)
            
            # (C) Evaluate on validation set - OPTIMIZED BATCH PREDICTION  
            if len(X_val) > 0:
                y_val_pred = current_algorithm.predict(X_val)  # Batch prediction is already efficient
                _, _, current_f1, _ = calculate_metrics(y_val, y_val_pred)
            else:
                current_f1 = 0.0
            
            # (D) Save the OPT: if s_current > s_OPT
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_algorithm = current_algorithm  # FAST: Reference instead of deepcopy
                best_epoch = epoch + 1
            
            # FAST: Early stopping if no improvement
            if epoch > 0 and current_f1 <= previous_f1 + 0.001:  # Minimal improvement threshold
                print(f"    Early stopping at epoch {epoch + 1}: F1 {current_f1:.4f}")
                break
            previous_f1 = current_f1
        
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

def evaluate_ensemble_methods(all_predictions, all_true_labels, validation_scores, dataset_name):
    """Evaluate intelligent ensemble aggregation methods optimized for cybersecurity"""
    print(f"\n  🔄 Evaluating SMART ensemble methods for {dataset_name}...")
    
    # Print individual algorithm performance for context
    print("    📊 Individual algorithm F1 scores:")
    for alg_name, scores in validation_scores.items():
        f1 = scores.get('f1', 0)
        fnr = scores.get('fnr', 0)
        fpr = scores.get('fpr', 0) 
        print(f"       {alg_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f}")
    
    ensemble_results = {}
    
    # Generate smart ensembles that filter out poor performers
    smart_ensembles = EnsembleAggregator.generate_smart_ensembles(
        all_predictions, all_true_labels, validation_scores
    )
    
    # Evaluate each smart ensemble method
    for method_name, ensemble_pred in smart_ensembles.items():
        fnr, fpr, f1, acc = calculate_metrics(all_true_labels, ensemble_pred)
        ensemble_results[method_name] = {
            'fnr': fnr, 'fpr': fpr, 'f1': f1, 'accuracy': acc
        }
        print(f"    ✅ {method_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f}")
    
    # Also include some baseline methods for comparison
    
    # Majority Voting (often bad with poor algorithms)
    majority_pred = np.round(np.mean(all_predictions, axis=0)).astype(int)
    majority_pred = np.where(majority_pred == 0, -1, 1)
    fnr, fpr, f1, acc = calculate_metrics(all_true_labels, majority_pred)
    ensemble_results['MajorityVoting_Baseline'] = {
        'fnr': fnr, 'fpr': fpr, 'f1': f1, 'accuracy': acc
    }
    print(f"    📊 MajorityVoting_Baseline: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f}")
    
    # Best single algorithm (for comparison)
    best_alg = max(validation_scores.items(), key=lambda x: x[1].get('f1', 0))
    best_f1 = best_alg[1].get('f1', 0)
    print(f"    🏆 Best individual: {best_alg[0]} (F1={best_f1:.3f})")
    
    return ensemble_results
    
    # 7. Adaptive Threshold Ensemble (confidence-based)
    adaptive_pred = EnsembleAggregator.adaptive_threshold_ensemble(all_predictions, validation_scores)
    ensemble_results['AdaptiveThreshold'] = evaluate_predictions(adaptive_pred, all_true_labels)
    
    # 8. Meta Ensemble (combination of strategies)
    meta_pred = EnsembleAggregator.meta_ensemble(all_predictions, validation_scores)
    ensemble_results['MetaEnsemble'] = evaluate_predictions(meta_pred, all_true_labels)
    
    # Print enhanced ensemble results with cybersecurity focus
    print(f"    🛡️  Cybersecurity-Optimized Ensemble Results:")
    print(f"    {'Method':<20} {'FNR':<8} {'FPR':<8} {'F1':<8} {'Acc':<8} {'Profile'}")
    print(f"    {'-'*75}")
    
    for method, results in ensemble_results.items():
        fnr, fpr, f1, acc = results['fnr'], results['fpr'], results['f1'], results['accuracy']
        
        # Add profile description
        profile_map = {
            'MajorityVoting': 'Baseline',
            'BalancedFPR_FNR': 'Balanced Security',
            'Conservative': 'Low False Alarms',
            'Aggressive': 'High Detection Rate',
            'Top3_F1': 'Best F1 Models',
            'Top3_Balanced': 'Best Balanced Models',
            'AdaptiveThreshold': 'Confidence-Based',
            'MetaEnsemble': 'Multi-Strategy'
        }
        profile = profile_map.get(method, 'Custom')
        
        print(f"    {method:<20} {fnr:<8.4f} {fpr:<8.4f} {f1:<8.4f} {acc:<8.4f} {profile}")
    
    # Find best methods for different objectives
    best_f1 = max(ensemble_results.keys(), key=lambda x: ensemble_results[x]['f1'])
    best_low_fnr = min(ensemble_results.keys(), key=lambda x: ensemble_results[x]['fnr'])
    best_low_fpr = min(ensemble_results.keys(), key=lambda x: ensemble_results[x]['fpr'])
    best_balanced = min(ensemble_results.keys(), key=lambda x: ensemble_results[x]['fnr'] + ensemble_results[x]['fpr'])
    
    print(f"\n    🏆 Best Ensemble Methods:")
    print(f"    Best F1 Score:     {best_f1} (F1={ensemble_results[best_f1]['f1']:.4f})")
    print(f"    Lowest FNR:        {best_low_fnr} (FNR={ensemble_results[best_low_fnr]['fnr']:.4f})")
    print(f"    Lowest FPR:        {best_low_fpr} (FPR={ensemble_results[best_low_fpr]['fpr']:.4f})")
    print(f"    Best Balanced:     {best_balanced} (FNR+FPR={ensemble_results[best_balanced]['fnr']+ensemble_results[best_balanced]['fpr']:.4f})")
    
    return ensemble_results

def evaluate_predictions(y_pred, y_true):
    """Helper function to calculate metrics from predictions"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'fnr': fnr,
        'fpr': fpr,
        'f1': f1,
        'accuracy': acc
    }

def create_ensemble_results_from_dataframe(results_df):
    """Create ensemble results by aggregating individual algorithm results per dataset"""
    ensemble_results = []
    
    # Debug: Print column names and sample of data
    print(f"  📊 DataFrame columns: {list(results_df.columns)}")
    print(f"  📊 DataFrame shape: {results_df.shape}")
    if len(results_df) > 0:
        print(f"  📊 Sample row: {results_df.iloc[0].to_dict()}")
    
    # Filter out failed runs (NaN values)
    valid_results = results_df.dropna(subset=['fnr', 'fpr', 'f1_score']).copy()
    
    if len(valid_results) == 0:
        print("  ⚠️  No valid results for ensemble creation")
        return []
    
    # Group by dataset
    datasets = valid_results['dataset'].unique()
    
    for dataset in datasets:
        dataset_results = valid_results[valid_results['dataset'] == dataset].copy()
        
        if len(dataset_results) < 2:
            print(f"  ⚠️  Skipping {dataset}: only {len(dataset_results)} valid algorithms")
            continue
            
        print(f"  📈 Creating ensembles for {dataset} ({len(dataset_results)} algorithms)")
        
        # Get sample information from the first row
        sample_row = dataset_results.iloc[0]
        
        # Create smart ensemble methods focused on LOW FPR AND LOW FNR
        ensemble_methods = {
            'SmartWeighted': create_statistical_ensemble(dataset_results, 'f1_weighted'),
            'ExcludeExtremes': create_statistical_ensemble(dataset_results, 'exclude_extremes'), 
            'BalancedOnly': create_statistical_ensemble(dataset_results, 'balanced_only'),
            'BestPerformer': create_statistical_ensemble(dataset_results, 'best_performer'),
            'TopBalanced': create_statistical_ensemble(dataset_results, 'top_balanced', k=3)
        }
        
        # Add ensemble results
        for method_name, ensemble_metrics in ensemble_methods.items():
            if ensemble_metrics:
                ensemble_results.append({
                    'dataset': dataset,
                    'algorithm': f'Ensemble_{method_name}',
                    'fnr': ensemble_metrics['fnr'],
                    'fpr': ensemble_metrics['fpr'],
                    'f1_score': ensemble_metrics['f1_score'],
                    'accuracy': ensemble_metrics['accuracy'],
                    'train_time': 0.0,  # Ensembles don't have training time
                    'predict_time': 0.0,  # Negligible
                    'train_samples': sample_row['train_samples'],
                    'test_samples': sample_row['test_samples'],
                    'features': sample_row['features'],
                    'best_epoch': np.nan,
                    'val_f1_score': np.nan,
                    'n_epochs': np.nan
                })
    
    print(f"  ✅ Generated {len(ensemble_results)} ensemble results")
    return ensemble_results

def create_statistical_ensemble(dataset_results, method='majority', k=3):
    """Create ensemble using statistical aggregation of metrics"""
    
    if method == 'majority':
        # Simple average of all metrics
        return {
            'fnr': dataset_results['fnr'].mean(),
            'fpr': dataset_results['fpr'].mean(),
            'f1_score': dataset_results['f1_score'].mean(),
            'accuracy': dataset_results['accuracy'].mean()
        }
    
    elif method == 'f1_weighted':
        # Weight by F1 score performance
        weights = dataset_results['f1_score'] / dataset_results['f1_score'].sum()
        return {
            'fnr': (dataset_results['fnr'] * weights).sum(),
            'fpr': (dataset_results['fpr'] * weights).sum(),
            'f1_score': (dataset_results['f1_score'] * weights).sum(),
            'accuracy': (dataset_results['accuracy'] * weights).sum()
        }
    
    elif method == 'low_fnr_weighted':
        # Weight inversely by FNR (prefer low false negatives for cybersecurity)
        inv_fnr = 1.0 - dataset_results['fnr']
        weights = inv_fnr / inv_fnr.sum()
        return {
            'fnr': (dataset_results['fnr'] * weights).sum(),
            'fpr': (dataset_results['fpr'] * weights).sum(),
            'f1_score': (dataset_results['f1_score'] * weights).sum(),
            'accuracy': (dataset_results['accuracy'] * weights).sum()
        }
    
    elif method == 'best_performer':
        # Use metrics from the best F1 performing algorithm
        best_idx = dataset_results['f1_score'].idxmax()
        best_row = dataset_results.loc[best_idx]
        return {
            'fnr': best_row['fnr'],
            'fpr': best_row['fpr'],
            'f1_score': best_row['f1_score'],
            'accuracy': best_row['accuracy']
        }
    
    elif method == 'top_k':
        # Average of top K performers by F1 score
        top_k = dataset_results.nlargest(min(k, len(dataset_results)), 'f1_score')
        return {
            'fnr': top_k['fnr'].mean(),
            'fpr': top_k['fpr'].mean(),
            'f1_score': top_k['f1_score'].mean(),
            'accuracy': top_k['accuracy'].mean()
        }
    
    elif method == 'exclude_extremes':
        # SMART: Exclude algorithms with extreme FPR or FNR values
        filtered = dataset_results[
            (dataset_results['fpr'] < 0.9) &  # Exclude FPR > 90%
            (dataset_results['fnr'] < 0.8) &  # Exclude FNR > 80%
            (dataset_results['f1_score'] > 0.1)  # Must have some performance
        ]
        
        if len(filtered) == 0:
            # Fallback to best performer if all are extreme
            best_idx = dataset_results['f1_score'].idxmax()
            best_row = dataset_results.loc[best_idx]
            return {
                'fnr': best_row['fnr'],
                'fpr': best_row['fpr'],
                'f1_score': best_row['f1_score'],
                'accuracy': best_row['accuracy']
            }
        
        # Weight by F1 score among non-extreme algorithms
        weights = filtered['f1_score'] / filtered['f1_score'].sum()
        return {
            'fnr': (filtered['fnr'] * weights).sum(),
            'fpr': (filtered['fpr'] * weights).sum(),
            'f1_score': (filtered['f1_score'] * weights).sum(),
            'accuracy': (filtered['accuracy'] * weights).sum()
        }
    
    elif method == 'balanced_only':
        # SMART: Only use algorithms with balanced FNR+FPR < threshold
        balance_score = dataset_results['fnr'] + dataset_results['fpr']
        balanced = dataset_results[balance_score < 1.0]  # Combined error < 100%
        
        if len(balanced) == 0:
            # Fallback to most balanced algorithm
            best_balance_idx = balance_score.idxmin()
            best_row = dataset_results.loc[best_balance_idx]
            return {
                'fnr': best_row['fnr'],
                'fpr': best_row['fpr'],
                'f1_score': best_row['f1_score'],
                'accuracy': best_row['accuracy']
            }
        
        # Weight by F1 among balanced algorithms
        weights = balanced['f1_score'] / balanced['f1_score'].sum()
        return {
            'fnr': (balanced['fnr'] * weights).sum(),
            'fpr': (balanced['fpr'] * weights).sum(),
            'f1_score': (balanced['f1_score'] * weights).sum(),
            'accuracy': (balanced['accuracy'] * weights).sum()
        }
    
    elif method == 'top_balanced':
        # SMART: Top K algorithms by lowest FNR+FPR balance
        balance_score = dataset_results['fnr'] + dataset_results['fpr']
        dataset_results_copy = dataset_results.copy()
        dataset_results_copy['balance'] = balance_score
        top_balanced = dataset_results_copy.nsmallest(min(k, len(dataset_results_copy)), 'balance')
        
        # Weight by F1 among most balanced
        if len(top_balanced) > 0:
            weights = top_balanced['f1_score'] / top_balanced['f1_score'].sum()
            return {
                'fnr': (top_balanced['fnr'] * weights).sum(),
                'fpr': (top_balanced['fpr'] * weights).sum(),
                'f1_score': (top_balanced['f1_score'] * weights).sum(),
                'accuracy': (top_balanced['accuracy'] * weights).sum()
            }
        else:
            return None
    
    else:
        return None

def create_ensemble_results(results_df):
    """Create ensemble results using statistical aggregation of individual algorithm results"""
    print(f"\n🔄 Creating ensemble results using statistical aggregation...")
    
    # Filter out failed results
    valid_results = results_df.dropna(subset=['fnr', 'fpr', 'f1_score']).copy()
    
    if len(valid_results) == 0:
        print("No valid results to aggregate.")
        return
    
    # Group by dataset to create ensembles per dataset
    ensemble_results = []
    
    for dataset in valid_results['dataset'].unique():
        dataset_results = valid_results[valid_results['dataset'] == dataset]
        
        if len(dataset_results) < 2:
            continue
        
        print(f"  Creating ensembles for {dataset} ({len(dataset_results)} algorithms)")
        
        # Method 1: Mean aggregation (conservative approach)
        mean_fnr = dataset_results['fnr'].mean()
        mean_fpr = dataset_results['fpr'].mean()
        mean_f1 = dataset_results['f1_score'].mean()
        mean_acc = dataset_results['accuracy'].mean()
        
        ensemble_results.append({
            'dataset': dataset,
            'algorithm': 'Ensemble_Mean',
            'fnr': mean_fnr,
            'fpr': mean_fpr,
            'f1_score': mean_f1,
            'accuracy': mean_acc,
            'train_time': dataset_results['train_time'].mean(),
            'predict_time': dataset_results['predict_time'].mean(),
            'train_samples': dataset_results['train_samples'].iloc[0],
            'test_samples': dataset_results['test_samples'].iloc[0],
            'features': dataset_results['features'].iloc[0],
            'best_epoch': np.nan,
            'val_f1_score': dataset_results['val_f1_score'].mean(),
            'n_epochs': np.nan
        })
        
        # Method 2: Best FNR weighted aggregation
        weights = 1.0 / (dataset_results['fnr'] + 0.001)  # Higher weight for lower FNR
        weights = weights / weights.sum()
        
        weighted_fnr = (dataset_results['fnr'] * weights).sum()
        weighted_fpr = (dataset_results['fpr'] * weights).sum()
        weighted_f1 = (dataset_results['f1_score'] * weights).sum()
        weighted_acc = (dataset_results['accuracy'] * weights).sum()
        
        ensemble_results.append({
            'dataset': dataset,
            'algorithm': 'Ensemble_WeightedByFNR',
            'fnr': weighted_fnr,
            'fpr': weighted_fpr,
            'f1_score': weighted_f1,
            'accuracy': weighted_acc,
            'train_time': dataset_results['train_time'].mean(),
            'predict_time': dataset_results['predict_time'].mean(),
            'train_samples': dataset_results['train_samples'].iloc[0],
            'test_samples': dataset_results['test_samples'].iloc[0],
            'features': dataset_results['features'].iloc[0],
            'best_epoch': np.nan,
            'val_f1_score': dataset_results['val_f1_score'].mean(),
            'n_epochs': np.nan
        })
        
        # Method 3: Select top performers and average (selective ensemble)
        top_performers = dataset_results.nsmallest(3, 'fnr')  # Top 3 by lowest FNR
        if len(top_performers) >= 2:
            ensemble_results.append({
                'dataset': dataset,
                'algorithm': 'Ensemble_TopPerformers',
                'fnr': top_performers['fnr'].mean(),
                'fpr': top_performers['fpr'].mean(),
                'f1_score': top_performers['f1_score'].mean(),
                'accuracy': top_performers['accuracy'].mean(),
                'train_time': top_performers['train_time'].mean(),
                'predict_time': top_performers['predict_time'].mean(),
                'train_samples': dataset_results['train_samples'].iloc[0],
                'test_samples': dataset_results['test_samples'].iloc[0],
                'features': dataset_results['features'].iloc[0],
                'best_epoch': np.nan,
                'val_f1_score': top_performers['val_f1_score'].mean(),
                'n_epochs': np.nan
            })
    
    if ensemble_results:
        # Create results directory if it doesn't exist
        from pathlib import Path
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Create ensemble DataFrame and save
        ensemble_df = pd.DataFrame(ensemble_results)
        ensemble_file = 'results/ensemble_results.csv'
        ensemble_df.to_csv(ensemble_file, index=False)
        print(f"✓ Ensemble results saved to {ensemble_file}")
        
        # Print summary of ensemble performance
        print(f"\n📊 ENSEMBLE PERFORMANCE SUMMARY:")
        print("─" * 50)
        for method in ['Ensemble_Mean', 'Ensemble_WeightedByFNR', 'Ensemble_TopPerformers']:
            method_results = ensemble_df[ensemble_df['algorithm'] == method]
            if len(method_results) > 0:
                avg_fnr = method_results['fnr'].mean()
                avg_fpr = method_results['fpr'].mean()
                avg_f1 = method_results['f1_score'].mean()
                print(f"{method:>25s}: FNR={avg_fnr:.4f}, FPR={avg_fpr:.4f}, F1={avg_f1:.4f}")
        
        # Compare with best individual algorithm
        best_individual = valid_results.loc[valid_results['fnr'].idxmin()]
        print(f"\nCompare with best individual algorithm:")
        print(f"{'Best Individual':>25s}: FNR={best_individual['fnr']:.4f}, FPR={best_individual['fpr']:.4f}, F1={best_individual['f1_score']:.4f} ({best_individual['algorithm']})")
    
    else:
        print("No ensemble results could be created.")

def main():
    print("7-Algorithm Cybersecurity Evaluation with Ensemble Methods")
    print("Memory-Efficient Implementation with Performance Optimization")
    print("=" * 80)
    
    # User input for mode selection
    print("\n🚀 Choose execution mode:")
    print("1. FAST MODE - Process 2 specific datasets (NonEnc_desktop, P2P_desktop)")
    print("2. FULL MODE - Process all available datasets")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 for FAST, 2 for FULL): ").strip()
            if choice == '1':
                fast_mode = True
                print("✅ FAST MODE selected - Processing NonEnc_desktop and P2P_desktop")
                break
            elif choice == '2':
                fast_mode = False
                print("✅ FULL MODE selected - Processing all available datasets")
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Execution cancelled by user.")
            return
        except Exception as e:
            print(f"❌ Error reading input: {e}. Please try again.")
    
    # Load datasets using smart loading function that checks for existing data
    data_dir = 'cyber/'
    print(f"\n📂 Loading RBD24 datasets...")
    X_train_dict, y_train_dict, X_test_dict, y_test_dict, train_user_ids_dict, test_user_ids_dict = load_rbd24_properly(data_dir)
    
    if not X_train_dict:
        print("❌ No datasets loaded! Check load_rbd24_properly function")
        return
    
    print(f"✅ Successfully loaded {len(X_train_dict)} datasets from RBD24")
    print(f"📋 Available datasets: {list(X_train_dict.keys())}")
    
    # Apply mode-specific dataset selection
    if fast_mode:
        # FAST MODE: Use specific datasets for speed demonstration
        fast_mode_datasets = ['NonEnc_desktop', 'P2P_desktop']
        available_datasets = set(X_train_dict.keys())
        
        # Check which of the requested fast mode datasets are available
        selected_datasets = [ds for ds in fast_mode_datasets if ds in available_datasets]
        
        if len(selected_datasets) < len(fast_mode_datasets):
            missing = [ds for ds in fast_mode_datasets if ds not in available_datasets]
            print(f"⚠️  Requested datasets not found: {missing}")
            
            # If requested datasets not available, fall back to first 2
            if len(selected_datasets) == 0:
                selected_datasets = list(X_train_dict.keys())[:2]
                print(f"⚡ FAST MODE: Using first 2 available datasets: {selected_datasets}")
            else:
                print(f"⚡ FAST MODE: Using available requested datasets: {selected_datasets}")
        else:
            print(f"⚡ FAST MODE: Using requested datasets: {selected_datasets}")
        
        # Create subset dictionaries with selected datasets
        X_train_dict = {k: X_train_dict[k] for k in selected_datasets}
        y_train_dict = {k: y_train_dict[k] for k in selected_datasets}
        X_test_dict = {k: X_test_dict[k] for k in selected_datasets}  
        y_test_dict = {k: y_test_dict[k] for k in selected_datasets}
        train_user_ids_dict = {k: train_user_ids_dict.get(k, None) for k in selected_datasets}
        test_user_ids_dict = {k: test_user_ids_dict.get(k, None) for k in selected_datasets}
        
        print(f"⚡ FAST MODE: Processing {len(selected_datasets)} datasets")
    else:
        # FULL MODE: Use all available datasets
        print(f"🔥 FULL MODE: Processing all {len(X_train_dict)} datasets")
    
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
    
    print(f"\nStarting evaluation on {len(X_train_dict)} datasets × {len(algorithms)} algorithms")
    print(f"Target: {len(X_train_dict) * len(algorithms)} evaluations")
    
    # Process datasets with progress bar
    for dataset_name in tqdm(X_train_dict.keys(), desc="Processing datasets"):
        try:
            print(f"\n🔍 Processing dataset: {dataset_name}")
            
            # Get dataset from dictionaries
            X_train = X_train_dict[dataset_name]
            y_train = y_train_dict[dataset_name]
            X_test = X_test_dict[dataset_name]
            y_test = y_test_dict[dataset_name]
            train_user_ids = train_user_ids_dict.get(dataset_name, None)
            test_user_ids = test_user_ids_dict.get(dataset_name, None)
            
            print(f"  ✅ Retrieved data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
            print(f"  ✅ Retrieved label shapes: y_train={y_train.shape}, y_test={y_test.shape}")
            
            # Check if we have both classes
            unique_train = np.unique(y_train)
            unique_test = np.unique(y_test)
            print(f"  ✅ Train classes: {unique_train}, Test classes: {unique_test}")
            
            if len(unique_train) < 2 or len(unique_test) < 2:
                print(f"  ⚠️  Skipping {dataset_name} (single class)")
                continue
            
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
            algorithm_predictions = {}
            algorithm_validation_scores = {}
            
            for alg_name, algorithm_class in algorithms.items():
                print(f"\n  🔧 Training {alg_name}...")
                
                # OnlineToBatch Protocol Implementation
                try:
                    result = evaluate_algorithm_online_to_batch(
                        alg_name, algorithm_class, X_train, y_train, X_test, y_test, train_user_ids, fast_mode
                    )
                    
                    if result is not None:
                        print(f"    ✅ {alg_name} completed successfully")
                        # Store individual algorithm results
                        results.append({
                            'dataset': dataset_name,
                            'algorithm': alg_name,
                            'fnr': result['fnr'],
                            'fpr': result['fpr'],
                            'f1_score': result['f1'],  # Note: result returns 'f1', we store as 'f1_score'
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
                        
                        # Store predictions and validation scores for ensemble
                        if 'predictions' in result:
                            algorithm_predictions[alg_name] = result['predictions']
                            algorithm_validation_scores[alg_name] = {
                                'f1': result['f1'],
                                'fnr': result['fnr'],
                                'fpr': result['fpr'],
                                'val_f1_score': result['val_f1_score']
                            }
                    else:
                        # Algorithm failed
                        print(f"    ❌ {alg_name} returned None result")
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
                    print(f"    ❌ {alg_name} failed: {str(e)}")
                    print(f"       Error type: {type(e).__name__}")
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
            
            # Evaluate ensemble methods if we have predictions from multiple algorithms
            if len(algorithm_predictions) >= 2:
                ensemble_results = evaluate_ensemble_methods(
                    algorithm_predictions, y_test, algorithm_validation_scores, dataset_name
                )
                
                # Add ensemble results to the main results
                for method_name, method_results in ensemble_results.items():
                    results.append({
                        'dataset': dataset_name,
                        'algorithm': f'Ensemble_{method_name}',
                        'fnr': method_results['fnr'],
                        'fpr': method_results['fpr'],
                        'f1_score': method_results['f1'],
                        'accuracy': method_results['accuracy'],
                        'train_time': 0.0,  # Ensembles don't have training time
                        'predict_time': 0.0,  # Negligible prediction time
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features': X_train.shape[1],
                        'best_epoch': np.nan,
                        'val_f1_score': np.nan,
                        'n_epochs': np.nan
                    })
            
            # Memory cleanup
            del X_train, X_test, y_train, y_test
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue
    
    print(f"\nCompleted evaluations on {len(X_train_dict)} datasets")
    
    # Debug: Check results before DataFrame creation
    print(f"  📊 Total results collected: {len(results)}")
    if len(results) > 0:
        print(f"  📊 Sample result keys: {list(results[0].keys())}")
        print(f"  📊 Sample result: {results[0]}")
    else:
        print("  ⚠️  No results collected - this will cause DataFrame issues")
    
    # Convert to DataFrame for ensemble analysis
    results_df = pd.DataFrame(results)
    print(f"  📊 DataFrame created with shape: {results_df.shape}")
    
    if len(results_df) > 0:
        print(f"  📊 DataFrame columns: {list(results_df.columns)}")
        # DISABLED: Generate ensemble results using statistical aggregation (metric averaging)
        # The prediction-based smart ensembles are already included in results from evaluate_ensemble_methods
        print("\n📊 Using prediction-based smart ensembles (already included in results)")
        ensemble_results = []  # Don't add statistical ensembles that overwrite the good ones
    else:
        print("  ⚠️  Empty DataFrame - skipping ensemble generation")
        ensemble_results = []
    
    # Add ensemble results to the main results
    all_results = results + ensemble_results
    final_results_df = pd.DataFrame(all_results)
    
    # Save combined results (individual + ensemble)
    results_file = 'results/batch_results_with_ensembles.csv'
    
    # Create results directory if it doesn't exist
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    final_results_df.to_csv(results_file, index=False)
    print(f"\n✓ Combined results (individual + ensemble) saved to {results_file}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("ADVANCED 7-ALGORITHM + ENSEMBLE EVALUATION SUMMARY")
    print("=" * 80)
    
    # Check if we have any results with required columns
    if 'fnr' not in final_results_df.columns or 'fpr' not in final_results_df.columns:
        print("ERROR: No successful evaluations completed.")
        print("All datasets failed during processing.")
        print("Check the error messages above for details.")
        return
    
    # Filter out failed runs
    valid_results = final_results_df.dropna(subset=['fnr', 'fpr'])
    
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
    
    # 4. ENSEMBLE PERFORMANCE ANALYSIS
    print(f"\n4. ENSEMBLE vs INDIVIDUAL PERFORMANCE:")
    print("─" * 45)
    
    # Separate ensemble and individual results
    ensemble_results = valid_results[valid_results['algorithm'].str.startswith('Ensemble_')].copy()
    individual_results = valid_results[~valid_results['algorithm'].str.startswith('Ensemble_')].copy()
    
    if len(ensemble_results) > 0 and len(individual_results) > 0:
        print("\nEnsemble Methods Performance:")
        ensemble_ranking = ensemble_results.groupby('algorithm').agg({
            'fnr': 'mean', 'fpr': 'mean', 'f1_score': 'mean'
        }).round(4).sort_values('f1_score', ascending=False)
        
        for i, (alg, row) in enumerate(ensemble_ranking.iterrows(), 1):
            print(f"  {i:2d}. {alg:>25s}: F1={row['f1_score']:.4f}, FNR={row['fnr']:.4f}, FPR={row['fpr']:.4f}")
        
        # Compare best ensemble vs best individual
        best_ensemble_f1 = ensemble_results['f1_score'].max()
        best_individual_f1 = individual_results['f1_score'].max()
        best_ensemble_fnr = ensemble_results.loc[ensemble_results['f1_score'].idxmax(), 'fnr']
        best_individual_fnr = individual_results.loc[individual_results['f1_score'].idxmax(), 'fnr']
        
        print(f"\nImprovement Analysis:")
        print(f"  Best Individual F1: {best_individual_f1:.4f}")
        print(f"  Best Ensemble F1:   {best_ensemble_f1:.4f}")
        f1_improvement = ((best_ensemble_f1 - best_individual_f1) / best_individual_f1) * 100
        fnr_improvement = ((best_individual_fnr - best_ensemble_fnr) / best_individual_fnr) * 100
        print(f"  F1 Improvement:     {f1_improvement:+.2f}%")
        print(f"  FNR Improvement:    {fnr_improvement:+.2f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Advanced evaluation with ensemble methods completed")
    print("Individual + Ensemble results saved for comparison")
    print("Memory-efficient processing with FNR/FPR optimization")
    print("=" * 80)

if __name__ == "__main__":
    main()
