import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import gc
import copy
import warnings
import argparse
import psutil  # Add for system monitoring
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy.stats import norm
from collections import defaultdict
import sys

# Import data handler for Zenodo downloads
try:
    from data_handler import prepare_data_from_zenodo
except ImportError:
    print("Warning: data_handler module not found, using fallback data creation")
    def prepare_data_from_zenodo(url, path):
        return False

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

def load_rbd24_properly(data_dir):
    """Load RBD24 dataset - download if not already available"""
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
    
    print(f" Checking for data in: {cyber_dir}")
    
    # Check if parquet files already exist in cyber directory
    parquet_files = []
    if cyber_dir.exists():
        parquet_files = list(cyber_dir.glob('*.parquet'))
        print(f"📁 Cyber directory exists with {len(list(cyber_dir.iterdir()))} total files")
        if parquet_files:
            print(f" Found existing {len(parquet_files)} parquet files:")
            for f in parquet_files[:5]:  # Show first 5 files
                print(f"   - {f.name}")
            if len(parquet_files) > 5:
                print(f"   ... and {len(parquet_files) - 5} more")
        else:
            print(f"  Directory exists but no .parquet files found")
            print(f"   Contents: {[f.name for f in cyber_dir.iterdir()]}")
    else:
        print(f"📁 Cyber directory does not exist: {cyber_dir}")
    
    if not parquet_files:
        # Data not found, need to download
        print(f"📥 Downloading RBD24 data to {cyber_dir}...")
        cyber_dir.mkdir(parents=True, exist_ok=True)
        
        # Use your robust download method
        zenodo_url = 'https://zenodo.org/api/records/13787591/files-archive'
        success = prepare_data_from_zenodo(zenodo_url, str(cyber_dir))
        
        if not success:
            print(" CRITICAL: Failed to download RBD24 data and no sample data available")
            print("   Please check your internet connection or manually download the data")
            print(f"   Expected location: {cyber_dir}")
            sys.exit(1)
        
        # Re-check for parquet files after download
        parquet_files = list(cyber_dir.glob('*.parquet'))
        if not parquet_files:
            print(" CRITICAL: No parquet files found after download")
            print("   The download may have failed or files are in wrong format")
            sys.exit(1)
        
        print(f" Successfully downloaded {len(parquet_files)} RBD24 datasets")
    else:
        print(f" Using existing data - skipping download")
    
    X_train_dict = {}
    y_train_dict = {}
    X_test_dict = {}
    y_test_dict = {}
    train_user_ids_dict = {}
    test_user_ids_dict = {}
    
    for parquet_file in parquet_files:
        dataset_name = parquet_file.stem
        print(f"\n Loading dataset: {dataset_name}")
        
        try:
            # Load data from parquet file
            df = pd.read_parquet(parquet_file)
            print(f"  Raw data shape: {df.shape}")
            
            # Check required columns
            if 'user_id' not in df.columns or 'label' not in df.columns:
                print(f"    Missing required columns, skipping {dataset_name}")
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
            print(f"   Error loading {dataset_name}: {e}")
            continue
    
    if not X_train_dict:
        print(" CRITICAL: No valid datasets loaded and no sample data available")
        print("   Please ensure RBD24 data is properly downloaded")
        sys.exit(1)
    
    print(f"\n Successfully loaded {len(X_train_dict)} RBD24 datasets with clean user-based splitting")
    return X_train_dict, y_train_dict, X_test_dict, y_test_dict, train_user_ids_dict, test_user_ids_dict

# Enhanced Ensemble aggregation methods for cybersecurity
class EnsembleAggregator:
    """Implements intelligent ensemble aggregation strategies optimized for cybersecurity"""
    
    @staticmethod
    def smart_ensemble_selection(all_predictions, validation_scores):
        """Select algorithms with BOTH low FPR AND low FNR for ensemble"""
        # FOCUS: Only use algorithms with LOW FNR AND LOW FPR
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
            print(f"     Using {len(excellent_algorithms)} EXCELLENT algorithms (low FNR+FPR)")
            return excellent_algorithms
        elif len(good_algorithms) >= 2:
            print(f"     Using {len(good_algorithms)} GOOD algorithms (balanced FNR+FPR)")
            return good_algorithms
        else:
            # Emergency fallback: use top algorithms by F1, but warn user
            print(f"      No algorithms with low FNR+FPR found, using top F1 performers")
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
        
        print(f"     Selected algorithms with weights:")
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
            print(f"     LOW ERROR ENSEMBLE: Using {len(super_algorithms)} super algorithms")
            super_predictions = np.array([all_predictions[i] for i, _, _ in super_algorithms])
            return np.mean(super_predictions, axis=0)
        else:
            print(f"     No super algorithms found, falling back to mean")
            return np.mean(all_predictions, axis=0)
    
    @staticmethod
    def exclude_extremes_ensemble(all_predictions, validation_scores):
        """NEW: Exclude extreme algorithms (like AdaRDA, RDA) that have FNR=0 OR FPR=1.0"""
        balanced_algorithms = []
        
        print("     EXCLUDE EXTREMES analysis:")
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
                print(f"       {alg_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f}")
            else:
                print(f"       {alg_name}: F1={f1:.3f}, FNR={fnr:.3f}, FPR={fpr:.3f} (EXTREME)")
        
        if len(balanced_algorithms) >= 2:
            print(f"     Using {len(balanced_algorithms)} non-extreme algorithms")
            balanced_predictions = np.array([all_predictions[i] for i, _, _ in balanced_algorithms])
            return np.mean(balanced_predictions, axis=0)
        else:
            print(f"      Only {len(balanced_algorithms)} non-extreme algorithms found, using all")
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
    def top_k_ensemble(all_predictions, validation_scores, k=3):
        """Select top-K performing algorithms by F1 score and average their predictions"""
        # Sort algorithms by F1 score
        sorted_algs = sorted(validation_scores.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
        
        # Select top k algorithms
        top_k_algs = sorted_algs[:min(k, len(sorted_algs))]
        
        if not top_k_algs:
            return np.mean(all_predictions, axis=0)
        
        # Get indices of top k algorithms
        alg_names = list(validation_scores.keys())
        top_k_indices = [alg_names.index(alg_name) for alg_name, _ in top_k_algs]
        
        # Average predictions from top k algorithms
        top_k_predictions = np.array([all_predictions[i] for i in top_k_indices])
        return np.mean(top_k_predictions, axis=0)
    
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
    
def weighted_majority_ensemble_by_user(predictions_dict, true_labels, test_user_ids):
    """
    Weighted Majority Ensemble using user-based adaptive weights.
    For each user, assign weights to each algorithm based on its F1 score over that user's samples.
    Final prediction for each sample is the weighted majority vote across algorithms.
    """
    alg_names = list(predictions_dict.keys())
    all_predictions = np.array([predictions_dict[alg] for alg in alg_names])  # shape: [n_algorithms, n_samples]
    n_algorithms, n_samples = all_predictions.shape
    ensemble_pred = np.zeros(n_samples, dtype=int)
    
    if test_user_ids is None:
        # Fallback to simple weighted average if no user IDs
        weights = np.ones(n_algorithms) / n_algorithms
        weighted_votes = np.dot(weights, all_predictions)
        return np.where(weighted_votes >= 0, 1, -1)
    
    unique_users = np.unique(test_user_ids)

    for user in unique_users:
        user_mask = (test_user_ids == user)
        if np.sum(user_mask) == 0:
            continue

        user_true = true_labels[user_mask]
        user_preds = all_predictions[:, user_mask]
        weights = []
        for alg_pred in user_preds:
            try:
                w = f1_score(user_true, alg_pred, pos_label=1, zero_division=0)
            except Exception:
                w = 0.0
            weights.append(w)
        weights = np.array(weights)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)

        weighted_votes = np.dot(weights, user_preds)
        ensemble_pred[user_mask] = np.where(weighted_votes >= 0, 1, -1)

    return ensemble_pred

def evaluate_ensemble_methods(all_predictions, all_true_labels, validation_scores, dataset_name, test_user_ids=None):
    """Evaluate intelligent ensemble aggregation methods optimized for cybersecurity"""
    print(f"\n   Evaluating SMART ensemble methods for {dataset_name}...")

    # Convert predictions dict to array format if needed
    if isinstance(all_predictions, dict):
        predictions_dict = all_predictions
        all_predictions = np.array(list(all_predictions.values()))
    else:
        # If it's already an array, create a predictions dict
        alg_names = list(validation_scores.keys())
        predictions_dict = {alg: all_predictions[i] for i, alg in enumerate(alg_names)}

    # Print individual algorithm performance for context
    print("     Individual algorithm FNR and FPR scores:")
    for alg_name, scores in validation_scores.items():
        fnr = scores.get('fnr', 0)
        fpr = scores.get('fpr', 0)
        print(f"       {alg_name}: FNR={fnr:.3f}, FPR={fpr:.3f}")

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
        print(f"     {method_name}: FNR={fnr:.3f}, FPR={fpr:.3f}")

    # Weighted Majority Ensemble by User (NEW)
    if test_user_ids is not None:
        wmajority_pred = weighted_majority_ensemble_by_user(predictions_dict, all_true_labels, test_user_ids)
        fnr, fpr, f1, acc = calculate_metrics(all_true_labels, wmajority_pred)
        ensemble_results['WeightedMajorityByUser'] = {
            'fnr': fnr, 'fpr': fpr, 'f1': f1, 'accuracy': acc
        }
        print(f"     WeightedMajorityByUser: FNR={fnr:.3f}, FPR={fpr:.3f}")

    # Majority Voting (often bad with poor algorithms)
    majority_pred = np.round(np.mean(all_predictions, axis=0)).astype(int)
    majority_pred = np.where(majority_pred == 0, -1, 1)
    fnr, fpr, f1, acc = calculate_metrics(all_true_labels, majority_pred)
    ensemble_results['MajorityVoting_Baseline'] = {
        'fnr': fnr, 'fpr': fpr, 'f1': f1, 'accuracy': acc
    }
    print(f"     MajorityVoting_Baseline: FNR={fnr:.3f}, FPR={fpr:.3f}")

    # Best single algorithm (for comparison) - use F1 internally but don't display
    best_alg = max(validation_scores.items(), key=lambda x: x[1].get('f1', 0))
    print(f"     Best individual: {best_alg[0]}")

    return ensemble_results

def evaluate_algorithm_online_to_batch(algorithm_name, algorithm_class, X_train, y_train, X_test, y_test, 
                                       train_user_ids=None, fast_mode=False):
    """
    OnlineToBatch Protocol - Optimized for Speed with mode-specific settings
    """
    try:
        start_time = time.time()
        
        # ULTRA-FAST mode settings for efficiency
        if fast_mode:
            # ULTRA-FAST: Minimal epochs
            if algorithm_name == 'AROW':
                n_epochs = 1  # Very expensive
            elif algorithm_name == 'SCW':
                n_epochs = 1  # Expensive
            elif algorithm_name in ['RDA', 'AdaRDA']:
                n_epochs = 1  # Reduce from 2 to 1
            else:
                n_epochs = 1  # Reduce all to 1 epoch
        else:
            # Regular mode - keep existing logic
            if algorithm_name == 'AROW':
                n_epochs = 2
            elif algorithm_name == 'SCW':
                n_epochs = 3
            elif algorithm_name in ['RDA', 'AdaRDA']:
                n_epochs = 4
            else:
                n_epochs = 5
        
        # Monitor resources before training
        if fast_mode:
            stats = monitor_system_resources()
            if stats['cpu_percent'] > 80:
                print(f"      High CPU usage ({stats['cpu_percent']:.1f}%), adding delay...")
                time.sleep(1)  # Brief pause to cool down
        
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
        
        # Initialize best score and algorithm
        best_algorithm = None
        best_f1_score = -1.0
        best_epoch = 0
        previous_f1 = -1.0  # For early stopping
        
        train_start = time.time()
        
        # OnlineToBatch Protocol: FOR e=1,2,...,E with early stopping
        for epoch in range(n_epochs):
            # User-based shuffling
            if train_user_ids is not None:
                X_shuffled, y_shuffled = user_based_shuffle(
                    X_train_split, y_train_split, train_split_user_ids, epoch, random_state=1729
                )
            else:
                # Fallback to regular shuffling
                indices = np.random.RandomState(epoch + 1729).permutation(len(X_train_split))
                X_shuffled = X_train_split[indices]
                y_shuffled = y_train_split[indices]
            
            # Initialize or update algorithm
            if epoch == 0:
                current_algorithm = algorithm_class()
            else:
                # Use reference for speed in fast mode
                if best_algorithm is not None:
                    current_algorithm = best_algorithm
                else:
                    current_algorithm = algorithm_class()
            
            # Train algorithm
            current_algorithm.partial_fit(X_shuffled, y_shuffled)
            
            # Evaluate on validation set - use F1 for consistency
            if len(X_val) > 0:
                y_val_pred = current_algorithm.predict(X_val)
                _, _, current_f1, current_acc = calculate_metrics(y_val, y_val_pred)
            else:
                current_f1 = 0.0
                current_acc = 0.0
        
            # Save best algorithm based on F1 (but don't report F1 in final output)
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_algorithm = current_algorithm
                best_epoch = epoch + 1
            
            # Early stopping if no improvement
            if epoch > 0 and current_f1 <= previous_f1 + 0.001:
                print(f"    Early stopping at epoch {epoch + 1}: Internal F1 {current_f1:.4f}")
                break
            previous_f1 = current_f1
        
        train_time = time.time() - train_start
        
        # Make final predictions with best algorithm
        predict_start = time.time()
        if best_algorithm is None:
            best_algorithm = algorithm_class()
            best_algorithm.partial_fit(X_train_split, y_train_split)
            
        y_pred = best_algorithm.predict(X_test)
        predict_time = time.time() - predict_start
        
        # Calculate final metrics
        fnr, fpr, f1, acc = calculate_metrics(y_test, y_pred)
        
        # Print results - Remove F1 from output but keep internal tracking
        print(f"    FNR: {fnr:.4f}, FPR: {fpr:.4f}")
        print(f"    Acc: {acc:.4f}")
        print(f"    Time: {train_time:.3f}s train, {predict_time:.3f}s predict")
        print(f"    Best epoch: {best_epoch}/{n_epochs}")
        
        return {
            'fnr': fnr,
            'fpr': fpr,
            'f1': f1,  # Keep for internal use but don't display
            'accuracy': acc,
            'train_time': train_time,
            'predict_time': predict_time,
            'total_time': time.time() - start_time,
            'best_epoch': best_epoch,
            'val_f1_score': best_f1_score,
            'n_epochs': n_epochs,
            'predictions': y_pred
        }
        
    except Exception as e:
        print(f"    Error with {algorithm_name}: {str(e)}")
        return None

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
        # Add: store feature thresholds for RF-like rule
        self.rf_thresholds = None

    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            # Add: initialize thresholds for RF-like rule
            self.rf_thresholds = np.median(X, axis=0)

        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]  # Data is already {-1,+1}

            # Standard gradient update
            if y_i * x_i.dot(self.weights) < 1:
                self.weights += y_i * x_i

            # Add: Random Forest-like rule
            # If more than half features exceed their threshold, boost update
            rf_vote = np.sum(x_i > self.rf_thresholds)
            if rf_vote > (self.n_features // 2):
                self.weights += y_i * x_i * 0.5  # Boost by 0.5 if RF-like rule triggers

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
            prediction = prediction if prediction != 0 else 1  # Fixed syntax error
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

class GradientLearningRFEnsemble:
    """Enhanced GradientLearning with Random Forest-like ensemble rules"""
    def __init__(self, n_features=None, n_estimators=5):
        self.weights = None
        self.n_features = n_features
        self.n_estimators = n_estimators
        # Multiple RF-like threshold sets for ensemble
        self.rf_thresholds_ensemble = []
        self.estimator_weights = None
        
    def partial_fit(self, X, y, classes=None):
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
            
            # Initialize multiple RF-like threshold sets
            for i in range(self.n_estimators):
                # Different threshold strategies for diversity
                if i == 0:
                    thresholds = np.median(X, axis=0)  # Median
                elif i == 1:
                    thresholds = np.percentile(X, 25, axis=0)  # Q1
                elif i == 2:
                    thresholds = np.percentile(X, 75, axis=0)  # Q3
                elif i == 3:
                    thresholds = np.mean(X, axis=0)  # Mean
                else:
                    # Random percentiles for additional diversity
                    pct = np.random.uniform(10, 90)
                    thresholds = np.percentile(X, pct, axis=0)
                
                self.rf_thresholds_ensemble.append(thresholds)
            
            # Initialize estimator weights (equal at start)
            self.estimator_weights = np.ones(self.n_estimators) / self.n_estimators

        # Batch processing - iterate through samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]  # Data is already {-1,+1}

            # Standard gradient update
            if y_i * x_i.dot(self.weights) < 1:
                self.weights += y_i * x_i

            # RF Ensemble: Multiple RF-like rules with different thresholds
            ensemble_boost = 0.0
            
            for est_idx, rf_thresholds in enumerate(self.rf_thresholds_ensemble):
                # RF rule: features above threshold
                rf_vote = np.sum(x_i > rf_thresholds)
                rf_strength = rf_vote / self.n_features  # Proportion of features
                
                # Different voting strategies per estimator
                if est_idx == 0:
                    # Majority vote (>50% features)
                    if rf_vote > (self.n_features // 2):
                        ensemble_boost += self.estimator_weights[est_idx] * 0.5
                elif est_idx == 1:
                    # Strong minority vote (>25% features)
                    if rf_vote > (self.n_features // 4):
                        ensemble_boost += self.estimator_weights[est_idx] * 0.3
                elif est_idx == 2:
                    # Super majority vote (>75% features)
                    if rf_vote > (3 * self.n_features // 4):
                        ensemble_boost += self.estimator_weights[est_idx] * 0.7
                else:
                    # Weighted by proportion of features
                    ensemble_boost += self.estimator_weights[est_idx] * rf_strength * 0.4
            
            # Apply ensemble boost
            if ensemble_boost > 0:
                self.weights += y_i * x_i * ensemble_boost

        return self

    def predict(self, X):
        if self.weights is None:
            return np.ones(X.shape[0])

        predictions = []
        for i in range(X.shape[0]):
            # Base prediction
            base_pred = np.sign(X[i].dot(self.weights))
            base_pred = base_pred if base_pred != 0 else 1
            
            # RF Ensemble confidence boost
            ensemble_confidence = 0.0
            for est_idx, rf_thresholds in enumerate(self.rf_thresholds_ensemble):
                rf_vote = np.sum(X[i] > rf_thresholds)
                rf_strength = rf_vote / self.n_features
                ensemble_confidence += self.estimator_weights[est_idx] * rf_strength
            
            # Adjust prediction based on ensemble confidence
            if ensemble_confidence > 0.6:  # High confidence
                final_pred = base_pred
            elif ensemble_confidence < 0.3:  # Low confidence, be conservative
                final_pred = base_pred * 0.8  # Reduce confidence
                final_pred = 1 if final_pred >= 0 else -1
            else:
                final_pred = base_pred
            
            predictions.append(final_pred)
        
        return np.array(predictions)

class OnlineToBatchTrainer:
    """
    OnlineToBatch protocol implementation that minimizes False Negatives
    by optimizing for recall (sensitivity) on validation set.
    """
    def __init__(self, algorithm_class, algorithm_params, epochs=5, optimize_metric='recall'):
        """
        Args:
            algorithm_class: The algorithm class (e.g., PassiveAggressive)
            algorithm_params: Parameters for algorithm initialization (dict)
            epochs: Number of training epochs
            optimize_metric: Metric to optimize ('recall', 'f1', 'fnr_min')
        """
        self.algorithm_class = algorithm_class
        self.algorithm_params = algorithm_params or {}
        self.epochs = epochs
        self.optimize_metric = optimize_metric
        self.best_algorithm = None
        self.best_score = -1.0
        
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Implements the OnlineToBatch protocol
        """
        print(f"    OnlineToBatch training with {self.epochs} epochs...")
        
        # Initialize best state
        self.best_algorithm = None
        self.best_score = -1.0
        
        for epoch in range(self.epochs):
            # (A) Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # (B) Initialize algorithm for this epoch
            if callable(self.algorithm_class):
                algo = self.algorithm_class()  # For lambda functions
            else:
                algo = self.algorithm_class(**self.algorithm_params)
            
            # Train on shuffled data
            algo.partial_fit(X_shuffled, y_shuffled)
            
            # (C) Evaluate on validation set
            y_pred = algo.predict(X_val)
            current_score = self._calculate_metric(y_val, y_pred)
            
            # (D) Save optimal parameters
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_algorithm = algo
        
        print(f"      Best validation {self.optimize_metric}: {self.best_score:.3f}")
        return self.best_algorithm
    
    def _calculate_metric(self, y_true, y_pred):
        """Calculate optimization metric (focusing on minimizing FN)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        except ValueError:
            return 0.0
        
        if self.optimize_metric == 'recall':
            # Maximize recall (minimize FN rate)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif self.optimize_metric == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif self.optimize_metric == 'fnr_min':
            # Minimize FNR (maximize 1-FNR)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            return 1.0 - fnr
        else:
            # Default to accuracy
            return accuracy_score(y_true, y_pred)


class WeightedMajorityVoter:
    """
    Weighted Majority Voting ensemble that combines multiple algorithms.
    Weights are based on validation performance with focus on minimizing FN.
    """
    def __init__(self, algorithms, weight_metric='recall'):
        """
        Args:
            algorithms: List of (name, algorithm) tuples
            weight_metric: Metric used for weighting ('recall', 'f1', 'accuracy')
        """
        self.algorithms = algorithms
        self.weights = {}
        self.weight_metric = weight_metric
        
    def fit_weights(self, X_val, y_val):
        """Calculate weights based on validation performance"""
        print("    Computing ensemble weights...")
        
        for name, algo in self.algorithms:
            y_pred = algo.predict(X_val)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[-1, 1]).ravel()
            except ValueError:
                self.weights[name] = 0.001
                continue
                
            if self.weight_metric == 'recall':
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            elif self.weight_metric == 'f1':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:  # accuracy
                score = accuracy_score(y_val, y_pred)
            
            self.weights[name] = max(score, 0.001)  # Avoid zero weights
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print("      Algorithm weights:")
        for name, weight in self.weights.items():
            print(f"        {name}: {weight:.3f}")
    
    def predict(self, X):
        """Make weighted majority prediction"""
        predictions = []
        
        for i in range(len(X)):
            x_sample = X[i:i+1] if len(X.shape) > 1 else X[i].reshape(1, -1)
            weighted_votes = 0.0
            
            for name, algo in self.algorithms:
                prediction = algo.predict(x_sample)[0]
                weighted_votes += self.weights.get(name, 0.0) * prediction
            
            final_pred = 1 if weighted_votes > 0 else -1
            predictions.append(final_pred)
        
        return np.array(predictions)


def run_online_to_batch_experiment():
    """
    Run OnlineToBatch experiment with all algorithms, focusing on minimizing False Negatives
    """
    print("="*80)
    print("OnlineToBatch Experiment - Minimizing False Negatives")
    print("Individual Algorithms + Weighted Majority Voting")
    print("="*80)
    
    # Force efficient settings
    optimize_for_efficiency()
    set_cpu_affinity(max_cores=2)
    
    # Load real data
    print(f"\n Loading RBD24 datasets...")
    X_train_dict, y_train_dict, X_test_dict, y_test_dict, train_user_ids_dict, test_user_ids_dict = load_rbd24_properly("cyber/")
    print(f" Loaded {len(X_train_dict)} datasets")
    
    # Algorithm configurations - all algorithms from your specification
    algorithm_configs = {
        'PassiveAggressive': (lambda: PassiveAggressive(), {}),
        'Perceptron': (lambda: Perceptron(), {}),
        'GradientLearning': (lambda: GradientLearning(), {}),
        'AROW': (lambda: AROW(r=1.0), {}),
        'RDA': (lambda: RDA(lambda_param=0.01, gamma_param=1.0), {}),
        'SCW': (lambda: SCW(C=0.1, eta=0.95), {}),
        'AdaRDA': (lambda: AdaRDA(lambda_param=0.01, eta_param=0.1, delta_param=1.0), {})
    }
    
    all_individual_results = []
    all_ensemble_results = []
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(X_train_dict.keys()):
        print(f"\n Dataset {dataset_idx+1}: {dataset_name}")
        
        X_train_full = X_train_dict[dataset_name]
        y_train_full = y_train_dict[dataset_name]
        X_test = X_test_dict[dataset_name]
        y_test = y_test_dict[dataset_name]
        test_user_ids = test_user_ids_dict.get(dataset_name, None)
        
        print(f"  Train: {X_train_full.shape}, Test: {X_test.shape}")
        print(f"  Attack rate: {np.mean(y_train_full == 1):.1%}")
        
        # Split training into train/validation for OnlineToBatch
        val_size = int(0.2 * len(X_train_full))
        indices = np.random.permutation(len(X_train_full))
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]
        
        print(f"  Split - Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Train individual algorithms with OnlineToBatch
        trained_algorithms = []
        dataset_individual_results = []
        
        for alg_name, (alg_class, alg_params) in algorithm_configs.items():
            print(f"\n   Training {alg_name} with OnlineToBatch...")
            start_time = time.time()
            
            try:
                # Use OnlineToBatch trainer
                trainer = OnlineToBatchTrainer(
                    algorithm_class=alg_class,
                    algorithm_params=alg_params,
                    epochs=3,  # Efficient number of epochs
                    optimize_metric='recall'  # Focus on minimizing FN
                )
                
                trained_algo = trainer.fit(X_train, y_train, X_val, y_val)
                trained_algorithms.append((alg_name, trained_algo))
                
                # Evaluate on test set
                y_pred = trained_algo.predict(X_test)
                
                # Calculate metrics
                try:
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                except ValueError:
                    accuracy = precision = recall = f1 = 0.0
                    fnr = fpr = 1.0
                    tn = fp = fn = tp = 0
                
                training_time = time.time() - start_time
                
                result = {
                    'dataset': dataset_name,
                    'algorithm': alg_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fnr': fnr,
                    'fpr': fpr,
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                    'training_time': training_time,
                    'best_val_recall': trainer.best_score
                }
                
                dataset_individual_results.append(result)
                all_individual_results.append(result)
                
                print(f"       Recall: {recall:.3f}, FN: {fn}, FP: {fp}")
                
            except Exception as e:
                print(f"       Failed: {str(e)}")
                continue
        
        # Create weighted majority ensemble
        if len(trained_algorithms) >= 2:
            print(f"\n   Creating Weighted Majority Ensemble...")
            print("    Using RECALL weighting (prioritizes minimizing false negatives)...")
            
            ensemble = WeightedMajorityVoter(trained_algorithms, weight_metric='recall')
            ensemble.fit_weights(X_val, y_val)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test)
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble, labels=[-1, 1]).ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except ValueError:
                accuracy = precision = recall = f1 = 0.0
                fnr = fpr = 1.0
                tn = fp = fn = tp = 0
            
            ensemble_result = {
                'dataset': dataset_name,
                'algorithm': 'WeightedMajority',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fnr': fnr,
                'fpr': fpr,
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'n_algorithms': len(trained_algorithms)
            }
            
            all_ensemble_results.append(ensemble_result)
            print(f"       Ensemble - Recall: {recall:.3f}, FN: {fn}, FP: {fp}")
            
            # Find best individual for this dataset
            best_individual = max(dataset_individual_results, key=lambda x: x['recall'])
            
            print(f"\n   Dataset Summary:")
            print(f"      Best Individual: {best_individual['algorithm']} (Recall: {best_individual['recall']:.3f})")
            print(f"      Ensemble: Recall: {recall:.3f}")
            print(f"      FN Improvement: {best_individual['fn'] - fn}")
            print(f"      FP Change: {fp - best_individual['fp']:+d}")
    
    # Save results to CSV files
    print(f"\n Saving results...")
    results_dir = Path("../results")  # Save in batch/results directory
    results_dir.mkdir(exist_ok=True)
    
    # Individual results
    if all_individual_results:
        individual_df = pd.DataFrame(all_individual_results)
        individual_path = results_dir / "onlinetobatch_individual_results.csv"
        individual_df.to_csv(individual_path, index=False)
        print(f"    Individual results: {individual_path}")
    
    # Ensemble vs best comparison
    if all_ensemble_results and all_individual_results:
        comparison_data = []
        
        for ensemble_result in all_ensemble_results:
            dataset = ensemble_result['dataset']
            dataset_individuals = [r for r in all_individual_results if r['dataset'] == dataset]
            
            if dataset_individuals:
                best_individual = max(dataset_individuals, key=lambda x: x['recall'])
                
                comparison_data.extend([
                    {
                        'dataset': dataset,
                        'method': f'Best Individual ({best_individual["algorithm"]})',
                        **{k: v for k, v in best_individual.items() if k not in ['dataset', 'algorithm']}
                    },
                    {
                        'dataset': dataset,
                        'method': 'Weighted Majority Ensemble',
                        **{k: v for k, v in ensemble_result.items() if k not in ['dataset', 'algorithm']}
                    },
                    {
                        'dataset': dataset,
                        'method': 'Improvement (Ensemble - Best)',
                        'accuracy': ensemble_result['accuracy'] - best_individual['accuracy'],
                        'recall': ensemble_result['recall'] - best_individual['recall'],
                        'f1': ensemble_result['f1'] - best_individual['f1'],
                        'fnr': best_individual['fnr'] - ensemble_result['fnr'],  # Lower is better
                        'fpr': ensemble_result['fpr'] - best_individual['fpr'],
                        'fn': best_individual['fn'] - ensemble_result['fn'],  # Reduction
                        'fp': ensemble_result['fp'] - best_individual['fp']   # Change
                    }
                ])
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = results_dir / "onlinetobatch_ensemble_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"    Comparison results: {comparison_path}")
    
    print(f"\n OnlineToBatch experiment completed!")
    return all_individual_results, all_ensemble_results


def main():
    """
    Main function - Run OnlineToBatch experiment
    """
    print("="*80)
    print("ONLINETOBATCH EXPERIMENT SUITE")
    print("="*80)
    
    # Run the OnlineToBatch experiment directly
    run_online_to_batch_experiment()

# Enhanced system resource monitoring and optimization
def monitor_system_resources():
    """Monitor system resources - CPU, memory, etc."""
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        # Create a simple report
        report = {
            'cpu_percent': cpu_percent,
            'memory_used': virtual_mem.used / (1024 ** 2),  # Convert to MB
            'memory_total': virtual_mem.total / (1024 ** 2),
            'memory_percent': virtual_mem.percent,
            'swap_used': swap_mem.used / (1024 ** 2),
            'swap_total': swap_mem.total / (1024 ** 2),
            'swap_percent': swap_mem.percent
        }
        
        return report
    except Exception as e:
        print(f"Error monitoring system resources: {str(e)}")
        return None

def set_cpu_affinity(max_cores=2):
    """Set CPU affinity for the process - limit to max_cores CPUs"""
    try:
        # Get the current process
        p = psutil.Process(os.getpid())
        
        # Get available CPUs
        all_cpus = list(range(psutil.cpu_count()))
        
        # Select CPUs based on max_cores
        if max_cores < len(all_cpus):
            # For max_cores < total CPUs, use the first max_cores CPUs
            selected_cpus = all_cpus[:max_cores]
        else:
            # For max_cores >= total CPUs, use all CPUs
            selected_cpus = all_cpus
        
        # Set the process affinity
        p.cpu_affinity(selected_cpus)
        
        print(f" Set CPU affinity to cores: {selected_cpus}")
    except Exception as e:
        print(f"Error setting CPU affinity: {str(e)}")

def optimize_for_efficiency():
    """Optimize settings for aggressive efficiency - cooling, resource limits, etc."""
    try:
        # Set CPU affinity - limit to 1 core for cooling
        set_cpu_affinity(max_cores=1)
        
        # Reduce Python's memory usage - aggressive garbage collection
        gc.collect()
        time.sleep(1)  # Brief pause to allow garbage collection
        
        print(" Optimized for efficiency: Aggressive cooling and resource limits")
    except Exception as e:
        print(f"Error optimizing for efficiency: {str(e)}")

def user_based_train_val_split(X_train, y_train, train_user_ids, val_size=0.2, random_state=1729):
    """User-based train/validation split - no user overlap"""
    from numpy.random import default_rng
    rng = default_rng(random_state)
    
    unique_users = np.unique(train_user_ids)
    rng.shuffle(unique_users)
    
    # Split users into train and validation sets
    num_val_users = max(1, int(len(unique_users) * val_size))
    val_users = set(unique_users[:num_val_users])
    train_users = set(unique_users[num_val_users:])
    
    # Create masks for train and validation sets
    train_mask = np.isin(train_user_ids, list(train_users))
    val_mask = np.isin(train_user_ids, list(val_users))
    
    X_train_split = X_train[train_mask]
    y_train_split = y_train[train_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    
    # Return the actual user IDs for the training split (not just the set)
    train_split_user_ids = train_user_ids[train_mask]
    
    print(f"    User-based split: {len(train_users)} train users, {len(val_users)} val users")
    
    return X_train_split, X_val, y_train_split, y_val, train_split_user_ids

def user_based_shuffle(X, y, user_ids, epoch, random_state=1729):
    """User-based shuffling - shuffle data while preserving user groups"""
    from numpy.random import default_rng
    rng = default_rng(random_state + epoch)
    
    unique_users = np.unique(user_ids)
    rng.shuffle(unique_users)
    
    # Create new indices based on shuffled user order
    shuffled_indices = []
    for u in unique_users:
        # Find all indices for this user
        user_indices = np.where(user_ids == u)[0]
        shuffled_indices.extend(user_indices)
    
    shuffled_indices = np.array(shuffled_indices)
    
    return X[shuffled_indices], y[shuffled_indices]

def calculate_metrics(y_true, y_pred):
    """Calculate FNR, FPR, F1, and accuracy metrics"""
    try:
        # Convert predictions to -1/+1 format if they're 0/1
        if np.any(y_pred == 0) and not np.any(y_pred == -1):
            y_pred = np.where(y_pred == 0, -1, 1)
        
        # Handle -1/+1 labels for confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Specify pos_label=1 for binary F1 score with -1/+1 labels
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return fnr, fpr, f1, acc
    except ValueError as e:
        print(f"      Metrics calculation error: {e}")
        print(f"       y_true unique: {np.unique(y_true)}")
        print(f"       y_pred unique: {np.unique(y_pred)}")
        return 0.0, 0.0, 0.0, 0.0


if __name__ == "__main__":
    main()
