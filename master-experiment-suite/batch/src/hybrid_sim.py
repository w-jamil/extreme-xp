#!/usr/bin/env python3
"""
HYBRID TEMPORAL CYBERSECURITY EXPERIMENT
Clean implementation:
1. Aggregate features by timestamp for batch training
2. Sequential online adaptation per user
3. Compare with baseline RCL_BCE (offline only)
"""

import os
import sys
import copy
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron
from numpy.random import default_rng
# Import Zenodo data handler
from data_handler import prepare_data_from_zenodo

# Add path for algorithm imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import online learning algorithms
try:
    from algorithms import OGL, AROW, RDA, SCW, AdaRDA, RCL_BCE
    print("✓ Imported online learning algorithms")
except ImportError as e:
    print(f"✗ Error importing algorithms: {e}")
    sys.exit(1)

# Configuration
DATASETS = [
    'Phishing_desktop.parquet',
    'Phishing_smartphone.parquet', 
    'NonEnc_desktop.parquet',
    'NonEnc_smartphone.parquet',
    'P2P_desktop.parquet',
    'P2P_smartphone.parquet'
]

# Algorithm factories (using dl.py-compatible random states)
rng = default_rng(seed=1729)  # Match dl.py approach
algo_random_state = rng.integers(2**31)  # Derive random state like dl.py

ALGORITHMS = {
    'PassiveAggressive': lambda: PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=algo_random_state),
    'Perceptron': lambda: Perceptron(alpha=0.0001, max_iter=1000, random_state=algo_random_state),
    'OGL': lambda: OGL(learning_rate=0.01, l2_reg=0.01, random_state=algo_random_state),
    'AROW': lambda: AROW(r=0.1, random_state=algo_random_state),
    'RDA': lambda: RDA(learning_rate=0.01, l1_reg=0.001, random_state=algo_random_state),
    'SCW': lambda: SCW(phi=1.0, C=0.5, random_state=algo_random_state),
    'AdaRDA': lambda: AdaRDA(learning_rate=0.01, l1_reg=0.001, random_state=algo_random_state),
    'RCL_BCE': lambda: RCL_BCE(learning_rate=0.01, l2_reg=0.01, random_state=algo_random_state)
}


def temporal_aggregate_split(data, test_size=0.2):
    """Split data with temporal aggregation approach (matching dl.py method):
    1. Split users into train/validation/test using same approach as dl.py _split_and_scale
    2. Aggregate train data by timestamp (sum features across users)  
    3. Keep test data as individual user sequences
    """
    print("\n  === TEMPORAL AGGREGATION SPLIT ===")
    
    # Match dl.py approach exactly: use default_rng with seed 1729
    rng = default_rng(1729)
    
    # Get unique users and sort for repeatability (like dl.py)
    uids = data['user_id']
    unique_uids = sorted(set(uids))  # guarantee repeatability like dl.py
    rng.shuffle(unique_uids)
    
    # Calculate test and validation users exactly like dl.py (proportional_to_users=True)
    num_test_users = round(len(unique_uids) * test_size)
    num_val_users = round(len(unique_uids) * test_size)
    test_uids = unique_uids[:num_test_users]  # First users go to TEST (like dl.py)
    val_uids = unique_uids[num_test_users:num_test_users + num_val_users]  # Next users go to VALIDATION
    train_uids = unique_uids[num_test_users + num_val_users:]  # Remaining users go to TRAIN
    
    print(f"    User split (matching dl.py): {len(train_uids)} train, {len(val_uids)} validation, {len(test_uids)} test users")
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'user_id', 'label']]
    
    def aggregate_by_timestamp(user_ids):
        """Helper function to aggregate data by timestamp for given users"""
        subset_data = data[data['user_id'].isin(user_ids)]
        
        agg_data = subset_data.groupby('timestamp').agg({
            **{col: 'sum' for col in feature_cols},  # Sum features across users
            'label': lambda x: 1 if x.sum() > len(x)/2 else 0,  # Majority vote for labels
            'user_id': 'count'  # Number of users (becomes feature)
        }).reset_index()
        
        agg_data.rename(columns={'user_id': 'user_count'}, inplace=True)
        agg_data = agg_data.sort_values('timestamp')
        
        X = agg_data[feature_cols + ['user_count']].values
        y = np.where(agg_data['label'].values == 0, -1, 1)
        
        return X, y
    
    # === TRAINING DATA: Aggregate by timestamp ===
    X_train, y_train = aggregate_by_timestamp(train_uids)
    print(f"    Aggregated training: {len(X_train)} timestamps, {X_train.shape[1]} features")
    
    # === VALIDATION DATA: Aggregate by timestamp ===
    X_val, y_val = aggregate_by_timestamp(val_uids)
    print(f"    Aggregated validation: {len(X_val)} timestamps")
    
    # === TEST DATA: Individual user sequences ===
    test_data = data[data['user_id'].isin(test_uids)].sort_values(['user_id', 'timestamp'])
    
    user_sequences = {}
    for user in test_uids:
        user_data = test_data[test_data['user_id'] == user]
        if len(user_data) > 0:
            # Add user_count=1 for individual users
            X_user = user_data[feature_cols].values
            X_user = np.column_stack([X_user, np.ones(len(X_user))])  # Add user_count=1
            y_user = np.where(user_data['label'].values == 0, -1, 1)
            user_sequences[user] = (X_user, y_user)
    
    print(f"    Test sequences: {len(user_sequences)} users")
    
    return X_train, y_train, X_val, y_val, user_sequences


def train_batch_models(X_train, y_train, X_val, y_val, epochs=3):
    """Train all models on aggregated temporal data with multiple epochs and shuffling"""
    print(f"\n  === BATCH TRAINING ON AGGREGATED DATA ({epochs} EPOCHS) ===")
    print(f"    Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    trained_models = {}
    rng = np.random.default_rng(seed=1729)  # For consistent shuffling
    
    for alg_name, algorithm_factory in ALGORITHMS.items():
        print(f"    Training {alg_name} for {epochs} epochs...")
        
        try:
            start_time = time.time()
            algorithm = algorithm_factory()
            
            # Store models from each epoch to find best one
            epoch_models = []
            epoch_val_f1_scores = []
            
            for epoch in range(epochs):
                print(f"      Epoch {epoch+1}/{epochs}:")
                
                # SHUFFLE training data for this epoch
                indices = rng.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                # Train for one epoch
                if epoch == 0:
                    algorithm.fit(X_shuffled, y_shuffled)
                else:
                    # Continue training with partial_fit if available
                    if hasattr(algorithm, 'partial_fit'):
                        algorithm.partial_fit(X_shuffled, y_shuffled)
                    else:
                        # Re-fit for algorithms without partial_fit (using shuffled data)
                        algorithm.fit(X_shuffled, y_shuffled)
                
                # Evaluate on VALIDATION data (not training data - prevents overfitting)
                y_val_pred = algorithm.predict(X_val)
                tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred, labels=[-1, 1]).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store copy of model and VALIDATION F1 score
                epoch_models.append(copy.deepcopy(algorithm))
                epoch_val_f1_scores.append(f1)
                
                print(f"        Validation F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
            
            training_time = time.time() - start_time
            
            # Find best epoch based on VALIDATION F1 score
            best_epoch_idx = np.argmax(epoch_val_f1_scores)
            best_model = epoch_models[best_epoch_idx]
            best_val_f1 = epoch_val_f1_scores[best_epoch_idx]
            
            trained_models[alg_name] = {
                'model': best_model,
                'best_epoch': best_epoch_idx + 1,
                'best_val_f1': best_val_f1,
                'all_epoch_val_f1s': epoch_val_f1_scores,
                'training_time': training_time
            }
            
            print(f"    ✓ {alg_name} trained in {training_time:.2f}s")
            print(f"      Best: Epoch {best_epoch_idx+1} (Val F1={best_val_f1:.3f})")
            
        except Exception as e:
            print(f"    ✗ {alg_name} failed: {e}")
            continue
    
    return trained_models


def sequential_user_testing(trained_models, user_sequences):
    """
    WEIGHT USAGE STRATEGY:
    
    1. BATCH TRAINING PHASE (completed before this function):
       - All algorithms trained on aggregated temporal data
       - Each algorithm learns initial weights W₀ from collective patterns
    
    2. SEQUENTIAL USER TESTING (this function):
       For each test user:
       
       A) RCL_BCE (BASELINE):
          - Uses original batch-trained weights W₀ (NO adaptation)
          - Predicts all user samples at once: y = f(X_user, W₀)
          - Static approach: same weights for all users
       
       B) ONLINE ALGORITHMS:
          - START: Copy batch-trained weights W₀ for each user
          - ADAPT: For each sample x_t, y_t:
            * Predict: ŷ_t = f(x_t, W_t)
            * Update: W_{t+1} = update_rule(W_t, x_t, y_t, ŷ_t)
          - RESET: Fresh W₀ copy for next user (no knowledge transfer)
    
    This tests: Can initial batch weights + online adaptation beat static batch weights?
    """
    print("\n  === SEQUENTIAL USER TESTING (ONLINE UPDATES) ===")
    
    results = {alg_name: [] for alg_name in trained_models.keys()}
    
    for user_idx, (user, (X_user, y_user)) in enumerate(user_sequences.items()):
        print(f"    User {user_idx+1}/{len(user_sequences)}: {user} ({len(X_user)} samples)")
        
        for alg_name, model_info in trained_models.items():
            try:
                if alg_name == 'RCL_BCE':
                    # BASELINE: Use batch-trained weights W₀, NO adaptation
                    model = model_info['model']
                    y_pred = model.predict(X_user)
                    
                else:
                    # ONLINE: Copy batch weights W₀, then adapt sequentially per user
                    model = copy.deepcopy(model_info['model'])  # Fresh W₀ copy
                    y_pred = []
                    
                    for i in range(len(X_user)):
                        # Predict first
                        pred = model.predict(X_user[i:i+1])[0]
                        y_pred.append(pred)
                        
                        # Then update model with true label
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_user[i:i+1], y_user[i:i+1])
                        else:
                            # For algorithms without partial_fit, retrain on recent samples
                            recent_start = max(0, i-100)  # Use last 100 samples
                            model.fit(X_user[recent_start:i+1], y_user[recent_start:i+1])
                    
                    y_pred = np.array(y_pred)
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_user, y_pred, labels=[-1, 1]).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(y_user)
                
                results[alg_name].append({
                    'user': user,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                })
                
            except Exception as e:
                print(f"      ✗ {alg_name} failed on user {user}: {e}")
                continue
    
    return results


def best_weights_testing(trained_models, user_sequences):
    """
    BEST WEIGHTS STRATEGY (NO ONLINE UPDATES):
    
    - All algorithms use their best epoch weights (selected by F1 score during training)
    - NO online adaptation: same weights used for all users and all samples
    - Pure batch learning approach: W_best applied statically
    - Tests: How good are the best batch weights without any personalization?
    """
    print("\n  === BEST WEIGHTS TESTING (NO ONLINE UPDATES) ===")
    
    results = {alg_name: [] for alg_name in trained_models.keys()}
    
    for user_idx, (user, (X_user, y_user)) in enumerate(user_sequences.items()):
        print(f"    User {user_idx+1}/{len(user_sequences)}: {user} ({len(X_user)} samples)")
        
        for alg_name, model_info in trained_models.items():
            try:
                # Use best epoch model without any updates
                model = model_info['model']
                y_pred = model.predict(X_user)
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_user, y_pred, labels=[-1, 1]).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(y_user)
                
                results[alg_name].append({
                    'user': user,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                })
                
            except Exception as e:
                print(f"      ✗ {alg_name} failed on user {user}: {e}")
                continue
    
    return results


def aggregate_results(results, dataset_name, training_times, approach_name="ONLINE"):
    """Aggregate results across all users"""
    print(f"\n  === AGGREGATING RESULTS ({approach_name}) ===")
    
    final_results = []
    
    for alg_name, user_metrics in results.items():
        if not user_metrics:
            continue
        
        # Aggregate confusion matrix elements
        total_tp = sum(m['tp'] for m in user_metrics)
        total_tn = sum(m['tn'] for m in user_metrics)
        total_fp = sum(m['fp'] for m in user_metrics)
        total_fn = sum(m['fn'] for m in user_metrics)
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        
        # Calculate FPR and FNR
        overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0  # False Positive Rate
        overall_fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0  # False Negative Rate
        
        # Average per-user metrics
        avg_f1 = np.mean([m['f1'] for m in user_metrics])
        avg_precision = np.mean([m['precision'] for m in user_metrics])
        avg_recall = np.mean([m['recall'] for m in user_metrics])
        
        # Set approach based on algorithm and testing type
        if approach_name == "BEST_WEIGHTS":
            approach = "BEST_WEIGHTS"
        elif alg_name == "RCL_BCE":
            approach = "BASELINE"
        else:
            approach = "ONLINE"
        
        result = {
            'dataset': dataset_name,
            'algorithm': alg_name,
            'approach': approach,
            'overall_f1': overall_f1,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_accuracy': overall_accuracy,
            'overall_fpr': overall_fpr,
            'overall_fnr': overall_fnr,
            'avg_user_f1': avg_f1,
            'avg_user_precision': avg_precision,
            'avg_user_recall': avg_recall,
            'training_time': training_times.get(alg_name, 0),
            'n_users': len(user_metrics),
            'total_samples': total_tp + total_tn + total_fp + total_fn,
            'total_tp': total_tp,
            'total_tn': total_tn, 
            'total_fp': total_fp,
            'total_fn': total_fn
        }
        
        final_results.append(result)
        
        print(f"    {alg_name:>15} ({approach:>11}): F1={overall_f1:.3f} (user_avg={avg_f1:.3f}) "
              f"P={overall_precision:.3f} R={overall_recall:.3f}")
        print(f"                        FPR={overall_fpr:.3f} FNR={overall_fnr:.3f} "
              f"TP={total_tp} TN={total_tn} FP={total_fp} FN={total_fn}")
    
    return final_results


def main_experiment():
    """Main experiment function"""
    print("="*80)
    print("HYBRID TEMPORAL CYBERSECURITY EXPERIMENT")
    print("Aggregated Training + Sequential User Adaptation (3 EPOCHS)")
    print("="*80)
    
    all_results = []
    os.makedirs('results', exist_ok=True)
    
    for dataset_file in DATASETS:
        dataset_name = dataset_file.replace('.parquet', '')
        print(f"\n{'='*60}")
        print(f"PROCESSING: {dataset_name}")
        print('='*60)
        
        try:
            # Load dataset
            data_path = os.path.join('cyber', dataset_file)
            if not os.path.exists(data_path):
                print(f"✗ Dataset not found: {data_path}")
                continue
            
            data = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(data)} samples")
            
            # Check required columns
            required_cols = ['timestamp', 'user_id', 'label']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"✗ Missing columns: {missing_cols}")
                continue
            
            # Sort data properly
            data = data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
            print(f"✓ Data sorted: {data['user_id'].nunique()} users, {len(data)} samples")
            
            # Temporal aggregation split
            X_train, y_train, X_val, y_val, user_sequences = temporal_aggregate_split(data)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Scale user sequences
            user_sequences_scaled = {}
            for user, (X_user, y_user) in user_sequences.items():
                X_user_scaled = scaler.transform(X_user)
                user_sequences_scaled[user] = (X_user_scaled, y_user)
            
            print(f"✓ Data standardized")
            
            # Train models on aggregated data (3 epochs, select best using validation)
            trained_models = train_batch_models(X_train_scaled, y_train, X_val_scaled, y_val, epochs=3)
            
            if not trained_models:
                print("✗ No models trained successfully")
                continue
            
            # Show training summary
            print(f"\n  === TRAINING SUMMARY ===")
            for alg_name, model_info in trained_models.items():
                best_epoch = model_info['best_epoch']
                best_val_f1 = model_info['best_val_f1']
                all_val_f1s = model_info['all_epoch_val_f1s']
                print(f"    {alg_name:>15}: Best epoch {best_epoch}/3 (Val F1={best_val_f1:.3f}), "
                      f"All epochs Val F1: {[f'{f1:.3f}' for f1 in all_val_f1s]}")
            
            # APPROACH 1: Sequential user testing with online updates
            user_results_online = sequential_user_testing(trained_models, user_sequences_scaled)
            
            # APPROACH 2: Best weights testing (no online updates)
            user_results_best = best_weights_testing(trained_models, user_sequences_scaled)
            
            # Aggregate results for both approaches
            training_times = {name: info['training_time'] for name, info in trained_models.items()}
            
            # Online approach results
            dataset_results_online = aggregate_results(user_results_online, dataset_name, training_times, "ONLINE")
            all_results.extend(dataset_results_online)
            
            # Best weights approach results  
            dataset_results_best = aggregate_results(user_results_best, dataset_name, training_times, "BEST_WEIGHTS")
            all_results.extend(dataset_results_best)
            
        except Exception as e:
            print(f"✗ Dataset {dataset_name} failed: {e}")
            continue
    
    # Save and display results
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df = pd.DataFrame(all_results)
        results_file = f'results/hybrid_temporal_3epochs_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED")
        print("="*80)
        
        print("\nFINAL RESULTS SUMMARY:")
        print("-" * 80)
        
        for dataset in results_df['dataset'].unique():
            print(f"\nDATASET: {dataset}")
            dataset_results = results_df[results_df['dataset'] == dataset]
            
            # Group by approach
            for approach in ['BASELINE', 'ONLINE', 'BEST_WEIGHTS']:
                approach_results = dataset_results[dataset_results['approach'] == approach]
                if len(approach_results) == 0:
                    continue
                    
                print(f"\n  {approach} APPROACH:")
                for _, row in approach_results.iterrows():
                    print(f"    {row['algorithm']:>15}: "
                          f"F1={row['overall_f1']:.3f} P={row['overall_precision']:.3f} "
                          f"R={row['overall_recall']:.3f} FPR={row['overall_fpr']:.3f} "
                          f"({row['n_users']} users)")
        
        print(f"\n✓ Results saved: {results_file}")
        print("\nAPPROACH COMPARISON:")
        print("- BASELINE: RCL_BCE trained on aggregated data, no user adaptation")
        print("- ONLINE: All algorithms start with best epoch weights, then adapt per user")  
        print("- BEST_WEIGHTS: All algorithms use best epoch weights, NO adaptation")
        print("- Training: 3 epochs with SHUFFLED data, best epoch selected by VALIDATION F1 score")
        print("- Data split: 60% train, 20% validation, 20% test users")
        
    else:
        print("\n" + "="*80)
        print("NO RESULTS - ALL DATASETS FAILED")
        print("="*80)


if __name__ == "__main__":
    # --- Zenodo Data Preparation ---
    ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
    DATA_DIRECTORY = 'cyber/'
    print(f"Using data directory: {DATA_DIRECTORY}")
    missing_files = [f for f in DATASETS if not os.path.exists(os.path.join(DATA_DIRECTORY, f))]
    if missing_files:
        print(f"Missing files: {missing_files}. Downloading from Zenodo...")
        data_ready = prepare_data_from_zenodo(ZENODO_ARCHIVE_URL, DATA_DIRECTORY)
        if not data_ready:
            print("ERROR: Data preparation from Zenodo failed. Experiment will not run.")
            sys.exit(1)
    else:
        print("All required data files are present. Skipping download.")
    main_experiment()
