def main_experiment():
    """Main experiment function"""
    print("="*80)
    print("BATCH TEMPORAL CYBERSECURITY EXPERIMENT")
    print("Multiple Training Strategies + Fixed Weight Testing (3 EPOCHS)")
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
            X_train, y_train, X_val, y_val, train_user_sequences, val_user_sequences, test_user_sequences = temporal_aggregate_split(data)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Scale user sequences for all splits
            def scale_user_sequences(sequences):
                scaled_sequences = {}
                for user, (X_user, y_user) in sequences.items():
                    X_user_scaled = scaler.transform(X_user)
                    scaled_sequences[user] = (X_user_scaled, y_user)
                return scaled_sequences
            
            train_user_sequences_scaled = scale_user_sequences(train_user_sequences)
            val_user_sequences_scaled = scale_user_sequences(val_user_sequences)
            test_user_sequences_scaled = scale_user_sequences(test_user_sequences)
            
            print(f"✓ Data standardized")
            
            # Train models on aggregated data (3 epochs, select best using validation)
            trained_models = train_batch_models(X_train_scaled, y_train, X_val_scaled, y_val, epochs=3)
            
            if not trained_models:
                print("✗ No models trained successfully")
                continue
            
            # Show training summary
            print(f"\n  === AGGREGATED TRAINING SUMMARY ===")
            for alg_name, model_info in trained_models.items():
                best_epoch = model_info['best_epoch']
                best_val_f1 = model_info['best_val_f1']
                all_val_f1s = model_info['all_epoch_val_f1s']
                print(f"    {alg_name:>15}: Best epoch {best_epoch}/3 (Val F1={best_val_f1:.3f}), "
                      f"All epochs Val F1: {[f'{f1:.3f}' for f1 in all_val_f1s]}")
            
            # Train individual user models (select user with minimum F1)
            user_trained_models = train_individual_user_models(train_user_sequences_scaled, val_user_sequences_scaled, epochs=3)
            
            if not user_trained_models:
                print(" No individual user models trained successfully")
                continue
                
            # Show individual user training summary
            print(f"\n  === AVERAGED USER TRAINING SUMMARY ===")
            for alg_name, model_info in user_trained_models.items():
                avg_val_f1 = model_info['avg_val_f1']
                n_users = model_info['n_users_averaged']
                all_f1s = model_info['all_user_val_f1s']
                print(f"    {alg_name:>15}: Averaged {n_users} users (Avg val F1={avg_val_f1:.3f})")
                print(f"                        F1 range=[{min(all_f1s):.3f}, {max(all_f1s):.3f}]")
            
            # Best weights testing
            user_results_best = best_weights_testing(trained_models, test_user_sequences_scaled)
            
            # Individual user weights testing
            user_results_individual = individual_user_testing(user_trained_models, test_user_sequences_scaled)
            
            # Aggregate results for all approaches
            training_times = {name: info['training_time'] for name, info in trained_models.items()}
            user_training_times = {name: info['training_time'] for name, info in user_trained_models.items()}

            # Best weights approach results
            dataset_results_best = aggregate_results(user_results_best, dataset_name, training_times, "BEST_WEIGHTS")
            all_results.extend(dataset_results_best)
            
            # Individual user weights approach results
            dataset_results_individual = aggregate_results(user_results_individual, dataset_name, user_training_times, "INDIVIDUAL_USER")
            all_results.extend(dataset_results_individual)
            
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
            for approach in ['BASELINE', 'BATCH', 'BEST_WEIGHTS', 'INDIVIDUAL_USER']:
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
        print("- BASELINE: RCL_BCE trained on aggregated data, batch prediction")
        print("- BEST_WEIGHTS: All algorithms trained on aggregated data, best epoch selected, EFFICIENT batch prediction")  
        print("- INDIVIDUAL_USER: Each algorithm trained on individual users, AVERAGE best F1 weights, EFFICIENT sequential prediction")
        print("- Training: 3 epochs with SHUFFLED data, best epoch selected by VALIDATION F1 score")
        print("- Data split: 60% train, 20% validation, 20% test users")
        print("- Individual User Logic: Average the best F1 weights from all users → consensus model")
        print("- Testing: ALL approaches use FIXED weights (NO online updates during testing)")
        print("- Efficiency: Use matrix multiplication (X @ weights + bias) instead of model.predict() calls")
        print("- Sequential vs Batch: Sequential processes samples one-by-one, Batch processes all at once")
        print("- Weight Averaging: Creates robust model from individual user patterns")
        
    else:
        print("\n" + "="*80)
        print("NO RESULTS - ALL DATASETS FAILED")
        print("="*80)
#!/usr/bin/env python3
"""
EFFICIENT BATCH TEMPORAL CYBERSECURITY EXPERIMENT
Pure batch learning with efficient matrix operations:
1. Individual user training + WEIGHT AVERAGING (fast, efficient)
2. All testing done with FIXED weights (NO online updates)
3. Matrix multiplication for predictions (X @ weights + bias)
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
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, LogisticRegression
from numpy.random import default_rng
# Import Zenodo data handler
from data_handler import prepare_data_from_zenodo

# Add path for algorithm imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import algorithms
try:
    from algorithms import OGL, AROW, RDA, SCW, AdaRDA, RCL_BCE
    print("✓ Imported online learning algorithms")
except ImportError as e:
    print(f"✗ Error importing algorithms: {e}")
    sys.exit(1)

# Configuration
DATASETS = [
    'NonEnc_desktop.parquet',
    'NonEnc_smartphone.parquet',
    'Phishing_desktop.parquet',
    'Phishing_smartphone.parquet', 
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
    
    # === INDIVIDUAL USER SEQUENCES FOR ALL SPLITS ===
    def get_user_sequences(user_ids, split_name):
        """Helper to get individual user sequences"""
        split_data = data[data['user_id'].isin(user_ids)].sort_values(['user_id', 'timestamp'])
        sequences = {}
        
        for user in user_ids:
            user_data = split_data[split_data['user_id'] == user]
            if len(user_data) > 0:
                # Add user_count=1 for individual users
                X_user = user_data[feature_cols].values
                X_user = np.column_stack([X_user, np.ones(len(X_user))])  # Add user_count=1
                y_user = np.where(user_data['label'].values == 0, -1, 1)
                sequences[user] = (X_user, y_user)
        
        print(f"    {split_name} individual sequences: {len(sequences)} users")
        return sequences
    
    train_user_sequences = get_user_sequences(train_uids, "Training")
    val_user_sequences = get_user_sequences(val_uids, "Validation") 
    test_user_sequences = get_user_sequences(test_uids, "Test")
    
    return X_train, y_train, X_val, y_val, train_user_sequences, val_user_sequences, test_user_sequences


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
            
            print(f"    {alg_name} trained in {training_time:.2f}s")
            print(f"      Best: Epoch {best_epoch_idx+1} (Val F1={best_val_f1:.3f})")
            
        except Exception as e:
            print(f"    {alg_name} failed: {e}")
            continue
    
    return trained_models





def extract_model_weights(model, alg_name):
    """Extract weights and bias from trained model for efficient matrix operations"""
    try:
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            # sklearn-style models (PassiveAggressive, Perceptron)
            weights = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            bias = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
            return weights, bias
            
        elif hasattr(model, 'w') and hasattr(model, 'b'):
            # Custom algorithm style (OGL, AROW, etc.)
            weights = model.w
            bias = model.b if hasattr(model, 'b') else 0.0
            return weights, bias
            
        elif hasattr(model, 'weights'):
            # Alternative custom style
            weights = model.weights[:-1] if len(model.weights) > 1 else model.weights
            bias = model.weights[-1] if len(model.weights) > 1 else 0.0
            return weights, bias
            
        else:
            # Fallback - try to find any weight-like attributes
            for attr in ['w', 'weights', 'coef_', 'weight']:
                if hasattr(model, attr):
                    w = getattr(model, attr)
                    if isinstance(w, np.ndarray):
                        weights = w.flatten() if w.ndim > 1 else w
                        bias = 0.0
                        return weights, bias
            
        print(f"    Warning: Could not extract weights for {alg_name}, using slow predict()")
        return None, None
        
    except Exception as e:
        print(f"    Warning: Weight extraction failed for {alg_name}: {e}")
        return None, None


def efficient_batch_prediction(X, weights, bias, threshold=0.0):
    """Efficient batch prediction using matrix multiplication"""
    # Compute scores: X @ weights + bias
    scores = np.dot(X, weights) + bias
    # Convert to binary predictions (-1, 1)
    predictions = np.where(scores > threshold, 1, -1)
    return predictions


def efficient_sequential_prediction(X, weights, bias, threshold=0.0):
    """Efficient sequential prediction (same result as batch, but computed sequentially)"""
    predictions = []
    for i in range(len(X)):
        # Compute score for single sample: x_i @ weights + bias
        score = np.dot(X[i], weights) + bias
        pred = 1 if score > threshold else -1
        predictions.append(pred)
    return np.array(predictions)


def best_weights_testing(trained_models, user_sequences):
    """
    BEST WEIGHTS STRATEGY (EFFICIENT MATRIX OPERATIONS):
    
    - Extract fixed weights from best epoch models
    - Use efficient matrix multiplication for batch prediction
    - NO online adaptation: same weights used for all users and all samples
    """
    print("\n  === BEST WEIGHTS TESTING (EFFICIENT BATCH PREDICTION) ===")
    
    results = {alg_name: [] for alg_name in trained_models.keys()}
    
    # Extract weights once for all algorithms
    algorithm_weights = {}
    for alg_name, model_info in trained_models.items():
        model = model_info['model']
        weights, bias = extract_model_weights(model, alg_name)
        algorithm_weights[alg_name] = (weights, bias, model)  # Keep model as fallback
    
    for user_idx, (user, (X_user, y_user)) in enumerate(user_sequences.items()):
        print(f"    User {user_idx+1}/{len(user_sequences)}: {user} ({len(X_user)} samples)")
        
        for alg_name in trained_models.keys():
            try:
                weights, bias, model = algorithm_weights[alg_name]
                
                if weights is not None:
                    # EFFICIENT: Use matrix multiplication
                    y_pred = efficient_batch_prediction(X_user, weights, bias)
                else:
                    # FALLBACK: Use model.predict() if weight extraction failed
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
    

def train_individual_user_models(train_user_sequences, val_user_sequences, epochs=3):
    """
    EFFICIENT INDIVIDUAL USER TRAINING STRATEGY:
    
    - Train models on individual users (small datasets → fast training)
    - For each user, select best epoch based on validation F1
    - AVERAGE the best F1 weights across all users
    - This creates a "consensus" model that captures individual patterns
    
    Logic: Average of best performing weights from each user → robust generalization
    """
    print(f"\n  === EFFICIENT INDIVIDUAL USER TRAINING ({epochs} EPOCHS) ===")
    print(f"    Training on {len(train_user_sequences)} individual users")
    
    user_trained_models = {}
    
    for alg_name, algorithm_factory in ALGORITHMS.items():
        print(f"    Training {alg_name} across all users...")
        
        all_user_best_weights = []  # Store best weights from each user
        all_user_best_biases = []   # Store best biases from each user
        all_user_val_f1s = []       # Store validation F1s for reporting
        
        try:
            start_time = time.time()
            
            for user_idx, (train_user, (X_train_user, y_train_user)) in enumerate(train_user_sequences.items()):
                if len(X_train_user) < 10:  # Skip users with too little data
                    continue
                    
                # Train model for this specific user (FAST - small dataset)
                algorithm = algorithm_factory()
                rng = np.random.default_rng(seed=1729 + user_idx)
                
                best_val_f1 = -1
                best_weights = None
                best_bias = None
                
                for epoch in range(epochs):
                    # Shuffle this user's training data
                    indices = rng.permutation(len(X_train_user))
                    X_shuffled = X_train_user[indices]
                    y_shuffled = y_train_user[indices]
                    
                    # Train for one epoch
                    if epoch == 0:
                        algorithm.fit(X_shuffled, y_shuffled)
                    else:
                        if hasattr(algorithm, 'partial_fit'):
                            algorithm.partial_fit(X_shuffled, y_shuffled)
                        else:
                            algorithm.fit(X_shuffled, y_shuffled)
                    
                    # Evaluate on validation set
                    val_predictions = []
                    val_true_labels = []
                    
                    for val_user, (X_val_user, y_val_user) in val_user_sequences.items():
                        try:
                            y_pred = algorithm.predict(X_val_user)
                            val_predictions.extend(y_pred)
                            val_true_labels.extend(y_val_user)
                        except:
                            continue
                    
                    if len(val_predictions) > 0:
                        tn, fp, fn, tp = confusion_matrix(val_true_labels, val_predictions, labels=[-1, 1]).ravel()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        if f1 > best_val_f1:
                            best_val_f1 = f1
                            # Extract weights from best epoch
                            weights, bias = extract_model_weights(algorithm, alg_name)
                            if weights is not None:
                                best_weights = weights
                                best_bias = bias
                
                # Store best weights from this user
                if best_weights is not None:
                    all_user_best_weights.append(best_weights)
                    all_user_best_biases.append(best_bias)
                    all_user_val_f1s.append(best_val_f1)
                
                print(f"      User {user_idx+1}: Best val F1={best_val_f1:.3f}")
            
            training_time = time.time() - start_time
            
            # AVERAGE all user best weights to create consensus model
            if len(all_user_best_weights) > 0:
                avg_weights = np.mean(all_user_best_weights, axis=0)
                avg_bias = np.mean(all_user_best_biases)
                avg_val_f1 = np.mean(all_user_val_f1s)
                
                # Create a simple model wrapper for the averaged weights
                class AveragedWeightsModel:
                    def __init__(self, weights, bias):
                        self.weights = weights
                        self.bias = bias
                    
                    def predict(self, X):
                        scores = np.dot(X, self.weights) + self.bias
                        return np.where(scores > 0.0, 1, -1)
                
                averaged_model = AveragedWeightsModel(avg_weights, avg_bias)
                
                user_trained_models[alg_name] = {
                    'model': averaged_model,
                    'averaged_weights': avg_weights,
                    'averaged_bias': avg_bias,
                    'avg_val_f1': avg_val_f1,
                    'n_users_averaged': len(all_user_best_weights),
                    'all_user_val_f1s': all_user_val_f1s,
                    'training_time': training_time
                }
                
                print(f"    {alg_name}: Averaged {len(all_user_best_weights)} user models (Avg val F1={avg_val_f1:.3f})")
                print(f"     Val F1 range: [{min(all_user_val_f1s):.3f}, {max(all_user_val_f1s):.3f}]")
            
        except Exception as e:
            print(f"    {alg_name} failed: {e}")
            continue
    
    return user_trained_models


def individual_user_testing(user_trained_models, test_user_sequences):
    """
    AVERAGED USER WEIGHTS TESTING (EFFICIENT SEQUENTIAL PREDICTION):
    
    - Use averaged weights from all users' best F1 models
    - Apply efficient matrix operations for sequential prediction
    - NO online updates during testing (weights remain fixed)
    """
    print("\n  === AVERAGED USER WEIGHTS TESTING (EFFICIENT SEQUENTIAL PREDICTION) ===")
    
    results = {alg_name: [] for alg_name in user_trained_models.keys()}
    
    for user_idx, (test_user, (X_test_user, y_test_user)) in enumerate(test_user_sequences.items()):
        print(f"    User {user_idx+1}/{len(test_user_sequences)}: {test_user} ({len(X_test_user)} samples)")
        
        for alg_name, model_info in user_trained_models.items():
            try:
                # Use averaged weights
                averaged_weights = model_info['averaged_weights']
                averaged_bias = model_info['averaged_bias']
                n_users_averaged = model_info['n_users_averaged']
                
                # EFFICIENT: Use matrix operations for sequential prediction
                y_pred = efficient_sequential_prediction(X_test_user, averaged_weights, averaged_bias)
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_test_user, y_pred, labels=[-1, 1]).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(y_test_user)
                
                results[alg_name].append({
                    'user': test_user,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'n_users_averaged': n_users_averaged  # Track how many users were averaged
                })
                
            except Exception as e:
                print(f"      ✗ {alg_name} failed on user {test_user}: {e}")
                continue
    
    return results


def aggregate_results(results, dataset_name, training_times, approach_name="BATCH"):
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
        
        # Set approach based on testing type (all are batch approaches)
        if approach_name == "BEST_WEIGHTS":
            approach = "BEST_WEIGHTS"
        elif approach_name == "INDIVIDUAL_USER":
            approach = "INDIVIDUAL_USER"
        elif alg_name == "RCL_BCE":
            approach = "BASELINE"
        else:
            approach = "BATCH"  # All other algorithms use batch training
        
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
