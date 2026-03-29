#!/usr/bin/env python3
"""
Standalone OnlineToBatch experiment with synthetic data
This version will work even without the cyber dataset
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import copy

warnings.filterwarnings('ignore')

# Import our algorithms
import sys
sys.path.append('/home/wjamil/Documents/extreme-xp/master-experiment-suite/cl_case1/src')

from algorithms import (
    PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA,
    OnlineToBatchTrainer, WeightedMajorityVoter
)


def generate_synthetic_cybersecurity_data(n_samples=3000, n_features=25, n_users=50, random_state=42):
    """Generate realistic cybersecurity-like data with user patterns"""
    np.random.seed(random_state)
    
    # Generate users
    users = [f"user_{i:03d}" for i in range(n_users)]
    user_ids = np.random.choice(users, n_samples)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic patterns for malicious behavior
    # Features represent: network activity, file access patterns, time patterns, etc.
    
    # Network activity features (0-4)
    X[:, 0] = np.random.exponential(1, n_samples)  # Connection frequency
    X[:, 1] = np.random.gamma(2, 2, n_samples)     # Data transfer volume
    X[:, 2] = np.random.lognormal(0, 1, n_samples)  # Session duration
    X[:, 3] = np.random.poisson(3, n_samples)      # Unique destinations
    X[:, 4] = np.random.beta(2, 5, n_samples)      # Off-hours activity
    
    # File access features (5-9)
    X[:, 5] = np.random.exponential(0.5, n_samples)  # File access frequency
    X[:, 6] = np.random.gamma(1.5, 1, n_samples)     # Sensitive file access
    X[:, 7] = np.random.lognormal(-1, 0.5, n_samples)  # File modification rate
    X[:, 8] = np.random.poisson(2, n_samples)        # Admin privilege usage
    X[:, 9] = np.random.beta(3, 7, n_samples)        # Unusual file types
    
    # Behavioral features (10-14)
    X[:, 10] = np.random.exponential(2, n_samples)   # Login frequency
    X[:, 11] = np.random.gamma(1, 1.5, n_samples)    # Failed login attempts
    X[:, 12] = np.random.lognormal(0.5, 0.8, n_samples)  # Geographic anomaly
    X[:, 13] = np.random.poisson(1, n_samples)       # Device changes
    X[:, 14] = np.random.beta(1, 9, n_samples)       # Weekend activity
    
    # Additional technical features (15-24)
    for i in range(15, n_features):
        X[:, i] = np.random.randn(n_samples) * 0.5 + np.random.choice([-1, 0, 1], n_samples) * 2
    
    # Create realistic malicious labels (approximately 15% positive)
    # Malicious behavior correlates with multiple suspicious features
    suspicious_score = (
        2.0 * X[:, 0] +    # High connection frequency
        1.5 * X[:, 1] +    # High data transfer
        1.0 * X[:, 4] +    # Off-hours activity
        2.0 * X[:, 6] +    # Sensitive file access
        1.5 * X[:, 8] +    # Admin privilege usage
        1.0 * X[:, 11] +   # Failed logins
        1.0 * X[:, 12] +   # Geographic anomaly
        np.random.randn(n_samples) * 0.5  # Add noise
    )
    
    # Convert to binary labels with threshold to get ~15% malicious
    threshold = np.percentile(suspicious_score, 85)
    y = (suspicious_score > threshold).astype(int)
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1 format
    
    print(f"Generated {n_samples} samples with {np.sum(y == 1)} malicious ({np.mean(y == 1)*100:.1f}%)")
    
    return X, y, user_ids


def split_data_by_users(X, y, user_ids, val_size=0.2, test_size=0.2, random_state=42):
    """Split data ensuring no user overlap between train/val/test"""
    np.random.seed(random_state)
    
    unique_users = np.unique(user_ids)
    np.random.shuffle(unique_users)
    
    n_users = len(unique_users)
    n_train = int(n_users * (1 - val_size - test_size))
    n_val = int(n_users * val_size)
    
    train_users = set(unique_users[:n_train])
    val_users = set(unique_users[n_train:n_train + n_val])
    test_users = set(unique_users[n_train + n_val:])
    
    # Create masks
    train_mask = np.array([uid in train_users for uid in user_ids])
    val_mask = np.array([uid in val_users for uid in user_ids])
    test_mask = np.array([uid in test_users for uid in user_ids])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return {
        'X_train': X_scaled[train_mask],
        'y_train': y[train_mask],
        'X_val': X_scaled[val_mask],
        'y_val': y[val_mask],
        'X_test': X_scaled[test_mask],
        'y_test': y[test_mask],
        'test_users': user_ids[test_mask],
        'train_users_count': len(train_users),
        'val_users_count': len(val_users),
        'test_users_count': len(test_users)
    }


def evaluate_algorithm(algorithm, X_test, y_test):
    """Comprehensive algorithm evaluation"""
    y_pred = np.array([algorithm.predict(x) for x in X_test])
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()
    except ValueError:
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'fnr': 1.0, 'fpr': 1.0, 'tn': 0, 'fp': 0, 'fn': len(y_test), 'tp': 0
        }
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fnr': fnr,
        'fpr': fpr,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def main():
    print("="*70)
    print("OnlineToBatch Cybersecurity Experiment")
    print("Focus: Minimizing False Negatives with Weighted Majority Voting")
    print("="*70)
    
    start_time = time.time()
    
    # Generate synthetic cybersecurity data
    print("\n1. Generating synthetic cybersecurity data...")
    X, y, user_ids = generate_synthetic_cybersecurity_data(
        n_samples=4000, n_features=25, n_users=60
    )
    
    # Split data by users
    print("\n2. Splitting data by users (no user overlap)...")
    data = split_data_by_users(X, y, user_ids)
    print(f"   Train: {len(data['X_train'])} samples ({data['train_users_count']} users)")
    print(f"   Val:   {len(data['X_val'])} samples ({data['val_users_count']} users)")
    print(f"   Test:  {len(data['X_test'])} samples ({data['test_users_count']} users)")
    
    # Algorithm configurations (optimized for speed)
    algorithm_configs = {
        'PA': (PassiveAggressive, {}),
        'Perceptron': (Perceptron, {}),
        'GradLearning': (GradientLearning, {}),
        'AROW': (AROW, {'r': 1.0}),
        'RDA': (RDA, {'lambda_param': 0.01, 'gamma_param': 1.0}),
        'SCW': (SCW, {'C': 0.1, 'eta': 0.95}),
        'AdaRDA': (AdaRDA, {'lambda_param': 0.01, 'eta_param': 0.1, 'delta_param': 1.0})
    }
    
    # Train individual algorithms
    print(f"\n3. Training {len(algorithm_configs)} algorithms with OnlineToBatch protocol...")
    n_features = data['X_train'].shape[1]
    epochs = 4  # Efficient number of epochs
    
    trained_algorithms = []
    individual_results = []
    
    for name, (algo_class, params) in algorithm_configs.items():
        print(f"   Training {name}...", end=" ")
        algo_start = time.time()
        
        # Train using OnlineToBatch protocol
        trainer = OnlineToBatchTrainer(
            algorithm_class=algo_class,
            algorithm_params=params,
            n_features=n_features,
            epochs=epochs,
            optimize_metric='recall'  # Minimize FN
        )
        
        trained_algo = trainer.fit(
            data['X_train'], data['y_train'], 
            data['X_val'], data['y_val']
        )
        
        trained_algorithms.append((name, trained_algo))
        
        # Evaluate on test set
        metrics = evaluate_algorithm(trained_algo, data['X_test'], data['y_test'])
        metrics['algorithm'] = name
        metrics['training_time'] = time.time() - algo_start
        metrics['best_val_recall'] = trainer.best_score
        
        individual_results.append(metrics)
        print(f"Done ({metrics['training_time']:.2f}s) - Recall: {metrics['recall']:.3f}, FN: {metrics['fn']}")
    
    # Create weighted majority ensemble
    print(f"\n4. Creating weighted majority ensemble...")
    ensemble = WeightedMajorityVoter(trained_algorithms, weight_metric='recall')
    ensemble.fit_weights(data['X_val'], data['y_val'])
    
    print("   Algorithm weights based on validation recall:")
    for name, weight in ensemble.weights.items():
        print(f"     {name}: {weight:.3f}")
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_algorithm(ensemble, data['X_test'], data['y_test'])
    
    # Find best individual algorithm by recall
    best_individual = max(individual_results, key=lambda x: x['recall'])
    
    print(f"\n5. Results Summary:")
    print("="*70)
    
    # Individual results table
    print(f"\nIndividual Algorithm Results:")
    print(f"{'Algorithm':<12} {'Recall':<7} {'F1':<7} {'FNR':<7} {'FN':<4} {'FP':<4} {'Accuracy':<8}")
    print("-" * 70)
    for result in sorted(individual_results, key=lambda x: x['recall'], reverse=True):
        print(f"{result['algorithm']:<12} {result['recall']:<7.3f} {result['f1']:<7.3f} "
              f"{result['fnr']:<7.3f} {result['fn']:<4d} {result['fp']:<4d} {result['accuracy']:<8.3f}")
    
    # Comparison with ensemble
    print(f"\nBest Individual vs Weighted Majority Ensemble:")
    print("-" * 70)
    print(f"{'Method':<20} {'Recall':<7} {'F1':<7} {'FNR':<7} {'FN':<4} {'FP':<4} {'Accuracy':<8}")
    print(f"{'Best Individual':<20} {best_individual['recall']:<7.3f} {best_individual['f1']:<7.3f} "
          f"{best_individual['fnr']:<7.3f} {best_individual['fn']:<4d} {best_individual['fp']:<4d} "
          f"{best_individual['accuracy']:<8.3f}")
    print(f"{'Weighted Ensemble':<20} {ensemble_metrics['recall']:<7.3f} {ensemble_metrics['f1']:<7.3f} "
          f"{ensemble_metrics['fnr']:<7.3f} {ensemble_metrics['fn']:<4d} {ensemble_metrics['fp']:<4d} "
          f"{ensemble_metrics['accuracy']:<8.3f}")
    
    # Calculate improvements
    fn_improvement = best_individual['fn'] - ensemble_metrics['fn']
    fp_change = ensemble_metrics['fp'] - best_individual['fp']
    recall_improvement = ensemble_metrics['recall'] - best_individual['recall']
    
    print(f"\nImprovement Analysis:")
    print(f"  False Negative Reduction: {fn_improvement} (fewer missed threats)")
    print(f"  False Positive Change: {fp_change:+d}")
    print(f"  Recall Improvement: {recall_improvement:+.3f}")
    
    if fn_improvement > 0:
        print(f"  ✓ SUCCESS: Ensemble reduced false negatives by {fn_improvement}")
    else:
        print(f"   No FN improvement achieved")
    
    # Save results to CSV
    print(f"\n6. Saving results...")
    results_dir = Path('/home/wjamil/Documents/extreme-xp/master-experiment-suite/cl_case1/src/results')
    results_dir.mkdir(exist_ok=True)
    
    # Individual results
    individual_df = pd.DataFrame(individual_results)
    individual_path = results_dir / "individual_algorithms_onlinetobatch.csv"
    individual_df.to_csv(individual_path, index=False)
    
    # Comparison results
    comparison_data = [
        {
            'method': f'Best Individual ({best_individual["algorithm"]})',
            **{k: v for k, v in best_individual.items() if k != 'algorithm'}
        },
        {
            'method': 'Weighted Majority Ensemble',
            'algorithm': 'Ensemble',
            **ensemble_metrics
        },
        {
            'method': 'Improvement (Ensemble - Best)',
            'algorithm': 'N/A',
            'accuracy': ensemble_metrics['accuracy'] - best_individual['accuracy'],
            'precision': ensemble_metrics['precision'] - best_individual['precision'],
            'recall': recall_improvement,
            'f1': ensemble_metrics['f1'] - best_individual['f1'],
            'fnr': best_individual['fnr'] - ensemble_metrics['fnr'],  # Lower FNR is better
            'fpr': ensemble_metrics['fpr'] - best_individual['fpr'],
            'fn': -fn_improvement,  # Negative means reduction
            'fp': fp_change,
            'tn': ensemble_metrics['tn'] - best_individual['tn'],
            'tp': ensemble_metrics['tp'] - best_individual['tp']
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = results_dir / "ensemble_vs_best_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    total_time = time.time() - start_time
    
    print(f"\nFiles saved:")
    print(f"  - Individual results: {individual_path}")
    print(f"  - Comparison results: {comparison_path}")
    print(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    print("="*70)
    print("Experiment completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
