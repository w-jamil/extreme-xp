#!/usr/bin/env python3
"""
OnlineToBatch Experiment Runner with Weighted Majority Voting
Focuses on minimizing False Negatives while maintaining overall performance.
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
import json

warnings.filterwarnings('ignore')

from algorithms import (
    PassiveAggressive, Perceptron, GradientLearning, AROW, RDA, SCW, AdaRDA,
    OnlineToBatchTrainer, WeightedMajorityVoter
)
from data_processor import TaskGenerator
from data_handler import prepare_data_from_zenodo


class OnlineToBatchExperiment:
    """Comprehensive experiment runner for OnlineToBatch protocol"""
    
    def __init__(self, data_dir, results_dir, epochs=5, validation_size=0.2, test_size=0.2):
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.epochs = epochs
        self.validation_size = validation_size
        self.test_size = test_size
        
        # Algorithm configurations optimized for efficiency
        self.algorithm_configs = {
            'PA': (PassiveAggressive, {}),
            'Perceptron': (Perceptron, {}),
            'GL': (GradientLearning, {}),
            'AROW': (AROW, {'r': 1.0}),
            'RDA': (RDA, {'lambda_param': 0.01, 'gamma_param': 1.0}),
            'SCW': (SCW, {'C': 0.1, 'eta': 0.95}),
            'AdaRDA': (AdaRDA, {'lambda_param': 0.01, 'eta_param': 0.1, 'delta_param': 1.0})
        }
        
    def load_and_prepare_data(self):
        """Load and prepare data with user-based splitting"""
        print("Loading and preparing data...")
        
        # Try to load from existing processor
        try:
            processor = TaskGenerator(directory_path=self.data_dir)
            tasks = processor.generate_tasks()
            
            if not tasks:
                raise ValueError("No tasks generated")
                
            # Combine all tasks into single dataset for user-based splitting
            all_X, all_y = [], []
            user_ids = []
            
            for task_name, (X, y) in tasks.items():
                all_X.append(X)
                all_y.extend(y)
                # Create synthetic user IDs based on task and index
                user_ids.extend([f"{task_name}_{i}" for i in range(len(y))])
            
            X_combined = np.vstack(all_X)
            y_combined = np.array(all_y)
            user_ids = np.array(user_ids)
            
        except Exception as e:
            print(f"Error loading tasks: {e}")
            print("Generating synthetic data for testing...")
            return self._generate_synthetic_data()
        
        return self._split_data_by_users(X_combined, y_combined, user_ids)
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        np.random.seed(42)
        n_samples = 5000
        n_features = 20
        n_users = 100
        
        X = np.random.randn(n_samples, n_features)
        # Create some correlation for malicious behavior
        y = ((X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5) > 0).astype(int)
        y = np.where(y == 0, -1, 1)  # Convert to -1, 1 format
        
        user_ids = np.random.choice([f"user_{i}" for i in range(n_users)], n_samples)
        
        return self._split_data_by_users(X, y, user_ids)
    
    def _split_data_by_users(self, X, y, user_ids):
        """Split data ensuring no user overlap between sets"""
        unique_users = np.unique(user_ids)
        np.random.seed(42)
        np.random.shuffle(unique_users)
        
        n_users = len(unique_users)
        n_train = int(n_users * (1 - self.test_size - self.validation_size))
        n_val = int(n_users * self.validation_size)
        
        train_users = set(unique_users[:n_train])
        val_users = set(unique_users[n_train:n_train + n_val])
        test_users = set(unique_users[n_train + n_val:])
        
        # Split data based on user assignment
        train_mask = np.array([uid in train_users for uid in user_ids])
        val_mask = np.array([uid in val_users for uid in user_ids])
        test_mask = np.array([uid in test_users for uid in user_ids])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        data_splits = {
            'X_train': X_scaled[train_mask],
            'y_train': y[train_mask],
            'X_val': X_scaled[val_mask],
            'y_val': y[val_mask],
            'X_test': X_scaled[test_mask],
            'y_test': y[test_mask],
            'test_users': user_ids[test_mask],
            'scaler': scaler
        }
        
        print(f"Data split: Train={len(data_splits['X_train'])}, "
              f"Val={len(data_splits['X_val'])}, Test={len(data_splits['X_test'])}")
        print(f"Users: Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")
        
        return data_splits
    
    def train_individual_algorithms(self, data_splits):
        """Train individual algorithms using OnlineToBatch protocol"""
        print("\nTraining individual algorithms with OnlineToBatch protocol...")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        n_features = X_train.shape[1]
        
        trained_algorithms = []
        individual_results = []
        
        for name, (algo_class, params) in self.algorithm_configs.items():
            print(f"  Training {name}...")
            start_time = time.time()
            
            # Train using OnlineToBatch protocol
            trainer = OnlineToBatchTrainer(
                algorithm_class=algo_class,
                algorithm_params=params,
                n_features=n_features,
                epochs=self.epochs,
                optimize_metric='recall'  # Focus on minimizing FN
            )
            
            trained_algo = trainer.fit(X_train, y_train, X_val, y_val)
            trained_algorithms.append((name, trained_algo))
            
            # Evaluate on test set
            test_metrics = self._evaluate_algorithm(
                trained_algo, data_splits['X_test'], data_splits['y_test']
            )
            test_metrics['algorithm'] = name
            test_metrics['training_time'] = time.time() - start_time
            test_metrics['best_val_score'] = trainer.best_score
            
            individual_results.append(test_metrics)
            print(f"    Completed in {test_metrics['training_time']:.2f}s, "
                  f"Test Recall: {test_metrics['recall']:.3f}")
        
        return trained_algorithms, individual_results
    
    def _evaluate_algorithm(self, algorithm, X_test, y_test):
        """Evaluate algorithm and return comprehensive metrics"""
        y_pred = np.array([algorithm.predict(x) for x in X_test])
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()
        except ValueError:
            # Handle edge cases
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
    
    def create_weighted_majority_ensemble(self, trained_algorithms, data_splits):
        """Create and evaluate weighted majority voting ensemble"""
        print("\nCreating weighted majority ensemble...")
        
        # Create ensemble
        ensemble = WeightedMajorityVoter(trained_algorithms, weight_metric='recall')
        ensemble.fit_weights(data_splits['X_val'], data_splits['y_val'])
        
        print("  Algorithm weights:")
        for name, weight in ensemble.weights.items():
            print(f"    {name}: {weight:.3f}")
        
        # Evaluate ensemble
        ensemble_metrics = self._evaluate_algorithm(
            ensemble, data_splits['X_test'], data_splits['y_test']
        )
        ensemble_metrics['algorithm'] = 'WeightedMajority'
        
        return ensemble, ensemble_metrics
    
    def evaluate_per_user_performance(self, algorithms, data_splits):
        """Evaluate algorithms per user over time"""
        print("\nEvaluating per-user performance...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        test_users = data_splits['test_users']
        
        unique_users = np.unique(test_users)
        user_results = []
        
        for user in unique_users[:min(10, len(unique_users))]:  # Limit for efficiency
            user_mask = test_users == user
            X_user = X_test[user_mask]
            y_user = y_test[user_mask]
            
            if len(X_user) < 2:
                continue
                
            user_metrics = {'user': user, 'n_samples': len(X_user)}
            
            for name, algo in algorithms:
                metrics = self._evaluate_algorithm(algo, X_user, y_user)
                for metric, value in metrics.items():
                    user_metrics[f"{name}_{metric}"] = value
            
            user_results.append(user_metrics)
        
        return pd.DataFrame(user_results)
    
    def save_results(self, individual_results, ensemble_metrics, user_results):
        """Save all results to CSV files"""
        print("\nSaving results...")
        
        # Save individual algorithm results
        individual_df = pd.DataFrame(individual_results)
        individual_path = self.results_dir / "individual_algorithms_results.csv"
        individual_df.to_csv(individual_path, index=False)
        print(f"  Individual results saved to: {individual_path}")
        
        # Create comparison with ensemble
        best_individual = individual_df.loc[individual_df['recall'].idxmax()]
        
        comparison_data = [
            {
                'method': 'Best Individual (by Recall)',
                'algorithm': best_individual['algorithm'],
                'accuracy': best_individual['accuracy'],
                'precision': best_individual['precision'],
                'recall': best_individual['recall'],
                'f1': best_individual['f1'],
                'fnr': best_individual['fnr'],
                'fpr': best_individual['fpr'],
                'fn': best_individual['fn'],
                'fp': best_individual['fp']
            },
            {
                'method': 'Weighted Majority',
                'algorithm': 'Ensemble',
                'accuracy': ensemble_metrics['accuracy'],
                'precision': ensemble_metrics['precision'],
                'recall': ensemble_metrics['recall'],
                'f1': ensemble_metrics['f1'],
                'fnr': ensemble_metrics['fnr'],
                'fpr': ensemble_metrics['fpr'],
                'fn': ensemble_metrics['fn'],
                'fp': ensemble_metrics['fp']
            }
        ]
        
        # Calculate improvements
        fn_improvement = best_individual['fn'] - ensemble_metrics['fn']
        fp_improvement = best_individual['fp'] - ensemble_metrics['fp']
        
        comparison_data.append({
            'method': 'Improvement (Best - Ensemble)',
            'algorithm': 'N/A',
            'accuracy': ensemble_metrics['accuracy'] - best_individual['accuracy'],
            'precision': ensemble_metrics['precision'] - best_individual['precision'],
            'recall': ensemble_metrics['recall'] - best_individual['recall'],
            'f1': ensemble_metrics['f1'] - best_individual['f1'],
            'fnr': best_individual['fnr'] - ensemble_metrics['fnr'],  # Lower is better
            'fpr': best_individual['fpr'] - ensemble_metrics['fpr'],  # Lower is better
            'fn': fn_improvement,
            'fp': fp_improvement
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.results_dir / "ensemble_vs_best_individual.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  Comparison results saved to: {comparison_path}")
        
        # Save per-user results
        if not user_results.empty:
            user_path = self.results_dir / "per_user_results.csv"
            user_results.to_csv(user_path, index=False)
            print(f"  Per-user results saved to: {user_path}")
        
        return individual_path, comparison_path
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("="*60)
        print("OnlineToBatch Experiment with Weighted Majority Voting")
        print("Focus: Minimizing False Negatives")
        print("="*60)
        
        start_time = time.time()
        
        # Load and prepare data
        data_splits = self.load_and_prepare_data()
        
        # Train individual algorithms
        trained_algorithms, individual_results = self.train_individual_algorithms(data_splits)
        
        # Create ensemble
        ensemble, ensemble_metrics = self.create_weighted_majority_ensemble(
            trained_algorithms, data_splits
        )
        
        # Evaluate per-user performance (subset for efficiency)
        user_results = self.evaluate_per_user_performance(
            trained_algorithms + [('WeightedMajority', ensemble)], data_splits
        )
        
        # Save results
        individual_path, comparison_path = self.save_results(
            individual_results, ensemble_metrics, user_results
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        print("\nBest Individual Algorithm (by Recall):")
        best_idx = np.argmax([r['recall'] for r in individual_results])
        best = individual_results[best_idx]
        print(f"  Algorithm: {best['algorithm']}")
        print(f"  Recall: {best['recall']:.3f}")
        print(f"  F1: {best['f1']:.3f}")
        print(f"  FNR: {best['fnr']:.3f}")
        print(f"  FN: {best['fn']}")
        
        print(f"\nWeighted Majority Ensemble:")
        print(f"  Recall: {ensemble_metrics['recall']:.3f}")
        print(f"  F1: {ensemble_metrics['f1']:.3f}")
        print(f"  FNR: {ensemble_metrics['fnr']:.3f}")
        print(f"  FN: {ensemble_metrics['fn']}")
        
        print(f"\nImprovement (Ensemble vs Best Individual):")
        print(f"  FN Reduction: {best['fn'] - ensemble_metrics['fn']}")
        print(f"  FP Change: {ensemble_metrics['fp'] - best['fp']}")
        print(f"  Recall Improvement: {ensemble_metrics['recall'] - best['recall']:.3f}")
        
        total_time = time.time() - start_time
        print(f"\nTotal Experiment Time: {total_time:.2f} seconds")
        print(f"\nResults saved to:")
        print(f"  - {individual_path}")
        print(f"  - {comparison_path}")


def main():
    """Main function to run the experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OnlineToBatch Experiment')
    parser.add_argument('--data-dir', default='../cyber', help='Data directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3 for efficiency)')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = OnlineToBatchExperiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs
    )
    
    experiment.run_experiment()


if __name__ == "__main__":
    main()
