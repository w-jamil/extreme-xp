#!/usr/bin/env python3
"""
MNIST Binary Classification - Online to Batch Learning Experiment
Binary classification of MNIST digits with online learning algorithms
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time
from datetime import datetime
import pickle

# Algorithm functions will be defined below
from scipy.stats import norm

# =============================================================================
# ONLINE LEARNING ALGORITHMS
# =============================================================================

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

def AROW(X, y, r, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Online learning implementation of AROW with optional multi-epoch training and early stopping."""
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

def download_mnist_data():
    """Download MNIST dataset from sklearn/openml"""
    print("Downloading MNIST dataset...")
    try:
        # Download MNIST from OpenML (via sklearn)
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        print(f"✓ MNIST downloaded successfully")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    except Exception as e:
        print(f"✗ Error downloading MNIST: {e}")
        return None, None

def create_binary_classification_task(X, y, task_type="even_odd"):
    """
    Convert MNIST to binary classification task
    
    Args:
        X: Features
        y: Labels (0-9)
        task_type: "even_odd", "low_high", or "zero_nonzero"
    """
    print(f"Creating binary classification task: {task_type}")
    
    if task_type == "even_odd":
        # Even digits (0,2,4,6,8) vs Odd digits (1,3,5,7,9)
        y_binary = (y % 2).astype(int)  # 0=even, 1=odd
        task_name = "Even_vs_Odd"
        print("  Binary task: Even digits (0) vs Odd digits (1)")
        
    elif task_type == "low_high":
        # Low digits (0,1,2,3,4) vs High digits (5,6,7,8,9)
        y_binary = (y >= 5).astype(int)  # 0=low, 1=high
        task_name = "Low_vs_High"
        print("  Binary task: Low digits 0-4 (0) vs High digits 5-9 (1)")
        
    elif task_type == "zero_nonzero":
        # Zero vs Non-zero
        y_binary = (y != 0).astype(int)  # 0=zero, 1=non-zero
        task_name = "Zero_vs_NonZero"
        print("  Binary task: Zero (0) vs Non-zero (1)")
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Convert to -1/+1 format for online algorithms
    y_online = np.where(y_binary == 0, -1, 1)
    
    print(f"  Class distribution: {np.bincount(y_binary)}")
    print(f"  Class ratio: {np.mean(y_binary):.4f}")
    
    return X, y_binary, y_online, task_name

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive binary classification metrics"""
    # Convert predictions to 0/1 format for metrics
    y_true_01 = np.where(y_true == -1, 0, y_true) if np.any(y_true == -1) else y_true
    y_pred_01 = np.where(y_pred == -1, 0, y_pred) if np.any(y_pred == -1) else y_pred
    
    tn, fp, fn, tp = confusion_matrix(y_true_01, y_pred_01, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true_01, y_pred_01)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def online_to_batch_experiment(X_train, y_train_online, X_val, y_val_online, X_test, y_test_online, 
                               algorithm_name, algorithm_func, max_shuffles=10):
    """
    Perform online-to-batch experiment with validation-based early stopping
    
    Args:
        X_train, y_train_online: Training data
        X_val, y_val_online: Validation data  
        X_test, y_test_online: Test data
        algorithm_name: Name of the algorithm
        algorithm_func: Algorithm function
        max_shuffles: Maximum number of shuffles (epochs) to try
    
    Returns:
        dict: Results including best shuffle and test performance
    """
    print(f"  Running online-to-batch experiment for {algorithm_name}...")
    
    best_f1_val = -1
    best_shuffle = -1
    best_weights = None
    
    shuffle_results = []
    training_times = []
    
    total_start_time = time.time()
    
    for shuffle_idx in range(max_shuffles):
        print(f"    Epoch {shuffle_idx + 1}/{max_shuffles}")
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_online[indices]
        
        try:
            # Train algorithm on shuffled data with timing
            epoch_start_time = time.time()
            y_pred_train, weights_history = algorithm_func(X_train_shuffled, y_train_shuffled)
            epoch_time = time.time() - epoch_start_time
            training_times.append(epoch_time)
            
            # Get final weights
            final_weights = weights_history[-1] if len(weights_history) > 0 else np.zeros(X_train.shape[1])
            
            # Validate on validation set
            val_scores = X_val.dot(final_weights)
            y_pred_val = np.sign(val_scores)
            y_pred_val = np.where(y_pred_val == 0, 1, y_pred_val)
            
            # Calculate validation F1
            val_metrics = calculate_metrics(y_val_online, y_pred_val)
            val_f1 = val_metrics['f1']
            
            shuffle_results.append({
                'epoch': shuffle_idx + 1,
                'val_f1': val_f1,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'epoch_time': epoch_time
            })
            
            print(f"      Validation F1: {val_f1:.4f}, Time: {epoch_time:.2f}s")
            
            # Check if this is the best shuffle so far
            if val_f1 > best_f1_val:
                best_f1_val = val_f1
                best_shuffle = shuffle_idx + 1
                best_weights = final_weights.copy()
                print(f"      ✓ New best validation F1: {val_f1:.4f}")
                
        except Exception as e:
            print(f"      ✗ Error in epoch {shuffle_idx + 1}: {e}")
            training_times.append(0.0)  # Record 0 time for failed epochs
            continue
    
    total_training_time = time.time() - total_start_time
    avg_epoch_time = np.mean(training_times) if training_times else 0.0
    
    # Test with best weights
    if best_weights is not None:
        test_scores = X_test.dot(best_weights)
        y_pred_test = np.sign(test_scores)
        y_pred_test = np.where(y_pred_test == 0, 1, y_pred_test)
        
        test_metrics = calculate_metrics(y_test_online, y_pred_test)
        
        print(f"    ✓ Best epoch: {best_shuffle} (Val F1: {best_f1_val:.4f})")
        print(f"    ✓ Test performance: F1={test_metrics['f1']:.4f}, "
              f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")
        print(f"    ✓ Training time: Total={total_training_time:.2f}s, Avg/epoch={avg_epoch_time:.2f}s")
        
        return {
            'algorithm': algorithm_name,
            'best_epoch': best_shuffle,
            'best_val_f1': best_f1_val,
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_accuracy': test_metrics['accuracy'],
            'test_fpr': test_metrics['fpr'],
            'test_fnr': test_metrics['fnr'],
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'num_epochs': max_shuffles,
            'epoch_results': shuffle_results
        }
    else:
        print(f"    ✗ No successful training for {algorithm_name}")
        return {
            'algorithm': algorithm_name,
            'best_epoch': 0,
            'best_val_f1': 0.0,
            'test_f1': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_accuracy': 0.0,
            'test_fpr': 0.0,
            'test_fnr': 0.0,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'num_epochs': max_shuffles,
            'epoch_results': []
        }

def run_mnist_binary_experiment(task_type="even_odd", max_shuffles=10, sample_size=None):
    """
    Run complete MNIST binary classification experiment
    
    Args:
        task_type: Type of binary classification task
        max_shuffles: Maximum number of shuffles for validation
        sample_size: Limit dataset size for faster experiments (None = use all data)
    """
    print("="*80)
    print("MNIST BINARY CLASSIFICATION - ONLINE TO BATCH EXPERIMENT")
    print("="*80)
    
    # Download MNIST data
    X, y = download_mnist_data()
    if X is None:
        return
    
    # Limit dataset size if specified
    if sample_size and sample_size < len(X):
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"  Using subset of {sample_size} samples")
    
    # Create binary classification task
    X, y_binary, y_online, task_name = create_binary_classification_task(X, y, task_type)
    
    # Normalize features
    print("  Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: 60% train, 20% validation, 20% test
    print("  Splitting data: 60% train, 20% validation, 20% test")
    X_train, X_temp, y_train, y_temp, y_binary_train, y_binary_temp = train_test_split(
        X_scaled, y_online, y_binary, test_size=0.4, random_state=42, stratify=y_binary
    )
    X_val, X_test, y_val, y_test, y_binary_val, y_binary_test = train_test_split(
        X_temp, y_temp, y_binary_temp, test_size=0.5, random_state=42, stratify=y_binary_temp
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Define algorithms (matching online_kaggle_fraud.py)
    algorithms = {
        'PA': lambda X, y: AP(X, y),
        'Perceptron': lambda X, y: PERCEPT(X, y),
        'GLC': lambda X, y: OGL(X, y),
        'AROW': lambda X, y: AROW(X, y, r=0.1),
        'RDA': lambda X, y: RDA(X, y, lambda_param=1, gamma_param=1),
        'SCW': lambda X, y: SCW(X, y, C=1, eta=0.5),
        'AdaRDA': lambda X, y: AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1),
    }
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"RUNNING ONLINE-TO-BATCH EXPERIMENTS")
    print(f"Task: {task_name}, Max Shuffles: {max_shuffles}")
    print('='*60)
    
    for alg_name, alg_func in algorithms.items():
        print(f"\n--- {alg_name} ---")
        start_time = time.time()
        
        result = online_to_batch_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            alg_name, alg_func, max_shuffles
        )
        
        if result and result['test_f1'] > 0:  # Only add if successful
            results.append(result)
        
        print(f"  Total training time: {time.time() - start_time:.2f}s")
    
    # Save and display results
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS - MNIST BINARY CLASSIFICATION")
        print('='*80)
        
        print(f"\n{'Algorithm':<12} {'Best Epoch':<11} {'Val F1':<8} {'Test F1':<8} {'Test Prec':<10} {'Test Rec':<9} {'Tot Time(s)':<11} {'Avg/Epoch(s)':<12}")
        print(f"{'-'*12} {'-'*11} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*11} {'-'*12}")
        
        for _, row in results_df.iterrows():
            print(f"{row['algorithm']:<12} "
                  f"{row['best_epoch']:<11} "
                  f"{row['best_val_f1']:<8.3f} "
                  f"{row['test_f1']:<8.3f} "
                  f"{row['test_precision']:<10.3f} "
                  f"{row['test_recall']:<9.3f} "
                  f"{row['total_training_time']:<11.2f} "
                  f"{row['avg_epoch_time']:<12.2f}")
        
        # Save results
        os.makedirs('../results', exist_ok=True)
        results_file = f'../results/mnist_binary_{task_type}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved: {results_file}")
        
        # Performance summary
        best_alg = results_df.loc[results_df['test_f1'].idxmax()]
        print(f"\nBEST ALGORITHM:")
        print(f"- Algorithm: {best_alg['algorithm']}")
        print(f"- Best Epoch: {best_alg['best_epoch']}")
        print(f"- Test F1: {best_alg['test_f1']:.4f}")
        print(f"- Test Precision: {best_alg['test_precision']:.4f}")
        print(f"- Test Recall: {best_alg['test_recall']:.4f}")
        print(f"- Total Training Time: {best_alg['total_training_time']:.2f}s")
        print(f"- Average Time per Epoch: {best_alg['avg_epoch_time']:.2f}s")
        
    else:
        print("No successful algorithm runs")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print('='*80)
    
    return results

if __name__ == "__main__":
    # Run experiment with different binary tasks
    
    # Configuration
    TASK_TYPE = "even_odd"  # Options: "even_odd", "low_high", "zero_nonzero"
    MAX_SHUFFLES = 10  # Number of shuffles for validation
    SAMPLE_SIZE = 5000  # Use subset for faster testing (None = full dataset)
    
    print(f"Configuration:")
    print(f"- Task: {TASK_TYPE}")
    print(f"- Max shuffles: {MAX_SHUFFLES}")
    print(f"- Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
    
    # Run experiment
    results = run_mnist_binary_experiment(
        task_type=TASK_TYPE,
        max_shuffles=MAX_SHUFFLES, 
        sample_size=SAMPLE_SIZE
    )
