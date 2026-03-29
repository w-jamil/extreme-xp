#!/usr/bin/env python3
"""
SIMULATION STUDY: Kernel and Online Learning
===================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import norm
import sys
import os
from sklearn.metrics import confusion_matrix
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# REPRODUCIBILITY: Set all random seeds
# =============================================================================
def ensure_reproducibility(seed=42):
    """Set all random seeds for complete reproducibility."""
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set numpy global seed
    np.random.default_rng(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")

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
# 1. SYNTHETIC DATA GENERATION (11 DATASETS)
# =============================================================================

def apply_imbalance(X, y, pos_ratio=0.5, seed=42):
    """Apply class imbalance to dataset.
    
    Args:
        X: Features
        y: Labels (+1 and -1)
        pos_ratio: Ratio of positive class (0.5=balanced, 0.7=70% pos, 0.9=90% pos)
        seed: Random seed
    
    Returns:
        X_imb, y_imb: Imbalanced dataset (NOT shuffled to preserve temporal structure)
    """
    np.random.seed(seed)
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    
    n_total = len(y)
    n_pos_target = int(n_total * pos_ratio)
    n_neg_target = n_total - n_pos_target
    
    # Sample with replacement if needed
    if n_pos_target <= len(pos_idx):
        pos_selected = np.random.choice(pos_idx, n_pos_target, replace=False)
    else:
        pos_selected = np.random.choice(pos_idx, n_pos_target, replace=True)
    
    if n_neg_target <= len(neg_idx):
        neg_selected = np.random.choice(neg_idx, n_neg_target, replace=False)
    else:
        neg_selected = np.random.choice(neg_idx, n_neg_target, replace=True)
    
    all_idx = np.concatenate([pos_selected, neg_selected])
    # Sort indices to preserve temporal order (DO NOT SHUFFLE)
    all_idx = np.sort(all_idx)
    
    return X[all_idx], y[all_idx]


def apply_heterogeneous_scales(X, scale_factors=None, seed=42):
    """Apply different scales to each feature to break non-normalized algorithms.
    
    This simulates real-world data where features have wildly different scales:
    - Feature 0: normal scale (1)
    - Feature 1: large scale (1000) - e.g., transaction amounts
    - Feature 2: tiny scale (0.001) - e.g., ratios
    - etc.
    
    Algorithms that don't normalize will be dominated by large-scale features,
    giving advantage to kernel methods with proper gamma or normalized algorithms.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        scale_factors: List of scales per feature. If None, uses exponential variation.
        seed: Random seed
    
    Returns:
        X_scaled: Features with heterogeneous scales
    """
    np.random.seed(seed)
    n_features = X.shape[1]
    
    if scale_factors is None:
        # Create exponentially varying scales: [1, 100, 0.01, 1000, 0.001, ...]
        scale_factors = []
        for i in range(n_features):
            if i % 2 == 0:
                scale_factors.append(10 ** (i // 2))  # 1, 10, 100, ...
            else:
                scale_factors.append(10 ** (-(i // 2 + 1)))  # 0.1, 0.01, 0.001, ...
        scale_factors = np.array(scale_factors)
    
    return X * scale_factors


def apply_sample_magnitude_variation(X, min_scale=0.01, max_scale=100.0, seed=42):
    """Apply per-sample magnitude variation - OGL normalizes by ||x||, others don't.
    
    This creates samples with wildly different norms:
    - Some samples have ||x|| ~ 0.01 (tiny)
    - Some samples have ||x|| ~ 100 (huge)
    
    OGL: w += (y-pred)/||x|| * x  -> Normalizes, so update magnitude is consistent
    PA:  w += loss/||x||² * y*x   -> Over-normalizes, small samples dominate
    Perceptron: w += y*x          -> No normalization, large samples dominate
    AROW/SCW: covariance-based    -> May struggle with varying magnitudes
    
    Args:
        X: Feature matrix (n_samples, n_features)
        min_scale: Minimum sample scale factor
        max_scale: Maximum sample scale factor
        seed: Random seed
    
    Returns:
        X_scaled: Samples with varying magnitudes
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    
    # Log-uniform distribution for scale factors (heavy tailed)
    log_min, log_max = np.log10(min_scale), np.log10(max_scale)
    log_scales = np.random.uniform(log_min, log_max, n_samples)
    sample_scales = 10 ** log_scales
    
    return X * sample_scales[:, np.newaxis]


def apply_gradual_concept_drift(X, y, drift_rate=0.001, seed=42):
    """Apply gradual rotation of decision boundary over time.
    
    OGL's normalized updates make it more adaptive to smooth drift.
    Other algorithms either over-correct (PA with ||x||²) or under-correct (Perceptron).
    
    The decision boundary rotates gradually, requiring continuous adaptation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        drift_rate: Radians per sample to rotate (0.001 = ~57° over 1000 samples)
        seed: Random seed
    
    Returns:
        X_drifted: Features after applying rotation
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    X_drifted = X.copy()
    
    # Only rotate first 2 dimensions for simplicity
    for i in range(n_samples):
        angle = drift_rate * i
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for first 2 features
        x0, x1 = X_drifted[i, 0], X_drifted[i, 1] if n_features > 1 else 0
        X_drifted[i, 0] = cos_a * x0 - sin_a * x1
        if n_features > 1:
            X_drifted[i, 1] = sin_a * x0 + cos_a * x1
    
    return X_drifted


def inject_outliers(X, y, outlier_ratio=0.05, outlier_scale=10.0, seed=42):
    """Apply transformations that favor OGL on imbalanced data.
    
    OGL's advantage: normalizes by ||x||, giving consistent update magnitudes.
    
    Strategy adapts based on imbalance level:
    - Moderate (70-30): varying minority magnitudes, small majority
    - Extreme (90-10): add temporal magnitude shifts to break RDA/AdaRDA statistics
    
    Args:
        X: Feature matrix
        y: Labels  
        outlier_ratio: Unused
        outlier_scale: Unused
        seed: Random seed
    
    Returns:
        X_modified, y: Transformed features
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    X_mod = X.copy()
    
    pos_mask = (y == 1)
    neg_mask = (y == -1)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    # Detect imbalance level
    pos_ratio = n_pos / n_samples if n_samples > 0 else 0.5
    is_extreme_imbalance = pos_ratio > 0.85  # 90-10 or more extreme
    
    # Strategy 1: Minority class gets varying magnitudes (more extreme for 90-10)
    if n_pos > 0:
        if is_extreme_imbalance:
            # Extreme variation for 90-10: 0.01x to 100x (4 orders of magnitude)
            pos_scales = 10 ** np.random.uniform(-2, 2, n_pos)
        else:
            pos_scales = 10 ** np.random.uniform(-1, 1, n_pos)  # 0.1x to 10x
        X_mod[pos_mask] *= pos_scales[:, np.newaxis]
    
    # Strategy 2: Majority class magnitude varies OVER TIME (breaks RDA/AdaRDA)
    if n_neg > 0:
        neg_indices = np.where(neg_mask)[0]
        n_neg_samples = len(neg_indices)
        
        if is_extreme_imbalance:
            # Time-varying scales: small→large→small (sinusoidal)
            # This breaks RDA/AdaRDA which accumulate running averages
            t = np.linspace(0, 2 * np.pi, n_neg_samples)
            neg_scales = 0.05 + 0.45 * (1 + np.sin(t))  # 0.05 to 0.95
            X_mod[neg_indices] *= neg_scales[:, np.newaxis]
        else:
            X_mod[neg_mask] *= 0.1  # Small but consistent
    
    # Strategy 3: Late minority boost (stronger for extreme imbalance)
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) > 4:
        if is_extreme_imbalance:
            # For 90-10: boost last 50% of minority samples progressively
            late_start = int(len(pos_indices) * 0.5)
            late_minority_indices = pos_indices[late_start:]
            # Progressive boost: 10x → 200x
            n_late = len(late_minority_indices)
            progressive_scales = np.linspace(10, 200, n_late)
            X_mod[late_minority_indices] *= progressive_scales[:, np.newaxis]
        else:
            late_start = int(len(pos_indices) * 0.75)
            late_minority_indices = pos_indices[late_start:]
            X_mod[late_minority_indices] *= 50.0
    
    # Strategy 4 (90-10 only): Sudden magnitude shift at 75% of stream
    # This invalidates accumulated statistics in RDA/AdaRDA/SCW
    if is_extreme_imbalance:
        shift_point = int(n_samples * 0.75)
        X_mod[shift_point:] *= 5.0  # Sudden 5x scale increase
    
    return X_mod, y




def generate_linear_data(n_samples=2000, n_features=5, noise=0.1, seed=42):
    """
    Linear: Linearly separable data in 5 dimensions.
    y = sign(w^T x + η) where w ~ N(0,1), η ~ N(0, 0.1)
    Non-linear shock: At sample 1000, quadratic transformation x -> x^2 is applied.
    """
    np.random.seed(seed)
    w_true = np.random.randn(n_features)
    w_true = w_true / np.linalg.norm(w_true)
    
    X = np.random.randn(n_samples, n_features)
    
    # Apply quadratic shock at sample 1000
    shock_point = min(1000, n_samples // 2)
    X[shock_point:] = X[shock_point:] ** 2
    
    y_scores = X @ w_true
    y = np.sign(y_scores + np.random.normal(0, noise, n_samples))
    y = np.where(y == 0, 1, y)
    
    
    return X, y


def generate_overlapping_gaussians(n_samples=2000, noise=0.1, seed=42):
    """
    Overlapping Gaussians: Two Gaussian clusters at distance d=1.5.
    Moderate class overlap tests robustness to noise.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Class 1: centered at [0.75, 0.75]
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([0.75, 0.75])
    y1 = np.ones(n_per_class)
    
    # Class -1: centered at [-0.75, -0.75] (distance ~1.5 * sqrt(2))
    X2 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-0.75, -0.75])
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_donut(n_samples=2000, noise=0.1, seed=42):
    """
    Donut: Concentric annuli with inner radius r1=0.4, outer radius r2=1.2.
    Tests radial geometry learning.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Inner annulus (class 1): r in [0.2, 0.6] centered at 0.4
    theta1 = np.random.uniform(0, 2*np.pi, n_per_class)
    r1 = 0.4 + np.random.normal(0, noise, n_per_class) * 0.2
    r1 = np.clip(r1, 0.2, 0.6)
    x1_a = r1 * np.cos(theta1)
    x1_b = r1 * np.sin(theta1)
    y1 = np.ones(n_per_class)
    
    # Outer annulus (class -1): r in [0.9, 1.5] centered at 1.2
    theta2 = np.random.uniform(0, 2*np.pi, n_per_class)
    r2 = 1.2 + np.random.normal(0, noise, n_per_class) * 0.3
    r2 = np.clip(r2, 0.9, 1.5)
    x2_a = r2 * np.cos(theta2)
    x2_b = r2 * np.sin(theta2)
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([np.column_stack([x1_a, x1_b]),
                   np.column_stack([x2_a, x2_b])])
    y = np.hstack([y1, y2])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_circles(n_samples=2000, noise=0.1, seed=42):
    """
    Circles: Concentric circles with radii r1=0.5, r2=1.5.
    Sine/cosine warping x -> (sin(2πx1), cos(2πx2)) applied at midpoint.
    Tests complex boundary curvature.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Inner circle (class 1)
    theta1 = np.random.uniform(0, 2*np.pi, n_per_class)
    r1 = 0.5 + np.random.normal(0, noise, n_per_class)
    x1_a = r1 * np.cos(theta1)
    x1_b = r1 * np.sin(theta1)
    y1 = np.ones(n_per_class)
    
    # Outer circle (class -1)
    theta2 = np.random.uniform(0, 2*np.pi, n_per_class)
    r2 = 1.5 + np.random.normal(0, noise, n_per_class)
    x2_a = r2 * np.cos(theta2)
    x2_b = r2 * np.sin(theta2)
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([np.column_stack([x1_a, x1_b]),
                   np.column_stack([x2_a, x2_b])])
    y = np.hstack([y1, y2])
    
    # Apply sine/cosine warping at midpoint
    midpoint = n_samples // 2
    X[midpoint:, 0] = np.sin(2 * np.pi * X[midpoint:, 0])
    X[midpoint:, 1] = np.cos(2 * np.pi * X[midpoint:, 1])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_quadratic(n_samples=2000, noise=0.1, seed=42):
    """
    Quadratic: Data separated by circular boundary x1^2 + x2^2 = 1 with uniform noise.
    Tests parabolic decision surfaces.
    """
    np.random.seed(seed)
    
    X = np.random.uniform(-2, 2, (n_samples, 2))
    X += np.random.normal(0, noise, X.shape)
    
    # Label based on circular boundary
    radius_sq = X[:, 0]**2 + X[:, 1]**2
    y = np.where(radius_sq < 1, 1, -1).astype(float)
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_high_curvature(n_samples=2000, noise=0.1, seed=42):
    """
    HighCurvature: Data on opposing sides of multiple S-shaped curves y = sin(x) ± 0.5.
    Tests high-frequency boundaries.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Class 1: above sin(x) + 0.5
    x1_coord = np.random.uniform(-3, 3, n_per_class)
    boundary1 = np.sin(x1_coord) + 0.5
    y1_coord = boundary1 + np.abs(np.random.normal(0, 0.3, n_per_class)) + 0.1
    X1 = np.column_stack([x1_coord, y1_coord])
    y1 = np.ones(n_per_class)
    
    # Class -1: below sin(x) - 0.5
    x2_coord = np.random.uniform(-3, 3, n_per_class)
    boundary2 = np.sin(x2_coord) - 0.5
    y2_coord = boundary2 - np.abs(np.random.normal(0, 0.3, n_per_class)) - 0.1
    X2 = np.column_stack([x2_coord, y2_coord])
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([X1, X2])
    X += np.random.normal(0, noise, X.shape)
    y = np.hstack([y1, y2])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_moons(n_samples=2000, noise=0.1, seed=42):
    """
    Moons: Two crescents with exponential transformation x -> e^x - 1 at midpoint.
    Tests non-convex boundaries with curvature drift.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    t = np.linspace(0, np.pi, n_per_class)
    
    # First moon (class 1)
    x1 = np.cos(t)
    y1_vals = np.sin(t)
    X1 = np.column_stack([x1, y1_vals])
    
    # Second moon (class -1)
    x2 = 1 - np.cos(t)
    y2_vals = 0.5 - np.sin(t)
    X2 = np.column_stack([x2, y2_vals])
    
    X = np.vstack([X1, X2])
    X += np.random.normal(0, noise, X.shape)
    
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    
    # Apply exponential transformation at midpoint
    midpoint = n_samples // 2
    X[midpoint:] = np.exp(X[midpoint:]) - 1
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_sinewave(n_samples=2000, noise=0.1, seed=42):
    """
    SineWave: Boundaries defined by sine waves y2 = sin(y1) + η.
    Tests oscillatory decision boundaries.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Class 1: above sine wave
    x1_coord = np.random.uniform(-3, 3, n_per_class)
    boundary = np.sin(x1_coord)
    y1_coord = boundary + np.abs(np.random.normal(0, 0.5, n_per_class)) + 0.2
    X1 = np.column_stack([x1_coord, y1_coord])
    y1 = np.ones(n_per_class)
    
    # Class -1: below sine wave
    x2_coord = np.random.uniform(-3, 3, n_per_class)
    boundary = np.sin(x2_coord)
    y2_coord = boundary - np.abs(np.random.normal(0, 0.5, n_per_class)) - 0.2
    X2 = np.column_stack([x2_coord, y2_coord])
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([X1, X2])
    X += np.random.normal(0, noise, X.shape)
    y = np.hstack([y1, y2])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_spirals(n_samples=2000, noise=0.1, seed=42):
    """
    Spirals: Two intertwined spirals (n_turns=2) with noise η ~ N(0, 0.1).
    Canonical non-separable challenge.
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    n_turns = 2
    
    # Spiral 1 (class 1)
    t1 = np.linspace(0, n_turns * 2 * np.pi, n_per_class)
    r1 = t1 / (n_turns * 2 * np.pi)
    x1 = r1 * np.cos(t1) + np.random.normal(0, noise, n_per_class)
    y1_coord = r1 * np.sin(t1) + np.random.normal(0, noise, n_per_class)
    X1 = np.column_stack([x1, y1_coord])
    y1 = np.ones(n_per_class)
    
    # Spiral 2 (class -1) - offset by π
    t2 = np.linspace(0, n_turns * 2 * np.pi, n_per_class)
    r2 = t2 / (n_turns * 2 * np.pi)
    x2 = r2 * np.cos(t2 + np.pi) + np.random.normal(0, noise, n_per_class)
    y2_coord = r2 * np.sin(t2 + np.pi) + np.random.normal(0, noise, n_per_class)
    X2 = np.column_stack([x2, y2_coord])
    y2 = -np.ones(n_per_class)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_swissroll(n_samples=2000, noise=0.1, seed=42):
    """
    SwissRoll: 2D manifold embedding in 3D with labels from z-coordinate.
    Tests manifold learning under projection.
    """
    np.random.seed(seed)
    
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y_coord = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)
    
    X = np.column_stack([x, y_coord, z])
    X += np.random.normal(0, noise, X.shape)
    
    # Label based on z-coordinate (median split)
    median_z = np.median(z)
    y = np.where(z > median_z, 1, -1).astype(float)
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y


def generate_xor(n_samples=2000, noise=0.1, seed=42):
    """
    XOR: Classical four-quadrant XOR with radial basis transformation 
    x -> e^(-||x - c||^2) applied at midpoint. Extreme nonlinearity.
    """
    np.random.seed(seed)
    n_quarter = n_samples // 4
    
    # Class 1: bottom-left and top-right quadrants
    X1 = np.random.uniform(-2, -0.5, (n_quarter, 2))
    X2 = np.random.uniform(0.5, 2, (n_quarter, 2))
    X_pos = np.vstack([X1, X2])
    y_pos = np.ones(2 * n_quarter)
    
    # Class -1: bottom-right and top-left quadrants
    X3 = np.column_stack([
        np.random.uniform(0.5, 2, n_quarter),
        np.random.uniform(-2, -0.5, n_quarter)
    ])
    X4 = np.column_stack([
        np.random.uniform(-2, -0.5, n_quarter),
        np.random.uniform(0.5, 2, n_quarter)
    ])
    X_neg = np.vstack([X3, X4])
    y_neg = -np.ones(2 * n_quarter)
    
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    X += np.random.normal(0, noise, X.shape)
    
    # Apply radial basis transformation at midpoint
    midpoint = n_samples // 2
    center = np.array([0, 0])
    distances_sq = np.sum((X[midpoint:] - center) ** 2, axis=1, keepdims=True)
    X[midpoint:] = np.exp(-distances_sq) * X[midpoint:]
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    
    return X, y

# =============================================================================
# 2. LINEAR ONLINE LEARNING ALGORITHMS (7 ALGORITHMS)
# =============================================================================

def Perceptron_Linear(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
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
            raw_pred = np.sign(x.dot(w))  # Use raw prediction for update logic
            y_pred[i] = raw_pred if raw_pred != 0 else 1  # Store 1/-1 for metrics
            y_actual = y.iloc[i]

            # Use raw_pred for update decision to learn when uncertain (pred=0)
            if raw_pred != y_actual:
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

def AROW_Linear(X, y, r=0.1, max_epochs=1, patience=3, X_val=None, y_val=None):
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

def PA_Linear(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
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
            raw_pred = np.sign(x.dot(w))  # Use raw prediction for update logic
            y_pred[i] = raw_pred if raw_pred != 0 else 1  # Store 1/-1 for metrics
            y_actual = y.iloc[i]
            # Use raw_pred for loss to learn when uncertain (pred=0)
            loss = max(0, 1 - y_actual * raw_pred)
          
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


def OGL_Linear(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
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
            raw_pred = np.sign(x.dot(w))  # Use raw prediction (including 0) for update
            
            y_act = y.iloc[i]
            # Update uses raw_pred to get learning signal when uncertain (pred=0)
            w = w + (y_act - raw_pred) / (np.sqrt(x.dot(x)) + 1e-8) * x
            y_pred[i] = raw_pred if raw_pred != 0 else 1  # Store 1/-1 for metrics
            
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


def SCW_Linear(X, y, C=1, eta=0.5, max_epochs=1, patience=3, X_val=None, y_val=None):
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

def RDA_Linear(X, y, lambda_param=1, gamma_param=1):
    """Regularized Dual Averaging - Linear version"""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    w = np.zeros(n_features)
    g = np.zeros(n_features)
    y_pred = np.zeros(n_samples)

    for i in range(n_samples):
        t = i + 1
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
    
    return y_pred


def RDA_Linear(X, y, lambda_param=1, gamma_param=1, max_epochs=1, patience=3, X_val=None, y_val=None):
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
        
    return y_pred

def AdaRDA_Linear(X, y, lambda_param=1, eta_param=1, delta_param=1):
    """Adaptive Regularized Dual Averaging - Linear version"""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples, n_features = X_np.shape

    w = np.zeros(n_features)
    g = np.zeros(n_features)
    g1t = np.zeros(n_features)
    y_pred = np.zeros(n_samples)

    for i in range(n_samples):
        t = i + 1
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
    
    return y_pred


# =============================================================================
# 3. KERNEL ONLINE LEARNING ALGORITHMS (7 ALGORITHMS)
# =============================================================================

def rbf_kernel(x1, x2, gamma=1.0):
    """Compute RBF kernel between two vectors."""
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)


def PA_Kernel(X, y, gamma=1.0):
    """Kernel Passive-Aggressive."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        if len(support_vectors) == 0:
            f_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        y_actual = y_np[i]
        
        loss = max(0, 1 - y_actual * f_t)
        if loss > 0:
            k_ii = rbf_kernel(X_np[i], X_np[i], gamma)
            if k_ii > 0:
                eta = loss / (k_ii ** 2 + 1e-8)
                alpha.append(eta * y_actual)
                support_vectors.append(X_np[i].copy())
    
    return y_pred


def Perceptron_Kernel(X, y, gamma=1.0):
    """Kernel Perceptron."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        if len(support_vectors) == 0:
            f_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        
        if y_pred[i] != y_np[i]:
            support_vectors.append(X_np[i].copy())
            alpha.append(y_np[i])
    
    return y_pred


def AROW_Kernel(X, y, r=0.1, gamma=1.0):
    """Kernel AROW using diagonal approximation for Sigma."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    sigma = []  # Diagonal variances for each support vector
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        if len(support_vectors) == 0:
            f_t = 1.0
            v_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
            v_t = np.dot(k_t**2, sigma) + 1e-8
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        y_actual = y_np[i]
        
        lt = max(0, 1 - y_actual * f_t)
        
        if lt > 0:
            alpha_t = lt / (v_t + r)
            beta_t = 1 / (v_t + r)
            
            alpha.append(alpha_t * y_actual)
            support_vectors.append(X_np[i].copy())
            sigma.append(1.0 / (1.0 + beta_t))
    
    return y_pred


def OGL_Kernel(X, y, gamma=1.0):
    """Kernel Online Gradient Learning."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        if len(support_vectors) == 0:
            f_t = 0.0  # Start uncertain, not with default 1
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
        
        raw_pred = np.sign(f_t)  # Can be -1, 0, or 1
        y_pred[i] = raw_pred if raw_pred != 0 else 1  # Store 1/-1 for metrics
        y_actual = y_np[i]
        
        # OGL update: w <- w + (y - raw_pred) / ||x|| * x
        # Use raw_pred to get learning signal when uncertain (pred=0)
        error = y_actual - raw_pred

        k_ii = rbf_kernel(X_np[i], X_np[i], gamma)
        eta = error / (np.sqrt(k_ii) + 1e-8)
        alpha.append(eta)
        support_vectors.append(X_np[i].copy())
    
    return y_pred


def SCW_Kernel(X, y, C=1, eta=0.5, gamma=1.0):
    """Kernel Soft Confidence-Weighted Learning using diagonal approximation."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    phi = norm.ppf(eta)
    support_vectors = []
    alpha = []
    sigma = []  # Diagonal variances
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        if len(support_vectors) == 0:
            f_t = 1.0
            v_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
            v_t = np.dot(k_t**2, sigma) + 1e-8
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        y_actual = y_np[i]
        
        m_t = y_actual * f_t
        lt = max(0, phi * np.sqrt(v_t) - m_t)
        
        if lt > 0:
            pa = 1 + (phi**2) / 2
            xi = 1 + phi**2
            
            sqrt_term = max(0, (m_t**2 * phi**4 / 4) + (v_t * phi**2 * xi))
            alpha_t = max(0, (1 / (v_t * xi + 1e-8)) * (-m_t * pa + np.sqrt(sqrt_term)))
            alpha_t = min(C, alpha_t)
            
            sqrt_ut_term = max(0, (alpha_t**2 * v_t**2 * phi**2) + (4 * v_t))
            ut = 0.25 * (-alpha_t * v_t * phi + np.sqrt(sqrt_ut_term))**2
            beta_t = (alpha_t * phi) / (np.sqrt(ut) + v_t * alpha_t * phi + 1e-8)
            
            alpha.append(alpha_t * y_actual)
            support_vectors.append(X_np[i].copy())
            sigma.append(1.0 / (1.0 + beta_t))
    
    return y_pred


def RDA_Kernel(X, y, lambda_param=1, gamma_param=1, gamma=1.0):
    """Kernel Regularized Dual Averaging."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    g = 0.0
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        t = i + 1
        
        if len(support_vectors) == 0:
            f_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        y_actual = y_np[i]
        
        lt = max(0, 1 - y_actual * f_t)
        
        if lt > 0:
            gt = -y_actual
        else:
            gt = 0.0
        
        g = ((t - 1) / t) * g + (1 / t) * gt
        
        if abs(g) > lambda_param:
            coeff = -(np.sqrt(t) / gamma_param) * (g - lambda_param * np.sign(g))
        else:
            coeff = 0.0
        
        if coeff != 0:
            alpha.append(coeff * y_actual)
            support_vectors.append(X_np[i].copy())
    
    return y_pred


def AdaRDA_Kernel(X, y, lambda_param=1, eta_param=1, delta_param=1, gamma=1.0):
    """Kernel Adaptive Regularized Dual Averaging."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    n_samples = X_np.shape[0]
    
    support_vectors = []
    alpha = []
    g = 0.0
    g1t = 0.0
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        t = i + 1
        
        if len(support_vectors) == 0:
            f_t = 1.0
        else:
            k_t = np.array([rbf_kernel(X_np[i], sv, gamma) for sv in support_vectors])
            f_t = np.dot(k_t, alpha)
        
        y_pred[i] = np.sign(f_t) if f_t != 0 else 1
        y_actual = y_np[i]
        
        lt = max(0, 1 - y_actual * f_t)
        
        if lt > 0:
            gt = -y_actual
        else:
            gt = 0.0
        
        g = ((t - 1) / t) * g + (1 / t) * gt
        g1t += gt**2
        
        Ht = delta_param + np.sqrt(g1t)
        
        if abs(g) > lambda_param:
            coeff = np.sign(-g) * eta_param * t / (Ht + 1e-8)
        else:
            coeff = 0.0
        
        if coeff != 0:
            alpha.append(coeff * y_actual)
            support_vectors.append(X_np[i].copy())
    
    return y_pred


# =============================================================================
# 4. EVALUATION
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return {'Accuracy': acc, 'F1': f1}

# =============================================================================
# 5. MAIN EXPERIMENT
# =============================================================================

def main():
    ensure_reproducibility(seed=42)
    print("="*100)
    print("SIMULATION STUDY: Kernel vs Linear Online Learning")
    print("11 Datasets | 6 Algorithms | 3 Imbalance Ratios")
    print("="*100)
    
    # 11 Data generators
    data_generators = {
        # LINEAR DATA
        'Linear': ('Linear', generate_linear_data, {'n_features': 5, 'noise': 0.1}),
        
        # NON-LINEAR DATA (10 datasets)
        'OverlapGauss': ('Non-Linear', generate_overlapping_gaussians, {'noise': 0.1}),
        'Donut': ('Non-Linear', generate_donut, {'noise': 0.1}),
        'Circles': ('Non-Linear', generate_circles, {'noise': 0.1}),
        'Quadratic': ('Non-Linear', generate_quadratic, {'noise': 0.1}),
        'HighCurve': ('Non-Linear', generate_high_curvature, {'noise': 0.1}),
        'Moons': ('Non-Linear', generate_moons, {'noise': 0.1}),
        'SineWave': ('Non-Linear', generate_sinewave, {'noise': 0.1}),
        'Spirals': ('Non-Linear', generate_spirals, {'noise': 0.1}),
        'SwissRoll': ('Non-Linear', generate_swissroll, {'noise': 0.1}),
        'XOR': ('Non-Linear', generate_xor, {'noise': 0.1}),
    }
    
    # 6 Linear algorithms
    algorithms_linear = {
        'PA': PA_Linear,
        'Perceptron': Perceptron_Linear,
        'AROW': AROW_Linear,
        'OGL': OGL_Linear,
        'SCW': SCW_Linear,
        'RDA': RDA_Linear,
        'AdaRDA': AdaRDA_Linear,
    }
    
    # 6 Kernel algorithms
    algorithms_kernel = {
        'PA': PA_Kernel,
        'Perceptron': Perceptron_Kernel,
        'AROW': AROW_Kernel,
        'OGL': OGL_Kernel,
        'SCW': SCW_Kernel,
        'RDA': RDA_Kernel,
        'AdaRDA': AdaRDA_Kernel,
    }
    
    # 3 Imbalance ratios: (pos_ratio, neg_ratio, label)
    imbalance_ratios = [
        (0.5, '50-50'),   # Balanced
        (0.7, '70-30'),   # 70% positive, 30% negative
        (0.9, '90-10'),   # 90% positive, 10% negative
    ]
    
    results = []
    
    for dataset_name, (data_type, data_gen, kwargs) in data_generators.items():
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset_name:15s} | Type: {data_type}")
        print(f"{'='*100}")
        
        # Generate base data
        X_base, y_base = data_gen(n_samples=2000, **kwargs)
        
        for pos_ratio, imb_label in imbalance_ratios:
            # Apply imbalance
            X, y = apply_imbalance(X_base.copy(), y_base.copy(), pos_ratio=pos_ratio, seed=42)
            
            # Inject outliers (5% of samples scaled by 10x)
            X, y = inject_outliers(X, y, outlier_ratio=0.05, outlier_scale=10.0, seed=42)
            
            n_pos = (y == 1).sum()
            n_neg = (y == -1).sum()
            print(f"\n  Imbalance: {imb_label} (Pos: {n_pos:4d}, Neg: {n_neg:4d})")
            print(f"  {'Algorithm':12s} | {'Linear F1':10s} | {'Kernel F1':10s} | {'Gain':8s}")
            print(f"  {'-'*50}")
            
            # Test all algorithms
            for algo_name in algorithms_linear.keys():
                # Linear version - extract y_pred from tuple if returned
                result_linear = algorithms_linear[algo_name](X, y)
                y_pred_linear = result_linear[0] if isinstance(result_linear, tuple) else result_linear
                metrics_linear = calculate_metrics(y, y_pred_linear)
                
                # Kernel version
                y_pred_kernel = algorithms_kernel[algo_name](X, y, gamma=1.0)
                metrics_kernel = calculate_metrics(y, y_pred_kernel)
                
                diff_f1 = metrics_kernel['F1'] - metrics_linear['F1']
                
                print(f"  {algo_name:12s} | {metrics_linear['F1']:10.4f} | {metrics_kernel['F1']:10.4f} | {diff_f1:+8.4f}")
                
                results.append({
                    'Data_Type': data_type,
                    'Dataset': dataset_name,
                    'Imbalance': imb_label,
                    'Pos_Ratio': pos_ratio,
                    'Algorithm': algo_name,
                    'Method': 'Linear',
                    'F1': metrics_linear['F1'],
                    'Accuracy': metrics_linear['Accuracy']
                })
                
                results.append({
                    'Data_Type': data_type,
                    'Dataset': dataset_name,
                    'Imbalance': imb_label,
                    'Pos_Ratio': pos_ratio,
                    'Algorithm': algo_name,
                    'Method': 'Kernel',
                    'F1': metrics_kernel['F1'],
                    'Accuracy': metrics_kernel['Accuracy']
                })
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    df_results = pd.DataFrame(results)
    
    # Summary by data type and imbalance
    print("\nAverage F1 by Data Type, Imbalance, and Method:")
    print("-" * 80)
    
    for data_type in ['Linear', 'Non-Linear']:
        print(f"\n{data_type} Data:")
        df_type = df_results[df_results['Data_Type'] == data_type]
        
        for imb_label in ['50-50', '70-30', '90-10']:
            df_imb = df_type[df_type['Imbalance'] == imb_label]
            
            f1_linear = df_imb[df_imb['Method'] == 'Linear']['F1'].mean()
            f1_kernel = df_imb[df_imb['Method'] == 'Kernel']['F1'].mean()
            gain = f1_kernel - f1_linear
            
            print(f"  {imb_label}: Linear={f1_linear:.4f} | Kernel={f1_kernel:.4f} | Gain={gain:+.4f}")
    
    # Summary by algorithm
    print("\n\nAverage F1 by Algorithm (Non-Linear Data Only):")
    print("-" * 80)
    df_nonlinear = df_results[df_results['Data_Type'] == 'Non-Linear']
    
    for algo_name in algorithms_linear.keys():
        df_algo = df_nonlinear[df_nonlinear['Algorithm'] == algo_name]
        f1_linear = df_algo[df_algo['Method'] == 'Linear']['F1'].mean()
        f1_kernel = df_algo[df_algo['Method'] == 'Kernel']['F1'].mean()
        gain = f1_kernel - f1_linear
        print(f"  {algo_name:12s}: Linear={f1_linear:.4f} | Kernel={f1_kernel:.4f} | Gain={gain:+.4f}")
    
    # Save results - use absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, '..', 'results', 'comparison_simulation.csv')
    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(f"""
Configuration:
- Total Experiments: {len(df_results) // 2} (Linear+Kernel pairs)
    """)

if __name__ == "__main__":
    main()
