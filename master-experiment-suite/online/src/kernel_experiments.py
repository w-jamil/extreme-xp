#!/usr/bin/env python3
"""
Uses efficient vectorized kernel computations with configurable support vector budget.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import os

# =============================================================================
# VECTORIZED KERNEL FUNCTIONS
# =============================================================================

def rbf_kernel_matrix(X1, X2, gamma):
    """Vectorized RBF kernel matrix K[i,j] = exp(-gamma * ||X1[i] - X2[j]||^2)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    cross = X1 @ X2.T  # (n1, n2)
    dist_sq = X1_sq + X2_sq.T - 2 * cross  # (n1, n2)
    dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
    return np.exp(-gamma * dist_sq)


def polynomial_kernel_matrix(X1, X2, degree=3, c=1.0):
    """Vectorized polynomial kernel K[i,j] = (X1[i] @ X2[j] + c)^degree."""
    return (X1 @ X2.T + c) ** degree


def linear_kernel_matrix(X1, X2):
    """Vectorized linear kernel K[i,j] = X1[i] @ X2[j]."""
    return X1 @ X2.T


# =============================================================================
# VECTORIZED BATCH KERNEL ALGORITHMS
# =============================================================================

class KernelPerceptron:
    """Batch Kernel Perceptron with support vector budget."""
    
    def __init__(self, gamma=0.1, max_sv=1000):
        self.gamma = gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=5):
        """Train with multiple passes, budget limiting."""
        n_samples = X.shape[0]
        sv_mask = np.zeros(n_samples, dtype=bool)
        alpha = np.zeros(n_samples)
        
        for epoch in range(epochs):
            # Compute kernel matrix for current SVs
            if sv_mask.sum() == 0:
                K = np.zeros((n_samples, n_samples))
            else:
                X_sv = X[sv_mask]
                K_sv = rbf_kernel_matrix(X, X_sv, self.gamma)  # (n, n_sv)
                
            for i in range(n_samples):
                if sv_mask.sum() == 0:
                    pred = 0.0
                else:
                    pred = np.dot(K_sv[i], alpha[sv_mask])
                
                pred_label = 1 if pred >= 0 else -1
                
                if pred_label != y[i]:
                    alpha[i] += y[i]
                    sv_mask[i] = True
                    
                    # Budget: keep most recent
                    if sv_mask.sum() > self.max_sv:
                        oldest_idx = np.where(sv_mask)[0][0]
                        sv_mask[oldest_idx] = False
                        alpha[oldest_idx] = 0
                    
                    # Recompute K_sv after adding new SV
                    X_sv = X[sv_mask]
                    K_sv = rbf_kernel_matrix(X, X_sv, self.gamma)
        
        self.support_vectors = X[sv_mask]
        self.alpha = alpha[sv_mask]
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelPassiveAggressive:
    """Batch Kernel PA with support vector budget."""
    
    def __init__(self, gamma=0.1, max_sv=1000, C=1.0):
        self.gamma = gamma
        self.max_sv = max_sv
        self.C = C
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=3):
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i:i+1]
                
                if len(sv_indices) == 0:
                    pred = 0.0
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]
                    pred = np.dot(k_vec, alpha)
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                
                loss = max(0, 1 - y[i] * pred)
                
                if loss > 0:
                    # PA-I update: tau = min(C, loss / k(x,x))
                    tau = min(self.C, loss / (k_xi_xi + 1e-8))
                    
                    if i in sv_indices:
                        idx = sv_indices.index(i)
                        alpha_list[idx] += tau * y[i]
                    else:
                        sv_indices.append(i)
                        alpha_list.append(tau * y[i])
                    
                    # Budget
                    if len(sv_indices) > self.max_sv:
                        sv_indices.pop(0)
                        alpha_list.pop(0)
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelAROW:
    """Batch Kernel AROW (Adaptive Regularization of Weights)."""
    
    def __init__(self, gamma=0.1, r=1.0, max_sv=500):
        self.gamma = gamma
        self.r = r
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
        self.Sigma = None
    
    def fit(self, X, y, epochs=2):
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        
        # Sigma will be (n_sv x n_sv) for the covariance in kernel space
        Sigma = None
        
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i:i+1]
                
                if len(sv_indices) == 0:
                    pred = 0.0
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                    v_t = k_xi_xi
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]  # (n_sv,)
                    pred = np.dot(k_vec, alpha)
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                    
                    # v_t = k' @ Sigma @ k
                    Sigma_k = Sigma @ k_vec
                    v_t = np.dot(k_vec, Sigma_k)
                
                loss = max(0, 1 - y[i] * pred)
                
                if loss > 0:
                    beta_t = 1.0 / (v_t + self.r)
                    alpha_t = loss * beta_t
                    
                    if i in sv_indices:
                        idx = sv_indices.index(i)
                        alpha_list[idx] += alpha_t * y[i]
                        
                        # Update Sigma
                        k_vec = rbf_kernel_matrix(x_i, X[sv_indices], self.gamma)[0]
                        Sigma_k = Sigma @ k_vec
                        Sigma -= beta_t * np.outer(Sigma_k, Sigma_k)
                    else:
                        sv_indices.append(i)
                        alpha_list.append(alpha_t * y[i])
                        
                        # Expand Sigma
                        if Sigma is None:
                            Sigma = np.array([[1.0]])
                        else:
                            n_sv = Sigma.shape[0]
                            new_Sigma = np.eye(n_sv + 1)
                            new_Sigma[:n_sv, :n_sv] = Sigma
                            Sigma = new_Sigma
                    
                    # Budget
                    if len(sv_indices) > self.max_sv:
                        sv_indices.pop(0)
                        alpha_list.pop(0)
                        Sigma = Sigma[1:, 1:]
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelOGL:
    """Batch Kernel Online Gradient Learning with gradient averaging.
    
    Uses stochastic gradient descent on hinge loss with:
    - Cumulative gradient tracking
    - Class-weighted updates for imbalanced data
    - Scaling with sqrt(t) for growing confidence
    - No explicit regularization (parameter-free)
    """
    
    def __init__(self, gamma=0.1, max_sv=1000):
        self.gamma = gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=3):
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        grad_sum = {}  # Accumulated gradients per support vector
        
        # Class weights for imbalanced data (inverse frequency)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        pos_weight = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
        neg_weight = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
        
        t = 0
        for epoch in range(epochs):
            for i in range(n_samples):
                t += 1
                x_i = X[i:i+1]
                
                # Compute prediction
                if len(sv_indices) == 0:
                    score = 0.0
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]
                    score = np.dot(k_vec, alpha)
                
                # Hinge loss gradient: -y if margin < 1, else 0
                margin = y[i] * score
                if margin < 1.0:
                    # Class-weighted gradient
                    weight = pos_weight if y[i] == 1 else neg_weight
                    grad = -y[i] * weight
                else:
                    grad = 0.0
                
                # Update gradient sum and alpha
                if i in sv_indices:
                    idx = sv_indices.index(i)
                    grad_sum[i] = grad_sum.get(i, 0.0) + grad
                    
                    # OGL update: alpha = -sqrt(t) * avg_grad
                    avg_grad = grad_sum[i] / t
                    alpha_list[idx] = -np.sqrt(t) * avg_grad
                    
                elif grad != 0:
                    sv_indices.append(i)
                    grad_sum[i] = grad
                    avg_grad = grad_sum[i] / t
                    alpha_list.append(-np.sqrt(t) * avg_grad)
                
                # Budget: remove smallest magnitude alpha
                if len(sv_indices) > self.max_sv:
                    min_idx = np.argmin(np.abs(alpha_list))
                    removed_idx = sv_indices.pop(min_idx)
                    alpha_list.pop(min_idx)
                    grad_sum.pop(removed_idx, None)
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelRDA:
    """Batch Kernel Regularized Dual Averaging."""
    
    def __init__(self, gamma=0.1, lambda_param=0.01, rda_gamma=1.0, max_sv=500):
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.rda_gamma = rda_gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=3):
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        grad_sum = {}  # Accumulated gradients per support vector
        
        t = 0
        for epoch in range(epochs):
            for i in range(n_samples):
                t += 1
                x_i = X[i:i+1]
                
                if len(sv_indices) == 0:
                    pred = 0.0
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]
                    pred = np.dot(k_vec, alpha)
                
                loss = max(0, 1 - y[i] * pred)
                
                # Always update gradient average (RDA)
                if loss > 0:
                    grad = -y[i]
                else:
                    grad = 0.0
                
                if i in sv_indices:
                    idx = sv_indices.index(i)
                    old_sum = grad_sum.get(i, 0.0)
                    grad_sum[i] = old_sum + grad
                    
                    avg_grad = grad_sum[i] / t
                    if abs(avg_grad) > self.lambda_param:
                        alpha_list[idx] = -(np.sqrt(t) / self.rda_gamma) * \
                            (avg_grad - self.lambda_param * np.sign(avg_grad))
                    else:
                        alpha_list[idx] = 0.0
                elif grad != 0:
                    sv_indices.append(i)
                    grad_sum[i] = grad
                    avg_grad = grad_sum[i] / t
                    if abs(avg_grad) > self.lambda_param:
                        alpha_list.append(-(np.sqrt(t) / self.rda_gamma) * \
                            (avg_grad - self.lambda_param * np.sign(avg_grad)))
                    else:
                        alpha_list.append(0.0)
                
                # Budget
                if len(sv_indices) > self.max_sv:
                    removed_idx = sv_indices.pop(0)
                    alpha_list.pop(0)
                    grad_sum.pop(removed_idx, None)
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelSCW:
    """Batch Kernel Soft Confidence-Weighted Learning."""
    
    def __init__(self, gamma=0.1, C=1.0, phi=0.5, max_sv=500):
        self.gamma = gamma
        self.C = C
        self.phi = phi
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=2):
        from scipy.stats import norm
        
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        Sigma = None
        
        # SCW constants
        psi = 1 + self.phi ** 2 / 2
        zeta = 1 + self.phi ** 2
        
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i:i+1]
                
                if len(sv_indices) == 0:
                    pred = 0.0
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                    v_t = k_xi_xi
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]
                    pred = np.dot(k_vec, alpha)
                    k_xi_xi = rbf_kernel_matrix(x_i, x_i, self.gamma)[0, 0]
                    
                    Sigma_k = Sigma @ k_vec
                    v_t = np.dot(k_vec, Sigma_k)
                
                m_t = y[i] * pred
                loss = self.phi * np.sqrt(v_t) - m_t
                
                if loss > 0:
                    # SCW-I update
                    alpha_t = max(0, (-m_t * psi + np.sqrt(m_t ** 2 * self.phi ** 4 / 4 + 
                                    v_t * self.phi ** 2 * zeta)) / (v_t * zeta + 1e-8))
                    alpha_t = min(alpha_t, self.C)
                    
                    u_t = 0.25 * (-alpha_t * v_t * self.phi + 
                                  np.sqrt(alpha_t ** 2 * v_t ** 2 * self.phi ** 2 + 4 * v_t)) ** 2
                    beta_t = alpha_t * self.phi / (np.sqrt(u_t) + v_t * alpha_t * self.phi + 1e-8)
                    
                    if i in sv_indices:
                        idx = sv_indices.index(i)
                        alpha_list[idx] += alpha_t * y[i]
                        
                        k_vec = rbf_kernel_matrix(x_i, X[sv_indices], self.gamma)[0]
                        Sigma_k = Sigma @ k_vec
                        Sigma -= beta_t * np.outer(Sigma_k, Sigma_k)
                    else:
                        sv_indices.append(i)
                        alpha_list.append(alpha_t * y[i])
                        
                        if Sigma is None:
                            Sigma = np.array([[1.0]])
                        else:
                            n_sv = Sigma.shape[0]
                            new_Sigma = np.eye(n_sv + 1)
                            new_Sigma[:n_sv, :n_sv] = Sigma
                            Sigma = new_Sigma
                    
                    # Budget
                    if len(sv_indices) > self.max_sv:
                        sv_indices.pop(0)
                        alpha_list.pop(0)
                        Sigma = Sigma[1:, 1:]
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


class KernelAdaRDA:
    """Batch Kernel Adaptive RDA."""
    
    def __init__(self, gamma=0.1, lambda_param=0.01, delta=1.0, max_sv=500):
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.delta = delta
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None
    
    def fit(self, X, y, epochs=3):
        n_samples = X.shape[0]
        sv_indices = []
        alpha_list = []
        grad_sum = {}
        sq_grad_sum = {}
        
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i:i+1]
                
                if len(sv_indices) == 0:
                    pred = 0.0
                else:
                    X_sv = X[sv_indices]
                    alpha = np.array(alpha_list)
                    k_vec = rbf_kernel_matrix(x_i, X_sv, self.gamma)[0]
                    pred = np.dot(k_vec, alpha)
                
                loss = max(0, 1 - y[i] * pred)
                
                # Subgradient
                if loss > 0:
                    grad = -y[i]
                else:
                    grad = 0.0
                
                if i in sv_indices:
                    idx = sv_indices.index(i)
                    grad_sum[i] = grad_sum.get(i, 0.0) + grad
                    sq_grad_sum[i] = sq_grad_sum.get(i, 0.0) + grad ** 2
                    
                    H_t = self.delta + np.sqrt(sq_grad_sum[i])
                    avg_grad = grad_sum[i]
                    
                    if abs(avg_grad) > self.lambda_param * H_t:
                        alpha_list[idx] = -(1.0 / H_t) * \
                            (avg_grad - self.lambda_param * H_t * np.sign(avg_grad))
                    else:
                        alpha_list[idx] = 0.0
                elif grad != 0:
                    sv_indices.append(i)
                    grad_sum[i] = grad
                    sq_grad_sum[i] = grad ** 2
                    H_t = self.delta + np.sqrt(sq_grad_sum[i])
                    avg_grad = grad_sum[i]
                    
                    if abs(avg_grad) > self.lambda_param * H_t:
                        alpha_list.append(-(1.0 / H_t) * \
                            (avg_grad - self.lambda_param * H_t * np.sign(avg_grad)))
                    else:
                        alpha_list.append(0.0)
                
                # Budget
                if len(sv_indices) > self.max_sv:
                    removed = sv_indices.pop(0)
                    alpha_list.pop(0)
                    grad_sum.pop(removed, None)
                    sq_grad_sum.pop(removed, None)
        
        if len(sv_indices) > 0:
            self.support_vectors = X[sv_indices]
            self.alpha = np.array(alpha_list)
        else:
            self.support_vectors = np.array([]).reshape(0, X.shape[1])
            self.alpha = np.array([])
        return self
    
    def decision_function(self, X):
        """Return raw decision scores for ROC curves."""
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        return K @ self.alpha
    
    def predict(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.ones(len(X))
        K = rbf_kernel_matrix(X, self.support_vectors, self.gamma)
        scores = K @ self.alpha
        return np.where(scores >= 0, 1, -1)


# =============================================================================
# F1 SCORE PLOTTING (PER SHUFFLE)
# =============================================================================

def plot_f1_scores(f1_data, dataset_name, output_path=None, highlight='K-OGL'):
    """
    Plot F1 scores for each shuffle across all algorithms.
    
    Args:
        f1_data: Dict mapping algorithm name to list of F1 scores (one per shuffle)
        dataset_name: Name of dataset for the title
        output_path: Path to save figure (None = show only)
        highlight: Algorithm to highlight (default: K-OGL)
    """
    plt.figure(figsize=(12, 6))
    
    # Style settings for each algorithm
    styles = {
        'K-OGL': {'color': '#e41a1c', 'linewidth': 2.5, 'linestyle': '-', 'marker': 'o', 'markersize': 3},
        'K-Perceptron': {'color': '#377eb8', 'linewidth': 1.0, 'linestyle': '--', 'marker': '', 'markersize': 0},
        'K-PA': {'color': '#4daf4a', 'linewidth': 1.0, 'linestyle': '--', 'marker': '', 'markersize': 0},
        'K-AROW': {'color': '#984ea3', 'linewidth': 1.0, 'linestyle': '-.', 'marker': '', 'markersize': 0},
        'K-RDA': {'color': '#ff7f00', 'linewidth': 1.0, 'linestyle': '-.', 'marker': '', 'markersize': 0},
        'K-SCW': {'color': '#a65628', 'linewidth': 1.0, 'linestyle': ':', 'marker': '', 'markersize': 0},
        'K-AdaRDA': {'color': '#f781bf', 'linewidth': 1.0, 'linestyle': ':', 'marker': '', 'markersize': 0},
    }
    
    n_shuffles = len(list(f1_data.values())[0])
    x = np.arange(1, n_shuffles + 1)
    
    # Plot each algorithm
    for name, f1_scores in f1_data.items():
        style = styles.get(name, {'color': 'gray', 'linewidth': 1, 'linestyle': '--', 'marker': '', 'markersize': 0})
        
        if name == highlight:
            plt.plot(x, f1_scores, label=f'{name} (mean={np.mean(f1_scores):.4f})',
                     color=style['color'], linewidth=style['linewidth'], 
                     linestyle=style['linestyle'], marker=style['marker'],
                     markersize=style['markersize'], zorder=10, alpha=0.9)
        else:
            plt.plot(x, f1_scores, label=f'{name} (mean={np.mean(f1_scores):.4f})',
                     color=style['color'], linewidth=style['linewidth'],
                     linestyle=style['linestyle'], alpha=0.6)
    
    plt.xlabel('Shuffle Number', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'F1 Scores Across Shuffles - {dataset_name}\n(K-OGL highlighted)', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, n_shuffles)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"F1 plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()



# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_dataset(dataset_name, time_split=True, test_ratio=0.2, subsample=None):
    """
    Generic dataset loader for all parquet files in data directory.
    
    Args:
        dataset_name: Name of parquet file (without .parquet extension)
        time_split: If True, temporal split (no shuffle). If False, random shuffle.
        test_ratio: Fraction for test set (default 0.2)
        subsample: If set, subsample to this many rows
    
    Returns:
        X_train, y_train, X_test, y_test (scaled, labels in {-1, +1})
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    parquet_path = os.path.join(data_dir, f'{dataset_name}.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset not found at: {parquet_path}")
    
    print(f"Loading {dataset_name} from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if subsample and len(df) > subsample:
        df = df.head(subsample)
    
    # Identify label column and timestamp column
    if 'Class' in df.columns:
        label_col = 'Class'
        time_col = 'Time'
        drop_cols = ['Time', 'Class']
    elif 'label' in df.columns:
        label_col = 'label'
        time_col = 'timestamp'
        drop_cols = ['label', 'user_id', 'timestamp', 'entity']
        drop_cols = [c for c in drop_cols if c in df.columns]
    else:
        raise ValueError(f"No known label column in {dataset_name}")
    
    # Sort by time for temporal consistency
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    
    # Extract features and labels
    X = df.drop(drop_cols, axis=1).values.astype(np.float64)
    y = df[label_col].values
    y = np.where(y == 0, -1, 1)  # Convert to {-1, +1}
    
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    
    if time_split:
        # Time-based: first split_idx for train, rest for test (no shuffle)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        # Random shuffle then split
        indices = np.random.permutation(n)
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    pos_label = (y_train == 1).sum()
    pos_ratio = 100 * pos_label / len(y_train) if len(y_train) > 0 else 0
    print(f"  Train: {len(X_train)} samples, +1: {pos_label} ({pos_ratio:.3f}%)")
    pos_test = (y_test == 1).sum()
    pos_test_ratio = 100 * pos_test / len(y_test) if len(y_test) > 0 else 0
    print(f"  Test:  {len(X_test)} samples, +1: {pos_test} ({pos_test_ratio:.3f}%)")
    
    return X_train, y_train, X_test, y_test


def load_credit_fraud(time_split=True, test_ratio=0.2, subsample=None):
    """
    Load Credit Card Fraud dataset from local parquet with time-based split.
    
    Args:
        time_split: If True, use temporal split (first 80% train, last 20% test)
        test_ratio: Fraction for test set
        subsample: If set, subsample to this many rows (for faster testing)
    
    Returns:
        X_train, y_train, X_test, y_test (scaled, labels in {-1, +1})
    """
    # Load from local parquet file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    parquet_path = os.path.join(data_dir, 'CreditFraud_kaggle.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Credit Fraud dataset not found at: {parquet_path}")
    
    print(f"Loading Credit Fraud from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if subsample and len(df) > subsample:
        df = df.head(subsample)
    
    # Sort by Time for temporal split
    df = df.sort_values('Time').reset_index(drop=True)
    
    X = df.drop(['Time', 'Class'], axis=1).values
    y = df['Class'].values
    y = np.where(y == 0, -1, 1)  # Convert to {-1, +1}
    
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    
    if time_split:
        # Time-based: first split_idx for train, rest for test
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        # Random shuffle
        indices = np.random.permutation(n)
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)} samples, Fraud: {(y_train == 1).sum()} ({100 * (y_train == 1).mean():.3f}%)")
    print(f"  Test:  {len(X_test)} samples, Fraud: {(y_test == 1).sum()} ({100 * (y_test == 1).mean():.3f}%)")
    
    return X_train, y_train, X_test, y_test


def load_mnist(random_split=True, test_ratio=0.2, binary_class=(0, 1)):
    """
    Load MNIST dataset from local parquet with random shuffle split.
    
    Args:
        random_split: If True, shuffle before splitting
        test_ratio: Fraction for test set
        binary_class: Tuple of two digit classes for binary classification
    
    Returns:
        X_train, y_train, X_test, y_test (scaled, labels in {-1, +1})
    """
    # Load from local parquet file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    parquet_path = os.path.join(data_dir, 'MNIST_combined.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"MNIST dataset not found at: {parquet_path}")
    
    print(f"Loading MNIST from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Extract features and labels
    y = df['label'].values.astype(int)
    X = df.drop('label', axis=1).values.astype(np.float64)
    
    # Filter to binary classes
    mask = np.isin(y, binary_class)
    X = X[mask]
    y = y[mask]
    
    # Convert to {-1, +1}
    y = np.where(y == binary_class[0], -1, 1)
    
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    
    if random_split:
        indices = np.random.permutation(n)
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    else:
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Binary MNIST classes: {binary_class[0]} vs {binary_class[1]}")
    print(f"  Train: {len(X_train)} samples, Class +1: {(y_train == 1).sum()} ({100 * (y_train == 1).mean():.1f}%)")
    print(f"  Test:  {len(X_test)} samples, Class +1: {(y_test == 1).sum()} ({100 * (y_test == 1).mean():.1f}%)")
    
    return X_train, y_train, X_test, y_test


def load_iris(time_split=True, test_ratio=0.2, binary_class=('Iris-setosa', 'Iris-versicolor')):
    """
    Load Iris dataset from local parquet with time-based split.
    
    Args:
        time_split: If True, use sequential split (first 80% train, last 20% test)
        test_ratio: Fraction for test set
        binary_class: Tuple of two class names for binary classification
    
    Returns:
        X_train, y_train, X_test, y_test (scaled, labels in {-1, +1})
    """
    # Load from local parquet file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    parquet_path = os.path.join(data_dir, 'Iris_UCI.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Iris dataset not found at: {parquet_path}")
    
    print(f"Loading Iris from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Extract features and labels
    y = df['label'].values
    X = df.drop('label', axis=1).values.astype(np.float64)
    
    # Filter to binary classes
    mask = np.isin(y, binary_class)
    X = X[mask]
    y = y[mask]
    
    # Convert to {-1, +1}
    y = np.where(y == binary_class[0], -1, 1)
    
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    
    if time_split:
        # Time-based: first split_idx for train, rest for test (no shuffle)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        # Random shuffle
        indices = np.random.permutation(n)
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Binary Iris classes: {binary_class[0]} vs {binary_class[1]}")
    print(f"  Train: {len(X_train)} samples, Class +1: {(y_train == 1).sum()} ({100 * (y_train == 1).mean():.1f}%)")
    print(f"  Test:  {len(X_test)} samples, Class +1: {(y_test == 1).sum()} ({100 * (y_test == 1).mean():.1f}%)")
    
    return X_train, y_train, X_test, y_test


def load_heart_disease(random_split=True, test_ratio=0.2):
    """
    Load Heart Disease dataset from local parquet with random shuffle split.
    
    Args:
        random_split: If True, shuffle before splitting
        test_ratio: Fraction for test set
    
    Returns:
        X_train, y_train, X_test, y_test (scaled, labels in {-1, +1})
    """
    # Load from local parquet file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    parquet_path = os.path.join(data_dir, 'HeartDisease_UCI.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Heart Disease dataset not found at: {parquet_path}")
    
    print(f"Loading Heart Disease from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Extract features and labels
    y = df['label'].values.astype(int)
    X = df.drop('label', axis=1).values.astype(np.float64)
    
    # Convert to {-1, +1}: 0 = no disease (-1), 1 = disease (+1)
    y = np.where(y == 0, -1, 1)
    
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    
    if random_split:
        indices = np.random.permutation(n)
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    else:
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)} samples, Disease: {(y_train == 1).sum()} ({100 * (y_train == 1).mean():.1f}%)")
    print(f"  Test:  {len(X_test)} samples, Disease: {(y_test == 1).sum()} ({100 * (y_test == 1).mean():.1f}%)")
    
    return X_train, y_train, X_test, y_test


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(y_true, y_pred):
    """Compute classification metrics."""
    # Convert to 0/1 for sklearn metrics
    y_true_01 = np.where(y_true == -1, 0, 1)
    y_pred_01 = np.where(y_pred == -1, 0, 1)
    
    precision = precision_score(y_true_01, y_pred_01, zero_division=0)
    recall = recall_score(y_true_01, y_pred_01, zero_division=0)
    f1 = f1_score(y_true_01, y_pred_01, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true_01, y_pred_01, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FPR': fpr,
        'FNR': fnr,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }


# =============================================================================
# GAMMA TUNING
# =============================================================================

def tune_gamma(X_train, y_train, gamma_values=None, val_ratio=0.2, max_sv=200, epochs=1):
    """
    Find the best gamma using validation set performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        gamma_values: List of gamma values to try (default: logarithmic grid)
        val_ratio: Fraction of training data for validation
        max_sv: Max SVs for tuning (keep low for speed)
        epochs: Epochs per gamma trial
    
    Returns:
        best_gamma: The gamma with highest validation F1
    """
    if gamma_values is None:
        # Logarithmic grid from 0.0001 to 10
        gamma_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Split into train/val
    n = len(X_train)
    val_size = int(n * val_ratio)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    print(f"  Tuning gamma on {len(X_tr)} train / {len(X_val)} val samples...")
    
    best_gamma = gamma_values[0]
    best_f1 = -1
    
    for gamma in gamma_values:
        # Use K-PA as reference (fast and stable)
        algo = KernelPassiveAggressive(gamma=gamma, max_sv=max_sv, C=1.0)
        algo.fit(X_tr, y_tr, epochs=epochs)
        y_pred = algo.predict(X_val)
        metrics = evaluate(y_val, y_pred)
        f1 = metrics['F1']
        
        if f1 > best_f1:
            best_f1 = f1
            best_gamma = gamma
        
        print(f"    gamma={gamma:.4f}: F1={f1:.4f}")
    
    print(f"  Best gamma: {best_gamma} (F1={best_f1:.4f})")
    return best_gamma


def run_repeated_experiments(n_repeats=10, gamma='tune', max_sv=500, epochs=3, test_ratio=0.3, datasets=None):
    """
    Run experiments with multiple random shuffles and average results.
    
    Args:
        n_repeats: Number of random shuffle repeats
        gamma: RBF gamma ('auto', 'tune', or float)
        max_sv: Maximum support vectors
        epochs: Training epochs
        test_ratio: Test set fraction (default 0.3 = 70/30 split)
        datasets: List of dataset names (None = all)
    """
    
    # Define algorithms factory
    def get_algorithms(gamma_val, max_sv):
        return {
            'K-Perceptron': lambda: KernelPerceptron(gamma=gamma_val, max_sv=max_sv),
            'K-PA': lambda: KernelPassiveAggressive(gamma=gamma_val, max_sv=max_sv, C=1.0),
            'K-AROW': lambda: KernelAROW(gamma=gamma_val, r=1.0, max_sv=min(max_sv, 500)),
            'K-OGL': lambda: KernelOGL(gamma=gamma_val, max_sv=max_sv),
            'K-RDA': lambda: KernelRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
            'K-SCW': lambda: KernelSCW(gamma=gamma_val, C=1.0, phi=0.5, max_sv=min(max_sv, 500)),
            'K-AdaRDA': lambda: KernelAdaRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
        }
    
    dataset_list = ALL_DATASETS
    if datasets:
        dataset_list = [(d, s) for d, s in ALL_DATASETS if d in datasets]
    
    all_results = []
    per_shuffle_results = []
    
    for dataset_name, split_type in dataset_list:
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name} ({n_repeats} repeats, {int((1-test_ratio)*100)}/{int(test_ratio*100)} split)")
        print("=" * 70)
        
        # Load raw data once
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        parquet_path = os.path.join(data_dir, f'{dataset_name}.parquet')
        
        if dataset_name == 'MNIST_combined':
            df = pd.read_parquet(parquet_path)
            y_raw = df['label'].values.astype(int)
            X_raw = df.drop('label', axis=1).values.astype(np.float64)
            # Binary: 0 vs 1
            mask = np.isin(y_raw, [0, 1])
            X_raw, y_raw = X_raw[mask], y_raw[mask]
            y_raw = np.where(y_raw == 0, -1, 1)
        else:
            df = pd.read_parquet(parquet_path)
            if 'Class' in df.columns:
                X_raw = df.drop(['Time', 'Class'], axis=1).values.astype(np.float64)
                y_raw = np.where(df['Class'].values == 0, -1, 1)
            else:
                X_raw = df.drop('label', axis=1).values.astype(np.float64)
                y_raw = np.where(df['label'].values == 0, -1, 1)
        
        print(f"  Total samples: {len(X_raw)}, +1: {(y_raw == 1).sum()} ({100*(y_raw == 1).mean():.1f}%)")
        
        # Accumulate metrics per algorithm
        algo_metrics = {name: {'Precision': [], 'Recall': [], 'F1': [], 'SVs': []} 
                        for name in get_algorithms(0.1, max_sv).keys()}
        
        for rep in range(n_repeats):
            np.random.seed(rep * 42)  # Different seed each repeat
            
            # Shuffle and split
            n = len(X_raw)
            indices = np.random.permutation(n)
            split_idx = int(n * (1 - test_ratio))
            
            X_train = X_raw[indices[:split_idx]]
            y_train = y_raw[indices[:split_idx]]
            X_test = X_raw[indices[split_idx:]]
            y_test = y_raw[indices[split_idx:]]
            
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Tune gamma on first repeat only
            if rep == 0:
                if gamma == 'tune':
                    gamma_val = tune_gamma(X_train, y_train, max_sv=min(100, max_sv), epochs=1)
                elif gamma == 'auto':
                    gamma_val = 1.0 / (X_train.shape[1] * X_train.var() + 1e-8)
                else:
                    gamma_val = float(gamma) if isinstance(gamma, str) else gamma
                print(f"  Using gamma = {gamma_val:.6f}")
            
            algorithms = get_algorithms(gamma_val, max_sv)
            
            for name, algo_factory in algorithms.items():
                algo = algo_factory()
                algo.fit(X_train, y_train, epochs=epochs)
                y_pred = algo.predict(X_test)
                metrics = evaluate(y_test, y_pred)
                n_sv = len(algo.support_vectors) if algo.support_vectors is not None else 0
                
                algo_metrics[name]['Precision'].append(metrics['Precision'])
                algo_metrics[name]['Recall'].append(metrics['Recall'])
                algo_metrics[name]['F1'].append(metrics['F1'])
                algo_metrics[name]['SVs'].append(n_sv)
            
            print(f"  Repeat {rep+1}/{n_repeats} done", end='\r')
        
        print()
        
        # Print averaged results
        print(f"\n  {'Algorithm':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'SVs':>8}")
        print("  " + "-" * 55)
        
        for name in algo_metrics:
            p_mean = np.mean(algo_metrics[name]['Precision'])
            p_std = np.std(algo_metrics[name]['Precision'])
            r_mean = np.mean(algo_metrics[name]['Recall'])
            r_std = np.std(algo_metrics[name]['Recall'])
            f1_mean = np.mean(algo_metrics[name]['F1'])
            f1_std = np.std(algo_metrics[name]['F1'])
            sv_mean = np.mean(algo_metrics[name]['SVs'])
            
            print(f"  {name:<15} {p_mean:.4f}±{p_std:.2f} {r_mean:.4f}±{r_std:.2f} {f1_mean:.4f}±{f1_std:.2f} {sv_mean:>6.0f}")
            
            all_results.append({
                'Dataset': dataset_name,
                'Algorithm': name,
                'Gamma': gamma_val,
                'Precision_mean': p_mean,
                'Precision_std': p_std,
                'Recall_mean': r_mean,
                'Recall_std': r_std,
                'F1_mean': f1_mean,
                'F1_std': f1_std,
                'SVs_mean': sv_mean,
                'N_repeats': n_repeats,
            })
            
            # Store per-shuffle F1 scores for line plots
            for rep_idx, f1_val in enumerate(algo_metrics[name]['F1']):
                per_shuffle_results.append({
                    'Dataset': dataset_name,
                    'Algorithm': name,
                    'Shuffle': rep_idx + 1,
                    'F1': f1_val,
                })
    
    # Save results
    df_results = pd.DataFrame(all_results)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'kernel_repeated_results.csv')
    df_results.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {output_path}")
    
    # Save per-shuffle F1 scores
    df_per_shuffle = pd.DataFrame(per_shuffle_results)
    per_shuffle_path = os.path.join(results_dir, 'kernel_per_shuffle_f1.csv')
    df_per_shuffle.to_csv(per_shuffle_path, index=False, float_format='%.4f')
    print(f"Per-shuffle F1 saved to: {per_shuffle_path}")
    
    # Generate F1 line plots for each dataset
    algo_colors = {
        'K-OGL': '#e41a1c',      # Red - highlighted
        'K-Perceptron': '#377eb8',
        'K-PA': '#4daf4a',
        'K-AROW': '#984ea3',
        'K-RDA': '#ff7f00',
        'K-SCW': '#a65628',
        'K-AdaRDA': '#f781bf',
    }
    
    for dataset_name, _ in dataset_list:
        df_ds = df_per_shuffle[df_per_shuffle['Dataset'] == dataset_name]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = df_ds['Algorithm'].unique()
        for algo in algorithms:
            df_algo = df_ds[df_ds['Algorithm'] == algo].sort_values('Shuffle')
            color = algo_colors.get(algo, 'gray')
            linewidth = 3 if algo == 'K-OGL' else 1.5
            alpha = 1.0 if algo == 'K-OGL' else 0.7
            zorder = 10 if algo == 'K-OGL' else 1
            
            # Compute running average
            f1_values = df_algo['F1'].values
            running_avg = np.cumsum(f1_values) / np.arange(1, len(f1_values) + 1)
            
            ax.plot(df_algo['Shuffle'], running_avg, 
                   label=algo, color=color, linewidth=linewidth, 
                   alpha=alpha, zorder=zorder)
        
        ax.set_xlabel('Shuffle', fontsize=12)
        ax.set_ylabel('Running Average F1 Score', fontsize=12)
        ax.set_title(f'Running Average F1 Score - {dataset_name}\n(K-OGL highlighted)', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, n_repeats)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'f1_running_avg_{dataset_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    return df_results


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

# All datasets with their split types:
# All use random shuffle split for repeated experiments
ALL_DATASETS = [
    ('HeartDisease_UCI', 'random'),  # Heart Disease: random split (binary: disease vs no disease)
    ('HeartFailure_UCI', 'random'),  # Heart Failure: random split (binary: death vs survival)
]


def run_experiments(gamma='tune', max_sv=1000, epochs=3, subsample=10000, max_train=15000, datasets=None):
    """
    Run all kernel algorithms on all datasets.
    
    Args:
        gamma: RBF kernel parameter ('auto' computes from data)
        max_sv: Maximum support vectors to keep
        epochs: Training epochs
        subsample: Subsample each dataset for speed (None = full)
        max_train: Max training samples to avoid memory issues with full kernel matrix
        datasets: List of dataset names to run (None = all datasets)
    """
    
    np.random.seed(42)
    
    results = []
    
    # Define algorithms
    def get_algorithms(gamma_val, max_sv):
        return {
            'K-Perceptron': lambda: KernelPerceptron(gamma=gamma_val, max_sv=max_sv),
            'K-PA': lambda: KernelPassiveAggressive(gamma=gamma_val, max_sv=max_sv, C=1.0),
            'K-AROW': lambda: KernelAROW(gamma=gamma_val, r=1.0, max_sv=min(max_sv, 500)),
            'K-OGL': lambda: KernelOGL(gamma=gamma_val, max_sv=max_sv),
            'K-RDA': lambda: KernelRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
            'K-SCW': lambda: KernelSCW(gamma=gamma_val, C=1.0, phi=0.5, max_sv=min(max_sv, 500)),
            'K-AdaRDA': lambda: KernelAdaRDA(gamma=gamma_val, lambda_param=0.01, max_sv=min(max_sv, 500)),
        }
    
    # Filter datasets if specified
    dataset_list = ALL_DATASETS
    if datasets:
        dataset_list = [(d, s) for d, s in ALL_DATASETS if d in datasets]
    
    for idx, (dataset_name, split_type) in enumerate(dataset_list, 1):
        print("\n" + "=" * 70)
        print(f"DATASET {idx}/{len(dataset_list)}: {dataset_name} ({split_type.upper()} split)")
        print("=" * 70)
        
        try:
            time_split = (split_type == 'time')
            
            # Special handling for MNIST, Iris, and Heart Disease (binary classification)
            if dataset_name == 'MNIST_combined':
                X_train, y_train, X_test, y_test = load_mnist(
                    random_split=True,
                    binary_class=(0, 1)
                )
            elif dataset_name == 'Iris_UCI':
                X_train, y_train, X_test, y_test = load_iris(
                    time_split=True,
                    binary_class=('Iris-setosa', 'Iris-versicolor')
                )
            elif dataset_name == 'HeartDisease_UCI':
                X_train, y_train, X_test, y_test = load_heart_disease(
                    random_split=True
                )
            else:
                X_train, y_train, X_test, y_test = load_dataset(
                    dataset_name,
                    time_split=time_split,
                    subsample=subsample
                )
            
            # Limit training size to avoid memory issues with full kernel matrix
            if max_train and len(X_train) > max_train:
                print(f"  Limiting training to {max_train} samples (was {len(X_train)})")
                if time_split:
                    # Keep last max_train to preserve temporal order
                    X_train = X_train[-max_train:]
                    y_train = y_train[-max_train:]
                else:
                    # Random selection since already shuffled
                    indices = np.random.choice(len(X_train), max_train, replace=False)
                    X_train = X_train[indices]
                    y_train = y_train[indices]
            
            # Auto gamma: 1 / (n_features * variance)
            # Tune gamma: grid search on validation set
            if gamma == 'auto':
                gamma_val = 1.0 / (X_train.shape[1] * X_train.var() + 1e-8)
            elif gamma == 'tune':
                gamma_val = tune_gamma(X_train, y_train, max_sv=min(200, max_sv), epochs=1)
            else:
                gamma_val = float(gamma) if isinstance(gamma, str) else gamma
            print(f"  Using gamma = {gamma_val:.6f}")
            
            algorithms = get_algorithms(gamma_val, max_sv)
            
            for name, algo_factory in algorithms.items():
                print(f"\n  Training {name}...", end=" ", flush=True)
                
                try:
                    algo = algo_factory()
                    start = time.time()
                    algo.fit(X_train, y_train, epochs=epochs)
                    train_time = time.time() - start
                    
                    y_pred = algo.predict(X_test)
                    metrics = evaluate(y_test, y_pred)
                    
                    n_sv = len(algo.support_vectors) if algo.support_vectors is not None else 0
                    
                    print(f"Done ({train_time:.1f}s, {n_sv} SVs)")
                    print(f"    Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
                    
                    results.append({
                        'Dataset': dataset_name,
                        'Split': split_type,
                        'Algorithm': name,
                        'Gamma': gamma_val,
                        'SVs': n_sv,
                        'TrainTime': train_time,
                        **metrics
                    })
                except Exception as e:
                    print(f"Error: {e}")
                    results.append({
                        'Dataset': dataset_name,
                        'Split': split_type,
                        'Algorithm': name,
                        'Gamma': gamma_val,
                        'SVs': 0,
                        'TrainTime': 0,
                        'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0, 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0
                    })
                
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        summary_cols = ['Dataset', 'Split', 'Algorithm', 'Precision', 'Recall', 'F1', 'SVs', 'TrainTime']
        print(df[summary_cols].round(4).to_string(index=False))
        
        # Save results to results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'kernel_experiment_results.csv')
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nResults saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vectorized Kernel Classification Experiments')
    parser.add_argument('--gamma', type=str, default='tune', help='RBF gamma (float, "auto", or "tune")')
    parser.add_argument('--max-sv', type=int, default=1000, help='Max support vectors')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--repeats', type=int, default=100, help='Number of random shuffle repeats')
    parser.add_argument('--test-ratio', type=float, default=0.3, help='Test set fraction (default: 0.3 = 70/30 split)')
    
    args = parser.parse_args()
    
    gamma = args.gamma if args.gamma in ('auto', 'tune') else float(args.gamma)
    
    # Run experiments with F1 per-shuffle saved and plots generated
    run_repeated_experiments(
        n_repeats=args.repeats,
        gamma=gamma,
        max_sv=args.max_sv,
        epochs=args.epochs,
        test_ratio=args.test_ratio,
        datasets=args.datasets
    )
