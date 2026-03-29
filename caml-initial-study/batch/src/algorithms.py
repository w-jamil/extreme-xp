#!/usr/bin/env python3
"""
Online Learning Algorithms for Cybersecurity Experiments
Essential algorithms with numerical stability
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

class OGL:
    """Online Gradient Learning with L2 regularization"""
    def __init__(self, learning_rate=0.01, l2_reg=0.01, random_state=42):
        self.learning_rate = max(learning_rate, 1e-6)
        self.l2_reg = max(l2_reg, 1e-8)
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features).astype(np.float64)
        self.bias = 0.0
        
        for i in range(len(X)):
            x_i = X[i].astype(np.float64)
            y_i = float(y[i])
            
            if not np.isfinite(x_i).all():
                continue
                
            score = np.dot(x_i, self.weights) + self.bias
            prediction = 1 if score >= 0 else -1
            
            if prediction != y_i:
                gradient = -y_i * x_i + self.l2_reg * self.weights
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm > 10.0:
                    gradient *= 10.0 / gradient_norm
                
                if np.isfinite(gradient).all():
                    self.weights -= self.learning_rate * gradient
                    self.bias -= self.learning_rate * (-y_i)
                    
                    # Bound weights
                    weight_norm = np.linalg.norm(self.weights)
                    if weight_norm > 50.0:
                        self.weights *= 50.0 / weight_norm
        
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)
    
    def partial_fit(self, X, y):
        """Online update for a single sample or batch"""
        for i in range(len(X)):
            x_i = X[i].astype(np.float64)
            y_i = float(y[i])
            
            if not np.isfinite(x_i).all():
                continue
                
            score = np.dot(x_i, self.weights) + self.bias
            prediction = 1 if score >= 0 else -1
            
            if prediction != y_i:
                gradient = -y_i * x_i + self.l2_reg * self.weights
                self.weights -= self.learning_rate * gradient
                self.bias -= self.learning_rate * (-y_i)
        return self

class AROW:
    """Adaptive Regularization of Weight Vectors"""
    def __init__(self, r=0.1, random_state=42):
        self.r = max(r, 1e-8)
        self.random_state = random_state
        self.weights = None
        self.sigma = None
        self.bias = 0.0
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.sigma = np.eye(n_features, dtype=np.float64) * 1.0
        self.bias = 0.0
        
        for i in range(len(X)):
            x_i = X[i].astype(np.float64)
            y_i = float(y[i])
            
            if not np.isfinite(x_i).all():
                continue
                
            sigma_x = np.dot(self.sigma, x_i)
            confidence = np.dot(x_i, sigma_x)
            confidence = max(confidence, 1e-10)
            
            score = np.dot(x_i, self.weights) + self.bias
            margin = y_i * score
            
            if margin < 1:
                beta = 1.0 / (confidence + self.r)
                alpha = max(0, min(10.0, (1 - margin) * beta))
                
                if alpha > 1e-10:
                    self.weights += alpha * y_i * sigma_x
                    self.bias += alpha * y_i
                    
                    # Update covariance
                    if beta > 0:
                        outer_product = np.outer(sigma_x, sigma_x)
                        self.sigma -= beta * outer_product
        
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)

class RDA:
    """Regularized Dual Averaging"""
    def __init__(self, learning_rate=0.1, l1_reg=0.01, random_state=42):
        self.learning_rate = max(learning_rate, 1e-6)
        self.l1_reg = max(l1_reg, 1e-8)
        self.random_state = random_state
        self.weights = None
        self.avg_gradient = None
        self.bias = 0.0
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.avg_gradient = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        
        for t in range(len(X)):
            x_i = X[t].astype(np.float64)
            y_i = float(y[t])
            
            if not np.isfinite(x_i).all():
                continue
                
            score = np.dot(x_i, self.weights) + self.bias
            prediction = 1 if score >= 0 else -1
            
            if prediction != y_i:
                gradient = -y_i * x_i
                self.avg_gradient = ((t) * self.avg_gradient + gradient) / (t + 1)
                
                # RDA weight update with L1 regularization
                learning_rate_t = self.learning_rate / np.sqrt(t + 1)
                for j in range(n_features):
                    if abs(self.avg_gradient[j]) > self.l1_reg:
                        self.weights[j] = -learning_rate_t * (
                            self.avg_gradient[j] - np.sign(self.avg_gradient[j]) * self.l1_reg
                        )
                    else:
                        self.weights[j] = 0.0
                
                self.bias += learning_rate_t * y_i
        
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)

class SCW:
    """Soft Confidence Weighted Learning"""
    def __init__(self, phi=1.0, C=1.0, random_state=42):
        self.phi = max(phi, 0.1)
        self.C = max(C, 1e-6)
        self.random_state = random_state
        self.weights = None
        self.sigma = None
        self.bias = 0.0
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.sigma = np.eye(n_features, dtype=np.float64) * 0.1
        self.bias = 0.0
        
        for i in range(len(X)):
            x_i = X[i].astype(np.float64)
            y_i = float(y[i])
            
            if not np.isfinite(x_i).all():
                continue
                
            sigma_x = np.dot(self.sigma, x_i)
            confidence = np.dot(x_i, sigma_x)
            confidence = np.clip(confidence, 1e-10, 100.0)
            
            score = np.dot(x_i, self.weights) + self.bias
            margin = y_i * score
            
            if margin < self.phi:
                v = confidence + 1.0 / (2 * self.C)
                alpha = min(self.C, max(0, (self.phi - margin) / v))
                
                if alpha > 1e-10:
                    self.weights += alpha * y_i * sigma_x
                    self.bias += alpha * y_i
                    
                    # Update covariance
                    beta = alpha * confidence
                    if beta < 1.0:
                        update = beta / (1 + beta) * np.outer(sigma_x, sigma_x)
                        self.sigma -= update
        
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)

class AdaRDA:
    """Adaptive Regularized Dual Averaging"""
    def __init__(self, learning_rate=0.01, l1_reg=0.001, random_state=42):
        self.learning_rate = max(learning_rate, 1e-6)
        self.l1_reg = max(l1_reg, 1e-8)
        self.random_state = random_state
        self.weights = None
        self.avg_gradient = None
        self.sum_gradient_sq = None
        self.bias = 0.0
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.avg_gradient = np.zeros(n_features, dtype=np.float64)
        self.sum_gradient_sq = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        
        for t in range(len(X)):
            x_i = X[t].astype(np.float64)
            y_i = float(y[t])
            
            if not np.isfinite(x_i).all():
                continue
                
            score = np.dot(x_i, self.weights) + self.bias
            prediction = 1 if score >= 0 else -1
            
            if prediction != y_i:
                gradient = -y_i * x_i
                
                # Update running averages
                self.avg_gradient = ((t) * self.avg_gradient + gradient) / (t + 1)
                self.sum_gradient_sq += gradient ** 2
                
                # Adaptive learning rates
                adaptive_lr = self.learning_rate / (np.sqrt(self.sum_gradient_sq + 1e-8))
                
                # AdaRDA update with adaptive regularization
                for j in range(n_features):
                    if abs(self.avg_gradient[j]) > self.l1_reg * adaptive_lr[j]:
                        self.weights[j] = -adaptive_lr[j] * (
                            self.avg_gradient[j] - np.sign(self.avg_gradient[j]) * self.l1_reg
                        )
                    else:
                        self.weights[j] = 0.0
                
                self.bias += np.mean(adaptive_lr) * y_i
        
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)

class RCL_BCE:
    """Regularized Competitive Learning with Binary Cross Entropy"""
    def __init__(self, learning_rate=0.01, l2_reg=0.01, random_state=42):
        self.learning_rate = max(learning_rate, 1e-6)
        self.l2_reg = max(l2_reg, 1e-8)
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        
    def fit(self, X, y):
        """Use logistic regression as baseline"""
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(
            C=1.0/self.l2_reg,
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'
        )
        lr.fit(X, y)
        
        self.weights = lr.coef_[0].astype(np.float64)
        self.bias = float(lr.intercept_[0])
        return self
    
    def predict(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X):
        scores = np.dot(X.astype(np.float64), self.weights) + self.bias
        return np.where(np.isfinite(scores), scores, 0.0)
