import numpy as np

class Perceptron:
    """
    Perceptron algorithm
    """
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)

    def predict(self, x):
        return np.sign(x @ self.weights)

    def partial_fit(self, x, y_true):
        prediction = self.predict(x)
        if prediction != y_true:
            self.weights += y_true * x

class PassiveAggressive:
    """
    Passive-Aggressive algorithm
    """
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        
    def predict(self, x):
        return np.sign(x @ self.weights)
    
    def partial_fit(self, x, y_true):
        prediction = self.predict(x)
        loss = max(0, 1 - y_true * prediction)
        l2_norm_sq = x @ x
        
        # Avoid division by zero
        if l2_norm_sq > 0:
            eta = loss / l2_norm_sq
            self.weights += eta * y_true * x

class GradientLearning:
    """
    A Gradient Descent-based learning algorithm.
    """
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)

    def predict(self, x):
        return np.sign(x @ self.weights)

    def partial_fit(self, x, y_true):
        prediction = self.predict(x)
        loss = y_true - prediction
        l2_norm = np.sqrt(x @ x) + 1e-8
        eta = loss / l2_norm
        self.weights += eta * y_true * x

class WPA:
    """
    A modified Passive-Aggressive algorith.
    """
    def __init__(self, n_features, rho=0.01):
        self.weights = np.zeros(n_features)
        self.rho = rho
        
    def predict(self, x):
        return np.sign(x @ self.weights)
    
    def partial_fit(self, x, y_true):
        prediction = self.predict(x)
        loss = max(0, 1 - y_true * prediction)
        l2_norm_sq = x @ x
        
        # Avoid division by zero
        if l2_norm_sq > 0:
            eta = loss / l2_norm_sq
            self.weights += eta * y_true * x * self.rho