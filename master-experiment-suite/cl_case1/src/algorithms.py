import numpy as np
from scipy.stats import norm
import copy
from sklearn.metrics import confusion_matrix, accuracy_score

class PassiveAggressive:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if y_true * x.dot(self.weights) < 1:
            l2_norm_sq = x.dot(x)
            if l2_norm_sq > 0:
                eta = (1 - y_true * x.dot(self.weights)) / l2_norm_sq
                self.weights += eta * y_true * x

class Perceptron:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if self.predict(x) != y_true: self.weights += y_true * x

class GradientLearning:
    def __init__(self, n_features): self.weights = np.zeros(n_features)
    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1
    def partial_fit(self, x, y_true):
        if y_true * x.dot(self.weights) < 1: self.weights += y_true * x


class AROW:
    """
    Adaptive Regularization of Weight Vectors (AROW).
    """
    def __init__(self, n_features, r):
        """
        Args:
            n_features (int): The number of features in the dataset.
            r (float): Regularization parameter.
        """
        self.r = r
        # The mean vector 'u' is the user-facing 'weights'
        self.weights = np.zeros(n_features)
        # The covariance matrix is an internal state of the model
        self.Sigma = np.identity(n_features)

    def predict(self, x):
        """
        Makes a prediction for a single data sample.
        """
        prediction = np.sign(x.dot(self.weights))
        # Ensure prediction is always -1 or 1, not 0
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        """
        Processes a single data sample (x, y_true) to update the model's weights.
        """
        # Calculate hinge loss and confidence for the current sample
        lt = max(0, 1 - y_true * x.dot(self.weights))
        vt = x.T.dot(self.Sigma).dot(x)
        
        # Only update the model's state if there is a loss
        if lt > 0:
            # Calculate the update coefficients, beta_t and alpha_t
            beta_t = 1 / (vt + self.r) if (vt + self.r) > 0 else 0.0
            alpha_t = lt * beta_t
           
            # Update the internal state (weights and covariance matrix)
            self.weights += alpha_t * y_true * self.Sigma.dot(x)
            self.Sigma -= beta_t * self.Sigma.dot(np.outer(x, x)).dot(self.Sigma)


class RDA:
    """Regularized Dual Averaging (RDA)."""
    def __init__(self, n_features, lambda_param, gamma_param):
        self.weights = np.zeros(n_features)
        self.g = np.zeros(n_features)
        self.t = 0
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        self.t += 1
        lt = max(0, 1 - y_true * x.dot(self.weights))
        gt = -y_true * x if lt > 0 else np.zeros_like(x)
        self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
        update_mask = np.abs(self.g) > self.lambda_param
        self.weights.fill(0)
        self.weights[update_mask] = -(np.sqrt(self.t) / self.gamma_param) * \
                                   (self.g[update_mask] - self.lambda_param * np.sign(self.g[update_mask]))

class SCW:
    """Soft Confidence-Weighted (SCW)."""
    def __init__(self, n_features, C, eta):
        self.phi = norm.ppf(eta)
        self.C = C
        self.weights = np.zeros(n_features)
        self.Sigma = np.identity(n_features)

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        vt = x.T.dot(self.Sigma).dot(x)
        mt = y_true * x.dot(self.weights)
        lt = max(0, self.phi * np.sqrt(vt) - mt)
        if lt > 0:
            pa = 1 + (self.phi**2) / 2
            xi = 1 + self.phi**2
            sqrt_term = max(0, (mt**2 * self.phi**4 / 4) + (vt * self.phi**2 * xi))
            alpha_t = min(self.C, max(0, (1 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term))))
            sqrt_ut_term = max(0, (alpha_t**2 * vt**2 * self.phi**2) + (4 * vt))
            ut = 0.25 * (-alpha_t * vt * self.phi + np.sqrt(sqrt_ut_term))**2
            beta_t = (alpha_t * self.phi) / (np.sqrt(ut) + vt * alpha_t * self.phi + 1e-8)
            self.weights += alpha_t * y_true * self.Sigma.dot(x)
            self.Sigma -= beta_t * self.Sigma.dot(np.outer(x, x)).dot(self.Sigma)

class AdaRDA:
    """Adaptive Regularized Dual Averaging (AdaRDA)."""
    def __init__(self, n_features, lambda_param, eta_param, delta_param):
        self.weights = np.zeros(n_features)
        self.g, self.g1t = np.zeros(n_features), np.zeros(n_features)
        self.t = 0
        self.lambda_param, self.eta_param, self.delta_param = lambda_param, eta_param, delta_param

    def predict(self, x):
        prediction = np.sign(x.dot(self.weights))
        return prediction if prediction != 0 else 1

    def partial_fit(self, x, y_true):
        self.t += 1
        lt = max(0, 1 - y_true * x.dot(self.weights))
        gt = -y_true * x if lt > 0 else np.zeros_like(x)
        self.g = ((self.t - 1) / self.t) * self.g + (1 / self.t) * gt
        self.g1t += gt**2
        Ht = self.delta_param + np.sqrt(self.g1t)
        update_mask = np.abs(self.g) > self.lambda_param
        self.weights.fill(0)
        self.weights[update_mask] = np.sign(-self.g[update_mask]) * self.eta_param * self.t / (Ht[update_mask] + 1e-8)


class OnlineToBatchTrainer:
    """
    OnlineToBatch protocol implementation that minimizes False Negatives
    by optimizing for recall (sensitivity) on validation set.
    """
    def __init__(self, algorithm_class, algorithm_params, n_features, epochs=10, 
                 optimize_metric='recall'):
        """
        Args:
            algorithm_class: The algorithm class (e.g., PassiveAggressive)
            algorithm_params: Parameters for algorithm initialization
            n_features: Number of features
            epochs: Number of training epochs
            optimize_metric: Metric to optimize ('recall', 'f1', 'fnr_min')
        """
        self.algorithm_class = algorithm_class
        self.algorithm_params = algorithm_params
        self.n_features = n_features
        self.epochs = epochs
        self.optimize_metric = optimize_metric
        self.best_weights = None
        self.best_score = -1.0
        self.best_state = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Implements the OnlineToBatch protocol
        """
        # Initialize algorithm
        algo = self.algorithm_class(n_features=self.n_features, **self.algorithm_params)
        
        # Store initial state
        initial_state = self._save_algorithm_state(algo)
        self.best_state = copy.deepcopy(initial_state)
        self.best_score = -1.0
        
        for epoch in range(self.epochs):
            # (A) Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # (B) Train on shuffled data
            if epoch > 0:
                # Restore weights from previous epoch
                self._restore_algorithm_state(algo, epoch_state)
            
            # Train on all instances
            for x, y in zip(X_shuffled, y_shuffled):
                algo.partial_fit(x, y)
            
            # Store current epoch state
            epoch_state = self._save_algorithm_state(algo)
            
            # (C) Evaluate on validation set
            y_pred = np.array([algo.predict(x) for x in X_val])
            current_score = self._calculate_metric(y_val, y_pred)
            
            # (D) Save optimal parameters
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_state = copy.deepcopy(epoch_state)
        
        # Restore best state
        final_algo = self.algorithm_class(n_features=self.n_features, **self.algorithm_params)
        self._restore_algorithm_state(final_algo, self.best_state)
        return final_algo
    
    def _save_algorithm_state(self, algo):
        """Save algorithm state for restoration"""
        state = {'weights': algo.weights.copy()}
        
        # Save additional state for specific algorithms
        if hasattr(algo, 'Sigma'):
            state['Sigma'] = algo.Sigma.copy()
        if hasattr(algo, 'g'):
            state['g'] = algo.g.copy()
        if hasattr(algo, 'g1t'):
            state['g1t'] = algo.g1t.copy()
        if hasattr(algo, 't'):
            state['t'] = algo.t
        if hasattr(algo, 'phi'):
            state['phi'] = algo.phi
            
        return state
    
    def _restore_algorithm_state(self, algo, state):
        """Restore algorithm state"""
        algo.weights = state['weights'].copy()
        
        if 'Sigma' in state and hasattr(algo, 'Sigma'):
            algo.Sigma = state['Sigma'].copy()
        if 'g' in state and hasattr(algo, 'g'):
            algo.g = state['g'].copy()
        if 'g1t' in state and hasattr(algo, 'g1t'):
            algo.g1t = state['g1t'].copy()
        if 't' in state and hasattr(algo, 't'):
            algo.t = state['t']
        if 'phi' in state and hasattr(algo, 'phi'):
            algo.phi = state['phi']
    
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
        for name, algo in self.algorithms:
            y_pred = np.array([algo.predict(x) for x in X_val])
            
            try:
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[-1, 1]).ravel()
            except ValueError:
                self.weights[name] = 0.0
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
    
    def predict(self, x):
        """Make weighted majority prediction"""
        weighted_votes = 0.0
        for name, algo in self.algorithms:
            prediction = algo.predict(x)
            weighted_votes += self.weights.get(name, 0.0) * prediction
        
        return 1 if weighted_votes > 0 else -1
    
    def predict_batch(self, X):
        """Make predictions for a batch of samples"""
        return np.array([self.predict(x) for x in X])
