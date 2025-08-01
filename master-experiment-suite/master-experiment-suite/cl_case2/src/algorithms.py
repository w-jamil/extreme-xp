import numpy as np
from scipy.stats import norm

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
