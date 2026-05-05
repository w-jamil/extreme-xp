#!/usr/bin/env python3
"""
algos.py — Online regression algorithms for delayed/batched feedback study.

Four algorithm families, all Numba-accelerated:
  CIRR      — Competitive Iterated Ridge Regression (linear, immediate feedback base)
  K-CIRR    — Kernelised CIRR via RBF kernel (non-linear feature space)
  Q-CIRR    — Quantum-enhanced CIRR (HHL-simulated matrix inversion speedup)
  QK-CIRR   — Quantum + Kernelised CIRR

Each family exposes:
  · An @njit inner-loop kernel (for prequential / delayed / batched runs)
  · A Python wrapper class with predict() / update() interface


Information-constraint runners (delayed + batched) are in experiments.py.
"""

import numpy as np
from numba import njit

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — RBF KERNEL UTILITIES
# ══════════════════════════════════════════════════════════════════════

def rbf_kernel(x1, x2, gamma=1.0):
    """RBF (Gaussian) kernel between two vectors.  Returns scalar in (0, 1]."""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))


def rbf_kernel_matrix(X1, X2, gamma):
    """Build full pairwise RBF kernel matrix (n1, n2) from two sample sets."""
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    cross = X1 @ X2.T
    dist_sq = np.maximum(X1_sq + X2_sq.T - 2 * cross, 0.0)
    return np.exp(-gamma * dist_sq)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — NUMBA JIT INNER LOOPS
#
#  Convention: every kernel returns (predictions, cumulative_squared_loss).
#  State (A, b, w / alpha) is updated IN-PLACE.
#  For delayed / batched variants the outer Python loop is used because
#  the feedback queue logic is irregular; the linear algebra hotpaths
#  are still JIT-compiled.
# ══════════════════════════════════════════════════════════════════════

# ---------- 2a. CIRR immediate-feedback inner loop ----------
@njit(cache=True)
def _cirr_kernel(X, y, a):
    """CIRR prequential inner loop — adaptive D-matrix ridge (competitive iterated).

    $D_{i,j} = \sqrt{|w_i|} \cdot \sqrt{|w_j|}$ weights the precision matrix adaptively.
    Prediction uses $\tilde{A}_t = A_t + x_t x_t^\top$ (one-step lookahead on precision).
    $w$ is initialised to ones so $D$ starts as the identity.

    Parameters
    ----------
    X : $X \in \mathbb{R}^{T \times n}$
    y : $y \in \mathbb{R}^T$
    a : float  — regularisation parameter

    Returns
    -------
    predictions    : (T,) float64
    squared_losses : (T,) float64
    """
    T, n = X.shape
    A = np.zeros((n, n))
    b = np.zeros(n)
    w = np.ones(n)          # initialise to ones so D = I at start
    predictions = np.zeros(T)
    sq_losses = np.zeros(T)

    for t in range(T):
        x = X[t]

        # --- Compute D from current w ---
        d_sqrt = np.empty(n)
        for i in range(n):
            d_sqrt[i] = np.sqrt(np.abs(w[i]))
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = d_sqrt[i] * d_sqrt[j]

        # --- Predict: M = a*I + D * (A + x x^T), pw = (D * inv(M)) @ b ---
        M = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = D[i, j] * (A[i, j] + x[i] * x[j])
            M[i, i] += a
        InvM = np.linalg.inv(M)
        pred = 0.0
        for i in range(n):
            pw_i = 0.0
            for j in range(n):
                pw_i += D[i, j] * InvM[i, j] * b[j]
            pred += pw_i * x[i]
        predictions[t] = pred
        sq_losses[t] = (pred - y[t]) ** 2

        # --- Update A, b ---
        for i in range(n):
            b[i] += y[t] * x[i]
            for j in range(n):
                A[i, j] += x[i] * x[j]

        # --- Update w: M = a*I + D * A (same D as before update) ---
        for i in range(n):
            for j in range(n):
                M[i, j] = D[i, j] * A[i, j]
            M[i, i] += a
        InvM = np.linalg.inv(M)
        for i in range(n):
            wi = 0.0
            for j in range(n):
                wi += D[i, j] * InvM[i, j] * b[j]
            w[i] = wi

    return predictions, sq_losses


# ---------- 2b. Q-CIRR HHL simulation inner loop ----------
@njit(cache=True)
def _qcirr_kernel(X, y, a, epsilon):
    """Q-CIRR prequential inner loop — CIRR D-matrix + HHL quantum noise.

    Identical to _cirr_kernel but adds deterministic perturbation
    $\epsilon \sin(tn + j + 1)$ to each weight component after the solve,
    modelling quantum measurement error from the HHL algorithm.

    Parameters
    ----------
    X       : $X \in \mathbb{R}^{T \times n}$
    y       : $y \in \mathbb{R}^T$
    a       : float  — regularisation parameter
    epsilon : float  — quantum measurement error scale $\epsilon$ (e.g. 0.01)

    Returns
    -------
    predictions    : (T,) float64
    squared_losses : (T,) float64
    """
    T, n = X.shape
    A = np.zeros((n, n))
    b = np.zeros(n)
    w = np.ones(n)          # initialise to ones
    predictions = np.zeros(T)
    sq_losses = np.zeros(T)

    for t in range(T):
        x = X[t]

        # --- Compute D from current w ---
        d_sqrt = np.empty(n)
        for i in range(n):
            d_sqrt[i] = np.sqrt(np.abs(w[i]))
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = d_sqrt[i] * d_sqrt[j]

        # --- Predict ---
        M = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = D[i, j] * (A[i, j] + x[i] * x[j])
            M[i, i] += a
        InvM = np.linalg.inv(M)
        pred = 0.0
        for i in range(n):
            pw_i = 0.0
            for j in range(n):
                pw_i += D[i, j] * InvM[i, j] * b[j]
            pred += pw_i * x[i]
        predictions[t] = pred
        sq_losses[t] = (pred - y[t]) ** 2

        # --- Update A, b ---
        for i in range(n):
            b[i] += y[t] * x[i]
            for j in range(n):
                A[i, j] += x[i] * x[j]

        # --- Update w with quantum noise ---
        for i in range(n):
            for j in range(n):
                M[i, j] = D[i, j] * A[i, j]
            M[i, i] += a
        InvM = np.linalg.inv(M)
        for i in range(n):
            wi = 0.0
            for j in range(n):
                wi += D[i, j] * InvM[i, j] * b[j]
            noise = epsilon * np.sin(float(t * n + i + 1))
            w[i] = wi + noise

    return predictions, sq_losses


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — PYTHON WRAPPER CLASSES
#
#  Used by the delayed / batched runners which need per-sample control.
#  All heavy linear algebra is still done via numpy (fast BLAS) or the
#  Numba kernels where possible.
# ══════════════════════════════════════════════════════════════════════

class CIRR:
    """Competitive Iterated Ridge Regression — adaptive D-matrix online regression.

    At each step, $D_{i,j} = \sqrt{|w_i|} \cdot \sqrt{|w_j|}$ weights the precision matrix
    so that directions with larger current weights get less regularisation,
    implementing a self-normalising adaptive ridge (competitive iterated scheme).
    Prediction uses a one-step lookahead: $\tilde{A}_t = A_t + x_t x_t^\top$.
    """

    name = "CIRR"

    def __init__(self, n_features, a=None, total_samples=None):
        self.n = n_features
        self.a = (1.0 / total_samples) if (total_samples and total_samples > 0) else 1.0
        if a is not None:
            self.a = a
        self.A = np.zeros((n_features, n_features))
        self.b = np.zeros(n_features)
        self.w = np.ones(n_features)    # ones  D = I at initialisation
        self._aI = self.a * np.eye(n_features)

    def _D(self):
        d_sqrt = np.sqrt(np.abs(self.w))
        return np.outer(d_sqrt, d_sqrt)

    def predict(self, x):
        D = self._D()
        temp_A = self.A + np.outer(x, x)
        M = self._aI + D * temp_A
        try:
            InvM = np.linalg.inv(M)
            pw = (D * InvM) @ self.b
        except np.linalg.LinAlgError:
            pw = self.w
        return float(np.dot(pw, x))

    def update(self, x, y):
        D = self._D()           # use D from current w (before this update)
        self.A += np.outer(x, x)
        self.b += y * x
        M = self._aI + D * self.A
        try:
            InvM = np.linalg.inv(M)
            self.w = (D * InvM) @ self.b
        except np.linalg.LinAlgError:
            pass


class KCIRR:
    """Kernelised CIRR (K-CIRR) via Random Fourier Features + CIRR D-matrix.

    The RBF kernel is approximated by an explicit D-dimensional feature map
    (Rahimi & Recht, 2007):

        $\varphi(x) = \sqrt{2/D}\cos(Wx + b)$,  $W \sim \mathcal{N}(0, \gamma I)$,  $b \sim \mathcal{U}(0, 2\pi)$

    so that $\mathbb{E}[\varphi(x)^\top \varphi(x')] = \exp(-\gamma \|x - x'\|^2)$.

    The transformed vector φ(x) is then fed to the *identical* CIRR
    D-matrix adaptive ridge algorithm — same predict, same update, same
    diagonal initialisation.  The only difference from CIRR is the
    feature representation.
    """

    name = "K-CIRR"

    def __init__(self, n_features, gamma=None, a=None, total_samples=None,
                 n_components=50):
        gamma = gamma if gamma is not None else 1.0 / (2.0 * n_features)
        self.a = (1.0 / total_samples) if (total_samples and total_samples > 0) else 1.0
        if a is not None:
            self.a = a
        self.n_components = n_components

        # Fixed random Fourier feature weights (seed for reproducibility)
        rng = np.random.RandomState(0)
        self.W = rng.randn(n_components, n_features) * np.sqrt(2.0 * gamma)
        self.bias = rng.uniform(0.0, 2.0 * np.pi, n_components)
        self.scale = np.sqrt(2.0 / n_components)

        # CIRR state in RFF space (D × D)
        D = n_components
        self.A = np.zeros((D, D))
        self.b = np.zeros(D)
        self.w = np.ones(D)          # ones → D-mat = I at start
        self._aI = self.a * np.eye(D)  # precomputed; avoids per-step allocation

    def _phi(self, x):
        """RFF feature map $\varphi(x) \in \mathbb{R}^D$."""
        return self.scale * np.cos(self.W @ x + self.bias)

    def _D(self):
        d_sqrt = np.sqrt(np.abs(self.w))
        return np.outer(d_sqrt, d_sqrt)

    def predict(self, x):
        phi = self._phi(x)
        D_mat = self._D()
        temp_A = self.A + np.outer(phi, phi)          # one-step lookahead
        M = self._aI + D_mat * temp_A
        try:
            InvM = np.linalg.inv(M)
            pw = (D_mat * InvM) @ self.b
        except np.linalg.LinAlgError:
            pw = self.w
        return float(np.dot(pw, phi))

    def update(self, x, y):
        phi = self._phi(x)
        D_mat = self._D()                             # D from current w
        self.A += np.outer(phi, phi)
        self.b += y * phi
        M = self._aI + D_mat * self.A
        try:
            InvM = np.linalg.inv(M)
            self.w = (D_mat * InvM) @ self.b
        except np.linalg.LinAlgError:
            pass


class QCIRR:
    """Quantum-enhanced CIRR (Q-CIRR).

    Uses the same adaptive D-matrix as CIRR, then perturbs the solved weight
    vector with deterministic HHL quantum measurement noise:
      $\text{noise}_j = \epsilon \sin(tn + j + 1)$
    This models quantum measurement error without norm-scaling (bounded noise).
    """

    name = "Q-CIRR"

    def __init__(self, n_features, a=None, total_samples=None, epsilon=0.01):
        self.n = n_features
        self.a = (1.0 / total_samples) if (total_samples and total_samples > 0) else 1.0
        if a is not None:
            self.a = a
        self.epsilon = epsilon
        self.A = np.zeros((n_features, n_features))
        self.b = np.zeros(n_features)
        self.w = np.ones(n_features)    # ones → D = I at initialisation
        self._aI = self.a * np.eye(n_features)
        self._j_idx = np.arange(n_features, dtype=np.float64)
        self._t = 0

    def _D(self):
        d_sqrt = np.sqrt(np.abs(self.w))
        return np.outer(d_sqrt, d_sqrt)

    def predict(self, x):
        D = self._D()
        temp_A = self.A + np.outer(x, x)
        M = self._aI + D * temp_A
        try:
            InvM = np.linalg.inv(M)
            pw = (D * InvM) @ self.b
        except np.linalg.LinAlgError:
            pw = self.w
        return float(np.dot(pw, x))

    def update(self, x, y):
        D = self._D()
        self.A += np.outer(x, x)
        self.b += y * x
        M = self._aI + D * self.A
        try:
            InvM = np.linalg.inv(M)
            sol = (D * InvM) @ self.b
            noise = self.epsilon * np.sin(
                self._t * self.n + self._j_idx + 1.0
            )
            self.w = sol + noise
        except np.linalg.LinAlgError:
            pass
        self._t += 1


class QKCIRR:
    """Quantum Kernelised CIRR (QK-CIRR).

    Identical to K-CIRR (RFF feature map + CIRR D-matrix) with the addition
    of deterministic HHL quantum measurement noise on the weight vector:

        $\text{noise}_j = \epsilon \sin(tD + j + 1)$

    This is the kernel analogue of Q-CIRR: same quantum perturbation model,
    operating in the RFF-lifted feature space $\mathbb{R}^D$.
    """

    name = "QK-CIRR"

    def __init__(self, n_features, gamma=None, a=None, total_samples=None,
                 epsilon=0.01, n_components=50):
        gamma = gamma if gamma is not None else 1.0 / (2.0 * n_features)
        self.a = (1.0 / total_samples) if (total_samples and total_samples > 0) else 1.0
        if a is not None:
            self.a = a
        self.epsilon = epsilon
        self.n_components = n_components

        rng = np.random.RandomState(0)
        self.W = rng.randn(n_components, n_features) * np.sqrt(2.0 * gamma)
        self.bias = rng.uniform(0.0, 2.0 * np.pi, n_components)
        self.scale = np.sqrt(2.0 / n_components)

        D = n_components
        self.A = np.zeros((D, D))
        self.b = np.zeros(D)
        self.w = np.ones(D)
        self._aI = self.a * np.eye(D)         # precomputed
        self._j_idx = np.arange(D, dtype=np.float64)  # precomputed
        self._t = 0

    def _phi(self, x):
        return self.scale * np.cos(self.W @ x + self.bias)

    def _D(self):
        d_sqrt = np.sqrt(np.abs(self.w))
        return np.outer(d_sqrt, d_sqrt)

    def predict(self, x):
        phi = self._phi(x)
        D_mat = self._D()
        temp_A = self.A + np.outer(phi, phi)
        M = self._aI + D_mat * temp_A
        try:
            InvM = np.linalg.inv(M)
            pw = (D_mat * InvM) @ self.b
        except np.linalg.LinAlgError:
            pw = self.w
        return float(np.dot(pw, phi))

    def update(self, x, y):
        phi = self._phi(x)
        D_mat = self._D()
        self.A += np.outer(phi, phi)
        self.b += y * phi
        M = self._aI + D_mat * self.A
        try:
            InvM = np.linalg.inv(M)
            sol = (D_mat * InvM) @ self.b
            noise = self.epsilon * np.sin(
                self._t * self.n_components + self._j_idx + 1.0
            )
            self.w = sol + noise
        except np.linalg.LinAlgError:
            pass
        self._t += 1


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — ALGORITHM REGISTRY
# ══════════════════════════════════════════════════════════════════════

ALL_ALGORITHMS = {
    "CIRR":    CIRR,
    "K-CIRR":  KCIRR,
    "Q-CIRR":  QCIRR,
    "QK-CIRR": QKCIRR,
}

KERNEL_ALGORITHMS = {"K-CIRR", "QK-CIRR"}
QUANTUM_ALGORITHMS = {"Q-CIRR", "QK-CIRR"}


def warmup_jit():
    """Trigger Numba compilation on tiny synthetic data (run once at startup)."""
    print("Warming up Numba JIT compilation...")
    X_w = np.random.randn(5, 3).astype(np.float64)
    y_w = np.random.randn(5).astype(np.float64)
    _cirr_kernel(np.ascontiguousarray(X_w), np.ascontiguousarray(y_w), 1.0)
    _qcirr_kernel(np.ascontiguousarray(X_w), np.ascontiguousarray(y_w), 1.0, 0.01)
    print("  JIT warm-up complete.")
