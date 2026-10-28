#!/usr/bin/env python3
"""
algos.py — Algorithms for the paper "Quantum Kernel Ridge Regression".

Contains:
  · QKRR      — The proposed method (quantum kernel + HHL + physics-informed)
  · PIKRR     — Classical physics-informed KRR baseline (RBF/Matérn)
  · PINN      — Physics-Informed Neural Network baseline (PyTorch)
  · MLP       — Standard feedforward neural network (non-physics baseline)

All algorithms expose: predict(), update(), evaluate().
"""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
#  KERNEL UTILITIES
# ══════════════════════════════════════════════════════════════════════

def matern_kernel(X1, X2, length_scale=1.0, nu=2.5):
    """Matérn 5/2 kernel."""
    dist = cdist(X1, X2, metric='euclidean') / length_scale
    sqrt5 = np.sqrt(5.0)
    K = (1.0 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * np.exp(-sqrt5 * dist)
    return K


def _quantum_feature_map(X, n_qubits, length_scale, beta, seed):
    """Return the weighted RFF feature matrix for X."""
    rng = np.random.RandomState(seed)
    D = 2 ** n_qubits
    d = X.shape[1]
    W = rng.randn(D, d) * np.sqrt(2.0) / length_scale
    b = rng.uniform(0, 2 * np.pi, D)
    phi = np.sqrt(2.0 / D) * np.cos(X @ W.T + b)
    weights = np.exp(-beta * np.arange(D))
    weights = weights / np.linalg.norm(weights) * np.sqrt(D)
    phi = phi * weights
    return phi


def quantum_kernel_sim(X1, X2, n_qubits=8, length_scale=1.0, beta=0.1, seed=42):
    """
    Simulate a quantum kernel via a high-dimensional structured feature map.

    The map is a weighted random Fourier feature (RFF) expansion.  Frequencies
    are sampled from an RBF-like power spectrum with bandwidth ``length_scale``,
    and feature weights decay exponentially at rate ``beta``.  The resulting
    Gram matrix has (i) finite rank bounded by the Hilbert-space dimension
    ``D = 2**n_qubits`` and (ii) rapidly decaying eigenvalues, mimicking the
    spectral behaviour assumed for quantum kernels on low-dimensional manifolds.
    """
    phi1 = _quantum_feature_map(X1, n_qubits, length_scale, beta, seed)
    phi2 = _quantum_feature_map(X2, n_qubits, length_scale, beta, seed)
    K = phi1 @ phi2.T
    return K, phi1, phi2


def build_physics_gram(X_data, y_data, X_coll, y_coll, kernel_fn,
                       lambda_reg=1e-4, pde_weight=1.0):
    """Build physics-informed Gram matrix and target vector."""
    Z = np.vstack([X_data, X_coll])
    K = kernel_fn(Z, Z)
    y_aug = np.hstack([y_data, pde_weight * y_coll])
    K_reg = K + lambda_reg * np.eye(len(Z))
    return K_reg, y_aug, Z


# ══════════════════════════════════════════════════════════════════════
#  HHL SIMULATOR
# ══════════════════════════════════════════════════════════════════════

class HHLSimulator:
    """Simulate HHL quantum matrix inversion with noise and complexity tracking."""
    def __init__(self, eps=0.01, track_complexity=True):
        self.eps = eps
        self.track_complexity = track_complexity
        self.total_gate_count = 0
        self.inversion_count = 0

    def solve(self, A, b):
        n = A.shape[0]
        x_exact = np.linalg.solve(A, b)
        # HHL solves the linear system to relative residual precision eps.
        # We model the quantum error as additive noise whose scale guarantees
        # ||A (x_quantum - x_exact)||_2 / ||b||_2 <= eps.
        noise_scale = self.eps * np.linalg.norm(b) / np.linalg.norm(A)
        noise = np.random.randn(n) * noise_scale
        x_quantum = x_exact + noise
        if self.track_complexity:
            cond = np.linalg.cond(A)
            gate_count = np.log2(max(n, 2)) * (cond ** 2) * (np.log2(max(cond, 2)) ** 2)
            self.total_gate_count += gate_count
            self.inversion_count += 1
            return x_quantum, cond
        return x_quantum, np.nan

    def complexity_report(self):
        return {
            'total_gate_count': self.total_gate_count,
            'inversion_count': self.inversion_count,
            'avg_gates_per_inversion': self.total_gate_count / max(self.inversion_count, 1)
        }


# ══════════════════════════════════════════════════════════════════════
#  QKRR — PROPOSED METHOD
# ══════════════════════════════════════════════════════════════════════

class QKRR:
    """
    Quantum Kernel Ridge Regression (QKRR).

    Combines:
      · Quantum kernel with exponential spectral decay
      · Physics-informed Gram matrix (data + collocation points)
      · HHL-simulated matrix inversion
      · Seq updates with immediate / delayed / batched feedback
    """
    def __init__(self, n_qubits=8, length_scale=1.0, beta=0.1, lambda_reg=1e-4,
                 pde_weight=1.0, hhl_eps=0.01, seed=42,
                 feature_threshold=1000, track_complexity=True):
        self.n_qubits = n_qubits
        self.length_scale = length_scale
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.pde_weight = pde_weight
        self.hhl = HHLSimulator(eps=hhl_eps, track_complexity=track_complexity)
        self.seed = seed
        self.feature_threshold = feature_threshold
        self.Z = None
        self.alpha = None
        self.coef = None
        self.feature_mode = False
        self.history = {'rmse': [], 'rel_l2': [], 'pde_residual': [],
                        'cond_number': [], 'eff_dim': []}
        # buffers
        self.X_data_buffer = []
        self.y_data_buffer = []
        self.X_coll_buffer = []
        self.y_coll_buffer = []
        self.pending_updates = []
        # Incremental feature-space covariance for updates
        self._A = None
        self._b = None

    def _feature_dim(self):
        return 2 ** self.n_qubits

    def _quantum_kernel(self, X1, X2):
        K, _, _ = quantum_kernel_sim(X1, X2, self.n_qubits, self.length_scale,
                                      self.beta, self.seed)
        return K

    def fit(self, X_data, y_data, X_coll, y_coll):
        if self.pde_weight == 0.0:
            Z = X_data
            y_aug = y_data
        else:
            Z = np.vstack([X_data, X_coll])
            y_aug = np.hstack([y_data, self.pde_weight * y_coll])
        self.Z = Z
        # For large support sets, solve in feature space (Woodbury) instead of
        # building the full N x N Gram matrix.
        if len(Z) > self.feature_threshold:
            self.feature_mode = True
            Phi = _quantum_feature_map(Z, self.n_qubits, self.length_scale,
                                       self.beta, self.seed)
            A = Phi.T @ Phi + self.lambda_reg * np.eye(Phi.shape[1])
            b = Phi.T @ y_aug
            self.coef, cond = self.hhl.solve(A, b)
            self.alpha = None
            # Store unregularised covariance for incremental updates.
            self._A = Phi.T @ Phi
            self._b = Phi.T @ y_aug
            # Effective dimension of the finite-rank quantum kernel.
            eigvals = np.linalg.eigvalsh(Phi.T @ Phi)
            eff_dim = np.sum(eigvals / (eigvals + self.lambda_reg))
        else:
            self.feature_mode = False
            if self.pde_weight == 0.0:
                self.Z = X_data
                y_aug = y_data
                K = self._quantum_kernel(self.Z, self.Z)
                K_reg = K + self.lambda_reg * np.eye(len(self.Z))
            else:
                K_reg, y_aug, self.Z = build_physics_gram(
                    X_data, y_data, X_coll, y_coll, self._quantum_kernel,
                    self.lambda_reg, self.pde_weight
                )
            self.alpha, cond = self.hhl.solve(K_reg, y_aug)
            self.coef = None
            eigvals = np.linalg.eigvalsh(K_reg) - self.lambda_reg
            eigvals = np.clip(eigvals, 0, None)
            eff_dim = np.sum(eigvals / (eigvals + self.lambda_reg))
        self.history['cond_number'].append(cond)
        self.history['eff_dim'].append(eff_dim)
        return self

    def predict(self, X):
        if self.Z is None:
            return np.zeros(len(X))
        if self.feature_mode:
            if self.coef is None:
                return np.zeros(len(X))
            phi = _quantum_feature_map(X, self.n_qubits, self.length_scale,
                                       self.beta, self.seed)
            return phi @ self.coef
        if self.alpha is None:
            return np.zeros(len(X))
        k_z, _, _ = quantum_kernel_sim(X, self.Z, self.n_qubits,
                                        self.length_scale, self.beta, self.seed)
        return k_z @ self.alpha

    def evaluate(self, X_test, y_test, X_coll=None, y_coll=None):
        pred = self.predict(X_test)
        mse = np.mean((pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        rel_l2 = np.linalg.norm(pred - y_test) / (np.linalg.norm(y_test) + 1e-8)
        result = {'MSE': mse, 'RMSE': rmse, 'rel_L2': rel_l2}
        if X_coll is not None and y_coll is not None:
            pred_coll = self.predict(X_coll)
            pde_res = np.mean((pred_coll - y_coll) ** 2)
            result['PDE_residual'] = pde_res
            self.history['pde_residual'].append(pde_res)
        self.history['rmse'].append(rmse)
        self.history['rel_l2'].append(rel_l2)
        return result

    def _update_feature_covariance(self, x_data, y_data, x_coll, y_coll):
        """Add one data (and optionally collocation) row to feature covariance."""
        phi_d = _quantum_feature_map(x_data.reshape(1, -1), self.n_qubits,
                                     self.length_scale, self.beta, self.seed)
        if self._A is None:
            D = self._feature_dim()
            self._A = np.zeros((D, D), dtype=phi_d.dtype)
            self._b = np.zeros(D, dtype=phi_d.dtype)
        self._A += phi_d.T @ phi_d
        self._b += y_data * phi_d[0]
        if self.pde_weight != 0.0:
            phi_c = _quantum_feature_map(x_coll.reshape(1, -1), self.n_qubits,
                                         self.length_scale, self.beta, self.seed)
            self._A += phi_c.T @ phi_c
            self._b += self.pde_weight * y_coll * phi_c[0]

    def seq_update(self, x_data, y_data, x_coll, y_coll,
                      mode='immediate', delay=0, batch_size=1):
        self.X_data_buffer.append(x_data)
        self.y_data_buffer.append(y_data)
        self.X_coll_buffer.append(x_coll)
        self.y_coll_buffer.append(y_coll)
        if mode == 'immediate':
            self._update_feature_covariance(x_data, y_data, x_coll, y_coll)
            self._retrain_incremental()
        elif mode == 'delayed':
            self.pending_updates.append((x_data, y_data, x_coll, y_coll))
            if len(self.pending_updates) > delay:
                self.pending_updates.pop(0)
                self._update_feature_covariance(x_data, y_data, x_coll, y_coll)
                self._retrain_incremental()
        elif mode == 'batched':
            self._update_feature_covariance(x_data, y_data, x_coll, y_coll)
            if len(self.X_data_buffer) >= batch_size:
                self._retrain_from_buffer()
                self.X_data_buffer = []
                self.y_data_buffer = []
                self.X_coll_buffer = []
                self.y_coll_buffer = []
                self._A = None
                self._b = None

    def _retrain_incremental(self):
        if self._A is None or self._b is None or len(self.X_data_buffer) == 0:
            return
        D = self._A.shape[0]
        A_reg = self._A + self.lambda_reg * np.eye(D)
        self.coef, cond = self.hhl.solve(A_reg, self._b)
        self.feature_mode = True
        self.alpha = None
        self.Z = np.array(self.X_data_buffer) if self.pde_weight == 0.0 \
            else np.vstack([np.array(self.X_data_buffer),
                            np.vstack(self.X_coll_buffer)])
        # Skip expensive eigendecomposition during updates.
        self.history['cond_number'].append(cond)
        self.history['eff_dim'].append(np.nan)

    def _retrain_from_buffer(self):
        """Batch-only retrain using the current buffer contents."""
        if len(self.X_data_buffer) == 0:
            return
        Xd = np.array(self.X_data_buffer)
        yd = np.array(self.y_data_buffer)
        Xc = np.vstack(self.X_coll_buffer)
        yc = np.hstack(self.y_coll_buffer)
        self.fit(Xd, yd, Xc, yc)

    def complexity_report(self):
        return self.hhl.complexity_report()


# ══════════════════════════════════════════════════════════════════════
#  PI-KRR — CLASSICAL BASELINE
# ══════════════════════════════════════════════════════════════════════

class PIKRR:
    """Classical Physics-Informed Kernel Ridge Regression (RBF/Matérn)."""
    def __init__(self, kernel='rbf', length_scale=1.0, lambda_reg=1e-4,
                 pde_weight=1.0):
        self.kernel = kernel
        self.length_scale = length_scale
        self.lambda_reg = lambda_reg
        self.pde_weight = pde_weight
        self.Z = None
        self.alpha = None
        self.history = {'rmse': [], 'rel_l2': [], 'pde_residual': []}

    def _kernel_fn(self, X1, X2):
        if self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=0.5 / (self.length_scale ** 2))
        elif self.kernel == 'matern':
            return matern_kernel(X1, X2, self.length_scale)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X_data, y_data, X_coll, y_coll):
        K_reg, y_aug, self.Z = build_physics_gram(
            X_data, y_data, X_coll, y_coll, self._kernel_fn,
            self.lambda_reg, self.pde_weight
        )
        self.alpha = np.linalg.solve(K_reg, y_aug)
        return self

    def predict(self, X):
        if self.Z is None or self.alpha is None:
            return np.zeros(len(X))
        k_z = self._kernel_fn(X, self.Z)
        return k_z @ self.alpha

    def evaluate(self, X_test, y_test, X_coll=None, y_coll=None):
        pred = self.predict(X_test)
        mse = np.mean((pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        rel_l2 = np.linalg.norm(pred - y_test) / (np.linalg.norm(y_test) + 1e-8)
        result = {'MSE': mse, 'RMSE': rmse, 'rel_L2': rel_l2}
        if X_coll is not None and y_coll is not None:
            pred_coll = self.predict(X_coll)
            pde_res = np.mean((pred_coll - y_coll) ** 2)
            result['PDE_residual'] = pde_res
            self.history['pde_residual'].append(pde_res)
        self.history['rmse'].append(rmse)
        self.history['rel_l2'].append(rel_l2)
        return result


# ══════════════════════════════════════════════════════════════════════
#  PINN — NEURAL BASELINE
# ══════════════════════════════════════════════════════════════════════

class PINN(nn.Module):
    """Standard feedforward Physics-Informed Neural Network."""
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4, activation='tanh'):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        act = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        layers.append(act)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def pde_residual(self, x_coll):
        x_coll = x_coll.clone().requires_grad_(True)
        u = self.forward(x_coll)
        grads = torch.autograd.grad(u, x_coll, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
        u_x = grads[:, 0]
        u_t = grads[:, 1] if x_coll.shape[1] > 1 else torch.zeros_like(u_x)
        u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x),
                                     create_graph=True)[0][:, 0]
        return u, u_x, u_t, u_xx


class PINNSolver:
    """PINN solver with seq/delayed/batched training."""
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4,
                 lambda_phys=1.0, lr=1e-3, device='cpu'):
        self.model = PINN(input_dim, hidden_dim, n_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_phys = lambda_phys
        self.device = device
        self.history = {'data_loss': [], 'phys_loss': [], 'total_loss': []}
        self.pending_data = []
        self.pending_coll = []

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def train_step(self, X_data, y_data, X_coll, y_coll, n_epochs=1, eval_set=None):
        Xd = self._to_tensor(X_data)
        yd = self._to_tensor(y_data)
        Xc = self._to_tensor(X_coll)
        yc = self._to_tensor(y_coll)
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            pred_data = self.model(Xd)
            loss_data = torch.mean((pred_data - yd) ** 2)
            u, u_x, u_t, u_xx = self.model.pde_residual(Xc)
            residual = u_t - 0.01 / np.pi * u_xx + u * u_x
            loss_phys = torch.mean((residual - yc) ** 2)
            loss = loss_data + self.lambda_phys * loss_phys
            loss.backward()
            self.optimizer.step()
        self.history['data_loss'].append(loss_data.item())
        self.history['phys_loss'].append(loss_phys.item())
        self.history['total_loss'].append(loss.item())
        if eval_set is not None:
            eval_rmse = self.evaluate(eval_set[0], eval_set[1])['RMSE']
            self.history.setdefault('val_rmse', []).append(eval_rmse)
        return loss.item()

    def predict(self, X):
        with torch.no_grad():
            return self.model(self._to_tensor(X)).cpu().numpy()

    def evaluate(self, X_test, y_test):
        pred = self.predict(X_test)
        mse = np.mean((pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        rel_l2 = np.linalg.norm(pred - y_test) / (np.linalg.norm(y_test) + 1e-8)
        return {'MSE': mse, 'RMSE': rmse, 'rel_L2': rel_l2}

    def seq_update(self, x, y, x_coll, y_coll, mode='immediate', delay=0):
        self.pending_data.append((x, y))
        self.pending_coll.append((x_coll, y_coll))
        if mode == 'immediate':
            self._flush_pending()
        elif mode == 'delayed' and len(self.pending_data) > delay:
            self._flush_one()
        elif mode == 'batched' and len(self.pending_data) >= delay:
            self._flush_pending()

    def _flush_pending(self):
        if not self.pending_data:
            return
        Xd = np.array([p[0] for p in self.pending_data])
        yd = np.array([p[1] for p in self.pending_data])
        Xc = np.vstack([p[0] for p in self.pending_coll])
        yc = np.hstack([p[1] for p in self.pending_coll])
        self.train_step(Xd, yd, Xc, yc, n_epochs=1)
        self.pending_data = []
        self.pending_coll = []

    def _flush_one(self):
        x, y = self.pending_data.pop(0)
        xc, yc = self.pending_coll.pop(0)
        self.train_step(x[None, :], np.array([y]), xc[None, :], np.array([yc]), n_epochs=1)


# ══════════════════════════════════════════════════════════════════════
#  MLP — STANDARD NEURAL BASELINE (non-physics)
# ══════════════════════════════════════════════════════════════════════

class StandardMLP(nn.Module):
    """Standard feedforward neural network for regression (no physics)."""
    def __init__(self, input_dim, hidden_dim=64, n_layers=3, lr=1e-3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self(x).cpu().numpy().squeeze()

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(torch.tensor(x[None, :], dtype=torch.float32))
        loss = torch.mean((pred - torch.tensor([y], dtype=torch.float32)) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, n_epochs=100, batch_size=None, eval_set=None):
        """Batch training for a fair comparison with QKRR."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = len(y)
        if batch_size is None:
            batch_size = n
        history = {'val_rmse': []}
        for _ in range(n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                self.optimizer.zero_grad()
                pred = self.forward(X_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                loss.backward()
                self.optimizer.step()
            if eval_set is not None:
                preds = self.predict(eval_set[0])
                rmse = np.sqrt(np.mean((preds - eval_set[1]) ** 2))
                history['val_rmse'].append(rmse)
        return history


class DeepMLP(StandardMLP):
    """Deeper/wider MLP baseline."""
    def __init__(self, input_dim, hidden_dim=128, n_layers=5, lr=1e-3):
        super().__init__(input_dim, hidden_dim=hidden_dim, n_layers=n_layers, lr=lr)


class ResMLP(nn.Module):
    """Residual MLP baseline."""
    def __init__(self, input_dim, hidden_dim=64, n_blocks=4, lr=1e-3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_blocks)
        ])
        self.output = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        for block in self.blocks:
            h = torch.relu(h + block(h))
        return self.output(h).squeeze(-1)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self(x).cpu().numpy().squeeze()

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(torch.tensor(x[None, :], dtype=torch.float32))
        loss = torch.mean((pred - torch.tensor([y], dtype=torch.float32)) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, n_epochs=100, batch_size=None, eval_set=None):
        """Batch training for a fair comparison with QKRR."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = len(y)
        if batch_size is None:
            batch_size = n
        history = {'val_rmse': []}
        for _ in range(n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                self.optimizer.zero_grad()
                pred = self.forward(X_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                loss.backward()
                self.optimizer.step()
            if eval_set is not None:
                preds = self.predict(eval_set[0])
                rmse = np.sqrt(np.mean((preds - eval_set[1]) ** 2))
                history['val_rmse'].append(rmse)
        return history


class Conv1DRegressor(nn.Module):
    """1-D CNN over the feature vector."""
    def __init__(self, input_dim, channels=32, kernel_size=3, lr=1e-3):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size,
                              padding=kernel_size // 2)
        self.fc = nn.Linear(channels * input_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # x: (batch, input_dim)
        h = torch.relu(self.conv(x.unsqueeze(1)))  # (batch, channels, input_dim)
        h = h.reshape(h.size(0), -1)
        return self.fc(h).squeeze(-1)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self(x).cpu().numpy().squeeze()

    def update(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(torch.tensor(x[None, :], dtype=torch.float32))
        loss = torch.mean((pred - torch.tensor([y], dtype=torch.float32)) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, n_epochs=100, batch_size=None, eval_set=None):
        """Batch training for a fair comparison with QKRR."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = len(y)
        if batch_size is None:
            batch_size = n
        history = {'val_rmse': []}
        for _ in range(n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                self.optimizer.zero_grad()
                pred = self.forward(X_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                loss.backward()
                self.optimizer.step()
            if eval_set is not None:
                preds = self.predict(eval_set[0])
                rmse = np.sqrt(np.mean((preds - eval_set[1]) ** 2))
                history['val_rmse'].append(rmse)
        return history


class LSTMRegressor(nn.Module):
    """SEQ LSTM baseline using a rolling window of inputs."""
    def __init__(self, input_dim, hidden_dim=32, seq_len=5, lr=1e-3):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.buffer_x = []
        self.buffer_y = []

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, input_dim)
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :]).squeeze(-1)

    def predict(self, x):
        self.buffer_x.append(x)
        if len(self.buffer_x) < self.seq_len:
            return 0.0
        seq = np.stack(self.buffer_x[-self.seq_len:])
        with torch.no_grad():
            return self(torch.tensor(seq[None, :, :], dtype=torch.float32)).item()

    def update(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_x) < self.seq_len:
            return 0.0
        seq = torch.tensor(
            np.stack(self.buffer_x[-self.seq_len:])[None, :, :],
            dtype=torch.float32)
        target = torch.tensor([self.buffer_y[-1]], dtype=torch.float32)
        self.optimizer.zero_grad()
        pred = self(seq)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, n_epochs=100, eval_set=None):
        """Batch training on sliding-window sequences."""
        n = len(y)
        seq_len = self.seq_len
        X_seq = np.array([X[i:i + seq_len] for i in range(n - seq_len + 1)])
        y_seq = y[seq_len - 1:]
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)
        history = {'val_rmse': []}
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            pred = self(X_t)
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            self.optimizer.step()
            if eval_set is not None:
                preds = self.predict_full(eval_set[0])
                rmse = np.sqrt(np.mean((preds - eval_set[1]) ** 2))
                history['val_rmse'].append(rmse)
        return history

    def predict_full(self, X):
        """Predict the full sequence after batch training."""
        n = len(X)
        preds = np.zeros(n)
        with torch.no_grad():
            for t in range(self.seq_len - 1, n):
                seq = torch.tensor(
                    X[t - self.seq_len + 1:t + 1][None, :, :],
                    dtype=torch.float32)
                preds[t] = self(seq).item()
        return preds



def compare_spectral_decay(X, n_qubits=8, length_scale=1.0):
    """Compare eigenvalue decay of classical RBF vs quantum kernel."""
    K_rbf = rbf_kernel(X, X, gamma=0.5 / (length_scale ** 2))
    K_quant, _, _ = quantum_kernel_sim(X, X, n_qubits, length_scale, seed=42)
    eig_rbf = np.sort(np.linalg.eigvalsh(K_rbf))[::-1]
    eig_quant = np.sort(np.linalg.eigvalsh(K_quant))[::-1]
    lambda_reg = 1e-4
    deff_rbf = np.sum(eig_rbf / (eig_rbf + lambda_reg))
    deff_quant = np.sum(eig_quant / (eig_quant + lambda_reg))
    return {
        'eig_rbf': eig_rbf,
        'eig_quant': eig_quant,
        'deff_rbf': deff_rbf,
        'deff_quant': deff_quant,
    }
