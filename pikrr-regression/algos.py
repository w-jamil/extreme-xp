#!/usr/bin/env python3
"""Algorithms for the QIKRR physics-informed PDE experiments.

QIKRR uses iteratively reweighted regularisation and HHL-simulated solves.
PI-IKRR is its classical iterative physics-informed control, PI-KRR is the
non-iterative physics-informed benchmark, and IKRR/KRR are the corresponding
data-only iterative and non-iterative methods.  All accept the same kernel
backends (RFF surrogate, RBF, or exact commuting ZZ).  Matched comparisons
therefore separate solver/noise, iterative reweighting, physics constraints,
and kernel inductive bias.

No function in this file executes a quantum algorithm on quantum hardware.
HHLSimulator performs a classical solve, adds optional precision-matched
noise, and records a transparent fault-tolerant gate-count model.
"""

import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from kernels import QuantumKernel, RBFKernel, ZZKernel, build_pi_system


class HHLSimulator:
    """Classical HHL stand-in with additive noise and a gate-count model.

    The gate model is log2(n) * kappa**2 * log2(1/eps) per inversion.  For
    nonsymmetric Kansa matrices it is understood up to the constant overhead
    of a standard Hermitian dilation.  It is a cost model, not a measured
    quantum runtime or a hardware validation.
    """

    def __init__(self, eps=0.01, noise=True):
        if not 0 < eps < 1:
            raise ValueError('eps must lie in (0, 1)')
        self.eps = eps
        self.noise = noise
        self.total_gates = 0.0
        self.inversions = 0
        self.condition_numbers = []

    def solve(self, A, b, rng):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        x = np.linalg.solve(A, b)
        cond = float(np.linalg.cond(A))
        if self.noise:
            scale = self.eps * np.linalg.norm(b) / max(np.linalg.norm(A), 1e-300)
            x = x + rng.standard_normal(len(x)) * scale
        n = A.shape[0]
        self.total_gates += (np.log2(max(n, 2)) * cond ** 2
                             * np.log2(1.0 / self.eps))
        self.inversions += 1
        self.condition_numbers.append(cond)
        return x, cond


def make_kernel(pde, kind='rbf', n_qubits=8, length_scale=1.0,
                beta=0.1, seed=42):
    """Construct one of the three interchangeable kernel backends."""
    terms = pde.operator_terms if pde is not None else None
    if kind == 'rff':
        return QuantumKernel(n_qubits=n_qubits, length_scale=length_scale,
                             beta=beta, seed=seed, operator_terms=terms)
    if kind == 'rbf':
        return RBFKernel(length_scale=length_scale, operator_terms=terms)
    if kind == 'zz':
        if pde is None:
            raise ValueError('The ZZ backend requires a PDE problem')
        return ZZKernel(n_qubits=n_qubits, input_dim=pde.dim,
                        operator_terms=terms)
    raise ValueError(f'Unknown kernel kind: {kind}')


class PhysicsInformedKRR:
    """Physics-informed Kansa collocation with optional iterative reweighting.

    ``n_reweight=1`` gives the ordinary regularised solve.  Values greater
    than one add fixed-operator feature-space reweighted solves with
    ``G = diag(max(abs(previous_primal_coefficient), reweight_delta))``.
    Offline fits use the eigenbasis of the training-functional Gram matrix;
    sequential RFF fits use the explicit feature coordinates.  Every
    reweighted solve is counted separately in ``linear_solves``; ``refits``
    counts model updates.  Sequential feedback uses causal protocols.
    """

    def __init__(self, kernel, lam=1e-4, pde_weight=1.0, hhl=None, seed=42,
                 clip=None, n_reweight=1, reweight_delta=1e-3):
        if n_reweight < 1:
            raise ValueError('n_reweight must be at least one')
        self.kernel = kernel
        self.lam = float(lam)
        self.pde_weight = pde_weight
        self.hhl = hhl
        self.rng = np.random.RandomState(seed)
        self.clip = clip
        self.n_reweight = int(n_reweight)
        self.reweight_delta = float(reweight_delta)

        self.Xd = self.Xc = None
        self.coef_ = None
        self.build_time = 0.0
        self.fit_time = 0.0
        self.linear_solves = 0
        self.refits = 0
        self.inversions = 0       # backward-compatible alias for solves
        self.solves_t = []
        self.condition_numbers = []

        self.Xs, self.ys, self.Xa = [], [], []
        self.applied_indices = []
        self.chol = None
        self.alpha_ = None
        self.pending = []
        self.t = 0
        self._reweight_U = None
        self._reweight_g = None
        self._seq_g = None

    @property
    def modelled_gates(self):
        return float(self.hhl.total_gates) if self.hhl is not None else 0.0

    def _solve(self, A, rhs):
        t0 = time.perf_counter()
        if self.hhl is not None:
            x, cond = self.hhl.solve(A, rhs, self.rng)
        else:
            x = np.linalg.solve(A, rhs)
            cond = float(np.linalg.cond(A))
        self.linear_solves += 1
        self.inversions = self.linear_solves
        self.condition_numbers.append(cond)
        self.fit_time += time.perf_counter() - t0
        return x

    def _weighted_cross(self, cross):
        """Map evaluation functionals into the fixed reweighted feature basis."""
        if self._reweight_g is None:
            return cross
        return ((cross @ self._reweight_U) * self._reweight_g) @ self._reweight_U.T

    def _native_features(self, X):
        """Explicit RFF coordinates used by sequential feature reweighting."""
        if not hasattr(self.kernel, '_feat'):
            raise ValueError('Sequential iterative reweighting requires the '
                             'explicit RFF feature backend')
        zero = (0,) * np.atleast_2d(X).shape[1]
        return self.kernel._feat(np.atleast_2d(X), zero)

    def fit(self, Xd, yd, Xc, gc):
        self.Xd, self.Xc = np.atleast_2d(Xd), np.atleast_2d(Xc)
        t0 = time.perf_counter()
        A, rhs = build_pi_system(self.kernel, self.Xd, yd, self.Xc, gc,
                                 self.lam, self.pde_weight)
        self.build_time = time.perf_counter() - t0
        base_A = A - self.lam * np.eye(A.shape[0])
        self.fit_time = 0.0
        self.linear_solves = self.refits = self.inversions = 0
        self._reweight_U = self._reweight_g = None
        self.coef_ = self._solve(A, rhs)
        if self.n_reweight > 1:
            # H contains the unscaled data/collocation functionals and R is
            # the row scaling in build_pi_system: A = R H + lambda I.
            # Diagonalise H = U diag(s) U^T.  In this finite feature basis,
            # G = diag(g) gives H_G = U diag(s*g) U^T, while a cross-Gram row
            # h becomes h U G U^T.  The primal coordinates are
            # w = G diag(sqrt(s)) U^T alpha, exactly the object reweighted in
            # the theory; no dual-coefficient reweighting is used.
            n_d = len(self.Xd)
            s = (1.0 / max(np.mean(np.abs(gc)), 1e-12)
                 if len(self.Xc) else 1.0)
            row_scale = np.ones(A.shape[0])
            row_scale[n_d:] = s
            H = base_A / row_scale[:, None]
            H = (H + H.T) * 0.5
            eig, U = np.linalg.eigh(H)
            eig = np.clip(eig, 0.0, None)
            root = np.sqrt(eig)
            primal = root * (U.T @ self.coef_)
            for _ in range(self.n_reweight - 1):
                g = np.maximum(np.abs(primal), self.reweight_delta)
                H_g = (U * (eig * g)) @ U.T
                A_rw = H_g * row_scale[:, None] + self.lam * np.eye(A.shape[0])
                self.coef_ = self._solve(A_rw, rhs)
                primal = g * (root * (U.T @ self.coef_))
            self._reweight_U, self._reweight_g = U, g
        self.refits = 1
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.coef_ is not None:
            cross = np.concatenate([
                self.kernel.K(X, self.Xd),
                self.pde_weight * self.kernel.D2K(X, self.Xc)], axis=1)
            return self._weighted_cross(cross) @ self.coef_
        if self.alpha_ is not None and len(self.Xa):
            return self.kernel.K(X, np.asarray(self.Xa)) @ self.alpha_
        return np.zeros(len(X))

    def residual(self, X):
        X = np.atleast_2d(X)
        if self.coef_ is not None:
            cross = np.concatenate([
                self.kernel.D1K(X, self.Xd),
                self.pde_weight * self.kernel.D1D2K(X, self.Xc)], axis=1)
            return self._weighted_cross(cross) @ self.coef_
        if self.alpha_ is not None and len(self.Xa):
            return self.kernel.D1K(X, np.asarray(self.Xa)) @ self.alpha_
        return np.zeros(len(X))

    def start_round(self, x):
        """Predict causally using only the last completed refit."""
        x = np.atleast_2d(x)
        if self.alpha_ is None or not len(self.Xa):
            pred = 0.0
        else:
            if self._seq_g is not None:
                F = self._native_features(np.asarray(self.Xa))
                k = (self._native_features(x)[0] * self._seq_g) @ F.T
            else:
                k = self.kernel.K(x, np.asarray(self.Xa))[0]
            pred = float(k @ self.alpha_)
        self.t += 1
        if self.clip is not None:
            return float(np.clip(pred, -self.clip, self.clip))
        return pred

    def observe(self, x, y, mode='immediate', delay=0, batch_size=1):
        idx = len(self.Xs)
        self.Xs.append(np.atleast_2d(x)[0])
        self.ys.append(float(y))
        self.pending.append(idx)
        if mode == 'immediate':
            self._apply_update(self.pending.pop())
        elif mode == 'delayed':
            if len(self.pending) > delay:
                self._apply_update(self.pending.pop(0))
        elif mode == 'batched':
            if len(self.pending) >= batch_size:
                idxs = list(self.pending)
                self.pending = []
                self._apply_batch_update(idxs)
        else:
            raise ValueError(mode)

    def flush(self):
        if self.pending:
            idxs = list(self.pending)
            self.pending = []
            self._apply_batch_update(idxs)

    def _applied_targets(self):
        return np.asarray([self.ys[i] for i in self.applied_indices])

    def _refit_sequential(self):
        n = len(self.chol)
        A = self.chol @ self.chol.T
        ys = self._applied_targets()
        if self.hhl is None and self.n_reweight == 1:
            t0 = time.perf_counter()
            self.alpha_ = solve_triangular(
                self.chol.T, solve_triangular(self.chol, ys, lower=True))
            self.fit_time += time.perf_counter() - t0
            self.linear_solves += 1
            self.inversions = self.linear_solves
        else:
            self.alpha_ = self._solve(A, ys)
            self._seq_g = None
            if self.n_reweight > 1:
                F = self._native_features(np.asarray(self.Xa))
                primal = F.T @ self.alpha_
                for _ in range(self.n_reweight - 1):
                    g = np.maximum(np.abs(primal), self.reweight_delta)
                    A_rw = (F * g) @ F.T + self.lam * np.eye(n)
                    self.alpha_ = self._solve(A_rw, ys)
                    primal = g * (F.T @ self.alpha_)
                self._seq_g = g
        self.refits += 1
        self.solves_t.append(n)

    def _apply_update(self, idx):
        x_new = np.atleast_2d(self.Xs[idx])
        self.Xa.append(self.Xs[idx])
        self.applied_indices.append(idx)
        k_self = float(self.kernel.K(x_new, x_new)[0, 0]) + self.lam
        if self.chol is None:
            self.chol = np.array([[np.sqrt(k_self)]])
        else:
            X_old = np.asarray(self.Xa[:-1])
            c = self.kernel.K(x_new, X_old)[0]
            w = solve_triangular(self.chol, c, lower=True)
            piv = np.sqrt(max(k_self - w @ w, 1e-12))
            L = np.zeros((len(self.chol) + 1,) * 2)
            L[:-1, :-1] = self.chol
            L[-1, :-1] = w
            L[-1, -1] = piv
            self.chol = L
        self._refit_sequential()

    def _apply_batch_update(self, idxs):
        n_old = len(self.chol) if self.chol is not None else 0
        X_new = np.asarray([self.Xs[i] for i in idxs])
        self.Xa.extend(self.Xs[i] for i in idxs)
        self.applied_indices.extend(idxs)
        B = len(idxs)
        Kbb = self.kernel.K(X_new, X_new) + self.lam * np.eye(B)
        if self.chol is None:
            self.chol = cholesky(Kbb, lower=True)
        else:
            X_old = np.asarray(self.Xa[:n_old])
            C = self.kernel.K(X_old, X_new)
            W = solve_triangular(self.chol, C, lower=True)
            S = Kbb - W.T @ W
            from scipy.linalg import eigh
            eig, V = eigh((S + S.T) / 2)
            S = (V * np.clip(eig, 0.5 * self.lam, None)) @ V.T
            Ls = cholesky(S, lower=True)
            L = np.zeros((n_old + B,) * 2)
            L[:n_old, :n_old] = self.chol
            L[n_old:, :n_old] = W.T
            L[n_old:, n_old:] = Ls
            self.chol = L
        self._refit_sequential()


class CausalFullRefit:
    """Causal full-refit wrapper for streaming data with fixed collocation.

    Collocation points, when supplied, are known before the stream.  Only data
    targets whose feedback has been released enter each refit.  Predictions
    therefore use the last completed model under immediate, delayed, or
    batched feedback.  This wrapper is used for the matched sequential
    factorial experiment across RFF, RBF, and exact-ZZ kernels; it never
    reveals pending targets to the model.
    """

    def __init__(self, model, Xc, gc, clip=None):
        self.model = model
        self.Xc = np.atleast_2d(Xc)
        self.gc = np.asarray(gc)
        self.clip = clip
        self.Xa, self.ya, self.pending = [], [], []
        self.fitted = False
        self.refits = 0
        self.linear_solves = 0

    @property
    def modelled_gates(self):
        return float(getattr(self.model, 'modelled_gates', 0.0))

    def start_round(self, x):
        pred = (float(self.model.predict(np.atleast_2d(x))[0])
                if self.fitted else 0.0)
        if self.clip is not None:
            pred = float(np.clip(pred, -self.clip, self.clip))
        return pred

    def observe(self, x, y, mode='immediate', delay=0, batch_size=1):
        self.Xa.append(np.atleast_2d(x)[0])
        self.ya.append(float(y))
        self.pending.append(len(self.Xa) - 1)
        if mode == 'immediate':
            self._refit(self.pending.pop())
        elif mode == 'delayed':
            if len(self.pending) > delay:
                self._refit(self.pending.pop(0))
        elif mode == 'batched':
            if len(self.pending) >= batch_size:
                idxs = list(self.pending)
                self.pending = []
                self._refit(idxs)
        else:
            raise ValueError(mode)

    def _refit(self, idxs):
        if not isinstance(idxs, (list, tuple)):
            idxs = [idxs]
        # Xa/ya contain only released targets: pending entries are removed
        # before this call, so slice them out explicitly for delayed feedback.
        released = [i for i in range(len(self.Xa)) if i not in self.pending]
        X = np.asarray([self.Xa[i] for i in released])
        y = np.asarray([self.ya[i] for i in released])
        self.model.fit(X, y, self.Xc, self.gc)
        self.linear_solves += self.model.linear_solves
        self.refits += 1
        self.fitted = True

    def flush(self):
        if self.pending:
            idxs = list(self.pending)
            self.pending = []
            self._refit(idxs)


class QIKRR(PhysicsInformedKRR):
    """Quantum iterative kernel ridge regression with simulated HHL solves."""

    def __init__(self, pde, kernel_kind='rff', n_qubits=8, length_scale=1.0,
                 beta=0.1, lam=1e-4, pde_weight=1.0, hhl_eps=0.01,
                 hhl_noise=True, n_reweight=2, reweight_delta=1e-3,
                 seed=42, clip=None):
        kernel = make_kernel(pde, kernel_kind, n_qubits, length_scale,
                             beta, seed)
        super().__init__(kernel, lam=lam, pde_weight=pde_weight,
                         hhl=HHLSimulator(eps=hhl_eps, noise=hhl_noise),
                         seed=seed, clip=clip, n_reweight=n_reweight,
                         reweight_delta=reweight_delta)
        self.kernel_kind = kernel_kind
        self.n_qubits = n_qubits


class PIKRR(PhysicsInformedKRR):
    """Classical physics-informed KRR benchmark on any supplied kernel."""

    def __init__(self, pde, kernel_kind='rbf', n_qubits=8, length_scale=1.0,
                 beta=0.1, lam=1e-4, pde_weight=1.0, n_reweight=1,
                 reweight_delta=1e-3, seed=42, clip=None):
        kernel = make_kernel(pde, kernel_kind, n_qubits, length_scale,
                             beta, seed)
        super().__init__(kernel, lam=lam, pde_weight=pde_weight, hhl=None,
                         seed=seed, clip=clip, n_reweight=n_reweight,
                         reweight_delta=reweight_delta)
        self.kernel_kind = kernel_kind
        self.n_qubits = n_qubits


class PIIKRR(PIKRR):
    """Classical physics-informed iterative KRR (two solves by default)."""

    def __init__(self, *args, n_reweight=2, **kwargs):
        super().__init__(*args, n_reweight=n_reweight, **kwargs)


class IKRR(PIKRR):
    """Classical data-only iterative KRR when fitted without collocation."""

    def __init__(self, *args, n_reweight=2, **kwargs):
        super().__init__(*args, n_reweight=n_reweight, **kwargs)


class KRR:
    """Classical kernel ridge regression without collocation constraints."""

    def __init__(self, pde=None, kernel_kind='rbf', n_qubits=8,
                 length_scale=1.0, beta=0.1, lam=1e-4, seed=42):
        self.kernel = make_kernel(pde, kernel_kind, n_qubits, length_scale,
                                  beta, seed)
        self.lam = lam
        self.X = None
        self.alpha_ = None
        self.fit_time = 0.0
        self.linear_solves = 0
        self.modelled_gates = 0.0

    def fit(self, X, y, *_, **__):
        X = np.atleast_2d(X)
        K = self.kernel.K(X, X)
        t0 = time.perf_counter()
        self.alpha_ = np.linalg.solve(K + self.lam * np.eye(len(K)), y)
        self.fit_time = time.perf_counter() - t0
        self.linear_solves = 1
        self.X = X
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.alpha_ is None:
            return np.zeros(len(X))
        return self.kernel.K(X, self.X) @ self.alpha_

    def residual(self, X):
        """Evaluate D[u]; a data-only model still has a measurable residual."""
        X = np.atleast_2d(X)
        if self.alpha_ is None:
            return np.zeros(len(X))
        return self.kernel.D1K(X, self.X) @ self.alpha_


class PINN:
    """Physics-informed neural network (tanh MLP) with autograd residuals."""

    def __init__(self, pde, hidden=64, n_layers=4, lr=1e-3,
                 lambda_phys=1.0, seed=42):
        import torch
        self.torch = torch
        self.pde = pde
        self.lambda_phys = lambda_phys
        torch.manual_seed(seed)
        layers = [torch.nn.Linear(pde.dim, hidden), torch.nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden, 1)]
        self.net = torch.nn.Sequential(*layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.fit_time = 0.0
        self.linear_solves = 0
        self.modelled_gates = 0.0

    def _mixed(self, u, x, mi):
        torch = self.torch
        der = u
        for d, order in enumerate(mi):
            for _ in range(order):
                der = torch.autograd.grad(
                    der, x, grad_outputs=torch.ones_like(der),
                    create_graph=True)[0][:, d]
        return der

    def residual(self, x):
        torch = self.torch
        x = torch.as_tensor(x, dtype=torch.float32).requires_grad_(True)
        u = self.net(x).squeeze(-1)
        res = 0
        for coef, mi in self.pde.operator_terms:
            res = res + coef * self._mixed(u, x, mi)
        return res.detach().numpy()

    def predict(self, X):
        torch = self.torch
        with torch.no_grad():
            X = torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)
            return self.net(X).squeeze(-1).numpy()

    def fit(self, Xd, yd, Xc, gc, n_epochs=4000, verbose=False):
        torch = self.torch
        Xd = torch.as_tensor(Xd, dtype=torch.float32)
        yd = torch.as_tensor(yd, dtype=torch.float32)
        Xc = torch.as_tensor(Xc, dtype=torch.float32).requires_grad_(True)
        gc = torch.as_tensor(gc, dtype=torch.float32)
        t0 = time.perf_counter()
        for ep in range(n_epochs):
            self.opt.zero_grad()
            pred = self.net(Xd).squeeze(-1)
            loss_d = torch.mean((pred - yd) ** 2)
            u = self.net(Xc).squeeze(-1)
            res = 0
            for coef, mi in self.pde.operator_terms:
                res = res + coef * self._mixed(u, Xc, mi)
            loss_p = torch.mean((res - gc) ** 2)
            loss = loss_d + self.lambda_phys * loss_p
            loss.backward()
            self.opt.step()
            if verbose and ep % 300 == 0:
                print(f'    epoch {ep:5d}  data {loss_d.item():.3e}  '
                      f'phys {loss_p.item():.3e}')
        self.fit_time = time.perf_counter() - t0
        return self.fit_time
