#!/usr/bin/env python3
"""
kernels.py — Kernels with analytic derivative blocks for physics-informed
collocation.

Two kernel families are provided, both exposing the same interface:

  K(X, X2)          plain Gram matrix
  D1K(X, X2)        operator applied in the FIRST  argument:  D_x K(x, x2)
  D2K(X, X2)        operator applied in the SECOND argument:  D_x2 K(x, x2)
  D1D2K(X, X2)      operator applied in both arguments

where the operator is the linear constant-coefficient PDE operator passed as
``operator_terms = [(coef, multiindex), ...]``.

  QuantumKernel  — finite-dimensional RFF surrogate used to emulate a
                   quantum feature map on classical hardware.  It is not an
                   exact circuit simulation: phi(x) = sqrt(2/D) cos(Wx+b)*w,
                   features, W ~ N(0, 2/ell^2 I), and exponentially decaying
                   feature weights w_j ~ e^{-beta j} (normalised).  All kernel
                   derivatives are exact derivatives of the feature map.
  RBFKernel      — classical Gaussian kernel K(x,x') = exp(-gamma ||x-x'||^2)
                   with derivative blocks computed analytically through
                   Hermite-polynomial identities.

This dual implementation lets us cross-check the quantum-simulation against
the classical kernel on identical collocation systems.
"""

import numpy as np


# ----------------------------------------------------------------------
#  Quantum feature map (simulated) and its derivatives
# ----------------------------------------------------------------------

def quantum_feature_dict(X, n_qubits, length_scale, beta, seed, multiindices):
    """Feature matrix and analytic derivative features for each multiindex.

    For a multiindex m = (m_1,...,m_d),
        phi^{(m)}(x) = sqrt(2/D) cos(W x + b + pi|m|/2) * prod_j W[:,j]^{m_j} * w
    """
    X = np.atleast_2d(X)
    rng = np.random.RandomState(seed)
    D = 2 ** n_qubits
    n, d = X.shape
    if X.size == 0 or d == 0:
        return {tuple(m): np.zeros((max(n, 0), D)) for m in multiindices}
    W = rng.randn(D, d) * np.sqrt(2.0) / length_scale
    b = rng.uniform(0, 2 * np.pi, D)
    t = X @ W.T + b                                    # (n, D)
    w = np.exp(-beta * np.arange(D))
    w = w / np.linalg.norm(w) * np.sqrt(D)

    out = {}
    for m in multiindices:
        m = tuple(m)
        phase = 0.5 * np.pi * sum(m)
        coeff = np.ones(D)
        for j, mj in enumerate(m):
            if mj:
                coeff *= W[:, j] ** mj
        out[m] = np.sqrt(2.0 / D) * np.cos(t + phase) * coeff * w
    return out


class QuantumKernel:
    """Simulated quantum kernel with exact derivative features."""

    name = 'QIKRR'

    def __init__(self, n_qubits=8, length_scale=1.0, beta=0.1, seed=42,
                 operator_terms=None):
        self.n_qubits = n_qubits
        self.length_scale = length_scale
        self.beta = beta
        self.seed = seed
        self.operator_terms = operator_terms or [(1.0, tuple())]
        self._mis = list({mi for _, mi in self.operator_terms})

    def _feat(self, X, mi):
        return quantum_feature_dict(X, self.n_qubits, self.length_scale,
                                    self.beta, self.seed, [mi])[tuple(mi)]

    def K(self, X, X2):
        return self._feat(X, tuple([0] * np.atleast_2d(X).shape[1])) @ \
            self._feat(X2, tuple([0] * np.atleast_2d(X2).shape[1])).T

    def _op_feats(self, X):
        """Sum of derivative features weighted by operator coefficients."""
        acc = None
        for c, mi in self.operator_terms:
            f = self._feat(X, mi)
            acc = c * f if acc is None else acc + c * f
        return acc

    def D1K(self, X, X2):
        return self._op_feats(X) @ self._feat(X2, self._zero_mi(X)).T

    def D2K(self, X, X2):
        return self._feat(X, self._zero_mi(X)) @ self._op_feats(X2).T

    def D1D2K(self, X, X2):
        return self._op_feats(X) @ self._op_feats(X2).T

    def _zero_mi(self, X):
        return tuple([0] * np.atleast_2d(X).shape[1])


# ----------------------------------------------------------------------
#  RBF kernel with analytic derivative blocks (Hermite identities)
# ----------------------------------------------------------------------

def _hermite(n, t):
    """Physicists' Hermite polynomial H_n(t) for n = 0..4, elementwise."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return 2 * t
    if n == 2:
        return 4 * t ** 2 - 2
    if n == 3:
        return 8 * t ** 3 - 12 * t
    if n == 4:
        return 16 * t ** 4 - 48 * t ** 2 + 12
    raise ValueError("Hermite order > 4 not implemented")


def rbf_deriv_block(X, X2, gamma, a, b):
    """Block d^a_x d^b_x' K(x,x') for the Gaussian kernel.

    With z = x - x',  d^a_x d^b_x' K = (-1)^|b| d^{a+b}_z K, and
    d^c_z K = prod_i (-sqrt(gamma))^{c_i} H_{c_i}(sqrt(gamma) z_i) * K.
    """
    X = np.atleast_2d(X).astype(float)
    X2 = np.atleast_2d(X2).astype(float)
    a = tuple(a)
    b = tuple(b)
    d = X.shape[1]
    z = X[:, None, :] - X2[None, :, :]              # (n, m, dim)
    K0 = np.exp(-gamma * np.sum(z ** 2, axis=2))
    factor = np.ones_like(K0)
    for i in range(d):
        c = a[i] + b[i]
        if c == 0:
            continue
        ti = np.sqrt(gamma) * z[:, :, i]
        factor *= (-np.sqrt(gamma)) ** c * _hermite(c, ti)
    sign = (-1) ** sum(b)
    return sign * factor * K0


class RBFKernel:
    """Classical Gaussian kernel with analytic derivative blocks."""

    name = 'PI-KRR (RBF)'

    def __init__(self, length_scale=1.0, operator_terms=None):
        self.length_scale = length_scale
        self.gamma = 0.5 / length_scale ** 2
        self.operator_terms = operator_terms or [(1.0, tuple())]

    def K(self, X, X2):
        return rbf_deriv_block(X, X2, self.gamma,
                               tuple([0] * np.atleast_2d(X).shape[1]),
                               tuple([0] * np.atleast_2d(X2).shape[1]))

    def _op(self, X, X2, first):
        acc = None
        for c, mi in self.operator_terms:
            blk = rbf_deriv_block(X, X2, self.gamma, mi, tuple([0] * len(mi))) \
                if first else \
                rbf_deriv_block(X, X2, self.gamma, tuple([0] * len(mi)), mi)
            acc = c * blk if acc is None else acc + c * blk
        return acc

    def D1K(self, X, X2):
        return self._op(X, X2, first=True)

    def D2K(self, X, X2):
        return self._op(X, X2, first=False)

    def D1D2K(self, X, X2):
        acc = None
        for c1, m1 in self.operator_terms:
            for c2, m2 in self.operator_terms:
                blk = rbf_deriv_block(X, X2, self.gamma, m1, m2)
                term = (c1 * c2) * blk
                acc = term if acc is None else acc + term
        return acc


# ----------------------------------------------------------------------
#  Exact commuting (ZZ) ansatz — closed-form fidelity kernel
# ----------------------------------------------------------------------

class ZZKernel:
    """Exact fidelity kernel of the commuting ansatz of Section III-B.

    U(x) = bigotimes_{j=1}^{n} exp(-i * pi/2 * x_{c(j)} Z)  with c(j) = j mod d,
    so psi(x)[z] = 2^{-n/2} exp(-i pi/2 * sum_j x_{c(j)} z_j) and
        K(x,x') = prod_j cos^2( pi/2 * (x_{c(j)} - x'_{c(j)}) ).
    Each coordinate is spread over n/d qubits, so K factorises as
    prod_{j=1}^d cos^{2n/d}( pi/2 * (x_j - x'_j) ).  This is the kernel whose
    spectrum verifies Assumption (i) of Theorem 3 without any surrogate.
    """

    name = 'QIKRR (exact ZZ)'

    def __init__(self, n_qubits=8, input_dim=2, operator_terms=None):
        self.n = n_qubits
        self.d = input_dim
        assert n_qubits % input_dim == 0
        self.k = n_qubits // input_dim          # qubits per coordinate
        self.operator_terms = operator_terms or [(1.0, tuple())]
        self._f2_scale = None

    # -- per-coordinate factor f(Delta) = cos^{2k}(pi Delta / 2) ----------
    def _factors(self, X, X2):
        X = np.atleast_2d(X).astype(float)
        X2 = np.atleast_2d(X2).astype(float)
        z = X[:, None, :] - X2[None, :, :]              # (n, m, d)
        th = 0.5 * np.pi * z
        c, s = np.cos(th), np.sin(th)
        k = self.k
        F0 = c ** (2 * k)
        F1 = -k * np.pi * c ** (2 * k - 1) * s
        F2 = 0.5 * k * np.pi ** 2 * (
            (2 * k - 1) * c ** (2 * k - 2) * s ** 2 - c ** (2 * k))
        # f''' = (pi/2)^3 * g3, sympy: -2k(2k-1)(2k-2)c^{2k-3}s^3 + 4k(3k-1)c^{2k-1}s
        g3 = (-2 * k * (2 * k - 1) * (2 * k - 2) * c ** (2 * k - 3) * s ** 3
              + 4 * k * (3 * k - 1) * c ** (2 * k - 1) * s)
        F3 = (0.5 * np.pi) ** 3 * g3
        # f'''' = (pi/2)^4 * g4, sympy-verified
        g4 = (4 * k * c ** (2 * k - 4)
              * (4 * k ** 3 * s ** 4 + (8 * k - 12 * k ** 2 - 2) * s ** 2
                 + 3 * k - 1))
        F4 = (0.5 * np.pi) ** 4 * g4
        return z, (F0, F1, F2, F3, F4)

    def _mixed(self, X, X2, a, b):
        """prod_l (-1)^{b_l} f_l^{(a_l + b_l)} via the product rule."""
        _, Fs = self._factors(X, X2)
        out = np.ones_like(Fs[0][:, :, 0])
        for l in range(self.d):
            m = a[l] + b[l]
            out = out * ((1 if m == 0 else -1) ** b[l]) * Fs[m][:, :, l]
        return out

    def K(self, X, X2):
        return self._mixed(X, X2, (0,) * self.d, (0,) * self.d)

    def D1K(self, X, X2):
        out = np.zeros_like(self.K(X, X2))
        for c, mi in self.operator_terms:
            out += c * self._mixed(X, X2, mi, (0,) * self.d)
        return out

    def D2K(self, X, X2):
        out = np.zeros_like(self.K(X, X2))
        for c, mi in self.operator_terms:
            out += c * self._mixed(X, X2, (0,) * self.d, mi)
        return out

    def D1D2K(self, X, X2):
        out = np.zeros_like(self.K(X, X2))
        for c1, m1 in self.operator_terms:
            for c2, m2 in self.operator_terms:
                out += c1 * c2 * self._mixed(X, X2, m1, m2)
        return out


# ----------------------------------------------------------------------
#  Physics-informed collocation system (Kansa, unsymmetric)
# ----------------------------------------------------------------------

def build_pi_system(kb, Xd, yd, Xc, gc, lam, pde_weight=1.0):
    """Assemble the physics-informed ridge system.

        [ K(Xd,Xd)          w*D2K(Xd,Xc)    ] [alpha]   [yd]
        [ w*D1K(Xc,Xd)   w^2*D1D2K(Xc,Xc)  ] [beta ] = [w*gc]

    with collocation weight w = pde_weight.  Returns (A, rhs) with ridge
    regularisation lam on the diagonal.
    """
    if len(Xc) == 0:                       # physics-free ablation
        A = kb.K(Xd, Xd) + lam * np.eye(len(Xd))
        return A, np.asarray(yd)
    w = pde_weight
    # row-scale the collocation equations by the rhs magnitude so the
    # collocation weight is dimensionless and conditioning is scale free
    s = 1.0 / max(np.mean(np.abs(gc)), 1e-12)
    A = np.block([
        [kb.K(Xd, Xd), w * kb.D2K(Xd, Xc)],
        [w * s * kb.D1K(Xc, Xd), w * w * s * kb.D1D2K(Xc, Xc)],
    ])
    A = A + lam * np.eye(A.shape[0])
    rhs = np.concatenate([yd, w * s * np.asarray(gc)])
    return A, rhs


def pi_predict(kb, X, Xd, Xc, coef, pde_weight=1.0):
    """Predict u(X) from physics-informed coefficients."""
    n_d = len(Xd)
    alpha, beta = coef[:n_d], coef[n_d:]
    return kb.K(X, Xd) @ alpha + pde_weight * kb.D2K(X, Xc) @ beta


def pi_residual(kb, X, Xd, Xc, coef, pde_weight=1.0):
    """Evaluate the PDE residual D u(X) from physics-informed coefficients."""
    n_d = len(Xd)
    alpha, beta = coef[:n_d], coef[n_d:]
    return kb.D1K(X, Xd) @ alpha + pde_weight * kb.D1D2K(X, Xc) @ beta


if __name__ == '__main__':
    # --- correctness checks -------------------------------------------
    rng = np.random.RandomState(0)
    X = rng.rand(6, 2)
    X2 = rng.rand(5, 2)

    # 1) RBF first-derivative block vs central finite differences
    terms1 = [(1.0, (1, 0)), (0.5, (0, 1))]
    rk1 = RBFKernel(length_scale=0.8, operator_terms=terms1)
    h = 1e-5
    fd = np.zeros((len(X), len(X2)))
    for coef, mi in terms1:
        for d_i, order in enumerate(mi):
            if order != 1:
                continue
            for s in (1, -1):
                Xs = X2.copy()
                Xs[:, d_i] += s * h
                fd += s * coef * rk1.K(X, Xs) / (2 * h)
    print('RBF D2K  FD-vs-analytic max err:', np.abs(fd - rk1.D2K(X, X2)).max())

    # 2) RBF second-derivative block vs 3-point stencil
    terms2 = [(-1.0, (2, 0)), (-1.0, (0, 2))]
    rk2 = RBFKernel(length_scale=0.8, operator_terms=terms2)
    fd2 = np.zeros((len(X), len(X2)))
    for coef, mi in terms2:
        for d_i, order in enumerate(mi):
            if order != 2:
                continue
            Xp = X2.copy(); Xp[:, d_i] += h
            Xm = X2.copy(); Xm[:, d_i] -= h
            fd2 += coef * (rk2.K(X, Xp) - 2 * rk2.K(X, X2) + rk2.K(X, Xm)) / h ** 2
    ana2 = sum(c * rbf_deriv_block(X, X2, rk2.gamma, (0, 0), mi)
               for c, mi in terms2)
    print('RBF D2K(2nd) FD-vs-analytic max err:', np.abs(fd2 - ana2).max())

    # 3) RBF D1D2 self-adjointness for Laplacian: D1D2K should be PSD-ish
    blk = rk2.D1D2K(X, X)
    print('D1D2K symmetric max err:', np.abs(blk - blk.T).max())

    # 4) quantum self-consistency: K = phi . phi'
    qk = QuantumKernel(n_qubits=6, length_scale=0.8, beta=0.2,
                       operator_terms=terms2)
    Kq = qk.K(X, X2)
    f1 = quantum_feature_dict(X, 6, 0.8, 0.2, 42, [(0, 0)])[(0, 0)]
    f2 = quantum_feature_dict(X2, 6, 0.8, 0.2, 42, [(0, 0)])[(0, 0)]
    print('quantum K vs feature dot max err:', np.abs(Kq - f1 @ f2.T).max())
    print('OK')
