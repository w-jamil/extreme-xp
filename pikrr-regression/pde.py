#!/usr/bin/env python3
"""
pde.py — PDE benchmark problems for physics-informed experiments.

Each problem provides a manufactured solution u*, the right-hand side g of the
linear PDE  D u = g  with constant coefficients, and the operator terms as a
list of (coefficient, multi-index) pairs, e.g. -u'' = g  is  [(-1.0, (2,))].

Problems
  Poisson1D      -u''           = g  on (0,1),  u=0 on boundary, u*=sin(pi x)
  AdvecDiff1D    -nu u'' + u'   = g  on (0,1),  u=0 on boundary, nu=0.1
  Poisson2D      -(u_xx + u_yy) = g  on (0,1)^2, u=0 on boundary,
                 u* = sin(pi x) sin(pi y)
"""

import numpy as np


class PDEProblem:
    """Linear constant-coefficient PDE with manufactured solution."""

    def __init__(self, name, dim, domain, solution, rhs, operator_terms, nu=None):
        self.name = name
        self.dim = dim
        self.domain = np.asarray(domain, dtype=float)          # (dim, 2)
        self.solution = solution                               # u*(x)
        self.rhs = rhs                                         # g(x), D u* = g
        self.operator_terms = operator_terms                   # [(coef, multiindex)]
        self.nu = nu

    # ------------------------------------------------------------------
    def apply_operator(self, x, ufun, h=1e-4):
        """Evaluate D ufun at points x via the analytic derivative of ufun."""
        raise NotImplementedError

    def residual_of(self, x, u_vals_fun):
        """Compute D[u](x) for a callable u using finite differences of u."""
        x = np.atleast_2d(x)
        out = np.zeros(len(x))
        for coef, mi in self.operator_terms:
            out += coef * _mixed_partial(u_vals_fun, x, mi, h=1e-4)
        return out

    # ------------------------------------------------------------------
    def sample_data(self, n_interior, noise_std, rng, n_boundary=None):
        """Noisy observations of u* in the interior + exact Dirichlet data."""
        lo, hi = self.domain[:, 0], self.domain[:, 1]
        X_in = lo + (hi - lo) * rng.random((n_interior, self.dim))
        y_in = self.solution(X_in) + noise_std * rng.standard_normal(n_interior)
        Xb, yb = self.sample_boundary(n_boundary, rng)
        X = np.vstack([X_in, Xb]) if len(Xb) else X_in
        y = np.hstack([y_in, yb]) if len(Xb) else y_in
        return X, y

    def sample_boundary(self, n, rng):
        """Exact Dirichlet boundary values (u* = 0 on boundary)."""
        if not n:
            return np.zeros((0, self.dim)), np.zeros(0)
        pts = []
        lo, hi = self.domain[:, 0], self.domain[:, 1]
        per = int(np.ceil(n / (2 * self.dim)))
        for d in range(self.dim):
            for side in (lo[d], hi[d]):
                p = lo + (hi - lo) * rng.random((per, self.dim))
                p[:, d] = side
                pts.append(p)
        X = np.vstack(pts)[:n]
        return X, self.solution(X)

    def sample_collocation(self, n, rng):
        lo, hi = self.domain[:, 0], self.domain[:, 1]
        X = lo + (hi - lo) * rng.random((n, self.dim))
        return X, self.rhs(X)

    def test_grid(self, n):
        if self.dim == 1:
            lo, hi = self.domain[0]
            X = np.linspace(lo, hi, n)[:, None]
        else:
            lo, hi = self.domain[:, 0], self.domain[:, 1]
            m = max(2, int(round(n ** (1.0 / self.dim))))
            axes = [np.linspace(lo[j], hi[j], m) for j in range(self.dim)]
            mesh = np.meshgrid(*axes, indexing='ij')
            X = np.column_stack([A.ravel() for A in mesh])
        return X, self.solution(X)


def _mixed_partial(ufun, x, multiindex, h=1e-4):
    """Mixed partial of ufun at rows of x via successive 1-D central
    differences, one per dimension with nonzero order."""
    from math import comb
    vals = ufun(np.atleast_2d(x)).astype(float)
    pts = np.atleast_2d(x).copy()
    for d, order in enumerate(multiindex):
        if order == 0:
            continue
        new_vals = np.zeros(len(pts))
        for k in range(order + 1):
            coeff = (-1) ** k * comb(order, k)
            offset = (order / 2.0 - k) * h
            shifted = pts.copy()
            shifted[:, d] += offset
            new_vals += coeff * ufun(shifted)
        vals = new_vals / h ** order
        # pts stays the base point for the next dimension
    return vals


# ----------------------------------------------------------------------
#  Problem definitions
# ----------------------------------------------------------------------

def make_problems():
    """Return the three benchmark PDEs."""
    problems = {}

    # --- P1: 1-D Poisson ---------------------------------------------
    def u_p1(x):
        return np.sin(np.pi * x[:, 0])

    def g_p1(x):
        return (np.pi ** 2) * np.sin(np.pi * x[:, 0])

    problems['poisson1d'] = PDEProblem(
        name='Poisson-1D', dim=1, domain=[(0.0, 1.0)],
        solution=u_p1, rhs=g_p1, operator_terms=[(-1.0, (2,))])

    # --- P2: 1-D advection-diffusion ---------------------------------
    nu = 0.1

    def u_p2(x):
        return np.sin(np.pi * x[:, 0]) + 0.3 * np.sin(3 * np.pi * x[:, 0])

    def g_p2(x):
        xx = x[:, 0]
        u = np.sin(np.pi * xx) + 0.3 * np.sin(3 * np.pi * xx)
        up = np.pi * np.cos(np.pi * xx) + 0.9 * np.pi * np.cos(3 * np.pi * xx)
        upp = -(np.pi ** 2) * np.sin(np.pi * xx) - 0.3 * 9 * np.pi ** 2 * np.sin(3 * np.pi * xx)
        return -nu * upp + up

    problems['advecdiff1d'] = PDEProblem(
        name='AdvecDiff-1D', dim=1, domain=[(0.0, 1.0)],
        solution=u_p2, rhs=g_p2, operator_terms=[(-nu, (2,)), (1.0, (1,))], nu=nu)

    # --- P3: 2-D Poisson ---------------------------------------------
    def u_p3(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    def g_p3(x):
        return 2 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

    problems['poisson2d'] = PDEProblem(
        name='Poisson-2D', dim=2, domain=[(0.0, 1.0), (0.0, 1.0)],
        solution=u_p3, rhs=g_p3,
        operator_terms=[(-1.0, (2, 0)), (-1.0, (0, 2))])

    # --- P4: 4-D Poisson ----------------------------------------------
    def u_p4(x):
        out = np.ones(len(x))
        for j in range(4):
            out = out * np.sin(np.pi * x[:, j])
        return out

    def g_p4(x):
        return 4 * np.pi ** 2 * u_p4(x)

    problems['poisson4d'] = PDEProblem(
        name='Poisson-4D', dim=4, domain=[(0.0, 1.0)] * 4,
        solution=u_p4, rhs=g_p4,
        operator_terms=[(-1.0, (2, 0, 0, 0)), (-1.0, (0, 2, 0, 0)),
                        (-1.0, (0, 0, 2, 0)), (-1.0, (0, 0, 0, 2))])

    return problems


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    for key, p in make_problems().items():
        X, y = p.sample_data(10, 0.0, rng, n_boundary=8)
        Xc, gc = p.sample_collocation(50, rng)
        Xt, yt = p.test_grid(400)
        # verify residual_of reproduces g on the test grid
        res = p.residual_of(Xt, p.solution)
        print(f"{key:14s} dim={p.dim}  |D u* - g|_inf on grid = "
              f"{np.abs(res - p.rhs(Xt)).max():.2e}  "
              f"u range [{yt.min():.2f},{yt.max():.2f}]")
