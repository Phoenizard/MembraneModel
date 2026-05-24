"""Woodbury solve for  (D + sum_j alpha_j u_j u_j^T) x = b,
where D is the diagonal FFT operator (symbol L_hat) and the u_j are the
rank-1 vectors from the penalty terms (doc 5.6.2(e)).

    x = D^-1 b - D^-1 U (alpha^-1 + U^T D^-1 U)^-1 U^T D^-1 b

All inner products are the L^2 product (with cell volume dV).
"""

from __future__ import annotations

import numpy as np

from discretization.grid import SpectralGrid


def solve(grid: SpectralGrid, Lhat: np.ndarray, b: np.ndarray, us, alphas) -> np.ndarray:
    Dinv_b = grid.solve_symbol(b, Lhat)
    if not us:
        return Dinv_b

    m = len(us)
    Dinv_u = [grid.solve_symbol(u, Lhat) for u in us]

    # M = diag(1/alpha) + U^T D^-1 U
    M = np.zeros((m, m))
    for i in range(m):
        M[i, i] += 1.0 / alphas[i]
        for j in range(m):
            M[i, j] += grid.inner(us[i], Dinv_u[j])

    rhs = np.array([grid.inner(us[i], Dinv_b) for i in range(m)])
    coeffs = np.linalg.solve(M, rhs)

    x = Dinv_b.copy()
    for j in range(m):
        x -= coeffs[j] * Dinv_u[j]
    return x
