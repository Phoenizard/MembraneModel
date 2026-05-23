"""phi-substep of the decoupled IEQ scheme (Stage A: c=0, no eta).

Linear system  ( (1/tau+sigma) I + S_E + M2 e_A (x) e_A + M1 1 (x) 1 ) phi^{n+1} = b
with  b = (1/tau+sigma) phi^n + S_E[phi^n] - Cq*[q^n]
          - M2 (A^n-a0) e_A + M2 (e_A,phi^n) e_A + M1 v_d.

Derivation (doc 5.1-5.3, 5.6): the variable-coefficient bending operator's
remainder cancels under old-layer lagging, leaving the exact explicit residual
Cq*[q^n] (= bending part of delta E_M/delta phi at phi^n) plus the constant
skeleton correction S_E[phi^{n+1}] - S_E[phi^n]. Steady state phi^{n+1}=phi^n
reduces the system to delta E_M/delta phi = 0 (verified in tests).
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from ..model import auxiliary, constraints, energy
from . import symbols, woodbury


def _Cq_star_q(phi: np.ndarray, q: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    """C_q^*[q] = a (eps*lap q + g'(phi) q),  a = sqrt(k/eps)."""
    a = np.sqrt(p.k / p.eps)
    return a * (p.eps * grid.lap(q) + energy.gp(phi, p) * q)


def phi_substep(phi: np.ndarray, grid: SpectralGrid, p: Params):
    A = constraints.area(phi, grid, p)
    V = constraints.volume(phi, grid)
    e_A = constraints.dA_dphi(phi, grid, p)
    ones = np.ones_like(phi)

    q = auxiliary.q_var(phi, grid, p)
    cqq = _Cq_star_q(phi, q, grid, p)

    gbar = symbols.gbar_prime(phi, p)
    Lhat = symbols.L_hat(grid, p, gbar)
    Shat = symbols.skeleton_symbol(grid, p, gbar)
    S_E_phi = grid.apply_symbol(phi, Shat)

    b = (
        (1.0 / p.tau + p.sigma) * phi
        + S_E_phi
        - cqq
        - p.M2 * (A - p.a_0) * e_A
        + p.M2 * grid.inner(e_A, phi) * e_A
        + p.M1 * p.v_d * ones
    )

    us = [e_A, ones]
    alphas = [p.M2, p.M1]
    phi_new = woodbury.solve(grid, Lhat, b, us, alphas)
    return phi_new


def apply_LHS(phi_test, e_A, ones, Lhat, grid, p):
    """LHS operator (for residual verification in tests)."""
    D = grid.apply_symbol(phi_test, Lhat)
    return D + p.M2 * grid.inner(e_A, phi_test) * e_A + p.M1 * grid.inner(ones, phi_test) * ones
