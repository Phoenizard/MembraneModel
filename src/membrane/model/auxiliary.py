"""IEQ auxiliary variables (Stage A uses q, r; Stage B adds s, p).

    q = sqrt(k/eps) (eps*lap phi + g)        ->  E = 1/2 ||q||^2
    r = sqrt(2 W(phi) + C0)                  ->  A = 1/2 int (r^2 - C0)

The area-penalty rank-1 vector e_A = C_r^*[r^n] reduces exactly to dA/dphi
(the C0 / r cancel when the argument is r itself), so Stage A computes it
directly via constraints.dA_dphi.
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from . import energy


def q_var(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    return np.sqrt(p.k / p.eps) * energy.curvature(phi, grid, p)


def r_var(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    return np.sqrt(2.0 * energy.W_density(phi, grid, p) + p.C0)
