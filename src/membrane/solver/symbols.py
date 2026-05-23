"""Constant-coefficient skeleton symbols (doc 5.6.2 / 5.6.3).

Stage A phi-substep diagonal symbol:
    L_hat(k) = 1/tau + sigma + abar^2 (eps|k|^2 - gbar')^2 + b_L|k|^2
with b_L = 0 (no line tension).  abar^2 = k/eps is exactly constant for c=0;
gbar' is the spatial mean of g'(phi^n) (a tunable representative, doc 5.8).
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from ..model import energy


def gbar_prime(phi: np.ndarray, p: Params) -> float:
    """Spatial mean of g'(phi) = (1/eps)(1 - 3 phi^2)."""
    return float(np.mean(energy.gp(phi, p)))


def skeleton_symbol(grid: SpectralGrid, p: Params, gbar: float) -> np.ndarray:
    """S_E spectral symbol: abar^2 (eps|k|^2 - gbar')^2  (no 1/tau, sigma)."""
    return p.a2 * (p.eps * grid.K2 - gbar) ** 2


def L_hat(grid: SpectralGrid, p: Params, gbar: float) -> np.ndarray:
    return (1.0 / p.tau + p.sigma) + skeleton_symbol(grid, p, gbar)
