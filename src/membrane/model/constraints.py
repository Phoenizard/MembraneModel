"""Constraint functionals and their variational derivatives (Stage A: A, V).

    A(phi) = int W(phi) dx        dA/dphi = -eps*lap phi + (1/eps) phi(phi^2-1)
    V(phi) = int phi dx           dV/dphi = 1

Stage B will add D(phi,eta), N, P here.
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params


def area(phi: np.ndarray, grid: SpectralGrid, p: Params) -> float:
    from .energy import W_density

    return grid.integral(W_density(phi, grid, p))


def volume(phi: np.ndarray, grid: SpectralGrid) -> float:
    return grid.integral(phi)


def dA_dphi(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    return -p.eps * grid.lap(phi) + (1.0 / p.eps) * phi * (phi**2 - 1.0)


def residuals(phi: np.ndarray, grid: SpectralGrid, p: Params) -> dict:
    return {
        "V-v_d": volume(phi, grid) - p.v_d,
        "A-a_0": area(phi, grid, p) - p.a_0,
    }
