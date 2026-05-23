"""Energy densities, bending energy, total penalised energy, and the
continuous variational derivative delta E_M / delta phi (Stage A: c=0).

Symbols follow doc/note/IEQ_scheme_vesicle.md:
    g(phi)  = (1/eps) phi (1 - phi^2)
    g'(phi) = (1/eps) (1 - 3 phi^2)
    W(phi)  = (eps/2)|grad phi|^2 + (1/4eps)(phi^2-1)^2
    E       = int (k/2eps)(eps*lap phi + g)^2
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from . import constraints


def g(phi: np.ndarray, p: Params) -> np.ndarray:
    return (1.0 / p.eps) * phi * (1.0 - phi**2)


def gp(phi: np.ndarray, p: Params) -> np.ndarray:
    return (1.0 / p.eps) * (1.0 - 3.0 * phi**2)


def W_density(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    gx, gy, gz = grid.grad(phi)
    grad2 = gx**2 + gy**2 + gz**2
    return 0.5 * p.eps * grad2 + (1.0 / (4.0 * p.eps)) * (phi**2 - 1.0) ** 2


def curvature(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    """m = eps*lap phi + g(phi); the bending integrand is (k/2eps) m^2."""
    return p.eps * grid.lap(phi) + g(phi, p)


def E_bending(phi: np.ndarray, grid: SpectralGrid, p: Params) -> float:
    m = curvature(phi, grid, p)
    return grid.integral((p.k / (2.0 * p.eps)) * m**2)


def E_total(phi: np.ndarray, grid: SpectralGrid, p: Params) -> float:
    """Penalised energy E_M (Stage A: bending + volume + area penalties)."""
    A = constraints.area(phi, grid, p)
    V = constraints.volume(phi, grid)
    return (
        E_bending(phi, grid, p)
        + 0.5 * p.M1 * (V - p.v_d) ** 2
        + 0.5 * p.M2 * (A - p.a_0) ** 2
    )


def energy_breakdown(phi: np.ndarray, grid: SpectralGrid, p: Params) -> dict:
    A = constraints.area(phi, grid, p)
    V = constraints.volume(phi, grid)
    return {
        "E_bend": E_bending(phi, grid, p),
        "E_line": 0.0,  # Stage A: no line tension (filled in Stage B)
        "E_M": E_total(phi, grid, p),
        "A": A,
        "V": V,
    }


def variational_derivative(phi: np.ndarray, grid: SpectralGrid, p: Params) -> np.ndarray:
    """Continuous delta E_M / delta phi (Stage A).

    bending: k*lap(m) + (k/eps) g'(phi) m,  m = eps*lap phi + g
    area:    M2 (A-a0) * dA/dphi,  dA/dphi = -eps*lap phi + (1/eps) phi(phi^2-1)
    volume:  M1 (V-v_d) * 1
    """
    m = curvature(phi, grid, p)
    bend = p.k * grid.lap(m) + (p.k / p.eps) * gp(phi, p) * m
    dA = constraints.dA_dphi(phi, grid, p)
    A = constraints.area(phi, grid, p)
    V = constraints.volume(phi, grid)
    return bend + p.M2 * (A - p.a_0) * dA + p.M1 * (V - p.v_d)
