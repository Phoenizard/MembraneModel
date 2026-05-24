"""Pointwise field primitives shared across the energy/variation kernel.

Symbols follow doc/note/IEQ_scheme_vesicle.md:
    g(phi)  = (1/eps) phi (1 - phi^2)
    g'(phi) = (1/eps) (1 - 3 phi^2)
    W(phi)  = (eps/2)|grad phi|^2 + (1/4eps)(phi^2-1)^2
    m(phi)  = eps*lap phi + g(phi)        (bending integrand is (k/2eps) m^2)
"""

from __future__ import annotations

import numpy as np

from discretization.grid import SpectralGrid
from params import Params


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
