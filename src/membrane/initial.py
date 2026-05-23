"""Initial conditions. Stage A: a single spherical vesicle.

phi^0 = tanh((R - |x|)/(sqrt2 eps)),  phi=+1 inside |x|<R.
Radius from the volume target:  |Omega_i| = (|Omega| + v_d)/2,
R = (3 |Omega_i| / 4pi)^(1/3).
"""

from __future__ import annotations

import numpy as np

from .discretization.grid import SpectralGrid
from .params import Params


def sphere_radius_from_vd(grid: SpectralGrid, p: Params) -> float:
    box_vol = grid.length**3
    omega_i = 0.5 * (box_vol + p.v_d)
    return (3.0 * omega_i / (4.0 * np.pi)) ** (1.0 / 3.0)


def sphere(grid: SpectralGrid, p: Params, radius: float | None = None) -> np.ndarray:
    R = sphere_radius_from_vd(grid, p) if radius is None else radius
    rr = np.sqrt(grid.X**2 + grid.Y**2 + grid.Z**2)
    return np.tanh((R - rr) / (np.sqrt(2.0) * p.eps))


def eta_two_phase(grid: SpectralGrid, p: Params) -> np.ndarray:
    """Upper/lower hemisphere split: eta = tanh(z/(sqrt2 eps)).

    By z -> -z symmetry the two phases have ~equal area (D ~ 0), matching the
    small a_d of the Fig. 2 convergence test.
    """
    return np.tanh(grid.Z / (np.sqrt(2.0) * p.eps))
