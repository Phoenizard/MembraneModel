"""Energy / constraint / auxiliary diagnostics and a one-line status string."""

from __future__ import annotations

import numpy as np

from .discretization.grid import SpectralGrid
from .params import Params
from .model import auxiliary, constraints, energy


def report(phi: np.ndarray, grid: SpectralGrid, p: Params) -> dict:
    bd = energy.energy_breakdown(phi, grid, p)
    res = constraints.residuals(phi, grid, p)
    r = auxiliary.r_var(phi, grid, p)
    return {
        **bd,
        **res,
        # Lambda_i -> Lagrange multipliers as M_i -> inf (cf. paper Table 1)
        "Lambda1": p.M1 * res["V-v_d"],
        "Lambda2": p.M2 * res["A-a_0"],
        "r_min": float(r.min()),
        "phi_absmax": float(np.abs(phi).max()),
    }


def line(phi: np.ndarray, grid: SpectralGrid, p: Params) -> str:
    d = report(phi, grid, p)
    return (
        f"E_M={d['E_M']:.4f} E_b={d['E_bend']:.4f} "
        f"V-v_d={d['V-v_d']:+.3e} A-a_0={d['A-a_0']:+.3e} r_min={d['r_min']:.3e}"
    )
