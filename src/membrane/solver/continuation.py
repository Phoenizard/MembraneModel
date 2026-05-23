"""Inner steady-state loop + outer penalty continuation (doc 5.7)."""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from ..model import constraints
from .stepper import phi_substep


def steady_solve(phi, grid: SpectralGrid, p: Params, *, tol, max_steps, log_every=0):
    """Advance phi-flow at fixed penalties until ||delta phi|| < tol."""
    for step in range(1, max_steps + 1):
        phi_new = phi_substep(phi, grid, p)
        dnorm = grid.l2norm(phi_new - phi)
        phi = phi_new
        if log_every and step % log_every == 0:
            from ..diagnostics import line

            print(f"    step {step:5d}  {line(phi, grid, p)}  |dphi|={dnorm:.3e}")
        if dnorm < tol:
            return phi, step, dnorm, True
    return phi, max_steps, dnorm, False


def continuation(
    phi,
    grid: SpectralGrid,
    p: Params,
    *,
    M_start,
    M_max,
    rho=4.0,
    inner_tol=1e-7,
    outer_tol=1e-3,
    max_inner=20000,
    log_every=0,
):
    """Raise M1=M2 geometrically until constraint residuals < outer_tol."""
    M = M_start
    history = []
    while True:
        p.M1 = p.M2 = M
        phi, steps, dnorm, conv = steady_solve(
            phi, grid, p, tol=inner_tol, max_steps=max_inner, log_every=log_every
        )
        res = constraints.residuals(phi, grid, p)
        history.append({"M": M, "steps": steps, "dphi": dnorm, **res})
        print(
            f"  M={M:.3e}  steps={steps}  conv={conv}  "
            f"|V-v_d|={abs(res['V-v_d']):.3e}  |A-a_0|={abs(res['A-a_0']):.3e}"
        )
        ok = abs(res["V-v_d"]) < outer_tol and abs(res["A-a_0"]) < outer_tol
        if ok or M >= M_max:
            return phi, history
        M = min(M * rho, M_max)
