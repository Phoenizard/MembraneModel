"""Stage B coupled steppers (two fields, decoupled substeps).

Each substep is a damped modified-Newton / preconditioned-descent step using
the EXACT validated gradient:

    field^{n+1} = field^n - (P + H)^{-1} G

  G = delta E_M / delta field        (full.dEM_d*; FD-validated)
  P = (1/tau + sigma) I + L_diff      diagonal spectral skeleton (bending 4th
                                      order + line-tension 2nd order)
  H = sum_j M_j e_j (x) e_j           penalty Hessian principal part (rank-1,
                                      e_j = constraint-gradient directions)

The fixed point is G=0 (= true critical point of E_M) for ANY SPD P,H, so
correctness rests on the validated gradient; P,H only set stability/speed.
Solved with the Woodbury identity. Decoupling (doc D3): the eta-substep uses
phi^{n+1}.
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from ..model import full, energy
from . import woodbury


def _phi_precond(phi, eta, grid, p):
    abar2 = float(np.mean(full.k_eta(eta, p))) / p.eps
    gbar = float(np.mean(energy.gp(phi, p)))
    bL = p.eps * float(np.mean(full.Phi_eta(eta, grid, p)))
    return (1.0 / p.tau + p.sigma) + abar2 * (p.eps * grid.K2 - gbar) ** 2 + bL * grid.K2


def _eta_precond(phi, eta, grid, p):
    bL = p.delta * p.xi * float(np.mean(energy.W_density(phi, grid, p)))
    return (1.0 / p.tau + p.sigma) + bL * grid.K2


def _active(us, alphas):
    """Drop rank-1 terms whose penalty is zero (constraint turned off)."""
    pairs = [(u, a) for u, a in zip(us, alphas) if a > 0.0]
    if not pairs:
        return [], []
    u_keep, a_keep = zip(*pairs)
    return list(u_keep), list(a_keep)


def phi_step(phi, eta, grid: SpectralGrid, p: Params):
    G = full.dEM_dphi(phi, eta, grid, p)
    P = _phi_precond(phi, eta, grid, p)
    us = [
        full.dV_dphi(phi, grid, p),
        full.dA_dphi(phi, grid, p),
        full.dD_dphi(phi, eta, grid, p),
        full.dN_dphi(phi, eta, grid, p),
    ]
    us, alphas = _active(us, [p.M1, p.M2, p.M3, p.M4])
    x = woodbury.solve(grid, P, G, us, alphas)
    return phi - x


def eta_step(phi, eta, grid: SpectralGrid, p: Params):
    G = full.dEM_deta(phi, eta, grid, p)
    P = _eta_precond(phi, eta, grid, p)
    us = [
        full.dD_deta(phi, eta, grid, p),
        full.dN_deta(phi, eta, grid, p),
        full.dP_deta(eta, grid, p),
    ]
    us, alphas = _active(us, [p.M3, p.M4, p.M5])
    x = woodbury.solve(grid, P, G, us, alphas)
    return eta - x


def line(phi, eta, grid, p):
    d = full.energy_breakdown(phi, eta, grid, p)
    return (
        f"E_M={d['E_M']:.4f} E_b={d['E_bend']:.4f} E_l={d['E_line']:.4f} "
        f"V-v_d={d['V']-p.v_d:+.2e} A-a_0={d['A']-p.a_0:+.2e} "
        f"D-a_d={d['D']-p.a_d:+.2e} N={d['N']:.2e} P={d['P']:.2e}"
    )


def steady_solve(phi, eta, grid, p, *, tol, max_steps, log_every=0):
    for step in range(1, max_steps + 1):
        phi_new = phi_step(phi, eta, grid, p)
        eta_new = eta_step(phi_new, eta, grid, p)  # D3: eta-substep uses phi^{n+1}
        d = grid.l2norm(phi_new - phi) + grid.l2norm(eta_new - eta)
        phi, eta = phi_new, eta_new
        if log_every and step % log_every == 0:
            print(f"    step {step:6d}  {line(phi, eta, grid, p)}  |d|={d:.2e}")
        if not np.isfinite(d):
            raise FloatingPointError(f"non-finite at step {step}")
        if d < tol:
            return phi, eta, step, d, True
    return phi, eta, max_steps, d, False


def continuation(phi, eta, grid, p, *, M_start, M_max, rho=4.0,
                 inner_tol=1e-7, outer_tol=1e-3, max_inner=20000, log_every=0):
    """Raise M1=M2=M3 geometrically; M4=M5 (regularisers) stay at their value."""
    M = M_start
    history = []
    while True:
        p.M1 = p.M2 = p.M3 = M
        phi, eta, steps, d, conv = steady_solve(
            phi, eta, grid, p, tol=inner_tol, max_steps=max_inner, log_every=log_every
        )
        bd = full.energy_breakdown(phi, eta, grid, p)
        res = {"V-v_d": bd["V"] - p.v_d, "A-a_0": bd["A"] - p.a_0, "D-a_d": bd["D"] - p.a_d}
        history.append({"M": M, "steps": steps, **res, "E_M": bd["E_M"]})
        print(f"  M={M:.2e} steps={steps} conv={conv} {line(phi, eta, grid, p)}")
        ok = all(abs(res[k]) < outer_tol for k in res)
        if ok or M >= M_max:
            return phi, eta, history
        M = min(M * rho, M_max)
