"""IEQ coupled steppers (two fields phi, eta; decoupled substeps).

Each substep is a damped modified-Newton / preconditioned-descent step using
the EXACT validated gradient:

    field^{n+1} = field^n - (P + H)^{-1} G

  G = delta E_M / delta field        (energy.dEM_d*; FD-validated)
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

from discretization.grid import SpectralGrid
from params import Params
from model import energy, fields
from . import woodbury


def _phi_precond(phi, eta, grid, p):
    abar2 = float(np.mean(energy.k_eta(eta, p))) / p.eps
    gbar = float(np.mean(fields.gp(phi, p)))
    bL = p.eps * float(np.mean(energy.Phi_eta(eta, grid, p)))
    return (1.0 / p.tau + p.sigma) + abar2 * (p.eps * grid.K2 - gbar) ** 2 + bL * grid.K2


def _eta_precond(phi, eta, grid, p):
    bL = p.delta * p.xi * float(np.mean(fields.W_density(phi, grid, p)))
    return (1.0 / p.tau + p.sigma) + bL * grid.K2


def _active(us, alphas):
    """Drop rank-1 terms whose penalty is zero (constraint turned off)."""
    pairs = [(u, a) for u, a in zip(us, alphas) if a > 0.0]
    if not pairs:
        return [], []
    u_keep, a_keep = zip(*pairs)
    return list(u_keep), list(a_keep)


def phi_step(phi, eta, grid: SpectralGrid, p: Params):
    G = energy.dEM_dphi(phi, eta, grid, p)
    P = _phi_precond(phi, eta, grid, p)
    us = [
        energy.dV_dphi(phi, grid, p),
        energy.dA_dphi(phi, grid, p),
        energy.dD_dphi(phi, eta, grid, p),
        energy.dN_dphi(phi, eta, grid, p),
    ]
    us, alphas = _active(us, [p.M1, p.M2, p.M3, p.M4])
    x = woodbury.solve(grid, P, G, us, alphas)
    return phi - x


def eta_step(phi, eta, grid: SpectralGrid, p: Params):
    G = energy.dEM_deta(phi, eta, grid, p)
    P = _eta_precond(phi, eta, grid, p)
    us = [
        energy.dD_deta(phi, eta, grid, p),
        energy.dN_deta(phi, eta, grid, p),
        energy.dP_deta(eta, grid, p),
    ]
    us, alphas = _active(us, [p.M3, p.M4, p.M5])
    x = woodbury.solve(grid, P, G, us, alphas)
    return eta - x


def line_from(bd, p):
    """One-line status from a precomputed energy_breakdown dict."""
    return (
        f"E_M={bd['E_M']:.4f} E_b={bd['E_bend']:.4f} E_l={bd['E_line']:.4f} "
        f"V-v_d={bd['V']-p.v_d:+.2e} A-a_0={bd['A']-p.a_0:+.2e} "
        f"D-a_d={bd['D']-p.a_d:+.2e} N={bd['N']:.2e} P={bd['P']:.2e}"
    )


def line(phi, eta, grid, p):
    return line_from(energy.energy_breakdown(phi, eta, grid, p), p)


def _metrics(bd, p, d, d_rel):
    """Flat metric dict for the monitor (keys 'E_M' and 'd' drive detection)."""
    return {
        "E_M": bd["E_M"], "E_bend": bd["E_bend"], "E_line": bd["E_line"],
        "V_res": bd["V"] - p.v_d, "A_res": bd["A"] - p.a_0, "D_res": bd["D"] - p.a_d,
        "N": bd["N"], "P": bd["P"], "d": d, "d_rel": d_rel,
    }


def steady_solve(phi, eta, grid, p, *, max_steps, log_every=0, monitor=None,
                 patience=8, e_rtol=1e-3):
    """Inner phi/eta flow to PRACTICAL convergence via an energy plateau
    (ML-style early stopping): E_M is sampled every log_every steps; we track
    its best (lowest) value and stop once `patience` consecutive samples fail to
    beat it by > e_rtol (relative) — i.e. the energy can no longer go lower.

    Rationale: the per-step field change |d| stays ~constant through the whole
    descent (the interface moves slowly but steadily), so thresholding it stops
    far too early. E_M is the quantity we minimise, so its plateau is the right
    signal. |d|/d_rel are still logged, just not used to decide convergence.
    Needs log_every>0 (E_M is sampled on that cadence)."""
    d = d_rel = float("nan")
    e_best = float("inf")
    stale = 0          # consecutive samples with no real new low
    warned = False
    for step in range(1, max_steps + 1):
        phi_new = phi_step(phi, eta, grid, p)
        eta_new = eta_step(phi_new, eta, grid, p)  # D3: eta-substep uses phi^{n+1}
        d = grid.l2norm(phi_new - phi) + grid.l2norm(eta_new - eta)
        phi, eta = phi_new, eta_new
        if not np.isfinite(d):
            raise FloatingPointError(f"non-finite at step {step}")
        if log_every and step % log_every == 0:
            bd = energy.energy_breakdown(phi, eta, grid, p)
            scale = grid.l2norm(phi) + grid.l2norm(eta)
            d_rel = d / scale if scale > 0 else d
            print(f"    step {step:6d}  {line_from(bd, p)}  |d|={d:.2e} d_rel={d_rel:.2e}")
            if monitor is not None:
                monitor.log(step, _metrics(bd, p, d, d_rel), m_level=p.M1)
                stop, reason = monitor.should_stop()
                if stop and not warned:
                    print(f"    ⚠ monitor: {reason}")
                    warned = True
            E = bd["E_M"]
            if E < e_best * (1.0 - e_rtol):   # a real new low
                e_best, stale = E, 0
            else:
                stale += 1
                if stale >= patience:
                    return phi, eta, step, d, True
    return phi, eta, max_steps, d, False


def continuation(phi, eta, grid, p, *, M_start, M_max, rho=4.0,
                 e_rtol=1e-3, patience=8, outer_tol=1e-3, max_inner=20000,
                 log_every=0, monitor=None):
    """Raise M1=M2=M3 geometrically; M4=M5 (regularisers) stay at their value."""
    M = M_start
    history = []
    while True:
        p.M1 = p.M2 = p.M3 = M
        phi, eta, steps, d, conv = steady_solve(
            phi, eta, grid, p, max_steps=max_inner, log_every=log_every,
            monitor=monitor, patience=patience, e_rtol=e_rtol,
        )
        bd = energy.energy_breakdown(phi, eta, grid, p)
        res = {"V-v_d": bd["V"] - p.v_d, "A-a_0": bd["A"] - p.a_0, "D-a_d": bd["D"] - p.a_d}
        history.append({"M": M, "steps": steps, **res, "E_M": bd["E_M"]})
        print(f"  M={M:.2e} steps={steps} conv={conv} {line(phi, eta, grid, p)}")
        ok = all(abs(res[k]) < outer_tol for k in res)
        if ok or M >= M_max:
            return phi, eta, history
        M = min(M * rho, M_max)
