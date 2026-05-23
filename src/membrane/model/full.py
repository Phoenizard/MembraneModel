"""Full two-component energy E_M(phi, eta) and the variational derivatives
delta E_M/delta phi (doc 4.1.7) and delta E_M/delta eta (doc 4.2.6).

Reuses the densities g, g', W from energy.py. eta enters via
    k(eta)   = k + c tanh(eta/xi)        bending stiffness
    Phi_eta  = delta((xi/2)|grad eta|^2 + (1/4xi)(eta^2-1)^2)
    Pi(eta)  = (xi/2)|grad eta|^2 - (1/4xi)(eta^2-1)^2
"""

from __future__ import annotations

import numpy as np

from ..discretization.grid import SpectralGrid
from ..params import Params
from .energy import g, gp, W_density, curvature


# --- eta-dependent coefficients ---
def tanh_eta(eta, p: Params):
    return np.tanh(eta / p.xi)


def sech2_eta(eta, p: Params):
    return 1.0 - np.tanh(eta / p.xi) ** 2


def k_eta(eta, p: Params):
    return p.k + p.c * tanh_eta(eta, p)


def kp_eta(eta, p: Params):
    """k'(eta) = (c/xi) sech^2(eta/xi)."""
    return (p.c / p.xi) * sech2_eta(eta, p)


def _eta_bracket(eta, grid: SpectralGrid, p: Params):
    """(xi/2)|grad eta|^2 and (1/4xi)(eta^2-1)^2 split, plus |grad eta|^2."""
    ex, ey, ez = grid.grad(eta)
    grad2 = ex**2 + ey**2 + ez**2
    grad_term = 0.5 * p.xi * grad2
    well_term = (1.0 / (4.0 * p.xi)) * (eta**2 - 1.0) ** 2
    return grad_term, well_term


def Phi_eta(eta, grid: SpectralGrid, p: Params):
    gt, wt = _eta_bracket(eta, grid, p)
    return p.delta * (gt + wt)


def Pi_eta(eta, grid: SpectralGrid, p: Params):
    gt, wt = _eta_bracket(eta, grid, p)
    return gt - wt


# --- functionals ---
def E_bending(phi, eta, grid, p):
    m = curvature(phi, grid, p)
    return grid.integral((k_eta(eta, p) / (2.0 * p.eps)) * m**2)


def E_line(phi, eta, grid, p):
    return grid.integral(Phi_eta(eta, grid, p) * W_density(phi, grid, p))


def area(phi, grid, p):
    return grid.integral(W_density(phi, grid, p))


def volume(phi, grid):
    return grid.integral(phi)


def area_diff(phi, eta, grid, p):
    return grid.integral(tanh_eta(eta, p) * W_density(phi, grid, p))


def N_func(phi, eta, grid, p):
    sdot = grid.grad_dot_grad(phi, eta)
    return grid.integral(0.5 * p.eps * sdot**2)


def P_func(eta, grid, p):
    return grid.integral(Pi_eta(eta, grid, p) ** 2)


def E_total(phi, eta, grid, p):
    A = area(phi, grid, p)
    V = volume(phi, grid)
    D = area_diff(phi, eta, grid, p)
    N = N_func(phi, eta, grid, p)
    P = P_func(eta, grid, p)
    return (
        E_bending(phi, eta, grid, p)
        + E_line(phi, eta, grid, p)
        + 0.5 * p.M1 * (V - p.v_d) ** 2
        + 0.5 * p.M2 * (A - p.a_0) ** 2
        + 0.5 * p.M3 * (D - p.a_d) ** 2
        + 0.5 * p.M4 * N**2
        + 0.5 * p.M5 * P**2
    )


def energy_breakdown(phi, eta, grid, p):
    A = area(phi, grid, p)
    V = volume(phi, grid)
    D = area_diff(phi, eta, grid, p)
    N = N_func(phi, eta, grid, p)
    P = P_func(eta, grid, p)
    return {
        "E_bend": E_bending(phi, eta, grid, p),
        "E_line": E_line(phi, eta, grid, p),
        "E_M": E_total(phi, eta, grid, p),
        "A": A, "V": V, "D": D, "N": N, "P": P,
    }


# --- constraint-gradient directions (rank-1 Newton vectors, doc 5.2/5.5) ---
# These are the bracket factors delta C_j / delta field (no penalty prefactor).
def dV_dphi(phi, grid, p):
    return np.ones_like(phi)


def dA_dphi(phi, grid, p):
    return -p.eps * grid.lap(phi) + (1.0 / p.eps) * phi * (phi**2 - 1.0)


def dD_dphi(phi, eta, grid, p):
    th = tanh_eta(eta, p)
    px, py, pz = grid.grad(phi)
    return -p.eps * grid.div(th * px, th * py, th * pz) + (1.0 / p.eps) * phi * (phi**2 - 1.0) * th


def dN_dphi(phi, eta, grid, p):
    px, py, pz = grid.grad(phi)
    ex, ey, ez = grid.grad(eta)
    sdot = px * ex + py * ey + pz * ez
    return -p.eps * grid.div(sdot * ex, sdot * ey, sdot * ez)


def dD_deta(phi, eta, grid, p):
    W = W_density(phi, grid, p)
    return (1.0 / p.xi) * W * sech2_eta(eta, p)


def dN_deta(phi, eta, grid, p):
    px, py, pz = grid.grad(phi)
    ex, ey, ez = grid.grad(eta)
    sdot = px * ex + py * ey + pz * ez
    return -p.eps * grid.div(sdot * px, sdot * py, sdot * pz)


def dP_deta(eta, grid, p):
    Pi = Pi_eta(eta, grid, p)
    ex, ey, ez = grid.grad(eta)
    return -2.0 * p.xi * grid.div(Pi * ex, Pi * ey, Pi * ez) - (2.0 / p.xi) * Pi * eta * (eta**2 - 1.0)


# --- variational derivative wrt phi (doc 4.1.7) ---
def dEM_dphi(phi, eta, grid, p):
    kk = k_eta(eta, p)
    m = curvature(phi, grid, p)
    W = W_density(phi, grid, p)
    A = area(phi, grid, p)
    V = volume(phi, grid)
    D = area_diff(phi, eta, grid, p)
    N = N_func(phi, eta, grid, p)
    th = tanh_eta(eta, p)
    Phi = Phi_eta(eta, grid, p)
    dwell = (1.0 / p.eps) * phi * (phi**2 - 1.0)  # (1/eps) phi(phi^2-1)

    px, py, pz = grid.grad(phi)
    ex, ey, ez = grid.grad(eta)
    sdot = px * ex + py * ey + pz * ez

    # bending
    out = grid.lap(kk * m) + (kk / p.eps) * gp(phi, p) * m
    # line tension
    out += -p.eps * grid.div(Phi * px, Phi * py, Phi * pz) + dwell * Phi
    # volume
    out += p.M1 * (V - p.v_d)
    # area
    out += p.M2 * (A - p.a_0) * (-p.eps * grid.lap(phi) + dwell)
    # area difference
    out += p.M3 * (D - p.a_d) * (
        -p.eps * grid.div(th * px, th * py, th * pz) + dwell * th
    )
    # orthogonality N
    out += -p.M4 * N * p.eps * grid.div(sdot * ex, sdot * ey, sdot * ez)
    return out


# --- variational derivative wrt eta (doc 4.2.6) ---
def dEM_deta(phi, eta, grid, p):
    m = curvature(phi, grid, p)
    W = W_density(phi, grid, p)
    D = area_diff(phi, eta, grid, p)
    N = N_func(phi, eta, grid, p)
    P = P_func(eta, grid, p)
    Pi = Pi_eta(eta, grid, p)
    sech2 = sech2_eta(eta, p)
    ewell = (1.0 / p.xi) * eta * (eta**2 - 1.0)  # (1/xi) eta(eta^2-1)

    px, py, pz = grid.grad(phi)
    ex, ey, ez = grid.grad(eta)
    sdot = px * ex + py * ey + pz * ez

    # bending: k'(eta)/(2eps) m^2
    out = (kp_eta(eta, p) / (2.0 * p.eps)) * m**2
    # line tension: -delta xi div(W grad eta) + delta (1/xi) W eta(eta^2-1)
    out += -p.delta * p.xi * grid.div(W * ex, W * ey, W * ez) + p.delta * W * ewell
    # area difference: M3 (D-a_d) (1/xi) W sech^2
    out += p.M3 * (D - p.a_d) * (1.0 / p.xi) * W * sech2
    # orthogonality N: -M4 N eps div((grad phi . grad eta) grad phi)
    out += -p.M4 * N * p.eps * grid.div(sdot * px, sdot * py, sdot * pz)
    # P term: M5 P (-2 xi div(Pi grad eta) - (2/xi) Pi eta(eta^2-1))
    out += p.M5 * P * (
        -2.0 * p.xi * grid.div(Pi * ex, Pi * ey, Pi * ez)
        - (2.0 / p.xi) * Pi * eta * (eta**2 - 1.0)
    )
    return out
