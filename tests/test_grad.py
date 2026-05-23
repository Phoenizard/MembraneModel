"""Core correctness gate: the coded variational derivative must equal the
finite-difference gradient of E_total, and the Woodbury solve must invert the
phi-substep LHS. These validate the math->code transcription independent of
any physical target.
"""

import numpy as np

from membrane import SpectralGrid, Params
from membrane.model import energy, constraints
from membrane.solver import symbols, stepper, woodbury


def _smooth_field(g):
    return 0.6 * np.sin(g.X) * np.cos(g.Y) * np.sin(g.Z) + 0.2 * np.cos(2 * g.X)


def test_variational_derivative_matches_fd():
    g = SpectralGrid(16)
    p = Params(eps=0.3, k=1.0, c=0.0, M1=100.0, M2=100.0, v_d=-50.0, a_0=20.0)
    phi = _smooth_field(g)
    u = np.sin(g.X + 0.7) * np.cos(g.Y - 0.3) * np.cos(g.Z + 0.2)

    grad = energy.variational_derivative(phi, g, p)
    analytic = g.inner(grad, u)  # dE[u] = (delta E/delta phi, u)

    h = 1e-5
    fd = (energy.E_total(phi + h * u, g, p) - energy.E_total(phi - h * u, g, p)) / (2 * h)

    assert abs(fd - analytic) <= 1e-6 * (1.0 + abs(analytic))


def test_dA_dphi_matches_fd():
    g = SpectralGrid(16)
    p = Params(eps=0.3)
    phi = _smooth_field(g)
    u = np.cos(g.X) * np.sin(2 * g.Y)
    analytic = g.inner(constraints.dA_dphi(phi, g, p), u)
    h = 1e-5
    fd = (constraints.area(phi + h * u, g, p) - constraints.area(phi - h * u, g, p)) / (2 * h)
    assert abs(fd - analytic) <= 1e-7 * (1.0 + abs(analytic))


def test_woodbury_inverts_lhs():
    g = SpectralGrid(16)
    p = Params(eps=0.3, M1=1e3, M2=1e3, tau=1e-3, sigma=0.0)
    phi = _smooth_field(g)
    e_A = constraints.dA_dphi(phi, g, p)
    ones = np.ones_like(phi)
    gbar = symbols.gbar_prime(phi, p)
    Lhat = symbols.L_hat(g, p, gbar)

    b = np.cos(g.X) * np.sin(g.Y) + 0.5
    x = woodbury.solve(g, Lhat, b, [e_A, ones], [p.M2, p.M1])
    lhs = stepper.apply_LHS(x, e_A, ones, Lhat, g, p)
    assert g.l2norm(lhs - b) <= 1e-9 * (1.0 + g.l2norm(b))
