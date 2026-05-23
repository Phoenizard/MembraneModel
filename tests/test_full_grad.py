"""Stage B correctness gate: full E_M(phi,eta) variational derivatives must
match finite differences. Term-isolation configs pinpoint any wrong term.
"""

import numpy as np
import pytest

from membrane import SpectralGrid, Params
from membrane.model import full


def _fields(g):
    phi = 0.6 * np.sin(g.X) * np.cos(g.Y) * np.sin(g.Z) + 0.2 * np.cos(2 * g.X)
    eta = 0.5 * np.cos(g.X) * np.sin(g.Y) + 0.3 * np.sin(g.Z) * np.cos(g.Y)
    return phi, eta


def _fd_phi(phi, eta, g, p, u, h=1e-5):
    return (full.E_total(phi + h * u, eta, g, p) - full.E_total(phi - h * u, eta, g, p)) / (2 * h)


def _fd_eta(phi, eta, g, p, w, h=1e-5):
    return (full.E_total(phi, eta + h * w, g, p) - full.E_total(phi, eta - h * w, g, p)) / (2 * h)


# Term-isolation parameter sets: zero out everything except the term under test.
ZERO = dict(k=0.0, c=0.0, delta=0.0, M1=0.0, M2=0.0, M3=0.0, M4=0.0, M5=0.0)

CASES = {
    "bending_keta": {**ZERO, "k": 1.3, "c": 0.7},     # E with k(eta), c!=0
    "line": {**ZERO, "delta": 2.0},                    # L
    "volume": {**ZERO, "M1": 50.0},                    # (M1/2)(V-v_d)^2
    "area": {**ZERO, "M2": 50.0},                      # (M2/2)(A-a0)^2
    "areadiff": {**ZERO, "M3": 50.0},                  # (M3/2)(D-a_d)^2
    "ortho_N": {**ZERO, "M4": 50.0},                   # (M4/2)N^2
    "profile_P": {**ZERO, "M5": 50.0},                 # (M5/2)P^2
    "all": dict(k=1.3, c=0.7, delta=2.0, M1=30.0, M2=30.0, M3=30.0, M4=30.0, M5=30.0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_dEM_dphi(name):
    g = SpectralGrid(16)
    p = Params(eps=0.3, v_d=-50.0, a_0=20.0, a_d=1.5, **CASES[name])
    phi, eta = _fields(g)
    u = np.sin(g.X + 0.7) * np.cos(g.Y - 0.3) * np.cos(g.Z + 0.2)
    analytic = g.inner(full.dEM_dphi(phi, eta, g, p), u)
    fd = _fd_phi(phi, eta, g, p, u)
    assert abs(fd - analytic) <= 1e-5 * (1.0 + abs(analytic)) + 1e-9, name


@pytest.mark.parametrize("name", list(CASES))
def test_dEM_deta(name):
    g = SpectralGrid(16)
    p = Params(eps=0.3, v_d=-50.0, a_0=20.0, a_d=1.5, **CASES[name])
    phi, eta = _fields(g)
    w = np.cos(g.X - 0.2) * np.sin(g.Y + 0.5) * np.cos(g.Z)
    analytic = g.inner(full.dEM_deta(phi, eta, g, p), w)
    fd = _fd_eta(phi, eta, g, p, w)
    assert abs(fd - analytic) <= 1e-5 * (1.0 + abs(analytic)) + 1e-9, name
