"""Spectral operators must be exact (to machine precision) on Fourier modes."""

import numpy as np

from membrane import SpectralGrid


def test_laplacian_eigenmode():
    g = SpectralGrid(16)
    f = np.sin(2 * g.X) * np.cos(3 * g.Y) * np.sin(g.Z)
    # lap of sin(2x)cos(3y)sin(z) = -(4+9+1) f
    assert np.allclose(g.lap(f), -14.0 * f, atol=1e-11)


def test_gradient():
    g = SpectralGrid(16)
    f = np.sin(2 * g.X) * np.cos(3 * g.Y)
    fx, fy, fz = g.grad(f)
    assert np.allclose(fx, 2 * np.cos(2 * g.X) * np.cos(3 * g.Y), atol=1e-11)
    assert np.allclose(fy, -3 * np.sin(2 * g.X) * np.sin(3 * g.Y), atol=1e-11)
    assert np.allclose(fz, 0.0, atol=1e-11)


def test_div_grad_is_lap():
    g = SpectralGrid(16)
    f = np.sin(g.X) * np.cos(2 * g.Y) * np.cos(g.Z)
    fx, fy, fz = g.grad(f)
    assert np.allclose(g.div(fx, fy, fz), g.lap(f), atol=1e-11)


def test_symbol_inverse_roundtrip():
    g = SpectralGrid(16)
    f = np.sin(g.X) * np.cos(g.Y) + 0.3 * np.cos(2 * g.Z)
    sym = 1.0 + g.K2**2
    assert np.allclose(g.solve_symbol(g.apply_symbol(f, sym), sym), f, atol=1e-11)
