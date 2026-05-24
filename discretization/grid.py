"""Fourier spectral grid on the periodic box Omega = [-pi, pi]^3.

All differential operators are diagonal in Fourier space. Fields are real;
FFTs are kept full-complex here for clear ik-indexing (swap to rfft / pyfftw
later without touching callers).
"""

from __future__ import annotations

import numpy as np


class SpectralGrid:
    def __init__(self, n: int, length: float = 2.0 * np.pi):
        self.n = n
        self.length = length
        self.h = length / n
        self.dV = self.h**3

        # physical coordinates, x_j = -L/2 + j*h
        x1 = -length / 2.0 + self.h * np.arange(n)
        self.X, self.Y, self.Z = np.meshgrid(x1, x1, x1, indexing="ij")

        # integer wavenumbers for L = 2*pi:  [0,1,...,n/2-1,-n/2,...,-1]
        k1 = np.fft.fftfreq(n, d=length / n) * (2.0 * np.pi)
        self.KX, self.KY, self.KZ = np.meshgrid(k1, k1, k1, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2

    # --- FFT wrappers ---
    def fft(self, f: np.ndarray) -> np.ndarray:
        return np.fft.fftn(f)

    def ifft(self, fhat: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(fhat).real

    # --- differential operators ---
    def grad(self, f: np.ndarray):
        fhat = self.fft(f)
        return (
            self.ifft(1j * self.KX * fhat),
            self.ifft(1j * self.KY * fhat),
            self.ifft(1j * self.KZ * fhat),
        )

    def lap(self, f: np.ndarray) -> np.ndarray:
        return self.ifft(-self.K2 * self.fft(f))

    def div(self, fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
        return self.ifft(
            1j * (self.KX * self.fft(fx) + self.KY * self.fft(fy) + self.KZ * self.fft(fz))
        )

    def grad_dot_grad(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        fx, fy, fz = self.grad(f)
        gx, gy, gz = self.grad(g)
        return fx * gx + fy * gy + fz * gz

    # --- integrals / inner products (L^2 with cell volume) ---
    def integral(self, f: np.ndarray) -> float:
        return float(np.sum(f) * self.dV)

    def inner(self, f: np.ndarray, g: np.ndarray) -> float:
        return float(np.sum(f * g) * self.dV)

    def l2norm(self, f: np.ndarray) -> float:
        return float(np.sqrt(self.inner(f, f)))

    # --- spectral application / inversion of a diagonal symbol ---
    def apply_symbol(self, f: np.ndarray, symbol: np.ndarray) -> np.ndarray:
        return self.ifft(symbol * self.fft(f))

    def solve_symbol(self, rhs: np.ndarray, symbol: np.ndarray) -> np.ndarray:
        return self.ifft(self.fft(rhs) / symbol)
