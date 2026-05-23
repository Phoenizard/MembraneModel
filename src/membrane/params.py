"""Model + scheme parameters. Stage A uses the phi-side subset (c=0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    # interface width (xi = eps)
    eps: float = 0.1964
    # bending stiffness k(eta) = k + c*tanh(eta/xi); Stage A: c=0 -> k const
    k: float = 1.0
    c: float = 0.0
    # line tension (unused in Stage A, no eta)
    delta: float = 10.0
    # constraints
    v_d: float = -216.52
    a_0: float = 29.46
    a_d: float = 0.23
    # penalty constants
    M1: float = 1.0e4
    M2: float = 1.0e4
    M3: float = 1.0e4
    M4: float = 1.0e4
    M5: float = 1.0e4
    # IEQ regularisation of r = sqrt(2W + C0)
    C0: float = 1.0e-2
    # time step + Eyre stabilisation
    tau: float = 1.0e-3
    sigma: float = 0.0

    @property
    def xi(self) -> float:
        return self.eps

    @property
    def a2(self) -> float:
        """a(eta)^2 = k(eta)/eps; constant for c=0."""
        return self.k / self.eps
