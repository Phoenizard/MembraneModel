"""exp1 Stage B driver — full two-component IEQ run (Fig. 2 target).

Target (paper p.355): E_M ~ 124.49 (eps=0.1964) = E_bend ~48 + E_line ~76.
Spec: doc/exp/exp1/README.md.

Usage:
    python run_stageB.py --quick          # 32^3 smoke
    python run_stageB.py --N 64 --eps 0.1964
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from membrane import SpectralGrid, Params
from membrane import initial
from membrane.solver import coupled

OUT = os.path.join(os.path.dirname(__file__), "out")


def run_case(N, eps, *, k, c, delta, tau, sigma, C0, M_start, M_max, rho,
             inner_tol, outer_tol, max_inner, M_reg, log_every=0):
    p = Params(eps=eps, k=k, c=c, delta=delta, v_d=-216.52, a_0=29.46, a_d=0.23,
               C0=C0, tau=tau, sigma=sigma, M4=M_reg, M5=M_reg)
    grid = SpectralGrid(N)
    phi = initial.sphere(grid, p)
    eta = initial.eta_two_phase(grid, p)
    print(f"\n=== Stage B  N={N}^3  eps={eps}  k={k} c={c} delta={delta} ===")
    print(f"  init: {coupled.line(phi, eta, grid, p)}")

    phi, eta, history = coupled.continuation(
        phi, eta, grid, p, M_start=M_start, M_max=M_max, rho=rho,
        inner_tol=inner_tol, outer_tol=outer_tol, max_inner=max_inner, log_every=log_every,
    )
    print(f"  final: {coupled.line(phi, eta, grid, p)}")
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"phiB_N{N}_eps{eps}.npy"), phi)
    np.save(os.path.join(OUT, f"etaB_N{N}_eps{eps}.npy"), eta)
    return phi, eta, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.1964)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=2.0)  # Eyre stabilisation (NaN at sigma=0,tau>=2e-3)
    ap.add_argument("--C0", type=float, default=1e-2)
    ap.add_argument("--Mreg", type=float, default=1e4)
    ap.add_argument("--log", type=int, default=0)
    args = ap.parse_args()

    common = dict(k=args.k, c=args.c, delta=args.delta, tau=args.tau, sigma=args.sigma,
                  C0=args.C0, M_reg=args.Mreg, rho=4.0, log_every=args.log)
    if args.quick:
        run_case(32, args.eps, M_start=1e4, M_max=3.2e5,
                 inner_tol=1e-6, outer_tol=3e-3, max_inner=3000, **common)
    else:
        run_case(args.N, args.eps, M_start=1e4, M_max=3.2e5,
                 inner_tol=1e-7, outer_tol=1e-3, max_inner=20000, **common)


if __name__ == "__main__":
    main()
