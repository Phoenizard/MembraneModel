"""exp1 Stage A driver — phi-only single-component IEQ run.

Spec: doc/exp/exp1/README.md (Fig. 2 convergence test, c=0).
Stage A validates the numerical machinery (spectral solve, Woodbury, penalty
continuation), NOT the 124.49 energy (that includes line tension -> Stage B).

Usage:
    python run.py            # 64^3, eps=0.1964
    python run.py --quick    # 32^3 smoke test
    python run.py --N 48 --eps 0.1964
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from membrane import SpectralGrid, Params
from membrane import initial, diagnostics
from membrane.solver.continuation import continuation

OUT = os.path.join(os.path.dirname(__file__), "out")


def run_case(N, eps, *, tau, sigma, C0, M_start, M_max, rho, inner_tol, outer_tol, max_inner):
    p = Params(eps=eps, k=1.0, c=0.0, v_d=-216.52, a_0=29.46,
               C0=C0, tau=tau, sigma=sigma)
    grid = SpectralGrid(N)
    phi = initial.sphere(grid, p)
    R = initial.sphere_radius_from_vd(grid, p)
    print(f"\n=== N={N}^3  eps={eps}  R0={R:.4f} ===")
    print(f"  init: {diagnostics.line(phi, grid, p)}")

    phi, history = continuation(
        phi, grid, p, M_start=M_start, M_max=M_max, rho=rho,
        inner_tol=inner_tol, outer_tol=outer_tol, max_inner=max_inner,
        log_every=0,
    )
    print(f"  final: {diagnostics.line(phi, grid, p)}")
    d = diagnostics.report(phi, grid, p)
    print(f"  Lambda1={d['Lambda1']:+.4f}  Lambda2={d['Lambda2']:+.4f}  "
          f"phi_absmax={d['phi_absmax']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"phi_N{N}_eps{eps}.npy"), phi)
    return phi, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.1964)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--C0", type=float, default=1e-2)
    args = ap.parse_args()

    if args.quick:
        run_case(32, args.eps, tau=args.tau, sigma=args.sigma, C0=args.C0,
                 M_start=1e4, M_max=3.2e5, rho=4.0,
                 inner_tol=1e-6, outer_tol=2e-3, max_inner=4000)
    else:
        run_case(args.N, args.eps, tau=args.tau, sigma=args.sigma, C0=args.C0,
                 M_start=1e4, M_max=3.2e5, rho=4.0,
                 inner_tol=1e-7, outer_tol=1e-3, max_inner=20000)


if __name__ == "__main__":
    main()
