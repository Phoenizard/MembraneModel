"""IEQ vesicle driver — full two-component run (exp1, Fig. 2 target).

Target (paper p.355): E_M ~ 124.49 (eps=0.1964) = E_bend ~48 + E_line ~76.
Spec: doc/exp/exp1/README.md.

Runs the continuation solver to a relaxed/converged state and saves phi, eta
(+ a same-named parameter sheet) to experiments/exp1/out/ (--name sets the
basename). Visualise the saved fields with visualize.py.

Usage:
    python run_IEQ.py --quick                              # 32^3 smoke
    python run_IEQ.py --N 32 --name exp1-1-N32 --log 25
    python run_IEQ.py --N 64 --max_inner 8000 --wandb
"""

from __future__ import annotations

import argparse
import os

from discretization.grid import SpectralGrid
from params import Params
import initial
import results
from model import energy
from solver import ieq
from monitor import RunMonitor

OUT = os.path.join(os.path.dirname(__file__), "experiments", "exp1", "out")


def run_case(N, eps, *, k, c, delta, tau, sigma, C0, M_start, M_max, rho,
             e_rtol, patience, outer_tol, max_inner, M_reg, name, log_every=0,
             use_wandb=False, wandb_mode="offline"):
    p = Params(eps=eps, k=k, c=c, delta=delta, v_d=-216.52, a_0=29.46, a_d=0.23,
               C0=C0, tau=tau, sigma=sigma, M4=M_reg, M5=M_reg)
    grid = SpectralGrid(N)
    phi = initial.sphere(grid, p)
    eta = initial.eta_two_phase(grid, p)
    print(f"\n=== IEQ vesicle  N={N}^3  eps={eps}  k={k} c={c} delta={delta}  name={name} ===")
    print(f"  init: {ieq.line(phi, eta, grid, p)}")

    settings = {"N": N, "M_start": M_start, "M_max": M_max, "rho": rho,
                "e_rtol": e_rtol, "patience": patience, "outer_tol": outer_tol,
                "max_inner": max_inner, "log_every": log_every}
    with RunMonitor(name, p, settings, out_dir=OUT, use_wandb=use_wandb,
                    wandb_mode=wandb_mode) as mon:
        phi, eta, history = ieq.continuation(
            phi, eta, grid, p, M_start=M_start, M_max=M_max, rho=rho,
            e_rtol=e_rtol, patience=patience, outer_tol=outer_tol,
            max_inner=max_inner, log_every=log_every, monitor=mon,
        )
    print(f"  final: {ieq.line(phi, eta, grid, p)}")

    # output interface (compute + save bundled): fields + same-named param sheet
    bd = energy.energy_breakdown(phi, eta, grid, p)
    final = {**{k: bd[k] for k in ("E_M", "E_bend", "E_line", "A", "V", "D", "N", "P")},
             "V-v_d": bd["V"] - p.v_d, "A-a_0": bd["A"] - p.a_0, "D-a_d": bd["D"] - p.a_d,
             "last_M_steps": history[-1]["steps"], "n_M_levels": len(history)}
    cont = {"M_start": M_start, "M_max": M_max, "rho": rho, "e_rtol": e_rtol,
            "patience": patience, "outer_tol": outer_tol, "max_inner": max_inner}
    paths = results.save_run(OUT, name, phi, p, grid, eta=eta,
                        sections={"continuation": cont, "final": final})
    print(f"  saved: {', '.join(os.path.basename(v) for v in paths.values())}")
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
    ap.add_argument("--name", type=str, default=None,
                    help="output basename, e.g. exp1-1-N32 (default: exp1-1-N{N})")
    ap.add_argument("--max_inner", type=int, default=2500)
    ap.add_argument("--e_rtol", type=float, default=1e-3,
                    help="energy-plateau tol: converged when no new E_M low beats best by >e_rtol")
    ap.add_argument("--patience", type=int, default=8,
                    help="consecutive log-samples with no new E_M low before declaring converged")
    ap.add_argument("--outer_tol", type=float, default=1e-3)
    ap.add_argument("--log", type=int, default=25,
                    help="print + monitor-log every N steps (0 = off, no monitor data)")
    ap.add_argument("--wandb", action="store_true",
                    help="enable wandb logging (default: local JSONL only)")
    ap.add_argument("--wandb-mode", type=str, default="offline",
                    choices=["offline", "online"], help="wandb mode when --wandb is set")
    args = ap.parse_args()

    N = 32 if args.quick else args.N
    name = args.name or f"exp1-1-N{N}"
    run_case(N, args.eps, k=args.k, c=args.c, delta=args.delta, tau=args.tau,
             sigma=args.sigma, C0=args.C0, M_reg=args.Mreg, rho=4.0,
             log_every=args.log, M_start=1e4, M_max=3.2e5, e_rtol=args.e_rtol,
             patience=args.patience, outer_tol=args.outer_tol, max_inner=args.max_inner,
             name=name, use_wandb=args.wandb, wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
