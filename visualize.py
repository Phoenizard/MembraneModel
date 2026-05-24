"""Render equilibrium fields to PNG for visual acceptance.

    python visualize.py --name exp1-1-N32            # loads out/exp1-1-N32_phi.npy + _eta.npy

Produces in out/:
    <name>_summary.png    3D phi=0 surface (coloured by eta) + cross-sections
    <name>_surface.png    surface only
    <name>_slices.png     cross-sections only
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from discretization.grid import SpectralGrid
import viz

OUT = os.path.join(os.path.dirname(__file__), "experiments", "exp1", "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True,
                    help="run basename, e.g. exp1-1-N32 (loads <name>_phi.npy [+ _eta.npy])")
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-60)
    args = ap.parse_args()

    base = os.path.join(OUT, args.name)
    phi = np.load(base + "_phi.npy")
    grid = SpectralGrid(phi.shape[0])

    eta = np.load(base + "_eta.npy")
    title = f"{args.name}  (phi + eta)"
    viz.summary_figure(phi, eta, grid, path=base + "_summary.png", title=title,
                       elev=args.elev, azim=args.azim)
    viz.surface_figure(phi, eta, grid, path=base + "_surface.png",
                       elev=args.elev, azim=args.azim)
    viz.cross_section_figure(phi, eta, grid, path=base + "_slices.png")
    for s in ("_summary.png", "_surface.png", "_slices.png"):
        print("wrote", base + s)


if __name__ == "__main__":
    main()
