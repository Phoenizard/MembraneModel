"""Visualisation interface for equilibrium fields (static matplotlib PNG).

Two paper-style views (cf. Wang & Du Fig. 2):
  - surface_figure: phi=0 vesicle iso-surface coloured by eta (red/blue phases)
  - cross_section_figure: x-z / x-y heatmaps of phi and eta, with the phi=0
    membrane contour overlaid on the eta panels
  - summary_figure: both, side by side, saved to one PNG

Backend is forced to Agg so it renders headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from skimage import measure

from discretization.grid import SpectralGrid


# --- slices ---
def _center_index(grid: SpectralGrid) -> int:
    return grid.n // 2


def slice_xz(field: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    """y = 0 plane; returns array indexed [x, z] for imshow(.T)."""
    return field[:, _center_index(grid), :]


def slice_xy(field: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    """z = 0 plane; returns array indexed [x, y]."""
    return field[:, :, _center_index(grid)]


# --- iso-surface (marching cubes on phi=0, coloured by eta) ---
def _isosurface(phi, eta, grid: SpectralGrid, level=0.0):
    verts, faces, _, _ = measure.marching_cubes(phi, level=level, spacing=(grid.h,) * 3)
    # verts are in physical length from index 0; sample eta at those positions
    idx = (verts / grid.h).T  # back to index coordinates
    eta_v = map_coordinates(eta, idx, order=1, mode="grid-wrap")
    # centre coordinates on the box origin (-L/2)
    verts = verts - grid.length / 2.0
    return verts, faces, eta_v


def surface_figure(phi, eta, grid: SpectralGrid, *, path=None, elev=22, azim=-60,
                   cmap="bwr", ax=None):
    verts, faces, eta_v = _isosurface(phi, eta, grid)
    face_eta = eta_v[faces].mean(axis=1)
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    colors = plt.get_cmap(cmap)(norm(face_eta))

    own = ax is None
    if own:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none", alpha=1.0)
    ax.add_collection3d(mesh)
    lim = grid.length / 2.0
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("phi=0 surface, coloured by eta (red:+1 / blue:-1)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if own and path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return ax


def cross_section_figure(phi, eta, grid: SpectralGrid, *, path=None, axes=None):
    lim = grid.length / 2.0
    ext = [-lim, lim, -lim, lim]
    panels = [
        ("phi  (x-z, y=0)", slice_xz(phi, grid), "coolwarm", (-1, 1)),
        ("eta  (x-z, y=0)", slice_xz(eta, grid), "bwr", (-1, 1)),
        ("phi  (x-y, z=0)", slice_xy(phi, grid), "coolwarm", (-1, 1)),
        ("eta  (x-y, z=0)", slice_xy(eta, grid), "bwr", (-1, 1)),
    ]
    phi_xz = slice_xz(phi, grid)
    phi_xy = slice_xy(phi, grid)

    own = axes is None
    if own:
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = np.asarray(axes).ravel()
    for k, (title, data, cmap, (vmin, vmax)) in enumerate(panels):
        ax = axes[k]
        im = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
        # overlay phi=0 membrane contour on the eta panels
        if "eta" in title:
            base = phi_xz if "x-z" in title else phi_xy
            ax.contour(base.T, levels=[0.0], colors="k", linewidths=1.0, extent=ext)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if own and path:
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return axes


def summary_figure(phi, eta, grid: SpectralGrid, *, path, title=None,
                   elev=22, azim=-60):
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 4, figure=fig)
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    surface_figure(phi, eta, grid, ax=ax3d, elev=elev, azim=azim)
    ax_pp = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
             fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]
    cross_section_figure(phi, eta, grid, axes=ax_pp)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
