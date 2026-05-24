"""Result I/O: save fields + a same-named .txt parameter sheet.

Naming convention (no decimal points in the basename):
    {name}_phi.npy   {name}_eta.npy   {name}.txt
where name = exp{n}-{run}-{control}, e.g. "exp1-1-N32".

The .txt records the full parameter set so the metadata-free .npy arrays stay
interpretable. eta is optional (single-component runs omit it).
"""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict

import numpy as np


def _fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def save_run(out_dir, name, phi, params, grid, *, eta=None, sections=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    pphi = os.path.join(out_dir, f"{name}_phi.npy")
    np.save(pphi, phi)
    paths["phi"] = pphi
    if eta is not None:
        peta = os.path.join(out_dir, f"{name}_eta.npy")
        np.save(peta, eta)
        paths["eta"] = peta

    lines = [
        f"# {name}",
        f"saved   : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"grid    : N={grid.n}^3, h={grid.h:.6f}, domain=[-pi,pi]^3",
        f"fields  : phi" + (", eta" if eta is not None else ""),
        "",
        "[params]",
    ]
    lines += [f"{k:8s} = {_fmt(v)}" for k, v in asdict(params).items()]
    for sec, d in (sections or {}).items():
        lines += ["", f"[{sec}]"] + [f"{k:8s} = {_fmt(v)}" for k, v in d.items()]

    ptxt = os.path.join(out_dir, f"{name}.txt")
    with open(ptxt, "w") as f:
        f.write("\n".join(lines) + "\n")
    paths["txt"] = ptxt
    return paths
