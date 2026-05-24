"""Run monitoring: per-step metrics -> wandb + local JSONL, plus
oscillation/stall detection for early stopping.

Decoupled from the solver: the solver only calls log()/should_stop() on an
optional monitor object; monitor=None means zero overhead. wandb is optional
(degrades to JSONL-only if not installed or disabled).

log() expects at least "E_M" and "d" (per-step field change) in the metrics
dict; those two drive the convergence diagnostics. Any other keys are recorded
and forwarded as-is.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np


class RunMonitor:
    def __init__(self, name, params, settings, *, out_dir,
                 use_wandb=True, wandb_mode="offline",
                 wandb_project="Membrane", wandb_entity="pheonizard-university-of-nottingham",
                 window=80, osc_rel_tol=5e-3, stall_flat_tol=0.1,
                 stall_floor=1e-3):
        self.name = name
        self.window = window
        self.osc_rel_tol = osc_rel_tol          # E_M relative-std threshold (oscillation)
        self.stall_flat_tol = stall_flat_tol    # |d| spread/mean threshold (flat => stall)
        self.stall_floor = stall_floor          # below this |d| (abs) is ~converging

        self.history = []          # full record, one dict per logged step
        self._cur_level = None     # current penalty level; window resets when it changes
        self._level_E = []         # E_M values within the current level
        self._level_d = []         # |d| values within the current level
        self._gstep = 0            # global monotonic step (wandb x-axis)

        os.makedirs(out_dir, exist_ok=True)
        self._log_path = os.path.join(out_dir, f"{name}.log")
        self._fh = open(self._log_path, "w")

        self.wandb = None
        if use_wandb:
            try:
                import wandb
                cfg = {**(asdict(params) if is_dataclass(params) else dict(params)),
                       **dict(settings)}
                self.run = wandb.init(entity=wandb_entity, project=wandb_project,
                                      name=name, config=cfg, mode=wandb_mode)
                self.wandb = wandb
            except Exception as e:
                print(f"[monitor] wandb disabled ({e}); local JSONL only at {self._log_path}")

    def log(self, step, metrics, *, m_level=None):
        # a new penalty level makes E_M jump: reset the rolling window
        if m_level != self._cur_level:
            self._cur_level = m_level
            self._level_E.clear()
            self._level_d.clear()

        self._gstep += 1
        rec = {"gstep": self._gstep, "step": int(step), "m_level": m_level,
               **{k: float(v) for k, v in metrics.items()}}
        self.history.append(rec)
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

        if "E_M" in metrics:
            self._level_E.append(float(metrics["E_M"]))
        if "d" in metrics:
            self._level_d.append(float(metrics["d"]))

        if self.wandb is not None:
            payload = dict(metrics)
            if m_level is not None:
                payload["M"] = m_level
            self.wandb.log(payload, step=self._gstep)

    def should_stop(self):
        """Detect non-convergence within the current penalty level.

        Returns (stop, reason). Logic: only fire if |d| is NOT making progress
        (well above the convergence floor and not shrinking across the window);
        then classify as oscillation (E_M bounces) or stall (|d| flat).
        """
        n = min(len(self._level_E), len(self._level_d))
        if n < self.window:
            return False, None

        Ew = np.asarray(self._level_E[-self.window:])
        dw = np.asarray(self._level_d[-self.window:])

        half = self.window // 2
        d_early, d_late = dw[:half].mean(), dw[half:].mean()
        shrinking = d_late < 0.9 * d_early          # |d| dropped >10% over the window
        above_floor = dw.mean() > self.stall_floor
        if shrinking or not above_floor:
            return False, None                       # genuine progress or near convergence

        rel_std = float(np.std(Ew) / (abs(np.mean(Ew)) + 1e-12))
        if rel_std > self.osc_rel_tol:
            return True, (f"oscillation: E_M rel-std={rel_std:.2e}, "
                          f"|d| stuck ~{dw.mean():.2e} over {self.window} steps")

        dspread = float((dw.max() - dw.min()) / (dw.mean() + 1e-12))
        if dspread < self.stall_flat_tol:
            return True, (f"stall: |d| flat ~{dw.mean():.2e}, "
                          f"E_M not converging over {self.window} steps")

        return False, None

    def finish(self):
        try:
            self._fh.close()
        except Exception:
            pass
        if self.wandb is not None:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False
