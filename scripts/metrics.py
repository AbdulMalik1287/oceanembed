#!/usr/bin/env python
"""Skill metrics, per depth level, in degrees C.

The PS asks for correlation, RMSE and bias. Correlation on absolute temperature
is close to 1.0 for anything that knows the seasonal cycle, so it flatters every
model and separates none of them. Anomaly correlation (ACC) — computed after the
climatology is removed from both prediction and truth — is the number that
actually says whether the model has skill beyond the calendar. Both are
reported; ACC is the one to argue from.
"""
from __future__ import annotations

import numpy as np


def _flat(pred, true, clim, mask):
    """(depth, samples) views over wet cells with finite truth."""
    n, nd = pred.shape[0], pred.shape[1]
    m = np.broadcast_to(mask, (n, nd, *mask.shape))
    ok = m & np.isfinite(true) & np.isfinite(pred)
    return [(pred[:, d][ok[:, d]], true[:, d][ok[:, d]], clim[:, d][ok[:, d]])
            for d in range(nd)]


def skill(pred, true, clim, mask, depths) -> list[dict]:
    """Per-depth RMSE / bias / correlation / ACC, all in degrees C."""
    rows = []
    for d, (p, t, c) in enumerate(_flat(pred, true, clim, mask)):
        if p.size == 0:
            rows.append(dict(depth=float(depths[d]), n=0))
            continue
        err = p - t
        pa, ta = p - c, t - c
        denom = np.sqrt((pa**2).sum() * (ta**2).sum())
        rows.append(dict(
            depth=float(depths[d]),
            n=int(p.size),
            rmse=float(np.sqrt((err**2).mean())),
            bias=float(err.mean()),
            corr=float(np.corrcoef(p, t)[0, 1]),
            acc=float((pa * ta).sum() / denom) if denom > 0 else np.nan,
        ))
    return rows


def table(rows, title="") -> str:
    out = [f"\n{title}" if title else ""]
    out.append(f"{'depth':>7} {'RMSE':>8} {'bias':>8} {'corr':>7} {'ACC':>7} {'n':>10}")
    out.append("-" * 50)
    for r in rows:
        if not r.get("n"):
            out.append(f"{r['depth']:7.0f} {'—':>8} {'—':>8} {'—':>7} {'—':>7} {0:>10}")
            continue
        out.append(f"{r['depth']:7.0f} {r['rmse']:8.3f} {r['bias']:8.3f} "
                   f"{r['corr']:7.3f} {r['acc']:7.3f} {r['n']:10d}")
    valid = [r for r in rows if r.get("n")]
    if valid:
        out.append("-" * 50)
        out.append(f"{'mean':>7} {np.mean([r['rmse'] for r in valid]):8.3f} "
                   f"{np.mean([r['bias'] for r in valid]):8.3f} "
                   f"{np.mean([r['corr'] for r in valid]):7.3f} "
                   f"{np.mean([r['acc'] for r in valid]):7.3f}")
    return "\n".join(out)
