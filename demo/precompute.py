#!/usr/bin/env python
"""Bake a compact, offline data bundle for the demo frontend.

    python -m demo.precompute

Writes demo/web/data.js, a single file the page loads with a <script> tag, so
the whole demo runs from the filesystem with no server, no network and no GPU
box. That matters more than elegance: a demo that needs WiFi is a demo that
fails in the room.

Fields are quantised to uint8 with a per-field min/max, which is invisible at
display resolution and keeps the bundle around 6 MB instead of 90.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.config import PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON

PRED = PROC.parent / "pred"
RUNS = PROC.parent / "runs"
ARGO = PROC.parent / "argo"
OUT = Path(__file__).parent / "web" / "data.js"

YEAR = 2021
COARSE = 2      # subsampling for the profile-only stacks (GLORYS, climatology)
# One per month: plus 29 August, the day Argo had zero profiles in the Bay of
# Bengal: which is the story the coverage figure tells.
DATES = [f"{YEAR}-{m:02d}-15" for m in range(1, 13)] + [f"{YEAR}-08-29"]


def q(arr, step: int = 1):
    """Quantise a 2-D field to uint8 + base64, carrying its own scale.

    `step` subsamples. Fields drawn as maps stay at full resolution; the GLORYS
    and climatology stacks are only ever read one cell at a time for the profile
    panel, so they go at half resolution and save two thirds of the bundle.
    """
    a = np.asarray(arr, dtype="float64")[::step, ::step]
    good = np.isfinite(a)
    if not good.any():
        return None
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    span = hi - lo if hi > lo else 1.0
    # 0 is reserved for "no data", so real values map to 1..255.
    u = np.where(good, 1 + np.round((a - lo) / span * 254), 0).astype("uint8")
    return {"lo": round(lo, 4), "hi": round(hi, 4),
            "d": base64.b64encode(u.tobytes()).decode("ascii")}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    pred = xr.open_dataarray(PRED / f"pred_{YEAR}.nc")
    truth = xr.open_dataarray(PROC / f"target_{YEAR}.nc").sel(time=pred.time)
    diag = xr.open_dataset(PRED / f"diag_{YEAR}.nc")

    land = ~np.isfinite(truth.isel(time=0, depth=0).values)
    dates_avail = pred["time"].dt.strftime("%Y-%m-%d").values

    frames = {}
    for day in DATES:
        hit = np.flatnonzero(dates_avail == day)
        if not len(hit):
            print(f"  skip {day} (not in prediction)")
            continue
        i = int(hit[0])
        p = pred.isel(time=i).values                      # (depth, lat, lon)
        t = truth.isel(time=i).values
        frames[day] = {
            "t": [q(p[k]) for k in range(len(STD_DEPTHS))],
            "truth": [q(t[k], COARSE) for k in range(len(STD_DEPTHS))],
            "tchp": q(np.where(land, np.nan, diag["tchp_pred"].values[i])),
            "tchp_true": q(np.where(land, np.nan, diag["tchp_true"].values[i])),
            "d26": q(np.where(land, np.nan, diag["d26_pred"].values[i])),
            "mld": q(np.where(land, np.nan, diag["mld_pred"].values[i])),
        }
        print(f"  baked {day}")

    # Climatology for the profile panel: the same day-of-year mean the model
    # predicts anomalies against, so the three curves are directly comparable.
    from scripts.data import day_of_year_climatology, load_years
    clim = day_of_year_climatology(load_years([2014, 2015, 2016, 2017, 2018, 2019])[1])
    clim_frames = {}
    for day in frames:
        doy = int(np.datetime64(day, "D").astype("datetime64[D]").astype(object).timetuple().tm_yday)
        c = clim.sel(dayofyear=doy).transpose("depth", "latitude", "longitude").values
        clim_frames[day] = [q(c[k], COARSE) for k in range(len(STD_DEPTHS))]

    scores = {m: json.load(open(RUNS / f"{m}.json")) for m in ("climatology", "linear", "unet")}
    argo = json.load(open(ARGO / "argo_scores.json")) if (ARGO / "argo_scores.json").exists() else None

    bundle = {
        "meta": {
            "lat": [float(TARGET_LAT[0]), float(TARGET_LAT[-1])],
            "lon": [float(TARGET_LON[0]), float(TARGET_LON[-1])],
            "ny": len(TARGET_LAT), "nx": len(TARGET_LON),
            "coarse": COARSE,
            "nyc": len(TARGET_LAT[::COARSE]), "nxc": len(TARGET_LON[::COARSE]),
            "depths": [float(d) for d in STD_DEPTHS],
            "dates": sorted(frames),
            "train_years": "2014-2019", "test_years": "2020-2021",
        },
        "land": base64.b64encode(land.astype("uint8").tobytes()).decode("ascii"),
        "frames": frames,
        "clim": clim_frames,
        "scores": {m: [r for r in scores[m]["rows"] if r.get("n")] for m in scores},
        "argo": argo,
    }

    payload = json.dumps(bundle, separators=(",", ":"))
    OUT.write_text("window.OCEAN_DATA = " + payload + ";", encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024**2:.1f} MB, {len(frames)} dates)")


if __name__ == "__main__":
    main()
