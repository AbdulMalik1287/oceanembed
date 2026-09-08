#!/usr/bin/env python
"""Run the trained U-Net over a year and write the standardized output product.

    python -m scripts.predict --train 2019 2020 --test 2021

Produces, in ~/ocean/pred/:
  pred_YYYY.nc   reconstructed temperature, degC, daily, 0.25 deg, 15 depths
                 (problem statement deliverable #4)
  diag_YYYY.nc   D26, TCHP, MLD and OHC for both the reconstruction and GLORYS,
                 so the operational diagnostics can be scored against truth
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import xarray as xr

from . import data as D
from .config import PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON
from .diagnostics import d26, mld, ohc, tchp
from .model import UNet

OUT = PROC.parent / "pred"
CKPT = PROC.parent / "runs" / "unet.pt"


def _cube(arr, times, name, units, long_name):
    return xr.DataArray(
        arr.astype("float32"),
        coords={"time": times, "depth": STD_DEPTHS,
                "latitude": TARGET_LAT, "longitude": TARGET_LON},
        dims=("time", "depth", "latitude", "longitude"),
        name=name,
        attrs={"units": units, "long_name": long_name},
    )


def _diagnostics(t4d, times, tag):
    """t4d is (time, depth, lat, lon); the diagnostics want depth last."""
    prof = np.moveaxis(t4d, 1, -1)
    fields = {
        f"d26_{tag}": (d26(prof, STD_DEPTHS), "m", "depth of the 26 degC isotherm"),
        f"tchp_{tag}": (tchp(prof, STD_DEPTHS), "kJ cm-2", "tropical cyclone heat potential"),
        f"mld_{tag}": (mld(prof, STD_DEPTHS), "m", "mixed layer depth (0.2 degC criterion)"),
        f"ohc_{tag}": (ohc(prof, STD_DEPTHS), "kJ cm-2", "ocean heat content 0-700 m"),
    }
    return {
        k: xr.DataArray(
            v.astype("float32"),
            coords={"time": times, "latitude": TARGET_LAT, "longitude": TARGET_LON},
            dims=("time", "latitude", "longitude"),
            attrs={"units": u, "long_name": ln},
        )
        for k, (v, u, ln) in fields.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[2019, 2020])
    ap.add_argument("--test", type=int, nargs="+", default=[2021])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train, test, stats = D.build(a.train, a.test)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet(train.x.shape[1], train.y.shape[1]).to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(test.x), a.batch):
            xb = torch.from_numpy(test.x[i:i + a.batch]).to(dev)
            preds.append(net(xb).float().cpu().numpy())
    pred_norm = np.concatenate(preds)

    pred = D.to_celsius(pred_norm, test.y_clim, stats["y_sd"])
    truth = D.to_celsius(test.y, test.y_clim, stats["y_sd"])

    # Land stays land: never emit a temperature where the target has none.
    land = ~np.isfinite(truth)
    pred = np.where(land, np.nan, pred)

    year = a.test[0]
    da = _cube(pred, test.time, "thetao", "degrees_C",
               "sea water potential temperature, reconstructed from surface satellite observations")
    da.attrs["source"] = "OceanEmbed U-Net; inputs SST/SSS/SLA/currents/wind"
    da.attrs["train_years"] = str(a.train)
    da.to_netcdf(OUT / f"pred_{year}.nc")
    print(f"  wrote {OUT / f'pred_{year}.nc'}  {dict(da.sizes)}")

    diags = {**_diagnostics(pred, test.time, "pred"),
             **_diagnostics(truth, test.time, "true")}
    xr.Dataset(diags).to_netcdf(OUT / f"diag_{year}.nc")
    print(f"  wrote {OUT / f'diag_{year}.nc'}  {list(diags)}")

    # Quick skill line for the headline diagnostic.
    m = stats["mask"]
    for k in ("tchp", "d26", "mld"):
        p = diags[f"{k}_pred"].values[:, m]
        t = diags[f"{k}_true"].values[:, m]
        ok = np.isfinite(p) & np.isfinite(t)
        rmse = float(np.sqrt(((p[ok] - t[ok]) ** 2).mean()))
        r = float(np.corrcoef(p[ok], t[ok])[0, 1])
        print(f"  {k:5s} RMSE {rmse:8.2f}  corr {r:.3f}  ({diags[f'{k}_true'].units})")


if __name__ == "__main__":
    main()
