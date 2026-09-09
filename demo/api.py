#!/usr/bin/env python
"""OceanEmbed REST API, serves the reconstruction from the real netCDF files.

    ~/envs/ocean/bin/python -m demo.api          # http://localhost:8000

Endpoints
    GET /api/health                          service + what is loaded
    GET /api/dates                           available days
    GET /api/field?date=&var=&depth=         one 2-D field as JSON
    GET /api/profile?date=&lat=&lon=         reconstruction, GLORYS, climatology
    GET /api/diagnostics?date=&lat=&lon=     TCHP, D26, MLD at a point
    GET /api/scores                          skill tables incl. Argo validation
    GET /                                    the console frontend

The frontend does not need this to run. It ships with a baked bundle so a demo
never depends on a server being up. The API is what an operational deployment
would actually expose, and it reads the same files the science does.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import uvicorn
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from scripts.config import PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON

PRED = PROC.parent / "pred"
RUNS = PROC.parent / "runs"
ARGO = PROC.parent / "argo"
WEB = Path(__file__).parent / "web"

app = FastAPI(title="OceanEmbed", version="1.0",
              description="Subsurface ocean temperature from surface satellite observations")

VARS = {"thetao": "temperature (degC)", "tchp": "cyclone heat potential (kJ/cm2)",
        "d26": "26 degC isotherm depth (m)", "mld": "mixed layer depth (m)"}


@lru_cache(maxsize=4)
def _pred(year: int):
    return xr.open_dataarray(PRED / f"pred_{year}.nc").load()


@lru_cache(maxsize=4)
def _truth(year: int):
    return xr.open_dataarray(PROC / f"target_{year}.nc").load()


@lru_cache(maxsize=4)
def _diag(year: int):
    return xr.open_dataset(PRED / f"diag_{year}.nc").load()


def _year(date: str) -> int:
    try:
        return int(date[:4])
    except ValueError:
        raise HTTPException(400, f"bad date {date!r}, expected YYYY-MM-DD")


def _sel(da, date: str):
    hit = da.sel(time=date, method="nearest")
    if str(hit["time"].values)[:10] != date:
        raise HTTPException(404, f"no data for {date}")
    return hit


def _clean(a):
    """JSON has no NaN. Land and missing values go out as null."""
    return [None if not np.isfinite(v) else round(float(v), 3) for v in np.ravel(a)]


@app.get("/api/health")
def health():
    years = sorted(int(p.stem.split("_")[1]) for p in PRED.glob("pred_*.nc"))
    return {"status": "ok", "years": years, "variables": VARS,
            "depths": [float(d) for d in STD_DEPTHS],
            "grid": {"ny": len(TARGET_LAT), "nx": len(TARGET_LON),
                     "lat": [float(TARGET_LAT[0]), float(TARGET_LAT[-1])],
                     "lon": [float(TARGET_LON[0]), float(TARGET_LON[-1])]}}


@app.get("/api/dates")
def dates(year: int = Query(2021)):
    return {"year": year,
            "dates": [str(t)[:10] for t in _pred(year)["time"].values]}


@app.get("/api/field")
def field(date: str, var: str = "thetao", depth: float = 100.0):
    if var not in VARS:
        raise HTTPException(400, f"unknown var {var!r}; one of {sorted(VARS)}")
    y = _year(date)
    if var == "thetao":
        k = int(np.argmin(np.abs(STD_DEPTHS - depth)))
        a = _sel(_pred(y), date).isel(depth=k).values
        depth_used = float(STD_DEPTHS[k])
    else:
        a = _sel(_diag(y)[f"{var}_pred"], date).values
        depth_used = None
    return {"date": date, "var": var, "units": VARS[var], "depth": depth_used,
            "shape": list(a.shape), "values": _clean(a)}


def _ij(lat: float, lon: float):
    if not (TARGET_LAT[0] - 0.2 <= lat <= TARGET_LAT[-1] + 0.2):
        raise HTTPException(400, f"lat {lat} outside 5-30N")
    if not (TARGET_LON[0] - 0.2 <= lon <= TARGET_LON[-1] + 0.2):
        raise HTTPException(400, f"lon {lon} outside 45-105E")
    return int(np.argmin(np.abs(TARGET_LAT - lat))), int(np.argmin(np.abs(TARGET_LON - lon)))


@app.get("/api/profile")
def profile(date: str, lat: float, lon: float):
    y = _year(date)
    i, j = _ij(lat, lon)
    clim = _climatology()
    doy = int(np.datetime64(date, "D").astype("datetime64[D]").astype(object).timetuple().tm_yday)
    return {
        "date": date,
        "lat": float(TARGET_LAT[i]), "lon": float(TARGET_LON[j]),
        "depths": [float(d) for d in STD_DEPTHS],
        "oceanembed": _clean(_sel(_pred(y), date).isel(latitude=i, longitude=j).values),
        "glorys": _clean(_sel(_truth(y), date).isel(latitude=i, longitude=j).values),
        "climatology": _clean(clim.sel(dayofyear=doy).isel(latitude=i, longitude=j).values),
    }


@lru_cache(maxsize=1)
def _climatology():
    from scripts.data import day_of_year_climatology, load_years
    return day_of_year_climatology(
        load_years([2014, 2015, 2016, 2017, 2018, 2019])[1]
    ).transpose("dayofyear", "depth", "latitude", "longitude").load()


@app.get("/api/diagnostics")
def diagnostics(date: str, lat: float, lon: float):
    y = _year(date)
    i, j = _ij(lat, lon)
    d = _diag(y)
    out = {"date": date, "lat": float(TARGET_LAT[i]), "lon": float(TARGET_LON[j])}
    for k, unit in (("tchp", "kJ cm-2"), ("d26", "m"), ("mld", "m"), ("ohc", "kJ cm-2")):
        v = float(_sel(d[f"{k}_pred"], date).isel(latitude=i, longitude=j).values)
        out[k] = {"value": None if not np.isfinite(v) else round(v, 2), "units": unit}
    t = out["tchp"]["value"]
    out["cyclone_risk"] = ("unknown" if t is None else
                           "high" if t >= 90 else "elevated" if t >= 60 else "low")
    return out


@app.get("/api/scores")
def scores():
    out = {"models": {}}
    for m in ("climatology", "linear", "unet"):
        p = RUNS / f"{m}.json"
        if p.exists():
            out["models"][m] = json.loads(p.read_text())
    p = ARGO / "argo_scores.json"
    if p.exists():
        out["argo_validation"] = json.loads(p.read_text())
    return out


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/data.js")
def data_bundle():
    """The page loads this as a sibling of index.html, so it must live at the
    root and not under /static."""
    p = WEB / "data.js"
    if not p.exists():
        raise HTTPException(404, "run `python -m demo.precompute` first")
    return FileResponse(p, media_type="application/javascript")


if __name__ == "__main__":
    print("OceanEmbed API  ->  http://localhost:8000   (docs at /docs)")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
