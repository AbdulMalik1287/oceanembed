#!/usr/bin/env python
"""Validation against independent in-situ profiles (problem statement #5).

    python -m scripts.argo inspect <file.nc>       # print one file's structure
    python -m scripts.argo collect --years 2020 2021
    python -m scripts.argo score  --years 2020 2021

Why this matters more than the GLORYS scores. GLORYS is a reanalysis that
assimilates Argo, so training against it is partly emulating a data-assimilation
system, and any validation against a gridded Argo product is only
semi-independent. Comparing directly against the profiles themselves is the
honest test.

The comparison is three-way on identical samples, reconstruction, GLORYS, and
climatology, each against the same observation. GLORYS's own error is the
reference: it had these profiles assimilated, so it is close to a best case, and
the gap between it and the reconstruction is the real cost of using surface data
alone.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON

ARGO_RAW = PROC.parent / "argo_raw"
ARGO_OUT = PROC.parent / "argo"
PRED = PROC.parent / "pred"

# EasyCORA names things in the OceanSITES style; fall back through the likely
# spellings rather than assuming one.
LAT_NAMES = ("LATITUDE", "latitude", "lat")
LON_NAMES = ("LONGITUDE", "longitude", "lon")
TIME_NAMES = ("TIME", "JULD", "time")
TEMP_NAMES = ("TEMP", "TEMP_ADJUSTED", "temperature")
DEPTH_NAMES = ("DEPH", "DEPTH", "PRES", "PRES_ADJUSTED", "depth")
QC_SUFFIX = "_QC"


def _first(ds, names):
    for n in names:
        if n in ds.variables or n in ds.coords:
            return n
    return None


def inspect(path: str) -> None:
    """Print enough of one file to write the reader against, not guess at it."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        print("dims:", dict(ds.sizes))
        print("\nvariables:")
        for v in ds.variables:
            var = ds[v]
            units = var.attrs.get("units", "")
            print(f"  {v:22s} {str(var.dims):32s} {str(var.dtype):10s} {units}")
        for group, names in (("lat", LAT_NAMES), ("lon", LON_NAMES),
                             ("time", TIME_NAMES), ("temp", TEMP_NAMES),
                             ("depth", DEPTH_NAMES)):
            print(f"resolved {group:6s} -> {_first(ds, names)}")
        for attr in ("platform_code", "wmo_platform_code", "source", "institution"):
            if attr in ds.attrs:
                print(f"attr {attr}: {ds.attrs[attr]}")


def pres_to_depth(p, lat):
    """Pressure (dbar) to depth (m), UNESCO 1983 / Fofonoff & Millard.

    EasyCORA reports PRES in dbar. Treating dbar as metres is wrong by about 1%,
    which is 1 m at 100 m, and in a thermocline running 0.05 degC per metre
    that is a systematic 0.05 degC error smeared across exactly the depths this
    project claims skill at. Cheap to do properly.
    """
    p = np.asarray(p, dtype="float64")
    x = np.sin(np.radians(lat)) ** 2
    g = 9.780318 * (1.0 + 5.2788e-3 * x + 2.36e-5 * x * x) + 1.092e-6 * p
    num = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return num / g


def _qc_mask(ds, var):
    """Keep only values flagged good (1) or probably good (2) where QC exists."""
    qc = var + QC_SUFFIX
    if qc not in ds.variables:
        return None
    flags = ds[qc].values
    if flags.dtype.kind in "SU":
        flags = np.char.strip(flags.astype(str))
        return np.isin(flags, ["1", "2"])
    return np.isin(flags, [1, 2])


def read_profiles(path: str) -> list[dict]:
    """Return in-region profiles as {lat, lon, time, depth[], temp[]}."""
    out = []
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        ln, lo = _first(ds, LAT_NAMES), _first(ds, LON_NAMES)
        tn, tv = _first(ds, TIME_NAMES), _first(ds, TEMP_NAMES)
        dn = _first(ds, DEPTH_NAMES)
        if not all((ln, lo, tn, tv, dn)):
            return out

        lat = np.atleast_1d(ds[ln].values)
        lon = np.atleast_1d(ds[lo].values)
        time = np.atleast_1d(ds[tn].values)
        temp = np.atleast_2d(ds[tv].values)
        dep = np.atleast_2d(ds[dn].values)
        if dep.shape != temp.shape:                      # depth may be 1-D
            dep = np.broadcast_to(dep.reshape(1, -1), temp.shape)

        good = _qc_mask(ds, tv)
        temp = np.where(good, temp, np.nan) if good is not None else temp
        # Drop levels whose pressure itself failed QC, else a bad depth silently
        # relocates a good temperature.
        good_z = _qc_mask(ds, dn)
        if good_z is not None:
            dep = np.where(good_z, dep, np.nan)

        is_pressure = "dbar" in str(ds[dn].attrs.get("units", "")).lower()

        inbox = ((lat >= LAT_MIN) & (lat <= LAT_MAX)
                 & (lon >= LON_MIN) & (lon <= LON_MAX))
        for i in np.flatnonzero(inbox):
            z, t = dep[i], temp[i]
            if is_pressure:
                z = pres_to_depth(z, lat[i])
            ok = np.isfinite(z) & np.isfinite(t) & (z >= 0) & (z <= 1100)
            if ok.sum() < 5:
                continue
            out.append(dict(lat=float(lat[i]), lon=float(lon[i]),
                            time=np.datetime64(time[i], "D"),
                            depth=z[ok], temp=t[ok]))
    return out


def to_std(prof: dict) -> np.ndarray:
    """Profile onto the 15 standard depths; NaN outside its sampled range."""
    z, t = prof["depth"], prof["temp"]
    order = np.argsort(z)
    z, t = z[order], t[order]
    out = np.interp(STD_DEPTHS, z, t, left=np.nan, right=np.nan)
    # np.interp clamps rather than extrapolating, so blank anything outside the
    # profile's actual span, an Argo float that stopped at 500 m must not be
    # credited with a value at 1000 m.
    return np.where((STD_DEPTHS >= z[0]) & (STD_DEPTHS <= z[-1]), out, np.nan)


def collect(years) -> Path:
    """Match every in-region profile to the model, GLORYS and climatology."""
    from .data import day_of_year_climatology, load_years

    files = sorted(glob.glob(str(ARGO_RAW / "**" / "*.nc"), recursive=True))
    if not files:
        raise SystemExit(f"no Argo files under {ARGO_RAW}")
    print(f"reading {len(files)} EasyCORA files")

    profs = []
    for f in files:
        try:
            profs.extend(read_profiles(f))
        except Exception as e:                            # one bad file must not stop the run
            print(f"  skip {Path(f).name}: {type(e).__name__}: {e}")
    print(f"  {len(profs)} profiles in region")
    if not profs:
        raise SystemExit("no profiles in the region, check the download filter")

    # Model output and truth for the test years.
    pred = xr.open_mfdataset([str(PRED / f"pred_{y}.nc") for y in years],
                             combine="by_coords")["thetao"].load()
    _, tgt = load_years(years)
    tgt = tgt.load()
    clim = day_of_year_climatology(load_years([2014, 2015, 2016, 2017, 2018, 2019])[1])

    rows = []
    for p in profs:
        if p["time"] not in pred["time"].dt.floor("D").values:
            continue
        obs = to_std(p)
        if not np.isfinite(obs).any():
            continue
        i = int(np.argmin(np.abs(TARGET_LAT - p["lat"])))
        j = int(np.argmin(np.abs(TARGET_LON - p["lon"])))
        sel = dict(time=p["time"], latitude=TARGET_LAT[i], longitude=TARGET_LON[j])
        rows.append(dict(
            obs=obs,
            model=pred.sel(**sel, method="nearest").values,
            glorys=tgt.sel(**sel, method="nearest").values,
            clim=clim.sel(dayofyear=int(pd.Timestamp(p["time"]).dayofyear),
                          latitude=TARGET_LAT[i], longitude=TARGET_LON[j],
                          method="nearest").values,
            lat=p["lat"], lon=p["lon"], time=p["time"],
        ))

    ARGO_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(ARGO_OUT / "matched.npz",
             obs=np.array([r["obs"] for r in rows]),
             model=np.array([r["model"] for r in rows]),
             glorys=np.array([r["glorys"] for r in rows]),
             clim=np.array([r["clim"] for r in rows]),
             lat=np.array([r["lat"] for r in rows]),
             lon=np.array([r["lon"] for r in rows]),
             time=np.array([str(r["time"]) for r in rows]))
    print(f"  matched {len(rows)} profiles -> {ARGO_OUT / 'matched.npz'}")
    return ARGO_OUT / "matched.npz"


def score() -> None:
    import json

    d = np.load(ARGO_OUT / "matched.npz", allow_pickle=True)
    obs, model, glorys, clim = d["obs"], d["model"], d["glorys"], d["clim"]
    print(f"\nIndependent validation against {len(obs)} in-situ profiles, 2020-2021")
    print("GLORYS assimilated these same profiles, so its column is a best case,")
    print("not an independent competitor.\n")
    print(f"{'depth':>6} {'n':>6} | {'clim':>6} | {'GLORYS':>6} | "
          f"{'ours':>6} {'bias':>6} {'corr':>6} | {'vs clim':>8}")
    print("-" * 66)

    rows, tot = [], {k: [] for k in ("clim", "glorys", "model")}
    for k, z in enumerate(STD_DEPTHS):
        ok = (np.isfinite(obs[:, k]) & np.isfinite(model[:, k])
              & np.isfinite(glorys[:, k]) & np.isfinite(clim[:, k]))
        if ok.sum() < 20:
            print(f"{z:6.0f} {int(ok.sum()):6d} | {'-':>6} | {'-':>6} | "
                  f"{'-':>6} {'-':>6} {'-':>6} | {'-':>8}")
            continue
        r = {"depth": float(z), "n": int(ok.sum())}
        for name, arr in (("clim", clim), ("glorys", glorys), ("model", model)):
            r[name] = float(np.sqrt(np.mean((arr[ok, k] - obs[ok, k]) ** 2)))
            tot[name].append(r[name])
        r["bias"] = float(np.mean(model[ok, k] - obs[ok, k]))
        r["corr"] = float(np.corrcoef(model[ok, k], obs[ok, k])[0, 1])
        r["gain"] = 100 * (r["clim"] - r["model"]) / r["clim"]
        rows.append(r)
        print(f"{z:6.0f} {r['n']:6d} | {r['clim']:6.3f} | {r['glorys']:6.3f} | "
              f"{r['model']:6.3f} {r['bias']:6.3f} {r['corr']:6.3f} | {r['gain']:+7.1f}%")

    print("-" * 66)
    mc, mg, mm = (np.mean(tot[k]) for k in ("clim", "glorys", "model"))
    print(f"{'mean':>6} {'':>6} | {mc:6.3f} | {mg:6.3f} | {mm:6.3f} "
          f"{'':>13} | {100 * (mc - mm) / mc:+7.1f}%")
    print(f"\nGLORYS is {100 * (mm - mg) / mm:.0f}% better than the reconstruction, the cost of")
    print("using surface satellite data alone, with no in-situ input.")

    ARGO_OUT.mkdir(parents=True, exist_ok=True)
    with open(ARGO_OUT / "argo_scores.json", "w") as fh:
        json.dump({"n_profiles": int(len(obs)), "rows": rows,
                   "mean": {"clim": mc, "glorys": mg, "model": mm}}, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("path")
    c = sub.add_parser("collect"); c.add_argument("--years", type=int, nargs="+", default=[2020, 2021])
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.path)
    elif a.cmd == "collect":
        collect(a.years)
    else:
        score()


if __name__ == "__main__":
    main()
