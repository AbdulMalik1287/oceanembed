#!/usr/bin/env python
"""OceanEmbed data pipeline: fetch from Copernicus Marine, regrid to the working grid.

    python -m scripts.pipeline fetch sst --year 2019
    python -m scripts.pipeline fetch-all
    python -m scripts.pipeline build-target 2019
    python -m scripts.pipeline build-inputs 2019

Credentials come from the environment (COPERNICUSMARINE_SERVICE_USERNAME /
COPERNICUSMARINE_SERVICE_PASSWORD) or ~/.copernicusmarine, never from this file.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from .config import (
    CHANNELS, INPUTS, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROC, RAW,
    RES, STD_DEPTHS, TARGET, TARGET_LAT, TARGET_LON, YEAR_END, YEAR_START,
)

# Fetch a slightly larger box than we keep: so interpolation onto the exact
# target grid has data on all four edges instead of producing a NaN border.
PAD = 0.5

# HDF5 is not thread-safe. Writing a dask-backed array to netCDF while reading
# from other netCDF files under dask's default threaded scheduler deadlocks:
# the process sits at 0% CPU forever with the output file frozen part-written,
# and nothing appears in any log. Observed here on build-inputs 2019, 65
# minutes wall clock for 16 seconds of CPU. The synchronous scheduler keeps the
# chunked streaming (so memory stays low) without the threading. Parallelism
# comes from running whole years as separate processes instead.
dask.config.set(scheduler="synchronous")


# --------------------------------------------------------------------------- #
# grid handling
# --------------------------------------------------------------------------- #
def _aligned(src, target, tol: float = 1e-4) -> bool:
    """True if every target coordinate exists in src (so we can select, not interp)."""
    if len(src) < len(target):
        return False
    idx = np.searchsorted(src, target)
    idx = np.clip(idx, 1, len(src) - 1)
    nearest = np.where(
        np.abs(src[idx] - target) < np.abs(src[idx - 1] - target), src[idx], src[idx - 1]
    )
    return bool(np.all(np.abs(nearest - target) < tol))


def _latlon_names(obj) -> tuple[str, str]:
    lat = next((n for n in ("latitude", "lat", "nav_lat") if n in obj.coords), None)
    lon = next((n for n in ("longitude", "lon", "nav_lon") if n in obj.coords), None)
    if lat is None or lon is None:
        raise KeyError(f"no lat/lon coords found in {list(obj.coords)}")
    return lat, lon


def to_target_grid(obj):
    """Regrid onto the canonical 0.25 deg grid (100 x 240).

    Native spacings are all exact integer divisors of 0.25 (GLORYS 1/12 -> 3,
    OSTIA 0.05 -> 5, wind 0.125 -> 2, DUACS 0.25 -> 1), so an area-mean coarsen
    is the correct anti-aliasing step and no external regridder is needed. The
    factor is derived from the file rather than hardcoded, so a product that
    changes resolution cannot silently produce a wrong answer. The final interp
    pins the result to the exact target coordinates regardless of any half-cell
    offset in the source grid.
    """
    lat, lon = _latlon_names(obj)
    step = float(abs(obj[lat].diff(lat).mean()))
    factor = int(round(RES / step))
    if factor < 1:
        raise ValueError(f"source spacing {step} is coarser than target {RES}")
    if factor > 1:
        obj = obj.coarsen({lat: factor, lon: factor}, boundary="trim").mean(skipna=True)

    # Every surface product lands exactly on the target grid after coarsening
    # (see config.TARGET_LAT). Selecting instead of interpolating there keeps the
    # land mask crisp, interpolation would average each coastal cell with its
    # NaN neighbour and eat a cell of ocean all the way round the basin.
    if _aligned(obj[lat].values, TARGET_LAT) and _aligned(obj[lon].values, TARGET_LON):
        out = obj.sel({lat: TARGET_LAT, lon: TARGET_LON}, method="nearest", tolerance=1e-4)
    else:
        out = obj.interp({lat: TARGET_LAT, lon: TARGET_LON})

    renames = {}
    if lat != "latitude":
        renames[lat] = "latitude"
    if lon != "longitude":
        renames[lon] = "longitude"
    if renames:
        out = out.rename(renames)

    # Stamp the exact target coordinates. `sel` returns the SOURCE coordinate
    # values, which are float32 and differ between products in the last bits
    # (~2e-6). xarray aligns on exact equality, so combining channels then
    # outer-joins them into a NaN-padded union, observed as a 144x394 cube
    # instead of 100x240, 78% NaN. Assigning the canonical float64 axes makes
    # every product bit-identical and the merge a no-op.
    # ponytail: GLORYS is the one product that misses the target grid (1/12 deg
    # origin 4.5, so coarsening lands 1/24 deg off) and therefore still goes
    # through interp, bleeding land NaNs one target cell into the coast. It is
    # the target field, so the cost is losing coastal pixels from training and
    # scoring, not corrupting them. Swap in a masked conservative regrid (xesmf)
    # if coastal skill turns out to matter.
    return out.assign_coords(latitude=TARGET_LAT, longitude=TARGET_LON)


def surface(obj):
    """Drop a singleton vertical dimension so a channel is purely (time, y, x).

    Several CMEMS surface products keep a length-1 depth axis (SSS) or a real
    one we only want the top of (currents, levels 0 and 15 m). Left in place it
    propagates into the assembled cube as a stray `depth: 1` dimension.
    """
    for name in ("depth", "elevation", "deptht"):
        if name in obj.dims:
            # Nearest to zero by value, not index, the axis may be stored as
            # negative elevation or in either order, so isel(0) can silently
            # pick 15 m instead of the surface.
            obj = obj.isel({name: int(np.abs(obj[name].values).argmin())}, drop=True)
        elif name in obj.coords:
            obj = obj.drop_vars(name)
    return obj


def to_std_depths(obj):
    """Interpolate the native vertical levels onto the 15 standard depths."""
    dname = next((n for n in ("depth", "elevation", "deptht") if n in obj.coords), None)
    if dname is None:
        raise KeyError(f"no depth coord in {list(obj.coords)}")
    if float(obj[dname].max()) <= 0:          # stored as negative elevation
        obj = obj.assign_coords({dname: -obj[dname]}).sortby(dname)
    # Only depth 0 m is an extrapolation, and only by 0.49 m past the shallowest
    # native level; everything else is interior (1000 m < 1062 m deepest level).
    out = obj.interp({dname: STD_DEPTHS}, kwargs={"fill_value": "extrapolate"})
    return out.rename({dname: "depth"}) if dname != "depth" else out


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def _atomic_write(obj, out: Path) -> None:
    """Write to a temp file in the same directory, then rename into place.

    Without this, a build killed part-way leaves a truncated .nc that the
    skip-if-exists check happily treats as finished, training then reads
    garbage, or an HDF error, with nothing in the log to say why. os.replace is
    atomic within a filesystem, so the destination either does not exist or is
    complete.
    """
    tmp = out.with_name(out.name + ".tmp")
    obj.to_netcdf(tmp)
    tmp.replace(out)


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def fetch(key: str, year: int, spec: dict, dest: Path, overwrite: bool = False) -> Path:
    import copernicusmarine as cm

    out = dest / f"{key}_{year}.nc"
    if out.exists() and not overwrite:
        print(f"  skip (exists) {out.name}")
        return out
    if _free_gb(dest) < 25:
        sys.exit(f"ABORT: under 25 GB free at {dest}; free space before fetching.")

    kw = dict(
        dataset_id=spec["id"],
        variables=spec["vars"],
        minimum_longitude=LON_MIN - PAD,
        maximum_longitude=LON_MAX + PAD,
        minimum_latitude=LAT_MIN - PAD,
        maximum_latitude=LAT_MAX + PAD,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=str(dest),
        output_filename=out.name,
        overwrite=True,
    )
    if "depth" in spec:
        kw["minimum_depth"], kw["maximum_depth"] = spec["depth"]

    print(f"  fetching {key} {year} <- {spec['id']}")
    cm.subset(**kw)
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_target(year: int) -> Path:
    """GLORYS 1/12 deg, 36 levels -> 0.25 deg, 15 standard depths."""
    src = RAW / f"glorys_{year}.nc"
    out = PROC / f"target_{year}.nc"
    PROC.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  skip (exists) {out.name}")
        return out
    with xr.open_dataset(src, chunks={"time": 30}) as ds:
        da = to_std_depths(to_target_grid(ds["thetao"]))
        da.name = "thetao"
        da.attrs["units"] = "degrees_C"
        _atomic_write(da.astype("float32").load(), out)   # ~50 MB; keep dask out of the write
    print(f"  wrote {out}")
    return out


def build_inputs(year: int) -> Path:
    """Assemble the 7 surface channels into one aligned daily 0.25 deg cube."""
    out = PROC / f"inputs_{year}.nc"
    PROC.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"  skip (exists) {out.name}")
        return out
    chans = {}

    with xr.open_dataset(RAW / f"sst_{year}.nc") as ds:
        chans["sst"] = to_target_grid(surface(ds["analysed_sst"])) - 273.15   # K -> degC

    with xr.open_dataset(RAW / f"sss_{year}.nc") as ds:
        chans["sss"] = to_target_grid(surface(ds["sos"]))

    with xr.open_dataset(RAW / f"sla_{year}.nc") as ds:
        chans["sla"] = to_target_grid(surface(ds["sla"]))

    with xr.open_dataset(RAW / f"cur_{year}.nc") as ds:
        cur = to_target_grid(surface(ds[["uo", "vo"]]))
        chans["uo"], chans["vo"] = cur["uo"], cur["vo"]

    # Wind is hourly; average to daily on the native grid before regridding.
    with xr.open_dataset(RAW / f"wind_{year}.nc", chunks={"time": 24 * 7}) as ds:
        w = surface(ds[["eastward_wind", "northward_wind"]]).resample(time="1D").mean()
        # Realise the daily means here (~300 MB) so nothing dask-backed reaches
        # to_netcdf; combined with the synchronous scheduler above this keeps
        # the write off HDF5's non-reentrant path entirely.
        w = to_target_grid(w).load()
        chans["wind_u"] = w["eastward_wind"]
        chans["wind_v"] = w["northward_wind"]

    # Align every channel on the intersection of available days.
    for k, v in chans.items():
        chans[k] = v.assign_coords(time=v["time"].dt.floor("D"))
    common = sorted(set.intersection(*(set(v["time"].values) for v in chans.values())))
    cube = xr.Dataset(chans).sel(time=common)[CHANNELS]
    _atomic_write(cube.astype("float32"), out)
    print(f"  wrote {out}  days={cube.sizes['time']}")
    return out


# --------------------------------------------------------------------------- #
def verify_year(year: int) -> bool:
    """Open both processed cubes and check their shape. Cheap, and the only
    thing standing between a truncated file and a silently poisoned training
    set."""
    try:
        with xr.open_dataset(PROC / f"target_{year}.nc") as ds:
            assert ds.sizes["latitude"] == 100 and ds.sizes["longitude"] == 240
            assert ds.sizes["depth"] == len(STD_DEPTHS)
        with xr.open_dataset(PROC / f"inputs_{year}.nc") as ds:
            assert ds.sizes["latitude"] == 100 and ds.sizes["longitude"] == 240
            assert "depth" not in ds.sizes and len(ds.data_vars) == len(CHANNELS)
        return True
    except Exception as e:
        print(f"  VERIFY FAILED {year}: {type(e).__name__}: {e}")
        return False


def stage_year(year: int, keep_raw: bool = False) -> bool:
    """fetch -> build -> verify -> delete raw, one year at a time.

    A full record does not fit on disk as raw files (~9.6 GB/year against
    ~137 GB free), but the processed cubes are only ~0.8 GB/year. Staging keeps
    peak usage at roughly one year of raw. Raw is deleted ONLY after both cubes
    have been opened and checked, never on the assumption that the build
    finished.
    """
    print(f"=== {year}")
    for key, spec in {**INPUTS, "glorys": TARGET["glorys"]}.items():
        fetch(key, year, spec, RAW)
    build_target(year)
    build_inputs(year)

    if not verify_year(year):
        print(f"  keeping raw for {year} so the build can be retried")
        return False
    if keep_raw:
        return True

    freed = 0
    for key in list(INPUTS) + ["glorys"]:
        f = RAW / f"{key}_{year}.nc"
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    print(f"  verified; freed {freed / 1024**3:.1f} GB of raw for {year}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("key")
    f.add_argument("--year", type=int, required=True)
    sub.add_parser("fetch-all")
    b = sub.add_parser("build-target")
    b.add_argument("year", type=int)
    i = sub.add_parser("build-inputs")
    i.add_argument("year", type=int)
    st = sub.add_parser("stage", help="fetch -> build -> verify -> delete raw, per year")
    st.add_argument("--years", type=int, nargs="+", required=True)
    st.add_argument("--keep-raw", action="store_true")

    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    if a.cmd == "fetch":
        fetch(a.key, a.year, {**INPUTS, **TARGET}[a.key], RAW)
    elif a.cmd == "fetch-all":
        print(f"free: {_free_gb(RAW):.0f} GB at {RAW}")
        for year in range(YEAR_START, YEAR_END + 1):
            for key, spec in {**INPUTS, "glorys": TARGET["glorys"]}.items():
                fetch(key, year, spec, RAW)
    elif a.cmd == "build-target":
        build_target(a.year)
    elif a.cmd == "build-inputs":
        build_inputs(a.year)
    elif a.cmd == "stage":
        failed = [y for y in a.years if not stage_year(y, keep_raw=a.keep_raw)]
        print(f"\nstaged {len(a.years) - len(failed)}/{len(a.years)} years"
              + (f"; FAILED: {failed}" if failed else ""))
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
