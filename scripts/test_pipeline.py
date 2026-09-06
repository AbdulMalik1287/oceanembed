#!/usr/bin/env python
"""Self-check for the regridding logic. No network, no data needed.

    python -m scripts.test_pipeline

Regridding and depth interpolation are the two places this pipeline can be
silently wrong: a half-cell offset or a flipped depth axis produces plausible
numbers that are quietly shifted. These assertions fail loudly instead.

The grid origins below are the real ones, read off one probe day of each
Copernicus product — not invented for the test.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .config import STD_DEPTHS, TARGET_LAT, TARGET_LON
from .pipeline import to_std_depths, to_target_grid

# label -> (native step, native origin in latitude), from real files
PRODUCTS = {
    "SLA / currents 0.25": (0.25, 4.625),
    "SSS / wind 0.125": (0.125, 4.5625),
    "SST (OSTIA) 0.05": (0.05, 4.525),
    "GLORYS 1/12": (1 / 12, 4.5),
}
ALIGNED = {"SLA / currents 0.25", "SSS / wind 0.125", "SST (OSTIA) 0.05"}


def _coords(step: float, origin: float):
    lat = origin + step * np.arange(int(round((30.5 - origin) / step)))
    lon = (origin + 40.0) + step * np.arange(int(round((105.5 - origin - 40.0) / step)))
    return lat, lon


def _analytic(lat, lon):
    """Smooth field: area-averaging and interpolation must both recover it."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return 20.0 + 0.1 * la + 0.05 * lo


def _da(step, origin, fill=None, name="x"):
    lat, lon = _coords(step, origin)
    vals = _analytic(lat, lon) if fill is None else np.full((len(lat), len(lon)), fill)
    # float32 coordinates, as the real netCDF files store them. The original
    # version of this test used float64 and so never reproduced the bug where
    # two products' axes differ by ~2e-6 and xarray outer-joins them.
    return xr.DataArray(
        vals,
        coords={"latitude": lat.astype("float32"), "longitude": lon.astype("float32")},
        dims=("latitude", "longitude"), name=name,
    )


def check_regrid(label: str) -> None:
    step, origin = PRODUCTS[label]
    out = to_target_grid(_da(step, origin))

    assert out.sizes["latitude"] == 100, f"{label}: lat {out.sizes['latitude']} != 100"
    assert out.sizes["longitude"] == 240, f"{label}: lon {out.sizes['longitude']} != 240"
    # BIT-EXACT, not allclose. xarray aligns on exact equality, so coordinates
    # that merely agree to 1e-6 silently outer-join into a padded union grid.
    assert np.array_equal(out["latitude"].values, TARGET_LAT), f"{label}: lat not bit-exact"
    assert np.array_equal(out["longitude"].values, TARGET_LON), f"{label}: lon not bit-exact"

    err = float(np.nanmax(np.abs(out.values - _analytic(TARGET_LAT, TARGET_LON))))
    assert err < 1e-3, f"{label}: max regrid error {err:.2e} on a linear field"
    assert not np.isnan(out.values).any(), f"{label}: NaNs in interior"
    print(f"  ok  {label:22s} step={step:<9.5f} max_err={err:.2e}")


def check_merge_across_products() -> None:
    """Regridded channels from different products must merge without growing.

    This is the assertion that would have caught the 144x394 cube: every
    product goes through to_target_grid, then they are combined into one
    Dataset exactly as build_inputs does it. If any pair of coordinate axes
    disagrees by even one float bit, xarray outer-joins and the grid grows.
    """
    chans = {}
    for i, label in enumerate(PRODUCTS):
        step, origin = PRODUCTS[label]
        chans[f"c{i}"] = to_target_grid(_da(step, origin, name=f"c{i}"))

    cube = xr.Dataset(chans)
    assert cube.sizes["latitude"] == 100 and cube.sizes["longitude"] == 240, (
        f"merged grid is {cube.sizes['latitude']}x{cube.sizes['longitude']}, "
        f"expected 100x240 — coordinates disagree between products"
    )
    for v in cube.data_vars:
        assert not np.isnan(cube[v].values).any(), f"{v}: NaN padding from an outer join"
    print(f"  ok  merge {len(chans)} products    stays 100x240, no NaN padding")


def check_surface_squeeze() -> None:
    """A singleton or two-level vertical axis must not survive into a channel."""
    from .pipeline import surface

    base = _da(0.25, 4.625)
    one = base.expand_dims(depth=[0.494])
    assert "depth" not in surface(one).dims, "singleton depth survived"

    # negative elevation, deepest-first: index 0 is 15 m, not the surface
    two = xr.concat([base, base + 5.0], dim="elevation").assign_coords(
        elevation=[-15.0, 0.0]
    )
    got = surface(two)
    assert "elevation" not in got.dims, "elevation axis survived"
    assert float(got[0, 0]) == float(base[0, 0] + 5.0), "picked 15 m instead of the surface"
    print("  ok  surface squeeze        drops depth axis, picks the level nearest 0")


def check_no_nan_bleed() -> None:
    """On an aligned product, land NaNs must not spread into neighbouring cells.

    This is the whole reason the target grid is cell-centred. A partially-land
    coarse cell keeps its ocean mean (skipna); only a fully-NaN block goes NaN.
    """
    step, origin = PRODUCTS["SST (OSTIA) 0.05"]
    da = _da(step, origin)
    factor = 5

    # one isolated fine cell of "land" inside an otherwise wet block
    speckled = da.copy()
    speckled[40, 60] = np.nan
    assert not np.isnan(to_target_grid(speckled).values).any(), \
        "a single fine-cell NaN must be absorbed by the block mean, not propagated"

    # two whole coarse blocks fully land -> exactly two NaN output cells
    solid = da.copy()
    r0, c0 = 10 * factor, 20 * factor
    solid[r0:r0 + factor, c0:c0 + 2 * factor] = np.nan
    n = int(np.isnan(to_target_grid(solid).values).sum())
    assert n == 2, f"expected exactly 2 NaN cells, got {n} — mask is bleeding"
    print("  ok  land mask stays crisp   1 speckle absorbed, 2 solid blocks -> 2 NaN")


def check_glorys_uses_interp() -> None:
    """GLORYS is the known-unaligned product; confirm it still lands correctly."""
    step, origin = PRODUCTS["GLORYS 1/12"]
    out = to_target_grid(_da(step, origin))
    np.testing.assert_allclose(out["latitude"].values, TARGET_LAT, atol=1e-6)
    err = float(np.nanmax(np.abs(out.values - _analytic(TARGET_LAT, TARGET_LON))))
    assert err < 1e-3, f"GLORYS interp error {err:.2e}"
    print(f"  ok  GLORYS off-grid path   interpolated onto target, max_err={err:.2e}")


def check_depths() -> None:
    """A field linear in depth must be recovered exactly at the standard depths."""
    native = np.array(
        [0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.930, 9.573, 11.405, 13.467,
         15.810, 18.496, 21.599, 25.211, 29.445, 34.434, 40.344, 47.374, 55.764,
         65.807, 77.854, 92.326, 109.729, 130.666, 155.851, 186.126, 222.475,
         266.040, 318.127, 380.213, 453.938, 541.089, 643.567, 763.333, 902.339,
         1062.440]
    )
    assert len(native) == 36, "GLORYS has 36 levels between 0 and 1062 m"
    da = xr.DataArray(30.0 - 0.02 * native, coords={"depth": native}, dims=("depth",))
    out = to_std_depths(da)
    np.testing.assert_allclose(out["depth"].values, STD_DEPTHS)
    np.testing.assert_allclose(out.values, 30.0 - 0.02 * STD_DEPTHS, atol=1e-6)
    print(f"  ok  depth interp           36 levels -> {len(STD_DEPTHS)} standard depths")


def check_negative_elevation() -> None:
    """CMEMS sometimes exposes the vertical axis as negative 'elevation'."""
    elev = np.array([-1000.0, -500.0, -100.0, -10.0, -0.5])
    da = xr.DataArray(30.0 + 0.02 * elev, coords={"elevation": elev}, dims=("elevation",))
    out = to_std_depths(da)
    assert "depth" in out.coords and "elevation" not in out.coords
    assert out["depth"].values[0] < out["depth"].values[-1], "depth must be ascending"
    np.testing.assert_allclose(out.sel(depth=100.0).item(), 30.0 - 2.0, atol=1e-6)
    print("  ok  negative elevation      flipped to positive depth")


def main() -> None:
    print("regrid (real product grid registrations):")
    for label in PRODUCTS:
        check_regrid(label)
    print("cross-product alignment:")
    check_merge_across_products()
    check_surface_squeeze()
    print("land mask:")
    check_no_nan_bleed()
    check_glorys_uses_interp()
    print("vertical:")
    check_depths()
    check_negative_elevation()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
