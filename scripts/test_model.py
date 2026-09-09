#!/usr/bin/env python
"""Self-check for the model, loss, climatology and metrics. No data needed.

    python -m scripts.test_model

Targets the things that fail silently: a U-Net that quietly crops the wrong
region because 100 is not divisible by 8, a loss that trains on land, a
climatology whose smoothing eats the turn of the year, and skill metrics that
look fine because they are averaging over NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .data import CLIM_SMOOTH_DAYS, N_IN, N_OUT, day_of_year_climatology, to_celsius
from .metrics import skill

NY, NX = 100, 240


def check_metrics() -> None:
    rng = np.random.default_rng(0)
    n, nd = 6, 3
    depths = [0.0, 100.0, 500.0]
    mask = np.zeros((NY, NX), dtype=bool)
    mask[10:40, 20:80] = True

    clim = rng.normal(20, 1, (n, nd, NY, NX))
    true = clim + rng.normal(0, 2, (n, nd, NY, NX))

    r = skill(true.copy(), true, clim, mask, depths)
    assert all(abs(x["rmse"]) < 1e-9 for x in r), "perfect prediction must give RMSE 0"
    assert all(abs(x["acc"] - 1) < 1e-9 for x in r), "perfect prediction must give ACC 1"
    assert all(x["n"] == int(mask.sum()) * n for x in r), "wrong sample count, mask ignored?"

    r = skill(true + 0.5, true, clim, mask, depths)
    assert all(abs(x["bias"] - 0.5) < 1e-9 for x in r), "bias not detected"

    # predicting the climatology exactly => zero anomaly => ACC undefined (nan)
    r = skill(clim.copy(), true, clim, mask, depths)
    assert all(np.isnan(x["acc"]) for x in r), "flat climatology should give ACC nan"

    # NaNs in truth must be dropped, not averaged in
    t2 = true.copy()
    t2[:, :, 10:20, 20:40] = np.nan
    r = skill(true.copy(), t2, clim, mask, depths)
    assert all(np.isfinite(x["rmse"]) for x in r), "NaN truth leaked into RMSE"
    assert all(x["n"] < int(mask.sum()) * n for x in r), "NaN cells were not dropped"
    print("  ok  metrics            perfect/biased/NaN cases behave")


def check_climatology() -> None:
    time = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    # constant in space and time except a clean seasonal cycle
    doy = time.dayofyear.values
    sig = 25 + 3 * np.sin(2 * np.pi * doy / 365.25)
    da = xr.DataArray(
        np.broadcast_to(sig[:, None, None, None], (len(time), 2, 4, 5)).copy(),
        coords={"time": time, "depth": [0.0, 10.0],
                "latitude": np.arange(4), "longitude": np.arange(5)},
        dims=("time", "depth", "latitude", "longitude"),
    )
    clim = day_of_year_climatology(da)
    assert clim.sizes["dayofyear"] == 366, f"got {clim.sizes['dayofyear']} days"
    assert np.isfinite(clim.values).all(), "climatology has NaNs after smoothing"

    # circular smoothing: 31 Dec and 1 Jan must stay close, not collapse
    jump = abs(float(clim.isel(dayofyear=365).mean()) - float(clim.isel(dayofyear=0).mean()))
    assert jump < 0.2, f"year boundary discontinuity {jump:.3f}, wrap failed"
    # a 31-day mean of a smooth annual cycle barely shifts the amplitude
    err = float(np.abs(clim.mean(("depth", "latitude", "longitude")).values
                       - (25 + 3 * np.sin(2 * np.pi * np.arange(1, 367) / 365.25))).max())
    assert err < 0.15, f"climatology distorted the seasonal cycle by {err:.3f}"
    print(f"  ok  climatology        366 doy, wrap intact (jump {jump:.4f})")


def check_static_channels() -> None:
    """The 4 appended static channels: day-of-year sin/cos, then lat and lon.

    Previously untested, and it broke the first real run, `TARGET_LAT.ptp()`
    is gone in NumPy 2. Cheap to cover, so cover it.
    """
    from .data import N_STATIC, _static_channels

    times = pd.date_range("2019-01-01", periods=5, freq="D").values
    s = _static_channels(times, (NY, NX))
    assert s.shape == (5, N_STATIC, NY, NX), f"bad shape {s.shape}"
    assert np.isfinite(s).all(), "non-finite static channel"

    # doy channels are constant in space, varying in time
    assert np.allclose(s[0, 0], s[0, 0, 0, 0]), "doy sin varies in space"
    assert not np.isclose(s[0, 0, 0, 0], s[4, 0, 0, 0]), "doy sin constant in time"
    # 1 Jan: sin(2*pi/365.25) ~ 0.0172, cos ~ 0.9999
    assert abs(s[0, 0, 0, 0] - np.sin(2 * np.pi / 365.25)) < 1e-6, "doy phase wrong"

    # lat varies down rows only, lon across columns only, both spanning [-1, 1]
    assert np.allclose(s[0, 2], s[0, 2][:, :1]), "lat channel varies along longitude"
    assert np.allclose(s[0, 3], s[0, 3][:1, :]), "lon channel varies along latitude"
    for c in (2, 3):
        assert abs(s[0, c].min() + 1) < 1e-6 and abs(s[0, c].max() - 1) < 1e-6, \
            f"channel {c} not normalised to [-1, 1]"
    print("  ok  static channels      doy sin/cos + lat/lon, normalised, right axes")


def check_roundtrip() -> None:
    rng = np.random.default_rng(1)
    y_sd = np.array([2.0, 1.0, 0.5])
    clim = rng.normal(20, 1, (4, 3, 8, 9))
    absolute = clim + rng.normal(0, 1, (4, 3, 8, 9))
    norm = (absolute - clim) / y_sd[None, :, None, None]
    np.testing.assert_allclose(to_celsius(norm, clim, y_sd), absolute, atol=1e-10)
    print("  ok  anomaly round-trip  normalise -> to_celsius is lossless")


def check_unet() -> None:
    import torch

    from .model import UNet, masked_mse

    net = UNet(N_IN, N_OUT, width=8)
    x = torch.randn(2, N_IN, NY, NX)
    out, emb = net(x, return_embedding=True)
    assert out.shape == (2, N_OUT, NY, NX), f"bad output shape {tuple(out.shape)}"
    # 100 is not divisible by 8: the pad-and-crop must restore the exact grid.
    assert emb.shape[-2:] == (13, 30), f"bottleneck {tuple(emb.shape[-2:])} != (13, 30)"

    mask = torch.zeros(NY, NX, dtype=torch.bool)
    mask[20:60, 30:90] = True
    true = torch.randn(2, N_OUT, NY, NX)
    true[:, :, :10, :] = float("nan")            # land, outside the mask

    loss = masked_mse(out, true, mask)
    assert torch.isfinite(loss), "loss went NaN, land leaked in"
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"

    # land must not influence the loss at all
    o2, t2 = out.detach().clone(), true.clone()
    t2[:, :, 80:, :] = 999.0                      # far outside the mask
    a = masked_mse(o2, true, mask)
    b = masked_mse(o2, t2, mask)
    assert torch.allclose(a, b), "values outside the mask changed the loss"
    print(f"  ok  unet               {sum(p.numel() for p in net.parameters())/1e3:.0f}k params, "
          f"{NY}x{NX} preserved, loss masked")


def main() -> None:
    print("data + metrics:")
    check_metrics()
    check_climatology()
    check_static_channels()
    check_roundtrip()
    print("model:")
    try:
        check_unet()
    except ImportError:
        print("  -- torch not installed yet, skipped")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
