#!/usr/bin/env python
"""Assemble model-ready arrays from the processed cubes.

Everything the model sees is built here: the input stack, the day-of-year
climatology, the anomaly target, the land mask, and the normalisation
statistics. All of them are derived from the TRAINING years only — a
climatology or a mean computed over the test years leaks the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .config import CHANNELS, PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON

# 7 surface channels + day-of-year (sin, cos) + normalised lat, lon.
# The convolutions are translation-equivariant, so without lat/lon the network
# cannot tell the Bay of Bengal from the Arabian Sea — two basins with very
# different stratification under similar surface signatures.
N_STATIC = 4
N_IN = len(CHANNELS) + N_STATIC
N_OUT = len(STD_DEPTHS)
CLIM_SMOOTH_DAYS = 31


@dataclass
class Split:
    x: np.ndarray          # (n, N_IN, lat, lon)  normalised, NaN-filled
    y: np.ndarray          # (n, N_OUT, lat, lon) anomaly, normalised, NaN kept
    y_clim: np.ndarray     # (n, N_OUT, lat, lon) climatology in degrees C
    time: np.ndarray

    def __len__(self) -> int:
        return len(self.time)


def load_years(years) -> tuple[xr.Dataset, xr.DataArray]:
    """Open processed input/target cubes for the given years, aligned on time."""
    ins = xr.concat([xr.open_dataset(PROC / f"inputs_{y}.nc") for y in years], dim="time")
    tgt = xr.concat(
        [xr.open_dataarray(PROC / f"target_{y}.nc") for y in years], dim="time"
    )
    common = np.intersect1d(ins.time.values, tgt.time.values)
    return ins.sel(time=common), tgt.sel(time=common)


def day_of_year_climatology(tgt: xr.DataArray) -> xr.DataArray:
    """Smoothed day-of-year climatology, from the training years only.

    With only a couple of training years a raw per-day mean is far too noisy, so
    it is smoothed with a circular 31-day window — circular because 31 December
    and 1 January are one day apart and a plain rolling mean would pull the
    ends of the year toward nothing.
    """
    doy = tgt.groupby("time.dayofyear").mean("time")
    doy = doy.reindex(dayofyear=np.arange(1, 367)).interpolate_na(
        "dayofyear", fill_value="extrapolate"
    )
    half = CLIM_SMOOTH_DAYS // 2
    wrapped = xr.concat(
        [doy.isel(dayofyear=slice(-half, None)), doy, doy.isel(dayofyear=slice(0, half))],
        dim="dayofyear",
    )
    smoothed = wrapped.rolling(dayofyear=CLIM_SMOOTH_DAYS, center=True).mean()
    out = smoothed.isel(dayofyear=slice(half, half + 366))
    return out.assign_coords(dayofyear=np.arange(1, 367))


def _static_channels(times, shape) -> np.ndarray:
    """(n, 4, lat, lon): day-of-year sin/cos, then normalised lat and lon."""
    n, ny, nx = len(times), *shape
    doy = xr.DataArray(times, dims="time").dt.dayofyear.values.astype("float32")
    ang = 2 * np.pi * doy / 365.25
    # np.ptp(x), not x.ptp() — the ndarray method was removed in NumPy 2.
    lat = (TARGET_LAT - TARGET_LAT.mean()) / (np.ptp(TARGET_LAT) / 2)
    lon = (TARGET_LON - TARGET_LON.mean()) / (np.ptp(TARGET_LON) / 2)

    out = np.empty((n, N_STATIC, ny, nx), dtype="float32")
    out[:, 0] = np.sin(ang)[:, None, None]
    out[:, 1] = np.cos(ang)[:, None, None]
    out[:, 2] = lat[None, :, None]
    out[:, 3] = lon[None, None, :]
    return out


def build(train_years, test_years) -> tuple[Split, Split, dict]:
    """Build train/test splits plus the statistics needed to invert them."""
    ins_tr, tgt_tr = load_years(train_years)
    ins_te, tgt_te = load_years(test_years)

    clim = day_of_year_climatology(tgt_tr)

    # Land mask: ocean if the target is finite at the surface on every training
    # day. Deliberately NOT "finite at every depth" — the target is NaN below
    # the seafloor, so requiring all 15 levels would discard every shelf cell
    # (~26% of the ocean here, including most of the coastal Bay of Bengal, the
    # region that matters most for hazards). Depth-varying bathymetry is handled
    # where it belongs: masked_mse and the metrics both intersect this mask with
    # isfinite(truth), so a shelf cell trains the levels above the seabed and is
    # skipped for the ones below it.
    mask = np.isfinite(tgt_tr.values[:, 0]).all(axis=0)         # (lat, lon)

    # Input normalisation, training years only, land excluded from the stats.
    x_tr_raw = ins_tr.to_array("channel").transpose("time", "channel", ...).values
    x_te_raw = ins_te.to_array("channel").transpose("time", "channel", ...).values
    wet = np.broadcast_to(mask, x_tr_raw.shape[1:])
    mu = np.array([np.nanmean(x_tr_raw[:, c][:, wet[c]]) for c in range(x_tr_raw.shape[1])])
    sd = np.array([np.nanstd(x_tr_raw[:, c][:, wet[c]]) for c in range(x_tr_raw.shape[1])])
    sd[sd == 0] = 1.0

    def prep_x(raw, times):
        z = (raw - mu[None, :, None, None]) / sd[None, :, None, None]
        # A NaN input cell (cloud gap, coastal mask difference between products)
        # becomes 0, i.e. the channel mean. ponytail: no per-channel validity
        # mask channel yet — add one if gap-heavy days turn out to hurt.
        z = np.nan_to_num(z, nan=0.0).astype("float32")
        # Blank everything outside the ocean mask. The input products do not
        # share the target's land mask: OSTIA carries inland lakes, and this
        # box reaches the Tibetan Plateau, so SST over "land" can read 0.7 degC
        # — physically impossible seawater. Those cells are already outside the
        # loss and the normalisation statistics, but convolutions have a
        # receptive field, so junk on land propagates into coastal ocean
        # predictions. Zeroing gives land a single consistent value, which also
        # acts as an implicit land channel.
        z[:, :, ~mask] = 0.0
        return np.concatenate([z, _static_channels(times, z.shape[-2:])], axis=1)

    # Target: anomaly from the training climatology, normalised per depth so the
    # near-constant deep levels do not dominate the loss and no hand-tuned depth
    # weights are needed. Metrics are reported back in degrees C.
    def clim_for(tgt):
        doy = tgt["time"].dt.dayofyear
        return clim.sel(dayofyear=doy).transpose("time", "depth", ...).values

    c_tr, c_te = clim_for(tgt_tr), clim_for(tgt_te)
    a_tr = tgt_tr.values - c_tr
    a_te = tgt_te.values - c_te
    y_sd = np.nanstd(a_tr, axis=(0, 2, 3))
    y_sd[y_sd == 0] = 1.0

    stats = dict(mu=mu, sd=sd, y_sd=y_sd, mask=mask, clim=clim,
                 depths=STD_DEPTHS, channels=list(ins_tr.data_vars))

    train = Split(prep_x(x_tr_raw, ins_tr.time.values),
                  (a_tr / y_sd[None, :, None, None]).astype("float32"),
                  c_tr.astype("float32"), ins_tr.time.values)
    test = Split(prep_x(x_te_raw, ins_te.time.values),
                 (a_te / y_sd[None, :, None, None]).astype("float32"),
                 c_te.astype("float32"), ins_te.time.values)
    return train, test, stats


def to_celsius(y_norm: np.ndarray, y_clim: np.ndarray, y_sd: np.ndarray) -> np.ndarray:
    """Undo the anomaly normalisation: back to absolute temperature."""
    return y_norm * y_sd[None, :, None, None] + y_clim
