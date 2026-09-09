#!/usr/bin/env python
"""Operational diagnostics derived from a reconstructed temperature profile.

These are what a hazard-warning centre actually consumes. None of them can be
computed from SST alone, they all need the vertical structure, which is the
argument for reconstructing it at all.

    D26   depth of the 26 degC isotherm (m)
    TCHP  tropical cyclone heat potential (kJ/cm2), heat stored above 26 degC.
          The operational predictor of cyclone rapid intensification in the Bay
          of Bengal: a deep warm layer keeps feeding a storm even after its own
          winds have mixed the surface, where a shallow one cools and starves it.
    OHC   ocean heat content over a depth range (kJ/cm2)
    MLD   mixed layer depth (m), 0.2 degC criterion referenced to 10 m

All take an array whose LAST axis is depth, plus the depth levels in metres.
"""
from __future__ import annotations

import numpy as np

RHO = 1026.0      # kg m-3, representative seawater density
CP = 3985.0       # J kg-1 K-1, specific heat capacity
J_PER_M2_TO_KJ_PER_CM2 = 1e-7


def _crossing_depth(t: np.ndarray, depths: np.ndarray, value: float) -> np.ndarray:
    """Depth at which the profile first drops below `value`, linearly interpolated.

    NaN where the surface is already colder than `value` (no warm layer at all)
    or where the profile never crosses it within the given levels.
    """
    t = np.asarray(t, dtype="float64")
    warm = t >= value
    # Number of leading levels above the threshold. 0 means the surface is
    # already below it; nz means it never crosses within our depth range.
    n = np.argmin(np.concatenate([warm, np.zeros(warm.shape[:-1] + (1,), bool)],
                                 axis=-1), axis=-1)

    nz = len(depths)
    lo = np.clip(n - 1, 0, nz - 1)
    hi = np.clip(n, 0, nz - 1)
    t_lo = np.take_along_axis(t, lo[..., None], -1)[..., 0]
    t_hi = np.take_along_axis(t, hi[..., None], -1)[..., 0]
    z_lo, z_hi = depths[lo], depths[hi]

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (t_lo - value) / (t_lo - t_hi)
        out = z_lo + frac * (z_hi - z_lo)
        out = np.where((n == 0) | (n >= nz), np.nan, out)
    return np.where(np.isfinite(t[..., 0]), out, np.nan)


def d26(t: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Depth of the 26 degC isotherm, metres."""
    return _crossing_depth(t, depths, 26.0)


def mld(t: np.ndarray, depths: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """Mixed layer depth, metres. Temperature criterion referenced to 10 m."""
    ref_i = int(np.argmin(np.abs(depths - 10.0)))
    ref = np.take(t, ref_i, axis=-1)
    # Reuse the crossing search per-profile threshold by shifting the profile.
    shifted = t - (ref[..., None] - delta) + 26.0
    return _crossing_depth(shifted, depths, 26.0)


def tchp(t: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Tropical cyclone heat potential, kJ/cm2.

    Integral of rho*cp*(T - 26) over the depth range where T exceeds 26 degC.

    ponytail: trapezoidal integration on the 15 standard levels, with the
    excess clipped at zero rather than integrating exactly to the interpolated
    D26. Level spacing is 5-25 m through the warm layer so the truncation is
    small; switch to an exact partial-layer term if the absolute value (rather
    than its spatial pattern) starts being quoted operationally.
    """
    excess = np.clip(np.asarray(t, dtype="float64") - 26.0, 0.0, None)
    integral = np.trapezoid(excess, x=depths, axis=-1)     # K m
    out = RHO * CP * integral * J_PER_M2_TO_KJ_PER_CM2
    return np.where(np.isfinite(np.asarray(t)[..., 0]), out, np.nan)


def ohc(t: np.ndarray, depths: np.ndarray, zmax: float = 700.0) -> np.ndarray:
    """Ocean heat content above `zmax`, kJ/cm2, relative to 0 degC."""
    keep = depths <= zmax
    integral = np.trapezoid(np.asarray(t, dtype="float64")[..., keep],
                            x=depths[keep], axis=-1)
    out = RHO * CP * integral * J_PER_M2_TO_KJ_PER_CM2
    return np.where(np.isfinite(np.asarray(t)[..., 0]), out, np.nan)


# --------------------------------------------------------------------------- #
def _self_check() -> None:
    """Analytic profiles with known answers. Run: python -m scripts.diagnostics"""
    depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500,
                       700, 1000], dtype=float)

    # Profile A: 30 degC at the surface falling 0.04 degC/m -> 26 degC at 100 m
    a = 30.0 - 0.04 * depths
    got = d26(a, depths)
    assert abs(got - 100.0) < 1e-6, f"D26 {got} != 100"

    # TCHP for that profile: integral of (4 - 0.04z) from 0 to 100 = 200 K m
    expected = RHO * CP * 200.0 * J_PER_M2_TO_KJ_PER_CM2
    got = tchp(a, depths)
    assert abs(got - expected) / expected < 0.02, f"TCHP {got} vs {expected}"
    assert 70 < got < 90, f"TCHP {got} kJ/cm2 outside a plausible tropical range"

    # Profile B: isothermal 20 degC, never reaches 26, so no warm layer
    b = np.full_like(depths, 20.0)
    assert np.isnan(d26(b, depths)), "D26 should be NaN with no warm layer"
    assert tchp(b, depths) == 0.0, "TCHP should be zero with no water above 26"

    # Profile C: 50 m mixed layer at 29 degC, then a sharp drop
    c = np.where(depths <= 50, 29.0, 29.0 - 0.5 * (depths - 50))
    got = mld(c, depths)
    assert 50 <= got <= 60, f"MLD {got} not at the base of a 50 m mixed layer"

    # Vectorised over a grid, with land as NaN
    grid = np.stack([a, b, c, np.full_like(depths, np.nan)]).reshape(2, 2, -1)
    for f in (d26, tchp, ohc, mld):
        out = f(grid, depths)
        assert out.shape == (2, 2), f"{f.__name__} shape {out.shape}"
        assert np.isnan(out[1, 1]), f"{f.__name__} did not propagate land NaN"

    print("diagnostics self-check passed")
    print(f"  D26  (30 degC surface, -0.04 degC/m) = {d26(a, depths):.1f} m")
    print(f"  TCHP same profile                    = {tchp(a, depths):.1f} kJ/cm2")
    print(f"  MLD  (50 m mixed layer)              = {mld(c, depths):.1f} m")
    print(f"  OHC  0-700 m, profile A              = {ohc(a, depths):.0f} kJ/cm2")


if __name__ == "__main__":
    _self_check()
