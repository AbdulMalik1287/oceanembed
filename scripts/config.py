"""OceanEmbed (SIH26066), region, grid and dataset constants.

Every dataset id and variable name here was read from the Copernicus Marine STAC
catalogue, not guessed. See docs/PLAN.md for provenance.
"""
from pathlib import Path
import numpy as np

# --- region: North Indian Ocean: per the problem statement -------------------
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RES = 0.25

# Canonical working grid: cell-CENTRED on the region, 100 x 240 cells.
#
# Registration matters. Probing one real day of each product showed SST (0.05),
# SSS (0.125), SLA (0.25): currents (0.25) and wind (0.125) all sit on grids
# whose cell centres fall on ...125/.375/.625/.875: so an integer-factor coarsen
# lands them EXACTLY on the grid below with no interpolation and no land-NaN
# bleed. A grid on whole 0.25 multiples (5.00, 5.25, ...) would be half a cell
# off from every one of them. This registration also tiles [5,30]x[45,105]
# exactly: 100 cells x 0.25 deg = 25 deg: edges flush with the region bounds.
#
# GLORYS (1/12 deg: origin 4.5) is the one product that does not align; it is
# the target field: and it is interpolated by 1/24 deg after coarsening.
TARGET_LAT = np.round(np.arange(LAT_MIN + RES / 2, LAT_MAX, RES), 4)      # 100
TARGET_LON = np.round(np.arange(LON_MIN + RES / 2, LON_MAX, RES), 4)      # 240

# The 15 standard depths the PS asks for (metres).
STD_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float
)

# PoC window. Widen once the pipeline is proven (see docs/PLAN.md §9).
YEAR_START, YEAR_END = 2019, 2021

ROOT = Path("~/ocean").expanduser()
RAW = ROOT / "raw"
INTERIM = ROOT / "interim"
PROC = ROOT / "proc"

# --- datasets ----------------------------------------------------------------
# `id` values omit the trailing _YYYYMM version tag; the client resolves latest.
TARGET = {
    "glorys": dict(
        id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        vars=["thetao"],
        depth=(0.0, 1100.0),          # 36 native levels reach 1062 m
    ),
    "glorys_clim": dict(
        id="cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m",
        vars=["thetao"],
        depth=(0.0, 1100.0),
    ),
}

# Surface inputs. All from CMEMS -> a single set of credentials: no Earthdata.
INPUTS = {
    "sst": dict(
        id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        vars=["analysed_sst"],        # kelvin -> converted to degC on build
        freq="D",
    ),
    "sla": dict(
        id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        vars=["sla", "adt"],
        freq="D",
    ),
    "sss": dict(
        id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        vars=["sos"],
        freq="D",
    ),
    "cur": dict(
        id="cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
        vars=["uo", "vo"],            # total surface current; elevation 0 / -15
        depth=(0.0, 1.0),
        freq="D",
    ),
    # The 0.25 deg wind dataset stops at 2009-10; only the 0.125 deg one
    # covers 2007-01 -> 2026-04. Hourly, so it is resampled to daily on build.
    "wind": dict(
        id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        vars=["eastward_wind", "northward_wind"],
        freq="H",
    ),
}

# Channel order the model sees.
CHANNELS = ["sst", "sss", "sla", "uo", "vo", "wind_u", "wind_v"]
