#!/usr/bin/env python
"""Figures for the SIH submission. Writes PNGs to ~/ocean/figs/.

    python -m scripts.figures

Coastlines are drawn from the data's own land mask rather than a mapping
library, so there is no cartopy/proj dependency to install or break.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .config import PROC, STD_DEPTHS, TARGET_LAT, TARGET_LON

FIGS = PROC.parent / "figs"
RUNS = PROC.parent / "runs"
PRED = PROC.parent / "pred"
YEAR = 2021
EXTENT = [TARGET_LON[0], TARGET_LON[-1], TARGET_LAT[0], TARGET_LAT[-1]]

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})
C = {"climatology": "#9aa0a6", "linear": "#e8a33d", "unet": "#1f6feb"}


def _coast(ax, land):
    """Outline the land mask so maps read as a basin, not a rectangle."""
    ax.contour(TARGET_LON, TARGET_LAT, land.astype(float), levels=[0.5],
               colors="k", linewidths=0.6)
    ax.set_facecolor("#f2f2f2")


def _mapfmt(ax, title):
    ax.set_title(title)
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_aspect("equal")


# --------------------------------------------------------------------------- #
def fig_skill_by_depth():
    runs = {m: json.load(open(RUNS / f"{m}.json")) for m in C}
    rows = {m: [r for r in d["rows"] if r.get("n")] for m, d in runs.items()}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    for m, rs in rows.items():
        z = [r["depth"] for r in rs]
        a1.plot([r["rmse"] for r in rs], z, "o-", color=C[m], label=m, lw=2, ms=4)
        if m != "climatology":
            a2.plot([r["acc"] for r in rs], z, "o-", color=C[m], label=m, lw=2, ms=4)

    a1.set_xlabel("RMSE (°C)   — lower is better")
    a1.set_ylabel("depth (m)")
    a1.set_title("Reconstruction error")
    a1.invert_yaxis()
    a1.set_yscale("symlog", linthresh=100)
    a1.set_yticks([0, 25, 50, 100, 200, 500, 1000])
    a1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.legend(frameon=False)
    a1.grid(alpha=0.25)

    a2.set_xlabel("anomaly correlation (ACC)   — higher is better")
    a2.set_title("Skill beyond the seasonal cycle")
    a2.axvline(0, color="k", lw=0.8)
    a2.grid(alpha=0.25)
    a2.legend(frameon=False)

    for ax in (a1, a2):
        ax.axhspan(75, 150, color="#1f6feb", alpha=0.07, zorder=0)
    a1.text(a1.get_xlim()[1] * 0.97, 110, "thermocline", ha="right", va="center",
            fontsize=9, color="#1f6feb", style="italic")

    # Title the actual test years rather than hardcoding one, so the figure can
    # never quietly disagree with the run it was made from.
    n_days = max(r["n"] for rs in rows.values() for r in rs) // (100 * 240) or 0
    span = "2020–2021" if n_days > 500 else "2021"
    fig.suptitle(f"Subsurface temperature skill by depth — test {span}, unseen",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "01_skill_by_depth.png")
    plt.close(fig)
    print("  01_skill_by_depth.png")


def fig_acc_map(pred, truth, clim, land):
    """Where the model has skill, at the depth where it matters most."""
    zi = int(np.argmin(np.abs(STD_DEPTHS - 100)))
    p, t, c = pred[:, zi], truth[:, zi], (truth - (truth - truth))[:, zi] * 0
    pa = p - clim[:, zi]
    ta = t - clim[:, zi]
    num = np.nansum(pa * ta, axis=0)
    den = np.sqrt(np.nansum(pa**2, axis=0) * np.nansum(ta**2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(den > 0, num / den, np.nan)
    acc = np.where(land, np.nan, acc)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    im = ax.imshow(acc, origin="lower", extent=EXTENT, vmin=0, vmax=1,
                   cmap="viridis", interpolation="nearest")
    _coast(ax, land)
    _mapfmt(ax, "Anomaly correlation at 100 m — 2021")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("ACC")
    ax.text(0.99, 0.03, f"basin mean {np.nanmean(acc):.2f}", transform=ax.transAxes,
            ha="right", color="white", fontsize=9, fontweight="bold")
    fig.savefig(FIGS / "02_acc_map_100m.png")
    plt.close(fig)
    print("  02_acc_map_100m.png")


def fig_profiles(pred, truth, times, land):
    """Predicted vs GLORYS profiles at two contrasting sites."""
    sites = [(15.125, 65.125, "Arabian Sea (15°N, 65°E)"),
             (15.125, 88.125, "Bay of Bengal (15°N, 88°E)")]
    dates = [0, 120, 240, 350]

    fig, axes = plt.subplots(1, len(sites), figsize=(9, 5), sharey=True)
    for ax, (la, lo, name) in zip(axes, sites):
        i = int(np.argmin(np.abs(TARGET_LAT - la)))
        j = int(np.argmin(np.abs(TARGET_LON - lo)))
        for k, d in enumerate(dates):
            lab_t = "GLORYS (truth)" if k == 0 else None
            lab_p = "OceanEmbed" if k == 0 else None
            ax.plot(truth[d, :, i, j], STD_DEPTHS, "-", color="0.35", lw=1.8, label=lab_t)
            ax.plot(pred[d, :, i, j], STD_DEPTHS, "--", color=C["unet"], lw=1.8, label=lab_p)
        ax.set_title(name)
        ax.set_xlabel("temperature (°C)")
        ax.grid(alpha=0.25)
        ax.set_ylim(1000, 0)
    axes[0].set_ylabel("depth (m)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Reconstructed profiles, four dates across 2021", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "03_profiles.png")
    plt.close(fig)
    print("  03_profiles.png")


def fig_tchp(diag, land):
    """The operational payoff: cyclone heat potential, which SST cannot give."""
    d = 240  # late August, peak Bay of Bengal cyclone season
    p = np.where(land, np.nan, diag["tchp_pred"].values[d])
    t = np.where(land, np.nan, diag["tchp_true"].values[d])
    date = str(diag.time.values[d])[:10]

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.35, wspace=0.25)

    vmax = float(np.nanpercentile(t, 99))
    for k, (arr, name) in enumerate([(t, "GLORYS (truth)"), (p, "OceanEmbed reconstruction")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(arr, origin="lower", extent=EXTENT, vmin=0, vmax=vmax,
                       cmap="inferno", interpolation="nearest")
        _coast(ax, land)
        _mapfmt(ax, name)
        # Both panels get a colorbar so they end up the same size; only the
        # right one is labelled. A colorbar on one panel alone shrinks it and
        # the pair reads as mismatched on a slide.
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("TCHP (kJ cm$^{-2}$)" if k == 1 else "")
        if k == 0:
            cb.ax.set_yticklabels([])

    ax = fig.add_subplot(gs[1, 0])
    ok = np.isfinite(p) & np.isfinite(t)
    ax.hexbin(t[ok], p[ok], gridsize=45, cmap="Blues", bins="log", mincnt=1)
    lim = [0, vmax]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("GLORYS TCHP (kJ cm$^{-2}$)")
    ax.set_ylabel("reconstructed")
    ax.set_title(f"Agreement, {date}")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 1])
    m = ~land
    ax.plot(diag.time, np.nanmean(np.where(m, diag["tchp_true"].values, np.nan), axis=(1, 2)),
            color="0.35", lw=1.8, label="GLORYS")
    ax.plot(diag.time, np.nanmean(np.where(m, diag["tchp_pred"].values, np.nan), axis=(1, 2)),
            color=C["unet"], lw=1.5, ls="--", label="OceanEmbed")
    ax.set_ylabel("basin-mean TCHP (kJ cm$^{-2}$)")
    ax.set_title("Seasonal cycle, 2021")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)

    # Compute the correlation here rather than hardcoding it, so the headline on
    # the figure can never drift from the run that produced it.
    pa = np.where(land, np.nan, diag["tchp_pred"].values).ravel()
    ta = np.where(land, np.nan, diag["tchp_true"].values).ravel()
    ok = np.isfinite(pa) & np.isfinite(ta)
    r = float(np.corrcoef(pa[ok], ta[ok])[0, 1])
    fig.suptitle("Tropical Cyclone Heat Potential — reconstructed from surface data alone "
                 f"(corr {r:.2f})", fontweight="bold")
    fig.savefig(FIGS / "04_tchp.png")
    plt.close(fig)
    print("  04_tchp.png")


def fig_argo():
    """Validation against profiles nothing in this pipeline ever saw."""
    m = PROC.parent / "argo" / "matched.npz"
    if not m.exists():
        print("  (no Argo matches yet, skipping 05)")
        return
    d = np.load(m, allow_pickle=True)
    obs, model, glorys, clim = d["obs"], d["model"], d["glorys"], d["clim"]

    rmse, bias, n = {}, {}, []
    for name, arr in (("clim", clim), ("glorys", glorys), ("model", model)):
        rmse[name], bias[name] = [], []
    for k in range(len(STD_DEPTHS)):
        ok = (np.isfinite(obs[:, k]) & np.isfinite(model[:, k])
              & np.isfinite(glorys[:, k]) & np.isfinite(clim[:, k]))
        n.append(ok.sum())
        for name, arr in (("clim", clim), ("glorys", glorys), ("model", model)):
            if ok.sum() < 20:
                rmse[name].append(np.nan); bias[name].append(np.nan)
            else:
                rmse[name].append(np.sqrt(np.mean((arr[ok, k] - obs[ok, k]) ** 2)))
                bias[name].append(np.mean(arr[ok, k] - obs[ok, k]))

    style = {"clim": ("climatology", C["climatology"]),
             "glorys": ("GLORYS (saw these profiles)", "#2ea043"),
             "model": ("OceanEmbed", C["unet"])}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 5.4), sharey=True)
    for key, (label, col) in style.items():
        a1.plot(rmse[key], STD_DEPTHS, "o-", color=col, label=label, lw=2, ms=4)
        a2.plot(bias[key], STD_DEPTHS, "o-", color=col, label=label, lw=2, ms=4)

    a1.set_xlabel("RMSE vs Argo (°C)")
    a1.set_ylabel("depth (m)")
    a1.set_title("Error against independent profiles")
    a1.invert_yaxis()
    a1.set_yscale("symlog", linthresh=100)
    a1.set_yticks([0, 25, 50, 100, 200, 500, 1000])
    a1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.legend(frameon=False, fontsize=9)
    a1.grid(alpha=0.25)

    a2.axvline(0, color="k", lw=0.8)
    a2.set_xlabel("bias vs Argo (°C)   — positive is too warm")
    a2.set_title("Inherited warm bias at the thermocline")
    a2.grid(alpha=0.25)

    for ax in (a1, a2):
        ax.axhspan(75, 150, color="#1f6feb", alpha=0.07, zorder=0)

    fig.suptitle(f"Independent validation — {len(obs):,} Argo profiles, 2020–2021",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "05_argo_validation.png")
    plt.close(fig)
    print("  05_argo_validation.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    pred = xr.open_dataarray(PRED / f"pred_{YEAR}.nc")
    truth = xr.open_dataarray(PROC / f"target_{YEAR}.nc")
    truth = truth.sel(time=pred.time)
    diag = xr.open_dataset(PRED / f"diag_{YEAR}.nc")

    p, t = pred.values, truth.values
    land = ~np.isfinite(t[0, 0])

    # Climatology used by the model: truth minus its own anomaly is not stored,
    # so rebuild the day-of-year mean of the TRAINING years the same way.
    from .data import day_of_year_climatology, load_years
    _, tgt_tr = load_years([2019, 2020])
    clim_doy = day_of_year_climatology(tgt_tr)
    clim = clim_doy.sel(dayofyear=truth["time"].dt.dayofyear).transpose(
        "time", "depth", "latitude", "longitude").values

    fig_skill_by_depth()
    fig_acc_map(p, t, clim, land)
    fig_profiles(p, t, pred.time.values, land)
    fig_tchp(diag, land)
    fig_argo()
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
