# OceanEmbed — engineering plan

Status: pre-build. Written 2026-09-05.

## 1. Problem shape (why this is tractable)

Region 5–30°N, 45–105°E at 0.25° → **100 lat × 240 lon** (cell-centred, 5.125–29.875 / 45.125–104.875). Daily, 1993-01-01 → 2026-06-23 → **~12,200 days**.
(GLORYS12V1 multiyear now runs to mid-2026 — confirmed from the CMEMS STAC, not 2021 as first assumed.)

- Input tensor per day: 7 channels × 101 × 241
- Target per day: 15 depths × 101 × 241
- Whole dataset, float32: `12200 × 22 × 100 × 240 × 4 B ≈ 26 GB`

That fits **entirely in one 96 GB GPU's memory**. No sharding, no streaming dataloader, no DDP.
Preprocess once into a single memory-mapped `.npy`/`zarr`, load to GPU, train.

### Grid registration (decided from real files, not assumed)

Probing one day of every product showed SST (0.05°), SSS (0.125°), SLA (0.25°), currents (0.25°)
and wind (0.125°) all sit on grids whose cell centres fall on `…125 / .375 / .625 / .875`. So the
working grid is **cell-centred** — 5.125…29.875 by 45.125…104.875 — and every surface input reaches
it by an exact integer-factor area-mean coarsen with **no interpolation at all**. A grid on whole
0.25° multiples would have been half a cell off from all five.

This is not cosmetic. Interpolating across the offset averages each coastal cell with its land
neighbour, and measured on real DUACS data it turned 48.4% of the domain NaN into 50.5% — about
515 ocean cells per day thrown away around the basin rim. It also tiles the region exactly:
100 × 0.25° = 25°, edges flush with 5°N and 30°N.

GLORYS (1/12°, origin 4.5) is the one product that does not align, and it is interpolated by 1/24°
after coarsening. It is the target field, so the cost is losing coastal pixels from training and
scoring rather than corrupting them.

> Consequence: the bottleneck is **data acquisition and regridding**, not compute. Downloads are the
> long pole (days), training runs will be minutes-to-hours.

## 2. Data acquisition

### Accounts needed (blocking)
- **Copernicus Marine (CMEMS)** — free account, for OSTIA SST, DUACS SLA, SSS, GLORYS.
  Tool: `copernicusmarine` python client (subset by bbox/time/depth server-side).
- **NASA Earthdata Login** — for PODAAC OSCAR currents and CCMP winds.

### Products — authoritative ids, resolved from the DOIs in the PS

Every id below came from the CMEMS STAC catalogue via the PS's own DOIs, and every variable name
was read from the dataset metadata. None are guessed.

| Var | Dataset id | Native | Coverage |
|---|---|---|---|
| SST | `METOFFICE-GLO-SST-L4-REP-OBS-SST` (`analysed_sst`, **kelvin**) | 0.05deg daily | 1981-10 -> 2026-03 |
| SLA | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` (`sla`, `adt`) | 0.25deg daily | 1993-01 -> 2026-01 |
| SSS | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` (`sos`) | 0.25deg daily | 1993-01 -> **2024-12** |
| Currents | `cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m` (`uo`, `vo`) | 0.25deg daily | 1993-01 -> 2026-03 |
| Winds | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` (`eastward_wind`, `northward_wind`) | 0.125deg **hourly** | **2007-01** -> 2026-04 |
| Target | `cmems_mod_glo_phy_my_0.083deg_P1D-m` (`thetao`) | 1/12deg daily, 50 lev | 1993-01 -> 2026-06 |

**Everything comes from Copernicus Marine.** The currents and wind products replace the PODAAC
OSCAR/CCMP suggestions in the PS, which means **one set of credentials and no NASA Earthdata
account** - and it avoids CCMP's global-file distribution, which has no bbox subsetting.

### Two corrections to earlier assumptions

1. **SSS is not the binding constraint.** The product the PS cites (`moi-00051`) is a *multi-
   observation* reconstruction running from **1993**, not a raw SMAP/SMOS retrieval starting 2015.
   All seven channels are available far earlier than first assumed.
   Caveat worth stating in the submission: before 2010 there was no satellite salinity sensor at
   all, so early SSS is a reconstruction that likely ingests in-situ data - a mild leakage path when
   the target is also in-situ-informed. The PS names this product explicitly, so it is sanctioned,
   but say so rather than let a reviewer find it.

2. **Wind is the binding constraint instead.** The 0.25deg wind dataset stops at **2009-10**; only
   the 0.125deg one reaches the present, and it starts **2007-01**. So the continuous all-7-channel
   record is **2007 -> 2024** (~6,500 days), bounded below by wind and above by SSS. Splicing the
   two wind products to reach 1993 means a mid-record resolution and product change - not worth it.

Wind is also hourly, so it is resampled to daily means on the native grid before regridding.
For 3 years that is ~20 GB unpacked (~10 GB packed) - the largest surface input by far. Stage it
monthly and average as it lands.

GLORYS download size — **corrected**. The CMEMS toolbox subsets depth by *range*, not by level list,
and GLORYS has **36 native levels between 0 and 1062 m** (the 15 standard depths must be interpolated
from those). Region at 1/12° is 301 × 721 cells:

```
301 × 721 × 36 levels × 4 B  =  31 MB / day  =  11.4 GB / year  (unpacked float32)
full 1993–2026  ≈ 382 GB unpacked
```

GLORYS netCDF is int16-packed with scale_factor, so **actual transfer is roughly half** the above
(~5.7 GB/yr). Pull yearly, interpolate to the 15 standard depths + regrid to 0.25° on arrival,
delete the 1/12° source. Post-regrid target store: ~18 GB for the full record.

### CMEMS subdataset IDs (product `GLOBAL_MULTIYEAR_PHY_001_030`)

| Subdataset | Use | Size (region) |
|---|---|---|
| `cmems_mod_glo_phy_my_0.083deg_P1D-m_202311` | **training target** (`thetao`), daily, 50 lev | 11.4 GB/yr |
| `cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m_202311` | **anomaly reference + climatology baseline**, 12 month-of-year fields | ~375 MB |
| `..._static_202311--ext--bathy` | bathymetry (static model input) | MB |
| `..._static_202311--ext--coords` | land/sea mask, cell coords | MB |

Useful variables in the daily set beyond `thetao`: `mlotst` (mixed layer depth) and `zos` come
free — use `mlotst` directly as an evaluation diagnostic instead of deriving MLD ourselves.

**The climatology subdataset is a real shortcut.** Computing a day-of-year climatology ourselves
would require downloading the entire 33-year daily record first. CMEMS ships it precomputed at
~375 MB — interpolate the 12 monthly fields to daily and use that as the anomaly reference.

**Fallback / secondary target:** ARMOR3D (CMEMS, 0.25°, weekly, 1993–present,
https://doi.org/10.48670/moi-00052). Cheap to fetch, useful as a baseline to beat.

## 3. The scientific honesty problem (address this explicitly, judges are INCOIS)

GLORYS is a **reanalysis that already assimilates** SST, altimetry, and ARGO. A model trained to
predict GLORYS from surface fields is learning to **emulate a data-assimilation system**, not to
discover subsurface structure from scratch. Two consequences:

1. Validation against **gridded** ARGO is only semi-independent (GLORYS ate those profiles).
2. The stronger validation is against **raw ARGO profiles** (Argo GDAC / Indian Argo Project),
   matched nearest-day / nearest-grid-cell, and ideally restricted to test years.

Do both. State the distinction in the submission — it is the difference between a project that
looks like a class assignment and one that looks like ocean science.

## 4. Split strategy (the #1 way this goes wrong)

**Never** random-day split. Consecutive days are near-identical; a random split leaks the answer
and produces a fake 0.99 correlation.

**3-year PoC (2019–2021):**
- Train 2019–2020, test 2021. One year of test is thin, but it is enough to tell whether the model
  beats climatology, which is the only question the PoC has to answer.

**Full record, once the pipeline is proven.** The all-7-channel window is 2007–2024 (wind starts
2007-01, SSS ends 2024-12), so:
- Train: 2007–2018
- Val: 2019–2020
- Test: 2021–2024 (held out, touched once)

Also report a **spatial** holdout (e.g., train excluding one 5°×5° box) to show the model is not
memorizing geography.

## 5. Model

Predict **anomaly from a day-of-year climatology**, not absolute temperature. The climatology alone
explains most of the variance; predicting it back is not skill. Baseline table:

| Baseline | Purpose |
|---|---|
| Monthly climatology (WOA18 / GLORYS clim) | floor — anything below this is a failed project |
| Per-depth linear regression on surface vars | classical |
| XGBoost per depth | what most published papers actually use |
| **OceanEmbed (ours)** | must beat all three |

Architecture — the embedding requirement is naturally satisfied by a bottleneck:

```
7×101×241 surface maps
  + static: land mask, bathymetry, lat/lon sin-cos, day-of-year sin-cos
        ↓
  U-Net / ConvNeXt encoder  →  latent embedding (the "satellite embedding engine")
        ↓
  decoder → 15×101×241 temperature anomaly
        ↓
  + climatology → absolute T
```

- Loss: land-masked MSE, weighted per depth (thermocline depths matter most, and 1000 m is nearly
  constant — unweighted loss lets the deep levels dominate and the model learns nothing useful).
- Add a small ViT/attention variant as an ablation — the PS explicitly lists ViT/GNN/attention,
  and having a comparison table is worth more than picking one blind.

Skip until measured as needed: GNN (no natural graph here beyond the grid), diffusion, foundation
models, multi-GPU training.

## 6. Evaluation

Per depth level, on the test years:
- RMSE, bias, correlation (the PS asks for exactly these)
- vs raw ARGO profiles, separately
- Derived diagnostics that oceanographers actually care about:
  **mixed layer depth**, **D20 (20 °C isotherm depth)**, **ocean heat content 0–700 m**
- A marine-heatwave case study (e.g., 2019 Arabian Sea) — this is the demo that sells it

## 7. Hardware

`blackwell` — 24 cores, 377 GB RAM, and **MIG is enabled**. `nvidia-smi --query-gpu` advertises
97,887 MiB per card, but each physical GPU is split into four `1g.24gb` instances: torch sees
**2 devices of 23.6 GiB with 46 SMs each**, about a quarter of a card apiece. Budget against 24 GB.

It is still ample. The 3-year tensors are ~1.2 GB of inputs and ~1.6 GB of target, and the U-Net at
width 32 is a few hundred thousand parameters. Use:
- device 0: main model training
- device 1: parallel ablations / hyperparameter sweeps

Skipped: DDP, FSDP, gradient checkpointing. There is no NVLink peer access between MIG slices
anyway, so multi-GPU data parallelism would be a poor trade even if the model needed it.

torch 2.11.0+cu128 verified working on sm_120 with bf16.

## 8. Order of work

1. Register CMEMS + Earthdata accounts *(blocking, do first)*
2. Download DUACS SLA + OSTIA SST for one test year → build and validate the regrid pipeline on a
   small slice before pulling 138 GB
3. Full download in background (days) while the model code is written against synthetic tensors
4. Climatology + baselines (this alone produces the first real numbers)
5. U-Net anomaly model, temporal split, per-depth metrics
6. ARGO validation harness
7. PoC demo: BoB / Arabian Sea maps, profile plots, MHW case study

Idea submission (deadline 20 Sep 2026) only needs steps 1–4's framing — write the abstract from
this plan; do not wait for the full build.

## 9. Disk budget

Grid cell counts for the region (5–30°N, 45–105°E), inclusive of both edges:

| Grid | lat × lon | cells |
|---|---|---|
| 1/12° (GLORYS native, padded) | 313 × 733 | 229,429 |
| 0.25° (working grid) | 100 × 240 | 24,000 |
| 0.05° (OSTIA native, padded) | 520 × 1220 | 634,400 |

Per-day sizes, float32:

- GLORYS raw, 36 levels: `229,429 × 36 × 4 B` = 33.0 MB/day unpacked; **measured 15.8 MB/day as delivered** → 5.8 GB/year
- Processed target, 15 depths at 0.25°: `24,000 × 15 × 4 B` = **1.44 MB/day** → 0.53 GB/year
- Processed inputs, 7 channels at 0.25°: `24,000 × 7 × 4 B` = **0.67 MB/day** → 0.25 GB/year

CMEMS ships int16-packed netCDF with `scale_factor`. The 2× packing assumption is now **measured**:
one GLORYS day is 33.0 MB unpacked and 15.8 MB as delivered, a ratio of 2.09.

### What actually has to sit on disk

The key distinction: **382 GB is cumulative transfer, not required disk.** If each year's 1/12°
source is deleted after regridding, the retained footprint is far smaller.

| Item | 3-year PoC | Full record 1993–2026 |
|---|---|---|
| GLORYS staging (1 yr at a time, packed) | 6 GB | 6 GB |
| Processed target (0.25°, 15 depths) | 1.6 GB | 17.8 GB |
| Processed inputs (0.25°, 7 ch) | 0.8 GB | 8.3 GB |
| Raw surface products | 3 GB | ~20 GB |
| Climatology + statics | 0.3 GB | 0.3 GB |
| Checkpoints, logs, predictions | 5 GB | 23 GB |
| **Peak retained** | **~17 GB** | **~60 GB** |
| Cumulative download | ~20 GB | ~190 GB packed |

**Both fit in the 154 GB free on `blackwell`.** The full record is not ruled out — it costs an
overnight download and ~60 GB retained, provided the staged delete-as-you-go loop is honoured. What
is ruled out is keeping the raw 1/12° GLORYS around.

Still start with 3 years: it costs ~20 GB and one evening, and it tells you whether the approach
beats climatology before committing to the long pull.
