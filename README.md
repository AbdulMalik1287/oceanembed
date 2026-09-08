# OceanEmbed — SIH26066

Reconstruct daily depth-wise subsurface ocean temperature over the North Indian Ocean
(5–30°N, 45–105°E) at 0.25°, from surface satellite observations only.

- **[`docs/PPT_HANDOFF.md`](docs/PPT_HANDOFF.md)** — asset pack, architecture diagrams and slide copy for whoever builds the deck
- **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — how to run, check and explain this without help
- **[`docs/RESULTS.md`](docs/RESULTS.md)** — measured skill, by depth, with the failure analysis
- **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** — how it is implemented, stage by stage
- [`docs/PS.md`](docs/PS.md) — problem statement and dataset table
- [`docs/PLAN.md`](docs/PLAN.md) — engineering plan, budgets, and the decisions that matter
- `docs/INCOIS_PS_official.pdf` — official INCOIS problem statement

## Status

| | |
|---|---|
| Pipeline | complete, 15 self-checks passing |
| Data | **8 years (2014–2021)** built and verified; staged so peak disk stays under 10 GB |
| Split | train 2014–2019 · test **2020 and 2021**, both held out |
| Baselines + U-Net | U-Net mean RMSE **0.783 °C** vs **0.889 °C** climatology (+11.9%) |
| Peak skill | **+20.2% at 100 m**, ACC 0.608 — the thermocline |
| Diagnostics | TCHP corr **0.897**, D26 0.830, MLD 0.741 — none computable from SST alone |
| Independent check | **8,015 Argo profiles**: 0.852 °C vs 0.900 climatology (+5.3%; +8–11% at 75–200 m) |
| Coverage | Argo gives **11 profiles/day** in this basin; the model gives **11,759** |
| Output | `pred_2020.nc`, `pred_2021.nc` standardized daily netCDF; 6 figures in `figs/` |
| Open | cyclone case study, thermocline bias correction, uncertainty, NRT demo |

Full numbers and the honest comparison against the earlier 3-year run are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Environment

Everything runs on the GPU box, reached as `blackwell`.

```
ssh blackwell
~/envs/ocean/bin/python -V        # 3.12.3
```

The venv `~/envs/ocean` has copernicusmarine 2.4.1, xarray, netCDF4, dask, scipy, numpy, pandas.
Source lives at `~/oceanembed/`, data under `~/ocean/{raw,interim,proc}`.

Note for the model stage: this box is **sm_120**, so torch must come from the cu128 index and use
bf16 — cu121/cu126 builds fail at the first matmul despite reporting CUDA available. torch
2.11.0+cu128 is installed and verified.

**MIG is enabled.** `nvidia-smi` advertises 97,887 MiB per card, but torch sees two `1g.24gb`
instances — 23.6 GiB and 46 SMs each, roughly a quarter of a card. Size everything against 24 GB.

## Credentials

One account only — [Copernicus Marine](https://data.marine.copernicus.eu/register). No NASA
Earthdata account is needed; the currents and wind products used here are CMEMS equivalents of the
PODAAC ones the PS suggests.

Preferred — let the client store them, so no secret ever sits in a shell profile or in `env`:

```
ssh blackwell
~/envs/ocean/bin/copernicusmarine login          # prompts for username + password
~/envs/ocean/bin/copernicusmarine login --check-credentials-valid
```

That writes `~/.copernicusmarine/.copernicusmarine-credentials`. Confirm it is not world-readable:

```
chmod 600 ~/.copernicusmarine/.copernicusmarine-credentials
```

Environment variables (`COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD`) also work and take
precedence, but putting them in `~/.bashrc` leaves the password in plaintext and exposes it to
anything that reads `env`. Use the credentials file unless something needs otherwise.

## Checking on a run

```
ssh blackwell ~/oceanembed/status.sh
```

Byte-level download bar per product per year, build progress, and the mean
RMSE / ACC of whichever models have finished. Read-only, safe to run any time.

## Running

```
cd ~/oceanembed

# self-check: regridding and depth interpolation, no network needed
~/envs/ocean/bin/python -m scripts.test_pipeline

# smallest useful first pull, to prove credentials and the grid end to end
~/envs/ocean/bin/python -m scripts.pipeline fetch sla --year 2019

# everything for the 3-year PoC (~20 GB)
~/envs/ocean/bin/python -m scripts.pipeline fetch-all

# regrid to the working grid
~/envs/ocean/bin/python -m scripts.pipeline build-target 2019
~/envs/ocean/bin/python -m scripts.pipeline build-inputs 2019

# stage many years: fetch -> build -> verify -> delete raw, one year at a time
~/envs/ocean/bin/python -m scripts.pipeline stage --years 2014 2015 2016 2017 2018

# baselines then the model — identical scoring path for all three
TR="2014 2015 2016 2017 2018 2019"; TE="2020 2021"
~/envs/ocean/bin/python -m scripts.train climatology --train $TR --test $TE
~/envs/ocean/bin/python -m scripts.train linear      --train $TR --test $TE
~/envs/ocean/bin/python -m scripts.train unet        --train $TR --test $TE --epochs 250

# standardized output + operational diagnostics, then figures
~/envs/ocean/bin/python -m scripts.predict --train $TR --test 2021
~/envs/ocean/bin/python -m scripts.figures

# independent validation against Argo profiles
~/envs/ocean/bin/copernicusmarine get   --dataset-id cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr   --filter "*global/202[01]/*_PR_PF.nc" --output-directory ~/ocean/argo_raw
~/envs/ocean/bin/python -m scripts.argo collect --years 2020 2021
~/envs/ocean/bin/python -m scripts.argo score
```

`fetch` skips files that already exist, so it is safe to re-run after an interruption. It aborts if
free disk drops below 25 GB.

## Grid

**100 × 240 cells at 0.25°**, cell-centred: 5.125–29.875 N by 45.125–104.875 E, so the grid tiles
the region exactly. 15 standard depths (0–1000 m), daily.

That registration was chosen from real files, not convention. SST (0.05°), SSS (0.125°), SLA
(0.25°), currents (0.25°) and wind (0.125°) all have cell centres on `…125/.375/.625/.875`, so each
reaches the working grid by an exact integer-factor area-mean coarsen with **no interpolation** —
which keeps the land mask crisp. Only GLORYS (1/12°, origin 4.5) is offset and needs an interp.
