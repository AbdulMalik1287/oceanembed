# Results — 8-year run

**Train 2014–2019** (2,192 days), **test 2020 and 2021** (731 days, both held out and scored once).
Epoch 177 of 250 selected on a 431-day validation split of interleaved 10-day blocks that never
touches the test years. U-Net 1.93 M parameters, 36 minutes on one MIG slice. ~8.6 M scored samples
per depth level.

Method in [`METHODOLOGY.md`](METHODOLOGY.md). Raw scores in `~/ocean/runs/*.json`; the earlier
3-year run is preserved in `~/ocean/runs_3yr/`.

## Skill by depth

RMSE in °C, lower is better. ACC is anomaly correlation, higher is better. Climatology has no ACC
by construction — its anomaly is identically zero.

| depth (m) | clim RMSE | linear RMSE | linear ACC | **unet RMSE** | **unet ACC** | unet vs clim |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.620 | 0.577 | 0.371 | **0.571** | **0.430** | +7.9% |
| 5 | 0.607 | 0.565 | 0.367 | **0.564** | **0.420** | +7.1% |
| 10 | 0.605 | **0.563** | 0.369 | 0.565 | **0.413** | +6.6% |
| 20 | 0.653 | **0.604** | 0.379 | 0.616 | 0.397 | +5.6% |
| 30 | 0.754 | **0.694** | 0.393 | 0.708 | **0.409** | +6.2% |
| 50 | 1.053 | 0.947 | 0.439 | **0.931** | **0.489** | +11.6% |
| 75 | 1.500 | 1.293 | 0.511 | **1.238** | **0.575** | +17.5% |
| **100** | 1.773 | 1.485 | 0.555 | **1.415** | **0.608** | **+20.2%** |
| 125 | 1.714 | 1.419 | 0.572 | **1.376** | **0.600** | +19.7% |
| 150 | 1.460 | 1.219 | 0.561 | **1.205** | **0.572** | +17.5% |
| 200 | 0.952 | **0.830** | 0.498 | 0.836 | 0.495 | +12.1% |
| 300 | 0.532 | **0.505** | **0.316** | 0.544 | 0.245 | −2.3% |
| 500 | 0.356 | **0.347** | **0.224** | 0.377 | 0.159 | −6.0% |
| 700 | 0.367 | **0.360** | **0.191** | 0.384 | 0.155 | −4.8% |
| 1000 | 0.387 | **0.382** | **0.170** | 0.410 | 0.101 | −5.9% |
| **mean** | **0.889** | **0.786** | **0.394** | **0.783** | **0.405** | **+11.9%** |

## Operational diagnostics

Scored on 2021 against GLORYS, over wet cells. None of these can be computed from SST alone.

| Diagnostic | RMSE | correlation | 3-year run |
|---|---:|---:|---:|
| **TCHP** — tropical cyclone heat potential | 17.6 kJ cm⁻² | **0.897** | 0.858 |
| D26 — depth of the 26 °C isotherm | 14.3 m | **0.830** | 0.759 |
| MLD — mixed layer depth | 15.3 m | **0.741** | 0.669 |

**TCHP is the headline for the Disaster Management theme.** It is the operational predictor of
cyclone rapid intensification in the Bay of Bengal: a deep warm layer keeps feeding a storm even
after its own winds have mixed the surface, whereas a shallow one cools and starves it. Two
cyclones over identical SST can behave completely differently, and the difference is invisible to a
satellite. Reconstructing the profile makes it visible — at 0.90 correlation, daily, basin-wide.

## Findings

### 1. Skill concentrates at the thermocline

Peak improvement is **+20.2% at 100 m with ACC 0.608**, and 75–150 m all sit near +20%. The U-Net
beats the context-free linear baseline by 4.7% there.

Those are the depths where the answer cannot be read off the temperature directly overhead and must
be inferred from mesoscale structure in the surface pattern — exactly what the encoder's receptive
field supplies and what a per-cell regression cannot see. Near the surface the margin is small
because SST is an input and the answer is nearly given.

### 2. No skill below 300 m — the U-Net is slightly *worse* than climatology there

Below 300 m the network trails both baselines (−6.0% at 500 m) and its ACC decays to 0.10, while
the linear model stays marginally ahead of climatology.

This is physically honest rather than a defect: daily surface fields carry almost no information
about 500–1000 m, and the network spends capacity trying anyway. The useful vertical range is
**surface to roughly 200–300 m** — the mixed layer, the thermocline, and the depths that drive
upper-ocean heat content and cyclone heat potential. An operational product should fall back to
climatology below 300 m.

### 3. Most of the bias is now corrected

| depth (m) | clim bias | linear bias | unet bias |
|---:|---:|---:|---:|
| 50 | −0.290 | −0.135 | **−0.049** |
| 75 | −0.444 | −0.207 | **−0.057** |
| **100** | **−0.596** | −0.318 | **−0.158** |
| 125 | −0.602 | −0.335 | **−0.212** |
| 150 | −0.476 | −0.252 | **−0.175** |
| 200 | −0.267 | −0.136 | **−0.105** |

The climatology is still −0.596 °C cold at 100 m; the U-Net removes **73%** of that, up from 35% in
the 3-year run. This is the clearest evidence that the network is learning real structure rather
than reproducing the reference.

### 4. Honest comparison with the 3-year run

**The raw RMSE values are not comparable between the two runs.** The 8-year run tests on 2020+2021
with a climatology built from six years; the 3-year run tested on 2021 with a two-year climatology.
Both the test set and the reference changed.

| | mean RMSE | vs climatology | @100 m vs clim | vs linear @100 m |
|---|---:|---:|---:|---:|
| 3-year (train 2019–20, test 2021) | 0.890 | +13.7% | +22.9% | +11.6% |
| **8-year (train 2014–19, test 2020–21)** | **0.783** | **+11.9%** | **+20.2%** | **+4.7%** |

The relative margin *shrank* even though the reconstruction improved, because a six-year climatology
is a much fairer baseline than a two-year one — it carries a −0.596 °C bias at 100 m instead of
−0.628 °C, and is better everywhere else too. **Part of what the 3-year model was credited for was
compensating for a weak reference.** The 11.9% is the number that survives scrutiny.

The U-Net's advantage over the linear baseline also narrowed in the mean, to a near-tie
(0.783 vs 0.786), while holding at the thermocline. Read together with finding 2, the network earns
its keep in a band, not everywhere.

Note that two things changed at once between the runs — training data and the blanking of input
channels over land — so the improvement cannot be attributed to either alone. Under a deadline that
was the right trade; it is not an ablation study.

### 5. Still data-limited, not capacity-limited

Train loss 0.069 against validation 0.233 — a 3.4× gap (was 4.2× on three years), with validation
flat from about epoch 130. More capacity would widen that gap, not close it. If more skill is
wanted, the lever is still more data or better inputs, not a bigger network.

## Figures

| File | Shows |
|---|---|
| `figs/01_skill_by_depth.png` | RMSE and ACC vs depth, all three models, thermocline band marked |
| `figs/02_acc_map_100m.png` | where the model has skill, at 100 m |
| `figs/03_profiles.png` | reconstructed vs GLORYS profiles, Arabian Sea and Bay of Bengal, four dates |
| `figs/04_tchp.png` | TCHP maps, agreement scatter, and the 2021 seasonal cycle |

## Next steps

1. **Independent ARGO validation** — the one remaining problem-statement deliverable, and the
   answer to the sharpest objection (GLORYS assimilates Argo, so training against it is partly
   emulating a data-assimilation system).
2. **Fall back to climatology below 300 m** in the delivered product, and say why.
3. **More years** — 2010–2024 is available; the record is bounded by wind (2007) and SSS (2024).

## Problem statement deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | End-to-end preprocessing pipeline | done |
| 2 | Satellite embedding engine | done — the U-Net bottleneck, readable directly |
| 3 | DL reconstruction model | done |
| 4 | Standardized daily 0.25° output | done — `scripts/predict.py` writes `pred_YYYY.nc` |
| 5 | Validation against independent ARGO | **open** |
| 6 | Bay of Bengal / Arabian Sea PoC | figures done; a named-cyclone case study would finish it |
