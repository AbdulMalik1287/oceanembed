# PPT handoff. OceanEmbed (SIH26066)

Everything needed to build the deck without reading the code. Slide copy is
paste-ready; diagrams are drawn in text for you to redraw in PowerPoint.

**Internal round: 10 September.** Full speaker notes live in
[`SUBMISSION.md`](SUBMISSION.md); this file is the asset and diagram pack.

---

## 1. The one-line pitch

> Satellites see only the ocean's skin. We reconstruct the temperature of the
> water underneath it, every day, across the whole North Indian Ocean, at
> depths no satellite can reach, and we validated it against 8,015 floats that
> the model has never seen.

**The differentiator to lead with: this is built and measured, not proposed.**
Most entries at the idea stage describe an intended approach.

---

## 2. Figures: where they are, what they show

All in `figs/`. Drop them in at full width; they are already sized for slides.

| File | Use it on | Shows |
|---|---|---|
| `06_coverage_gap.png` | **The problem slide** | 11 Argo profiles that day vs 11,759 reconstructed. The single most persuasive image we have. |
| `01_skill_by_depth.png` | Results | RMSE and ACC vs depth for all three models, thermocline band shaded |
| `04_tchp.png` | Why INCOIS cares | Cyclone heat potential maps, agreement scatter, seasonal cycle |
| `05_argo_validation.png` | Independent validation | Error and bias against real floats; shows the inherited bias honestly |
| `03_profiles.png` | Optional, method | Reconstructed vs GLORYS profiles at two sites, four dates |
| `02_acc_map_100m.png` | Optional, results | Where the model has skill, spatially |

**Live demo:** run `python -m http.server 8080` inside `demo/web`, open
`http://127.0.0.1:8080/index.html`. Or open `demo/web/standalone.html` directly, one file, no server, no network.

---

## 3. Architecture flow

### A. Data pipeline: acquisition to training-ready cubes

```
  COPERNICUS MARINE  (one account, six products)
  ┌───────────────────────────────────────────────────────────┐
  │ SST 0.05° daily      SSS 0.25° daily     SLA 0.25° daily   │
  │ Currents 0.25° daily Wind 0.125° HOURLY                    │
  │ GLORYS reanalysis 1/12°, 50 levels   ← the target          │
  └───────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  HARMONISATION                │
              │  • wind: hourly → daily mean  │
              │  • area-mean coarsen to 0.25° │
              │    (exact integer factors,    │
              │     so NO interpolation)      │
              │  • GLORYS: 36 native levels   │
              │    → the 15 standard depths   │
              │  • land mask, unit fixes      │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  ALIGNED DAILY CUBES          │
              │  inputs  7 × 100 × 240        │
              │  target 15 × 100 × 240        │
              │  2014-2021, ~8 GB             │
              └───────────────────────────────┘
```

**Point worth making on this slide:** staged *fetch → build → verify → delete*,
one year at a time, so 77 GB of raw downloads only ever needs ~10 GB of disk.

### B. Model: the satellite embedding engine

```
   INPUT  11 × 100 × 240
   ┌────────────────────────────────────────────┐
   │ SST · SSS · SLA · current u,v · wind u,v   │  7 observed
   │ day-of-year sin,cos · latitude · longitude │  4 static
   └────────────────────────────────────────────┘
                     │
        ENCODER      │  32 → 64 → 128 channels, halving in space
                     ▼
   ╔════════════════════════════════════════════╗
   ║  BOTTLENECK   256 × 13 × 30                ║
   ║  = the SATELLITE EMBEDDING                 ║   ← what the PS asks for
   ║  99,840 numbers, 2.6× compression          ║
   ╚════════════════════════════════════════════╝
                     │
        DECODER      │  128 → 64 → 32, each level joined to its
                     │  encoder skip connection
                     ▼
   OUTPUT 15 × 100 × 240
   temperature at 0, 5, 10, 20, 30, 50, 75, 100,
   125, 150, 200, 300, 500, 700, 1000 m
```

**Say on this slide:** the embedding is not a separate pre-trained autoencoder. It is the bottleneck of one network trained end to end on the actual objective,
so the latent space is optimised for reconstruction rather than for compressing
the inputs. Encoder = embedding engine, decoder = reconstruction model.

**Why an encoder/decoder at all:** subsurface temperature at a point is *not* a
function of the surface directly above it. An eddy or a displaced thermocline is
a pattern hundreds of kilometres wide, readable only from context, which the
encoder's growing receptive field supplies. The skip connections then restore
the sharp local SST and SLA that fix the mixed layer.

### C. System: how it runs

```
   satellite surface fields (near-real-time twins exist for every input)
                     │
                     ▼
          ┌─────────────────────┐
          │  OceanEmbed U-Net   │   1.93 M params · inference in seconds
          └─────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  3-D temperature          DERIVED PRODUCTS
  netCDF, daily,           • TCHP  cyclone heat potential
  0.25°, 15 depths         • D26   26 °C isotherm depth
  (PS deliverable #4)      • MLD   mixed layer depth
                           • OHC   heat content 0-700 m
                                  │
                                  ▼
                      REST API  →  operator console
                      /api/field, /api/profile,
                      /api/diagnostics, /api/scores
```

---

## 4. Numbers: copy-paste, do not retype

### Headline skill (train 2014-2019: test 2020 **and** 2021, both held out)

| | mean RMSE | @100 m RMSE | @100 m ACC |
|---|---|---|---|
| Climatology (floor) | 0.889 °C | 1.773 °C | - |
| Per-depth linear | 0.786 °C | 1.485 °C | 0.555 |
| **OceanEmbed U-Net** | **0.783 °C** | **1.415 °C** | **0.608** |

**+11.9% over climatology overall, +20.2% at 100 m.**

### Operational diagnostics (2021)

| | RMSE | correlation |
|---|---|---|
| **TCHP** | 17.6 kJ cm⁻² | **0.897** |
| D26 | 14.3 m | 0.830 |
| Mixed layer depth | 15.3 m | 0.741 |

### Independent Argo validation, **the honest headline**

8,015 quality-controlled float profiles, 2020-2021, never seen in training.

| | mean RMSE vs Argo |
|---|---|
| Climatology | 0.900 °C |
| **OceanEmbed** | **0.852 °C** → **+5.3%**, and **+8-11%** through 75-200 m |
| GLORYS reanalysis | 0.615 °C |

### Coverage

| | profiles per day, North Indian Ocean |
|---|---|
| Argo floats | **11** |
| **OceanEmbed** | **11,759** |

Bay of Bengal averages **3.3 profiles a day**, and has **none at all on 8% of days**.

### Scale and cost

- 8 years of data · 2,192 training days · two full years held out
- U-Net **1.93 M parameters**, trains in **36 minutes** on a quarter of one GPU
- Inference: **seconds**. The constraint is bandwidth, never compute.
- 15 automated correctness checks, no network or data required to run them

---

## 5. Suggested slide order

1. **Title**
2. **The problem**: the ocean is opaque below the surface; Argo is sparse
3. **The gap, in one picture**, `06_coverage_gap.png`
4. **Solution**, surface → embedding → profile (diagram B)
5. **Technical approach**, diagram A + the grid/harmonisation points
6. **Results**, `01_skill_by_depth.png` + the skill table
7. **Why INCOIS cares**, `04_tchp.png`, cyclone heat potential
8. **Independent validation**, `05_argo_validation.png`, the 5.3%
9. **Live demo**: the console
10. **Feasibility, impact, references**

---

## 6. Two things to say before a judge says them

**"GLORYS already assimilates Argo, isn't this circular?"**
Raise it yourself on the validation slide. Against GLORYS we beat climatology by
11.9%; against real floats, 5.3%. We inherit about three quarters of GLORYS's
+0.451 °C warm bias at 100 m. Quote the 5.3%. GLORYS's own 0.615 is *not* a fair
comparison. It assimilated those very profiles, so it is a best case, and it
needs in-situ data and is not available at real-time latency.

**"It is worse than climatology below 300 m."**
True, and we report it. Surface fields carry almost no information about
500-1000 m. The useful range is surface to ~300 m, which covers the mixed layer,
the thermocline, and everything driving heat content and cyclone potential. An
operational product should serve climatology below 300 m.

---

## 7. Do not put these on a slide

- **The old 3-year numbers**: 13.7%, +22.9%, TCHP 0.858. They came from a
  two-year climatology that was too weak a baseline. Superseded.
- **"Beats GLORYS"**: it does not, and GLORYS is not the competitor anyway.
- **Any skill claim below 300 m.**
- **"Real-time"**: the pipeline *could* run on near-real-time inputs, but we
  have not demonstrated it. Say "designed for near-real-time operation".

---

## 8. Demo script, 90 seconds

1. Open the console. It starts on **29 August 2021**, Arabian Sea, peak cyclone
   season, and a day Argo had **zero** profiles in the Bay of Bengal.
2. Point at the two tiles: heat potential and the warm-layer depth, read live at
   that point. Note the dashed line in the second illustration is the **real**
   26 °C isotherm depth, not decoration.
3. **Click into the Bay of Bengal.** Everything updates, the profile redraws,
   the tiles re-read, the basin name changes.
4. Drag the depth slider to 100 m and switch the map to **Heat**. Say: *this is
   the variable that decides whether a cyclone intensifies, and no satellite can
   measure it.*
5. Scroll to the Argo table: *checked against 8,015 floats we never trained on.*

---

## 9. Where everything lives

```
Repo   github.com/AbdulMalik1287/oceanembed   (private)
Docs   docs/RESULTS.md        every number, with caveats
       docs/METHODOLOGY.md    how it works, stage by stage
       docs/SUBMISSION.md     slide-by-slide copy + speaker notes
       docs/RUNBOOK.md        how to run and troubleshoot it
Figures figs/*.png
Demo   demo/web/standalone.html   one file, no server
Data   local backup at Claude VS\sih-26066-data (7.8 GB, verified)
```
