# SIH26066 OceanEmbed — submission content

Slide-by-slide content for the SIH idea submission and the internal round.
Written to the standard five-section SIH template; paste into the official
deck and re-style. Speaker notes are what to *say*, not what to put on the slide.

**The one thing that differentiates this submission: it is not a proposal. It is
built, it runs, and the numbers below are measured on two held-out years.** Most
entries at this stage describe an intended approach. Lead with that difference
and never bury it.

---

## Slide 1 — Title

**OceanEmbed**
Satellite Embedding-Based Deep Learning Framework for Reconstruction of
Subsurface Ocean Temperature from Surface Satellite Observations

PS SIH26066 · Ministry of Earth Sciences · INCOIS · Software · Team <name>

> **Say:** "We've built and evaluated this, not just designed it. Everything you'll
> see is measured on two full years the model never saw."

---

## Slide 2 — The problem

**The ocean is opaque below the surface.**

- Subsurface temperature drives cyclone intensification, marine heatwaves,
  fisheries, and monsoon coupling.
- It is measured by **Argo floats** — roughly one profile per 3°×3° box per
  10 days. Sparse in space, sparse in time.
- Satellites see the surface **continuously, basin-wide, daily** — but only the
  surface.
- Result: for any given day, the 3D state of the North Indian Ocean is unknown
  almost everywhere.

> **Say:** "The gap isn't sensors, it's dimensionality. We have excellent daily
> coverage of a 2D slice of a 3D problem."

---

## Slide 3 — Proposed solution

**Learn the mapping from the surface pattern to the vertical profile.**

Surface fields carry indirect signatures of what lies beneath — thermocline
displacement shows up in sea level, eddies in surface currents, mixing in wind
history. A deep network can learn that relationship where a formula cannot.

```
SST, SSS, SLA, currents (u,v), winds (u,v)     7 daily surface fields
                    ↓
        satellite embedding (encoder)          compact latent ocean state
                    ↓
          reconstruction (decoder)
                    ↓
   temperature at 15 depths, 0–1000 m          daily, 0.25°, basin-wide
```

**Innovation & uniqueness**

1. The **embedding is the bottleneck of a single U-Net**, not a separately
   pre-trained autoencoder — encoder and decoder are trained jointly on the
   actual objective, so the latent space is optimised for reconstruction rather
   than for compressing the inputs.
2. The model predicts **anomaly from climatology**, not absolute temperature, so
   its reported skill is skill beyond the seasonal cycle rather than credit for
   knowing the calendar.
3. We output **operational diagnostics**, not just temperature — including
   Tropical Cyclone Heat Potential, which no satellite can measure directly.

---

## Slide 4 — Technical approach

**Data** — all from Copernicus Marine, one credential, fully reproducible:
OSTIA SST (0.05°), multi-obs SSS, DUACS SLA, GLOBCURRENT surface currents,
L4 scatterometer winds. Target: GLORYS12V1 reanalysis at 1/12°, 50 levels.

**Pipeline** — 100 × 240 cells at 0.25°, cell-centred so that five of six inputs
regrid by exact integer-factor area-mean coarsening with **no interpolation at
all**, which keeps the coastline sharp. Vertical interpolation from GLORYS's 36
native levels to the 15 depths the PS specifies.

**Model** — U-Net, 1.93 M parameters. Encoder 32→64→128, bottleneck 256 × 13 × 30
(the satellite embedding, 2.6× compression), decoder with skip connections,
1×1 head to 15 depths. AdamW, cosine schedule, bf16, masked MSE on per-depth
normalised anomalies. **Trains in 36 minutes** on one MIG slice of an
RTX PRO 6000 Blackwell — 250 epochs, of which epoch 177 is selected.

**Evaluation** — train 2014–2019 (2,192 days), test on **2020 and 2021** in
full (731 days), held out and scored once. Epoch count chosen on a validation
split of every 5th 10-day block, never on the test years.

> **Say:** "Split by year, never randomly. Consecutive days in the ocean are
> nearly identical, so a random split would hand us a fake 0.99 correlation.
> Even our validation split is by 10-day blocks spread across all seasons — we
> found that a plain chronological holdout validates you on one season and tells
> you to stop training after two epochs."

---

## Slide 5 — Results

**We beat the climatology baseline by 11.9%, and by 20.2% at the thermocline —
tested on two full years the model has never seen.**

| | mean RMSE | RMSE @100 m | ACC @100 m |
|---|---:|---:|---:|
| Climatology (floor) | 0.889 °C | 1.773 °C | — |
| Per-depth linear | 0.786 °C | 1.485 °C | 0.555 |
| **OceanEmbed U-Net** | **0.783 °C** | **1.415 °C** | **0.608** |

*Figure: `01_skill_by_depth.png`*

The gain is concentrated at **75–150 m** — exactly the depths where the answer
cannot be read off the temperature directly overhead and must be inferred from
mesoscale structure. That is the physical hypothesis, and it held.

> **Say:** "The linear model sees each cell's own surface state. The U-Net sees
> the surrounding pattern. The gap between them at 100 m *is* the value of
> spatial context — that's why the architecture is what it is."

**We also report where it does not work:** below 300 m the model is slightly
*worse* than climatology. Daily surface fields carry almost no information about
500–1000 m. An operational product should fall back to climatology there, and we
say so rather than average it away.

> **Say:** "Any model claiming skill at 1000 m from surface data alone should
> make you suspicious."


> **Say:** "One number moved the wrong way and it's worth explaining. Against
> our first three-year run the margin over climatology was 13.7%; here it's
> 11.9%. The reconstruction got *better* — every diagnostic improved on the same
> test year. What changed is that six years of data make a much fairer
> climatology baseline. Part of what the earlier model was credited for was
> fixing a weak reference. We'd rather quote the number that survives scrutiny."

---

## Slide 6 — Why INCOIS should care: cyclone heat potential

**Tropical Cyclone Heat Potential — reconstructed at 0.90 correlation.**

*Figure: `04_tchp.png`*

| Diagnostic | RMSE | correlation |
|---|---:|---:|
| **TCHP** | 17.6 kJ cm⁻² | **0.897** |
| D26 (26 °C isotherm depth) | 14.3 m | 0.830 |
| Mixed layer depth | 15.3 m | 0.741 |

TCHP is the operational predictor of **cyclone rapid intensification** in the Bay
of Bengal. Two cyclones crossing water with *identical* sea surface temperature
behave completely differently depending on how deep the warm layer runs: a deep
one keeps feeding the storm even after its own winds churn the surface, a
shallow one cools and starves it.

**That difference is invisible to a satellite.** It requires the vertical
profile — which is what this system reconstructs, daily and basin-wide.

> **Say:** "This is the answer to 'so what'. We're not producing a temperature
> field for its own sake, we're producing the variable a warning centre actually
> uses, on days when no float is anywhere near the storm."

---

## Slide 7 — Independent validation

**We checked against 8,015 Argo profiles the model has never seen.**

*Figure: `05_argo_validation.png`*

| | mean RMSE vs Argo |
|---|---:|
| Climatology | 0.900 °C |
| **OceanEmbed** | **0.852 °C** — +5.3%, and +8–11% through 75–200 m |
| GLORYS reanalysis | 0.615 °C |

Two things this says, and we would rather say them than have you find them.

**The margin is smaller than the reanalysis-based score suggests.** Against GLORYS we beat
climatology by 11.9%; against real observations, 5.3%. The difference is the cost of training
against a reanalysis — some of the apparent skill was agreement with GLORYS's errors, not the ocean.

**GLORYS's column is not a fair competitor.** It assimilated these very profiles, so 0.615 is a fit
statistic, not an independent score. It also needs in-situ data and is not available at real-time
latency. Our model runs from satellite surface fields alone, in seconds.

**And we can point at the largest remaining error.** GLORYS is +0.451 °C warm against Argo at 100 m;
we inherit about three quarters of that. Fine-tuning against Argo directly, or fitting a
depth-dependent bias correction, is the clear next step.

> **Say:** "The obvious challenge to this project is that GLORYS already assimilates Argo, so
> training on it is partly emulating a data-assimilation system. That's fair. So we went and
> checked against the floats themselves. The margin drops from 11.9% to 5.3% — and we'd rather
> quote the 5.3%, because it's the one that's true."

---

## Slide 8 — Feasibility and viability

**Already demonstrated**

- End-to-end pipeline runs unattended: acquisition → harmonisation → training →
  scoring → standardized netCDF output.
- 8 years of real data processed end to end, staged so that peak disk stays
  under 10 GB; two full years held out for testing.
- 16 automated correctness assertions covering regridding, land masking, depth
  interpolation and loss behaviour.

**Cost is trivial.** A full 250-epoch run over eight years of data takes 36
minutes on a quarter of one GPU. Scaling to the full 2010–2024 record
is ~144 GB and one overnight download — the constraint is bandwidth, never
compute. Inference for an operational daily product is seconds.

**Risks, and how they are handled**

| Risk | Mitigation |
|---|---|
| GLORYS assimilates Argo, so training on it partly emulates a DA system | **Done** — validated against 8,015 raw Argo profiles; margin quoted from that, not from GLORYS |
| Satellite salinity before 2010 is a reconstruction, not a retrieval | Restrict the operational record to 2007+, where all inputs are genuine retrievals |
| Coastal cells lost to regridding | Masked conservative regridding if coastal skill proves to matter |
| Short climatology biases the reference | Swap in the CMEMS long-term monthly climatology |

> **Say:** "We know the sharpest question you can ask is that GLORYS already ate
> the Argo data. We'd rather raise it ourselves than have you find it."

---

## Slide 9 — Impact and benefits

**Direct users:** INCOIS ocean forecasting and hazard-warning services, IMD
cyclone forecasting, Indian Navy, marine fisheries advisories.

- **Disaster management** — daily basin-wide TCHP for cyclone intensity guidance,
  in a basin responsible for a disproportionate share of global cyclone deaths.
- **Fisheries** — thermocline and mixed layer depth drive the Potential Fishing
  Zone advisories that reach lakhs of small-vessel fishers.
- **Climate monitoring** — upper-ocean heat content and marine heatwave detection
  with vertical structure, not just a surface signature.
- **Cost** — built entirely on open Copernicus data with a 1.9 M parameter model.
  No new sensors, no new missions, no proprietary inputs.

---

## Slide 10 — Research and references

- Problem statement SIH26066, INCOIS / Ministry of Earth Sciences
- GLORYS12V1 Global Ocean Physics Reanalysis — https://doi.org/10.48670/moi-00021
- OSTIA SST — https://doi.org/10.48670/moi-00168
- DUACS sea level — https://doi.org/10.48670/moi-00145
- Multi-observation sea surface salinity — https://doi.org/10.48670/moi-00051
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for
  Biomedical Image Segmentation*
- Argo programme, and the INCOIS Live Access Server gridded Argo product
- Leipper & Volgenau (1972), *Hurricane heat potential of the Gulf of Mexico* —
  the origin of TCHP as an operational quantity

**Code:** github.com/AbdulMalik1287/oceanembed

---

## Anticipated questions

**"Isn't this circular — GLORYS already assimilates Argo?"**
Partly, and we measured exactly how much. Against 8,015 independent Argo
profiles the margin over climatology falls from 11.9% to 5.3%, and we inherit
about three quarters of GLORYS's +0.451 °C warm bias at 100 m. We quote the
5.3%. The operational value stands regardless: GLORYS needs in-situ input and is
not available at real-time latency, while this model runs from surface fields
alone in seconds.

**"Why not just use Argo directly?"**
Coverage. One profile per 3°×3° per 10 days cannot produce a daily basin-wide
field, and cyclone forecasting needs the field on the day the storm is there.

**"Why a U-Net and not a transformer?"**
731 training days. At that scale a 2 M parameter convolutional model is the right
capacity, and ViT variants are in the plan as an ablation. We picked based on
data size, not fashion.

**"Your model is worse than climatology below 500 m."**
Correct, and reported. Surface observations carry almost no information about
those depths on daily timescales. The useful range is surface to ~300 m — which
covers the mixed layer, the thermocline, and everything that drives heat content
and cyclone potential.

**"What's the real-world latency?"**
Inference is seconds. The binding constraint is input availability — the
near-real-time versions of these same products are published daily, so an
operational build would run on NRT inputs with the same weights.
