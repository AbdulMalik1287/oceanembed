# SIH26066 — OceanEmbed

**Title:** OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of
Subsurface Ocean Temperature from Surface Satellite Observations

| Field | Value |
|---|---|
| PS Number | SIH26066 (S.No. 66) |
| Organization | Ministry of Earth Sciences (MoES) |
| Department | INCOIS, Ocean Valley, Hyderabad |
| Category | Software |
| Theme | Space Technology (listed as "Disaster Management" in the official INCOIS PDF) |
| Idea submission deadline | 20 September 2026 |
| Official PDF | `docs/INCOIS_PS_official.pdf` |

## Task

Reconstruct daily, depth-wise subsurface ocean temperature over the North Indian Ocean
(5°N–30°N, 45°E–105°E) at 0.25° resolution, using **only surface satellite observations**.

- **Inputs:** SST, SSS, SSH/SLA, surface currents (U, V), surface winds (U, V)
- **Output depths (m):** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
- **Grid:** 0.25° × 0.25°, daily
- **Method:** satellite *embeddings* — compact latent representations of the surface state
  (CNN / ViT / autoencoder / GNN / attention hybrids) feeding a profile-reconstruction model
- **Metrics:** correlation, RMSE, bias vs independent observations

## Datasets (from official PDF)

### Training inputs

| Variable | Product & native resolution | Source |
|---|---|---|
| SST | OSTIA, 0.05°, daily | https://doi.org/10.48670/moi-00168 |
| SSS | SMAP, SMOS, 0.125°, daily | https://doi.org/10.48670/moi-00051 |
| SSH / SLA | DUACS, 0.25°, daily | https://doi.org/10.48670/moi-00145 |
| Currents (U,V) | 0.25°, daily | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0 |
| Winds (U,V) | 0.25°, daily | https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal <br> https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 |

### Training target (subsurface temperature)

- GLORYS Global Ocean Reanalysis — https://doi.org/10.48670/moi-00021 (variable: `thetao`)

### Independent validation

- Gridded ARGO, INCOIS Live Access Server (LAS)

## Expected deliverables

1. End-to-end preprocessing / harmonization pipeline for the multi-source datasets
2. Satellite embedding engine (surface state → latent representation)
3. DL reconstruction model (latent → temperature profile)
4. Standardized output: daily, 0.25°, 15 standard depths
5. Validation framework against independent ARGO observations
6. Working PoC over the Bay of Bengal / Arabian Sea

## Extra info

- Idea PDF (Google Drive): https://drive.google.com/file/d/1TrME3MMW-aYaf7KNmDXpwl2CcvLYjp-T/view
- Note the PDF's own dataset table says "If a dataset is not available at the required resolution,
  the team may select the openly available product and perform appropriate interpolation/regridding."
