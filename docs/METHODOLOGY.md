# Methodology

How OceanEmbed is actually implemented, stage by stage. For the problem statement see
[`PS.md`](PS.md); for measured skill see [`RESULTS.md`](RESULTS.md).

## 1. The learning problem

One training sample is **one calendar day over the whole basin**, image-to-image, not per-pixel
regression.

```
X : (11, 100, 240)   one day of surface state
Y : (15, 100, 240)   the same day, temperature at 15 standard depths
```

2019-2020 gives 731 samples, 2021 gives 365 held out for test. This is a small dataset by
deep-learning standards, and it is the binding constraint on model capacity, not compute.

There is **no lead time**. Input day *t* produces output day *t*. This is reconstruction of the
vertical dimension satellites cannot see, not forecasting.

## 2. Grid and region

North Indian Ocean, 5-30°N by 45-105°E, at 0.25°: **100 × 240 cells**, cell-centred on
5.125-29.875 N and 45.125-104.875 E.

The registration was chosen from real files rather than convention. Probing one day of each product
showed SST (0.05°), SSS (0.125°), SLA (0.25°), currents (0.25°) and wind (0.125°) all have cell
centres on `…125/.375/.625/.875`. On this grid each of them reaches the target by an exact
integer-factor area-mean coarsen with **no interpolation at all**, which keeps the land mask sharp.
A grid on whole 0.25° multiples would have been half a cell off from every one of them, and
interpolating across that offset averages each coastal cell with its land neighbour, measured on
real DUACS data, that pushed the domain from 48.4% NaN to 50.5%, discarding ~515 ocean cells per
day around the basin rim.

The grid also tiles the region exactly: 100 × 0.25° = 25°, flush with 5°N and 30°N.

GLORYS (1/12°, grid origin 4.5) is the one product that does not align, and it is interpolated by
1/24° after coarsening. It is the target field, so the cost is losing coastal pixels from training
and scoring rather than corrupting them.

## 3. Acquisition

Six `copernicusmarine.subset` calls per year, bbox-clipped server-side with 0.5° of padding so the
regrid has data on all four edges instead of a NaN border. One file per product per year, resumable,
aborts under 25 GB free.

All six products come from Copernicus Marine, so a **single account** covers everything. The CMEMS
currents and wind products replace the PODAAC OSCAR/CCMP suggestions in the PS, avoiding a second
credential and CCMP's global-file distribution which has no bbox subsetting.

| Variable | Dataset id | Native |
|---|---|---|
| SST | `METOFFICE-GLO-SST-L4-REP-OBS-SST` (`analysed_sst`, kelvin) | 0.05° daily |
| SSS | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` (`sos`) | 0.25° daily |
| SLA | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` (`sla`) | 0.25° daily |
| Currents | `cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m` (`uo`, `vo`) | 0.25° daily |
| Wind | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | 0.125° **hourly** |
| Target | `cmems_mod_glo_phy_my_0.083deg_P1D-m` (`thetao`) | 1/12° daily, 50 levels |

Measured cost: **~9.6 GB per year** (GLORYS 5.75 GB, wind 3.39 GB, the rest under 0.7 GB combined).

## 4. Harmonization

`to_target_grid()` handles every product with one code path. It reads the file's own latitude
spacing, computes `factor = round(0.25 / step)`, area-mean coarsens by that factor, and then
**selects** rather than interpolates when the coarsened centres already match the target.

Two details that are easy to get wrong and were both caught by testing:

- The factor is derived from the file, not hardcoded, so a product that silently changes resolution
  cannot produce a quietly wrong answer.
- After select or interp, the canonical float64 axes are **stamped onto the result**. `sel` returns
  the *source's* coordinate values, which netCDF stores as float32; different products then differ
  in the last bits (~2e-6), and since xarray aligns on exact equality, combining the channels
  outer-joins them into a NaN-padded union. That produced a 144 × 394 cube at 78% NaN before it was
  fixed.

`to_std_depths()` flips a negative `elevation` axis to positive depth where needed and interpolates
GLORYS's 36 native levels between 0 and 1062 m onto the 15 the PS specifies. Only 0 m is an
extrapolation, and only 0.49 m past the shallowest native level.

`surface()` drops singleton vertical axes from the surface products, choosing the level nearest zero
**by value**, picking by index would silently take 15 m instead of the surface for currents,
whose axis may be stored as negative elevation in either order.

Wind is averaged from hourly to daily **on its native grid** before regridding, so the day boundary
is applied before any spatial operation.

Output: `inputs_YYYY.nc` (7 channels) and `target_YYYY.nc` (15 depths), identical 100 × 240 grid.

## 5. Array assembly

This is where the modelling decisions live. **Every statistic below is computed from the training
years only.**

1. **Climatology**, group the training target by day-of-year, mean, smooth with a 31-day *circular*
   window so 31 December and 1 January stay continuous.
2. **Anomaly target**, `Y = target - climatology`. The model never predicts absolute temperature,
   so its score reflects only what it adds beyond the seasonal cycle.
3. **Per-depth normalisation**, divide the anomaly by its per-depth training standard deviation.
   This replaces hand-tuned depth weights: unnormalised, the near-constant 1000 m level and the
   highly variable surface enter the loss on wildly different scales and the thermocline is ignored.
4. **Land mask**, ocean where the target is finite at the surface on every training day.
   Deliberately *not* "finite at every depth": the target is NaN below the seafloor, so requiring
   all 15 levels discards every shelf cell (~26% of the ocean here, including most of the coastal
   Bay of Bengal). Depth-varying bathymetry is handled in the loss and the metrics, which intersect
   this mask with `isfinite(truth)`, a shelf cell trains the levels above the seabed and is skipped
   below it.
5. **Input normalisation**, per-channel mean and standard deviation over wet cells. Residual NaNs
   (cloud gaps, differing coastal masks between products) become 0, the channel mean.
6. **Static channels**, day-of-year sin/cos plus normalised latitude and longitude, appended to the
   7 surface fields for 11 total. Convolutions are translation-equivariant, so without position the
   network cannot distinguish the Bay of Bengal from the Arabian Sea, two basins with very
   different stratification under similar surface signatures.

## 6. Architecture

U-Net, **1,929,775 parameters**, width 32, depth 3.

```
input       (11, 100, 240)    7 surface channels + doy sin/cos + lat + lon
   │ encoder: 32 → 64 → 128, halving spatially at each step
bottleneck  (256, 13, 30)     99,840 values, the satellite embedding, 2.64x compression
   │ decoder: 128 → 64 → 32, each level concatenated with its encoder skip
output      (15, 100, 240)    temperature at the 15 standard depths
```

Each block is Conv3×3 → BatchNorm → ReLU, twice. Upsampling is `ConvTranspose2d` stride 2, then
concatenation with the encoder skip, then another double conv. The head is a 1×1 convolution mapping
32 channels to 15 depths. 100 is not divisible by 8, so the forward pass reflect-pads to 104 and
crops back.

**The bottleneck is the "satellite embedding engine"** the problem statement asks for, and it is
readable directly via `forward(..., return_embedding=True)`. Encoder = embedding engine, decoder =
reconstruction model. No separate autoencoder pre-training stage is required.

Why the encoder/decoder shape suits the physics: subsurface temperature at a point is not a function
of the surface directly above it. A mesoscale eddy or a displaced thermocline is a pattern hundreds
of kilometres across, legible only from context, which the growing receptive field supplies. The
skip connections then restore the sharp local SST and SLA values that fix the mixed layer.

Training: AdamW at 3e-4, cosine decay, weight decay 1e-4, batch 8, 40 epochs, bf16 autocast, masked
MSE on the per-depth-normalised anomaly. **65 seconds** on one MIG slice.

## 7. Baselines

Scored through the identical code path, so the comparison is not confounded by scoring differences.

- **Climatology**, predict anomaly zero. The floor. A model that does not beat this has learned
  nothing the calendar did not already know.
- **Linear**, per-depth ordinary least squares on the 11 channels, pooled over cells. Deliberately
  has *no* spatial context: each cell sees only its own surface state. It exists to isolate how much
  of the U-Net's skill comes from context rather than from the temperature directly overhead.

## 8. Evaluation

Predictions are returned to °C (`× per-depth std + climatology`) before scoring. Per depth: RMSE,
bias, correlation, and **ACC**, correlation after removing the climatology from both prediction and
truth.

ACC is the metric to argue from. Plain correlation on absolute temperature sits near 0.9 for
anything that knows the seasonal cycle, so it flatters every model equally and separates none.

**Split discipline:** train 2019-2020, test 2021. Never a random-day split, consecutive days are
near-identical, and a random split manufactures ~0.99 correlation from autocorrelation alone.

## 9. Verification

15 checks across two self-check files, plus a third covering the operational
diagnostics. None need network or data:

- Regrid correctness at every real product registration, including a half-cell-offset grid
- Coordinates **bit-exact** against the target axes, not merely close: the failure mode requires
  exact equality, and an `allclose` check passed while the pipeline was broken
- Cross-product merge stays 100 × 240 with no NaN padding
- Land mask stays crisp: an isolated land speckle is absorbed by the block mean, two fully-land
  blocks give exactly two NaN cells
- Depth interpolation exact on a linear profile across the real 36 GLORYS levels
- Negative `elevation` axis flipped correctly
- Climatology keeps 366 days with the year-boundary wrap intact
- Static channels normalised on the right axes with the right phase
- U-Net preserves 100 × 240 through pad-and-crop
- The loss provably ignores everything outside the mask

## 10. Known limitations

- **Coastal NaN bleed on the target only.** GLORYS goes through interp, so it loses one target cell
  at the coast. Inputs do not. Fixable with a masked conservative regrid if coastal skill matters.
- **No per-channel validity mask.** NaN inputs are filled with the channel mean; gap-heavy days are
  not flagged to the network.
- **Two-year climatology.** Measurably biased against the test year, see
  [`RESULTS.md`](RESULTS.md). The CMEMS long-term monthly climatology is the fix.
- **Pre-2010 SSS is a reconstruction, not a satellite retrieval.** No satellite salinity sensor
  existed before SMOS, so the multi-observation product likely ingests in-situ data in early years, a mild leakage path when the target is also in-situ-informed. The PS names this product
  explicitly, but it should be disclosed rather than left for a reviewer to find.
- **GLORYS is a reanalysis that assimilates SST, altimetry and Argo.** Training against it is partly
  emulating a data-assimilation system, and gridded-Argo validation is therefore only
  semi-independent. Raw Argo profiles are the honest test, and that harness is not built yet.
