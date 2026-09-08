# Runbook — operating OceanEmbed without help

Everything needed to re-run, re-check or explain this project on your own.
Written 2026-09-08.

## The 60-second version

- Code: `~/oceanembed/` on the GPU box, reached as `ssh blackwell`. Same code in this repo.
- Data: `~/ocean/` on the box. **Full backup on your laptop at
  `C:\Users\abdul\Documents\Claude VS\sih-26066-data\` (7.8 GB, checksum-verified).**
- Python: `~/envs/ocean/bin/python`. Never the system python — it has none of the packages.
- One command for status: `ssh blackwell ~/oceanembed/status.sh`

## If someone asks you to prove it works

```
ssh blackwell
cd ~/oceanembed
~/envs/ocean/bin/python -m scripts.test_pipeline   # 10 checks, ~5 s, no data needed
~/envs/ocean/bin/python -m scripts.test_model      # 5 checks, ~20 s
~/envs/ocean/bin/python -m scripts.diagnostics     # TCHP/D26/MLD self-check
```

All three should end with a pass line. They need no network and no data, so they work even if the
box's disk is full or CMEMS is down. This is the safest thing to run in front of anyone.

## Re-running the whole result

Data is already built, so this is the short path:

```
cd ~/oceanembed
TR="2014 2015 2016 2017 2018 2019"; TE="2020 2021"
P=~/envs/ocean/bin/python

$P -m scripts.train climatology --train $TR --test $TE    # seconds
$P -m scripts.train linear      --train $TR --test $TE    # seconds
$P -m scripts.train unet        --train $TR --test $TE --epochs 250   # ~36 min
$P -m scripts.predict --train $TR --test 2021             # ~2 min
$P -m scripts.figures                                     # ~1 min
```

Results land in `~/ocean/runs/*.json`, figures in `~/ocean/figs/`.

**Expected numbers** (if they differ by more than ~0.01, something changed):

| | mean RMSE | @100 m | ACC @100 m |
|---|---:|---:|---:|
| climatology | 0.889 | 1.773 | — |
| linear | 0.786 | 1.485 | 0.555 |
| unet | 0.783 | 1.415 | 0.608 |

TCHP correlation 0.897 · D26 0.830 · MLD 0.741 · Argo mean RMSE 0.852 vs 0.900 climatology.

## Getting the figures onto your laptop

```
scp "blackwell:~/ocean/figs/*.png" .
```

They are also committed in the repo under `figs/`, so you do not need the box for the deck.

## If the box is unreachable

`ssh: connect to host ... Connection timed out` almost always means **your laptop is off the
college network**, not that the box is down. It happened twice on 6–7 September; both times the box
had been up for 17+ days. Check WiFi or VPN first.

Everything needed for the presentation — figures, docs, numbers — is in the GitHub repo and in the
local data folder. **You do not need the box to present.**

## If the disk fills up

The box's root volume runs at 90%+ and it is not mostly you. Your home is ~80 GB of which
OceanEmbed's data is 7.8 GB. Safe to delete, in order:

```
rm -rf ~/.cache/pip            # ~11 GB, zero risk
rm -rf ~/.vscode-server        # ~4.6 GB, re-downloads on next connect
rm -rf ~/ocean/pred            # 1.6 GB, regenerate with scripts.predict
rm -rf ~/ocean/argo_raw        # 525 MB, re-downloadable
```

**Do not delete `~/ocean/proc`** unless you are done for good — it is 8 years of built cubes and
rebuilding means re-downloading 77 GB, about 8 hours. A verified copy is on your laptop.

`~/.cache/huggingface` is 14 GB but belongs to your NLP capstone (Mistral-7B), not this project.

## The numbers to quote, and the ones not to

**Quote these:**
- +11.9% over climatology on two held-out years (against GLORYS)
- **+5.3% against 8,015 independent Argo profiles** — this is the honest headline
- +20.2% at 100 m; +8–11% at 75–200 m against Argo
- TCHP correlation 0.897
- 11 Argo profiles/day vs 11,759 reconstructed

**Do not quote** the old 3-year figures (13.7%, 22.9%, TCHP 0.858) — they came from a two-year
climatology that was measurably too weak a baseline. `~/ocean/runs_3yr/` keeps them only for the
comparison written up in RESULTS.md.

## Questions you will be asked, and where the answer lives

| Question | Answer |
|---|---|
| "GLORYS already assimilates Argo — isn't this circular?" | `RESULTS.md`, Argo section. Margin drops 11.9% → 5.3%; we inherit ~75% of GLORYS's +0.451 °C warm bias at 100 m. Say it before they do. |
| "Why not just use Argo?" | 11 profiles/day for the whole basin; 3.3/day in the Bay of Bengal; none at all on 8% of days. |
| "Why a U-Net and not a transformer?" | 2,192 training days. Train loss 0.069 vs val 0.233 — already overfitting, so more capacity widens the gap. Data-limited, not capacity-limited. |
| "It's worse than climatology below 300 m." | Correct, and reported. Surface fields carry almost no information about 500–1000 m. An operational product should serve climatology below 300 m. |
| "How fast is it?" | 36 min to train on 8 years; inference is seconds. The constraint is bandwidth, never compute. |

Full script in `SUBMISSION.md` under "Anticipated questions".

## What is deliberately not done

- **Thermocline bias correction** — the largest remaining error (+0.603 °C at 100 m against Argo).
  Fine-tuning on Argo profiles, or a depth-dependent correction, is the obvious next work.
- **Uncertainty estimates** — the model gives point predictions with no error bars. A second output
  head with a Gaussian NLL loss would fix this, and the Argo set is already there to check
  calibration against.
- **Near-real-time demo** — every input has an NRT twin, so the pipeline could produce *today's*
  ocean. This is the strongest deployability argument and it is not built.
- **ViT / GNN ablations** — the PS lists them as candidate architectures.

## Repo layout

```
scripts/
  config.py       region, grid, depths, dataset ids
  pipeline.py     fetch / regrid / stage
  data.py         splits, climatology, normalisation, land mask
  model.py        U-Net + masked loss
  train.py        climatology | linear | unet, one scoring path
  predict.py      standardized netCDF output + diagnostics
  diagnostics.py  TCHP, D26, MLD, OHC
  argo.py         independent validation
  figures.py      all six figures
  metrics.py      RMSE / bias / correlation / ACC
  test_*.py       15 checks, no data needed
  status.sh       one-command dashboard
docs/
  RESULTS.md      every number, with the caveats
  METHODOLOGY.md  how it works, stage by stage
  SUBMISSION.md   slide-by-slide content + speaker notes
  PLAN.md         budgets and decisions
  PS.md           the problem statement
  RUNBOOK.md      this file
```

## Credentials

Copernicus Marine only, stored at `~/.copernicusmarine/.copernicusmarine-credentials` (mode 600).
Re-run `~/envs/ocean/bin/copernicusmarine login` if it ever stops working. No NASA Earthdata
account is needed. Nothing in the repo contains a secret.
