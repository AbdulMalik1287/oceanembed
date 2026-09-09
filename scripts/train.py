#!/usr/bin/env python
"""Train and score a reconstruction model. One entry point for every model, so
the numbers in the comparison table are produced by identical code paths.

    python -m scripts.train climatology --train 2019 2020 --test 2021
    python -m scripts.train linear      --train 2019 2020 --test 2021
    python -m scripts.train unet        --train 2019 2020 --test 2021 --epochs 40

Climatology is the floor. A model that does not beat it has learned nothing that
the calendar did not already know.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from .config import PROC
from .metrics import skill, table

OUT = PROC.parent / "runs"


def _report(name, pred_norm, split, stats, extra=None):
    """Score in degrees C and write the run out."""
    pred = D.to_celsius(pred_norm, split.y_clim, stats["y_sd"])
    true = D.to_celsius(split.y, split.y_clim, stats["y_sd"])
    rows = skill(pred, true, split.y_clim, stats["mask"], stats["depths"])
    print(table(rows, f"{name}  (test n={len(split)} days)"))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as fh:
        json.dump(dict(model=name, rows=rows, **(extra or {})), fh, indent=2)
    return rows


# --------------------------------------------------------------------------- #
def run_climatology(train, test, stats):
    """Predict the climatology exactly: anomaly zero everywhere."""
    return _report("climatology", np.zeros_like(test.y), test, stats)


def run_linear(train, test, stats, max_rows=2_000_000, seed=0):
    """Per-depth ordinary least squares on the input channels, pooled over cells.

    The classical comparison. Deliberately has no spatial context, it sees each
    cell's surface state alone, which is exactly the limitation the U-Net's
    receptive field is supposed to fix.
    """
    rng = np.random.default_rng(seed)
    mask = stats["mask"]
    n, c = train.x.shape[0], train.x.shape[1]

    X = train.x.transpose(0, 2, 3, 1)[:, mask].reshape(-1, c)
    Yall = train.y.transpose(0, 2, 3, 1)[:, mask].reshape(-1, train.y.shape[1])
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, Yall = X[idx], Yall[idx]
    X = np.concatenate([X, np.ones((len(X), 1), dtype="float32")], axis=1)

    ok = np.isfinite(Yall).all(axis=1) & np.isfinite(X).all(axis=1)
    coef, *_ = np.linalg.lstsq(X[ok].astype("float64"), Yall[ok].astype("float64"), rcond=None)

    Xt = test.x.transpose(0, 2, 3, 1).reshape(-1, c)
    Xt = np.concatenate([Xt, np.ones((len(Xt), 1), dtype="float32")], axis=1)
    pred = (Xt @ coef).astype("float32").reshape(
        test.x.shape[0], *test.x.shape[2:], test.y.shape[1]
    ).transpose(0, 3, 1, 2)
    return _report("linear", pred, test, stats, dict(n_fit=int(ok.sum())))


def run_unet(train, test, stats, epochs=40, batch=8, lr=3e-4, width=32, seed=0):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .model import UNet, masked_mse

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mask_t = torch.from_numpy(stats["mask"]).to(dev)

    # Validation split used ONLY to choose the epoch count, never the test year,
    # otherwise "we trained longer" is just tuning on the answer.
    #
    # Interleaved 10-day blocks, every 5th block held out. Two failure modes are
    # being avoided at once:
    #   - a random day split leaks, because consecutive ocean days are nearly
    #     identical;
    #   - taking the last 15% chronologically hands you one contiguous season
    #     (Sep-Dec here), so the model is judged on a regime it barely trained
    #     on. Measured: that split picked epoch 2 of 250, and the model it chose
    #     was worse on the test year than one trained 20x longer.
    # Blocks keep adjacent days together while spreading validation across every
    # season. Residual leakage is the two boundary days per block.
    block = np.arange(len(train.x)) // 10
    is_val = (block % 5) == 4
    xtr, ytr = train.x[~is_val], train.y[~is_val]
    xva = torch.from_numpy(train.x[is_val]).to(dev)
    yva = torch.from_numpy(train.y[is_val]).to(dev)
    n_val = int(is_val.sum())
    print(f"  split: {len(xtr)} train days, {n_val} val days "
          f"(every 5th 10-day block, all seasons covered)")

    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    net = UNet(train.x.shape[1], train.y.shape[1], width=width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"UNet {n_par/1e6:.2f}M params on {dev}, {len(ds)} train days")

    def val_loss() -> float:
        net.eval()
        tot = k = 0.0
        with torch.no_grad():
            for i in range(0, len(xva), batch):
                tot += float(masked_mse(net(xva[i:i + batch]).float(),
                                        yva[i:i + batch], mask_t))
                k += 1
        return tot / max(k, 1)

    t0 = time.time()
    best = (float("inf"), -1, None)
    for ep in range(epochs):
        net.train()
        tot = k = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            # bf16, not fp16: this card is sm_120 and bf16 avoids loss scaling.
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = masked_mse(net(xb).float(), yb, mask_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); k += 1
        sched.step()

        vl = val_loss()
        if vl < best[0]:
            best = (vl, ep, {k2: v.detach().clone() for k2, v in net.state_dict().items()})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train {tot/k:.4f}  val {vl:.4f}  {time.time()-t0:5.0f}s")

    print(f"  best val {best[0]:.4f} at epoch {best[1]} of {epochs}")
    if best[2] is not None:
        net.load_state_dict(best[2])          # test the selected model, not the last one

    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test.x), batch):
            xb = torch.from_numpy(test.x[i:i + batch]).to(dev)
            preds.append(net(xb).float().cpu().numpy())
    pred = np.concatenate(preds)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT / "unet.pt")
    return _report("unet", pred, test, stats,
                   dict(params=n_par, epochs=epochs, best_epoch=best[1],
                        best_val_loss=best[0], val_days=n_val,
                        seconds=round(time.time() - t0)))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=["climatology", "linear", "unet"])
    ap.add_argument("--train", type=int, nargs="+", default=[2019, 2020])
    ap.add_argument("--test", type=int, nargs="+", default=[2021])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=32)
    a = ap.parse_args()

    print(f"train years {a.train}  test years {a.test}")
    train, test, stats = D.build(a.train, a.test)
    print(f"x {train.x.shape}  y {train.y.shape}  wet cells {int(stats['mask'].sum())}")

    if a.model == "climatology":
        run_climatology(train, test, stats)
    elif a.model == "linear":
        run_linear(train, test, stats)
    else:
        run_unet(train, test, stats, epochs=a.epochs, width=a.width)


if __name__ == "__main__":
    main()
