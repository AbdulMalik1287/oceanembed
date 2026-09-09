#!/usr/bin/env python
"""U-Net mapping surface fields to a subsurface temperature profile per cell.

The encoder compresses the surface state to a coarse, wide-channel bottleneck. This is the "satellite embedding" the problem statement asks for, and it can be
read out directly via ``forward(..., return_embedding=True)``. The decoder is
the reconstruction model. Skip connections carry the sharp local SST/SLA signal
past the bottleneck, which otherwise throws away exactly the detail that fixes
the mixed layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth
        widths = [width * 2**i for i in range(depth + 1)]

        self.down = nn.ModuleList()
        c = n_in
        for w in widths[:-1]:
            self.down.append(_block(c, w))
            c = w
        self.bottleneck = _block(c, widths[-1])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w_skip, w_in in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.up.append(nn.ConvTranspose2d(w_in, w_skip, 2, stride=2))
            self.dec.append(_block(w_skip * 2, w_skip))

        self.head = nn.Conv2d(widths[0], n_out, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # Pad to a multiple of 2**depth so the pooling is exact, then crop back.
        _, _, h, w = x.shape
        k = 2 ** self.depth
        ph, pw = (-h) % k, (-w) % k
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        embedding = x

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        out = self.head(x)[:, :, :h, :w]
        return (out, embedding) if return_embedding else out


def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over wet cells with finite truth only.

    Land is NaN in the target; letting it into the loss would either poison the
    gradient with NaN or, if zero-filled, teach the model to predict zeros over
    half the domain.
    """
    valid = mask & torch.isfinite(true)
    if not valid.any():
        return pred.sum() * 0.0
    diff = (pred - torch.nan_to_num(true))[valid]
    return (diff**2).mean()
