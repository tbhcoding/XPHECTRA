"""
Crude baseline CRN (Convolutional Regression Network) for LIGTAS-pH
====================================================================

Predicts a dense per-pixel pH map from a 6-band multispectral cube.
Trained (see train_crn.py) using ONLY the sparse probe labels per sample
plus a total-variation smoothness prior -- the full dense map
(`*_phtrue.npy`) stays hidden during training and is used solely for
evaluation-time sanity checking.

Priority right now is "does sparse supervision + a smoothness prior even
work on our own synthetic data" -- not architecture quality. This is
deliberately a small U-Net: two downsamples, two upsamples, skip
connections. Get this crude version training end-to-end before spending
time on anything fancier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CrudeCRN(nn.Module):
    """
    Input:  (B, 6, in_res, in_res) multispectral cube.
    Output: (B, out_res, out_res)  dense pH prediction (pH units).

    The network always decodes back to full input resolution internally,
    then an adaptive average pool maps down to `out_res` if a coarser
    prediction grid is requested. A coarser cell is therefore the
    area-averaged pH over that cell -- the physically sensible reading of
    "predict pH on a coarser grid" (see PART B: out_res is a parameter,
    not hardcoded to full resolution).
    """

    def __init__(self, in_bands=6, base=16, in_res=256, out_res=256, ph_prior=5.9):
        super().__init__()
        if out_res > in_res:
            raise ValueError(f"out_res ({out_res}) cannot exceed in_res ({in_res})")
        self.in_res = in_res
        self.out_res = out_res

        self.enc1 = conv_block(in_bands, base)        # in_res
        self.enc2 = conv_block(base, base * 2)         # in_res/2
        self.enc3 = conv_block(base * 2, base * 4)     # in_res/4

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)
        # pH lives in ~5.2-6.8, nowhere near a default near-zero conv output.
        # Bias-init the head to the physiological midpoint so training starts
        # in-range and spends its (few, crude-pass) epochs learning spatial
        # detail instead of first learning a large constant offset.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, ph_prior)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        full = self.head(d1).squeeze(1)  # (B, in_res, in_res)

        if self.out_res == self.in_res:
            return full
        return F.adaptive_avg_pool2d(full.unsqueeze(1), self.out_res).squeeze(1)
