"""
check_spatial_skill.py
=======================

Separates two questions the single reported R² currently conflates:

  (a) "Can the CRN measure pH from an image?"      -> POOLED R²
      One R² over all val meat pixels, SS_tot against the GRAND mean.
      Rewards getting each sample's overall pH LEVEL right.

  (b) "Can the CRN reconstruct the spatial field?" -> PER-SAMPLE R²
      R² computed within each sample, SS_tot against THAT SAMPLE's own
      mean, then averaged over samples. Rewards only within-sample
      spatial structure -- which is the project's actual stated claim
      (see CLAUDE.md: "can the CRN reconstruct a full field from 4
      points using spectral shape alone?").

These can disagree sharply, because between-sample pH spread is several
times larger than within-sample spread, so (a) is dominated by level.

Also reports:

  - LEVEL-ONLY REFERENCE: predict a constant = the mean of that sample's
    4 sparse pH points, everywhere. NOTE: this is an ORACLE reference,
    NOT a competing method -- it uses the sparse pH values at test time,
    which `CrudeCRN` does NOT receive (`model(x)` takes the 6-band cube
    only; coords/values enter solely through sparse_loss). It exists to
    quantify how much of pooled R² is attributable to level alone.
    Do not report it as a baseline the CRN "loses to".

  - SPATIAL DIAGNOSTICS: within-sample true vs predicted spatial std,
    their ratio, the within-sample correlation, and the level error.
    These distinguish "over-smoothed" (pred std << true std) from
    "over-amplified" (pred std >> true std) from "no signal"
    (correlation ~ 0).

Reports numbers only. Does not modify PARAMS, CrudeCRN, or train_crn.py.

Usage:
    python check_spatial_skill.py [--data DIR] [--ckpt-glob GLOB] [--out JSON]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_crn import SparseLIGTASDataset
from crn_model import CrudeCRN

IN_RES = OUT_RES = 256
N_POINTS = 4
BATCH_SIZE = 8


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def collect(model, loader):
    """Returns per-sample (true, pred) arrays over meat pixels, in dataset order."""
    out = []
    with torch.no_grad():
        for x, coords, values, ph_dense, mask, _ in loader:
            pred = model(x)
            if pred.shape[-1] != IN_RES:
                pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                     mode="bilinear", align_corners=False).squeeze(1)
            for b in range(pred.shape[0]):
                m = mask[b]
                out.append((ph_dense[b][m].numpy(), pred[b][m].numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--ckpt-glob", default="crn_check/seed_*/crn_best.pt")
    ap.add_argument("--out", default="metric_check_outputs/spatial_skill.json")
    a = ap.parse_args()

    ds = SparseLIGTASDataset(os.path.join(a.data, "val"), N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- variance decomposition: why pooled and per-sample can disagree -----
    sample_means, within_stds, true_all = [], [], []
    for i in range(len(ds)):
        _, _, _, ph, m, _ = ds[i]
        y = ph[m].numpy()
        sample_means.append(y.mean()); within_stds.append(y.std()); true_all.append(y)
    between_std = float(np.std(sample_means))
    within_std = float(np.mean(within_stds))

    # --- level-only ORACLE reference (see module docstring) -----------------
    ys, flat = [], []
    for i in range(len(ds)):
        _, _, values, ph, m, _ = ds[i]
        y = ph[m].numpy()
        ys.append(y); flat.append(np.full_like(y, values.numpy().mean()))
    y_cat = np.concatenate(ys); f_cat = np.concatenate(flat)
    oracle = dict(pooled_r2=r2(y_cat, f_cat),
                  mae=float(np.abs(y_cat - f_cat).mean()),
                  rmse=float(np.sqrt(((y_cat - f_cat) ** 2).mean())))

    print(f"Val set: {len(ds)} samples")
    print(f"  between-sample pH std (sample means) : {between_std:.4f}")
    print(f"  within-sample  pH std (mean/sample)  : {within_std:.4f}")
    print(f"  ratio                                 : {between_std / within_std:.2f}x")
    print(f"\nLEVEL-ONLY ORACLE (mean of the 4 sparse points; ORACLE -- CRN never sees these at test time):")
    print(f"  pooled R2={oracle['pooled_r2']:.4f}  MAE={oracle['mae']:.4f}  RMSE={oracle['rmse']:.4f}")
    print(f"  -> {100 * oracle['pooled_r2']:.1f}% of pooled variance is explainable by LEVEL alone.\n")

    rows = []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        model = CrudeCRN(in_res=IN_RES, out_res=OUT_RES)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()
        per = collect(model, loader)

        y_all = np.concatenate([t for t, _ in per])
        p_all = np.concatenate([p for _, p in per])
        per_r2 = np.array([r2(t, p) for t, p in per])
        t_std = np.array([t.std() for t, _ in per])
        p_std = np.array([p.std() for _, p in per])
        corr = np.array([np.corrcoef(t, p)[0, 1] for t, p in per])
        lvl = np.array([abs(p.mean() - t.mean()) for t, p in per])

        row = dict(
            checkpoint=ck,
            pooled_r2=r2(y_all, p_all),
            mae=float(np.abs(y_all - p_all).mean()),
            rmse=float(np.sqrt(((y_all - p_all) ** 2).mean())),
            per_sample_r2_mean=float(per_r2.mean()),
            per_sample_r2_median=float(np.median(per_r2)),
            per_sample_r2_negative_count=int((per_r2 < 0).sum()),
            n_samples=len(per),
            true_spatial_std=float(t_std.mean()),
            pred_spatial_std=float(p_std.mean()),
            amplitude_ratio=float(p_std.mean() / t_std.mean()),
            within_sample_corr=float(corr.mean()),
            level_error=float(lvl.mean()),
            # Best per-sample R2 reachable by optimally rescaling this model's
            # spatial pattern: r^2. An upper bound, not a promise.
            per_sample_r2_ceiling_if_rescaled=float((corr.mean()) ** 2),
        )
        rows.append(row)
        print(f"{ck}")
        print(f"  pooled R2={row['pooled_r2']:.4f}  MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}")
        print(f"  per-sample R2 mean={row['per_sample_r2_mean']:+.4f} "
              f"median={row['per_sample_r2_median']:+.4f}  "
              f"negative on {row['per_sample_r2_negative_count']}/{row['n_samples']}")
        print(f"  spatial: true_std={row['true_spatial_std']:.4f} "
              f"pred_std={row['pred_spatial_std']:.4f} "
              f"({row['amplitude_ratio']:.2f}x)  corr={row['within_sample_corr']:+.4f}  "
              f"level_err={row['level_error']:.4f}")

    agg = {}
    for k in ("pooled_r2", "mae", "rmse", "per_sample_r2_mean", "amplitude_ratio",
              "within_sample_corr", "per_sample_r2_ceiling_if_rescaled"):
        v = np.array([r[k] for r in rows])
        agg[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        print(f"\n{k:38s} mean={v.mean():+.4f} std={agg[k]['std']:.4f}" if k == "pooled_r2"
              else f"{k:38s} mean={v.mean():+.4f} std={agg[k]['std']:.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(between_sample_ph_std=between_std, within_sample_ph_std=within_std,
                       level_only_oracle=oracle, per_checkpoint=rows, aggregate=agg), fh, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
