"""
check_calibration.py
=====================

Measures the CRN's calibration and what a post-hoc correction recovers.

WHY THIS EXISTS
---------------
The network systematically compresses its predictions toward the middle of
the pH range: regressing each sample's predicted mean pH on its true mean
gives a slope near 0.77 rather than 1.0, and prediction bias correlates
about -0.95 with true pH. That is regression to the mean -- a signature of
UNDER-fitting, which is the expected consequence of a deliberately small
network trained on four sparse points under a strong smoothness prior.

It matters in a specific way: the compression COSTS accuracy rather than
flattering it, so the uncorrected headline is conservative, not inflated.

THE CORRECTION, AND WHY IT IS SPLIT IN TWO
------------------------------------------
Two separate miscalibrations act in opposite directions:

  LEVEL   -- between samples, predictions are compressed (slope < 1).
             Corrected by regressing true sample mean on predicted sample
             mean. Fitting this needs only each calibration sample's MEAN
             pH, which sparse probe readings can supply. DEPLOYABLE.

  TEXTURE -- within a sample, spatial variation is over-expressed
             (predicted spatial SD is ~1.3x the true). Corrected by scaling
             deviations about the sample mean. Fitting this needs the
             within-sample SD of true pH, i.e. a dense ground-truth map,
             which a real deployment does not have. NOT DEPLOYABLE from
             sparse probes alone.

The ablation below reports each part separately, because the deployable and
non-deployable halves should not be quoted as one number.

NO LEAKAGE
----------
All correction parameters are fitted on the VALIDATION split and applied to
the held-out 500-sample test set, which influenced neither training nor
checkpoint selection nor the correction itself.

The network is NOT retrained. Checkpoints are read, never written.

Usage:
    python check_calibration.py
    python check_calibration.py --fit-data ligtas_synthetic_dataset --fit-split val \
                                --eval-data ligtas_test_extended --eval-split test
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
BATCH = 8


def collect(model, root):
    """Per-sample (true, predicted) arrays over tissue pixels, dataset order."""
    ds = SparseLIGTASDataset(root, N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    out = []
    with torch.no_grad():
        for x, _c, _v, ph, mask, _m in loader:
            p = model(x)
            if p.shape[-1] != IN_RES:
                p = F.interpolate(p.unsqueeze(1), size=IN_RES,
                                  mode="bilinear", align_corners=False).squeeze(1)
            for b in range(p.shape[0]):
                mk = mask[b]
                out.append((ph[b][mk].numpy(), p[b][mk].numpy()))
    return out


def fit_correction(pairs):
    """Level (slope, intercept) mapping predicted mean -> true mean, and texture scale."""
    tm = np.array([t.mean() for t, _ in pairs])
    pm = np.array([p.mean() for _, p in pairs])
    slope, intercept = np.polyfit(pm, tm, 1)
    t_sd = np.mean([t.std() for t, _ in pairs])
    p_sd = np.mean([p.std() for _, p in pairs])
    return float(slope), float(intercept), float(t_sd / p_sd)


def apply_correction(p, level=None, texture=None):
    mu = p.mean()
    new_mu = (level[1] + level[0] * mu) if level else mu
    dev = (p - mu) * (texture if texture else 1.0)
    return new_mu + dev


def score(pairs, level=None, texture=None):
    Y, P, per = [], [], []
    for t, p in pairs:
        q = apply_correction(p, level, texture)
        Y.append(t); P.append(q)
        per.append(1 - ((t - q) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    y = np.concatenate(Y); p = np.concatenate(P)
    per = np.array(per)
    return dict(
        r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        mae=float(np.abs(y - p).mean()),
        rmse=float(np.sqrt(((y - p) ** 2).mean())),
        per_sample_r2=float(per.mean()),
        pct_per_sample_positive=float((per > 0).mean() * 100),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-data", default="ligtas_synthetic_dataset")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-data", default="ligtas_test_extended")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--out", default="metric_check_outputs/calibration.json")
    a = ap.parse_args()

    fit_root = os.path.join(a.fit_data, a.fit_split)
    eval_root = os.path.join(a.eval_data, a.eval_split)
    for r in (fit_root, eval_root):
        if not os.path.isdir(r):
            raise SystemExit(
                f"Dataset not found: {r}\n\n"
                "Regenerate with:\n"
                "    python generate_dataset.py\n"
                "    python generate_dataset.py --n 500 --seed 777 --all-test "
                "--out ligtas_test_extended\n")

    print(f"fit on   {fit_root}")
    print(f"evaluate {eval_root}   (never seen by training, selection, or the correction)\n")

    variants = ["uncorrected", "level only", "texture only", "level + texture"]
    rows, params = {v: [] for v in variants}, []

    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = CrudeCRN(in_res=IN_RES, out_res=OUT_RES)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()

        slope, intercept, texture = fit_correction(collect(m, fit_root))
        ev = collect(m, eval_root)
        params.append(dict(checkpoint=ck, level_slope=slope,
                           level_intercept=intercept, texture_scale=texture))

        rows["uncorrected"].append(score(ev))
        rows["level only"].append(score(ev, level=(slope, intercept)))
        rows["texture only"].append(score(ev, texture=texture))
        rows["level + texture"].append(score(ev, level=(slope, intercept), texture=texture))
        print(f"  {os.path.basename(os.path.dirname(ck)):8s} "
              f"level slope={slope:.3f}  texture scale={texture:.3f}", flush=True)

    def agg(v, k):
        return float(np.mean([r[k] for r in rows[v]]))

    print(f"\n{'variant':17s} {'R2':>8s} {'MAE':>9s} {'RMSE':>9s} "
          f"{'per-sample R2':>14s} {'% samples +ve':>14s}")
    for v in variants:
        print(f"{v:17s} {agg(v,'r2'):>8.4f} {agg(v,'mae'):>9.4f} {agg(v,'rmse'):>9.4f} "
              f"{agg(v,'per_sample_r2'):>14.3f} {agg(v,'pct_per_sample_positive'):>13.1f}%")

    base, lvl, both = agg("uncorrected", "r2"), agg("level only", "r2"), agg("level + texture", "r2")
    share = 100 * (lvl - base) / (both - base) if both != base else float("nan")
    print(f"\n  Total R2 gain from the full correction : {both - base:+.4f}")
    print(f"  Share attributable to the LEVEL part   : {share:.0f}%"
          f"   <- the part a deployment could actually fit")
    print(f"  Share attributable to the TEXTURE part : {100 - share:.0f}%"
          f"   <- needs dense ground truth; not deployable")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(fit_on=fit_root, evaluated_on=eval_root,
                       per_checkpoint_parameters=params,
                       results={v: {k: agg(v, k) for k in rows[v][0]} for v in variants},
                       level_share_of_gain_pct=share), f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
