"""
check_calibration_probe.py
===========================

Refits the level calibration the way a deployment would have to: from the
four sparse probe readings per sample, not from the true dense mean.

WHY THIS EXISTS
---------------
`check_calibration.py` fits the level correction by regressing each
calibration sample's TRUE DENSE mean pH on its predicted mean. Chapter 3
describes that correction as fittable from sparse probe readings, which a
real deployment could supply. Those are not the same thing: the dense mean
is exact, while four probe readings give a noisy estimate of it.

A reviewer raised this, and the objection is correct as the original script
was written. This script answers it directly by substituting the mean of the
four committed probe readings for the dense mean at fit time. Everything else
is unchanged: parameters are still fitted on the validation split and applied
to the held-out 500-sample test set, and the network is not retrained.

If the corrected R-squared survives that substitution, the claim in Chapter 3
holds and is now measured rather than assumed. If it does not, the figure is
a ceiling and must be reported as one.

Reads committed checkpoints and the committed sparse labels only.

Usage:
    python check_calibration_probe.py
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

IN_RES = 256
N_POINTS = 4
BATCH = 8


def collect(model, root):
    """Returns per-sample (true dense, predicted dense, probe-reading mean)."""
    ds = SparseLIGTASDataset(root, N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    out, idx = [], 0
    ids = sorted({f.split("_msi.npy")[0] for f in os.listdir(root)
                  if f.endswith("_msi.npy")})
    with torch.no_grad():
        for x, _c, values, ph, mask, _m in loader:
            p = model(x)
            if p.shape[-1] != IN_RES:
                p = F.interpolate(p.unsqueeze(1), size=IN_RES,
                                  mode="bilinear", align_corners=False).squeeze(1)
            for b in range(p.shape[0]):
                mk = mask[b]
                # `values` holds the four committed probe pH readings for this
                # sample. Their mean is all a physical deployment would have.
                out.append((ph[b][mk].numpy(), p[b][mk].numpy(),
                            float(values[b].mean())))
                idx += 1
    return out


def fit_level(pairs, use_probe):
    """Regress true sample mean on predicted sample mean.

    use_probe=True  -> the target is the mean of the four probe readings,
                       which is what a deployment could measure.
    use_probe=False -> the target is the true dense mean, which it could not.
    """
    pm = np.array([p.mean() for _, p, _ in pairs])
    tm = np.array([(probe if use_probe else t.mean()) for t, _, probe in pairs])
    slope, intercept = np.polyfit(pm, tm, 1)
    return float(slope), float(intercept)


def score(pairs, level):
    Y, P = [], []
    for t, p, _ in pairs:
        mu = p.mean()
        Y.append(t)
        P.append((level[1] + level[0] * mu) + (p - mu))
    y, q = np.concatenate(Y), np.concatenate(P)
    return dict(
        r2=float(1 - ((y - q) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        mae=float(np.abs(y - q).mean()),
        rmse=float(np.sqrt(((y - q) ** 2).mean())),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-data", default="ligtas_synthetic_dataset")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-data", default="ligtas_test_extended")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--out", default="metric_check_outputs/calibration_probe.json")
    a = ap.parse_args()

    fit_root = os.path.join(a.fit_data, a.fit_split)
    eval_root = os.path.join(a.eval_data, a.eval_split)
    for r in (fit_root, eval_root):
        if not os.path.isdir(r):
            raise SystemExit("Dataset not found: %s\n\nRegenerate with:\n"
                             "    python generate_dataset.py\n"
                             "    python generate_dataset.py --n 500 --seed 777 "
                             "--all-test --out ligtas_test_extended\n" % r)

    print("fit on   %s   (level target varied)" % fit_root)
    print("evaluate %s\n" % eval_root)

    rows = {"uncorrected": [], "level from dense mean": [], "level from 4 probes": []}
    params = []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = CrudeCRN(in_res=IN_RES, out_res=IN_RES)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        fit_pairs = collect(m, fit_root)
        ev = collect(m, eval_root)

        dense = fit_level(fit_pairs, use_probe=False)
        probe = fit_level(fit_pairs, use_probe=True)
        params.append(dict(checkpoint=ck,
                           dense_slope=dense[0], dense_intercept=dense[1],
                           probe_slope=probe[0], probe_intercept=probe[1]))

        rows["uncorrected"].append(score(ev, (1.0, 0.0)))
        rows["level from dense mean"].append(score(ev, dense))
        rows["level from 4 probes"].append(score(ev, probe))
        print("  %-8s dense slope %.3f | probe slope %.3f"
              % (os.path.basename(os.path.dirname(ck)), dense[0], probe[0]))

    def agg(v, k):
        return float(np.mean([r[k] for r in rows[v]]))

    print("%-24s %8s %9s %9s" % ("fit source", "R2", "MAE", "RMSE"))
    for v in rows:
        print("%-24s %8.4f %9.4f %9.4f"
              % (v, agg(v, "r2"), agg(v, "mae"), agg(v, "rmse")))

    d = agg("level from dense mean", "r2") - agg("level from 4 probes", "r2")
    print("\n  R2 lost by using probe readings instead of the dense mean: %.4f" % d)
    print("  Reading: if this gap is small, the correction is genuinely")
    print("  deployable and Chapter 3's claim stands as written.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(fit_on=fit_root, evaluated_on=eval_root,
                       n_seeds=len(params), per_checkpoint_parameters=params,
                       results={v: {k: agg(v, k) for k in rows[v][0]} for v in rows},
                       r2_lost_using_probes=d), f, indent=2)
    print("\nWrote %s" % a.out)


if __name__ == "__main__":
    main()
