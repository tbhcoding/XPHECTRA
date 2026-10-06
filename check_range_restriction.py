"""
check_range_restriction.py
===========================

Measures what happens to the reported metrics when evaluation is restricted
to the pH range the cited reference scale actually covers.

WHY THIS EXISTS
---------------
The generator draws each sample's mean pH uniformly across 5.35 to 6.45 and
clips the field to 5.2 to 6.8. The reference scale cited in Chapter 2
(Sristi et al., 2025, Figure 1) runs only to 6.0. Measured on the 500-sample
test set, 43% of pixels and 43% of samples sit above that ceiling.

A panelist is entitled to ask what the headline looks like on the part of the
range the literature covers. This script answers that, so the answer is a
measured number rather than an assertion.

The expected result is that pooled R-squared FALLS while MAE IMPROVES.
R-squared is scale-relative: narrowing the range removes variance from the
denominator, so identical per-pixel accuracy scores lower. That is a property
of the metric, not a deterioration of the model, and reporting both together
is what makes the point defensible.

Restriction is applied at sample level on each sample's mean pH, which is the
quantity a grader would use to decide whether a cut falls in the normal range.

Reads committed checkpoints only. Trains nothing, writes no checkpoint.

Usage:
    python check_range_restriction.py
    python check_range_restriction.py --ceiling 6.2
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


def score(pairs):
    y = np.concatenate([t for t, _ in pairs])
    p = np.concatenate([q for _, q in pairs])
    return dict(
        n_samples=len(pairs),
        n_pixels=int(y.size),
        pooled_r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        mae=float(np.abs(y - p).mean()),
        rmse=float(np.sqrt(((y - p) ** 2).mean())),
        true_sd=float(y.std()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_test_extended")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ceiling", type=float, default=6.0,
                    help="upper bound of the cited reference scale")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--out", default="metric_check_outputs/range_restriction.json")
    a = ap.parse_args()

    root = os.path.join(a.data, a.split)
    if not os.path.isdir(root):
        raise SystemExit(
            "Dataset not found: {}\n\n"
            "Regenerate with:\n"
            "    python generate_dataset.py --n 500 --seed 777 --all-test "
            "--out ligtas_test_extended\n".format(root))

    print("evaluating {}".format(root))
    print("restriction: samples whose MEAN pH <= {}\n".format(a.ceiling))

    full_rows, rest_rows = [], []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = CrudeCRN(in_res=IN_RES, out_res=IN_RES)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        pairs = collect(m, root)
        kept = [(t, p) for t, p in pairs if t.mean() <= a.ceiling]
        full_rows.append(score(pairs))
        rest_rows.append(score(kept))
        print("  {:8s} full n={:3d} R2={:.4f} MAE={:.4f}   "
              "restricted n={:3d} R2={:.4f} MAE={:.4f}".format(
                  os.path.basename(os.path.dirname(ck)),
                  full_rows[-1]["n_samples"], full_rows[-1]["pooled_r2"],
                  full_rows[-1]["mae"],
                  rest_rows[-1]["n_samples"], rest_rows[-1]["pooled_r2"],
                  rest_rows[-1]["mae"]))

    def agg(rows, k):
        return float(np.mean([r[k] for r in rows]))

    print("\n{:14s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "", "n", "R2", "MAE", "RMSE", "true SD"))
    for lab, rows in (("full range", full_rows), ("restricted", rest_rows)):
        print("{:14s} {:>7.0f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            lab, agg(rows, "n_samples"), agg(rows, "pooled_r2"),
            agg(rows, "mae"), agg(rows, "rmse"), agg(rows, "true_sd")))

    dr2 = agg(rest_rows, "pooled_r2") - agg(full_rows, "pooled_r2")
    dmae = agg(rest_rows, "mae") - agg(full_rows, "mae")
    print("\n  R2  change on restriction : {:+.4f}".format(dr2))
    print("  MAE change on restriction : {:+.4f}  ({})".format(
        dmae, "improves" if dmae < 0 else "worsens"))
    print("\n  Reading: R-squared is scale-relative. Removing the upper part of")
    print("  the range removes variance from the denominator, so the same model")
    print("  scores lower while its per-pixel error is unchanged or better.")

    out = dict(
        data="{}/{}".format(a.data, a.split),
        ceiling=a.ceiling,
        ceiling_source="Sristi et al. 2025 Figure 1 upper anchor, DOI 10.55002/mr.5.3.117",
        restriction="samples whose mean pH <= ceiling",
        n_seeds=len(full_rows),
        full_range={k: agg(full_rows, k) for k in full_rows[0]},
        restricted={k: agg(rest_rows, k) for k in rest_rows[0]},
        delta_r2=dr2,
        delta_mae=dmae,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote {}".format(a.out))


if __name__ == "__main__":
    main()
