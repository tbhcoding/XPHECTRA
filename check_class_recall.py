"""
check_class_recall.py
======================

Per-class recall before and after the level correction.

WHY THIS EXISTS
---------------
Overall quality-class accuracy (87.7%) is carried by the most common class.
DFD occupies about 63% of tissue pixels and is recovered at 96%, which hides
how the model performs on PSE -- the defect the study is motivated by, and
only about 6% of pixels. Reporting one accuracy figure for all three classes
therefore understates a real weakness.

The weakness is not an inability to detect the condition. It is the
between-sample compression measured by check_calibration.py: predictions are
pulled toward the centre of the pH range, and the lowest-pH pixels are the
first to cross a class boundary when that happens. PSE sits at the bottom of
the range, so it is affected first and worst.

This script measures both halves of that claim: the per-class recall, and
what the level correction recovers. The correction is the deployable one --
fitted from the mean of each calibration sample's four probe readings, never
from the dense ground-truth map (see check_calibration.py for why the two
agree to within 0.0001 in R-squared).

NO LEAKAGE
----------
Correction parameters are fitted on the VALIDATION split. Recall is scored on
the held-out 500-sample set, which influenced neither training, nor
checkpoint selection, nor the fit. The network is not retrained; checkpoints
are read, never written.

CLASS BOUNDARIES
----------------
5.40 and 5.80 pH, the midpoints between the 5.2 / 5.6 / 6.0 anchors of the
scale cited in Chapter 2 (Sristi et al., 2025), which gives anchors rather
than limits. check_class_thresholds.py reports how overall accuracy moves
under three other conventions; this script holds the adopted one fixed,
because its question is which CLASS is missed, not where the lines fall.

Usage:
    python check_class_recall.py
    python check_class_recall.py --lower 5.5 --upper 6.1
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
CLASSES = ["PSE", "normal", "DFD"]


def collect(model, root):
    """Per-sample (true, predicted, probe values) over tissue pixels."""
    ds = SparseLIGTASDataset(root, N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    out = []
    with torch.no_grad():
        for x, _coords, values, ph, mask, _meta in loader:
            p = model(x)
            if p.shape[-1] != IN_RES:
                p = F.interpolate(p.unsqueeze(1), size=IN_RES,
                                  mode="bilinear", align_corners=False).squeeze(1)
            for b in range(p.shape[0]):
                mk = mask[b]
                out.append((ph[b][mk].numpy(), p[b][mk].numpy(), values[b].numpy()))
    return out


def fit_level(pairs):
    """Slope/intercept mapping predicted sample mean -> true sample mean,
    estimated from the four probe readings only (what a deployment has)."""
    pm = np.array([p.mean() for _, p, _ in pairs])
    probe = np.array([v.mean() for _, _, v in pairs])
    slope, intercept = np.polyfit(pm, probe, 1)
    return float(slope), float(intercept)


def recall(pairs, edges, level=None):
    """Per-class recall and overall accuracy over all tissue pixels."""
    Y, P = [], []
    for t, p, _ in pairs:
        if level is None:
            q = p
        else:
            mu = p.mean()
            q = (level[1] + level[0] * mu) + (p - mu)
        Y.append(t); P.append(q)
    y = np.concatenate(Y); q = np.concatenate(P)
    cy = np.digitize(y, edges); cq = np.digitize(q, edges)
    rec = [float((cq[cy == k] == k).mean() * 100) if (cy == k).any() else float("nan")
           for k in range(3)]
    shares = [float((cy == k).mean() * 100) for k in range(3)]
    return rec, float((cq == cy).mean() * 100), shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-data", default="ligtas_synthetic_dataset")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval-data", default="ligtas_test_extended")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--lower", type=float, default=5.40)
    ap.add_argument("--upper", type=float, default=5.80)
    ap.add_argument("--out", default="metric_check_outputs/class_recall.json")
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

    edges = [a.lower, a.upper]
    print(f"fit on   {fit_root}   (level correction, from probe readings only)")
    print(f"evaluate {eval_root}")
    print(f"class boundaries {a.lower} / {a.upper} pH\n")

    rows, params, shares = [], [], None
    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = CrudeCRN(in_res=IN_RES, out_res=OUT_RES)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        slope, intercept = fit_level(collect(m, fit_root))
        ev = collect(m, eval_root)
        before, acc_b, shares = recall(ev, edges)
        after, acc_a, _ = recall(ev, edges, level=(slope, intercept))
        rows.append(dict(checkpoint=ck, level_slope=slope, level_intercept=intercept,
                         recall_before_pct=before, recall_after_pct=after,
                         overall_accuracy_before_pct=acc_b,
                         overall_accuracy_after_pct=acc_a))
        print(f"  {os.path.basename(os.path.dirname(ck)):8s} "
              f"PSE {before[0]:5.1f}% -> {after[0]:5.1f}%   "
              f"normal {before[1]:5.1f}% -> {after[1]:5.1f}%   "
              f"DFD {before[2]:5.1f}% -> {after[2]:5.1f}%", flush=True)

    def agg(key, idx=None):
        v = [r[key][idx] if idx is not None else r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    print(f"\n{'class':8s} {'true share':>11s} {'recall before':>14s} {'recall after':>13s}")
    summary = {}
    for k, name in enumerate(CLASSES):
        b, bs = agg("recall_before_pct", k)
        aft, afs = agg("recall_after_pct", k)
        summary[name] = dict(true_share_pct=shares[k],
                             recall_before_pct=[b, bs], recall_after_pct=[aft, afs])
        print(f"{name:8s} {shares[k]:10.1f}% {b:12.1f}% {aft:12.1f}%")
    ab, abs_ = agg("overall_accuracy_before_pct")
    aa, aas_ = agg("overall_accuracy_after_pct")
    print(f"{'overall':8s} {'':11s} {ab:12.1f}% {aa:12.1f}%")
    pse_lo = min(r["recall_before_pct"][0] for r in rows)
    pse_hi = max(r["recall_before_pct"][0] for r in rows)
    print(f"\n  PSE recall before correction ranges {pse_lo:.1f}% to {pse_hi:.1f}% across seeds.")
    print(f"  The level correction raises the mean from {summary['PSE']['recall_before_pct'][0]:.1f}% "
          f"to {summary['PSE']['recall_after_pct'][0]:.1f}%,")
    print(f"  and overall accuracy from {ab:.1f}% to {aa:.1f}%.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(fit_on=fit_root, evaluated_on=eval_root,
                       class_boundaries=edges, n_seeds=len(rows),
                       correction="level only, fitted from the mean of four probe readings",
                       summary=summary,
                       overall_accuracy_before_pct=[ab, abs_],
                       overall_accuracy_after_pct=[aa, aas_],
                       pse_recall_before_range_pct=[pse_lo, pse_hi],
                       per_checkpoint=rows), f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
