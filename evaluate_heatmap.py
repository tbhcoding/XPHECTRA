"""
evaluate_heatmap.py
====================

Final evaluation of the predicted pH heatmap -- the single script to run
for Chapter 4 numbers. Scores already-trained checkpoints; does not train,
does not touch PARAMS, CrudeCRN, or train_crn.py.

WHY THIS EXISTS
---------------
Pooled R^2 alone does not describe a heatmap well, in either direction:

  - It FLATTERS, because between-sample pH spread (SD ~0.30) is ~3.6x
    larger than within-sample spread (SD ~0.084), so a model that only
    gets each sample's overall LEVEL right already scores highly.
  - Per-sample R^2 PUNISHES, because it measures against each sample's own
    tiny internal variance -- errors that are irrelevant at the scale meat
    science cares about still drive it sharply negative.

So this script reports three families of number together, and deliberately
prints them in ONE block so the favourable ones cannot be quoted without
the limitation beside them:

  1. Standard regression metrics : pooled R^2, MAE, RMSE
  2. Practical heatmap accuracy  : % of pixels within a pH tolerance
                                   (THRESHOLD-FREE -- report these), with
                                   quality-class agreement demoted to 2b as
                                   weaker, caveated evidence
  3. Spatial honesty            : per-sample R^2, within-sample correlation,
                                   and the amplitude ratio that explains it

WHICH NUMBER TO PUT IN THE MANUSCRIPT
-------------------------------------
Lead with the tolerance bands, e.g. "82.8% of test pixels fall within
+/-0.15 pH of ground truth". They depend on no class definition and no
class balance, so there is no interpretive choice in them to challenge.

The quality-class figure is more evocative for a meat-science audience but
is weaker evidence twice over -- the boundaries are interpolated and the
class mix follows the sampling design. Use it as support, never as the
headline, or omit it with --no-class.

QUALITY-CLASS THRESHOLDS -- READ BEFORE QUOTING
-----------------------------------------------
Defaults are 5.40 and 5.80, DERIVED from the manuscript's own Figure 1
(DOI: 10.55002/mr.5.3.117), which anchors pH 5.2 = PSE, 5.6 = normal,
6.0 = DFD. Note what that figure does and does not give:

  - It GIVES three labelled anchor points on a continuum.
  - It does NOT give class boundaries.

5.40 and 5.80 are the MIDPOINTS between those anchors -- a reasonable
interpolation, but an interpolation, not a value the source states. Say so
if asked; do not present the edges themselves as cited.

Class proportions also inherit the generator's sampling design, which draws
between-sample pH uniformly over 5.35-6.45 for even coverage rather than
imitating the population distribution of commercial pork. Under the edges
above that puts ~63% of pixels in the DFD class, so a classifier that
always guessed DFD would already score ~63%. The script therefore reports
the MAJORITY-CLASS BASELINE next to the accuracy, and the lift over it --
quote the lift, not the raw accuracy alone.

The tolerance-band figures depend on no threshold and no class balance, so
they are the safer number if the boundaries are ever challenged.

Usage:
    python evaluate_heatmap.py
    python evaluate_heatmap.py --ckpt-glob "crn_5seed_final/seed_*/crn_best.pt"
    python evaluate_heatmap.py --class-edges 5.60 6.00
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
TOLERANCES = [0.05, 0.10, 0.15, 0.20]
CLASS_NAMES = ["PSE (low)", "normal", "DFD (high)"]


def pooled_r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def score_checkpoint(ckpt, loader, class_edges):
    model = CrudeCRN(in_res=IN_RES, out_res=OUT_RES)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    per_sample = []
    with torch.no_grad():
        for x, coords, values, ph_dense, mask, _ in loader:
            pred = model(x)
            if pred.shape[-1] != IN_RES:
                pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                     mode="bilinear", align_corners=False).squeeze(1)
            for b in range(pred.shape[0]):
                m = mask[b]
                per_sample.append((ph_dense[b][m].numpy(), pred[b][m].numpy()))

    y = np.concatenate([t for t, _ in per_sample])
    p = np.concatenate([q for _, q in per_sample])
    err = np.abs(y - p)

    # 1. standard regression metrics
    out = dict(
        checkpoint=ckpt,
        pooled_r2=pooled_r2(y, p),
        mae=float(err.mean()),
        rmse=float(np.sqrt(((y - p) ** 2).mean())),
    )

    # 2. practical heatmap accuracy
    out["within_tolerance_pct"] = {f"{t:.2f}": float(100 * (err <= t).mean())
                                   for t in TOLERANCES}
    y_cls = np.digitize(y, class_edges)
    p_cls = np.digitize(p, class_edges)
    n_cls = len(class_edges) + 1
    counts = np.bincount(y_cls, minlength=n_cls)
    # A classifier that always guessed the commonest true class would score
    # this. Report the LIFT over it, never the raw accuracy alone -- the
    # class balance here is a product of the generator's sampling design.
    majority = float(100 * counts.max() / len(y_cls))
    acc = float(100 * (y_cls == p_cls).mean())
    out["class_accuracy_pct"] = acc
    out["majority_baseline_pct"] = majority
    out["class_lift_pct"] = acc - majority
    out["class_true_proportions_pct"] = [float(100 * c / len(y_cls)) for c in counts]
    out["class_confusion"] = [[int(((y_cls == a) & (p_cls == b)).sum())
                               for b in range(n_cls)] for a in range(n_cls)]

    # 3. spatial honesty -- reported in the same dict on purpose
    per_r2 = np.array([pooled_r2(t, q) for t, q in per_sample])
    corr = np.array([np.corrcoef(t, q)[0, 1] for t, q in per_sample])
    t_std = np.array([t.std() for t, _ in per_sample])
    p_std = np.array([q.std() for _, q in per_sample])
    out.update(
        per_sample_r2_mean=float(per_r2.mean()),
        per_sample_r2_negative_count=int((per_r2 < 0).sum()),
        n_samples=len(per_sample),
        within_sample_corr=float(corr.mean()),
        amplitude_ratio=float(p_std.mean() / t_std.mean()),
    )
    return out


def agg(rows, key):
    v = np.array([r[key] for r in rows])
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--class-edges", type=float, nargs="+", default=[5.40, 5.80],
                     help="pH boundaries between quality classes. Default: "
                          "midpoints between the PSE/normal/DFD anchors in "
                          "Figure 1 (DOI: 10.55002/mr.5.3.117). The anchors are "
                          "cited; these midpoints are interpolated from them.")
    ap.add_argument("--no-class", action="store_true",
                     help="omit the quality-class section entirely. The "
                          "tolerance bands in section 2 make the same point "
                          "without importing any threshold choice.")
    ap.add_argument("--out", default="metric_check_outputs/final_evaluation.json")
    a = ap.parse_args()

    split_dir = os.path.join(a.data, a.split)
    if not os.path.isdir(split_dir):
        raise SystemExit(
            "Dataset not found: {}\n\n"
            "The datasets are not committed (~1.7 GB together) but both\n"
            "regenerate bit-for-bit from the frozen PARAMS:\n\n"
            "    python generate_dataset.py\n"
            "        the frozen 400-sample set (seed 42, 300/50/50)\n\n"
            "    python generate_dataset.py --n 500 --seed 777 --all-test --out ligtas_test_extended\n"
            "        the 500-sample held-out set the headline figures use".format(split_dir))

    ds = SparseLIGTASDataset(os.path.join(a.data, a.split), N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    ckpts = sorted(glob.glob(a.ckpt_glob))
    if not ckpts:
        raise SystemExit(f"No checkpoints matched {a.ckpt_glob}")

    print(f"Data: {a.data}/{a.split}  ({len(ds)} samples)")
    print(f"Checkpoints: {len(ckpts)}\n")

    rows = []
    for ck in ckpts:
        r = score_checkpoint(ck, loader, a.class_edges)
        rows.append(r)
        print(f"{ck}")
        print(f"  R2={r['pooled_r2']:.4f}  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  "
              f"class={r['class_accuracy_pct']:.1f}%  "
              f"per-sample R2={r['per_sample_r2_mean']:+.3f}")

    print("\n" + "=" * 72)
    print("FINAL EVALUATION  --  mean +/- sd over "
          f"{len(rows)} seed{'s' if len(rows) > 1 else ''}")
    print("=" * 72)

    print("\n1. STANDARD REGRESSION METRICS")
    for k, lbl in [("pooled_r2", "R^2 (pooled)"), ("mae", "MAE (pH units)"),
                   ("rmse", "RMSE (pH units)")]:
        m, s = agg(rows, k)
        print(f"   {lbl:<22} {m:.4f} +/- {s:.4f}")

    print("\n2. PRACTICAL HEATMAP ACCURACY  <-- REPORT THESE")
    print("   Threshold-free: depends on no class definition and no class")
    print("   balance, so there is nothing in them to dispute.")
    for t in TOLERANCES:
        v = np.array([r["within_tolerance_pct"][f"{t:.2f}"] for r in rows])
        star = "  <- headline" if abs(t - 0.15) < 1e-9 else ""
        print(f"   pixels within +/-{t:.2f} pH   {v.mean():5.1f}% +/- {v.std():.1f}{star}")

    if not a.no_class:
        m, s = agg(rows, "class_accuracy_pct")
        mb, _ = agg(rows, "majority_baseline_pct")
        ml, mls = agg(rows, "class_lift_pct")
        edges = ", ".join(f"{e:.2f}" for e in a.class_edges)
        props = " / ".join(f"{n} {p:.0f}%" for n, p in
                           zip(CLASS_NAMES, rows[0]["class_true_proportions_pct"]))
        print("\n   2b. SECONDARY -- quality class. Weaker evidence; see caveats.")
        print(f"       correct quality class  {m:5.1f}% +/- {s:.1f}   (edges {edges})")
        print(f"       majority-class baseline{mb:5.1f}%      (true mix: {props})")
        print(f"       LIFT over baseline     {ml:+5.1f} pts +/- {mls:.1f}")
        print("       CAVEAT 1: the edges are OUR interpolation of Figure 1's")
        print("                 anchors (5.2/5.6/6.0); the source states no")
        print("                 boundaries. Different edges give a different %.")
        print("       CAVEAT 2: the class mix follows the uniform sampling")
        print("                 design, not a real pork population.")
        print("       If challenged, fall back on section 2 -- it needs neither.")

    print("\n3. SPATIAL LIMITATION  --  report this alongside section 1")
    m, s = agg(rows, "per_sample_r2_mean")
    print(f"   per-sample R^2         {m:+.4f} +/- {s:.4f}")
    neg = np.array([r["per_sample_r2_negative_count"] for r in rows])
    print(f"   negative on            {neg.min()}-{neg.max()} of {rows[0]['n_samples']} samples")
    m, s = agg(rows, "within_sample_corr")
    print(f"   within-sample corr     {m:+.4f} +/- {s:.4f}   (signal IS present)")
    m, s = agg(rows, "amplitude_ratio")
    print(f"   pred/true spatial SD   {m:.4f} +/- {s:.4f}   (over-amplified)")

    print("\n   Reading: the heatmap places structure correctly but over-expresses")
    print("   its magnitude. Quote section 1 and 2 only with this section beside them.")

    summary = {}
    for k in ("pooled_r2", "mae", "rmse", "class_accuracy_pct",
              "majority_baseline_pct", "class_lift_pct",
              "per_sample_r2_mean", "within_sample_corr", "amplitude_ratio"):
        m, s = agg(rows, k)
        summary[k] = dict(mean=m, std=s)
    summary["within_tolerance_pct"] = {
        f"{t:.2f}": dict(
            mean=float(np.mean([r["within_tolerance_pct"][f"{t:.2f}"] for r in rows])),
            std=float(np.std([r["within_tolerance_pct"][f"{t:.2f}"] for r in rows])))
        for t in TOLERANCES}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(data=f"{a.data}/{a.split}", n_samples=len(ds),
                       class_edges=a.class_edges,
                       class_edges_source="midpoints interpolated from Figure 1 anchors, DOI: 10.55002/mr.5.3.117 (anchors cited; midpoints are our interpolation)",
                       class_names=CLASS_NAMES,
                       per_checkpoint=rows, summary=summary), f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
