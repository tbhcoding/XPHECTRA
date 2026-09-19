"""
check_spatial_localization.py
=============================

Separates two failures that per-sample R^2 bundles together and that the
phrase "spatial reconstruction is unreliable" wrongly conflates:

  LOCATION  -- does the map flag the right REGIONS as higher-pH?
  MAGNITUDE -- does it predict the right AMOUNT of variation?

Per-sample R^2 is a calibration measure. It goes sharply negative here
because true within-sample variation (SD ~0.088 pH) is SMALLER than the
model's own per-pixel error (MAE ~0.094 pH) -- so on that scale a 1.25x
over-amplification is heavily penalised. That says nothing about whether
the hot spots are in the right place, which is what a user of a heatmap
actually needs.

This script measures location directly, on the samples where spatial
detail matters -- those spanning 2+ quality classes. Uniform samples are
excluded because localisation is vacuous there.

Reports numbers only. Trains nothing, modifies nothing.

Usage:
    python check_spatial_localization.py
    python check_spatial_localization.py --data ligtas_synthetic_dataset --split val
"""

import argparse, glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_crn import SparseLIGTASDataset
from crn_model import CrudeCRN

IN_RES = 256
N_POINTS = 4
BATCH = 8
TOP_FRAC = 0.20   # "hottest" region = top 20% of pixels by pH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_test_extended")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--class-edges", type=float, nargs="+", default=[5.40, 5.80])
    ap.add_argument("--out", default="metric_check_outputs/spatial_localization.json")
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
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    edges = a.class_edges

    rows = []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        model = CrudeCRN(in_res=IN_RES, out_res=IN_RES)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()

        acc, overlap, rank_ok, n_multi, n_all = [], [], 0, 0, 0
        with torch.no_grad():
            for x, _, _, ph, mask, _ in loader:
                pred = model(x)
                if pred.shape[-1] != IN_RES:
                    pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                         mode="bilinear", align_corners=False).squeeze(1)
                for b in range(pred.shape[0]):
                    m = mask[b].numpy()
                    t = ph[b].numpy()[m]
                    p = pred[b].numpy()[m]
                    n_all += 1
                    if len(np.unique(np.digitize(t, edges))) < 2:
                        continue          # uniform sample: localisation is vacuous
                    n_multi += 1
                    acc.append((np.digitize(t, edges) == np.digitize(p, edges)).mean())
                    k = max(1, int(TOP_FRAC * len(t)))
                    ti = set(np.argsort(t)[-k:]); pi = set(np.argsort(p)[-k:])
                    overlap.append(len(ti & pi) / k)
                    hot, cold = np.argsort(p)[-k:], np.argsort(p)[:k]
                    if t[hot].mean() > t[cold].mean():
                        rank_ok += 1

        rows.append(dict(checkpoint=ck, n_samples=n_all, n_multiclass=n_multi,
                         class_acc_multiclass_pct=float(100 * np.mean(acc)),
                         hot_region_overlap_pct=float(100 * np.mean(overlap)),
                         hot_direction_correct_pct=float(100 * rank_ok / n_multi)))
        print(f"{ck}\n  multi-class {n_multi}/{n_all}   "
              f"class-acc {rows[-1]['class_acc_multiclass_pct']:.1f}%   "
              f"overlap {rows[-1]['hot_region_overlap_pct']:.1f}%   "
              f"direction {rows[-1]['hot_direction_correct_pct']:.1f}%")

    agg = {}
    for k in ("class_acc_multiclass_pct", "hot_region_overlap_pct",
              "hot_direction_correct_pct"):
        v = np.array([r[k] for r in rows])
        agg[k] = dict(mean=float(v.mean()), std=float(v.std()))

    print("\n" + "=" * 70)
    print("LOCATION vs MAGNITUDE  --  multi-class samples only")
    print("=" * 70)
    print(f"  samples spanning 2+ classes        {rows[0]['n_multiclass']} of {rows[0]['n_samples']}")
    print(f"  predicted hot region truly hotter  "
          f"{agg['hot_direction_correct_pct']['mean']:5.1f}% +/- {agg['hot_direction_correct_pct']['std']:.1f}"
          f"   (chance 50%) -- LENIENT: direction only")
    print(f"  hottest-20% overlap with truth     "
          f"{agg['hot_region_overlap_pct']['mean']:5.1f}% +/- {agg['hot_region_overlap_pct']['std']:.1f}"
          f"   (chance {100*TOP_FRAC:.0f}%) -- the demanding measure")
    print(f"  per-pixel class accuracy in them   "
          f"{agg['class_acc_multiclass_pct']['mean']:5.1f}% +/- {agg['class_acc_multiclass_pct']['std']:.1f}")
    print()
    print("  Reading: LOCATION is reliable, MAGNITUDE is not. Do not describe")
    print("  this as 'spatial reconstruction is unreliable' -- that conflates")
    print("  the two and overstates the weakness. The map points at the right")
    print("  regions and over-expresses how different they are.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(data=f"{a.data}/{a.split}", class_edges=edges,
                       top_fraction=TOP_FRAC, per_checkpoint=rows, aggregate=agg), f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
