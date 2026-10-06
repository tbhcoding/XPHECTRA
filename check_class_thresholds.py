"""
check_class_thresholds.py
==========================

Tests whether the quality-class result depends on the class boundaries chosen.

WHY THIS EXISTS
---------------
Chapter 4 reports per-pixel quality-class accuracy using boundaries of 5.4 and
5.8 pH, obtained as midpoints between the anchors of the reference scale cited
in Chapter 2 (5.2 / 5.6 / 6.0).

Those are not the only boundaries in use. The pork quality literature reports
no agreed criterion: commonly cited alternatives place PSE below an ultimate pH
of 5.5 and DFD above 6.1, with the normal band in between. A panelist who works
from those figures will observe that a DFD boundary at 5.8 is lower than the
value they know.

This script recomputes the class result under several boundary sets so that the
conclusion can be shown to hold, or not hold, independently of which convention
is applied. It reports accuracy against the majority-class baseline in each
case, since a three-class problem with an uneven split flatters a classifier
that simply guesses the largest class.

Reads committed checkpoints only. Trains nothing, writes no checkpoint.

Usage:
    python check_class_thresholds.py
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

# (label, lower boundary, upper boundary, provenance)
SCHEMES = [
    ("adopted (Fig. 1 midpoints)", 5.4, 5.8,
     "midpoints between the cited scale's 5.2 / 5.6 / 6.0 anchors"),
    ("literature A", 5.5, 6.1,
     "PSE below 5.5, DFD above 6.1; a commonly cited ultimate-pH convention"),
    ("literature B", 5.5, 6.2,
     "PSE below 5.5, DFD above 6.2; the stricter DFD boundary in common use"),
    ("cited anchors as limits", 5.2, 6.0,
     "the reference scale's own PSE and DFD anchor values used directly"),
]


def collect(model, root):
    ds = SparseLIGTASDataset(root, N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    T, P = [], []
    with torch.no_grad():
        for x, _c, _v, ph, mask, _m in loader:
            p = model(x)
            if p.shape[-1] != IN_RES:
                p = F.interpolate(p.unsqueeze(1), size=IN_RES,
                                  mode="bilinear", align_corners=False).squeeze(1)
            for b in range(p.shape[0]):
                mk = mask[b]
                T.append(ph[b][mk].numpy())
                P.append(p[b][mk].numpy())
    return np.concatenate(T), np.concatenate(P)


def classify(v, lo, hi):
    return np.digitize(v, [lo, hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_test_extended")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--out", default="metric_check_outputs/class_thresholds.json")
    a = ap.parse_args()

    root = os.path.join(a.data, a.split)
    if not os.path.isdir(root):
        raise SystemExit(
            "Dataset not found: {}\n\nRegenerate with:\n"
            "    python generate_dataset.py --n 500 --seed 777 --all-test "
            "--out ligtas_test_extended\n".format(root))

    print("evaluating {}\n".format(root))

    rows = {lab: [] for lab, _, _, _ in SCHEMES}
    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = CrudeCRN(in_res=IN_RES, out_res=IN_RES)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        t, p = collect(m, root)
        for lab, lo, hi, _ in SCHEMES:
            ct, cp = classify(t, lo, hi), classify(p, lo, hi)
            acc = float((ct == cp).mean() * 100)
            _, counts = np.unique(ct, return_counts=True)
            maj = float(counts.max() / counts.sum() * 100)
            shares = [float((ct == i).mean() * 100) for i in range(3)]
            rows[lab].append(dict(accuracy_pct=acc, majority_pct=maj,
                                  lift_pts=acc - maj, true_shares_pct=shares))
        print("  {} done".format(os.path.basename(os.path.dirname(ck))))

    def agg(lab, k):
        return float(np.mean([r[k] for r in rows[lab]]))

    print("\n{:30s} {:>8s} {:>10s} {:>8s}   {}".format(
        "scheme", "accuracy", "majority", "lift", "true PSE/normal/DFD %"))
    out = []
    for lab, lo, hi, note in SCHEMES:
        shares = np.mean([r["true_shares_pct"] for r in rows[lab]], axis=0)
        print("{:30s} {:>7.1f}% {:>9.1f}% {:>+7.1f}   {:.1f} / {:.1f} / {:.1f}".format(
            lab, agg(lab, "accuracy_pct"), agg(lab, "majority_pct"),
            agg(lab, "lift_pts"), *shares))
        out.append(dict(scheme=lab, lower=lo, upper=hi, provenance=note,
                        accuracy_pct=agg(lab, "accuracy_pct"),
                        majority_baseline_pct=agg(lab, "majority_pct"),
                        lift_pts=agg(lab, "lift_pts"),
                        true_shares_pct=[float(x) for x in shares]))

    print("\n  Reading: the lift over the majority-class baseline is the figure that")
    print("  matters. If it stays positive and comparable across schemes, the result")
    print("  is not an artefact of where the boundaries were placed.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(data="{}/{}".format(a.data, a.split),
                       n_seeds=len(rows[SCHEMES[0][0]]), schemes=out), f, indent=2)
    print("\nWrote {}".format(a.out))


if __name__ == "__main__":
    main()
