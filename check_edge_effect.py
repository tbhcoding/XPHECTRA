"""
check_edge_effect.py
====================

Quantifies a boundary artefact visible in the predicted maps: a rim of
elevated error along the tissue edge. The network sees less surrounding
context there, and convolutional padding supplies partial background, so
predictions degrade near the mask boundary.

Reports MAE inside an N-pixel band along the tissue boundary against MAE
in the interior. Trains nothing, modifies nothing.

Why it matters for the write-up: the headline MAE averages the degraded
boundary into the interior. Reporting both is more informative and more
honest than reporting either alone -- the interior figure is what a
quality inspector would act on, and the boundary figure is a disclosed,
named artefact with a known remedy (boundary-aware padding or masked
convolution).

Usage:
    python check_edge_effect.py
    python check_edge_effect.py --band 8 --split test
"""

import argparse, glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

from train_crn import SparseLIGTASDataset
from crn_model import CrudeCRN

IN_RES = 256
N_POINTS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt-glob", default="crn_5seed_final/seed_*/crn_best.pt")
    ap.add_argument("--band", type=int, default=8, help="boundary band width in pixels")
    ap.add_argument("--out", default="metric_check_outputs/edge_effect.json")
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
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    rows = []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        model = CrudeCRN(in_res=IN_RES, out_res=IN_RES)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()
        edge, interior, worse = [], [], 0
        with torch.no_grad():
            for x, _, _, ph, mask, _ in loader:
                pred = model(x)
                if pred.shape[-1] != IN_RES:
                    pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                         mode="bilinear", align_corners=False).squeeze(1)
                for b in range(pred.shape[0]):
                    mk = mask[b].numpy()
                    err = np.abs(ph[b].numpy() - pred[b].numpy())
                    inner = ndimage.binary_erosion(mk, iterations=a.band)
                    rim = mk & ~inner
                    if rim.sum() == 0 or inner.sum() == 0:
                        continue
                    e, i = err[rim].mean(), err[inner].mean()
                    edge.append(e); interior.append(i)
                    worse += int(e > i)
        e, i = float(np.mean(edge)), float(np.mean(interior))
        rows.append(dict(checkpoint=ck, edge_mae=e, interior_mae=i,
                         ratio=e / i, pct_samples_edge_worse=100 * worse / len(edge),
                         n_samples=len(edge)))
        print(f"{ck}\n  edge {e:.4f}  interior {i:.4f}  ratio {e/i:.2f}x")

    agg = {k: dict(mean=float(np.mean([r[k] for r in rows])),
                   std=float(np.std([r[k] for r in rows])))
           for k in ("edge_mae", "interior_mae", "ratio", "pct_samples_edge_worse")}

    print("\n" + "=" * 62)
    print(f"BOUNDARY ARTEFACT  --  {a.band}px band, {rows[0]['n_samples']} samples, "
          f"{len(rows)} seeds")
    print("=" * 62)
    print(f"  MAE, boundary band   {agg['edge_mae']['mean']:.4f} +/- {agg['edge_mae']['std']:.4f} pH")
    print(f"  MAE, interior        {agg['interior_mae']['mean']:.4f} +/- {agg['interior_mae']['std']:.4f} pH")
    print(f"  ratio                {agg['ratio']['mean']:.2f}x")
    print(f"  samples where edge is worse   {agg['pct_samples_edge_worse']['mean']:.0f}%")
    print("\n  Reading: the headline MAE averages the degraded boundary into the")
    print("  interior. Report both. The cause is convolutional context loss at")
    print("  the mask edge, with a known remedy (boundary-aware padding).")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(band_px=a.band, data=f"{a.data}/{a.split}",
                       per_checkpoint=rows, aggregate=agg), f, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
