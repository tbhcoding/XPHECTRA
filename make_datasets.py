"""
make_datasets.py
================

Regenerates the two datasets the evaluation scripts expect. Neither is
committed -- together they are ~1.7 GB -- but both are deterministic, so
this reproduces them bit-for-bit from the frozen PARAMS.

    ligtas_synthetic_dataset/   400 samples, seed 42, split 300/50/50
        The frozen dataset the model was trained on.

    ligtas_test_extended/       500 samples, seed 777, all test
        An additional held-out set used for the reported headline figures.
        Seed 777 is deliberately NOT 42: reusing 42 would reproduce the
        training draws exactly and overlap the training data.

Reproducibility was verified on 2026-09-13 by regenerating a sample and
comparing against the on-disk copy: the reflectance cube was bit-for-bit
identical, and the pH map matched once NaN background pixels were
compared NaN-aware.

Run this after cloning, before the evaluation scripts. Takes about a
minute for both.

Usage:
    python make_datasets.py              # both, skipping any that exist
    python make_datasets.py --extended   # only the 500-sample set
    python make_datasets.py --force      # regenerate even if present
"""

import argparse
import os
import time

import numpy as np

import generate_dataset as G


def count_samples(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith("_msi.npy")])


def build(out_root, splits, seed, force):
    have = sum(count_samples(os.path.join(out_root, s)) for s, _ in splits)
    want = sum(n for _, n in splits)
    if have == want and not force:
        print(f"  {out_root}: already complete ({have} samples) -- skipping")
        return
    if have and not force:
        print(f"  {out_root}: incomplete ({have}/{want}) -- regenerating")

    rng = np.random.default_rng(seed)
    t0 = time.time()
    idx = 1
    for split, n in splits:
        d = os.path.join(out_root, split)
        os.makedirs(d, exist_ok=True)
        for _ in range(n):
            G.generate_sample(f"sample_{idx:03d}" if out_root.endswith("dataset")
                              else f"xtest_{idx:04d}", d, rng)
            idx += 1
    print(f"  {out_root}: {want} samples, seed {seed}, {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true", help="only the 400-sample frozen dataset")
    ap.add_argument("--extended", action="store_true", help="only the 500-sample extended set")
    ap.add_argument("--force", action="store_true", help="regenerate even if present")
    a = ap.parse_args()
    both = not (a.frozen or a.extended)

    print(f"denat_amplitude = {G.PARAMS['denat_amplitude']['value']}  "
          f"(regenerating against the current PARAMS)")

    if both or a.frozen:
        build("ligtas_synthetic_dataset",
              [("train", 300), ("val", 50), ("test", 50)], 42, a.force)
    if both or a.extended:
        build("ligtas_test_extended", [("test", 500)], 777, a.force)

    print("\nReady. The evaluation scripts can now run:")
    print("  python evaluate_heatmap.py")
    print("  python check_spatial_localization.py")
    print("  python check_edge_effect.py")


if __name__ == "__main__":
    main()
