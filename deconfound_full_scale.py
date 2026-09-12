"""
deconfound_full_scale.py
==========================

Follow-up to sweep_denat_amplitude.py's finding that the CRN lost to
linear/PLSR baselines at denat_amplitude=0.4/0.6/0.8 in the small
(120-sample) sweep. The amp=0.4 case was already deconfounded by hand
(reusing the existing 400-sample ligtas_synthetic_dataset/ and the
already-completed crn_5seed_final/ seed 0-2 runs) and confirmed the
CRN wins again at full scale -- see docs/TEAM_LOG.md and this
session's conversation for that result.

This script repeats the SAME deconfounding procedure at full scale
(400 samples, 300 train / 50 val, matching ligtas_synthetic_dataset/'s
own split and generation seed) for denat_amplitude=0.8, then 0.6 if
time allows -- these two didn't have a pre-existing 400-sample dataset
to reuse, so both the dataset and the CRN runs are generated fresh
here.

Same discipline as sweep_denat_amplitude.py:
  - Does not touch CrudeCRN, sparse_loss, tv_loss, or the
    val_loss-only checkpoint-selection criterion in train_crn.py.
  - Does not tune anything toward a preferred outcome.
  - PARAMS['denat_amplitude']['value'] is restored to its original
    value when finished, even on error.
  - Reports numbers only -- no conclusions about the thesis write-up.

Usage:
    python deconfound_full_scale.py
"""

import os
import json
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

import generate_dataset as G
import train_crn
import sweep_denat_amplitude as S  # reuse sample_pixels(), not reimplemented

AMPLITUDES = [0.8, 0.6]   # order per the task: 0.8 first, then 0.6 if time allows
N_TRAIN = 300              # matches ligtas_synthetic_dataset/'s own split exactly
N_VAL = 50
GEN_SEED = 42               # matches generate_dataset.py main()'s own fixed seed --
                             # same seed across amplitudes isolates the amplitude's
                             # effect (same pH fields/masks/nuisance draws), only the
                             # amplitude-driven scattering physics differs
CRN_SEEDS = [0, 1, 2]
CRN_PATIENCE = 10
CRN_EPOCHS_CAP = 50
OUT_ROOT = "deconfound_outputs"


def generate_full_dataset(amp, out_dir):
    rng = np.random.default_rng(GEN_SEED)
    idx = 1
    for split, count in [("train", N_TRAIN), ("val", N_VAL)]:
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for _ in range(count):
            G.generate_sample(f"sample_{idx:03d}", d, rng)
            idx += 1


def run_one_amplitude(amp):
    print(f"\n{'#' * 70}\n# denat_amplitude = {amp}  (full-scale deconfound)\n{'#' * 70}")
    G.PARAMS["denat_amplitude"]["value"] = amp

    data_dir = os.path.join(OUT_ROOT, f"data_amp_{amp}")
    if os.path.isdir(os.path.join(data_dir, "train")) and \
       len([f for f in os.listdir(os.path.join(data_dir, "train")) if f.endswith("_msi.npy")]) == N_TRAIN:
        print(f"  reusing already-generated dataset at {data_dir}")
    else:
        generate_full_dataset(amp, data_dir)
        print(f"  generated {N_TRAIN} train / {N_VAL} val samples (seed={GEN_SEED}) -> {data_dir}")

    X_train, y_train = S.sample_pixels(data_dir, "train", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    X_val, y_val = S.sample_pixels(data_dir, "val", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    print(f"  train pixels: {len(y_train)}   val pixels: {len(y_val)}")

    # Linear -- fit on train, eval in-sample + held-out
    Xa_train = np.c_[X_train, np.ones(len(X_train))]
    coef, *_ = np.linalg.lstsq(Xa_train, y_train, rcond=None)
    pred_train_lin = Xa_train @ coef
    r2_lin_in = 1 - ((y_train - pred_train_lin) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
    Xa_val = np.c_[X_val, np.ones(len(X_val))]
    pred_val_lin = Xa_val @ coef
    r2_lin_held = 1 - ((y_val - pred_val_lin) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum()

    # PLSR -- fit on train, eval in-sample + held-out
    pls = PLSRegression(n_components=4)
    pls.fit(X_train, y_train)
    r2_pls_in = r2_score(y_train, pls.predict(X_train).ravel())
    r2_pls_held = r2_score(y_val, pls.predict(X_val).ravel())

    print(f"  Linear:  in-sample R2={r2_lin_in:.4f}   held-out R2={r2_lin_held:.4f}")
    print(f"  PLSR:    in-sample R2={r2_pls_in:.4f}   held-out R2={r2_pls_held:.4f}")

    # CRN -- 3 fresh seeds, same --patience/--epochs convention as every other run
    crn_r2 = []
    for s in CRN_SEEDS:
        crn_out = os.path.join(OUT_ROOT, f"crn_amp_{amp}_seed_{s}")
        ap = train_crn.build_argparser()
        args = ap.parse_args([
            "--data", data_dir,
            "--seed", str(s),
            "--patience", str(CRN_PATIENCE),
            "--epochs", str(CRN_EPOCHS_CAP),
            "--out", crn_out,
        ])
        result = train_crn.train(args)
        crn_r2.append(result["best_val_r2"])
        print(f"  CRN seed={s}  R2={result['best_val_r2']:.4f}  MAE={result['best_val_mae']:.4f}")

    crn_r2 = np.array(crn_r2)
    row = dict(
        denat_amplitude=amp,
        linear_in_sample_r2=float(r2_lin_in), linear_held_out_r2=float(r2_lin_held),
        plsr_in_sample_r2=float(r2_pls_in), plsr_held_out_r2=float(r2_pls_held),
        crn_r2_per_seed=crn_r2.tolist(),
        crn_r2_mean=float(crn_r2.mean()), crn_r2_std=float(crn_r2.std()),
    )
    print(f"\n  SUMMARY amp={amp}: Linear(held-out)={r2_lin_held:.4f}  "
          f"PLSR(held-out)={r2_pls_held:.4f}  CRN(mean+/-std)={crn_r2.mean():.4f}+/-{crn_r2.std():.4f}")
    beats = (crn_r2.mean() - r2_lin_held > 0.05) and (crn_r2.mean() - r2_pls_held > 0.05)
    print(f"  CRN beats both (held-out) by >0.05: {'YES' if beats else 'NO'}")
    return row


def main():
    original_amp = G.PARAMS["denat_amplitude"]["value"]
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    try:
        for amp in AMPLITUDES:
            row = run_one_amplitude(amp)
            rows.append(row)
            with open(os.path.join(OUT_ROOT, "deconfound_results.json"), "w") as f:
                json.dump(rows, f, indent=2)
    finally:
        G.PARAMS["denat_amplitude"]["value"] = original_amp
        print(f"\ndenat_amplitude restored to {original_amp} (original value).")

    print(f"\n{'=' * 90}\nFINAL SUMMARY (full-scale, 400 samples, 300 train/50 val, 3 seeds)\n{'=' * 90}")
    print(f"{'amp':>6} {'Linear(held)':>13} {'PLSR(held)':>11} {'CRN mean+/-std':>18} {'beats both>0.05?':>18}")
    for row in rows:
        beats = (row["crn_r2_mean"] - row["linear_held_out_r2"] > 0.05) and \
                (row["crn_r2_mean"] - row["plsr_held_out_r2"] > 0.05)
        print(f"{row['denat_amplitude']:>6.2f} {row['linear_held_out_r2']:>13.4f} "
              f"{row['plsr_held_out_r2']:>11.4f} "
              f"{row['crn_r2_mean']:>10.4f}+/-{row['crn_r2_std']:<6.4f} "
              f"{'YES' if beats else 'NO':>18}")


if __name__ == "__main__":
    main()
