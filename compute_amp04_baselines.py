"""
compute_amp04_baselines.py
===========================

Computes held-out Linear and PLSR R^2 for denat_amplitude=0.4 on the
frozen dataset (ligtas_synthetic_dataset/), using the exact same
sampling/fit/eval logic as deconfound_full_scale.py's run_one_amplitude()
(reuses sweep_denat_amplitude.sample_pixels() -- not reimplemented).

Why this script exists instead of just reusing deconfound_full_scale.py
directly: amp=0.4 is the live PARAMS value and already has a completed,
trusted CRN result on this exact dataset (crn_5seed_final/seed_0,1,2 --
mean R^2=0.7788, std=0.0629), so retraining the CRN for this amplitude
would be redundant compute. This script computes only the two pieces
that were actually missing -- Linear and PLSR held-out R^2 -- against
the SAME frozen dataset crn_5seed_final was trained on, instead of
regenerating a fresh amp=0.4 dataset under deconfound_outputs/ like the
other three amplitudes.

Does not touch PARAMS, CrudeCRN, or train_crn.py. Reports numbers only.

Usage:
    python compute_amp04_baselines.py
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

import sweep_denat_amplitude as S

DATA_DIR = "ligtas_synthetic_dataset"
GEN_SEED = 42  # matches generate_dataset.py main()'s own fixed seed,
               # and sample_pixels()'s seed in every other amplitude run


def main():
    X_train, y_train = S.sample_pixels(DATA_DIR, "train", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    X_val, y_val = S.sample_pixels(DATA_DIR, "val", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    print(f"train pixels: {len(y_train)}   val pixels: {len(y_val)}")

    # Linear -- fit on train, eval in-sample + held-out
    Xa_train = np.c_[X_train, np.ones(len(X_train))]
    coef, *_ = np.linalg.lstsq(Xa_train, y_train, rcond=None)
    pred_train = Xa_train @ coef
    r2_lin_in = 1 - ((y_train - pred_train) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
    Xa_val = np.c_[X_val, np.ones(len(X_val))]
    pred_val = Xa_val @ coef
    r2_lin_held = 1 - ((y_val - pred_val) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum()

    # PLSR -- fit on train, eval in-sample + held-out
    pls = PLSRegression(n_components=S.PLSR_N_COMPONENTS)
    pls.fit(X_train, y_train)
    r2_pls_in = r2_score(y_train, pls.predict(X_train).ravel())
    r2_pls_held = r2_score(y_val, pls.predict(X_val).ravel())

    print(f"Linear: in-sample R2={r2_lin_in:.4f}  held-out R2={r2_lin_held:.4f}")
    print(f"PLSR:   in-sample R2={r2_pls_in:.4f}  held-out R2={r2_pls_held:.4f}")
    print()
    print("CRN for amp=0.4 is NOT computed here -- reused directly from")
    print("crn_5seed_final/seed_0,1,2 (already trained on this same dataset).")

    return dict(
        linear_in_sample_r2=float(r2_lin_in), linear_held_out_r2=float(r2_lin_held),
        plsr_in_sample_r2=float(r2_pls_in), plsr_held_out_r2=float(r2_pls_held),
    )


if __name__ == "__main__":
    main()
