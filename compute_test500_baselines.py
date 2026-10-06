"""
compute_test500_baselines.py
=============================

Computes Linear and PLSR R^2 on the 500-sample held-out set -- the SAME set
the reported CRN headline is measured on.

WHY THIS EXISTS
---------------
The existing baseline comparison (compute_amp04_baselines.py, and the
denat_amplitude sweep table) fits on `ligtas_synthetic_dataset/train` and
evaluates on `ligtas_synthetic_dataset/val`, n=50. That comparison is
internally like-for-like: all three methods are scored on the same 50
samples. But the CRN's REPORTED headline is measured on
`ligtas_test_extended/test`, n=500, and the baselines were never run on
that set.

So the chapter could state a margin on the validation scale, or a headline
on the 500-sample scale, but not both on one basis. This script supplies
the missing half: the same baselines, fitted on the same training pixels,
evaluated on the 500-sample set.

METHOD
------
Deliberately identical to compute_amp04_baselines.py except for the
evaluation split:

  fit    ligtas_synthetic_dataset/train   (the split the CRN trained on)
  eval   ligtas_test_extended/test        (the 500-sample reported set)

Reuses sweep_denat_amplitude.sample_pixels() and PLSR_N_COMPONENTS rather
than reimplementing either, so the pixel sampling and the PLSR component
count match every other baseline number in the project.

PLSR uses 4 components, fixed and not tuned -- the same disclosed
simplification as elsewhere.

Trains nothing, writes no checkpoint, touches no PARAMS. Reports numbers.

Usage:
    python compute_test500_baselines.py
"""

import argparse
import json
import os

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

import sweep_denat_amplitude as S

FIT_DATA = "ligtas_synthetic_dataset"
FIT_SPLIT = "train"
EVAL_DATA = "ligtas_test_extended"
EVAL_SPLIT = "test"
GEN_SEED = 42  # matches compute_amp04_baselines.py and sample_pixels()
               # elsewhere, so the pixel draw is the project-wide one


def pooled_r2(y, p):
    """Pooled over all pixels -- the same definition as the CRN headline."""
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metric_check_outputs/baselines_test500.json")
    a = ap.parse_args()

    for d, sp in ((FIT_DATA, FIT_SPLIT), (EVAL_DATA, EVAL_SPLIT)):
        if not os.path.isdir(os.path.join(d, sp)):
            raise SystemExit(
                "Dataset not found: {}/{}\n\n"
                "Not committed, but regenerates bit-for-bit:\n"
                "    python generate_dataset.py\n"
                "    python generate_dataset.py --n 500 --seed 777 --all-test "
                "--out ligtas_test_extended\n".format(d, sp))

    print("fit  on  {}/{}".format(FIT_DATA, FIT_SPLIT))
    print("eval on  {}/{}   (the reported headline set)\n".format(EVAL_DATA, EVAL_SPLIT))

    X_fit, y_fit = S.sample_pixels(FIT_DATA, FIT_SPLIT, S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    X_ev, y_ev = S.sample_pixels(EVAL_DATA, EVAL_SPLIT, S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    print("fit pixels: {}   eval pixels: {}\n".format(len(y_fit), len(y_ev)))

    # --- Linear ---
    Xa_fit = np.c_[X_fit, np.ones(len(X_fit))]
    coef, *_ = np.linalg.lstsq(Xa_fit, y_fit, rcond=None)
    Xa_ev = np.c_[X_ev, np.ones(len(X_ev))]
    pred_lin = Xa_ev @ coef
    r2_lin = pooled_r2(y_ev, pred_lin)
    mae_lin = float(np.abs(y_ev - pred_lin).mean())
    rmse_lin = float(np.sqrt(((y_ev - pred_lin) ** 2).mean()))

    # --- PLSR ---
    pls = PLSRegression(n_components=S.PLSR_N_COMPONENTS)
    pls.fit(X_fit, y_fit)
    pred_pls = pls.predict(X_ev).ravel()
    r2_pls = pooled_r2(y_ev, pred_pls)
    mae_pls = float(np.abs(y_ev - pred_pls).mean())
    rmse_pls = float(np.sqrt(((y_ev - pred_pls) ** 2).mean()))

    print("{:22s} {:>8s} {:>9s} {:>9s}".format("method", "R2", "MAE", "RMSE"))
    print("{:22s} {:>8.4f} {:>9.4f} {:>9.4f}".format(
        "Linear regression", r2_lin, mae_lin, rmse_lin))
    print("{:22s} {:>8.4f} {:>9.4f} {:>9.4f}".format(
        "PLSR ({} comp)".format(S.PLSR_N_COMPONENTS), r2_pls, mae_pls, rmse_pls))

    # The CRN figure for the SAME set, read from the reported file rather
    # than restated, so this script cannot drift from the headline.
    crn = None
    try:
        with open("RESULTS.json", encoding="utf-8") as f:
            crn = json.load(f)["headline"]
        print("{:22s} {:>8.4f} {:>9.4f} {:>9.4f}   (5-seed mean, RESULTS.json)".format(
            "CRN", crn["r2"][0], crn["mae_ph"][0], crn["rmse_ph"][0]))
        print("\n  Margin over Linear : {:+.4f} R2".format(crn["r2"][0] - r2_lin))
        print("  Margin over PLSR   : {:+.4f} R2".format(crn["r2"][0] - r2_pls))
    except Exception as e:
        print("\n  (RESULTS.json unreadable, CRN row omitted: {})".format(e))

    out = dict(
        fit_on="{}/{}".format(FIT_DATA, FIT_SPLIT),
        evaluated_on="{}/{}".format(EVAL_DATA, EVAL_SPLIT),
        n_eval_pixels=int(len(y_ev)),
        pixels_per_sample=int(S.PIXELS_PER_SAMPLE),
        pixel_sampling_seed=GEN_SEED,
        plsr_n_components=int(S.PLSR_N_COMPONENTS),
        linear=dict(r2=r2_lin, mae=mae_lin, rmse=rmse_lin),
        plsr=dict(r2=r2_pls, mae=mae_pls, rmse=rmse_pls),
    )
    if crn:
        out["crn_reported"] = dict(r2=crn["r2"][0], mae=crn["mae_ph"][0],
                                   rmse=crn["rmse_ph"][0], seeds=5)
        out["margin_over_linear_r2"] = crn["r2"][0] - r2_lin
        out["margin_over_plsr_r2"] = crn["r2"][0] - r2_pls

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote {}".format(a.out))


if __name__ == "__main__":
    main()
