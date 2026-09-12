"""
sweep_denat_amplitude.py
=========================

Chapter 4 defensibility check: does the CRN's advantage over simple
baselines (OLS linear regression, PLSR) hold across the full disclosed
denat_amplitude sweep range [0.2, 0.4, 0.6, 0.8], not just at the
current operating point (0.4)?

This is a REPORTING / VALIDATION script, not an architecture change.
It does not modify CrudeCRN, sparse_loss, tv_loss, or the
leakage-prevention discipline in train_crn.py -- train_crn.train() is
imported and called as-is (see that function's own docstring for what
"structural extraction only" means there: it is the former main()
body, unchanged, just made callable).

It does NOT tune denat_amplitude toward whichever value makes the CRN
look best. generate_dataset.PARAMS['denat_amplitude']['value'] is
restored to its original value when the sweep finishes (even on
error), and the point of this script is to report the result across
the whole range, not pick a winner.

Scale, deliberately kept small -- this is a defensibility check, not a
new headline dataset:
  - N_TRAIN/N_VAL = 100/20 samples per amplitude, not the full 400.
    Precedent: the project's own notebook (LIGTAS_calibration.ipynb,
    Section 5) used only 12 samples for its original
    amplitude-vs-linear-R^2 sweep.
  - CRN_SEEDS = 3, not 5 -- keeps total runtime manageable. On this
    machine (CPU only, no CUDA), a single seed took ~20-35 min in the
    project's earlier 5-seed run on the full 400-sample dataset; with
    100 train samples here each epoch is proportionally cheaper, but
    4 amplitudes x 3 seeds is still a multi-hour job.
  - PLSR n_components is FIXED at 4 across every amplitude, not tuned
    per amplitude -- a disclosed simplification (6 input bands, 4
    components is an unremarkable choice; tuning it per amplitude
    would itself be a small form of chasing the best-looking number,
    which this script is explicitly not supposed to do).
  - Both baselines (linear and PLSR) are fit and evaluated IN-SAMPLE on
    the same sampled pixels -- the same convention used everywhere else
    in this project's non-triviality checks (generate_dataset.py's own
    self_test()/_linear_baseline_once(), and the notebook's Sections
    3b/5): the question being asked is "can a trivial static formula
    solve this," which is inherently an in-sample question, not a
    held-out generalization test.

Explicitly does NOT:
  - Change CrudeCRN's architecture, sparse_loss, tv_loss, or the
    leakage-prevention discipline in train_crn.py.
  - Run the full 400-sample dataset or 5+ seeds per amplitude.
  - Propose or make any other changes to the model, training logic, or
    physics parameters -- this reports findings, nothing else.

Usage:
    python sweep_denat_amplitude.py
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_absolute_error

import generate_dataset as G
import train_crn

# ---------------------------------------------------------------------------
# Configuration -- deliberately small-scale, see module docstring.
# ---------------------------------------------------------------------------
AMPLITUDES = [0.2, 0.4, 0.6, 0.8]
N_TRAIN = 100
N_VAL = 20
PIXELS_PER_SAMPLE = 400            # matches the project-wide convention:
                                    # generate_dataset.py's own
                                    # _linear_baseline_once() and the
                                    # notebook's Sections 3b/5 both use 400
PLSR_N_COMPONENTS = 4              # fixed, not tuned -- disclosed simplification
CRN_SEEDS = [0, 1, 2]
CRN_EPOCHS_CAP = 50
CRN_PATIENCE = 10
OUT_DIR = "sweep_outputs"
BEAT_MARGIN = 0.05                 # "clearly beats" threshold, per the task


def generate_amplitude_dataset(amp, out_dir):
    """
    Writes N_TRAIN + N_VAL samples at the given denat_amplitude into
    out_dir/{train,val}/, using generate_dataset.py's OWN
    generate_sample() writer (reused, not reimplemented). Fixed seed
    per amplitude, derived deterministically from amp itself, so the
    whole sweep is reproducible from this script alone.
    """
    seed = int(round(amp * 1000))  # 0.2->200, 0.4->400, 0.6->600, 0.8->800
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    idx = 1
    for split, count in [("train", N_TRAIN), ("val", N_VAL)]:
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for _ in range(count):
            G.generate_sample(f"sample_{idx:03d}", d, rng)
            idx += 1
    return seed


def sample_pixels(data_dir, split, n_pixels_per_sample, seed):
    """
    The pixel-sampling logic already used in LIGTAS_calibration.ipynb
    Sections 3b/5 (and generate_dataset.py's own
    _linear_baseline_once()) -- factored into one shared function here,
    since the notebook itself duplicates it inline between the two
    cells rather than factoring it. For each generated sample, load its
    cube + hidden pH map and draw n_pixels_per_sample random meat
    pixels; stack across all samples in the split into (X, y).
    """
    d = os.path.join(data_dir, split)
    ids = sorted({f.split("_msi.npy")[0] for f in os.listdir(d) if f.endswith("_msi.npy")})
    rng = np.random.default_rng(seed)
    X, y = [], []
    for sid in ids:
        cube = np.load(os.path.join(d, f"{sid}_msi.npy"))
        ph = np.load(os.path.join(d, f"{sid}_phtrue.npy"))
        mask = np.isfinite(ph)
        flat_idx = np.flatnonzero(mask)
        n = min(n_pixels_per_sample, len(flat_idx))
        idx = rng.choice(flat_idx, n, replace=False)
        X.append(cube.reshape(-1, 6)[idx])
        y.append(ph.reshape(-1)[idx])
    return np.vstack(X), np.concatenate(y)


def linear_baseline(X, y):
    """
    Exactly the pixel-sampling + lstsq logic from
    LIGTAS_calibration.ipynb Sections 3b/5 (and generate_dataset.py's
    _linear_baseline_once()): OLS via np.linalg.lstsq on [X, 1],
    in-sample R^2/MAE. See module docstring for why in-sample is the
    right convention here.
    """
    Xa = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = np.abs(y - pred).mean()
    return float(r2), float(mae)


def plsr_baseline(X, y, n_components=PLSR_N_COMPONENTS):
    """
    PLSR baseline on the SAME sampled pixels as the linear baseline,
    for a like-for-like comparison. n_components is fixed (not tuned
    per amplitude) -- see module docstring.
    """
    model = PLSRegression(n_components=n_components)
    model.fit(X, y)
    pred = model.predict(X).ravel()
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    return float(r2), float(mae)


def run_crn(data_dir, seed, out_dir):
    """
    Calls train_crn.train() directly (imported, not shelled out to).
    See train_crn.py's train() docstring: this is the exact former
    main() body, unchanged, just made callable -- same --patience
    early-stopping / val_loss-only checkpoint discipline, same
    leakage-prevention (phtrue.npy never touches gradients or the
    checkpoint decision), as every other CRN run in this project.
    """
    ap = train_crn.build_argparser()
    args = ap.parse_args([
        "--data", data_dir,
        "--seed", str(seed),
        "--patience", str(CRN_PATIENCE),
        "--epochs", str(CRN_EPOCHS_CAP),
        "--out", out_dir,
    ])
    result = train_crn.train(args)
    return result["best_val_r2"], result["best_val_mae"]


def main():
    original_amp = G.PARAMS["denat_amplitude"]["value"]
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []  # one dict per (amplitude, method, seed_if_applicable, R2, MAE)
    crn_by_amp = {amp: [] for amp in AMPLITUDES}
    linear_by_amp = {}
    plsr_by_amp = {}

    try:
        for amp in AMPLITUDES:
            print(f"\n{'=' * 70}\ndenat_amplitude = {amp}\n{'=' * 70}")
            G.PARAMS["denat_amplitude"]["value"] = amp

            data_dir = os.path.join(OUT_DIR, f"data_amp_{amp}")
            seed = generate_amplitude_dataset(amp, data_dir)
            print(f"  generated {N_TRAIN} train / {N_VAL} val samples "
                  f"(seed={seed}) -> {data_dir}")

            X, y = sample_pixels(data_dir, "train", PIXELS_PER_SAMPLE, seed=seed)
            print(f"  sampled {len(y)} pixels for baselines")

            lin_r2, lin_mae = linear_baseline(X, y)
            linear_by_amp[amp] = (lin_r2, lin_mae)
            rows.append(dict(denat_amplitude=amp, method="linear", seed="",
                              r2=lin_r2, mae=lin_mae))
            print(f"  linear   R2={lin_r2:.4f}  MAE={lin_mae:.4f}")

            pls_r2, pls_mae = plsr_baseline(X, y)
            plsr_by_amp[amp] = (pls_r2, pls_mae)
            rows.append(dict(denat_amplitude=amp, method="plsr", seed="",
                              r2=pls_r2, mae=pls_mae))
            print(f"  PLSR     R2={pls_r2:.4f}  MAE={pls_mae:.4f} "
                  f"(n_components={PLSR_N_COMPONENTS}, fixed)")

            for s in CRN_SEEDS:
                crn_out = os.path.join(OUT_DIR, f"crn_amp_{amp}_seed_{s}")
                r2, mae = run_crn(data_dir, s, crn_out)
                crn_by_amp[amp].append((r2, mae))
                rows.append(dict(denat_amplitude=amp, method="crn", seed=s,
                                  r2=r2, mae=mae))
                print(f"  CRN seed={s}  R2={r2:.4f}  MAE={mae:.4f}")
    finally:
        # Never leave PARAMS mutated, even if something above raises --
        # this script must not silently change the project's live physics.
        G.PARAMS["denat_amplitude"]["value"] = original_amp
        print(f"\ndenat_amplitude restored to {original_amp} (original value).")

    # ---- save results table ----
    results_csv = os.path.join(OUT_DIR, "sweep_results.csv")
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["denat_amplitude", "method", "seed", "r2", "mae"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved results table to {results_csv} (and sweep_results.json).")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    lin_y = [linear_by_amp[a][0] for a in AMPLITUDES]
    pls_y = [plsr_by_amp[a][0] for a in AMPLITUDES]
    crn_mean = [float(np.mean([r for r, _ in crn_by_amp[a]])) for a in AMPLITUDES]
    crn_std = [float(np.std([r for r, _ in crn_by_amp[a]])) for a in AMPLITUDES]

    ax.plot(AMPLITUDES, lin_y, "o-", label="linear (OLS)")
    ax.plot(AMPLITUDES, pls_y, "s-", label=f"PLSR (n_components={PLSR_N_COMPONENTS})")
    ax.errorbar(AMPLITUDES, crn_mean, yerr=crn_std, fmt="^-",
                capsize=4, label=f"CRN (mean +/- std, {len(CRN_SEEDS)} seeds)")
    ax.set_xlabel("denaturation amplitude (assumed)")
    ax.set_ylabel("R2")
    ax.set_title("CRN vs. linear/PLSR baselines across denat_amplitude  [CHAPTER 4 FIGURE]")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "sweep_crn_vs_baselines.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # ---- summary table ----
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'amp':>6} {'linear R2':>10} {'PLSR R2':>10} "
          f"{'CRN R2 (mean+/-std)':>24} {'beats both by >' + str(BEAT_MARGIN) + '?':>20}")
    flagged = []
    for i, amp in enumerate(AMPLITUDES):
        beats = (crn_mean[i] - lin_y[i] > BEAT_MARGIN) and (crn_mean[i] - pls_y[i] > BEAT_MARGIN)
        print(f"{amp:>6.2f} {lin_y[i]:>10.4f} {pls_y[i]:>10.4f} "
              f"{crn_mean[i]:>10.4f} +/- {crn_std[i]:<9.4f} {'YES' if beats else 'NO':>20}")
        if not beats:
            flagged.append(f"amp={amp} (CRN mean R2={crn_mean[i]:.4f} "
                            f"vs linear={lin_y[i]:.4f}, PLSR={pls_y[i]:.4f})")
        # Also flag any INDIVIDUAL seed that doesn't clearly beat both
        # baselines -- same standard as flagging seed 0 at amp=0.4 before.
        for s, (r2, _mae) in zip(CRN_SEEDS, crn_by_amp[amp]):
            seed_beats = (r2 - lin_y[i] > BEAT_MARGIN) and (r2 - pls_y[i] > BEAT_MARGIN)
            if not seed_beats:
                flagged.append(f"  amp={amp} seed={s}: CRN R2={r2:.4f} "
                                f"(linear={lin_y[i]:.4f}, PLSR={pls_y[i]:.4f})")

    print()
    if flagged:
        print(f"FLAGGED -- does not clearly beat both baselines by >{BEAT_MARGIN} "
              f"margin (disclose, do not discover later):")
        for f_ in flagged:
            print(f"  {f_}")
    else:
        print(f"No amplitude/seed combination failed to beat both baselines "
              f"by >{BEAT_MARGIN} margin.")


if __name__ == "__main__":
    main()
