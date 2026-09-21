"""
make_figures.py
===============

Generates the manuscript figures that are not already produced as a
side-effect of training. Reads the frozen dataset only; trains nothing,
modifies nothing.

Produces (into --out, default figures/):

  fig_dataset_sample.png
      What one sample contains: all six spectral bands, the RGB
      composite, the hidden dense pH map with the four sparse
      supervision points marked, and the tissue mask. This is the
      "describe your dataset" figure for Chapter 3 -- a reader cannot
      assume familiarity with a synthetic dataset, so show it.

  fig_loss_curves.png
      Training and validation loss for one representative seed, with the
      early-stopping checkpoint marked. Chapter 4.

  fig_sweep_comparison.png
      CRN vs Linear vs PLSR across the denat_amplitude sweep, with per-seed
      error bars on the CRN series. Built from the CORRECTED table, not the
      pre-metric-fix sweep_outputs/ data. Chapter 4.

  fig_heatmap_example.png
      True / predicted / |error| for the median-accuracy sample. Chapter 4.

  fig_ph_distribution.png
      How pH is distributed between and within samples. Makes two
      methodology points visually: the uniform between-sample sampling
      design, and that within-sample variation is small but not
      negligible -- which is the context needed to read the spatial
      results in Chapter 4 correctly.

Usage:
    python make_figures.py
    python make_figures.py --sample sample_351 --out figures/
"""

import argparse
import json
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVELENGTHS = [481, 525, 573, 600, 730, 970]
DPI = 200

# Categorical series palette. Validated for colour-vision deficiency:
# worst adjacent-pair separation is OKLab dE 12.2 (deuteranopia), above the
# dE >= 8 target; all three clear 3:1 contrast on white. Line style varies
# alongside hue so identity never rests on colour alone.
C_LIN = "#1F6F78"    # Linear regression  (teal)
C_PLS = "#C9761A"    # PLSR               (orange)
C_CRN = "#8C1D2E"    # CRN                (deep red -- the project's primary)
CMAP_PH = "turbo"    # author's choice -- kept for continuity with the
                     # figures the team and panel have already seen
CMAP_ERR = "magma"   # sequential; dark = low error
C_TRAIN = C_LIN      # training loss
C_VAL = C_CRN        # validation loss


def fig_dataset_sample(data_dir, sample_id, out_path):
    cube = np.load(os.path.join(data_dir, f"{sample_id}_msi.npy"))
    ph = np.load(os.path.join(data_dir, f"{sample_id}_phtrue.npy"))
    with open(os.path.join(data_dir, f"{sample_id}_labels.json")) as f:
        meta = json.load(f)
    mask = np.isfinite(ph)
    pts = meta["sparse_measurements"][:4]

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25], hspace=0.28, wspace=0.12)

    # --- row 1: the six spectral bands ---------------------------------
    vmin, vmax = cube[mask].min(), cube[mask].max()
    for i, wl in enumerate(WAVELENGTHS):
        ax = fig.add_subplot(gs[0, i])
        band = np.where(mask, cube[:, :, i], np.nan)
        ax.imshow(band, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{wl} nm", fontsize=10)
        ax.axis("off")
    fig.text(0.5, 0.955, "Six-band multispectral cube (one sample)",
             ha="center", fontsize=11, weight="bold")

    # --- row 2: RGB, pH map with sparse points, mask --------------------
    ax = fig.add_subplot(gs[1, 0:2])
    rgb = np.dstack([cube[:, :, 3], cube[:, :, 1], cube[:, :, 0]])  # 600/525/481
    rgb = np.clip(rgb / max(rgb.max(), 1e-6), 0, 1)
    ax.imshow(rgb)
    ax.set_title("RGB composite (600/525/481 nm)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2:4])
    im = ax.imshow(np.where(mask, ph, np.nan), cmap=CMAP_PH)
    for p in pts:
        r, c = p["pixel_coord"]
        ax.plot(c, r, "x", color="white", markersize=9, markeredgewidth=2.2)
        ax.plot(c, r, "x", color="black", markersize=9, markeredgewidth=1.0)
    ax.set_title("Hidden dense pH map\n(× = 4 sparse supervision points)", fontsize=10)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("pH", fontsize=9)

    ax = fig.add_subplot(gs[1, 4:6])
    ax.imshow(mask, cmap="gray")
    ax.set_title(f"Tissue mask\n({100*mask.mean():.0f}% of frame)", fontsize=10)
    ax.axis("off")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_ph_distribution(data_dir_glob, out_path, max_samples=400, scope_label=None):
    files = sorted(glob.glob(data_dir_glob))[:max_samples]
    sample_means, within_sds, all_vals = [], [], []
    for f in files:
        ph = np.load(f)
        y = ph[np.isfinite(ph)]
        sample_means.append(y.mean())
        within_sds.append(y.std())
        all_vals.append(y[::37])          # thin for the pooled histogram
    sample_means = np.array(sample_means)
    within_sds = np.array(within_sds)
    pooled = np.concatenate(all_vals)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    ax[0].hist(sample_means, bins=28, color="#8C1D2E", alpha=.85, edgecolor="white")
    ax[0].set_title("Between-sample pH\n(one value per sample)", fontsize=10)
    ax[0].set_xlabel("mean pH of sample"); ax[0].set_ylabel("samples")
    ax[0].text(.03, .93, f"SD = {sample_means.std():.3f}", transform=ax[0].transAxes,
               fontsize=9, va="top")

    ax[1].hist(within_sds, bins=28, color="#1F6F78", alpha=.85, edgecolor="white")
    ax[1].set_title("Within-sample variation\n(SD inside each sample)", fontsize=10)
    ax[1].set_xlabel("pH SD within sample"); ax[1].set_ylabel("samples")
    ax[1].text(.03, .93, f"mean = {within_sds.mean():.3f}", transform=ax[1].transAxes,
               fontsize=9, va="top")

    ax[2].hist(pooled, bins=50, color="0.45", alpha=.85, edgecolor="white")
    ax[2].set_title("All pixels pooled", fontsize=10)
    ax[2].set_xlabel("pH"); ax[2].set_ylabel("pixels (thinned)")
    ax[2].text(.03, .93, f"{pooled.min():.2f} – {pooled.max():.2f}",
               transform=ax[2].transAxes, fontsize=9, va="top")

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8.5)

    # State the scope on the figure. RESULTS.md section 4 quotes the same two
    # statistics computed on the 50-sample VALIDATION split (0.3036 / 0.0843);
    # this figure covers all splits. Without the label the two read as a
    # contradiction rather than as two different scopes.
    if scope_label:
        ax[0].set_ylabel(f"samples\n({scope_label})", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def pick_sample_by_variation(data_dir, n_check=20):
    """
    Objective rule for the Chapter 3 sample figure: among the first `n_check`
    samples in sorted order, take the one with the highest within-sample pH
    SD. A flat sample would not show what the dense map is for, but the pick
    must not be made by scrolling through images until one looks good -- so
    the candidate window is fixed in advance and the winner is arithmetic.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*_phtrue.npy")))[:n_check]
    best, best_sd = None, -1.0
    for f in files:
        y = np.load(f)
        y = y[np.isfinite(y)]
        if y.std() > best_sd:
            best_sd = float(y.std())
            best = os.path.basename(f).replace("_phtrue.npy", "")
    return best, best_sd


def pick_representative_seed(eval_path):
    """
    Objective rule for the Chapter 4 loss-curve figure: the seed whose
    CORRECTED held-out pooled R^2 is closest to the 5-seed mean -- not the
    seed whose curve looks tidiest. Reads the post-metric-fix evaluation
    file, not history.json (whose r2 fields predate the fix).
    """
    d = json.load(open(eval_path))
    pc = d["per_checkpoint"]
    r2 = np.array([c["pooled_r2"] for c in pc])
    seeds = [int(re.search(r"seed_(\d+)", c["checkpoint"]).group(1)) for c in pc]
    i = int(np.argmin(np.abs(r2 - r2.mean())))
    return seeds[i], float(r2[i]), float(r2.mean()), float(r2.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--sample", default=None,
                    help="override the rule-selected Chapter 3 sample")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None,
                    help="override the rule-selected checkpoint")
    ap.add_argument("--sweep-table", default="deconfound_outputs/full_table.json")
    ap.add_argument("--eval-json", default="metric_check_outputs/final_evaluation_TEST500.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    d = os.path.join(a.data, a.split)
    manifest = {}

    # -- which seed and which sample, by rule ---------------------------
    seed, seed_r2, mean_r2, sd_r2 = (None, None, None, None)
    if os.path.exists(a.eval_json):
        seed, seed_r2, mean_r2, sd_r2 = pick_representative_seed(a.eval_json)
        print(f"  representative seed = {seed} "
              f"(R2 {seed_r2:.4f} vs 5-seed mean {mean_r2:.4f} +/- {sd_r2:.4f})")
        manifest["representative_seed"] = dict(seed=seed, seed_r2=seed_r2,
                                               mean_r2=mean_r2, sd_r2=sd_r2)
    ck = a.ckpt or (f"crn_5seed_final/seed_{seed}/crn_best.pt" if seed is not None else None)

    if a.sample:
        sample, sample_sd = a.sample, None
    else:
        sample, sample_sd = pick_sample_by_variation(d)
        print(f"  Chapter 3 sample = {sample} (within-sample SD {sample_sd:.4f}, "
              f"highest of the first 20 in {a.split})")
    manifest["chapter3_sample"] = dict(sample=sample, within_sample_sd=sample_sd)

    # -- Chapter 3 ------------------------------------------------------
    fig_dataset_sample(d, sample, os.path.join(a.out, "fig_dataset_sample.png"))
    fig_ph_distribution(os.path.join(a.data, "*", "*_phtrue.npy"),
                        os.path.join(a.out, "fig_ph_distribution.png"),
                        scope_label="all 400 frozen samples")

    # -- Chapter 4 ------------------------------------------------------
    if os.path.exists(a.sweep_table):
        manifest["sweep"] = fig_sweep_comparison(
            a.sweep_table, os.path.join(a.out, "fig_sweep_comparison.png"))
    else:
        print(f"  skipped sweep figure -- not found: {a.sweep_table}")

    if seed is not None:
        hp = os.path.join("crn_5seed_final", f"seed_{seed}", "history.json")
        if os.path.exists(hp):
            manifest["loss_curves"] = fig_loss_curves(
                hp, os.path.join(a.out, "fig_loss_curves.png"), seed=seed)
        else:
            print(f"  skipped loss curves -- not found: {hp}")

    if ck and os.path.exists(ck):
        fig_system_output(d, sample, ck, os.path.join(a.out, "fig_system_output.png"))
        fig_prediction_gallery(d, ck, os.path.join(a.out, "fig_prediction_gallery.png"))
        manifest["heatmap_example"] = fig_heatmap_example(
            d, ck, os.path.join(a.out, "fig_heatmap_example.png"))
    else:
        print(f"  skipped model figures -- checkpoint not found: {ck}")

    with open(os.path.join(a.out, "figure_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    print(f"\nFigures in {a.out}/ — 200 dpi. Selection rules recorded in "
          f"{a.out}/figure_manifest.json")


def fig_system_output(data_dir, sample_id, ckpt, out_path):
    """
    The deliverable as an operator would see it: a sample goes in, a pH
    map comes out. Chapter 4 / Objective 3. Deliberately NOT the
    diagnostic true/predicted/error view -- this is the product.
    """
    import torch, torch.nn.functional as F
    from crn_model import CrudeCRN

    cube = np.load(os.path.join(data_dir, f"{sample_id}_msi.npy"))
    ph = np.load(os.path.join(data_dir, f"{sample_id}_phtrue.npy"))
    mask = np.isfinite(ph)

    model = CrudeCRN(in_res=256, out_res=256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0)
        pred = model(x)
        if pred.shape[-1] != 256:
            pred = F.interpolate(pred.unsqueeze(1), size=256, mode="bilinear",
                                 align_corners=False).squeeze(1)
    pred = pred.squeeze(0).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.6))
    rgb = np.dstack([cube[:, :, 3], cube[:, :, 1], cube[:, :, 0]])
    rgb = np.clip(rgb / max(rgb.max(), 1e-6), 0, 1)
    ax[0].imshow(rgb); ax[0].axis("off")
    ax[0].set_title("Input: six-band multispectral capture\n(RGB composite shown)", fontsize=10.5)

    im = ax[1].imshow(np.where(mask, pred, np.nan), cmap=CMAP_PH)
    ax[1].axis("off")
    ax[1].set_title("Output: predicted pH distribution", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label("pH", fontsize=9.5)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_prediction_gallery(data_dir, ckpt, out_path, n=4):
    """
    Several samples spanning the range of per-sample accuracy, selected by
    error percentile rather than by eye -- so the figure is representative
    rather than curated. Includes the worst case on purpose.
    """
    import torch, torch.nn.functional as F
    from crn_model import CrudeCRN

    ids = sorted({f.split("_msi.npy")[0] for f in os.listdir(data_dir)
                  if f.endswith("_msi.npy")})
    model = CrudeCRN(in_res=256, out_res=256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    recs = []
    with torch.no_grad():
        for sid in ids:
            cube = np.load(os.path.join(data_dir, f"{sid}_msi.npy"))
            ph = np.load(os.path.join(data_dir, f"{sid}_phtrue.npy"))
            m = np.isfinite(ph)
            x = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0)
            p = model(x)
            if p.shape[-1] != 256:
                p = F.interpolate(p.unsqueeze(1), size=256, mode="bilinear",
                                  align_corners=False).squeeze(1)
            p = p.squeeze(0).numpy()
            recs.append((sid, np.abs(ph[m] - p[m]).mean(), ph, p, m))

    recs.sort(key=lambda r: r[1])
    picks = [recs[0], recs[len(recs)//3], recs[2*len(recs)//3], recs[-1]][:n]
    labels = ["Best case", "Lower quartile", "Upper quartile", "Worst case"][:n]

    fig, ax = plt.subplots(2, n, figsize=(3.1*n, 6.6))
    for j, ((sid, mae, ph, p, m), lbl) in enumerate(zip(picks, labels)):
        lo = np.nanmin(np.where(m, ph, np.nan)); hi = np.nanmax(np.where(m, ph, np.nan))
        ax[0, j].imshow(np.where(m, ph, np.nan), cmap=CMAP_PH, vmin=lo, vmax=hi)
        ax[0, j].set_title(f"{lbl}\nground truth", fontsize=9.5); ax[0, j].axis("off")
        im = ax[1, j].imshow(np.where(m, p, np.nan), cmap=CMAP_PH, vmin=lo, vmax=hi)
        ax[1, j].set_title(f"predicted — MAE {mae:.3f} pH", fontsize=9.5); ax[1, j].axis("off")
        fig.colorbar(im, ax=ax[1, j], fraction=0.046, pad=0.03)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  (MAE range {picks[0][1]:.3f} to {picks[-1][1]:.3f} pH)")

# --------------------------------------------------------------------------
# Manuscript Figure 4.1 -- training / validation loss curves
# --------------------------------------------------------------------------
def fig_loss_curves(hist_path, out_path, seed=None):
    """
    Train vs validation loss per epoch for ONE representative seed, with the
    early-stopping checkpoint marked and the post-checkpoint divergence left
    visible.

    Seed choice is by rule, not by which curve looks cleanest: the seed whose
    corrected held-out pooled R^2 is closest to the 5-seed mean. Passed in by
    main() so the rule lives in one place.

    Plots the SPARSE loss only (what the model is actually trained on).
    Deliberately does NOT plot R^2 from history.json: these histories were
    written by the pre-2026-09-13 training loop, when R^2 was averaged per
    mini-batch rather than pooled, so the r2 fields here are superseded. The
    loss fields are unaffected by that bug.
    """
    path = hist_path
    h = json.load(open(path))
    ep = np.array([e["epoch"] for e in h])
    tr = np.array([e["train_sparse"] for e in h])
    va = np.array([e["val_sparse"] for e in h])
    vloss = np.array([e["val_loss"] for e in h])
    best_i = int(np.argmin(vloss))
    best_ep = ep[best_i]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))

    ax.plot(ep, tr, "-", color=C_TRAIN, lw=2, marker="o", ms=4.5,
            label="training loss", zorder=3)
    ax.plot(ep, va, "--", color=C_VAL, lw=2, marker="s", ms=4.5,
            label="validation loss", zorder=3)

    # post-checkpoint region -- shaded after the axes are scaled to the data
    if best_i < len(ep) - 1:
        ax.axvspan(best_ep, ep[-1], color="0.88", alpha=.55, zorder=0, lw=0)
        ax.text(best_ep + (ep[-1] - best_ep) / 2, ax.get_ylim()[1] * 0.92,
                "after checkpoint", ha="center", va="top",
                fontsize=8, color="0.5", zorder=1)

    ax.axvline(best_ep, color="0.25", lw=1.4, ls=":", zorder=2)
    ax.plot([best_ep], [va[best_i]], "o", ms=11, mfc="none",
            mec="0.15", mew=2, zorder=4)
    ax.annotate(f"checkpoint (epoch {best_ep})",
                xy=(best_ep, va[best_i]), xytext=(26, 12),
                textcoords="offset points", fontsize=8.5,
                ha="left", color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=.9))

    ax.set_xlabel("epoch")
    ax.set_ylabel("sparse-point MSE loss")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(f"Training and validation loss (seed {seed})", fontsize=11, pad=10)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  (seed {seed}, {len(ep)} epochs, checkpoint epoch {best_ep})")
    return dict(seed=seed, epochs=int(len(ep)), best_epoch=int(best_ep),
                train_at_best=float(tr[best_i]), val_at_best=float(va[best_i]),
                val_final=float(va[-1]), history_file=path)


# --------------------------------------------------------------------------
# Manuscript Figure 4.3 -- amplitude sweep, CRN vs conventional baselines
# --------------------------------------------------------------------------
def fig_sweep_comparison(table_path, out_path):
    """
    CRN vs Linear vs PLSR held-out R^2 across the swept denat_amplitude.

    Reads deconfound_outputs/full_table.json -- the CORRECTED, pooled-metric
    table. It does NOT read sweep_outputs/sweep_results.json, which predates
    the 2026-09-13 metric fix and shows a materially different (and wrong)
    picture; see docs/TEAM_LOG.md.

    Error bars on the CRN series are +/- 1 SD across seeds at that amplitude.
    The baselines are closed-form fits with no seed variance, so they carry
    no error bars -- that asymmetry is real, not an omission.
    """
    d = json.load(open(table_path))
    rows = sorted(d["amplitudes"], key=lambda r: r["denat_amplitude"])
    amp = np.array([r["denat_amplitude"] for r in rows])
    lin = np.array([r["linear_held_out_r2"] for r in rows])
    pls = np.array([r["plsr_held_out_r2"] for r in rows])
    crn = np.array([r["crn_r2_mean"] for r in rows])
    sd = np.array([r["crn_r2_std"] for r in rows])
    nseed = [len(r.get("crn_r2_per_seed", [])) for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    ax.errorbar(amp, crn, yerr=sd, color=C_CRN, lw=2, marker="o", ms=8,
                capsize=4, capthick=1.4, elinewidth=1.4,
                label="CRN (mean ± SD across seeds)", zorder=4)
    ax.plot(amp, lin, "-", color=C_LIN, lw=2, marker="s", ms=7.5,
            label="Linear regression", zorder=3)
    ax.plot(amp, pls, "--", color=C_PLS, lw=2, marker="^", ms=7.5,
            label="PLSR", zorder=3)

    # the adopted operating point
    ax.axvline(0.4, color="0.55", lw=1, ls=":", zorder=1)
    ax.annotate("operating point", xy=(0.4, 0.29), xytext=(0.412, 0.275),
                fontsize=8, color="0.45", ha="left")

    for x, y, e in zip(amp, crn, sd):
        ax.annotate(f"{y:.3f}", xy=(x, y + e), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=8.2, color="0.25")

    ax.set_xlabel("denat_amplitude")
    ax.set_ylabel("held-out R²")
    ax.set_title("CRN vs conventional baselines across the parameter sweep",
                 fontsize=11, pad=12)
    ax.set_xticks(amp)
    ax.set_ylim(0.25, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return dict(amplitudes=amp.tolist(), linear=lin.tolist(), plsr=pls.tolist(),
                crn_mean=crn.tolist(), crn_sd=sd.tolist(), n_seeds=nseed)


# --------------------------------------------------------------------------
# Manuscript Figure 4.2 -- true / predicted / |error| for one sample
# --------------------------------------------------------------------------
def fig_heatmap_example(data_dir, ckpt, out_path, prefer=None):
    """
    The diagnostic three-panel view. Sample chosen by rule: the sample whose
    per-sample R^2 is the MEDIAN of the split under this checkpoint, so the
    figure is typical rather than flattering.

    Returns the stats needed to caption it honestly -- including where the
    chosen sample sits in the per-sample R^2 distribution. Note that
    per-sample R^2 is a WITHIN-sample calibration measure and is mostly
    negative here by design; it is not comparable to the pooled headline R^2.
    """
    import torch, torch.nn.functional as F
    from crn_model import CrudeCRN

    ids = sorted({f.split("_msi.npy")[0] for f in os.listdir(data_dir)
                  if f.endswith("_msi.npy")})
    model = CrudeCRN(in_res=256, out_res=256)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    recs = {}
    with torch.no_grad():
        for sid in ids:
            cube = np.load(os.path.join(data_dir, f"{sid}_msi.npy"))
            ph = np.load(os.path.join(data_dir, f"{sid}_phtrue.npy"))
            m = np.isfinite(ph)
            x = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0)
            p = model(x)
            if p.shape[-1] != 256:
                p = F.interpolate(p.unsqueeze(1), size=256, mode="bilinear",
                                  align_corners=False).squeeze(1)
            p = p.squeeze(0).numpy()
            t, q = ph[m], p[m]
            ss_res = float(((t - q) ** 2).sum())
            ss_tot = float(((t - t.mean()) ** 2).sum())
            recs[sid] = dict(r2=1 - ss_res / ss_tot, mae=float(np.abs(t - q).mean()),
                             sd=float(t.std()), ph=ph, pred=p, mask=m)

    order = sorted(recs, key=lambda s: recs[s]["r2"])
    median_sid = order[len(order) // 2]
    sid = prefer if (prefer in recs) else median_sid
    r = recs[sid]
    rank = order.index(sid) + 1

    ph, pred, mask = r["ph"], r["pred"], r["mask"]
    t_masked = np.where(mask, ph, np.nan)
    p_masked = np.where(mask, pred, np.nan)
    err = np.where(mask, np.abs(ph - pred), np.nan)
    vmin, vmax = np.nanmin(t_masked), np.nanmax(t_masked)

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 4.3))
    im0 = ax[0].imshow(t_masked, cmap=CMAP_PH, vmin=vmin, vmax=vmax)
    ax[0].set_title(f"Hidden true pH ({sid})", fontsize=10.5); ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=.046, pad=.03).set_label("pH", fontsize=9)

    im1 = ax[1].imshow(p_masked, cmap=CMAP_PH, vmin=vmin, vmax=vmax)
    ax[1].set_title("CRN prediction", fontsize=10.5); ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=.046, pad=.03).set_label("pH", fontsize=9)

    im2 = ax[2].imshow(err, cmap=CMAP_ERR)
    ax[2].set_title(f"|error|  —  MAE {r['mae']:.3f} pH", fontsize=10.5); ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=.046, pad=.03).set_label("|Δ pH|", fontsize=9)

    fig.text(0.5, 0.03, f"{sid} — median-accuracy case (rank {rank} of {len(order)})",
             ha="center", fontsize=8.5, color="0.4")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({sid}, per-sample R2 {r['r2']:.3f}, MAE {r['mae']:.3f})")

    allr2 = np.array([recs[s]["r2"] for s in order])
    return dict(sample=sid, per_sample_r2=r["r2"], mae=r["mae"],
                within_sample_sd=r["sd"], rank=rank, n=len(order),
                split_median_r2=float(np.median(allr2)),
                split_mean_r2=float(allr2.mean()),
                negative_count=int((allr2 < 0).sum()),
                others={s: dict(r2=recs[s]["r2"], mae=recs[s]["mae"])
                        for s in recs if s in ("sample_351", "sample_301", "sample_370")})


if __name__ == "__main__":
    main()
