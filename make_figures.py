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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVELENGTHS = [481, 525, 573, 600, 730, 970]
DPI = 200


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
    im = ax.imshow(np.where(mask, ph, np.nan), cmap="turbo")
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

    res = meta.get("resolution", "256×256")
    dims = meta.get("roi_dimensions_cm", "")
    fig.text(0.5, 0.015,
             f"{sample_id} — {res} px, ROI {dims} cm. The model receives only the six bands; "
             f"the dense pH map is never seen during training.",
             ha="center", fontsize=8.5, color="0.35")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_ph_distribution(data_dir_glob, out_path, max_samples=400):
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

    ratio = sample_means.std() / within_sds.mean()
    fig.text(0.5, -0.06,
             f"Between-sample spread is {ratio:.1f}× larger than within-sample spread. "
             f"pH is drawn uniformly per sample by design, for even coverage of the "
             f"physiological range — not to mimic a population distribution.",
             ha="center", fontsize=8.5, color="0.35")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--sample", default="sample_351")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="crn_5seed_final/seed_1/crn_best.pt")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    d = os.path.join(a.data, a.split)

    fig_dataset_sample(d, a.sample, os.path.join(a.out, "fig_dataset_sample.png"))
    fig_ph_distribution(os.path.join(a.data, "*", "*_phtrue.npy"),
                        os.path.join(a.out, "fig_ph_distribution.png"))
    ck = a.ckpt
    if os.path.exists(ck):
        fig_system_output(d, a.sample, ck, os.path.join(a.out, "fig_system_output.png"))
        fig_prediction_gallery(d, ck, os.path.join(a.out, "fig_prediction_gallery.png"))
    else:
        print(f"  skipped model figures -- checkpoint not found: {ck}")
    print(f"\nFigures in {a.out}/ — 200 dpi, ready for the manuscript.")




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

    im = ax[1].imshow(np.where(mask, pred, np.nan), cmap="turbo")
    ax[1].axis("off")
    ax[1].set_title("Output: predicted pH distribution", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label("pH", fontsize=9.5)

    fig.text(0.5, 0.02, "The system requires no ground-truth pH at inference; "
                        "the map is estimated from spectral reflectance alone.",
             ha="center", fontsize=8.5, color="0.35")
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
        ax[0, j].imshow(np.where(m, ph, np.nan), cmap="turbo", vmin=lo, vmax=hi)
        ax[0, j].set_title(f"{lbl}\nground truth", fontsize=9.5); ax[0, j].axis("off")
        im = ax[1, j].imshow(np.where(m, p, np.nan), cmap="turbo", vmin=lo, vmax=hi)
        ax[1, j].set_title(f"predicted — MAE {mae:.3f} pH", fontsize=9.5); ax[1, j].axis("off")
        fig.colorbar(im, ax=ax[1, j], fraction=0.046, pad=0.03)

    fig.text(0.5, 0.015, "Samples selected by per-sample error percentile, not by inspection. "
                         "Each column shares a colour scale between truth and prediction.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  (MAE range {picks[0][1]:.3f} to {picks[-1][1]:.3f} pH)")

if __name__ == "__main__":
    main()
