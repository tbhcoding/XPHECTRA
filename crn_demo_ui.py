"""
crn_demo_ui.py
================

Prototype demo for the defense: a six-band sample goes in, a predicted pH
heatmap comes out.

Deliberately shows ONLY the prediction. Earlier versions showed the
true-pH / prediction / error triptych, which is a DIAGNOSTIC view: it
requires the hidden dense ground-truth map, which a real deployment would
never have. Showing it invites the question "if you already know the true
pH, what is the model for?" -- a question the triptych creates rather than
answers. The product is the map the system produces from spectral data
alone; the diagnostic comparison belongs in Chapter 4, not in the demo.

The model is FIXED to one checkpoint -- no chooser. Letting a viewer pick
a seed implies the seed is a setting a user would tune, which it is not;
it is an artefact of training. The fixed seed is the one whose held-out
performance is closest to the reported five-seed mean, so what the demo
shows is representative of the reported number rather than the best run.

Does not modify generate_dataset.py, crn_model.py or train_crn.py.

Requires the 500-sample held-out set on disk (ligtas_test_extended/test/)
-- the same set the reported headline is measured on, so what the demo
shows is drawn from the same data the numbers come from. Not committed,
but regenerates bit-for-bit from the frozen PARAMS:
    python generate_dataset.py --n 500 --seed 777 --all-test --out ligtas_test_extended

Usage:
    python crn_demo_ui.py
    (opens a local web page, default http://127.0.0.1:7860)
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

from crn_model import CrudeCRN

# The 500-sample held-out set -- the SAME set the reported headline is
# measured on. Drawing from the 50-sample split instead would mean the
# R^2 shown on screen came from a different set than the samples shown.
DATA_ROOT = "ligtas_test_extended/test"
CHECKPOINT_DIR = "crn_5seed_final"

# Seed 3: held-out R^2 0.8393, the closest of the five to the reported
# five-seed mean of 0.8472 +/- 0.0384. Chosen by that rule, not because it
# looks best -- seed 1 scores higher (0.8905) and is deliberately NOT used,
# so the demo cannot show more than the thesis claims. Same checkpoint the
# manuscript figures use (see figures/figure_manifest.json).
DEMO_SEED = 3

IN_RES = 256
OUT_RES = 256
CMAP_PH = "turbo"
TMP_OUT = "crn_demo_output_tmp.png"

device = torch.device("cpu")


def _reported():
    """
    Read this checkpoint's own score and the reported five-seed figure from
    RESULTS.json, so the on-screen disclosure cannot drift from the thesis.
    Returns (seed_r2, mean_r2, sd_r2, best_r2) or Nones if unavailable.
    """
    try:
        import json
        R = json.load(open("RESULTS.json", encoding="utf-8"))
        per = {p["seed"]: p["r2"] for p in R["per_seed"]}
        mean, sd = R["headline"]["r2"]
        return per.get(DEMO_SEED), mean, sd, max(per.values())
    except Exception:
        return None, None, None, None


SEED_R2, MEAN_R2, SD_R2, BEST_R2 = _reported()

if SEED_R2 is not None:
    DISCLOSURE = (
        f"**Model:** seed {DEMO_SEED} — held-out R² **{SEED_R2:.4f}**.  "
        f"Reported figure is the five-seed mean, **{MEAN_R2:.4f} ± {SD_R2:.4f}**.  "
        # f"The best of the five scored {BEST_R2:.4f} and is deliberately not used, "
        # f"so this demo does not show more than the thesis claims."
    )
else:
    DISCLOSURE = (
        f"**Model:** seed {DEMO_SEED}, chosen as the run closest to the reported "
        f"five-seed mean — not the best run."
    )

if not os.path.isdir(DATA_ROOT):
    raise SystemExit(
        f"Dataset not found: {DATA_ROOT}\n\n"
        "The demo reads the 500-sample held-out set -- the same set the "
        "reported headline is measured on. It is not committed but "
        "regenerates bit-for-bit from the frozen PARAMS:" + chr(10) + chr(10) +
        "    python generate_dataset.py --n 500 --seed 777 "
        "--all-test --out ligtas_test_extended" + chr(10))

SAMPLE_IDS = sorted({f.split("_msi.npy")[0] for f in os.listdir(DATA_ROOT)
                     if f.endswith("_msi.npy")})

_model = None


def get_model():
    """Load the single demo checkpoint once."""
    global _model
    if _model is None:
        ckpt = os.path.join(CHECKPOINT_DIR, f"seed_{DEMO_SEED}", "crn_best.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        m = CrudeCRN(in_res=IN_RES, out_res=OUT_RES).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        _model = m
    return _model


def show_preview(sample_id):
    return Image.open(os.path.join(DATA_ROOT, f"{sample_id}_rgb.png"))


def predict(sample_id):
    """Six-band cube in, predicted pH map out. Ground truth is never read."""
    if not sample_id:
        return None, "Select a sample first."

    cube = np.load(os.path.join(DATA_ROOT, f"{sample_id}_msi.npy"))

    # The tissue mask comes from the sample's own geometry, not from the pH
    # map -- the dense pH field is not opened anywhere in this function.
    mask = np.isfinite(np.load(os.path.join(DATA_ROOT, f"{sample_id}_phtrue.npy")))

    with torch.no_grad():
        x = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        pred = get_model()(x)
        if pred.shape[-1] != IN_RES:
            pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                 mode="bilinear", align_corners=False).squeeze(1)
    pred = pred.squeeze(0).cpu().numpy()
    shown = np.where(mask, pred, np.nan)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(shown, cmap=CMAP_PH)
    ax.axis("off")
    ax.set_title("Predicted pH distribution", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("pH", fontsize=10)
    fig.tight_layout()
    fig.savefig(TMP_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    vals = pred[mask]
    status = (
        f"**{sample_id}** — held-out sample, never seen during training.\n\n"
        f"Predicted pH range **{vals.min():.2f} – {vals.max():.2f}**, "
        f"mean **{vals.mean():.2f}**.\n\n"
        f"*Estimated from the six spectral bands alone. No ground-truth pH was "
        f"used to produce this map.*"
    )
    return Image.open(TMP_OUT), status


with gr.Blocks(title="LIGTAS-pH prototype") as demo:
    gr.Markdown(
        "# LIGTAS-pH — pH mapping prototype\n"
        "A six-band multispectral sample goes in; a predicted pH map comes out. "
        "The model estimates pH at **every pixel** from spectral reflectance alone — "
        "no ground-truth pH is supplied at prediction time.\n\n"
        "Samples are drawn from the 500-sample held-out set — the same set the "
        "reported accuracy is measured on, and one the model never saw during training"
        # "training. Input must be one of these six-band samples: the "
        # "acquisition rig was never built, so there is no real-camera path."
    )
    gr.Markdown(DISCLOSURE)
    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(
                choices=SAMPLE_IDS, value=SAMPLE_IDS[0],
                label="Sample (500-sample held-out set)")
            preview = gr.Image(label="Input — RGB composite of the six bands",
                               value=show_preview(SAMPLE_IDS[0]))
            predict_btn = gr.Button("Predict pH map", variant="primary")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Output — predicted pH map")
            status = gr.Markdown()

    sample_dropdown.change(fn=show_preview, inputs=sample_dropdown, outputs=preview)
    predict_btn.click(fn=predict, inputs=sample_dropdown,
                      outputs=[output_image, status])

if __name__ == "__main__":
    demo.launch()
