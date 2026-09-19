"""
crn_demo_ui.py
================

Interactive demo UI -- does NOT modify generate_dataset.py, crn_model.py,
or train_crn.py. Reuses train_crn.py's own SparseLIGTASDataset and
save_heatmap_figure() by import, same pattern already used by
sweep_denat_amplitude.py / deconfound_full_scale.py.

Lets you pick a held-out test sample (by its RGB preview, never seen by
the CRN during training) and a trained checkpoint (one of the 5
crn_5seed_final seeds), then shows the true-pH / predicted-pH /
error-map figure -- the same figure train_crn.py itself saves during
training, just on demand for any test sample instead of one fixed one.

Input is always one of the existing synthetic 6-band samples in
ligtas_synthetic_dataset/test/ -- there is no real-camera pipeline in
this project, so an arbitrary photo cannot be used as input (see
docs/TEAM_LOG.md / README.md's "why this exists" section).

Usage:
    python crn_demo_ui.py
    (opens a local web page, default http://127.0.0.1:7860)
"""

import os
import torch
import gradio as gr
from PIL import Image

from crn_model import CrudeCRN
from train_crn import SparseLIGTASDataset, save_heatmap_figure

DATA_ROOT = "ligtas_synthetic_dataset/test"
CHECKPOINT_DIR = "crn_5seed_final"
AVAILABLE_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SEED = 1  # best val R^2 (0.8388) of the 5 -- see docs/TEAM_LOG.md
IN_RES = 256
OUT_RES = 256
N_POINTS = 4
TMP_OUT = "crn_demo_output_tmp.png"

device = torch.device("cpu")
dataset = SparseLIGTASDataset(DATA_ROOT, n_points=N_POINTS, in_res=IN_RES)
_model_cache = {}


def get_model(seed):
    if seed not in _model_cache:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"seed_{seed}", "crn_best.pt")
        model = CrudeCRN(in_res=IN_RES, out_res=OUT_RES).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        _model_cache[seed] = model
    return _model_cache[seed]


def show_preview(sample_id):
    path = os.path.join(DATA_ROOT, f"{sample_id}_rgb.png")
    return Image.open(path)


def run_prediction(sample_id, seed):
    if not sample_id:
        return None, "Pick a sample first."
    model = get_model(seed)
    sample_idx = dataset.ids.index(sample_id)

    import argparse
    fig_args = argparse.Namespace(in_res=IN_RES, out_res=OUT_RES, n_points=N_POINTS)
    save_heatmap_figure(model, dataset, device, fig_args, TMP_OUT, sample_idx=sample_idx)

    status = (f"Sample: {sample_id}  (held-out test set -- never seen during training)  |  "
              f"Model: crn_5seed_final/seed_{seed}")
    return Image.open(TMP_OUT), status


with gr.Blocks(title="LIGTAS-pH CRN Demo") as demo:
    gr.Markdown(
        "# LIGTAS-pH -- CRN prediction demo\n"
        "Pick a **held-out synthetic test sample** (the model never saw these during "
        "training) and a trained checkpoint, then predict its pH heatmap from the "
        "6-band spectral image alone, given only 4 sparse ground-truth points "
        "(marked with white x's). Input must be one of the existing synthetic "
        "samples -- there is no real camera pipeline in this project."
    )
    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(
                choices=dataset.ids, value=dataset.ids[0], label="Test sample (50 held-out)"
            )
            preview = gr.Image(label="RGB preview of selected sample", value=show_preview(dataset.ids[0]))
            seed_dropdown = gr.Dropdown(
                choices=AVAILABLE_SEEDS, value=DEFAULT_SEED, label="Trained model (crn_5seed_final seed)"
            )
            predict_btn = gr.Button("Predict pH heatmap", variant="primary")
        with gr.Column(scale=2):
            output_image = gr.Image(label="True pH / CRN prediction / error map")
            status = gr.Markdown()

    sample_dropdown.change(fn=show_preview, inputs=sample_dropdown, outputs=preview)
    predict_btn.click(fn=run_prediction, inputs=[sample_dropdown, seed_dropdown],
                       outputs=[output_image, status])

if __name__ == "__main__":
    demo.launch()
