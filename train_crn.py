"""
Priority 2 -- crude CRN training on sparse-point supervision + TV smoothness
=============================================================================

Team decision (see TEAM_LOG.md, Part B): a 4-point sparse loss alone is
underdetermined -- 4 labels vs. up to 65,536 pixels in a 256x256 map --
so a total-variation smoothness regularizer on the full predicted map is
REQUIRED, not optional, alongside the sparse point loss.

This is a crude first pass. Goal: one working predicted heatmap and a
loss curve to sanity-check that the approach converges at all, not a
tuned model. See crn_model.py for the network.

Configurable, per Part B requirements:
  --out-res    prediction grid resolution (default: full input resolution,
                256). A value < in-res predicts a coarser, area-averaged
                pH grid instead -- see CrudeCRN's adaptive-pool step.
  --n-points   number of supervised sparse points per sample (default 4).
               NOTE: the current dataset only stores 4 probe points per
               sample (one per quadrant -- see generate_dataset.py
               generate_sample()). Requesting more than 4 here does NOT
               create new data: it warns and clamps to 4. True 8/16/32
               -point ablation requires regenerating the dataset with
               more embedded probes first -- an explicit follow-up, not
               done in this pass.
  --tv-weight  weight on the total-variation smoothness term.

Do NOT add a dataset-size ablation (400/800/1600/3200) here -- that is a
follow-up experiment after this crude version trains successfully, not
part of this pass.

Usage:
    python train_crn.py --data ligtas_synthetic_dataset --epochs 15
    python train_crn.py --out-res 32 --n-points 4   # coarse-grid variant
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from crn_model import CrudeCRN

MAX_AVAILABLE_POINTS = 4  # see docstring -- what generate_dataset.py actually stores


class SparseLIGTASDataset(Dataset):
    """
    Loads (cube, sparse points) for training and, separately, the hidden
    dense pH map for evaluation-only use. The training loop must not let
    `ph_dense` influence gradients -- see run_epoch().
    """

    def __init__(self, root, n_points, in_res):
        self.root = root
        self.n_points = n_points
        self.in_res = in_res
        self.ids = sorted({f.split("_msi.npy")[0] for f in os.listdir(root)
                            if f.endswith("_msi.npy")})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        cube = np.load(os.path.join(self.root, f"{sid}_msi.npy"))       # (H,W,6)
        ph_raw = np.load(os.path.join(self.root, f"{sid}_phtrue.npy"))  # (H,W), NaN outside mask
        with open(os.path.join(self.root, f"{sid}_labels.json")) as f:
            meta = json.load(f)

        pts = meta["sparse_measurements"][:self.n_points]
        coords = np.array([p["pixel_coord"] for p in pts], dtype=np.int64)
        values = np.array([p["ph_value"] for p in pts], dtype=np.float32)

        mask_np = np.isfinite(ph_raw)
        x = torch.from_numpy(cube.transpose(2, 0, 1)).float()
        ph_dense = torch.from_numpy(np.nan_to_num(ph_raw, nan=0.0)).float()
        mask = torch.from_numpy(mask_np)
        coords = torch.from_numpy(coords)
        values = torch.from_numpy(values)

        return x, coords, values, ph_dense, mask, sid


def map_coords_to_grid(coords, in_res, out_res):
    """Full-res pixel coords (row, col) -> nearest index on the out_res grid."""
    if out_res == in_res:
        return coords
    scaled = (coords.float() * (out_res / in_res)).long()
    return scaled.clamp(0, out_res - 1)


def sparse_loss(pred, coords_grid, values):
    """pred: (B, out_res, out_res); coords_grid: (B, P, 2); values: (B, P)."""
    losses = []
    for b in range(pred.shape[0]):
        rr, cc = coords_grid[b, :, 0], coords_grid[b, :, 1]
        pred_pts = pred[b, rr, cc]
        losses.append(F.mse_loss(pred_pts, values[b]))
    return torch.stack(losses).mean()


def tv_loss(pred):
    """Total variation over the predicted grid -- the smoothness prior
    that makes the sparse-point problem well-posed."""
    dy = (pred[:, 1:, :] - pred[:, :-1, :]).abs().mean()
    dx = (pred[:, :, 1:] - pred[:, :, :-1]).abs().mean()
    return dy + dx


def full_field_eval(pred, ph_dense, mask, in_res):
    """
    Sanity check ONLY -- never used for gradients. Upsamples pred to full
    resolution (if it's on a coarser grid) so it can be compared against
    the hidden dense map. MAE + R^2 over meat pixels.
    """
    if pred.shape[-1] != in_res:
        pred = F.interpolate(pred.unsqueeze(1), size=in_res, mode="bilinear",
                              align_corners=False).squeeze(1)
    diffs, y_all, p_all = [], [], []
    for b in range(pred.shape[0]):
        m = mask[b]
        y = ph_dense[b][m]
        p = pred[b][m]
        diffs.append((y - p).abs())
        y_all.append(y); p_all.append(p)
    mae = torch.cat(diffs).mean().item()
    y_all = torch.cat(y_all); p_all = torch.cat(p_all)
    ss_res = ((y_all - p_all) ** 2).sum()
    ss_tot = ((y_all - y_all.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return mae, r2


def run_epoch(model, loader, opt, device, args, train=True):
    model.train(train)
    total_sparse, total_tv, total_mae, total_r2, n = 0.0, 0.0, 0.0, 0.0, 0
    for x, coords, values, ph_dense, mask, _ in loader:
        x = x.to(device)
        coords_grid = map_coords_to_grid(coords, args.in_res, args.out_res).to(device)
        values = values.to(device)
        ph_dense, mask = ph_dense.to(device), mask.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            l_sparse = sparse_loss(pred, coords_grid, values)
            l_tv = tv_loss(pred)
            loss = l_sparse + args.tv_weight * l_tv
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        bs = x.shape[0]
        total_sparse += l_sparse.item() * bs
        total_tv += l_tv.item() * bs
        mae, r2 = full_field_eval(pred.detach(), ph_dense, mask, args.in_res)
        total_mae += mae * bs
        total_r2 += r2 * bs
        n += bs

    return total_sparse / n, total_tv / n, total_mae / n, total_r2 / n


def save_heatmap_figure(model, dataset, device, args, out_path, sample_idx=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, coords, values, ph_dense, mask, sid = dataset[sample_idx]
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
    if pred.shape[-1] != args.in_res:
        pred = F.interpolate(pred[None, None], size=args.in_res, mode="bilinear",
                              align_corners=False)[0, 0]

    pred_masked = torch.where(mask, pred, torch.nan)
    true_masked = torch.where(mask, ph_dense, torch.nan)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    vmin = min(ph_dense[mask].min().item(), pred[mask].min().item())
    vmax = max(ph_dense[mask].max().item(), pred[mask].max().item())

    im0 = ax[0].imshow(true_masked, cmap="turbo", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"hidden true pH ({sid})"); ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(pred_masked, cmap="turbo", vmin=vmin, vmax=vmax)
    ax[1].set_title(f"CRN prediction (out_res={args.out_res}, n_points={args.n_points})")
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    rr, cc = coords[:, 0].numpy(), coords[:, 1].numpy()
    ax[1].scatter(cc, rr, c="white", edgecolors="black", s=30, marker="x")

    err = torch.abs(true_masked - pred_masked)
    im2 = ax[2].imshow(err, cmap="magma")
    ax[2].set_title("|error|"); ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved heatmap comparison to {out_path}")


def save_loss_curve(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h["train_sparse"] for h in history], label="train sparse MSE")
    ax[0].plot(epochs, [h["val_sparse"] for h in history], label="val sparse MSE")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("sparse point loss (what the model is actually trained on)")

    ax[1].plot(epochs, [h["train_mae"] for h in history], label="train MAE")
    ax[1].plot(epochs, [h["val_mae"] for h in history], label="val MAE")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("pH MAE"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("full-field MAE vs hidden dense map (sanity check only)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ligtas_synthetic_dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                     help="L2 weight decay on Adam; 0.0 = none (current default, "
                          "no regularization beyond the TV term)")
    ap.add_argument("--in-res", type=int, default=256,
                     help="input cube resolution (matches the generator's H=W)")
    ap.add_argument("--out-res", type=int, default=256,
                     help="prediction grid resolution; < in-res = coarser, "
                          "area-averaged pH grid")
    ap.add_argument("--n-points", type=int, default=4,
                     help="sparse supervised points per sample; dataset "
                          "currently has 4 max -- see module docstring")
    ap.add_argument("--tv-weight", type=float, default=0.05,
                     help="weight on the total-variation smoothness term; "
                          "required because a handful of points alone "
                          "underdetermine a dense map")
    ap.add_argument("--out", default="crn_outputs")
    args = ap.parse_args()

    if args.n_points > MAX_AVAILABLE_POINTS:
        print(f"WARNING: --n-points={args.n_points} requested, but the "
              f"dataset only embeds {MAX_AVAILABLE_POINTS} probes per sample "
              f"(generate_dataset.py generate_sample()). Clamping to "
              f"{MAX_AVAILABLE_POINTS}. Ablating past that requires "
              f"regenerating the dataset with more embedded probes first -- "
              f"a follow-up, not done here.")
        args.n_points = MAX_AVAILABLE_POINTS

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  out_res: {args.out_res}  n_points: {args.n_points}  "
          f"tv_weight: {args.tv_weight}")

    train_ds = SparseLIGTASDataset(os.path.join(args.data, "train"), args.n_points, args.in_res)
    val_ds = SparseLIGTASDataset(os.path.join(args.data, "val"), args.n_points, args.in_res)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CrudeCRN(in_res=args.in_res, out_res=args.out_res).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print()
    print("Loss = sparse-point MSE + tv_weight * total-variation on the full")
    print("predicted map. The 'MAE/R2 vs hidden dense map' columns are a")
    print("SANITY CHECK only -- phtrue.npy is never used for gradients.\n")

    history = []
    best_val_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_sparse, tr_tv, tr_mae, tr_r2 = run_epoch(model, train_loader, opt, device, args, train=True)
        val_sparse, val_tv, val_mae, val_r2 = run_epoch(model, val_loader, opt, device, args, train=False)
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train: sparse={tr_sparse:.4f} tv={tr_tv:.4f}  "
              f"[hidden-map] MAE={tr_mae:.3f} R2={tr_r2:.3f}  |  "
              f"val: sparse={val_sparse:.4f} tv={val_tv:.4f} MAE={val_mae:.3f} R2={val_r2:.3f}")
        history.append(dict(epoch=epoch, train_sparse=tr_sparse, val_sparse=val_sparse,
                             train_mae=tr_mae, val_mae=val_mae,
                             train_r2=tr_r2, val_r2=val_r2))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.out, "crn_best.pt"))

    save_loss_curve(history, os.path.join(args.out, "loss_curve.png"))

    # Reload the BEST checkpoint (lowest val MAE seen during training) before
    # generating the reported heatmap -- the in-memory `model` here is
    # whatever the LAST epoch left it at, which is not necessarily the best
    # one and was silently mismatched with "best_val_mae" in earlier runs.
    model.load_state_dict(torch.load(os.path.join(args.out, "crn_best.pt"), map_location=device))
    save_heatmap_figure(model, val_ds, device, args, os.path.join(args.out, "sanity_check_heatmap.png"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val MAE (hidden-map check) = {best_val_mae:.3f} pH units.")
    print(f"Checkpoint, loss curve, heatmap, history in {args.out}/")


if __name__ == "__main__":
    main()
