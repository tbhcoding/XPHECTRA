"""
check_metric_mismatch.py
=========================

Checks whether the CRN-vs-baseline comparison in deconfound_full_scale.py
is metric-consistent, and specifically whether amp=0.8's "CRN does NOT
beat both by >0.05" verdict survives a like-for-like comparison.

THE ISSUE
---------
deconfound_full_scale.py compares two R^2 values computed differently:

  - Baselines (Linear/PLSR): ONE pooled R^2 over all sampled held-out
    val pixels -- `1 - ss_res/ss_tot` on the concatenated array (and
    sklearn's r2_score, also pooled).
  - CRN: `train_crn.train()`'s `best_val_r2`, which run_epoch() computes
    as a BATCH-AVERAGED value -- per-batch R^2, weighted by batch size,
    averaged over batches.

R^2 is not linear in the data, so these are different quantities. Each
batch's R^2 is measured against the variance *within that batch*, so a
batch whose samples happen to share a narrow pH range scores far lower
for the same absolute error. With 50 val samples at batch_size=8 the
final batch holds only 2 samples, which makes this worse.

Measured on identical amp=0.4 predictions, batch-averaged R^2 came out
~0.067 BELOW pooled R^2 -- i.e. the mismatch is biased AGAINST the CRN.
At amp=0.8 the reported margin over the baselines was only 0.029, so a
bias of that size could flip that row's verdict.

A second, smaller mismatch: the baselines score a 400-pixel-per-sample
random subsample, while the CRN's metric uses every meat pixel. This
script measures that effect too, so the two can be told apart.

WHAT THIS DOES
--------------
For each amplitude, reports the baselines exactly as the original
scripts compute them (unchanged method, reusing their own functions),
and the CRN scored three ways from the SAME checkpoints:

  batch_avg   -- train_crn's own metric (what the original table used)
  pooled_all  -- pooled over every val meat pixel (standard R^2)
  pooled_sub  -- pooled over exactly the baselines' sampled pixels
                 (replicates sample_pixels()'s RNG draws), the truly
                 like-for-like number

amp=0.4 reuses already-trained checkpoints (no retraining). amp=0.8
regenerates its dataset and trains fresh, because neither the .pt
checkpoints nor data_amp_*/ are committed.

Does not modify PARAMS permanently (restored in `finally`), CrudeCRN,
train_crn.py, or any baseline method. Reports numbers only.

Usage:
    python check_metric_mismatch.py
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

import generate_dataset as G
import train_crn
import sweep_denat_amplitude as S
from train_crn import SparseLIGTASDataset
from crn_model import CrudeCRN

GEN_SEED = 42          # matches deconfound_full_scale.py and the frozen dataset
N_TRAIN, N_VAL = 300, 50
CRN_SEEDS = [0, 1, 2]  # same three seeds deconfound_full_scale.py used
BATCH_SIZE = 8         # train_crn.py's default -- required to reproduce batch_avg
IN_RES = OUT_RES = 256
N_POINTS = 4
OUT_ROOT = "metric_check_outputs"
BEAT_MARGIN = 0.05


def pooled_r2(y, p):
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def score_checkpoint(ckpt_path, data_dir):
    """
    Scores one checkpoint three ways on data_dir/val. Returns dict of
    batch_avg / pooled_all / pooled_sub R^2.

    Iterates the val set with shuffle=False at BATCH_SIZE so the
    batch_avg figure reproduces train_crn.run_epoch()'s convention
    exactly, while also retaining per-sample predictions for the two
    pooled figures.
    """
    ds = SparseLIGTASDataset(os.path.join(data_dir, "val"), N_POINTS, IN_RES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model = CrudeCRN(in_res=IN_RES, out_res=OUT_RES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    bw_r2, seen = 0.0, 0
    per_sample = []          # (ph_full HxW, pred_full HxW, mask HxW), dataset order
    all_y, all_p = [], []
    with torch.no_grad():
        for x, coords, values, ph_dense, mask, _ in loader:
            pred = model(x)
            if pred.shape[-1] != IN_RES:
                pred = F.interpolate(pred.unsqueeze(1), size=IN_RES,
                                     mode="bilinear", align_corners=False).squeeze(1)
            ys, ps = [], []
            for b in range(pred.shape[0]):
                m = mask[b]
                ys.append(ph_dense[b][m]); ps.append(pred[b][m])
                per_sample.append((ph_dense[b].numpy(), pred[b].numpy(), m.numpy()))
            y = torch.cat(ys); p = torch.cat(ps)
            all_y.append(y); all_p.append(p)
            # train_crn.run_epoch(): per-batch R^2, weighted by batch size
            bw_r2 += pooled_r2(y, p) * pred.shape[0]
            seen += pred.shape[0]

    y_all = torch.cat(all_y).numpy(); p_all = torch.cat(all_p).numpy()

    # pooled_sub: replicate sample_pixels()'s draws exactly -- same sorted id
    # order, same rng, same rng.choice call -- then index the CRN's prediction
    # at those same pixels instead of the raw cube.
    rng = np.random.default_rng(GEN_SEED)
    sub_y, sub_p = [], []
    for ph_full, pred_full, m in per_sample:
        flat_idx = np.flatnonzero(m)
        n = min(S.PIXELS_PER_SAMPLE, len(flat_idx))
        idx = rng.choice(flat_idx, n, replace=False)
        sub_y.append(ph_full.reshape(-1)[idx])
        sub_p.append(pred_full.reshape(-1)[idx])
    sub_y = np.concatenate(sub_y); sub_p = np.concatenate(sub_p)

    return dict(batch_avg=bw_r2 / seen,
                pooled_all=pooled_r2(y_all, p_all),
                pooled_sub=pooled_r2(sub_y, sub_p))


def baselines(data_dir):
    """Linear + PLSR held-out pooled R^2 -- unchanged from the original scripts."""
    X_tr, y_tr = S.sample_pixels(data_dir, "train", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    X_va, y_va = S.sample_pixels(data_dir, "val", S.PIXELS_PER_SAMPLE, seed=GEN_SEED)
    Xa_tr = np.c_[X_tr, np.ones(len(X_tr))]
    coef, *_ = np.linalg.lstsq(Xa_tr, y_tr, rcond=None)
    lin = pooled_r2(y_va, np.c_[X_va, np.ones(len(X_va))] @ coef)
    pls = PLSRegression(n_components=S.PLSR_N_COMPONENTS)
    pls.fit(X_tr, y_tr)
    plsr = float(r2_score(y_va, pls.predict(X_va).ravel()))
    return lin, plsr


def generate_dataset_at(amp, out_dir):
    """Same generation path deconfound_full_scale.py uses (seed 42, 300/50)."""
    rng = np.random.default_rng(GEN_SEED)
    idx = 1
    for split, count in [("train", N_TRAIN), ("val", N_VAL)]:
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for _ in range(count):
            G.generate_sample(f"sample_{idx:03d}", d, rng)
            idx += 1


def run_amplitude(amp, data_dir, ckpt_paths, train_if_missing):
    print(f"\n{'#' * 72}\n# denat_amplitude = {amp}\n{'#' * 72}", flush=True)
    G.PARAMS["denat_amplitude"]["value"] = amp

    if train_if_missing:
        train_dir = os.path.join(data_dir, "train")
        have = (os.path.isdir(train_dir) and
                len([f for f in os.listdir(train_dir) if f.endswith("_msi.npy")]) == N_TRAIN)
        if have:
            print(f"  reusing dataset at {data_dir}", flush=True)
        else:
            generate_dataset_at(amp, data_dir)
            print(f"  generated {N_TRAIN}/{N_VAL} at {data_dir}", flush=True)
        ckpt_paths = {}
        for s in CRN_SEEDS:
            out = os.path.join(OUT_ROOT, f"crn_amp_{amp}_seed_{s}")
            ck = os.path.join(out, "crn_best.pt")
            if os.path.exists(ck):
                print(f"  reusing checkpoint {ck}", flush=True)
            else:
                args = train_crn.build_argparser().parse_args([
                    "--data", data_dir, "--seed", str(s),
                    "--patience", "10", "--epochs", "50", "--out", out])
                train_crn.train(args)
            ckpt_paths[s] = ck

    lin, plsr = baselines(data_dir)
    print(f"  Linear(held-out, pooled)={lin:.4f}   PLSR(held-out, pooled)={plsr:.4f}", flush=True)

    rows = {}
    for s, ck in sorted(ckpt_paths.items()):
        r = score_checkpoint(ck, data_dir)
        rows[s] = r
        print(f"  seed {s}: batch_avg={r['batch_avg']:.4f}  "
              f"pooled_all={r['pooled_all']:.4f}  pooled_sub={r['pooled_sub']:.4f}", flush=True)

    out = dict(amp=amp, linear_held=lin, plsr_held=plsr, per_seed=rows)
    for key in ("batch_avg", "pooled_all", "pooled_sub"):
        v = np.array([rows[s][key] for s in rows])
        out[f"crn_{key}_mean"] = float(v.mean())
        out[f"crn_{key}_std"] = float(v.std())
        worst = min(lin, plsr)
        out[f"margin_{key}"] = float(v.mean() - max(lin, plsr))
        out[f"beats_{key}"] = bool((v.mean() - lin > BEAT_MARGIN) and (v.mean() - plsr > BEAT_MARGIN))
        print(f"  -> CRN {key:11s} mean={v.mean():.4f} std={v.std():.4f}  "
              f"margin={out[f'margin_{key}']:+.4f}  beats both by >0.05: "
              f"{'YES' if out[f'beats_{key}'] else 'NO'}", flush=True)
    return out


def main():
    original = G.PARAMS["denat_amplitude"]["value"]
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    try:
        # amp=0.4: reuse the frozen dataset + already-trained checkpoints.
        ck04 = {s: f"crn_check/seed_{s}/crn_best.pt" for s in CRN_SEEDS}
        missing = [p for p in ck04.values() if not os.path.exists(p)]
        if missing:
            print(f"SKIPPING amp=0.4 -- missing checkpoints: {missing}")
        else:
            results.append(run_amplitude(0.4, "ligtas_synthetic_dataset", ck04, False))

        # amp=0.8: the row whose verdict is in question. Needs fresh training.
        results.append(run_amplitude(0.8, os.path.join(OUT_ROOT, "data_amp_0.8"),
                                     None, True))
    finally:
        G.PARAMS["denat_amplitude"]["value"] = original
        print(f"\ndenat_amplitude restored to {original}")

    with open(os.path.join(OUT_ROOT, "metric_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 96}\nMETRIC COMPARISON\n{'=' * 96}")
    print(f"{'amp':>5} {'Linear':>8} {'PLSR':>8} | "
          f"{'CRN batch_avg':>14} {'CRN pooled_all':>15} {'CRN pooled_sub':>15} | "
          f"{'verdict batch_avg':>18} {'verdict pooled_sub':>19}")
    for r in results:
        print(f"{r['amp']:>5.1f} {r['linear_held']:>8.4f} {r['plsr_held']:>8.4f} | "
              f"{r['crn_batch_avg_mean']:>14.4f} {r['crn_pooled_all_mean']:>15.4f} "
              f"{r['crn_pooled_sub_mean']:>15.4f} | "
              f"{('YES' if r['beats_batch_avg'] else 'NO'):>18} "
              f"{('YES' if r['beats_pooled_sub'] else 'NO'):>19}")
    print(f"\nWrote {os.path.join(OUT_ROOT, 'metric_comparison.json')}")


if __name__ == "__main__":
    main()
