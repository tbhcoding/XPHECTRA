"""970 nm across local NHSI cubes: original find_meat() method vs. a
water-band tissue mask inside the tray, broken down by tray column.

Tissue mask: meat is ~73% water and shows a deep ~1450 nm absorption dip;
the tray/background is spectrally flat. Index = mean R(~1010-1090 nm) -
mean R(~1440-1500 nm), Otsu-thresholded inside the tray. (A brightness-only
Otsu split bright tissue from lean tissue instead of tissue from background.)

Tray layout (HSI frame, from cube 01 render; RGB photo is rotated 90 deg):
  C1 x<137   : 6 small pale pieces  -> chicken (RGB bottom row)
  C2 137-292 : 2 long red pieces    -> RGB row 4
  C3 292-492 : fatty mixed pieces   -> RGB row 3
  C4 492-660 : 3 pink slices        -> RGB row 2
  C5 x>=660  : striped fillet       -> salmon (RGB top row)
Species of C2-C4 is NOT established here.

Reads each cube in one go (~770 MB); per-band h5py slicing re-decompresses
chunks and is far slower. All 18 cubes take several minutes.

Usage:
    python analysis/nhsi_970_breakdown.py "NHSI dataset/mat_data" <out_dir> [glob, default *.mat]
Writes tissue_masks.png (cubes 01/09/19) to <out_dir> -- check it before
trusting the numbers.
"""
import glob, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B970 = 38                       # int(argmin |linspace(900,1700,431) - 970|)
NIR_ON = slice(60, 100)         # ~1010-1090 nm (assumed axis)
WATER = slice(290, 320)         # ~1440-1500 nm
EDGES = [0, 137, 292, 492, 660, 811]
NAMES = ["C1 chicken", "C2 row4", "C3 fatty", "C4 row2", "C5 salmon"]
Y0, Y1 = 85, 480                # tray rows; excludes the bottom strip
SHOW = ("01", "09", "19")


def otsu(x, nbins=256):
    hist, e = np.histogram(x, bins=nbins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


out_dir = sys.argv[2]
pattern = sys.argv[3] if len(sys.argv) > 3 else "*.mat"
files = sorted(glob.glob(os.path.join(sys.argv[1], pattern)))
rows, keep = [], {}
for f in files:
    with h5py.File(f, "r") as h:
        cube = np.array(h["hyper_image"])       # (B, H, W); one read -- per-band slicing re-decompresses chunks
    full_mean = cube.mean(axis=0, dtype=np.float64)
    b970 = cube[B970].astype(np.float64)
    index = cube[NIR_ON].mean(axis=0, dtype=np.float64) - cube[WATER].mean(axis=0, dtype=np.float64)
    del cube
    # identical to extract_sensor_params.find_meat() (after its transpose)
    orig_mask = full_mean > np.percentile(full_mean, 60)
    orig = b970[orig_mask].mean()

    tray = np.zeros_like(orig_mask); tray[Y0:Y1, :] = True
    thr = otsu(index[tray])
    tissue = tray & (index > thr)
    cols = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = tissue.copy(); m[:, :a] = False; m[:, b:] = False
        cols.append(b970[m].mean())
    name = os.path.basename(f)[:2]
    rows.append((name, orig, b970[tissue].mean(), *cols, orig_mask[Y1:].mean(), thr))
    if name in SHOW:
        keep[name] = (full_mean, tissue)
    print(name, "done", flush=True)

hdr = (f"{'cube':>4} {'orig':>6} {'tissue':>7} " + " ".join(f"{n:>10}" for n in NAMES)
       + f" {'orig%bgstrip':>12} {'idx_thr':>7}")
print("\n" + hdr)
for r in rows:
    print(f"{r[0]:>4} {r[1]:6.4f} {r[2]:7.4f} " + " ".join(f"{v:10.4f}" for v in r[3:8])
          + f" {100*r[8]:11.1f}% {r[9]:7.4f}")
arr = np.array([r[1:8] for r in rows])
if len(rows) > 1:
    print(f"{'mean':>4} {arr[:,0].mean():6.4f} {arr[:,1].mean():7.4f} " + " ".join(f"{v:10.4f}" for v in arr[:, 2:].mean(0)))
    print(f"{'sd':>4} {arr[:,0].std(ddof=1):6.4f} {arr[:,1].std(ddof=1):7.4f} " + " ".join(f"{v:10.4f}" for v in arr[:, 2:].std(0, ddof=1)))

if keep:
    fig, ax = plt.subplots(1, len(keep), figsize=(7 * len(keep), 5), squeeze=False)
    for a, (k, (fm, t)) in zip(ax[0], sorted(keep.items())):
        a.imshow(fm, cmap="gray")
        a.imshow(np.ma.masked_where(~t, t), cmap="autumn", alpha=0.45)
        for e in EDGES[1:-1]:
            a.axvline(e, color="c", lw=1)
        a.set_title(f"cube {k}: water-band tissue mask")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tissue_masks.png"), dpi=80)
