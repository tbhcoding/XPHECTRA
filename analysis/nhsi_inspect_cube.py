"""Render what an NHSI cube contains and what extract_sensor_params.find_meat() selects.

Four panels: band-mean image, the find_meat() mask (top 40% brightest
pixels), the ~970 nm band, and the mean spectrum inside/outside the mask
against the ASSUMED 900-1700 nm linear axis (the .mat stores no wavelength
vector; the ~1450 nm water dip landing near band 300 is the sanity check).

Usage:
    python analysis/nhsi_inspect_cube.py "NHSI dataset/mat_data/01.mat" cube01.png
"""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out = sys.argv[1], sys.argv[2]
with h5py.File(path, "r") as h:
    cube = np.array(h["hyper_image"])          # (B, H, W) as stored
cube = np.transpose(cube, (1, 2, 0))           # same as orient(): bands last
H, W, B = cube.shape
wl = np.linspace(900, 1700, B)                  # the script's ASSUMED axis
i970 = int(np.argmin(np.abs(wl - 970)))

bright = cube.mean(axis=2)
mask = bright > np.percentile(bright, 60)       # identical to find_meat()

fig, ax = plt.subplots(2, 2, figsize=(13, 11))
im = ax[0, 0].imshow(bright, cmap="gray"); ax[0, 0].set_title("band-mean reflectance")
plt.colorbar(im, ax=ax[0, 0], fraction=0.03)
ax[0, 1].imshow(bright, cmap="gray")
ax[0, 1].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.45)
ax[0, 1].set_title(f"find_meat() mask (top 40%) -- {mask.sum()} px")
im = ax[1, 0].imshow(cube[:, :, i970], cmap="viridis", vmin=0, vmax=0.6)
ax[1, 0].set_title(f"band {i970} (assumed {wl[i970]:.0f} nm)")
plt.colorbar(im, ax=ax[1, 0], fraction=0.03)

spec = cube[mask].mean(axis=0)
ax[1, 1].plot(np.arange(B), spec, label="mean over mask")
bg = cube[~mask].mean(axis=0)
ax[1, 1].plot(np.arange(B), bg, label="mean outside mask", alpha=0.6)
ax[1, 1].axvline(i970, color="k", ls="--", lw=0.8, label="assumed 970 nm")
sec = ax[1, 1].secondary_xaxis("top", functions=(lambda b: 900 + b * 800 / (B - 1),
                                                   lambda l: (l - 900) * (B - 1) / 800))
sec.set_xlabel("assumed nm")
ax[1, 1].set_xlabel("band index"); ax[1, 1].legend(); ax[1, 1].set_title("mean spectrum")
fig.tight_layout(); fig.savefig(out, dpi=80)

print(f"shape {cube.shape}, range {cube.min():.3f}..{cube.max():.3f}")
print(f"970 band idx {i970}: mask mean {cube[:, :, i970][mask].mean():.4f}")
print("band of min in mask spectrum:", int(spec.argmin()), f"(assumed {wl[spec.argmin()]:.0f} nm)")
print("spectrum every 25 bands:", np.round(spec[::25], 3))
