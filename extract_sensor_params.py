"""
Extract measurable parameters from the NHSI-meat-overtime dataset
==================================================================

Turns three PLACEHOLDER entries in generate_dataset.py into measured,
citable values:

    sensor_sigma       -> measured read noise
    texture amplitude  -> measured muscle-fibre spatial variation
    970 nm reflectance -> external sanity check on your simulated band

Dataset: Wang, Tang, Li & Chen (2026), NHSI-meat-overtime.
Cubes are NOT in the GitHub repo -- download from the Baidu Pan or
Google Drive link in the README. Get ONE pork timepoint first and
check the file size before pulling the whole set.

Usage:
    python extract_sensor_params.py path/to/pork_cube.mat
    python extract_sensor_params.py path/to/cube.mat --lam-min 900 --lam-max 1700
"""

import sys
import argparse
import numpy as np


def load_cube(path):
    """Load a hyperspectral cube from .mat (v7.3 or older), .npy, or ENVI."""
    if path.endswith(".npy"):
        return np.load(path)

    if path.endswith(".mat"):
        try:
            import h5py
            with h5py.File(path, "r") as h:
                keys = [k for k in h.keys() if not k.startswith("#")]
                arrs = [(k, np.array(h[k])) for k in keys]
                arrs = [(k, a) for k, a in arrs if a.ndim == 3]
                if not arrs:
                    print(f"  no 3-D array found. keys present: {keys}")
                    sys.exit(1)
                k, cube = max(arrs, key=lambda t: t[1].size)
                print(f"  loaded '{k}' from v7.3 .mat")
                return cube
        except (OSError, ImportError):
            import scipy.io as sio
            d = sio.loadmat(path)
            arrs = [(k, v) for k, v in d.items()
                    if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
            if not arrs:
                print(f"  no 3-D array found. keys: {list(d.keys())}")
                sys.exit(1)
            k, cube = max(arrs, key=lambda t: t[1].size)
            print(f"  loaded '{k}' from legacy .mat")
            return cube

    print("  unsupported format. Expected .mat or .npy")
    sys.exit(1)


def orient(cube):
    """h5py may return (bands, W, H). Put bands last."""
    if cube.shape[0] < cube.shape[-1] and cube.shape[0] < 600:
        cube = np.transpose(cube, (1, 2, 0))
        print(f"  transposed to bands-last: {cube.shape}")
    return cube


def find_meat(cube):
    """
    Meat is bright relative to the dark background in NIR.
    Threshold on mean reflectance across bands.
    """
    bright = cube.mean(axis=2)
    thr = np.percentile(bright, 60)
    mask = bright > thr
    print(f"  meat pixels: {mask.sum()} of {mask.size} "
          f"({100*mask.sum()/mask.size:.0f}%)")
    return mask


def flattest_patch(cube, mask, size=24):
    """
    Find the most spatially uniform meat patch. Within a uniform patch,
    remaining pixel-to-pixel variance is dominated by sensor noise,
    not by real tissue structure.
    """
    bright = cube.mean(axis=2)
    H, W = bright.shape
    best, best_var = None, np.inf
    for r in range(0, H - size, size // 2):
        for c in range(0, W - size, size // 2):
            if not mask[r:r+size, c:c+size].all():
                continue
            v = bright[r:r+size, c:c+size].var()
            if v < best_var:
                best_var, best = v, (r, c)
    if best is None:
        print("  no fully-meat patch found; try a smaller --patch")
        sys.exit(1)
    print(f"  flattest patch at row {best[0]}, col {best[1]} ({size}x{size})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("--lam-min", type=float, default=900.0)
    ap.add_argument("--lam-max", type=float, default=1700.0)
    ap.add_argument("--patch", type=int, default=24)
    a = ap.parse_args()

    print("Loading...")
    cube = orient(load_cube(a.cube)).astype(np.float64)
    H, W, B = cube.shape
    print(f"  cube: {H} x {W} x {B} bands")
    print(f"  value range: {cube.min():.4f} to {cube.max():.4f}")
    if cube.max() > 1.5:
        print("  NOTE: values exceed 1.0 -- may be raw DN, not calibrated")
        print("        reflectance. Check the dataset README before using.")
    print()

    wl = np.linspace(a.lam_min, a.lam_max, B)
    mask = find_meat(cube)
    r0, c0 = flattest_patch(cube, mask, a.patch)
    patch = cube[r0:r0+a.patch, c0:c0+a.patch, :]
    print()

    # --- 1. SENSOR NOISE ----------------------------------------------------
    # Remove any smooth gradient across the patch, then take the residual std.
    yy, xx = np.mgrid[0:a.patch, 0:a.patch]
    A = np.c_[yy.ravel(), xx.ravel(), np.ones(a.patch**2)]
    sigmas = []
    for b in range(B):
        y = patch[:, :, b].ravel()
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sigmas.append((y - A @ coef).std())
    sigmas = np.array(sigmas)

    print("=" * 62)
    print("1. SENSOR NOISE")
    print("=" * 62)
    print(f"  median across bands : {np.median(sigmas):.5f}")
    print(f"  range               : {sigmas.min():.5f} to {sigmas.max():.5f}")
    rel = np.median(sigmas) / patch.mean()
    print(f"  relative to signal  : {rel:.5f}")
    print()
    print(f'  -> sensor_sigma value: {rel:.4f}')
    print('     status: "MEASURED"')
    print('     source: "Measured from NHSI-meat-overtime pork cube,')
    print('              Wang et al. 2026"')
    print()

    # --- 2. TEXTURE AMPLITUDE ----------------------------------------------
    # Larger-scale variation across the whole meat region, minus the noise
    # floor, is real muscle structure.
    band_mid = B // 2
    meat_vals = cube[:, :, band_mid][mask]
    total_rel = meat_vals.std() / meat_vals.mean()
    noise_rel = sigmas[band_mid] / meat_vals.mean()
    texture_rel = np.sqrt(max(total_rel**2 - noise_rel**2, 0))

    print("=" * 62)
    print("2. TEXTURE AMPLITUDE")
    print("=" * 62)
    print(f"  total spatial variation : {total_rel:.4f}")
    print(f"  noise floor             : {noise_rel:.4f}")
    print(f"  structure (quadrature)  : {texture_rel:.4f}")
    print()
    print(f"  -> in generate_sample(), replace the hardcoded 0.03 with")
    print(f"     {texture_rel:.4f}  and cite Wang et al. 2026")
    print()

    # --- 3. 970 nm ANCHOR ---------------------------------------------------
    print("=" * 62)
    print("3. 970 nm REALITY CHECK")
    print("=" * 62)
    if a.lam_min <= 970 <= a.lam_max:
        i = int(np.argmin(np.abs(wl - 970)))
        v = cube[:, :, i][mask]
        print(f"  nearest band: index {i} = {wl[i]:.1f} nm")
        print(f"  mean   : {v.mean():.4f}")
        print(f"  std    : {v.std():.4f}")
        print(f"  p5-p95 : {np.percentile(v,5):.4f} to {np.percentile(v,95):.4f}")
        print()
        print("  -> Compare against band 5 of your simulated cubes:")
        print("       cube[:,:,5][mask].mean()")
        print("     Same ballpark = your 970 nm channel is plausible.")
        print("     Way off = revisit water absorption or the K-M constants.")
        print()
        print("     This is ONE band out of six, on real pork, measured by")
        print("     someone else. It is the only external check you have.")
    else:
        print(f"  970 nm outside {a.lam_min}-{a.lam_max} nm. Check the README")
        print("  for the true wavelength axis and pass --lam-min/--lam-max.")
    print()

    print("=" * 62)
    print("CAVEATS FOR YOUR LIMITATIONS SECTION")
    print("=" * 62)
    print("  - Different camera than your intended Arducam OV9281. Noise")
    print("    characteristics transfer only approximately. Say so.")
    print("  - Their NIR sensor differs from a visible-range sensor; noise")
    print("    at 970 nm is not necessarily noise at 481 nm.")
    print("  - Texture measured on their samples, their illumination,")
    print("    their working distance.")
    print()
    print("  These are real limitations. State them plainly -- measured")
    print("  numbers with stated caveats still beat invented ones.")


if __name__ == "__main__":
    main()
