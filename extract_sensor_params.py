"""
Extract measurable parameters from the NHSI-meat-overtime dataset
==================================================================

Pulls three things out of an NHSI cube (a MIXED TRAY -- chicken, salmon
and three unidentified red-meat/fat columns; NOT established as pork):

    sensor_sigma       -> measured additive read noise  (USED: 0.0054)
    texture amplitude  -> fine-scale surface structure  (see caveat below)
    970 nm reflectance -> external reality check on the simulated 970 nm band

TWO MASKS, ON PURPOSE (2026-09-13, verified on real cubes 01 and 19 --
see docs/TEAM_LOG.md). These three measurements do not want the same
pixels, and using one mask for all of them caused a real mislabelling:

  - sensor_sigma wants a UNIFORM, UNTEXTURED region. flattest_patch()
    over the brightness mask finds exactly that -- and on cube 01 the
    patch it picks is 0% tissue (bare tray). That is CORRECT for read
    noise: absolute residual sd is 0.00039 on tray vs 0.00377 on muscle,
    and that 10x gap is biological micro-texture, not electronics.
    The published 0.0054 therefore stands, but it was NOT "measured on
    pork" as this file and PARAMS previously claimed.
  - texture and the 970 nm check want TISSUE ONLY, so they use the
    water-band mask (find_meat_waterband).

Switching sensor_sigma to the tissue mask would give ~0.035 (cube 01) or
~0.057 (cube 19) -- a different physical quantity that double-counts what
texture_amplitude was meant to model. Do not do it without a team
decision; see the 2026-09-13 TEAM_LOG entry.

TEXTURE CAVEAT (2026-09-03): the whole-region spatial std of these cubes is
~0.45-0.60, but that is fat seams / muscle groups / surface geometry / drip
on a rough, aged, whole muscle imaged at 900-1700 nm -- NOT the fine
marbling/fibre mottle that generate_dataset.py's texture_amplitude models,
and not the surface a trimmed 5x5x2.5 cm chop presents. Within a genuinely
uniform 24x24 patch, residual variation is ~0.005 relative -- essentially
sensor noise. So texture_amplitude is kept at 0.030 as an ASSUMED value
anchored to that smooth-patch floor, not set from the numbers below.

970 nm CHECK (2026-09-03): real NHSI mean ~0.19 vs simulated ~0.54 -- a ~3x
gap, still unresolved. See the Parameterized Data Sheet.

Dataset: Wang, Tang, Li & Chen (2026), NHSI-meat-overtime.
Cubes are NOT in the GitHub repo -- download separately. Get ONE pork
timepoint first and check the file size before pulling the whole set.

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


def _otsu(x, nbins=256):
    """Otsu threshold. Same implementation as analysis/nhsi_970_breakdown.py."""
    hist, e = np.histogram(x, bins=nbins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def find_meat_brightness(cube):
    """
    ORIGINAL method: top-40%-brightest pixels. KEPT so the published
    sensor_sigma = 0.0054 (NHSI cube 01) stays exactly reproducible --
    do not delete.

    KNOWN FAILURE MODE (measured 2026-09-11, docs/TEAM_LOG.md): this assumes
    meat occupies ~40% of the frame. As tissue darkens over the NHSI time
    series it does not: from cube 03 on, this mask pulls 62-71% of the known
    background strip in. Cubes 01/02 were unaffected, which is why the
    published value stands. Prefer find_meat_waterband() for anything else.
    """
    bright = cube.mean(axis=2)
    mask = bright > np.percentile(bright, 60)
    print(f"  meat pixels (brightness, top 40%): {mask.sum()} of {mask.size} "
          f"({100*mask.sum()/mask.size:.0f}%)")
    return mask


def find_meat_waterband(cube, wl, tray_rows=None):
    """
    Tissue mask via the ~1450nm water absorption band -- meat is ~73% water
    and shows a deep dip there, while tray/background is spectrally flat.
    Index = mean R(1010-1090nm) - mean R(1440-1500nm), Otsu-thresholded.

    Ported from analysis/nhsi_970_breakdown.py, where it was checked visually
    on cubes 01/09/19: covers every piece, excludes background. A
    brightness-only Otsu variant was tried there first and rejected (it split
    bright tissue from lean tissue instead of tissue from background).

    Requires a wavelength axis covering both bands; returns None if not.
    tray_rows=(y0, y1) optionally restricts to the tray (NHSI: 85, 480).
    """
    if wl.min() > 1010 or wl.max() < 1500:
        return None
    on = (wl >= 1010) & (wl <= 1090)
    water = (wl >= 1440) & (wl <= 1500)
    if on.sum() == 0 or water.sum() == 0:
        return None

    index = cube[:, :, on].mean(axis=2) - cube[:, :, water].mean(axis=2)
    region = np.zeros(index.shape, dtype=bool)
    if tray_rows is None:
        region[:] = True
    else:
        region[tray_rows[0]:tray_rows[1], :] = True
    thr = _otsu(index[region])
    mask = region & (index > thr)
    print(f"  meat pixels (water-band, Otsu thr={thr:.4f}): {mask.sum()} of "
          f"{mask.size} ({100*mask.sum()/mask.size:.0f}%)")
    return mask


def find_meat(cube, wl=None, method="auto", tray_rows=None):
    """
    Returns the tissue mask, preferring the water-band method when the
    wavelength axis supports it. Also reports how much the two methods
    disagree, which is the background-leak diagnostic.
    """
    want_wb = method in ("auto", "waterband")
    wb = find_meat_waterband(cube, wl, tray_rows) if (want_wb and wl is not None) else None

    if method == "waterband" and wb is None:
        print("  ERROR: --mask waterband requested but the wavelength axis does")
        print("         not cover 1010-1090nm and 1440-1500nm.")
        sys.exit(1)

    if method == "brightness":
        return find_meat_brightness(cube)

    br = find_meat_brightness(cube)
    if wb is None:
        print("  WARNING: wavelength axis does not cover the ~1450nm water band,")
        print("           falling back to the brightness mask. That mask assumes")
        print("           meat fills ~40% of the frame -- if it does not, background")
        print("           leaks in (see docs/TEAM_LOG.md 2026-09-11). Check the mask.")
        return br

    leak = (br & ~wb).sum()
    print(f"  background-leak check: {leak} px ({100*leak/max(br.sum(),1):.1f}% of the"
          f" brightness mask) are NOT tissue by the water-band test.")
    if leak / max(br.sum(), 1) > 0.10:
        print("    ^ the original brightness mask would have been substantially")
        print("      contaminated on this cube. Using the water-band mask.")
    return wb


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
    ap.add_argument("--mask", choices=["auto", "waterband", "brightness"],
                     default="auto",
                     help="tissue mask method. 'auto' (default) uses the "
                          "~1450nm water band when the wavelength axis allows "
                          "and falls back to brightness otherwise. "
                          "'brightness' is the ORIGINAL top-40%% method -- use "
                          "it to reproduce the published sensor_sigma=0.0054 "
                          "from NHSI cube 01 exactly.")
    ap.add_argument("--tray-rows", type=int, nargs=2, metavar=("Y0", "Y1"),
                     default=None,
                     help="restrict the water-band mask to these rows "
                          "(NHSI trays: 85 480). Default: whole frame.")
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

    # Two masks, because the three measurements below want different things.
    # Verified on real NHSI cubes 01 and 19 (2026-09-13, docs/TEAM_LOG.md):
    #   - sensor read noise wants a UNIFORM, untextured region. Bare tray is
    #     the ideal target for that and is what the original brightness mask
    #     + flattest_patch() actually found (cube 01 patch = 0% tissue).
    #   - texture and the 970nm check want TISSUE and nothing else.
    # Using one mask for all three is what produced the confusion this
    # comment exists to prevent.
    tissue_mask = find_meat(cube, wl, a.mask, a.tray_rows)
    noise_mask = find_meat_brightness(cube) if a.mask != "brightness" else tissue_mask

    # Sensor noise: search the brightness mask, NOT the tissue mask. This
    # keeps the published sensor_sigma=0.0054 (NHSI cube 01) reproducible and
    # keeps muscle micro-texture out of a read-noise figure. Measured on cube
    # 01: uniform-region residual sd 0.00039 vs tissue 0.00377 -- a 10x gap
    # that is biology, not electronics.
    r0, c0 = flattest_patch(cube, noise_mask, a.patch)
    patch = cube[r0:r0+a.patch, c0:c0+a.patch, :]
    patch_is_tissue = 100.0 * tissue_mask[r0:r0+a.patch, c0:c0+a.patch].mean()
    print(f"  noise patch is {patch_is_tissue:.0f}% tissue by the water-band test")
    if patch_is_tissue < 50:
        print("    -> a uniform NON-tissue region (expected, and correct for")
        print("       read noise). Report it as such, NOT as 'measured on pork'.")
    else:
        print("    -> this patch is mostly TISSUE, so the figure below includes")
        print("       muscle micro-texture and is NOT pure read noise. Treat it")
        print("       as an upper bound and say so.")
    if not (a.tray_rows and a.tray_rows[0] <= r0 <= a.tray_rows[1]) and a.tray_rows:
        print(f"    WARNING: patch row {r0} is outside --tray-rows "
              f"{a.tray_rows[0]}-{a.tray_rows[1]}. Check what it landed on.")
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
    if patch_is_tissue < 50:
        print('     source: "Read noise measured from NHSI-meat-overtime cube')
        print('              01 (Wang et al. 2026): flattest 24x24 patch, which')
        print('              is a UNIFORM NON-TISSUE (tray) region -- the correct')
        print('              target for sensor read noise. NOT measured on pork,')
        print('              nor on any tissue; earlier wording saying so was')
        print('              inaccurate. Read noise is a property of the sensor,')
        print('              not of what was imaged, so the VALUE stands. Absolute')
        print(f'              residual sd {np.median(sigmas):.5f}, normalised by the')
        print(f'              patch mean {patch.mean():.4f}."')
    else:
        print('     source: "Measured from NHSI-meat-overtime cube (Wang et al.')
        print('              2026) on a TISSUE patch -- includes muscle')
        print('              micro-texture, so this is an upper bound on read')
        print('              noise, not pure read noise. Species NOT established:')
        print('              each cube is a mixed tray and the red-meat columns')
        print('              are unidentified."')
    print()

    # --- 2. TEXTURE AMPLITUDE (fine-scale) --------------------------------
    # texture_amplitude in generate_sample() is the FRACTIONAL multiplicative
    # std of FINE surface structure (muscle fibre / marbling, ~1 mm), applied
    # band-independently. The whole-region std ("macro" below) is the WRONG
    # quantity -- it is dominated by fat seams, muscle groups, surface
    # geometry and drip pooling, which the generator already models
    # separately (pH field + illumination). So we high-pass each band by a
    # normalized-convolution detrend (Gaussian, scale = sigma px), take the
    # residual std / mean, and remove the sensor-noise floor in quadrature.
    # Reported at several sigmas so the scale sensitivity is visible, and as
    # a median over bands (the generator's texture term is band-flat).
    from scipy.ndimage import gaussian_filter, binary_erosion

    band_mid = B // 2
    mid_meat = cube[:, :, band_mid][tissue_mask]
    noise_rel = sigmas[band_mid] / mid_meat.mean()
    total_rel = mid_meat.std() / mid_meat.mean()
    m = tissue_mask.astype(np.float64)
    band_step = max(B // 20, 1)

    print("=" * 62)
    print("2. TEXTURE AMPLITUDE  (fine-scale surface structure)")
    print("=" * 62)
    print(f"  whole-region std/mean (MACRO -- not what you want) : {total_rel:.4f}")
    print(f"  sensor-noise floor                                 : {noise_rel:.4f}")
    print()
    print("  high-pass residual std/mean, noise removed in quadrature:")
    results = {}
    for sig in (15.0, 25.0, 40.0):
        denom = gaussian_filter(m, sig)
        denom[denom < 1e-6] = 1e-6
        ev = binary_erosion(tissue_mask, iterations=int(sig))
        if ev.sum() < 1000:
            ev = tissue_mask
        fr = []
        for b in range(0, B, band_step):
            img = cube[:, :, b]
            resid = img - gaussian_filter(img * m, sig) / denom
            fr.append(resid[ev].std() / img[tissue_mask].mean())
        fine_denoised = float(np.sqrt(max(np.median(fr) ** 2 - noise_rel ** 2, 0.0)))
        results[sig] = fine_denoised
        print(f"    sigma = {sig:4.0f} px  ->  {fine_denoised:.4f}")
    print()
    print(f"  -> texture_amplitude value: {results[25.0]:.4f}   (sigma=25 px row;")
    print(f"     use the sigma that best matches your fibre/marbling scale)")
    print('     status: "MEASURED (fine-scale, high-pass)"')
    print('     source: "Fine-scale surface texture, NHSI-meat-overtime pork')
    print('              cube (Wang et al. 2026): median-over-bands high-pass')
    print('              residual std/mean, Gaussian detrend sigma=25 px,')
    print('              sensor-noise floor removed in quadrature."')
    print()

    # --- 3. 970 nm ANCHOR ---------------------------------------------------
    print("=" * 62)
    print("3. 970 nm REALITY CHECK")
    print("=" * 62)
    if a.lam_min <= 970 <= a.lam_max:
        i = int(np.argmin(np.abs(wl - 970)))
        v = cube[:, :, i][tissue_mask]
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
