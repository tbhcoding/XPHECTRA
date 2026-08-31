"""
LIGTAS-pH : Physics-based synthetic multispectral dataset generator
===================================================================

Combines the file schema and sparse-supervision design from the team's
technical spec with a biophysical forward model instead of a linear
SHAP-weighted mapping.

Forward model chain (this is the "equation chain" for Chapter 3):

    pH  ->  protein denaturation  ->  reduced scattering  mu_s'(lambda, pH)
    myoglobin fractions           ->  absorption          mu_a(lambda)
    (mu_a, mu_s')                 ->  Kubelka-Munk        R_inf(lambda)
    R_inf                         ->  sensor              pixel value

WHY THIS IS NOT CIRCULAR
------------------------
Reflectance is NOT a direct function of pH. pH acts only on scattering.
Absorption is driven independently by myoglobin state, which varies
sample to sample as a nuisance variable. The network must disentangle
the two using spectral shape across all six bands. Run self_test() to
confirm a linear pixel-wise fit does poorly -- that is the evidence
that the CNN is doing real work.

BEFORE YOU DEFEND THIS
----------------------
Every entry in PARAMS below marked status="PLACEHOLDER" must be replaced
with a cited value. The script prints a loud warning while any remain.
That warning is your week-1 to-do list.

Usage:
    python generate_dataset.py            # generate 400 samples
    python generate_dataset.py --selftest # run physics + invertibility checks
"""

import os
import json
import shutil
import tempfile
import argparse
import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

# ============================================================================
# PARAMETER TABLE  --  this IS your week-1 spreadsheet, in code form.
# Replace every PLACEHOLDER with a cited value and change status to "CITED".
# ============================================================================

WAVELENGTHS = np.array([481.0, 525.0, 573.0, 600.0, 730.0, 970.0])

PARAMS = {

    # ---- Table A: myoglobin molar extinction coefficients -------------------
    # Units: relative (any consistent scale; only ratios matter after
    # c_Mb is fitted). Source: derive from the modified Krzywicki equations
    # in Piao et al. (2025), Meat and Muscle Biology 9(1):18338, open access.
    # Krzywicki isosbestic points are 474/525/572/610 nm -- your 481/525/573/600
    # bands sit on or beside them, so the published coefficients transfer.
    "eps_deoxy": {
        "value": np.array([0.55, 0.62, 0.80, 0.30, 0.0, 0.0]),
        "status": "PLACEHOLDER",
        "source": "TODO: Piao et al. 2025, Table of extinction coefficients",
    },
    "eps_oxy": {
        "value": np.array([0.60, 0.62, 1.05, 0.18, 0.0, 0.0]),
        "status": "PLACEHOLDER",
        "source": "TODO: Piao et al. 2025",
    },
    "eps_met": {
        "value": np.array([0.85, 0.62, 0.45, 0.42, 0.0, 0.0]),
        "status": "PLACEHOLDER",
        "source": "TODO: Piao et al. 2025",
    },
    # NOTE: 525 nm is an isosbestic point -- all three forms should be EQUAL
    # there. That is a built-in sanity check on whatever numbers you enter.
    # 730 and 970 nm are outside the myoglobin bands; zero is defensible,
    # and 730 nm as a scattering baseline is the AMSA/Krzywicki convention.

    "c_Mb_mean": {
        "value": 2.2,
        "status": "PLACEHOLDER",
        "source": "TODO: pork Longissimus myoglobin concentration, mg/g",
    },
    "c_Mb_sd": {
        "value": 0.4,
        "status": "PLACEHOLDER",
        "source": "TODO: between-animal variation",
    },

    # ---- Table B: scattering ------------------------------------------------
    # mu_s'(lambda) = a * (lambda/500)^(-b)
    # Jacques (2013) Phys Med Biol 58:R37, Table 2, "other soft tissues".
    # CSV: omlc.org/news/dec14/Jacques_PMB2013/table2_JacquesPMB2013.csv
    # LIMITATION for Chapter 5: skeletal muscle is not broken out separately.
    "scatter_a": {
        "value": 18.9,
        "status": "CITED",
        "source": "Jacques 2013 PMB 58:R37 Table 2, other soft tissues (cm^-1)",
    },
    "scatter_b": {
        "value": 1.286,
        "status": "CITED",
        "source": "Jacques 2013 PMB 58:R37 Table 2, other soft tissues",
    },

    # ---- The pH -> scattering link : YOUR WEAKEST ASSUMPTION ----------------
    # Low pH near the isoelectric point (~5.4) denatures sarcoplasmic proteins,
    # raising mu_s'. That is why PSE meat is pale. Direction is well established;
    # the exact magnitude and shape are not.
    # DO NOT hide this. Sweep it (see sweep_denaturation.py idea in the docstring)
    # and report the sweep as a sensitivity analysis in Chapter 4.
    "denat_amplitude": {
        "value": 0.45,
        "status": "PLACEHOLDER",
        "source": "TODO: sweep 0.2-0.8; report sensitivity, do not pin one value",
    },
    "denat_midpoint": {
        "value": 5.70,
        "status": "PLACEHOLDER",
        "source": "TODO: pH at half-maximal denaturation effect",
    },
    "denat_width": {
        "value": 0.22,
        "status": "PLACEHOLDER",
        "source": "TODO: steepness of the transition",
    },

    # ---- Table C: water -----------------------------------------------------
    "mua_water": {
        "value": np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.45]),
        "status": "PLACEHOLDER",
        "source": "TODO: omlc.org tabulated water absorption (cm^-1) at 6 bands",
    },
    "water_fraction": {
        "value": 0.75,
        "status": "PLACEHOLDER",
        "source": "TODO: pork loin water content, ~0.75 by mass",
    },

    # ---- Sensor and tissue texture -----------------------------------------
    # BOTH of these can be MEASURED from the NHSI-meat-overtime cubes.
    # Run extract_sensor_params.py on a downloaded pork cube and paste the
    # numbers here. That converts two assumptions into cited measurements.
    "sensor_sigma": {
        "value": 0.015,
        "status": "ASSUMED",
        "source": "TODO: measure via extract_sensor_params.py (Wang et al. 2026)",
    },
    "texture_amplitude": {
        "value": 0.030,
        "status": "ASSUMED",
        "source": "TODO: measure via extract_sensor_params.py (Wang et al. 2026)",
    },
}


def check_params():
    """Print the citation status. This is your week-1 progress bar."""
    todo = [k for k, v in PARAMS.items() if v["status"] in ("PLACEHOLDER", "ASSUMED")]
    print("-" * 70)
    print("PARAMETER CITATION STATUS")
    print("-" * 70)
    for k, v in PARAMS.items():
        mark = {"CITED": "  OK  ", "MEASURED": " MEAS ", "ASSUMED": " TODO ", "PLACEHOLDER": " TODO "}[v["status"]]
        print(f"[{mark}] {k:20s} {v['source']}")
    if todo:
        print()
        print(f"!! {len(todo)} parameter(s) still uncited. Do not present results")
        print("!! from this dataset until each has a source. This is week 1.")
    print("-" * 70)
    print()
    return len(todo)


def P(key):
    return PARAMS[key]["value"]


# ============================================================================
# PHYSICS
# ============================================================================

def mu_s_prime(pH):
    """
    Reduced scattering coefficient, cm^-1. Shape (..., 6).

    Baseline wavelength dependence from Jacques (2013) power law, then
    scaled by a pH-driven denaturation factor.

    SIGN CHECK: low pH -> MORE denaturation -> MORE scattering -> PALER meat
    (higher reflectance). This is the PSE mechanism. The team's earlier
    generator had this inverted.
    """
    base = P("scatter_a") * (WAVELENGTHS / 500.0) ** (-P("scatter_b"))

    # Monotonic decreasing in pH: high at low pH, low at high pH.
    pH_arr = np.asarray(pH)[..., None]
    denat = 1.0 / (1.0 + np.exp((pH_arr - P("denat_midpoint")) / P("denat_width")))
    factor = 1.0 + P("denat_amplitude") * denat

    return base * factor


def mu_a(f_deoxy, f_oxy, f_met, c_Mb):
    """
    Absorption coefficient, cm^-1. Shape (6,).

    Note: NO pH term here. Absorption is set by myoglobin state, which is
    independent of pH in this model. That independence is what stops the
    forward model from being a one-to-one pH -> reflectance lookup.
    """
    mb = c_Mb * (f_deoxy * P("eps_deoxy")
                 + f_oxy * P("eps_oxy")
                 + f_met * P("eps_met"))
    water = P("water_fraction") * P("mua_water")
    return mb + water


def kubelka_munk(mu_a_v, mu_s_v):
    """
    Diffuse reflectance of a semi-infinite scattering slab.

        K = 2*mu_a,  S = (3/4)*mu_s'
        R_inf = 1 + K/S - sqrt((K/S)^2 + 2*(K/S))

    Cite a standard optics text for the K,S relations in Chapter 3.
    """
    K = 2.0 * mu_a_v
    S = 0.75 * mu_s_v
    x = K / np.maximum(S, 1e-9)
    return 1.0 + x - np.sqrt(x * x + 2.0 * x)


# ============================================================================
# SPATIAL FIELDS
# ============================================================================

def smooth_field(shape, scale, rng):
    """Smooth zero-mean random field via FFT-filtered Gaussian noise."""
    h, w = shape
    noise = rng.normal(0, 1, (h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    filt = np.exp(-(r * scale) ** 2)
    out = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
    return out / (out.std() + 1e-9)


def make_ph_field(shape, rng):
    """
    Base pH varies BETWEEN samples across the full physiological range.
    Spatial variation WITHIN one 5x5 cm sample is small (~0.1 pH units),
    which is what real loin actually does. The earlier generator stretched
    every sample across the whole range, which is not physical.
    """
    base = rng.uniform(5.35, 6.45)
    spread = rng.uniform(0.04, 0.14)
    field = smooth_field(shape, scale=18.0, rng=rng)
    return np.clip(base + spread * field, 5.2, 6.8)


def make_tissue_mask(shape, rng):
    """Irregular blob of meat on a dark background."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.normal(0, 6), w / 2 + rng.normal(0, 6)
    r = np.sqrt(((yy - cy) / (h * 0.42)) ** 2 + ((xx - cx) / (w * 0.42)) ** 2)
    wobble = 0.12 * smooth_field(shape, scale=28.0, rng=rng)
    return (r + wobble) < 1.0


# ============================================================================
# SAMPLE GENERATION
# ============================================================================

def generate_sample(sample_id, out_dir, rng, size=256):
    H = W = size

    ph_true = make_ph_field((H, W), rng)
    mask = make_tissue_mask((H, W), rng)

    # Myoglobin state: nuisance variables, independent of pH.
    # These are what make the inverse problem non-trivial.
    f_met = rng.uniform(0.05, 0.45)
    f_oxy = rng.uniform(0.20, 0.75) * (1.0 - f_met)
    f_deoxy = 1.0 - f_met - f_oxy
    c_Mb = max(0.3, rng.normal(P("c_Mb_mean"), P("c_Mb_sd")))

    absorb = mu_a(f_deoxy, f_oxy, f_met, c_Mb)          # (6,)
    scatter = mu_s_prime(ph_true)                        # (H, W, 6)
    R = kubelka_munk(absorb[None, None, :], scatter)     # (H, W, 6)

    # Muscle fibre / marbling texture: multiplicative, same across bands
    texture = 1.0 + P("texture_amplitude") * smooth_field((H, W), scale=6.0,
                                                          rng=rng)[..., None]

    # Illumination falloff from the LED ring
    yy, xx = np.mgrid[0:H, 0:W]
    illum = 1.0 - 0.10 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4
    illum = illum[..., None]

    cube = R * texture * illum
    cube = cube + rng.normal(0, P("sensor_sigma"), cube.shape)
    cube = np.clip(cube, 0.0, 1.0).astype(np.float32)

    # Background pixels: dark, not meat
    cube[~mask] = np.clip(rng.normal(0.03, 0.01, (int((~mask).sum()), 6)), 0, 1)
    ph_out = np.where(mask, ph_true, np.nan).astype(np.float32)

    # ---- sparse ground truth: 4 probe points, one per quadrant -------------
    pts, labels = {}, []
    for pid, (r0, r1, c0, c1) in {
        "P1": (0, H // 2, 0, W // 2),
        "P2": (0, H // 2, W // 2, W),
        "P3": (H // 2, H, 0, W // 2),
        "P4": (H // 2, H, W // 2, W),
    }.items():
        for _ in range(200):
            r = rng.integers(r0 + 20, r1 - 20)
            c = rng.integers(c0 + 20, c1 - 20)
            if mask[r, c]:
                break
        pts[pid] = (int(r), int(c))
        labels.append({
            "point_id": pid,
            "pixel_coord": [int(r), int(c)],
            "ph_value": float(round(ph_true[r, c], 3)),
        })

    meta = {
        "sample_id": sample_id,
        "roi_dimensions_cm": [5.0, 5.0],
        "resolution": [H, W],
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "sparse_measurements": labels,
        "generation_params": {
            "f_deoxy": round(f_deoxy, 4),
            "f_oxy": round(f_oxy, 4),
            "f_met": round(f_met, 4),
            "c_Mb": round(c_Mb, 4),
        },
    }

    np.save(os.path.join(out_dir, f"{sample_id}_msi.npy"), cube)
    np.save(os.path.join(out_dir, f"{sample_id}_phtrue.npy"), ph_out)
    with open(os.path.join(out_dir, f"{sample_id}_labels.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if HAVE_CV2:
        rgb = np.dstack([cube[:, :, 3], cube[:, :, 1], cube[:, :, 0]])  # 600/525/481
        rgb = (np.clip(rgb / max(rgb.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_rgb.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return cube, ph_true, mask


# ============================================================================
# SELF-TESTS  --  run these before you trust anything
# ============================================================================

def self_test():
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("TEST 1 -- sign of the pH / reflectance relationship")
    print("=" * 70)
    print("Expect reflectance to FALL as pH rises (low pH = PSE = pale).\n")

    absorb = mu_a(0.25, 0.60, 0.15, P("c_Mb_mean"))
    for ph in [5.4, 5.7, 6.0, 6.3]:
        R = kubelka_munk(absorb, mu_s_prime(np.array(ph)))
        print(f"  pH {ph}:  " + "  ".join(f"{int(w)}nm={r:.3f}"
                                          for w, r in zip(WAVELENGTHS, R)))
    lo = kubelka_munk(absorb, mu_s_prime(np.array(5.4)))
    hi = kubelka_munk(absorb, mu_s_prime(np.array(6.3)))
    ok = np.all(hi < lo)
    print(f"\n  Reflectance falls with pH in all bands: {ok}")
    print("  " + ("PASS" if ok else "FAIL -- check denat sign"))

    print()
    print("=" * 70)
    print("TEST 2 -- is the problem trivially invertible?")
    print("=" * 70)
    print("A linear fit on raw pixels should do POORLY. If R^2 is near 1,")
    print("the CNN is redundant and the result proves nothing.\n")

    tmp_dir = tempfile.mkdtemp(prefix="ligtas_selftest_")
    X, y = [], []
    try:
        for i in range(40):
            cube, ph, mask = generate_sample(f"tmp_{i}", tmp_dir, rng)
            idx = rng.choice(np.flatnonzero(mask), 400, replace=False)
            X.append(cube.reshape(-1, 6)[idx])
            y.append(ph.reshape(-1)[idx])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    X = np.vstack(X); y = np.concatenate(y)
    Xa = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  Linear baseline R^2  = {r2:.4f}")
    print(f"  Linear baseline MAE  = {np.abs(y - pred).mean():.4f} pH units")
    print()
    if r2 > 0.9:
        print("  WARNING: too easy. Widen the myoglobin nuisance ranges.")
    else:
        print("  GOOD: a linear model cannot solve this. Report this number")
        print("  in Chapter 4 as your baseline -- the CNN must beat it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ligtas_synthetic_dataset")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    check_params()
    if args.selftest:
        self_test()
        return

    rng = np.random.default_rng(42)
    splits = [("train", int(args.n * 0.75)),
              ("val", int(args.n * 0.125)),
              ("test", args.n - int(args.n * 0.75) - int(args.n * 0.125))]

    idx = 1
    for name, count in splits:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for _ in range(count):
            generate_sample(f"sample_{idx:03d}", d, rng)
            idx += 1
        print(f"  {name}: {count} samples -> {d}")
    print(f"\nDone. {args.n} samples in {args.out}/")


if __name__ == "__main__":
    main()
