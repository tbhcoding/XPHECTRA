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
Every entry in PARAMS below marked status "PLACEHOLDER", "ASSUMED" or
"PARTIAL" must be replaced with a cited or measured value. The script
prints a loud warning while any remain. That warning is your to-do list.

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
# PARAMETER TABLE  --  this IS your parameter spreadsheet, in code form.
# Replace every PLACEHOLDER / ASSUMED / PARTIAL entry with a cited or
# measured value and update its status. See the "XpHectra Parameterized
# Data Sheet" for the running record of sources.
# ============================================================================

WAVELENGTHS = np.array([481.0, 525.0, 573.0, 600.0, 730.0, 970.0])

PARAMS = {

    # ---- Table A: myoglobin molar extinction coefficients -------------------
    # Units: DECADIC millimolar extinction coefficients, mM^-1 cm^-1, taken
    # from the primary source (not digitized from a figure):
    #   Tang, Faustman & Hoagland (2004), J. Food Sci. 69(9):C717-C720,
    #   Table 2, p.C718.
    # Band order: [481, 525, 573, 600, 730, 970] nm.
    #   525 nm  -- read directly from Table 2 (isosbestic; all 3 forms = 7.60,
    #              itself sourced by Tang to Bowen 1949 / Krzywicki 1982 --
    #              ONE shared literature value, not three independent
    #              measurements that happened to agree).
    #   573 nm  -- Tang measured 557 and 582 nm but not 573. Value below is
    #              LINEARLY INTERPOLATED between those two rows (573 sits 64%
    #              of the way from 557 to 582). Replaces an earlier flat copy
    #              of the 582 nm value.
    #   481, 600 nm -- STILL OPEN, NOT a citation. Tang's 4 measured points
    #              (503/525/557/582) don't reach either band closely enough
    #              to call "nearest value." The numbers below are the OLD
    #              arbitrary-scale placeholders, rescaled by 7.60/0.62 (new
    #              cited 525 nm value over the old placeholder's 525 nm value)
    #              so they sit at the right ORDER OF MAGNITUDE next to the
    #              cited bands and don't silently corrupt self_test(). This is
    #              a provisional stand-in for pipeline testing only -- NOT
    #              defensible for the actual thesis dataset. Next source:
    #              Bowen (1949), J Biol Chem 179:235-245, full spectrum,
    #              rescaled onto Tang's scale at a shared wavelength.
    #   730, 970 nm -- myoglobin does not meaningfully absorb here; 0 is the
    #              standard AMSA/Krzywicki convention.
    "eps_deoxy": {
        "value": np.array([6.74, 7.60, 9.96, 3.68, 0.0, 0.0]),
        "status": "PARTIAL",
        "source": "Tang et al. 2004, Table 2, p.C718 (525 cited, 573 interpolated). "
                   "481, 600 nm are PROVISIONAL rescaled placeholders, not citations.",
    },
    "eps_oxy": {
        "value": np.array([7.35, 7.60, 12.61, 2.21, 0.0, 0.0]),
        "status": "PARTIAL",
        "source": "Tang et al. 2004, Table 2, p.C718 (525 cited, 573 interpolated). "
                   "481, 600 nm are PROVISIONAL rescaled placeholders, not citations.",
    },
    "eps_met": {
        "value": np.array([10.42, 7.60, 3.56, 5.15, 0.0, 0.0]),
        "status": "PARTIAL",
        "source": "Tang et al. 2004, Table 2, p.C718 (525 cited, 573 interpolated). "
                   "481, 600 nm are PROVISIONAL rescaled placeholders, not citations.",
    },
    # NOTE: 525 nm is an isosbestic point -- all three forms must be EQUAL
    # there. Built-in sanity check. CONFIRMED PASSING: 7.60 == 7.60 == 7.60
    # (was 7.97 for MetMb in an earlier figure-digitized version -- a ~5%
    # reading error).
    #
    # eps values above are DECADIC (Beer-Lambert, base-10). Kubelka-Munk's
    # K = 2*mu_a term expects a NAPIERIAN (natural-log) absorption
    # coefficient. See MB_DECADIC_TO_NAPIERIAN below and its use in mu_a().

    "c_Mb_mean": {
        "value": 0.87,
        "status": "CITED",
        "source": "Cross, King, Shackelford, Wheeler, Nonneman, Keel & Rohrer (2018). "
                   "Genome-wide association of myoglobin concentrations in pork loins. "
                   "Meat and Muscle Biology 2(1):189-196. doi:10.22175/mmb2017.08.0042. "
                   "n=599 pigs, Longissimus thoracis et lumborum, mean 0.87 mg/g tissue.",
    },
    "c_Mb_sd": {
        "value": 0.12,
        "status": "CITED / BACK-CALCULATED",
        "source": "Same source as c_Mb_mean. Paper reports 0.87 +/- 0.005 mg/g; "
                   "back-calculated SD = SE * sqrt(n) = 0.005 * sqrt(599) = 0.12 mg/g "
                   "(CV ~14%, biologically plausible). The paper's own '+/-' label "
                   "(SE vs SD) was not directly confirmable -- flag this inference in Ch.3.",
    },

    # ---- Table B: scattering ----------------------------------------------
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

    # ---- The pH -> scattering link : YOUR WEAKEST ASSUMPTION --------------
    # Low pH near the isoelectric point (~5.4) denatures sarcoplasmic
    # proteins, raising mu_s'. That is why PSE meat is pale. Direction is
    # well established; the exact magnitude and shape are not.
    # DO NOT hide this. Sweep denat_amplitude (0.2-0.8) and report the sweep
    # as a sensitivity analysis in Chapter 4.
    "denat_amplitude": {
        "value": 0.45,
        "status": "PLACEHOLDER",
        "source": "TODO: sweep 0.2-0.8; report sensitivity, do not pin one value",
    },
    "denat_midpoint": {
        "value": 5.70,
        "status": "CITED (PROXY)",
        "source": "Cross et al. (2018), Meat and Muscle Biology 2(1):189-196 -- same "
                   "599-pig population as c_Mb_mean. Reports ultimate pH 5.70 +/- 0.006. "
                   "Used as the denaturation-onset midpoint: this is a PROXY (population "
                   "mean ultimate pH, not a directly measured sigmoid midpoint). Flag as "
                   "a modeling assumption in Ch.3.",
    },
    "denat_width": {
        "value": 0.22,
        "status": "PLACEHOLDER",
        "source": "TODO: steepness of the transition; or derive from the sweep results",
    },

    # ---- Table C: water --------------------------------------------------
    "mua_water": {
        "value": np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.45]),
        "status": "PLACEHOLDER",
        "source": "TODO: Hale & Querry 1973 via omlc.org tabulated water absorption "
                   "(cm^-1) at all 6 bands. 970 nm confirmed 0.45; 730 nm ~0.018.",
    },
    "water_fraction": {
        "value": 0.75,
        "status": "PLACEHOLDER",
        "source": "TODO: pork loin water content ~0.75 by mass; confirm exact figure/page "
                   "in Honikel 1998 (manuscript ref [6]) before citing.",
    },

    # ---- Sensor and tissue texture --------------------------------------
    # BOTH can be MEASURED from the NHSI-meat-overtime cubes (Wang et al.
    # 2026). Run extract_sensor_params.py on a downloaded pork cube and
    # paste the numbers here. Watch the ">1.0 = raw sensor counts, not
    # calibrated reflectance" caveat from the README.
    "sensor_sigma": {
        "value": 0.0054,
        "status": "MEASURED",
        "source": "Measured from NHSI-meat-overtime pork cube 01.mat (Wang et al. 2026) via "
                   "extract_sensor_params.py: additive noise std relative to signal = 0.0054, "
                   "on a calibrated reflectance cube (values 0-1). Ch.5 caveats: different "
                   "camera (not the intended Arducam OV9281), NIR sensor, their illumination "
                   "and working distance -- transfers only approximately.",
    },
    # texture_amplitude (muscle-fibre/marbling mottle) was CUT -- was an
    # ASSUMED, uncited multiplicative nuisance term. extract_sensor_params.py
    # on NHSI 01.mat measured only ~0.005 relative residual on genuinely
    # smooth meat patches (indistinguishable from sensor noise); the term
    # added no mechanistic value over what sensor_sigma already provides, so
    # it was removed from generate_sample() rather than left as a floating
    # uncited knob. See docs/TEAM_LOG.md for the decision record.

    # ---- Non-myoglobin baseline absorption ------------------------------
    # Without this term, mu_a at 730/970 nm is ~0 (myoglobin eps=0 there,
    # water is tiny) and Kubelka-Munk returns ~90-92% reflectance in the
    # NIR -- far above the ~50-70% real pork loin shows (manuscript Fig. 6,
    # and the NHSI 970 nm reality-check in Ch.4). Stands in for everything
    # else that absorbs a little light in real tissue. Wavelength-independent
    # by construction (does not touch the pH sign test). TUNED, not cited --
    # state that plainly in Ch.3/Ch.5 and consider sweeping it (0.2/0.3/0.4)
    # alongside the denat_amplitude sweep.
    "mu_a_baseline": {
        "value": 0.3,
        "status": "TUNED -- NOT CITED, documented limitation",
        "source": "Chosen so 730 nm reflectance lands near realistic pork loin values "
                   "instead of ~0.90-0.92 with no baseline. Fitted nuisance term.",
    },
}

# Kubelka-Munk's K = 2*mu_a term expects a NAPIERIAN (natural-log)
# absorption coefficient. Tang's extinction coefficients are DECADIC
# (Beer-Lambert, base-10): mu_a(decadic) = eps * c. Standard conversion is
# mu_a(Napierian) = ln(10) * mu_a(decadic). Applied to the myoglobin term
# ONLY -- mua_water (Hale & Querry 1973 via omlc.org) is already tabulated
# as a physical/Napierian coefficient and does NOT get this factor.
MB_DECADIC_TO_NAPIERIAN = np.log(10.0)  # ~2.303

# c_Mb_mean/c_Mb_sd are in mg myoglobin per g tissue (Cross et al. 2018).
# eps is in mM^-1 cm^-1, so c_Mb must be converted from a mass fraction to
# a molar concentration in mM before it can multiply eps:
#   c_Mb [mg/g] * density [g/cm^3] / MW [g/mol]  ->  mmol/cm^3
#   * 1000 cm^3/L                                ->  mmol/L = mM
#   0.87 * 1.06 / 17000 * 1000 = 0.0542 mM
# The *1000 (CM3_PER_L) is the bridge people forget: eps is per mole-per-
# LITRE, the concentration above is built per cm^3.
MUSCLE_DENSITY_G_PER_CM3 = 1.06     # BioNumbers BNID 111214, mammalian skeletal muscle
MB_MOLAR_MASS_G_PER_MOL = 17000.0   # myoglobin MW ~17 kDa -- cite a species-specific
                                     # value (e.g. UniProt pig myoglobin) in Ch.3
CM3_PER_L = 1000.0                    # unit bridge, mmol/cm^3 -> mmol/L (mM)


def check_params():
    """Print the citation status. This is your progress bar."""
    blocking_statuses = ("PLACEHOLDER", "ASSUMED", "PARTIAL")
    todo = [k for k, v in PARAMS.items() if v["status"] in blocking_statuses]
    print("-" * 70)
    print("PARAMETER CITATION STATUS")
    print("-" * 70)
    mark_map = {"CITED": "  OK  ", "MEASURED": " MEAS ",
                "ASSUMED": " TODO ", "PLACEHOLDER": " TODO ", "PARTIAL": " PART "}
    for k, v in PARAMS.items():
        mark = mark_map.get(v["status"], " note ")
        print(f"[{mark}] {k:20s} {v['status']:32s} {v['source'][:52]}")
    if todo:
        print()
        print(f"!! {len(todo)} parameter(s) still uncited/unmeasured. Do not present")
        print("!! results from this dataset until each has a source.")
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
    Absorption coefficient, cm^-1 (Napierian). Shape (6,).

    No pH term here. Absorption is set by myoglobin state, which is
    independent of pH in this model. That independence is what stops the
    forward model from being a one-to-one pH -> reflectance lookup.

    Unit handling (see PARAMS / module constants for citations):
      1. c_Mb arrives in mg/g and is converted to mM (mmol/L) -- including
         the cm^3 -> L factor -- before multiplying the mM^-1cm^-1 eps.
      2. eps * c_Mb is a DECADIC absorption coefficient; * ln(10) converts
         it to the Napierian coefficient Kubelka-Munk expects.
      3. mua_water is already Napierian -- no conversion applied.
      4. mu_a_baseline is a tuned, uncited nuisance term -- see PARAMS.
    """
    c_Mb_mM = (c_Mb * MUSCLE_DENSITY_G_PER_CM3 / MB_MOLAR_MASS_G_PER_MOL) * CM3_PER_L
    mb_decadic = c_Mb_mM * (f_deoxy * P("eps_deoxy")
                             + f_oxy * P("eps_oxy")
                             + f_met * P("eps_met"))
    mb = mb_decadic * MB_DECADIC_TO_NAPIERIAN
    water = P("water_fraction") * P("mua_water")
    baseline = P("mu_a_baseline")
    return mb + water + baseline


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

    TEAM DECISION (see TEAM_LOG.md): scale was 18.0 (~0.35 cm correlation
    length), which verification found produced a mottled/speckled field
    rather than the smooth gradient described above. Raised to 77.0
    (~1.5 cm, at 256 px / 5 cm = 51.2 px/cm) for a visibly smoother,
    few-blob field closer to what a real 5x5 cm sample should look like.
    """
    base = rng.uniform(5.35, 6.45)
    spread = rng.uniform(0.04, 0.14)
    field = smooth_field(shape, scale=77.0, rng=rng)
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

    # Illumination falloff from the LED ring
    yy, xx = np.mgrid[0:H, 0:W]
    illum = 1.0 - 0.10 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4
    illum = illum[..., None]

    cube = R * illum
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
