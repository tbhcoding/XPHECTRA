# LIGTAS-pH — Synthetic Dataset Generator

Physics-based generator for 6-band multispectral pork images with
pixel-wise pH ground truth.

## Why this exists (read first)

The hardware acquisition rig short-circuited before data collection, so the
real pork MSI dataset could not be built. With the adviser's approval, the
project pivoted to a **literature-constrained synthetic dataset**: same
objectives, same CRN experiment, "system" now scoped as the software
pipeline rather than a physical scanner. Every physical constant in the
forward model must trace to a published measurement — the running record of
that is the **Parameterized Data Sheet** (Google Sheet, team drive), which
`generate_dataset.py`'s `PARAMS` mirrors. Keep the two in sync.

## Current status — 2026-09-07

`generate_dataset.py` is the **only** generator now — the earlier
`generate_dataset_new_plus_eps.py` candidate was fully merged into it
(scattering values + eps NIR values, see below) and retired. **Use
`generate_dataset.py`, not any other file, from here on.**

Forward model runs end-to-end and passes both self-tests (sign check,
and linear-baseline R² = 0.690 ± 0.017, 10-seed mean — non-trivial).
**Do not generate the final dataset or run any CRN experiment you intend
to report** until the 5 blockers below are closed — every reported
number will move.

| Parameter | State |
|---|---|
| `eps_*` @ 525, 573 nm | CITED / interpolated — Tang, Faustman & Hoagland 2004 Table 2 |
| `eps_*` @ 481, 600 nm | **PROVISIONAL invented values** — need Bowen 1949 (or the haemoglobin-proxy alternative, see `CLAUDE.md`). These are the manuscript's diagnostic bands. #1 blocker. |
| `eps_*` @ 730, 970 nm | Small non-zero values (was the AMSA/Krzywicki 0 convention) — tuned, not cited; adopted after isolated testing showed a real 970nm-match improvement at no R² cost |
| `c_Mb_mean` 0.87, `c_Mb_sd` 0.12 mg/g | CITED — Cross et al. 2018 (n=599); SD back-calculated from SE |
| unit reconciliation (decadic→Napierian, mg/g→mM) | IMPLEMENTED in `mu_a()`; a real 1000× units bug was caught and fixed here |
| `scatter_a` 8.7436, `scatter_b` 1.6618 | **RESOLVED** — porcine-muscle refit (approx. Bergmann 2021) adopted, replacing the generic Jacques 2013 soft-tissue values. Formerly a `"DECISION"` blocker; now `FITTED`, not blocking. |
| `denat_midpoint` 5.70 | CITED (proxy) — Cross et al. 2018 ultimate pH |
| `denat_amplitude` | OPEN — must be **swept** (0.2/0.4/0.6/0.8) and reported as a Ch. 4 result, not pinned |
| `denat_width` 0.28 | OPEN — derived estimate (two independent published transitions), not a direct citation |
| `mua_water` | CITED — Hale & Querry 1973 via omlc.org, wired in with linear interpolation (970 nm = 0.45 exact) |
| `water_fraction` 0.732 | CITED — Wojtasik-Kalinowska et al. 2016 (LWT 67:112-117), mean of 4 diet groups |
| `sensor_sigma` 0.0054 | MEASURED — `extract_sensor_params.py` on NHSI cube `01.mat` |
| `mu_a_baseline` 0.8 | TUNED, uncited nuisance term (documented limitation) — re-swept fresh after the scattering/eps changes; this is a disclosed trade-off (closer 970nm match, real R² cost), see `docs/TEAM_LOG.md` |
| **970 nm external reality check** | Improved but not closed — sim ~0.30 vs real NHSI ~0.19 (was ~0.54). Decide: tune further, or report as a stated Ch. 5 limitation. |

**Current blocking count: 5** (eps @481/600nm ×3, `denat_amplitude`,
`denat_width`). `python generate_dataset.py --selftest` prints the live
blocking list — always trust that over this table if they disagree.

## Files

| File | What it does |
|---|---|
| `generate_dataset.py` | Builds the dataset. `PARAMS` mirrors the Parameterized Data Sheet. **The only generator — use this one.** |
| `extract_sensor_params.py` | Measures `sensor_sigma` + fine-scale texture + the 970 nm anchor from the NHSI cubes. |
| `crn_model.py` | The CRN (`CrudeCRN`) — a small U-Net predicting a dense pH map from the 6-band cube. |
| `train_crn.py` | Trains the CRN on the 4 sparse points + a smoothness prior. See Step 5. |
| `requirements.txt` | Dependencies (now includes `torch`, `matplotlib`). |
| `docs/TEAM_LOG.md` | Session-by-session log of what was done/verified/decided/still open. Read this before assuming any number here is final. |

---

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Check it works:

```bash
python generate_dataset.py --selftest
```

You should see `PASS` on test 1 and a linear baseline R² well under 0.9
on test 2. You'll also see a list of parameters still marked TODO —
that's step 1 below.

---

## Step 1 — Fill in the parameter table

> **Status note (2026-09-03):** this step is partly done — see *Current
> status* above and the Parameterized Data Sheet for the authoritative
> record. Myoglobin extinction coefficients now come from **Tang, Faustman
> & Hoagland 2004** Table 2 (decadic mM⁻¹cm⁻¹), read directly from the
> primary PDF, not digitized from Piao et al. 2025. The `c_Mb`, scattering,
> `denat_midpoint` and `sensor_sigma` rows are done. Still open: `eps` at
> 481/600 nm (Bowen 1949), `mua_water`, `water_fraction`, the
> `denat_amplitude` sweep, `denat_width`, and the 970 nm gap.

Open `generate_dataset.py` and find the `PARAMS` dictionary near the top.
Each entry has a `value`, a `status`, and a `source`. Your job is to turn
every `PLACEHOLDER`, `ASSUMED` and `PARTIAL` into `CITED` or `MEASURED`.

### 1a. Myoglobin extinction coefficients — from Piao et al. (2025)

Open **Piao, Ramanathan, Denzer, Pfeiffer & Mafi (2025)**, *Meat and
Muscle Biology* 9(1):18338. Open access, already reference [8] in your
manuscript.

Find the extinction coefficients for deoxymyoglobin, oxymyoglobin and
metmyoglobin. Fill in `eps_deoxy`, `eps_oxy`, `eps_met` — one value per
wavelength, in the order `[481, 525, 573, 600, 730, 970]`.

Notes:
- The paper uses Krzywicki's wavelengths (474, 525, 572, 610 nm). Yours
  are 481, 525, 573, 600. Use the nearest published value and say so in
  Chapter 3.
- **525 nm is isosbestic** — all three forms must have the SAME value
  there. If yours differ, you misread the table.
- 730 and 970 nm sit outside the myoglobin bands. Zero is defensible.

Set `status` to `"CITED"` and put the real citation in `source`.

### 1b. Water absorption — from omlc.org

**Done.** `mua_water` is wired in from Hale & Querry 1973 via the
omlc.org data file (`omlc.org/spectra/water/data/hale73.dat`), read at
the nearest tabulated wavelength to each band. 970 nm = 0.45 exact; the
visible bands are near-zero. `water_fraction` is `0.732`, cited to
Wojtasik-Kalinowska et al. 2016 (LWT 67:112-117).

### 1c. Myoglobin concentration

`c_Mb_mean` and `c_Mb_sd` — pork *Longissimus* myoglobin concentration
and its between-animal variation. Any meat science reference works.

### 1d. The denaturation parameters — DO NOT just pick a number

`denat_amplitude`, `denat_midpoint`, `denat_width` control how strongly
pH affects scattering. This is your weakest assumption and you should
treat it as one.

**Sweep it.** Generate datasets at `denat_amplitude` = 0.2, 0.4, 0.6, 0.8
and report how your results change. That sensitivity analysis IS a
Chapter 4 result, and it's more honest than pinning a value you can't
source.

### 1e. Scattering — RESOLVED

`scatter_a`/`scatter_b` were originally Jacques (2013), *Phys. Med.
Biol.* 58:R37, Table 2, "other soft tissues" — a generic soft-tissue
average, not porcine muscle. **Now resolved**: a porcine-muscle refit
(the same power-law formula, re-fit to approximate Bergmann et al. 2021)
has been adopted — `scatter_a=8.7436`, `scatter_b=1.6618`, status
`FITTED`. This closed ~47% of the 970nm real-world gap and dropped the
blocking count by 2, at a disclosed R² cost (0.530→0.647 at the time of
adoption). Full before/after numbers in `docs/TEAM_LOG.md`.

For your limitations section: the refit approximates a porcine-specific
study rather than being read directly from a table, and the underlying
Jacques data still doesn't break out skeletal muscle separately — both
worth stating plainly in Ch.3/Ch.5.

---

## Step 2 — Measure sensor parameters from the downloaded dataset

You have the NHSI-meat-overtime cubes from Wang, Tang, Li & Chen (2026).
They have no pH labels and their wavelengths don't overlap yours, so they
can't train your model. But two things in your parameter table are
sensor characteristics you can measure from them directly.

```bash
python extract_sensor_params.py path/to/pork_cube.mat
```

The script prints three blocks:

1. **Sensor noise** → paste into `sensor_sigma`, status `"MEASURED"`
2. **970 nm reflectance stats** → your one external reality check

(`extract_sensor_params.py` also reports a fine-scale texture residual,
but `texture_amplitude` was **cut from the model** on 2026-09-05 — it was
an uncited nuisance term indistinguishable from sensor noise on smooth
patches. See `docs/TEAM_LOG.md`.)

### About block 3

970 nm is the only wavelength where their camera and your bands overlap.
After you generate your dataset, compare:

```python
import numpy as np
cube = np.load("ligtas_synthetic_dataset/train/sample_001_msi.npy")
ph   = np.load("ligtas_synthetic_dataset/train/sample_001_phtrue.npy")
mask = np.isfinite(ph)
print("simulated 970nm mean:", cube[:, :, 5][mask].mean())
```

Same ballpark as their measured value → your 970 nm channel is
physically plausible. Way off → revisit water absorption or the
Kubelka-Munk constants.

One band out of six, on real pork, measured by someone else. It is the
only external validation available to you. Put it in Chapter 4 as a
figure.

### First, check what the values mean

If the script warns that values exceed 1.0, the cubes are raw sensor
counts rather than calibrated reflectance. The noise figure is still
usable as a relative number, but the 970 nm comparison isn't — you'd
need their white and dark reference frames first. Check their README.

### Caveats for Chapter 5

- Different camera than your intended Arducam OV9281
- Their NIR sensor ≠ a visible-range sensor; noise at 970 nm is not
  necessarily noise at 481 nm
- Their samples, illumination and working distance

Measured numbers with stated caveats still beat invented ones.

---

## Step 3 — Generate

> **Do not run this for the reported dataset yet (2026-09-03).** The
> parameter table still has provisional/uncited entries (see *Current
> status*). The 400-sample dataset currently on disk predates all of the
> September parameter work and must not be trained on or quoted. Regenerate
> only after the blockers are closed, then re-run Step 4.

```bash
python generate_dataset.py --n 400 --out ligtas_synthetic_dataset
```

Produces a 75/12.5/12.5 train/val/test split. Per sample:

```
sample_001_msi.npy      (256, 256, 6) float32  — model input
sample_001_phtrue.npy   (256, 256)    float32  — hidden dense ground truth
sample_001_labels.json                         — 4 sparse probe points
sample_001_rgb.png      (256, 256, 3) uint8    — visualisation only
```

Train on the **4 sparse points only**. Keep `phtrue.npy` hidden during
training and use it at evaluation time to measure full-field
reconstruction. That separation is the core experiment.

---

## Step 4 — Verify before you trust it

```bash
python generate_dataset.py --selftest
```

**Test 1** checks the sign: reflectance must FALL as pH rises. Low pH
means PSE means pale means bright. If this fails, the physics is
inverted.

**Test 2** checks the task isn't trivial. A linear fit on raw pixels
should get R² well below 0.9. If it climbs near 1.0, your generator has
collapsed into a linear mapping and a CNN result would prove nothing.

Run this after **every** parameter change. Report the linear baseline in
Chapter 4 — it's the number your CNN has to beat.

---

## Step 5 — Train the CRN (Priority 2)

Once you have a generated dataset (Step 3), train the dense pH predictor
on the **4 sparse points only** — `phtrue.npy` stays hidden from training
and is used only as a sanity check, same rule as everywhere else in this
project.

```bash
python train_crn.py --data ligtas_synthetic_dataset --epochs 50
```

Loss = sparse-point MSE + a total-variation smoothness term (required:
4 points alone can't determine a 65,536-pixel map). `--out-res` (coarser
prediction grid) and `--n-points` (currently capped at 4 — see the
script's docstring) are configurable, not hardcoded.

**Learning rate note:** the default is `1e-4`. An earlier default of
`1e-3` caused wild epoch-to-epoch validation swings (val MAE bouncing
between ~0.06 and 1.17 pH). Diagnosed and confirmed at a 50-epoch run —
see `docs/TEAM_LOG.md` for the full investigation. `1e-4` is
**significantly more stable, not perfectly stable** — a mild late-run
uptick and grainy per-pixel texture at full resolution are still open.
Don't treat any single "best epoch" number as final without checking the
loss curve.

Output per run (`--out`, default `crn_outputs/`):

```
crn_best.pt              best checkpoint by val MAE (not committed to git)
loss_curve.png           train/val sparse loss + MAE per epoch
sanity_check_heatmap.png true pH | prediction | |error|, for one val sample
history.json             full per-epoch train/val MAE and R²
```

---

## Why this isn't circular

pH does not map directly to reflectance. pH acts only on **scattering**.
Absorption is driven independently by myoglobin state, which varies
sample to sample as a nuisance variable. The same brightness can come
from different pH values depending on oxidation state, so the network
must use spectral *shape* across all six bands to separate them.

That's why 525 nm and 730 nm matter: 525 is isosbestic (anchors
myoglobin concentration) and 730 is nearly absorption-free (reads
scattering almost directly).

Test 2 is your evidence. Linear R² ≈ 0.45 means the problem is real.

---

## What to say at the defense

**What this is:** a literature-constrained synthetic benchmark for
evaluating the CRN architecture and the software pipeline under sparse
supervision.

**What this is not:** validation on real pork.

**Why it's still valid:** every physical constant traces to a published
measurement. The forward model was fixed and shown to the adviser before
any training. The sensitivity sweep reports how conclusions change under
the weakest assumption.

Say the second line out loud before the panel asks.

---

## Which file do I open?

| File | When |
|---|---|
| `LIGTAS_calibration.ipynb` | **Start here.** Filling in parameters, checking physics, making figures. |
| `generate_dataset.py` | Edit the `PARAMS` table here. Run from terminal to build the 400 samples. |
| `extract_sensor_params.py` | Once, on a downloaded NHSI cube, to measure sensor noise. |
| `train_crn.py` | Once a dataset exists. Trains the CRN — see Step 5. |
| `docs/TEAM_LOG.md` | Before touching the CRN or dataset params — check what's actually settled vs. still open. |

The notebook imports the script, so the physics lives in one place only.
Edit `PARAMS` in the `.py`, then **restart the notebook kernel** to pick
up the change.

```bash
pip install -r requirements.txt jupyter matplotlib
jupyter notebook LIGTAS_calibration.ipynb
```

In VS Code you can open the `.ipynb` directly — pick the venv as the
kernel when prompted.
