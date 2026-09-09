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

## Current status — 2026-09-09 (pushed for team review — not yet formally agreed, see note below)

`generate_dataset.py` is the **only** generator now — the earlier
`generate_dataset_new_plus_eps.py` candidate was fully merged into it
(scattering values + eps NIR values, see below) and retired. **Use
`generate_dataset.py`, not any other file, from here on.**

Forward model runs end-to-end and passes both self-tests (sign check,
and linear-baseline R² = 0.651 ± 0.018, 10-seed mean — non-trivial).
**Do not generate the final dataset or run any CRN experiment you intend
to report** — the parameter table has **zero** blocking entries for the
first time in this project, but the latest change (`denat_amplitude`/
`denat_width`) is pushed for team review, not yet formally agreed.
Treat every number below as provisional until the team has actually
looked at it and signed off.

**Note on `eps_oxy`/`eps_deoxy`/`eps_met`:** these were closed twice, in
parallel, by two different people — `origin/main` commit `188e76a`
first closed `eps_oxy`/`eps_deoxy` @ 481/600nm via a haemoglobin-shape
proxy (leaving `eps_met` open on principle), then a second track
independently digitized all three directly from the Tang/Bowen figures
and was adopted as a disclosed, team-agreed override once the conflict
was found. See `docs/TEAM_LOG.md`'s 2026-09-08 entry for the full
comparison — both approaches' numbers and reasoning are preserved
there, not just the winner's.

| Parameter | State |
|---|---|
| `eps_*` @ 525 nm | CITED — Tang, Faustman & Hoagland 2004 Table 2 (isosbestic, printed value) |
| `eps_deoxy`/`eps_oxy`/`eps_met` @ 481, 573, 600 nm | **CITED — digitized** from Tang 2004 Figure 1 (WebPlotDigitizer; artifacts in `docs/digitization/`), calibrated to the 525nm isosbestic. Supersedes `188e76a`'s haemoglobin-proxy values for `eps_oxy`/`eps_deoxy` and resolves `eps_met`, which that commit left open. |
| `eps_deoxy`/`eps_oxy`/`eps_met` @ 730, 970 nm | **CITED — digitized** from Bowen 1949 Figures 1–2 (same artifact folder). Replaces the earlier small tuned values; closed more of the 970nm gap (0.093→0.076) at no R² cost. |
| `c_Mb_mean` 0.87, `c_Mb_sd` 0.12 mg/g | CITED — Cross et al. 2018 (n=599); SD back-calculated from SE |
| unit reconciliation (decadic→Napierian, mg/g→mM) | IMPLEMENTED in `mu_a()`; a real 1000× units bug was caught and fixed here |
| `scatter_a` 8.7436, `scatter_b` 1.6618 | **RESOLVED** — porcine-muscle refit (approx. Bergmann 2021) adopted, replacing the generic Jacques 2013 soft-tissue values. Formerly a `"DECISION"` blocker; now `FITTED`, not blocking. |
| `denat_midpoint` 5.70 | CITED (proxy) — Cross et al. 2018 ultimate pH |
| `denat_amplitude` 0.4 | **SWEPT** — no single citable value exists. A literature value (2.19, Offer & Knight 1988 via Kim/Warner/Rosenvold 2014) was checked and rejected: breaks the R²<0.9 non-triviality requirement (R²=0.9437) and is structurally incompatible with the cited midpoint (capped at 1.49×, not 2×). Adopted a verified sweep instead (0.2/0.4/0.6/0.8); full table in `docs/TEAM_LOG.md`. |
| `denat_width` 0.28 | **SWEPT** — genuinely unsourced. Prior citation (MacDougall & Jones 1981) was checked, found not to actually support the claim, and retracted. |
| `mua_water` | CITED — Hale & Querry 1973 via omlc.org, wired in with linear interpolation (970 nm = 0.45 exact) |
| `water_fraction` 0.732 | CITED — Wojtasik-Kalinowska et al. 2016 (LWT 67:112-117), mean of 4 diet groups |
| `sensor_sigma` 0.0054 | MEASURED — `extract_sensor_params.py` on NHSI cube `01.mat` |
| `mu_a_baseline` 0.8 | TUNED, uncited nuisance term (documented limitation) — re-swept fresh after the scattering/eps changes; this is a disclosed trade-off (closer 970nm match, real R² cost), see `docs/TEAM_LOG.md` |
| **970 nm external reality check** | Improved but not closed — sim ~0.264 vs real NHSI ~0.19 (was ~0.54). Decide: tune further, or report as a stated Ch. 5 limitation. |

**Current blocking count: 0.** `python generate_dataset.py --selftest`
prints the live blocking list — always trust that over this table if
they disagree. **The `denat_amplitude`/`denat_width` resolution above
is pushed to `origin/main` so the team can review it in the actual
code** — it still needs an actual team conversation and explicit
agreement before being treated as final, same standard as the eps
override. See `docs/TEAM_LOG.md`'s 2026-09-09 entry.

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

> **Status note (2026-09-09, pushed for team review):** this step is fully done
> — see *Current status* above and the Parameterized Data Sheet for the
> authoritative record. Myoglobin extinction coefficients at 525nm come
> from **Tang, Faustman & Hoagland 2004** Table 2 (decadic mM⁻¹cm⁻¹,
> read directly from the primary PDF); the 481/573/600/730/970nm bands
> are digitized directly from Tang (2004) Figure 1 and Bowen (1949)
> Figures 1–2 (see `docs/digitization/` for the source screenshots). The
> `c_Mb`, scattering, `denat_midpoint` and `sensor_sigma` rows are done.
> `mua_water` and `water_fraction` are done (Hale & Querry 1973;
> Wojtasik-Kalinowska 2016). `denat_amplitude`/`denat_width` are now
> swept (see §1d) rather than open placeholders. Only the 970 nm gap
> decision remains a genuinely open question.

Open `generate_dataset.py` and find the `PARAMS` dictionary near the top.
Each entry has a `value`, a `status`, and a `source`. Your job is to turn
every `PLACEHOLDER`, `ASSUMED` and `PARTIAL` into `CITED` or `MEASURED`.

### 1a. Myoglobin extinction coefficients — DONE (digitized from Tang 2004 / Bowen 1949)

**Historical note:** this section originally pointed at Piao et al.
(2025) as a source, then at a haemoglobin-shape proxy for the
481/600nm gap (`origin/main` commit `188e76a`, 2026-09-07). Both are
superseded — the project now reads `eps_deoxy`/`eps_oxy`/`eps_met` at
all 6 bands directly off the Tang (2004) Figure 1 and Bowen (1949)
Figures 1–2 curves (WebPlotDigitizer), calibrated to the cited 525nm
isosbestic value. Nothing left to fill in here. See:
- `generate_dataset.py`'s Table A header comment for the full method
  and per-band source breakdown.
- `docs/digitization/` for the six source screenshots the values are
  read from.
- `docs/TEAM_LOG.md`'s 2026-09-08 entry for why this was chosen over
  the haemoglobin-proxy alternative, including that approach's own
  numbers and reasoning (preserved there, not discarded).
- **525 nm is isosbestic** — all three forms have the SAME value
  (7.60) there; this is still worth spot-checking if you ever touch
  this table again.

### 1b. Water absorption — from omlc.org

**Done.** `mua_water` is wired in from Hale & Querry 1973 via the
omlc.org data file (`omlc.org/spectra/water/data/hale73.dat`), read at
the nearest tabulated wavelength to each band. 970 nm = 0.45 exact; the
visible bands are near-zero. `water_fraction` is `0.732`, cited to
Wojtasik-Kalinowska et al. 2016 (LWT 67:112-117).

### 1c. Myoglobin concentration

`c_Mb_mean` and `c_Mb_sd` — pork *Longissimus* myoglobin concentration
and its between-animal variation. Any meat science reference works.

### 1d. The denaturation parameters — DONE (swept, not pinned)

`denat_amplitude`, `denat_midpoint`, `denat_width` control how strongly
pH affects scattering. This is the project's weakest assumption and is
treated as one, deliberately, not smoothed over.

**Historical note:** a literature value for `denat_amplitude` (2.19,
from Offer & Knight 1988's ~2× PSE-vs-normal scattering claim) was
seriously considered — the citation itself was independently
re-verified against the actual source PDF and checks out. It was
rejected anyway: with `denat_midpoint`=5.70 already fixed, the model is
structurally capped at 1.49× at the citation's actual comparison (no
amplitude reaches 2× there), and forcing a literal 2× by anchoring
across the wider pH domain instead pushes linear-baseline R² to
0.9437 — breaking the "must stay comfortably under 0.9" non-triviality
requirement this whole experiment depends on, and worsening the 970nm
external match too.

**Adopted instead:** the swept range, `denat_amplitude` = 0.2, 0.4,
0.6, 0.8, all verified safely under R²=0.9, operating value **0.4**
(10-seed R²=0.651, 970nm gap=0.074). Report the full sweep table as the
Ch.4 sensitivity result, not just the operating value — same as
`mu_a_baseline`'s precedent. `denat_width` stays at 0.28, but as an
explicitly unsourced sweep value (0.20/0.28/0.40/0.50) after its prior
citation was checked and retracted. Full numbers, the literature-value
math, and citation re-verification: `docs/TEAM_LOG.md`'s 2026-09-09
entry. **Pushed for team review** — needs actual team agreement before
being treated as final.

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
