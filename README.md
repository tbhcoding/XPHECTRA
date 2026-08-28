# LIGTAS-pH — Synthetic Dataset Generator

Physics-based generator for 6-band multispectral pork images with
pixel-wise pH ground truth.

## Files

| File | What it does |
|---|---|
| `generate_dataset.py` | Builds the dataset. Contains the parameter table. |
| `extract_sensor_params.py` | Measures sensor noise + texture from the NHSI cubes. |
| `requirements.txt` | Dependencies. |

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

## Step 1 — Fill in the parameter table (this is week 1)

Open `generate_dataset.py` and find the `PARAMS` dictionary near the top.
Each entry has a `value`, a `status`, and a `source`. Your job is to turn
every `PLACEHOLDER` and `ASSUMED` into `CITED` or `MEASURED`.

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

Download the tabulated water absorption spectrum from omlc.org. Read off
the absorption coefficient at each of your six wavelengths. Only 970 nm
will be meaningfully non-zero.

Fill in `mua_water`. Set `water_fraction` to pork loin water content
(~0.75) with a citation from any meat science text.

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

### 1e. Scattering — already done

`scatter_a` and `scatter_b` come from Jacques (2013), *Phys. Med. Biol.*
58:R37, Table 2, "other soft tissues". Already filled in and cited.

For your limitations section: skeletal muscle isn't broken out separately
in that table, so you're using a soft-tissue average. Say so.

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
2. **Texture amplitude** → paste into `texture_amplitude`, status `"MEASURED"`
3. **970 nm reflectance stats** → your one external reality check

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

The notebook imports the script, so the physics lives in one place only.
Edit `PARAMS` in the `.py`, then **restart the notebook kernel** to pick
up the change.

```bash
pip install -r requirements.txt jupyter matplotlib
jupyter notebook LIGTAS_calibration.ipynb
```

In VS Code you can open the `.ipynb` directly — pick the venv as the
kernel when prompted.
