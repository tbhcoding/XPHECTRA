# LIGTAS-pH — System Architecture Reference

Complete technical reference for the current state of the codebase: what
exists, how it works, what's cited vs. assumed, and how to run it. Written
to answer "what did you use for X" panel questions directly, and to help
divide remaining work among the team.

---

## 1. What this system is, in one paragraph

LIGTAS-pH predicts pork pH from a 6-band multispectral image (481, 525,
573, 600, 730, 970 nm). The intended pipeline was: real camera → real MSI
cubes → CRN model → dense pH map. The camera failed before data
collection, so the project pivoted (adviser-approved) to a **literature-
constrained physics simulation** that generates synthetic MSI cubes with
known pH ground truth, used to test whether the CRN architecture and a
sparse-supervision training approach can reconstruct a dense pH field from
just 4 measured points per sample. The claim being tested is narrower than
"this measures real pork pH" — it's "this pipeline and architecture work
under a stated, physically-motivated set of assumptions."

---

## 2. Repository structure

| File | Role |
|---|---|
| `generate_dataset.py` | **The only dataset generator.** Contains the `PARAMS` table (see §4) and the forward-model equations. A parallel candidate file, `generate_dataset_new_plus_eps.py`, existed 2026-09-04 through 09-07 to test scattering/eps changes in isolation before adopting them into this file; it's now fully merged and retired. |
| `crn_model.py` | The neural network (`CrudeCRN`) — see §5. |
| `train_crn.py` | Training loop: data loading, loss functions, early stopping, evaluation, checkpointing — see §6. |
| `extract_sensor_params.py` | One-off script that measures sensor noise + a spatial-texture check from a real (different-purpose) hyperspectral dataset (Wang et al. 2026, NHSI-meat-overtime) — used to source `sensor_sigma` and to reality-check the 970nm band. |
| `make_figures.py` | Produces every manuscript figure from the frozen dataset and the saved result files; records the selection rule for each in `figures/figure_manifest.json`. A notebook (`LIGTAS_calibration.ipynb`) used to do this interactively; it was removed in `42e3deb` because it drifted out of sync with `PARAMS`. Do not recreate it. |
| `README.md` | Setup + step-by-step walkthrough. |
| `docs/TEAM_LOG.md` | Dated, structured log of every session's changes/verified numbers/decisions/open items. **The authoritative history — read this, not memory, before claiming any past result.** |
| `requirements.txt` | `numpy`, `scipy`, `h5py`, `opencv-python`, `matplotlib`, `torch`. |

Generated artifacts (never committed, always regenerable — see §7):
dataset folders (`ligtas_synthetic_dataset*/`), CRN output folders
(`crn_outputs*/`, `crn_5seed*/` — some of these *are* committed as
reference results, checkpoints `*.pt` never are).

---

## 3. The forward model — physics chain

```
pH  ──────────────────────────►  protein denaturation  ──►  scattering  μs'(λ, pH)
myoglobin state (deoxy/oxy/met, independent of pH)  ──►  absorption  μa(λ)
(μa, μs')  ──►  Kubelka-Munk radiative transfer  ──►  diffuse reflectance R∞(λ)
R∞  ──►  illumination falloff + sensor noise  ──►  final pixel value
```

**Why this isn't circular** (a likely panel question): reflectance is not
a direct function of pH. pH only drives scattering. Absorption is driven
independently by myoglobin oxidation state, which is randomized per
sample as a nuisance variable uncorrelated with pH. Two samples at the
same pH can look different depending on myoglobin state, and vice versa
— the network has to use the *spectral shape* across all 6 bands to
separate the two effects, not just overall brightness. This is verified
empirically: a plain linear regression on raw pixel values only reaches
R²≈0.651 ± 0.018 on the current `generate_dataset.py` (10-seed mean,
post scatter_a/b + fully digitized eps + retuned `mu_a_baseline` +
swept `denat_amplitude`=0.4, now **reviewed, agreed, and frozen**),
well under 1.0, proving the mapping isn't trivial (see §6.5 and
`self_test()`). Note this ~0.65 is the *linear non-triviality baseline*,
deliberately kept under 0.9 — it is NOT the headline result. The headline
is the CRN's 0.8472; see §8.

### 3.1 Scattering — `mu_s_prime(pH)`

```
μs'(λ) = scatter_a · (λ/500)^(-scatter_b) · [1 + denat_amplitude · sigmoid((pH - denat_midpoint)/denat_width)]
```

A Jacques (2013)-style power law for the wavelength dependence, multiplied
by a logistic (sigmoid) factor that increases scattering as pH drops
toward the isoelectric point — the PSE (pale, soft, exudative) mechanism:
low pH → protein denaturation → more scattering → paler meat → higher
reflectance.

### 3.2 Absorption — `mu_a(f_deoxy, f_oxy, f_met, c_Mb)`

```
c_Mb [mg/g] → mM:  c_Mb_mM = c_Mb · 1.06 [g/cm³] / 17000 [g/mol] · 1000 [cm³/L]
mb_decadic  = c_Mb_mM · (f_deoxy·ε_deoxy + f_oxy·ε_oxy + f_met·ε_met)
mb          = mb_decadic · ln(10)                      [decadic → Napierian]
μa(λ)       = mb + water_fraction · mua_water(λ) + mu_a_baseline
```

No pH term — absorption is set entirely by myoglobin state, which is
randomized independently per sample (`f_deoxy`, `f_oxy`, `f_met` drawn
per-sample; `f_met ∈ [0.05, 0.45]`, `f_oxy ∈ [0.20, 0.75]·(1−f_met)`,
`f_deoxy` = remainder). This independence is what stops the whole thing
from being a one-to-one pH → reflectance lookup.

### 3.3 Combining into reflectance — Kubelka-Munk

```
K = 2·μa,  S = 0.75·μs'
R∞ = 1 + K/S − √((K/S)² + 2·K/S)
```

Standard diffuse-reflectance solution for a semi-infinite scattering slab
(cite a radiative-transfer/optics text, e.g. Kubelka & Munk 1931, in
Chapter 3).

### 3.4 Sensor model

```
cube = R∞ · illumination_falloff
cube = cube + Normal(0, sensor_sigma)
cube = clip(cube, 0, 1)
```

Illumination falloff is a radial vignette from the LED ring (deterministic
geometric term, not fit to data). Sensor noise is additive Gaussian, std
measured from a real hyperspectral cube (`sensor_sigma`, see §4).

### 3.5 Spatial fields

- **pH field**: base pH drawn per-sample from U(5.35, 6.45); a smooth
  spatial variation (~±0.04–0.14) added via FFT-filtered noise (spatial
  correlation length ~1.5cm, tuned after verification found the original
  scale looked speckled rather than a smooth gradient — see TEAM_LOG,
  2026-09-04).
- **Tissue mask**: an irregular blob (randomized ellipse + smooth wobble)
  simulating a real trimmed chop's silhouette on a dark background.
- **4 sparse probe points**: one randomly placed point per image quadrant
  — the *only* labels the CRN is allowed to train on (see §6).

---

## 4. Parameter table — current citation status

Run `python generate_dataset.py --selftest` (prints this live, plus
physics sanity checks) — the table below is a snapshot, always re-verify
against the live output before quoting it.

| Parameter | Value | Status | What it controls |
|---|---|---|---|
| `eps_deoxy/oxy/met` @ 525nm | Tang et al. (2004) Table 2, printed | **CITED** | Isosbestic point, all 3 forms = 7.60 |
| `eps_deoxy/oxy/met` @ 481, 573, 600nm | digitized from Tang (2004) Figure 1 | **CITED — digitized** | Calibrated to the 525nm isosbestic. Supersedes `origin/main` commit `188e76a`'s haemoglobin-shape proxy for `eps_oxy`/`eps_deoxy` (adopted as a disclosed, team-agreed override — see `docs/TEAM_LOG.md` 2026-09-08) and resolves `eps_met`, which that commit left open. |
| `eps_deoxy/oxy/met` @ 730, 970nm | digitized from Bowen (1949) Figures 1–2 | **CITED — digitized** | Replaces earlier small tuned values; closed more of the 970nm gap (0.093→0.076) at no R² cost. Source screenshots in `docs/digitization/`. |
| `c_Mb_mean`, `c_Mb_sd` | 0.87, 0.12 mg/g | CITED / back-calculated, justified | Myoglobin concentration (Cross et al. 2018, n=599 pigs) |
| `scatter_a`, `scatter_b` | 8.7436, 1.6618 | **FITTED — resolved** | Scattering power law — refit to approximate a porcine-specific study (Bergmann et al. 2021) using the original 2-parameter formula. Formerly a `"DECISION"` blocker; adopted. |
| `denat_amplitude` | 0.4 | **SWEPT — resolved** | How strongly pH affects scattering. A literature value (2.19, Offer & Knight 1988) was checked and rejected — structurally capped at 1.49x vs. the cited 2x, and breaks R²<0.9 (R²=0.9437) if forced. Adopted a verified sweep (0.2/0.4/0.6/0.8) instead; pushed for team review. |
| `denat_midpoint` | 5.70 | CITED (proxy) | pH at half-maximal denaturation — proxy from population mean ultimate pH; source cohort was non-PSE (caveat documented) |
| `denat_width` | 0.28 | **SWEPT — resolved** | Steepness of the pH transition. Prior citation (MacDougall & Jones 1981) checked and retracted — did not actually support the claim made for it. Now an explicitly unsourced sweep value (0.20/0.28/0.40/0.50). |
| `mua_water` | Hale & Querry 1973 values | **CITED** | Water absorption spectrum, wired in with linear interpolation between tabulated grid points |
| `water_fraction` | 0.732 | CITED | Pork loin water content (Wojtasik-Kalinowska et al. 2016, n=24 pigs) |
| `sensor_sigma` | 0.0054 | MEASURED | Sensor noise, measured directly from a real hyperspectral cube |
| `mu_a_baseline` | 0.8 | TUNED, not cited (documented limitation) | Non-myoglobin absorption baseline — re-swept fresh after scattering/eps changes; a disclosed trade-off (closer 970nm match, real R² cost), not a free win |
| ~~`texture_amplitude`~~ | — | **CUT** | Removed — verified (10-seed test) that it added no mechanistic value beyond what `sensor_sigma` already provides |

**Current blocking count: 0.** All three eps arrays dropped off
2026-09-08 (fully digitized, superseding `188e76a`'s haemoglobin
proxy), and `denat_amplitude`/`denat_width` dropped off 2026-09-09
(swept and verified). This is the first time the parameter table has
been fully clean. The 2026-09-09 change was pushed to `origin/main`,
**reviewed, agreed, and is now frozen** — the full experiment has been
run on these values. The two SWEPT entries stay labelled as swept rather
than cited; the amplitude sweep is what establishes the conclusion
survives their uncertainty. Verified live via `check_params()`.

---

## 5. The model — `CrudeCRN`

A small U-Net, deliberately minimal (crude-first-pass philosophy — get
something training end-to-end before optimizing architecture).

```
Input:  (B, 6, 256, 256)  — the 6-band multispectral cube
Output: (B, out_res, out_res)  — dense pH prediction, out_res ≤ 256

enc1: Conv3x3(6→16) → BN → ReLU → Conv3x3(16→16) → BN → ReLU        [256×256]
      ↓ MaxPool2
enc2: Conv3x3(16→32) → BN → ReLU → Conv3x3(32→32) → BN → ReLU       [128×128]
      ↓ MaxPool2
enc3: Conv3x3(32→64) → BN → ReLU → Conv3x3(64→64) → BN → ReLU       [64×64]

up2:  ConvTranspose2d(64→32, stride 2) → concat(skip from enc2) → conv_block(64→32)  [128×128]
up1:  ConvTranspose2d(32→16, stride 2) → concat(skip from enc1) → conv_block(32→16)  [256×256]

head: Conv1x1(16→1)
      — weight zero-initialized, bias initialized to 5.9 (physiological
        pH midpoint), so training starts in-range instead of wasting
        early epochs correcting a large constant offset
      — if out_res < 256: adaptive average pool down to out_res
        (a coarser cell = the area-averaged pH over that cell)
```

**Parameters:** 2 downsample / 2 upsample stages, base channel width 16,
standard U-Net skip connections. `BatchNorm2d` throughout (kept after
diagnosis found learning rate, not normalization choice, was the actual
cause of early instability — see §6.4).

---

## 6. Training — `train_crn.py`

### 6.1 The core design decision: sparse supervision

The CRN is trained using **only the 4 sparse probe points per sample** —
never the dense hidden pH map. This mirrors the real deployment scenario
(a few physical pH-meter probe readings, dense prediction everywhere
else) and is the actual research question: can the network learn a
useful *dense* field from *sparse* labels.

### 6.2 Loss function

```
loss = sparse_point_MSE + tv_weight · total_variation(prediction)
```

- **`sparse_point_MSE`**: MSE between the predicted value at each of the
  4 probe pixel locations and their true labels. This alone is required,
  but underdetermined — 4 labels can't meaningfully constrain up to
  65,536 pixels.
- **`total_variation`**: mean absolute difference between horizontally/
  vertically adjacent pixels in the *full* predicted map. This is what
  makes the problem well-posed — it's the smoothness prior that fills in
  the space between the 4 known points. `tv_weight = 0.05` (default).

### 6.3 What's evaluation-only, never touching gradients

`val_MAE` / `val_R²` are computed against the **full hidden dense pH
field** (confirmed directly in code — `full_field_eval()` uses
`ph_dense`+`mask`, all meat pixels, not the 4 points). This is a **sanity
check only** — the training loop never lets it influence gradients. This
separation is enforced throughout the codebase and is a repeatedly-tested
invariant, not an incidental detail.

### 6.4 Hyperparameters (current defaults)

| | Default | Note |
|---|---|---|
| Learning rate | `1e-4` | Was `1e-3` originally — caused wild val instability (MAE swinging 0.06–1.17 pH). Diagnosed against weight decay (ruled out overfitting) and architecture (BatchNorm vs GroupNorm — ruled out, LR was the dominant cause). Lowering to 1e-4 roughly halved the instability. |
| Batch size | 8 | |
| Optimizer | Adam | `weight_decay=0.0` by default (no L2 regularization beyond the TV term) |
| `tv_weight` | 0.05 | |
| `out_res` | 256 (= full resolution) | Configurable to a coarser grid |
| `n_points` | 4 | Configurable, but dataset currently only embeds 4 probes/sample — requesting more warns and clamps |

### 6.5 Early stopping (added most recently)

`--patience N` enables early stopping: checkpoints on the **lowest
validation loss** (`val_sparse_MSE + tv_weight·val_TV`, computed from the
4 sparse val points — deliberately *not* the hidden-field val_MAE, to
keep that metric sanity-check-only even for model selection). Stops after
`N` epochs with no improvement, hard-capped at `--epochs`. `--seed N`
sets `torch.manual_seed()` for reproducible weight init + data shuffling.
Both default to off/unset — old behavior (run the full `--epochs`,
checkpoint on val_MAE) is unchanged unless these are passed.

### 6.6 Training instability — characterised and disclosed, not open

Even after the LR fix, validation performance swings epoch-to-epoch more
than desired, though **less than earlier entries in this file claimed** —
see the correction below.

**SUPERSEDED NUMBERS (2026-09-13).** This section previously reported
**val R² = 0.79 ± 0.07–0.08** against a linear baseline of ~0.68. Both
were wrong, for a reason this section itself had already flagged as an
untried fix: `run_epoch()` computed R² per mini-batch and averaged the
batches, which is not the epoch's R², while the baselines were scored
pooled. That was found independently on a second machine and **fixed at
source** in `train_crn.py`. Corrected figures, same checkpoints, no
retraining:

| | superseded | **current** |
|---|---|---|
| CRN val R² (5 seeds) | 0.79 ± 0.07 | **0.8486 ± 0.0368** |
| CRN **test** R² (5 seeds, untouched split) | — | **0.8389 ± 0.0412** |
| Linear baseline | ~0.68 | **0.6212** held-out / 0.6509 self-test |

The same error **exaggerated the instability described here**: rescored,
seed 0's epoch-1 val R² moves from −0.34 to +0.003 with MAE unchanged, so
the "crashes to R² ≈ −6" quoted in older entries are partly a scoring
artefact. The instability is real (val_loss genuinely spikes) but milder
than logged. Early stopping on val_loss continues to select a good
checkpoint. GroupNorm remains untried and is now lower priority.

---

## 7. How to run everything

```bash
# Setup
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Sanity-check the physics (run after any parameter change)
python generate_dataset.py --selftest

# Generate the dataset (400 samples, 300/50/50 train/val/test split)
python generate_dataset.py --n 400 --out ligtas_synthetic_dataset

# Train the CRN
python train_crn.py --data ligtas_synthetic_dataset --epochs 50

# Train with early stopping + a reproducible seed (recommended going forward)
python train_crn.py --data ligtas_synthetic_dataset --epochs 50 --patience 10 --seed 0
```

Dataset generation is **fully deterministic** (hardcoded seed in
`main()`) — the same script version always produces a byte-identical
dataset, verified directly by regenerating and comparing.

Each `train_crn.py` run produces, in its `--out` folder: `crn_best.pt`
(checkpoint, never committed), `loss_curve.png`, `sanity_check_heatmap.png`
(true pH | prediction | error, for one validation sample), `history.json`
(full per-epoch metrics).

---

## 8. Likely panel questions, answered directly

**"Is this trained on real data?"** No — real camera acquisition failed
before data collection (adviser-approved pivot). Every physical constant
in the simulation traces to either a published measurement, a direct
measurement from a real (different-purpose) dataset, or is explicitly
flagged as an assumption. See §4.

**"How do you know the problem isn't trivially easy for the model?"** A
plain linear regression on raw pixel values is run as a baseline
(`self_test()`, Test 2) — it reaches **R² = 0.6509 ± 0.0180**, averaged
over 10 random seeds (not a single lucky draw — this was itself a bug
found and fixed: single-seed numbers were found to sit 0.05–0.08 above
the true average). On the frozen dataset's own held-out split a linear
fit reaches **0.6212**, and PLSR **0.6206**. Well under 0.9, meaning the
mapping genuinely requires non-trivial spatial/spectral reasoning.
(Earlier figures of 0.587/0.691 in this file predate the retirement of
`generate_dataset_new_plus_eps.py` and no longer apply.)

**"Why only 4 points?"** Mirrors a realistic deployment: a few physical
pH-probe measurements per sample, dense prediction everywhere else. This
is deliberately configurable (`--n-points`), though the current dataset
only embeds 4 — ablating to more requires regenerating the dataset with
more embedded probes (explicit, logged follow-up, not done yet).

**"Is the model actually working?"** Yes, with two stated caveats. The
reported figure is **R² = 0.8472 ± 0.0384** with **MAE 0.0936 pH** and
**RMSE 0.1282 pH**, measured on 500 fresh held-out samples (seed 777),
five seeds. It clearly beats both baselines: 0.6212 (linear) and 0.6206
(PLSR). Validation (n=50) gives 0.8486 ± 0.0368 and MAE 0.0902 — close
enough to the held-out figure to be direct evidence of no overfitting to
the split used for early stopping. **Quote the 500-sample number**; the
n=50 splits are supporting detail.

All R² values here are **pooled over all pixels**, not averaged over
batches. An earlier batch-averaged implementation understated the
headline as 0.7884; it was found, fixed at source in `train_crn.py`, and
the correction is documented in `docs/TEAM_LOG.md`.

The two caveats, both disclosed rather than hidden:
1. **Training instability** epoch-to-epoch is real, though milder than
   older entries claimed (see §6.6).
2. **Spatial MAGNITUDE is over-expressed — but LOCATION is reliable.**
   These are two different claims and must not be collapsed into one.
   **Never write "spatial reconstruction is unreliable."** It is not what
   the measurements say, and it overstates our own weakness.

   - *LOCATION (reliable):* the predicted hot region is genuinely hotter
     **99.6%** of the time; hottest-20% overlap with truth is **51.3%**
     against 20% chance; per-pixel class accuracy on multi-class samples
     is **79.0%**. Measured by `check_spatial_localization.py`.
   - *MAGNITUDE (weak):* per-sample within-sample R² is negative. This is
     a calibration measure, and it goes negative because true
     within-sample variation (SD ≈ 0.088 pH) is *smaller* than the
     model's own per-pixel error (MAE ≈ 0.094 pH) — so a ~1.25×
     over-amplification is heavily penalised on that scale. Measured by
     `check_spatial_skill.py`.

   Practically the heatmap holds up: **81.5% of held-out pixels fall
   within ±0.15 pH**. Run `evaluate_heatmap.py` for the full set.

3. **A tissue-boundary artefact**, quantified rather than hidden:
   interior MAE is **0.070** against the 0.094 headline. It is
   convolutional context loss at the mask edge. See
   `check_edge_effect.py`.

**"What's your one real-world validation?"** 970nm is the only band
where the simulated bands overlap a real, independently-measured
hyperspectral pork dataset (Wang et al. 2026). The simulated 970nm value
is **0.2645** against real pork **0.198** (freshest cube) / **0.217**
(18-cube mean) — about 1.2–1.3×, down from ~0.54 before the scattering
and eps work. The species question behind it is closed: tray column C4 is
pork, established 2026-09-18 from the dataset authors' own annotation.
The residual gap is **reported honestly as a stated Chapter 5
limitation**, not concealed and not being chased further.

---

## 9. Status — FROZEN as of 2026-09-19

**Nothing in this section is open work.** Every item that used to be
listed here is closed:

1. **Team agreement on the 2026-09-09 `denat_amplitude`/`denat_width`
   resolution** — obtained. Parameters frozen.
2. **Model stabilization** — not pursued further. The instability is
   characterised in §6.6, mitigated by the LR fix and early stopping, and
   disclosed. Five-seed reporting (mean ± std) is what makes the
   headline robust to it, per Henderson et al. 2018.
3. **970nm gap** — decided: documented as a stated Chapter 5 limitation
   (see §8). Not being tuned further.
4. **Manuscript** — the only remaining work on the project, and it is
   writing, not computation. Any chapter text saying "spatial
   reconstruction is unreliable" must be corrected per §8.
5. **Regenerate + run the CRN experiment** — done. It is the headline
   result: R² = 0.8472 ± 0.0384 over five seeds on 500 held-out samples.
   Evidence: `metric_check_outputs/final_evaluation_TEST500.json`.

`RESULTS.md` is the single source of truth for every reportable number
and names the evidence file behind each one. `CLAUDE.md` carries the
current status block and a list of things that look like bugs but are
deliberate and already investigated — read it before raising a finding.

See `docs/TEAM_LOG.md` for the full dated history behind every number and
decision above.
