# XPHECTRA / LIGTAS-pH — context for Claude

This file exists so a fresh Claude session (or a human pasting this
elsewhere) can pick up this project without re-deriving everything from
git history. It is a summary and a map, not the source of truth — where
it says "see X," X is more authoritative than this file.

## The one-sentence version

This is a thesis project that was supposed to measure pork pH from a
physical multispectral camera rig; the rig broke before any real data
was collected, so the project pivoted to a **physics-based synthetic
dataset generator**, and the current work is about making every number
in that simulation traceable to a real published source.

## How we got here: the failed hardware

The original plan was a physical acquisition rig: a multispectral camera
scanning pork samples to build a real image dataset with pH ground
truth, which a neural network (the CRN — see below) would then learn to
predict from. **That rig short-circuited before any data collection
happened.** There is no real pork MSI dataset and there will not be one
on this project's timeline.

With the adviser's approval, the project pivoted:
- Same objectives, same CRN experiment.
- "The system" is now scoped as **the software pipeline**, not a
  physical scanner.
- Instead of real photos, `generate_dataset.py` *simulates* 6-band
  multispectral pork images from a biophysical forward model (see
  below), with pixel-wise pH ground truth that a real camera could never
  give you at that resolution.
- The tradeoff this pivot makes: nothing here is validated against real
  pork data end-to-end. The defense against "this is just made up" is
  that **every physical constant in the simulation must trace to a
  published measurement** — that discipline is the whole point of the
  Parameterized Data Sheet (below). One real external dataset (see NHSI,
  below) exists and is used as a partial reality-check, but it's not the
  training data — it has no pH labels and its wavelengths don't overlap
  cleanly with ours.

This is documented in `README.md`'s opening section ("Why this exists —
read first") and in `docs/TEAM_LOG.md`'s 2026-09-03 entry ("Pivot
confirmed with adviser... Settled — do not re-litigate").

## What's actually in this repo

| File | What it is |
|---|---|
| `generate_dataset.py` | The simulator. Contains `PARAMS` — the forward model's physical constants, each tagged with a citation status. This dict **mirrors the Parameterized Data Sheet** and must be kept in sync with it. Also has `--selftest`, which runs two sanity checks (see below). |
| `generate_dataset_new_plus_eps.py` | A tested-but-not-yet-adopted candidate replacement for part of `generate_dataset.py`'s `PARAMS` (see "Open decision: scattering values" below). Not the file actually used to generate data yet. |
| `extract_sensor_params.py` | Measures real sensor noise and texture statistics from an external real dataset (NHSI pork cubes, Wang et al. 2026) — the closest thing this project has to real measurements. |
| `crn_model.py` | The CRN (`CrudeCRN`): a small U-Net that predicts a dense pH map from a 6-band image. This is the actual model being evaluated — the thing the synthetic dataset exists to test. |
| `train_crn.py` | Trains the CRN on 4 sparse pH points per sample (never the full dense map — that's held out as a hidden evaluation target, which is the core experimental design: can the CRN reconstruct a full field from 4 points using spectral shape alone?). |
| `docs/TEAM_LOG.md` | Session-by-session log: what was changed, what was actually verified (with numbers), what was decided, what's still open. **This is the most current and most trustworthy record in the repo** — more current than `README.md`'s status table, which is dated 2026-09-03 and predates later sessions. |
| `README.md` | Onboarding walkthrough (setup, how to fill in parameters, how to run each step). Good for procedure; check `TEAM_LOG.md` for current numbers, since the README's status table is a few sessions stale. |
| `LIGTAS_calibration.ipynb` | Notebook for parameter work and figures; imports `generate_dataset.py` so the physics lives in one place. |

## The physics, briefly (why this isn't circular)

```
pH  →  protein denaturation  →  scattering  mu_s'(λ, pH)
myoglobin redox state (independent of pH)  →  absorption  mu_a(λ)
(mu_a, mu_s')  →  Kubelka-Munk  →  reflectance R(λ)
R(λ)  →  sensor model (noise, illumination falloff)  →  pixel value
```

pH does **not** map directly to reflectance — it only drives scattering.
Absorption is driven independently by myoglobin's redox state (deoxy /
oxy / met fractions), which varies sample-to-sample as a nuisance
variable uncorrelated with pH. A model has to use spectral *shape*
across all 6 bands to disentangle the two. `generate_dataset.py
--selftest` has two checks for this:

- **Test 1 (sign check):** reflectance must fall as pH rises (low pH =
  PSE = pale = high reflectance). This is a hard pass/fail on whether
  the physics direction is right.
- **Test 2 (non-triviality check):** a plain linear fit on raw pixels
  must do *poorly*. If a linear model could solve it, the CNN/CRN proves
  nothing. Target: R² comfortably under 0.9.

**Important recent correction on Test 2:** until 2026-09-05, this test
used a single fixed random seed. That number was found to be unreliable
— off from the true average by 0.05–0.08 depending on configuration, in
either direction. `self_test()` now averages 10 seeds and reports
mean/std. **Treat any single-number R² quoted anywhere in this project's
older history as suspect** unless it explicitly says "mean over N
seeds." Current honest baseline: **R² mean ≈ 0.59, std ≈ 0.02** (10
seeds), on the current `PARAMS`.

## The Parameterized Data Sheet

This is the project's provenance tracker — a Google Sheet, now split
into four tabs, that `generate_dataset.py`'s `PARAMS` dict must mirror:

1. **Parameters** — the actual free/tunable model inputs (14 of them,
   see below). Each needs a real citation or measurement before the
   dataset can be called defensible.
2. **Constants & Derivations** — fixed physical/mathematical constants
   used in unit conversions (e.g. muscle density, myoglobin molecular
   weight, the ln(10) decadic→Napierian factor). These aren't tunable
   and don't need a "sweep" — they're arithmetic, already implemented
   and verified in `mu_a()`.
3. **Validation Checks** — outputs, not inputs: the sign check, the
   isosbestic check, the non-triviality (linear baseline) check, and the
   970nm external reality check against real NHSI pork data (currently
   **failing** — simulated ~0.53-0.58 vs real ~0.19).
4. **Proposed Cuts** — simplification ideas that have been evaluated but
   not necessarily applied (e.g. folding `water_fraction` into
   `mua_water`, or using a haemoglobin spectrum as a proxy for the
   missing myoglobin data at 481/600nm).

The sheet was reorganized into these 4 tabs on 2026-09-05 after
recognizing the original single "Parameters" tab conflated real tunable
parameters with fixed constants and diagnostic checks — the constants
and checks will never move to "CITED" because they aren't citable
values, they're derivations and measurements of the model's output.

### The 14 real parameters, and where they stand

**Done (3):**
- `c_Mb_mean`, `c_Mb_sd` — myoglobin concentration (Cross et al. 2018)
- `sensor_sigma` — measured directly from real NHSI pork cubes

**Still open, roughly in the order they should be tackled (items 1–2 now
resolved; numbering kept so the "#1 / #3" cross-references below still
line up):**

1. ~~`mua_water`~~ — **DONE.** Wired in from Hale & Querry 1973 via the
   omlc.org data file: `[0.00025, 0.00032, 0.00079, 0.0023, 0.016, 0.45]`
   cm⁻¹ (970nm = 0.45 exact; visible bands near-zero). Status `CITED`.
2. ~~`water_fraction`~~ — **DONE.** `0.732`, cited to Wojtasik-Kalinowska
   et al. 2016 (LWT 67:112-117), mean of the study's 4 diet-treatment
   groups, pork *Longissimus dorsi*. Honikel 1998 is no longer the source.
3. `scatter_a` / `scatter_b` — **highest-priority item on the sheet.**
   Status in `PARAMS` is now `"DECISION"` (blocks in `--selftest`).
   Current values are a generic soft-tissue average (Jacques 2013); a
   porcine-muscle-specific refit (Bergmann et al. 2021) has already been
   tested and recommended in `generate_dataset_new_plus_eps.py` — it
   matches or beats every alternative on every metric tried (970nm
   match, R², data integrity) with no added complexity. This is a
   pending sign-off, not more research. It also directly affects the
   failing 970nm check.
4. `denat_midpoint` — currently 5.70 (Cross et al., unselected
   population); quality-stratified studies suggest normal-class pork
   sits lower (5.4–5.6). Recommended Ch.3 language already drafted;
   needs a decision.
5. `mu_a_baseline` — tuned/uncited nuisance term. Explicitly sequenced
   **after** #3 and #1 are resolved — it may shrink or become
   unnecessary once real scattering and water values are in.
6. `denat_amplitude` — not a single citable value by design; needs an
   actual sweep (0.2/0.4/0.6/0.8) run and reported as a Chapter 4
   sensitivity result. A literature-anchored ceiling exists (Offer &
   Knight 1988 via Kim et al. 2014).
7. `eps_oxy` at 481/600nm — **the sheet's #1 blocker.** No visible-range
   oxymyoglobin curve exists in Bowen 1949 at all; this is a genuine
   research gap, not just an unread table. 600nm is also the
   highest-weighted band in the design, making this the most exposed
   cell on the whole sheet.
8. `eps_deoxy`, `eps_met` at 481/600nm — same underlying gap. A
   haemoglobin-spectrum proxy (Prahl, omlc.org — fully tabulated, and
   spectroscopically similar to myoglobin in this range per Grabtchak
   et al. 2014) would close all three of these rows (#7, #8) in one
   team decision, at the cost of declaring the proxy explicitly in Ch.3
   and Ch.5.
9. `denat_width` — genuinely unsourced after an active search. Lowest
   priority of the remaining research gaps since there's already a
   fallback: report it as a second sensitivity axis alongside
   `denat_amplitude` (a joint sweep found it moves R² by 0.07–0.10, not
   safely ignorable — that number used the old single-seed method,
   though, and needs re-confirming with the 10-seed method before being
   treated as final).

## Recent history worth knowing about

- **2026-09-03:** the parameter table was seeded with real citations for
  the first time (Tang 2004 for myoglobin eps, Cross et al. 2018 for
  concentration, Jacques 2013 for scattering, measured `sensor_sigma`
  from real NHSI cubes). A real units bug was caught and fixed in the
  myoglobin-concentration unit conversion (was 1000x too small).
- **2026-09-04:** a full comparison of scattering models (generic vs.
  porcine-specific) plus a from-scratch investigation isolated which
  parts of that change actually helped, refit the original formula's
  constants instead of adopting a more complex one, and — separately —
  discovered the single-seed R² measurement bug described above.
- **2026-09-05:** `texture_amplitude` (a cosmetic, uncited surface-noise
  term) was cut from the model. The first attempt to justify this cut
  used the since-discredited single-seed method and wrongly concluded
  the cut was nearly free; re-testing with the honest 10-seed method
  showed it actually raises R² by a real +0.07 (comparable to the
  scattering-model change). The cut was kept anyway (still an uncited
  term, R² is still safely under 0.9), but documented as a real,
  disclosed trade-off rather than a free win. Full numbers in
  `docs/TEAM_LOG.md`'s 2026-09-05 entry.

## What to actually do next

1. ~~Fill in `mua_water` and `water_fraction`~~ — DONE (Hale & Querry
   1973 via omlc.org; Wojtasik-Kalinowska et al. 2016). See the "Still
   open" list above.
2. Get a decision on adopting `generate_dataset_new_plus_eps.py`'s
   `scatter_a`/`scatter_b`/`mu_a_baseline` values into `generate_dataset.py`
   — this is the single highest-leverage open item. Their `PARAMS` status
   is now `"DECISION"` so `--selftest` flags them as blocking.
3. ~~Decide `denat_midpoint`~~ — DECIDED: keep 5.70 as a labelled proxy,
   with a stated 5.4–5.7 uncertainty band in Ch.3. Status `CITED (PROXY)`.
4. Re-test `mu_a_baseline` once #2 is in.
5. Run and write up the `denat_amplitude` sweep as a Ch.4 result (use
   the 10-seed method).
6. Decide on the haemoglobin-proxy approach for `eps_oxy`/`eps_deoxy`/
   `eps_met` at 481/600nm — this is the project's real remaining
   research blocker, not just a TODO.
7. `denat_width` — report as a sensitivity axis if no citation surfaces.
8. Once the parameter table is fully CITED/MEASURED: regenerate the
   dataset, re-run `--selftest`, and only then treat any R² number or
   CRN result as reportable. Nothing generated so far should be quoted
   as final — every entry in `docs/TEAM_LOG.md` says so for a reason.

For anything not covered here — exact numbers, exact citations, what was
tried and rejected and why — read `docs/TEAM_LOG.md` in full. It's
structured chronologically, newest first, and is more detailed and more
current than this file will stay over time.
