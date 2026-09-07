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
| `generate_dataset.py` | The simulator — **the only generator now.** Contains `PARAMS` — the forward model's physical constants, each tagged with a citation status. This dict **mirrors the Parameterized Data Sheet** and must be kept in sync with it. Also has `--selftest`, which runs two sanity checks (see below). A parallel candidate file, `generate_dataset_new_plus_eps.py`, existed 2026-09-04 through 09-07 to test scattering/eps changes in isolation before adopting them; it has since been fully merged and retired — don't look for it. |
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
seeds." Current honest baseline: **R² mean ≈ 0.69, std ≈ 0.02** (10
seeds), on the current `PARAMS` (post scatter_a/b + eps NIR + retuned
`mu_a_baseline`) — this number has moved twice since it was 0.59 and
will move again once the remaining 5 blockers close; always re-run
`--selftest` rather than trust this file's snapshot.

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

**Done (6):**
- `c_Mb_mean`, `c_Mb_sd` — myoglobin concentration (Cross et al. 2018)
- `sensor_sigma` — measured directly from real NHSI pork cubes
- `mua_water` — Hale & Querry 1973 via omlc.org, linearly interpolated
- `water_fraction` — 0.732, Wojtasik-Kalinowska et al. 2016
- `denat_midpoint` — kept 5.70 as a labelled proxy (decided, not a TODO)
- `scatter_a`/`scatter_b` — see #1 below, now resolved

**Still open (3 blocking parameters), roughly in the order they should be
tackled:**

1. `eps_met` at 481/600nm — the last piece of the old "#1 blocker."
   `eps_oxy` and `eps_deoxy` at these bands were closed 2026-09-07 with
   a haemoglobin-shape proxy (Prahl omlc.org, anchored at the cited
   525nm isosbestic — see "Resolved since" below). Methemoglobin is
   **not** in the Prahl file, so that proxy doesn't extend to metMb, and
   a deoxy-Hb substitute is wrong in a known direction (misses metMb's
   ~630nm band). Next step, small and scoped: use Tang (2004) Table 2's
   own 503 and 582 nm metMb entries as direct anchors, else a digitized
   metHb spectrum (Zijlstra & Buursma). Do it in isolation.
2. `denat_amplitude` — not a single citable value by design; needs an
   actual sweep (0.2/0.4/0.6/0.8) run and reported as a Chapter 4
   sensitivity result. A literature-anchored ceiling exists (Offer &
   Knight 1988 via Kim et al. 2014). Plan agreed for a while, not yet
   executed — lowest-effort real blocker left.
3. `denat_width` — genuinely unsourced after an active search; current
   value (0.28) is a derived central estimate from two independent
   published transitions, not a direct citation. Report as a second
   sensitivity axis alongside `denat_amplitude` if nothing better
   surfaces.

**Resolved since the list above was first written:**
- `eps_oxy` / `eps_deoxy` at 481/600nm — **closed 2026-09-07 via
  haemoglobin-shape proxy.** Each value = 7.60 (the cited 525nm
  isosbestic) × that form's Prahl-omlc haemoglobin ratio HbX(band)/
  HbX(525nm); oxyMb←HbO2, deoxyMb←Hb. The same recipe reproduces the
  already-cited 573nm values to 0.0% (oxy) / 8% (deoxy), which is the
  evidence it holds. Declared a PROXY, not a citation — state in
  Ch.3/Ch.5; `check_params()` prints `[ note ]`, not `[ OK ]`. Blocking
  count 5→3. No measurable R² cost (0.6902→0.6897, inside seed noise).
  `eps_met` deliberately excluded — see open item 1. Full record:
  `docs/TEAM_LOG.md` 2026-09-07 (cont. 2).
- `scatter_a`/`scatter_b` — **adopted** (was the `"DECISION"` blocker,
  the former #1 item here). Jacques 2013 generic soft-tissue values
  (18.9/1.286) → refit to approximate Bergmann et al. 2021's
  porcine-specific curve (8.7436/1.6618), same formula. Closed ~47% of
  the 970nm gap, dropped blocking count 7→5, at a disclosed R² cost
  (0.530→0.647 at adoption time).
- eps NIR values (730/970nm) — adopted separately from the scattering
  merge (tested in isolation, not bundled). `0.0` → small non-zero
  values. Not a citation — a tuned adjustment, documented as such in
  code, that further improved the 970nm match at no measurable R² cost.
- `mu_a_baseline` — re-swept fresh (0.3–0.8) against the post-scattering
  PARAMS, as sequenced above. This time there was no free improvement —
  a genuine trade-off across the whole range. Adopted `0.8`, prioritizing
  the 970nm match; `0.3` remains an equally defensible alternative if
  the team would rather keep more R² margin. Full sweep table in
  `docs/TEAM_LOG.md`.
- `generate_dataset_new_plus_eps.py` — retired. Fully merged into
  `generate_dataset.py`; kept drifting out of sync with the official
  file's ongoing updates every time it wasn't touched, which is exactly
  the kind of confusion multiple people on the team ran into. Don't
  recreate a parallel candidate file for future changes — test in
  isolation on a throwaway copy instead, then merge directly.

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
- **2026-09-06/07:** the `"DECISION"` blocker on `scatter_a`/`scatter_b`
  was resolved and adopted, eps NIR values were adopted separately, and
  `mu_a_baseline` was re-swept fresh against the new state (adopted 0.8,
  a genuine trade-off this time, not a free win like earlier sweeps
  found). `generate_dataset_new_plus_eps.py` was retired after fully
  merging into `generate_dataset.py`. Then `eps_oxy`/`eps_deoxy` at
  481/600nm were closed with a haemoglobin-shape proxy (Prahl omlc.org,
  525nm-isosbestic anchor). Blocking count went 7→5→3. Full before/after
  numbers for every step in `docs/TEAM_LOG.md`'s four entries from this
  window — each change was tested in isolation on a throwaway copy
  before being applied to the real file, confirmed identical both times.

## What to actually do next

1. ~~Fill in `mua_water` and `water_fraction`~~ — DONE.
2. ~~Adopt `scatter_a`/`scatter_b`~~ — DONE. Resolved and merged.
3. ~~Decide `denat_midpoint`~~ — DONE. Kept 5.70 as a labelled proxy.
4. ~~Re-test `mu_a_baseline`~~ — DONE. Re-swept, adopted 0.8.
5. **Run and write up the `denat_amplitude` sweep as a Ch.4 result** (use
   the 10-seed method) — plan agreed for a while, not yet executed.
   Lowest-effort real blocker remaining.
6. ~~Haemoglobin-proxy for `eps_oxy`/`eps_deoxy` at 481/600nm~~ — DONE
   2026-09-07 (525nm-isosbestic anchor × Prahl omlc Hb shape; declared a
   proxy). **`eps_met` at 481/600nm still open** — the Prahl file has no
   methemoglobin. Next: anchor on Tang (2004)'s own 503/582nm metMb
   entries, else digitize metHb (Zijlstra & Buursma).
7. `denat_width` — report as a sensitivity axis if no citation surfaces.
8. Decide on the 970nm gap: it's improved a lot (sim ~0.30 vs real
   ~0.19, was ~0.54) but not closed. Tune further, or document as a
   stated Ch.5 limitation.
9. Once the parameter table is fully CITED/MEASURED: regenerate the
   dataset, re-run `--selftest`, and only then treat any R² number or
   CRN result as reportable. Nothing generated so far should be quoted
   as final — every entry in `docs/TEAM_LOG.md` says so for a reason.

For anything not covered here — exact numbers, exact citations, what was
tried and rejected and why — read `docs/TEAM_LOG.md` in full. It's
structured chronologically, newest first, and is more detailed and more
current than this file will stay over time.
