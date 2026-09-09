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
seeds." Current honest baseline: **R² mean ≈ 0.65, std ≈ 0.02** (10
seeds), on the current `PARAMS` (post scatter_a/b + fully digitized eps
+ retuned `mu_a_baseline` + swept `denat_amplitude`=0.4, **not yet
pushed**) — this number has moved several times as parameters got
resolved; always re-run `--selftest` rather than trust this file's
snapshot.

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
   970nm external reality check against real NHSI pork data (still
   **not closed**, but much improved — simulated ~0.27 vs real ~0.19,
   was ~0.53-0.58).
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

**Done (7):**
- `c_Mb_mean`, `c_Mb_sd` — myoglobin concentration (Cross et al. 2018)
- `sensor_sigma` — measured directly from real NHSI pork cubes
- `mua_water` — Hale & Querry 1973 via omlc.org, linearly interpolated
- `water_fraction` — 0.732, Wojtasik-Kalinowska et al. 2016
- `denat_midpoint` — kept 5.70 as a labelled proxy (decided, not a TODO)
- `scatter_a`/`scatter_b` — see "resolved" list below
- `eps_oxy`/`eps_deoxy`/`eps_met` (all 6 bands) — see "resolved" list
  below, now fully digitized
- `denat_amplitude`/`denat_width` — see "resolved" list below, now
  swept and verified

**Still open (0 blocking parameters).** For the first time in this
project's history, `check_params()` prints no `!!` warning line at
all. **Pushed 2026-09-09 for team review — not yet formally agreed by
the team**; see the `denat_amplitude`/`denat_width` entry in "resolved"
below before treating any downstream number as final.

**Resolved since the list above was first written:**
- `denat_amplitude`/`denat_width` — **swept and adopted, 2026-09-09,
  pushed for team review.** A literature-anchored value was proposed for
  `denat_amplitude` (2.19, from Offer & Knight 1988's ~2x PSE-vs-normal
  scattering claim, via Kim/Warner/Rosenvold 2014 — independently
  re-verified against the actual PDF text, citation confirmed accurate)
  but checked and REJECTED: the model is structurally capped at 1.49x
  at the citation's actual comparison (PSE vs. normal, i.e. pH 5.4 vs.
  the cited midpoint 5.70) for any amplitude, and forcing a literal 2x
  by anchoring across the wider domain instead pushes linear-baseline
  R² to 0.9437 — breaking the project's own "must stay comfortably
  under 0.9" non-triviality requirement — while also worsening the
  970nm match. Adopted instead: a verified sweep (0.2/0.4/0.6/0.8, real
  10-seed R²/970nm-gap numbers, not just planned), operating value 0.4.
  `denat_width` kept at 0.28 but its prior citation (MacDougall & Jones
  1981) was checked, found not to actually support the claim made for
  it, and retracted — now `"SWEPT -- unsourced, no citation found"`.
  Blocking count 2→0. Full sweep table, the literature-value math, and
  the citation re-verification against the actual source PDFs are in
  `docs/TEAM_LOG.md`'s 2026-09-09 entry. **Pushed to `origin/main` for
  the team to review directly in the code — same disclosed-decision
  process as the eps override, but this one doesn't overwrite anyone
  else's commit. Still needs actual team discussion and explicit
  agreement before treating it as the final, adopted state — a push is
  "here's what I propose, please check it," not "this is settled."**
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
- `eps_oxy`/`eps_deoxy`/`eps_met` at 481/600nm — **the sheet's former #1
  blocker, resolved — but as a DISCLOSED OVERRIDE of a teammate's
  already-pushed commit, not a clean-slate resolution.** A separate
  track independently closed `eps_oxy`/`eps_deoxy` @ 481/600nm first
  (`origin/main` commit `188e76a`), via a haemoglobin-shape proxy (Prahl
  omlc.org table, anchored at the 525nm isosbestic) — fully reproducible
  from two public tables, self-checked against the cited 573nm value,
  and independently re-verified by hand in this session. It left
  `eps_met` open on principle (no methemoglobin entry in the Prahl
  table). This session's work, done in parallel without knowing about
  that commit at first, instead digitized the myoglobin curves directly
  from Tang 2004 Fig.1 (481/573/600nm) and Bowen 1949 Figs.1-2
  (730/970nm) — avoiding the extra Hb-for-Mb substitution `188e76a`
  needed, and additionally resolving `eps_met`. Once the conflict was
  found, the digitization was re-verified against the raw
  WebPlotDigitizer screenshots (now checked in at `docs/digitization/`)
  before adopting it over `188e76a`'s values. Blocking count 5→2 (vs.
  `188e76a` alone reaching 3). 970nm gap improved further (0.093→0.076,
  the closest yet — `188e76a` didn't touch NIR) at no measurable R² cost
  across either version (0.6902 → 0.6897 `188e76a` / 0.6923 this
  version, all within the ~0.017 seed-noise band). **Team-agreed
  override, reconciled with `188e76a` via rebase** — see
  `docs/TEAM_LOG.md`'s 2026-09-08 entry for the full comparison and the
  reasoning for overriding rather than keeping `188e76a`'s values. Also
  flagged there: across old-placeholder / `188e76a` / this version,
  `eps_oxy`/`eps_deoxy` @ 600nm — the highest-weighted band in the
  design — span more than a 2.5x range, which is a real unresolved
  uncertainty no single version settles.

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
  525nm-isosbestic anchor, `origin/main` commit `188e76a`). Blocking
  count went 7→5→3. Full before/after numbers for every step in
  `docs/TEAM_LOG.md`'s four entries from this window — each change was
  tested in isolation on a throwaway copy before being applied to the
  real file, confirmed identical both times.
- **2026-09-08:** a second, parallel track closed the same eps
  481/600nm blocker independently, via a full digitization of
  `eps_deoxy`/`eps_oxy`/`eps_met` from Tang (2004)/Bowen (1949) figures
  (all 6 bands, not just 481/600). Once the two tracks were discovered
  to conflict, the digitization was adopted over `188e76a`'s proxy as a
  disclosed override — not silently — after independently re-verifying
  it against the raw digitizer screenshots (now in `docs/digitization/`)
  and weighing it against `188e76a`'s own, genuinely reproducible,
  method. Blocking count went 3→2 (`eps_met`, which `188e76a` left
  open, is now resolved too). 970nm gap improved further (0.093→0.076)
  at no measurable R² cost. Full comparison and reasoning in
  `docs/TEAM_LOG.md`'s 2026-09-08 entry.
- **2026-09-09:** `denat_amplitude`/`denat_width` closed. A literature
  value (2.19, from Offer & Knight 1988 via Kim/Warner/Rosenvold 2014 —
  independently re-verified against the actual PDF text) was checked
  and rejected: it breaks the non-triviality requirement (R²=0.9437)
  and is structurally incompatible with the already-cited midpoint
  (capped at 1.49x, not 2x, at the citation's actual comparison).
  Adopted a verified sweep instead (0.2/0.4/0.6/0.8, operating value
  0.4). `denat_width`'s prior citation was checked and retracted (not
  actually supported), kept at 0.28 as an unsourced sweep value.
  Blocking count 2→0 — the parameter table has zero uncited/unmeasured
  entries for the first time in this project. **Pushed for team
  review** — visible on `origin/main` now, but still needs an actual
  team conversation and agreement before treating it as final, same
  standard as the eps decision.
  Full sweep table and citation verification in `docs/TEAM_LOG.md`'s
  2026-09-09 entry.

## What to actually do next

1. ~~Fill in `mua_water` and `water_fraction`~~ — DONE.
2. ~~Adopt `scatter_a`/`scatter_b`~~ — DONE. Resolved and merged.
3. ~~Decide `denat_midpoint`~~ — DONE. Kept 5.70 as a labelled proxy.
4. ~~Re-test `mu_a_baseline`~~ — DONE. Re-swept, adopted 0.8.
5. ~~Decide on `eps_oxy`/`eps_deoxy`/`eps_met` at 481/600nm~~ — DONE. A
   haemoglobin-proxy (`origin/main` commit `188e76a`) closed `eps_oxy`/
   `eps_deoxy` first, leaving `eps_met` open; a parallel digitization
   track then closed all three, and was adopted over the proxy as a
   team-agreed, disclosed override (not a silent overwrite) — see
   `docs/TEAM_LOG.md`'s 2026-09-08 entry for the full reasoning and the
   `188e76a` reconciliation.
6. ~~Run and write up the `denat_amplitude` sweep~~ — DONE. A literature
   value (2.19) was proposed, checked, and rejected (breaks R²<0.9,
   structurally incompatible with the cited midpoint); adopted a
   verified sweep instead (0.2/0.4/0.6/0.8, operating value 0.4). **Not
   yet pushed** — needs a real team conversation, same as item 5.
7. ~~`denat_width`~~ — DONE. No citation surfaced; prior citation
   (MacDougall & Jones) was checked and found not to hold, retracted.
   Kept at 0.28 as an explicitly unsourced sweep value.
8. Decide on the 970nm gap: it's improved a lot (sim ~0.264 vs real
   ~0.19, was ~0.54) but not closed. Tune further, or document as a
   stated Ch.5 limitation. Now the top open item, along with pushing
   items 5/6 once agreed.
9. Once items 5/6 are pushed: regenerate the dataset, re-run
   `--selftest`, and only then treat any R² number or CRN result as
   reportable. Nothing generated so far should be quoted as final —
   every entry in `docs/TEAM_LOG.md` says so for a reason.
10. Run the actual CRN experiment (`train_crn.py` on the frozen
    dataset) — hasn't been touched by any recent session and is the
    actual point of the pipeline, not a footnote after the parameter
    table is clean.
11. Check the manuscript against the current code/TEAM_LOG state —
    flagged as possibly drifted, untouched by recent technical work.

For anything not covered here — exact numbers, exact citations, what was
tried and rejected and why — read `docs/TEAM_LOG.md` in full. It's
structured chronologically, newest first, and is more detailed and more
current than this file will stay over time.
