# Team Context Log

Short, structured entries — not prose. One entry per work session per
person/track. Newest entry on top. Purpose: so anyone (human or AI)
picking this up later doesn't have to re-derive what was actually done,
actually checked, and still open.

Template:

```
## [Date] — [Name/track]
**Changed:** what was actually done (files/commits touched)
**Verified:** what was actually checked, with numbers/evidence — not
  just "looks good" or "checks out." If something wasn't checked, say so.
**Decided:** any open decision that got resolved, and why
**Still open:** unresolved questions, explicitly owned or unowned
**Context to feed next session:** 2-3 lines of what the next person
  needs to know before touching this part of the project
```

---

## 2026-09-05 — Cut `texture_amplitude`; kept `denat_width` as-is (via Claude session)

**Changed:**
- `generate_dataset.py`: removed the `texture_amplitude` parameter from
  `PARAMS` and the multiplicative texture term from `generate_sample()`.
  Cube composition is now `R * illum` (was `R * texture * illum`). Left a
  comment in `PARAMS` pointing here for the decision record.
- No other parameters touched. `mu_a_baseline` and `denat_width` were both
  considered as cut candidates and explicitly rejected — see below.

**Verified (numbers):**
- Ran a joint `denat_amplitude` (0.2/0.4/0.6/0.8) x `denat_width`
  (0.15/0.22/0.30) grid using the same linear-baseline methodology as
  `self_test()` TEST 2 (n=40 samples, 400 px/sample, seed 0), via a
  throwaway script, not committed. `denat_width` moves R² by **0.07-0.10**
  at every amplitude level, monotonically (narrower width -> higher R²) —
  comparable in size to the scattering-model swap logged 2026-09-04
  (+0.069). **Not safe to treat as inert; not cut.**
- After cutting `texture_amplitude`: `--selftest` TEST 1 (sign) still
  PASS. TEST 2 linear-baseline R² moved **0.5839 -> 0.5660** (-0.018,
  MAE 0.166 -> 0.170) — a small move, in line with the parameter sheet's
  own prediction that this term "barely moves" R²/CRN performance.
  `check_params()` blocking count dropped **8 -> 7**.

**Decided:**
- Cut `texture_amplitude`: it was ASSUMED/uncited, self-flagged as
  cosmetic (doesn't encode the pH mechanism, unlike `eps_*`/`scatter_*`/
  `denat_*`), and its removal changed R² by only ~0.02 — confirmed
  empirically, not just predicted. One fewer uncited nuisance parameter
  to defend in Ch.3/Ch.5, no meaningful loss.
- Did NOT cut `mu_a_baseline`: it exists specifically to hold 730nm
  reflectance near realistic pork values (~0.70 instead of ~0.90-0.92).
  Removing it would push NIR reflectance brighter, worsening the
  already-failing 970nm real-world check (sim ~0.53 vs real ~0.19) rather
  than helping it. Left in place, still flagged TUNED/uncited.
- Did NOT fold `denat_width` into the `denat_amplitude` sweep as a
  "checked, doesn't matter" justification — the grid above shows it does
  matter. It stays a separate open TODO (find a PSE-transition-steepness
  citation, or report it as its own sensitivity axis in Ch.4 alongside
  amplitude).

**Still open:**
- `denat_width` still needs either a literature source or an explicit
  Ch.4 write-up as a second sensitivity axis (not yet written up either
  way).
- Parameterized Data Sheet (Google Sheet) not yet updated to mark
  `texture_amplitude` as CUT — do that alongside this entry so the sheet
  and code don't drift.
- All other open items from prior entries (eps 481/600nm, `mua_water`
  wiring, `water_fraction` citation, `denat_amplitude` sweep writeup,
  failing 970nm check) are unaffected by this session and still open.

**Context to feed next session:**
- `texture_amplitude` no longer exists in `generate_dataset.py` — don't
  re-add it without re-running the same before/after R² check done here.
- The `denat_amplitude` x `denat_width` sweep script used above was a
  throwaway, not committed. Re-run rather than assuming it still exists
  if this needs reproducing.
- Do not quote a final linear-baseline R² yet — still moving with the
  usual open blockers (eps 481/600, `mua_water`, `water_fraction`,
  `denat_width`).

---

## 2026-09-04 — Scattering-model comparison (via Claude session)

**Changed:**
- Nothing committed to `generate_dataset.py` yet. A candidate alternative,
  `generate_dataset_changed.py` (not yet tracked in git), replaces the
  Jacques 2013 generic "other soft tissues" scattering power law
  (`scatter_a`/`scatter_b`) with a **tissue-specific two-term
  Rayleigh+Mie model from Bergmann et al. 2021** (porcine muscle
  specifically) — adds `scatter_c`, `scatter_lambda0`. Also gives the
  myoglobin coefficients small non-zero values at 730/970nm (were flat
  0.0). Rewrote `mu_s_prime()` accordingly. Everything else (KM
  equations, pH→scattering causal link, myoglobin nuisance logic,
  texture, sensor noise) is untouched.
- Ran a full side-by-side comparison: both `--selftest`s, a 970nm
  real-world check, RPD, and an n=20 generation/integrity check into
  `test_original/` and `test_changed/` (neither committed).

**Verified (numbers):**
- Sign test: **PASS in both** — reflectance falls with pH in all 6
  bands, both versions.
- **970nm reflectance vs. the real NHSI reference (~0.19):** original
  avg 0.553 (gap 0.363), changed avg 0.382 (gap **0.192, ~half**).
  Changed is meaningfully closer to the one real external check this
  project has.
- **Linear-baseline non-triviality test:** R² 0.5839→**0.6532**
  (+0.069, MAE 0.1663→0.1514 pH). RPD (same self-test sample,
  RMSE-derived) 1.550→1.698. **R² is climbing further** on the trend
  already being watched (~0.45 pre-parameterization → 0.58 → 0.65 now)
  — still safely under the 0.9 "too easy" alarm, but the direction is
  unfavorable for the "the CRN has non-trivial work to do" argument.
  Note: RPD "weak" (<2) in both is *expected and desired* here — that
  literature bar (2.5–3 = strong) judges a deployed model's calibration,
  not this deliberately-dumb linear strawman. It belongs on the CRN's
  own eventual RPD, not this baseline.
- **Data integrity:** n=20 generated with each, 0 NaN / 0 negative / 0
  values >1.0 in either, across 737,188 meat pixels each. Per-band
  mean/std reflectance recorded for both (see chat log / re-run
  `check_generated.py`-style script if needed — not committed).

**Decided:**
- Nothing adopted yet. This session's output is a comparison + a
  recommendation (lean toward the changed scattering model, given the
  real-world 970nm match, *if* the R² trade-off is explicitly documented
  rather than discovered later) — **not** a decision. Whoever owns this
  parameter should make the actual call.

**Still open:**
- Whether to actually adopt `generate_dataset_changed.py`'s scattering
  model into `generate_dataset.py` — unresolved, needs an owner's
  decision, not just this comparison.
- The realism improvement is confirmed at **970nm only**. The same
  change shifts all 5 other bands too, and there is no real reference
  to check whether those moved toward or away from truth.
- The R² increase hasn't been attributed to a specific cause — the diff
  bundles the new scattering law AND new NIR myoglobin terms together;
  which one (or both) drives the R² climb is not separated.
- Not attempted: tuning an unrelated parameter (e.g. `texture_amplitude`)
  to try to recover a lower R² while keeping the 970nm improvement.
- All the usual open items from the parameter-sourcing track (eps at
  481/600nm, `mua_water`, `water_fraction`, `denat_amplitude` sweep, the
  failing 970nm check in the *current* `main` params) are unaffected by
  this comparison and still open.

**Context to feed next session:**
- `generate_dataset_changed.py`, `test_original/`, `test_changed/` are
  **local, untracked, not pushed**. If you need this comparison
  reproduced, the exact diff and commands are in this entry's numbers
  above — re-run rather than assuming stale local files still exist.
- Do not treat either version's linear-baseline R² as final — both are
  still running on the same 8 unresolved parameter blockers logged in
  the 2026-09-03 entry below.

---

## 2026-08-31 — Harvey & Raymond (model/data track, via Claude session)

**Changed:**
- Fixed `self_test()`'s hardcoded `/tmp` path in `generate_dataset.py`
  (commit `1d43b48`, pushed to `origin/main`) — crashed on Windows since
  `/tmp` doesn't exist there and was never created. Now uses
  `tempfile.mkdtemp()` + `shutil.rmtree()` cleanup.
- Increased the pH field's spatial smoothing scale in `make_ph_field()`
  from 18px (~0.35cm correlation length) to 77px (~1.5cm) — verification
  found the old scale produced a mottled/speckled field, not the "smooth
  gradient" the code's own docstring describes. Team decision (b) of 3
  options presented. Regenerated the dataset because this parameter
  actually changed.
- Added `crn_model.py` (`CrudeCRN`: small U-Net, configurable
  `in_res`/`out_res`, head bias-initialized to a pH prior of 5.9 so
  training doesn't waste early epochs correcting a large constant
  offset) and `train_crn.py` (sparse-point + total-variation-regularized
  training loop; `--out-res` and `--n-points` are CLI parameters, not
  hardcoded).
- `requirements.txt`: added `torch>=2.1`, `matplotlib>=3.7`.
  `.gitignore`: added `xphectra_venv/`, `crn_outputs/`.
- Created `xphectra_venv` (Python 3.11.5) as the project's venv.
- Fixed a second bug in `train_crn.py`: the final heatmap/report used
  whatever the *last* epoch left the model at, not the checkpoint
  actually saved as `crn_best.pt` (lowest val MAE). Now reloads
  `crn_best.pt` before generating the reported heatmap.
- Added `--weight-decay` CLI arg to `train_crn.py` (default 0.0 — no
  behavior change unless passed) to support a diagnostic run.
- Added `train_r2`/`val_r2` to the per-epoch `history.json` record
  (previously only MAE was stored there).
- Changed `train_crn.py`'s `--lr` default from `1e-3` to `1e-4` (see
  Decided below — confirmed stable over a full 50-epoch run, not just
  the 15-epoch diagnostic). Added a docstring note explaining why.
- Added a note to `crn_model.py`'s docstring on why BatchNorm was kept
  (LR was the diagnosed cause, not architecture — see Decided below).
- Committed `crn_outputs_lr1e4_50ep/loss_curve.png`, `sanity_check_heatmap.png`,
  and `history.json` as the reference 50-epoch result (the `crn_best.pt`
  checkpoint itself stays out of git, per the existing blanket `*.pt`
  rule — model binaries aren't normally committed).
- Added a Part-B section to `README.md` — it previously documented only
  `generate_dataset.py`/`extract_sensor_params.py`, with no mention that
  `crn_model.py`/`train_crn.py` existed or how to run them.

**Verified (numbers, not "looks good"):**
- Self-test after the pH-scale change: sign test **PASS** (reflectance
  falls with pH in all 6 bands); linear baseline **R²=0.4513** (was
  0.4529 before the change — task did not become trivial).
- Dataset regenerated cleanly: 400 samples, 300/50/50 train/val/test
  split, same file structure as before.
- CRN 15-epoch run (defaults: `out-res=256`, `n-points=4`, `batch=8`,
  `lr=1e-3`, `tv-weight=0.05`): train sparse MSE **0.089 → 0.012**
  (steady decrease); train MAE (hidden-map sanity check only)
  **0.246 → 0.095 pH**; train R² rising to **~0.7–0.8** by later epochs.
  **Best val MAE = 0.084 pH (epoch 8)** — but val metrics are **volatile
  across epochs**, not smoothly converging: val MAE swings between 0.084
  and 1.174 pH, val R² swings between +0.87 and −18.7 epoch to epoch.
  The saved sanity-check heatmap (val `sample_301`) is one of the
  **poorly-predicted** cases — true pH ≈5.8–6.05, predicted skewed low
  ≈5.0–6.0, near-uniform ~0.6–0.9 error across the region. Reported as-is,
  not cherry-picked.
- `--out-res 32`: ran cleanly, produced a coarse 32×32 grid correctly
  upsampled for the hidden-map comparison. Confirmed working.
- `--n-points 8`: correctly prints a warning and clamps to 4 (dataset
  only embeds 4 probes/sample); ran without crashing. Confirmed working.
- **Diagnosed the val instability with 3 isolated 15-epoch runs**, same
  dataset, one variable changed per run, no architecture change (per
  explicit instruction to understand the cause before touching
  BatchNorm/etc.):
  - Baseline repeat (`lr=1e-3`, no weight decay): val MAE range
    **0.059–1.065** (spread 1.006), val R² range −14.2 to +0.93.
    Confirms the instability reproduces — not a one-off fluke run.
  - `lowlr` (`lr=1e-4`): val MAE range **0.108–0.599** (spread 0.491 —
    roughly half of baseline). Clear stabilizing effect.
  - `wd` (`lr=1e-3`, `weight_decay=1e-4`): val MAE range **0.082–0.705**
    (spread 0.623) — barely different from baseline. Regularization is
    NOT the main lever.
  - Val set size confirmed: **50 samples**. Contributes some inherent
    per-epoch metric noise, but doesn't alone explain the swing — if it
    did, changing the learning rate wouldn't have nearly halved it.
  - With the checkpoint-reload bug fixed, the baseline's corrected
    best-epoch heatmap now visually matches the true field's zonal
    pattern well. The earlier "bad" heatmap shown to the team was
    largely a reporting-bug artifact, not proof the model wasn't
    learning.
- **50-epoch confirmation run at `lr=1e-4`** (`--out crn_outputs_lr1e4_50ep`):
  - Epochs 1–8: warmup (val MAE 0.26→0.13 pH), expected.
  - Epochs 9–50 (42 epochs, the real regime): val MAE range
    **0.083–0.318 pH (spread 0.235)** — narrower than the 15-epoch
    `lowlr` diagnostic's 0.491, i.e. it kept improving/stabilizing with
    more epochs rather than degrading. ~86% of these 42 epochs (36/42)
    landed in a genuinely good 0.08–0.17 pH band; the rest were
    mediocre bumps (0.20–0.32), none catastrophic (no repeat of the
    R²=−14 to −18 seen at `lr=1e-3`).
  - **Best checkpoint: epoch 13, val MAE=0.083 pH, val R²=0.834.** Part
    of a broad plateau of good epochs (9,10,12,13,15,18,21,24-26,28,29,
    32,35,36,38-41,43,44,47,49 all R²>0.6), not an isolated fluke.
  - Corrected heatmap (using the reloaded best checkpoint) visually
    matches the true field's zonal structure well — error concentrated
    at boundary/edge pixels, not spread uniformly.
  - **Not fully resolved**: a mild uptick in epochs 45–50 (0.151→0.258
    pH, ending on the run's second-worst epoch) — possibly early
    overfitting starting ~epoch 40-45, possibly noise; not enough data
    to tell yet. Also, the predicted map has a **grainy, per-pixel
    texture** within zones at full 256 resolution that the true field's
    smooth zones don't have — likely under-smoothing at this
    resolution/TV-weight combination, not investigated further this
    session.

**Decided:**
- pH field correlation length: 18px → 77px (~1.5cm). Applied and dataset
  regenerated (not left as a no-op) because this run actually changes
  generator output.
- Sparse-point loss requires a total-variation smoothness term (per Part
  B spec — 4 points alone underdetermine a 65,536-pixel map). Implemented
  at weight 0.05, not yet tuned.
- Root cause of val instability: **learning rate too high (1e-3)**, not
  overfitting (weight decay didn't help) and not purely a small-val-set
  metric artifact (the LR change had a real, reproducible stabilizing
  effect). No architecture change made this session.
- `lr=1e-4` confirmed at a 50-epoch horizon (not just the 15-epoch
  diagnostic) and **made the script's actual default** — no longer
  needs `--lr 1e-4` passed explicitly.
- Order-of-magnitude improvement over baseline (val MAE spread 0.235 vs.
  1.006), but explicitly **not** calling this fully resolved — see the
  late-run uptick and grainy-texture items below.

**Still open:**
- **Dataset size.** Currently 400 samples (`--n 400`) — the only number
  that has ever actually been run or documented in this repo. The
  **"~5000" figure that appeared in an earlier planning conversation was
  never a real decision** — it was carried over from a different Claude
  session's checklist text and never checked against the manuscript or
  any team agreement. Do not treat 5000 as a target until someone
  confirms it against the actual manuscript methodology (not in this
  repo) or a team decision.
- **CRN validation instability — significantly reduced, not fully
  resolved.** `lr=1e-4` is now confirmed at 50 epochs (spread 0.235 vs.
  baseline's 1.006) and is the script's default. Still open: (a) a mild
  uptick in epochs 45–50 of the 50-epoch run — early overfitting or
  noise, not yet distinguished; (b) grainy per-pixel texture in the
  predicted map at full 256 resolution, not present in the true field's
  smooth zones — possibly needs a stronger `--tv-weight` or a coarser
  `--out-res`, not investigated; (c) the val R² metric is still averaged
  per-batch rather than pooled globally across the whole val set, adding
  its own noise on top of whatever the model does — deliberately left
  unchanged through the diagnostic runs so every run's evaluation code
  was identical, still not fixed.
- 12 of 14 physical `PARAMS` in `generate_dataset.py` are still
  placeholders (AB's task) — the dataset and this CRN run are on
  placeholder physics, not yet citable.
- Dataset-size ablation (400/800/1600/3200) — explicitly **not started**,
  deferred as a follow-up per Part B instructions, do not add yet.
- Real-photo live-demo idea (separate from training data — one real
  photo supplies silhouette only, spectral/pH stays simulated) was
  scoped and agreed but not built. Was blocked on a trained CRN existing;
  that's now crudely true, so it's unblocked, but still needs the actual
  photo(s) from the team.

**Context to feed next session:**
- Dataset on disk right now uses the NEW pH scale (~1.5cm). If you see
  old figures/numbers referencing a "~0.35cm mottled" field, they're
  stale, from before this session's regeneration.
- `crn_outputs/`, `crn_outputs_lowlr/`, `crn_outputs_wd/` (all gitignored,
  not committed) are the 3 diagnostic runs from the LR investigation —
  each has a `history.json` with full per-epoch train/val MAE and R².
- `crn_outputs_lr1e4_50ep/` (loss curve, heatmap, history.json committed;
  checkpoint not) is the 50-epoch confirmation run — treat it as the
  current reference result, not the earlier 15-epoch runs.
- `lr=1e-4` is now the script's default (was `1e-3`). If you see old
  numbers/screenshots quoting wild val swings, they're from before this
  fix — don't assume that's still current behavior, but also don't
  assume the model is fully stable either (see the late-run uptick and
  grainy-texture items above).
- Don't re-introduce "5000 samples" as a target without a real source —
  see "Still open" above.

---

## 2026-09-03 — Bless (parameter / forward-model track, via Claude session)

**Changed:**
- Commit `40c6724`, pushed to `origin/main`. Touched `generate_dataset.py`,
  `extract_sensor_params.py`, `README.md`.
- Merged the team Parameterized Data Sheet into `generate_dataset.py` PARAMS:
  - **Table A (myoglobin eps)** now DECADIC millimolar extinction coefficients
    (mM^-1 cm^-1) from **Tang, Faustman & Hoagland 2004**, J. Food Sci.
    69(9):C717-C720, Table 2, p.C718, read from the primary PDF (not digitized
    off Piao et al. 2025). 525 nm = 7.60 for all three forms (isosbestic, one
    shared literature value); 573 nm linearly interpolated from Tang's 557/582
    rows (deoxy 9.96 / oxy 12.61 / met 3.56); **481 & 600 nm = PROVISIONAL
    rescaled placeholders**, flagged PARTIAL, NOT citations; 730/970 nm = 0.
  - `c_Mb_mean` 2.2 -> **0.87 mg/g**, `c_Mb_sd` 0.4 -> **0.12 mg/g**
    (Cross et al. 2018, Meat & Muscle Biology 2(1):189-196, n=599 LTL; SD
    back-calculated SE*sqrt(n)).
  - `denat_midpoint` 5.70 -> **CITED (proxy)** = Cross et al. 2018 population
    mean ultimate pH used as denaturation-onset midpoint.
  - `sensor_sigma` 0.015 -> **0.0054** (MEASURED).
  - `texture_amplitude` kept **0.030** (ASSUMED).
  - Added `mu_a_baseline` = 0.3 cm^-1 (TUNED, uncited).
- **Unit reconciliation in `mu_a()`** (was missing/wrong before):
  1. `MB_DECADIC_TO_NAPIERIAN = ln(10)` applied to the myoglobin term --
     Kubelka-Munk's K = 2*mu_a needs a Napierian coefficient; Tang's eps is
     decadic. Factor was absent before this session.
  2. `c_Mb` mg/g -> mM via `MUSCLE_DENSITY_G_PER_CM3 = 1.06` (BioNumbers BNID
     111214), `MB_MOLAR_MASS_G_PER_MOL = 17000`, `CM3_PER_L = 1000`.
  - **Bug caught & fixed:** first attempt computed 0.87*1.06/17000 = 5.4e-5 and
    used it as mM -- 1000x too small (that value is mol/L). `CM3_PER_L` bridge
    now applied.
- `check_params()`: PARTIAL now counts as blocking.
- Kept from `main` two fixes the working draft had reverted: `make_ph_field`
  scale = 77.0 (not 18.0), `self_test()` on `tempfile.mkdtemp` (not `/tmp`).
- `extract_sensor_params.py`: rewrote the texture block to measure FINE-scale
  structure -- normalized-convolution high-pass detrend at sigma in {15,25,40}
  px, median over bands, sensor-noise floor removed in quadrature -- instead of
  raw whole-region std. Docstrings + README updated with the synthetic-pivot
  framing and a parameter-status table.

**Verified (numbers):**
- `generate_dataset.py --selftest`: TEST 1 (sign) PASS -- reflectance falls
  with pH in all 6 bands. TEST 2: linear-baseline **R^2 = 0.58**, MAE = 0.166
  pH (was 0.39 / 0.20 before `sensor_sigma` dropped -- real sensor noise is
  ~3x below the old 0.015 assumption). Still < 0.9. `check_params()` = 8
  blocking params.
- Reflectance now physical: visible ~0.49-0.66, 573 nm dip ~0.50 (Q-band),
  730 nm plateau ~0.70, 970 nm ~0.54.
- `c_Mb` conversion: 0.87 mg/g -> 0.05425 mM (target 0.0542). mu_a(573) =
  1.62 cm^-1 (Q-band peak), mu_a(730) = 0.32 cm^-1 -- hand-recomputed from
  PARAMS, match code output.
- NHSI `mat_data/01.mat` (v7.3, 551x811x431, calibrated 0-1): `sensor_sigma`
  0.0054; fine-scale texture 0.42-0.47 (still macro-dominated -- NOT adopted);
  smooth 24x24 patch residual ~0.005 (~= sensor noise); **970 nm real mean
  0.19** (p5-p95 0.06-0.39) vs simulated ~0.54.

**Decided:**
- Pivot confirmed with adviser: synthetic dataset, objectives unchanged,
  "system" scoped as the software pipeline. Settled -- do not re-litigate.
- eps units = decadic mM^-1 cm^-1 (Tang convention); Napierian factor (x ln10)
  on the myoglobin term only (`mua_water` already Napierian).
- `sensor_sigma` = 0.0054 MEASURED, adopted (cube is calibrated reflectance,
  no rescaling caveat).
- `texture_amplitude` stays 0.030 ASSUMED -- NHSI cubes (whole aged muscle,
  NIR-heavy) are the wrong source for a trimmed-chop fine texture; anchored
  to the ~0.005 smooth-patch floor. Cosmetic nuisance term, lower bar.
- `c_Mb` conversion needs the cm^3->L (x1000) bridge -- now in code.

**Still open (blocks a defensible / reportable dataset):**
- **`eps` at 481 & 600 nm** -- invented. These are the manuscript's diagnostic
  bands (481 deoxyMb, 600 metMb). Need **Bowen 1949** (J Biol Chem
  179:235-245) full spectrum, rescaled onto Tang's scale at a shared
  wavelength (525 or 557 nm). **#1 blocker.**
- **`mua_water`** -- Hale & Querry 1973 values verified (970 nm = 0.45 exact,
  730 nm ~= 0.018) but NOT yet wired into code (still the placeholder). Also
  read the four visible-band values (all ~= 0) for completeness.
- **`water_fraction`** 0.75 -- confirm exact Honikel 1998 figure/page.
- **`denat_amplitude`** -- must be SWEPT (0.2/0.4/0.6/0.8), Ch. 4 sensitivity
  result, not pinned. **`denat_width`** 0.22 still unsourced.
- **970 nm external reality check FAILS** -- sim ~0.54 vs real NHSI ~0.19.
  Decision needed: (a) tune the NIR before generating (`scatter_a` has ~54%
  relative SD -- lower NIR-specific value or higher water/`mu_a_baseline`
  closes it), or (b) keep values and report as a stated Ch. 5 limitation on
  the one external validation. Explicit decision required.
- Optional insensitivity sweeps before locking: `mu_a_baseline` (0.2/0.3/0.4),
  `texture_amplitude` (0.02/0.03/0.05).
- Second person to eyeball Tang 2004 Table 2 (p. C718) vs the hand-read eps
  (7.60 @ 525; 9.96/12.61/3.56 @ 573).
- `c_Mb_sd`: confirm Cross et al.'s "+/- 0.005" is SE not SD (SE assumption
  gives CV ~14%).

**Context to feed next session:**
- The dataset on disk (`ligtas_synthetic_dataset/`, 400 samples) is STALE --
  predates every parameter change above. Do NOT train on it or quote any
  number from it. Regenerate only after the blockers close, then re-run
  `--selftest`.
- Do NOT quote a final linear-baseline R^2 or run reported CRN experiments
  yet -- R^2 moved 0.39 -> 0.58 on one parameter change and will move again
  with eps 481/600 + water.
- The Parameterized Data Sheet (Google Sheet) is the source of truth for
  parameter provenance; `generate_dataset.py` PARAMS mirrors it -- keep in
  sync. `check_params()` prints the live blocking list.
- Forward-model structure (KM, pH -> scattering only, myoglobin as an
  independent nuisance) is unchanged and sound. This session was parameter
  values + units only, no architecture change.
