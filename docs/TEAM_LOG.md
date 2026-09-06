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

## 2026-09-07 (cont.) — `generate_dataset_new_plus_eps.py` retired; docs synced (via Claude session)

**Changed:**
- Deleted `generate_dataset_new_plus_eps.py` (`git rm`). Team decision:
  **"we are going to use `generate_dataset.py` from now"** — explicit,
  direct instruction. The candidate file had fully served its purpose
  (every value it proposed is now merged into the official file) and had
  drifted out of sync twice already, which is exactly the confusion
  multiple people on the team ran into (testing the wrong/old generator).
- Updated the three docs that still described it as a live, pending
  candidate: `README.md` (Current Status table + the scattering section
  rewritten to say RESOLVED, not DECISION pending), `CLAUDE.md` (file
  table, the 14-parameters status list restructured into Done/Still
  open/Resolved-since sections, the recent-history log, the "what to do
  next" checklist, and a stale R²≈0.59 baseline number corrected to
  ≈0.69), `docs/SYSTEM_ARCHITECTURE.md` (repo structure table, §3's
  non-triviality R² figure, §4's full parameter table, §7's run
  commands). All three previously said "not yet adopted, pending
  sign-off" for values that have been live in `generate_dataset.py` for
  two commits now.

**Verified:**
- Re-ran `--selftest` on `generate_dataset.py` after all doc edits and
  the file deletion, to confirm neither touched the actual generator:
  sign PASS, R² mean=0.6902 std=0.0168 -- unchanged from before this
  entry, as expected (docs/deletion don't touch physics).
- Grepped all three updated docs for remaining references to the
  retired file -- confirmed every remaining mention is correctly
  past-tense ("was retired," "existed until 09-07"), not a live
  "use this file" pointer.

**Decided:**
- `generate_dataset.py` is the one and only generator going forward.
  This is now stated plainly in the repo structure table of all three
  reference docs, not just implied.

**Still open:**
- Unaffected by this entry -- same 5 blocking parameters as the prior
  two entries today: eps @481/600nm (×3), `denat_amplitude`,
  `denat_width`. See CLAUDE.md's "What to actually do next" for the
  current numbered priority order.

**Context to feed next session:**
- If you're looking for `generate_dataset_new_plus_eps.py`, it's gone —
  don't recreate a parallel candidate file for future parameter
  experiments. Test changes in isolation on a throwaway copy (pattern
  used throughout today's three entries: copy the file, edit the copy,
  verify, apply to the real file, re-verify identical), then discard the
  copy once merged.
- `README.md`, `CLAUDE.md`, and `docs/SYSTEM_ARCHITECTURE.md` should now
  all agree with each other and with `generate_dataset.py`'s live
  `check_params()` output. If you find a discrepancy, the live
  `--selftest` output is the authority, not any of these docs.

---

## 2026-09-07 — `mu_a_baseline` re-swept fresh against current PARAMS (via Claude session)

**Changed:**
- Re-swept `mu_a_baseline` (0.3-0.8, 10-seed averaged, same methodology
  as every prior sweep in this log) directly against the CURRENT
  `generate_dataset.py` -- i.e. post scatter_a/b adoption and eps NIR
  adoption (see the two entries above). Not reused from any other file
  or an earlier config -- loaded and swept live via `importlib`, on-disk
  file untouched until the result was applied.
- Updated `mu_a_baseline`: `0.3` -> `0.8`.

**Verified (numbers):**
| `mu_a_baseline` | 970nm gap | linear R² (10-seed mean) | R² std |
|---|---|---|---|
| 0.3 (old) | 0.185 | **0.643 (lowest in range)** | 0.0185 |
| 0.4 | 0.162 | 0.650 | 0.0183 |
| 0.5 | 0.141 | 0.660 | 0.0180 |
| 0.6 | 0.123 | 0.671 | 0.0176 |
| 0.7 | 0.107 | 0.681 | 0.0172 |
| **0.8 (new)** | **0.093 (lowest in range)** | 0.690 | 0.0168 |

**Unlike the earlier `mu_a_baseline` sweep in this log (2026-09-06,
which found a genuine no-cost improvement on a since-superseded config),
this sweep found NO free lunch** -- 970nm gap and linear R² trade off
monotonically across the entire tested range, and the R² change (+0.047
from 0.3->0.8) is well above the ~0.017-0.018 seed-noise floor, i.e. a
real cost, not noise.

Re-ran `--selftest` on the actual (now-edited) file after applying: R²
mean=0.6902, std=0.0168 -- matches the sweep's 0.8 row exactly. n=20
generation + integrity check: 732,743 meat pixels, 0 NaN/negative/>1.0.
Sign test PASS.

**Decided:**
- Picked `0.8`: prioritizes the 970nm real-world match (the project's
  only external reality check) while keeping the linear-baseline R²
  (0.690) safely under the 0.9 "too easy" ceiling. This is a disclosed
  value judgment, not a discovery -- the full trade-off table above is
  the actual justification, not just the endpoint chosen. If the team
  would rather prioritize a lower R² over a closer 970nm match, 0.3-0.5
  are equally defensible alternatives on this same table.
- Anti-bias note, since this came up when applying it: this parameter is
  explicitly labeled `"TUNED -- NOT CITED, documented limitation"` in
  code, not disguised as a citation. The full swept range (not just the
  winning value) is preserved in this entry specifically so the choice
  is auditable -- see the parameter's own `source` field in
  `generate_dataset.py`, which points back here.

**Still open:**
- Same 5 blocking parameters as before this entry -- unaffected: eps
  @481/600nm (x3), `denat_amplitude` (sweep planned, not run),
  `denat_width` (derived estimate, not cited).
- 970nm gap improved further (0.185->0.093) but still not closed --
  same open decision as before: tune further, or document as a stated
  Ch.5 limitation. Getting closer makes this decision more pressing,
  not less.

**Context to feed next session:**
- `mu_a_baseline` is `0.8` as of this entry, not `0.3` -- if you see the
  older value quoted anywhere (chat history, earlier log entries, a
  teammate's notes), it's stale.
- Don't re-sweep this again assuming a free improvement exists like the
  2026-09-06 entry found -- confirmed here that config no longer applies;
  this is a real trade-off now, re-litigate only with a stated reason.

---

## 2026-09-06 (cont. 2) — Scattering "DECISION" resolved; eps NIR values adopted (via Claude session)

**Changed:**
- Resolved the `"DECISION"`-status blocker on `scatter_a`/`scatter_b` in
  `generate_dataset.py`: `18.9`/`1.286` (Jacques 2013, generic soft
  tissue) → `8.7436`/`1.6618` (same power-law formula, re-fit to
  approximate Bergmann et al. 2021's porcine-muscle-specific curve).
  This is exactly the item `CLAUDE.md` tracked as the single
  highest-leverage open item. Status → `FITTED`.
- Separately, adopted the `eps_deoxy`/`eps_oxy`/`eps_met` NIR (730/970nm)
  values: `0.0, 0.0` → `[0.21, 0.29]` / `[0.175, 0.35]` / `[0.09, 0.10]`
  respectively. NOT a citation change (still not sourced to anything) —
  a small tuned/empirical adjustment, tested and adopted separately from
  the scattering merge, not bundled in without justification.
- Deliberately did **not** merge the eps NIR values in the same step as
  scattering, even though both came from the same exploratory candidate
  file — scattering was the tracked team decision; eps NIR was not, and
  earlier isolation testing (2026-09-04/05) found it moves almost
  nothing on its own. Tested it separately, on its own merits, after
  scattering landed.
- Rebuilt `generate_dataset_new_plus_eps.py` to match — it had drifted
  out of sync with the team's `mua_water`/`denat_width` updates (was
  built before those landed). Now functionally identical to
  `generate_dataset.py` (confirmed via diff — only comment text differs).
  **This file has fully served its purpose and is now redundant** — see
  Still Open.

**Verified (numbers, each tested in isolation before being applied,
confirmed twice — once on a throwaway copy, once on the real file — with
identical results both times):**

*Step 1 — scattering only, before any eps NIR change:*
| | Before | After |
|---|---|---|
| Sign test | PASS | PASS |
| Linear baseline R² (10-seed mean) | 0.530 ± 0.020 | 0.647 ± 0.018 |
| 970nm real-world gap | 0.366 | 0.195 |
| Blocking parameter count | 7 | 5 |
| Data integrity (732,743 px) | — | 0 NaN/negative/>1.0 |

*Step 2 — eps NIR added on top of the now-merged scattering:*
| | Without eps NIR | With eps NIR (adopted) |
|---|---|---|
| Linear baseline R² (10-seed mean) | 0.6471 ± 0.0179 | 0.6433 ± 0.0185 (within noise, not a real change) |
| 970nm real-world gap | 0.1950 | **0.1853** (real, deterministic improvement) |
| Blocking count | 5 | 5 (unchanged — not a citation either way) |
| Sign test / integrity | PASS / clean | PASS / clean |

**Final combined state, generate_dataset.py, right now:** sign PASS,
linear baseline R²=0.6433±0.0185, 970nm gap=0.1853, blocking count=5,
data integrity clean.

**Decided:**
- Adopted `scatter_a`/`scatter_b` refit — resolves the tracked
  `"DECISION"` item. Explicit, disclosed trade-off: R² rose from 0.530
  to 0.647 (a real cost, not hidden), in exchange for closing ~47% of
  the gap on the project's one real-world check and dropping 2 blocking
  parameters. Judged net-better, not a free win — see the objective
  verdict given to the team alongside this entry.
- Adopted the eps NIR values on top — small, real, no measurable cost.
  Documented in-code as NOT a citation (see the updated comment block
  above `eps_deoxy` in `generate_dataset.py`) with 0.0 (the AMSA/
  Krzywicki convention) named as the explicit fallback if this stops
  being defensible.
- Full parameter table (all 14 values + status) was shared with the team
  for review before this commit, per explicit request.

**Still open:**
- **`generate_dataset_new_plus_eps.py` is now redundant** (functionally
  identical to `generate_dataset.py`, confirmed by diff) — proposed for
  retirement, matching how `generate_dataset_changed.py` was retired
  once superseded. Not yet deleted — flagging for the team, same as the
  earlier retirement was flagged before acting.
- The 5 remaining blocking parameters are unchanged by this entry: eps
  @481/600nm (×3, the real #1 blocker), `denat_amplitude` (sweep planned,
  not run), `denat_width` (derived estimate, not a citation).
- 970nm gap improved (0.366→0.185) but is not closed — still an open
  decision: tune further, or document as a stated Ch.5 limitation.

**Context to feed next session:**
- `generate_dataset.py` now has NO remaining scattering/eps-NIR decisions
  pending — both were resolved and merged this session, not just
  proposed.
- If `generate_dataset_new_plus_eps.py` is retired, update any docs still
  pointing to it (`SYSTEM_ARCHITECTURE.md`, `CLAUDE.md`, `README.md` all
  reference it as "the candidate, not yet adopted" — that framing is now
  stale and needs a pass regardless of whether the file itself is
  deleted).
- Re-run `check_params()` before quoting the blocking count — it's 5 as
  of this entry, but has changed three times in two days.

---

## 2026-09-06 (cont.) — Independent 5-seed replication + a correction to both sessions above (via Claude session)

**⚠️ CORRECTION to this same day's earlier "5-seed" entry above and its
own uncertainty about dataset differences -- read this before trusting
that entry's "not guaranteed bit-identical" caveat.**

**Changed:**
- Made `--patience` and `--seed` PERMANENT flags in the shared
  `train_crn.py` (not a scratch script) -- `--patience` enables early
  stopping (checkpoint on lowest val_loss = val_sparse + tv_weight*val_tv,
  computed from the 4 sparse val points only, never the hidden field;
  stops after N epochs with no improvement); `--seed` calls
  `torch.manual_seed()` for reproducible init/shuffling. Unset (the
  default for both) reproduces the exact prior behavior -- nothing
  changed for existing callers.
- Ran the same bounded experiment independently: 5 seeded runs (0-4),
  lr=1e-4, patience=10, cap 50 epochs, against `ligtas_synthetic_dataset_v2/`
  (generated earlier the same day, same generator script state as the
  entry above's `ligtas_dataset_new_plus_eps/`).
- Committed the per-seed results (`crn_5seed_s0`.."s4"/ -- history.json,
  loss_curve.png, sanity_check_heatmap.png; checkpoints excluded per the
  existing blanket `*.pt` rule) alongside this entry, matching the
  existing-entry's precedent.

**Verified (numbers):**
- Per-seed: seed0 R²=0.7055/MAE=0.1091, seed1 R²=0.8434/MAE=0.0802,
  seed2 R²=0.7045/MAE=0.1124, seed3 R²=0.7932/MAE=0.0881, seed4
  R²=0.8922/MAE=0.0560.
- Mean +/- std: **R²=0.7878 +/- 0.0745, MAE=0.0892 +/- 0.0206** -- vs.
  the entry above's R²=0.8070 +/- 0.0763, MAE=0.0840 +/- 0.0191. Means
  agree within one std of each other; pattern shape does NOT replicate
  cleanly -- their run shows 4 seeds tightly clustered (0.82-0.87) +1
  clear outlier (0.66); ours shows 2 seeds low together (~0.70-0.71) and
  the other 3 more spread out (0.79/0.84/0.89), not a single outlier.
- **Dataset-identity check, resolving the entry-above's stated
  uncertainty:** the generator's `main()` uses a HARDCODED seed (42),
  not a random one. Regenerated a 4-sample dataset fresh and confirmed
  `sample_001_msi.npy` and `sample_001_phtrue.npy` are byte-identical
  (`np.array_equal`) to what's already in `ligtas_synthetic_dataset_v2/`
  -- confirming dataset generation is fully deterministic given the same
  script version. Since the entry-above's dataset was generated from the
  same committed script state (their own log confirms this via
  `check_params()` output), **their `ligtas_dataset_new_plus_eps/` and
  our `ligtas_synthetic_dataset_v2/` were almost certainly the SAME
  data, not different RNG draws as previously assumed.**

**Decided:**
- The "different dataset" explanation for the per-seed pattern mismatch
  is very likely WRONG -- correcting that assumption in both this entry
  and (by reference) the entry above's stated uncertainty. The two
  experiments were run on effectively identical data.
- Given that, the mismatch is better explained by either (a) genuine
  sensitivity of this still-unstable model to small seed differences, or
  (b) environment-level non-determinism (different machine/PyTorch
  build) that `torch.manual_seed()` does not fully eliminate across
  environments -- NOT distinguished from each other, not investigated
  further here (bounded experiment, not iterating).
- `--patience`/`--seed` are now permanent, reusable `train_crn.py`
  features -- this resolves the entry-above's open item asking whether
  early stopping should become a committed CLI feature. It should, and
  now does.

**Still open:**
- Whether the per-seed mismatch is model sensitivity or cross-environment
  non-determinism is unresolved -- would need same-machine, same-process
  reruns to distinguish, not attempted here.
- The AGGREGATE conclusion (CRN with early-stopping-selected checkpoint
  clearly beats the ~0.68 linear baseline) is now supported by two
  independent 5-seed samples landing within noise of each other -- treat
  this as reasonably solid. The exact per-seed reproducibility is not.
- All the usual parameter-citation blockers and the deeper instability
  root-cause (untested ideas: more epochs, GroupNorm, the val-R²
  per-batch-averaging issue) remain open, unaffected by this entry.

**Context to feed next session:**
- `train_crn.py --patience N --seed N` is now the standard way to run
  this kind of experiment -- don't reconstruct a scratch script, the
  capability is permanent and committed.
- Do not re-litigate "were the two datasets different" -- verified same,
  see above. If re-deriving this dataset for any reason, expect it to be
  byte-identical to both `ligtas_synthetic_dataset_v2/` and
  `ligtas_dataset_new_plus_eps/` as long as `generate_dataset_new_plus_eps.py`
  hasn't changed.
- Two independent 5-seed results now exist (`crn_5seed_new_plus_eps/`
  from the entry above, `crn_5seed_s0`.."s4"/` here) -- both legitimate,
  neither supersedes the other; read both before assuming either is "the"
  result.

---

## 2026-09-06 — 5-seed CRN early-stopping experiment on generate_dataset_new_plus_eps.py (via Claude session)

**Relation to the entry directly below (the "(night)" CRN retrain):** this
is a separate session, run against the current (post-4f879dd) state of
`generate_dataset_new_plus_eps.py` -- confirmed via `check_params()`
output at generation time (water_fraction CITED, eps citation chain
present, matching the 5 parameter updates already logged). It uses its
OWN freshly generated dataset (`ligtas_dataset_new_plus_eps/`, local
only, not the entry below's `ligtas_synthetic_dataset_v2/`) and a
different evaluation design (5 seeds + early stopping, vs. one 50-epoch
run's last-10-epoch stats). Read both; don't conflate the two datasets
or treat one as superseding the other.

**Changed:**
- Generated a fresh 400-sample dataset (300/50/50 split) from
  `generate_dataset_new_plus_eps.py` at its current committed state:
  `python generate_dataset_new_plus_eps.py --n 400 --out ligtas_dataset_new_plus_eps`.
  Local only, not committed (regenerable from the command above).
- Ran ONE bounded experiment, per explicit instruction ("no further
  tuning after this"): 5 independent CRN training runs, identical
  architecture and hyperparameters (lr=1e-4, tv_weight=0.05, batch=8,
  in_res=out_res=256, n_points=4), varying only the seed (0-4) that
  controls model weight init and training-data shuffle order -- the
  dataset itself is held fixed across all 5 runs.
- Added early stopping and per-seed checkpointing via a standalone
  script (NOT committed -- ran from a local scratch path, reuses
  `train_crn.py`'s existing `SparseLIGTASDataset` / `run_epoch` /
  `CrudeCRN` building blocks rather than modifying the shared file):
  patience = 10 epochs with no improvement, hard cap 50 epochs per run.
- **Explicit design choice, not yet reconciled with `train_crn.py`'s own
  convention:** "validation loss" for checkpoint selection and early
  stopping = val_sparse_MSE + tv_weight*val_TV (the actual optimized
  quantity, computed from the 4 sparse val points only) -- NOT the
  hidden-field val_MAE that `train_crn.py`'s own checkpointing currently
  uses. Chosen so model selection never touches `phtrue.npy`, consistent
  with the project's "hidden field is eval-only" rule. Flagged for the
  team to confirm or override, not silently adopted as the new default.
- Outputs (best checkpoint, loss curve, sanity-check heatmap, per-epoch
  history, `summary.json`) saved per seed under
  `crn_5seed_new_plus_eps/seed_{0-4}/` in the repo -- untracked so far.

**Verified (numbers):**
- All 5 runs completed; ~85s/epoch on CPU (no GPU in this environment),
  113.5 min total wall time.
- Confirmed by reading `full_field_eval()` in `train_crn.py`: reported
  val R²/MAE are computed against the FULL HIDDEN DENSE pH field (all
  meat pixels), not just the 4 sparse points. Checkpoint selection/early
  stopping used val_loss (sparse+TV on the 4 sparse val points only,
  see above) -- the two are deliberately different signals.
- Per-seed early-stopped results:
  | seed | best epoch | stopped at | val R² | val MAE (pH) |
  |---|---|---|---|---|
  | 0 | 11 | 21 | 0.6596 | 0.1199 |
  | 1 | 14 | 24 | 0.8663 | 0.0706 |
  | 2 | 16 | 26 | 0.8280 | 0.0765 |
  | 3 | 10 | 20 | 0.8162 | 0.0860 |
  | 4 | 10 | 20 | 0.8650 | 0.0668 |
- Mean +/- std across the 5 selected checkpoints: **val R² = 0.8070 +/-
  0.0763** (min 0.6596, max 0.8663); **val MAE = 0.0840 +/- 0.0191 pH**
  (min 0.0668, max 0.1199).
- Seed 0 is a clear low outlier (R²=0.66) versus the other four
  (clustering 0.82-0.87) -- pulls the mean down and inflates the std;
  flag the spread, don't just quote the mean.
- Every seed's per-epoch log shows the same qualitative pattern: sharp
  swings to strongly negative val R² (e.g. seed 3 epoch 8 R²=-2.45,
  seed 4 epoch 19 R²=-2.79) interleaved with good epochs, before
  stabilizing enough to trigger early stopping -- consistent with, and
  now shown across 5 seeds rather than 1, the instability the
  2026-08-31 and the entry-below sessions already documented at
  lr=1e-4.
- **Cross-reference to the entry below's open item #1** ("re-run
  lr=1e-4 with a different seed to check if instability was an unlucky
  draw"): this experiment does exactly that across 5 seeds, though on a
  separately-generated dataset and using early-stopping-by-val_loss
  rather than watching a fixed run's last-10 epochs -- not a strict
  controlled replication, so treat as suggestive, not conclusive. Result:
  with early stopping at the point of lowest val_loss, 4 of 5 seeds land
  at R²=0.82-0.87, clearly beating the ~0.68-0.69 linear baseline the
  entry below reports; only seed 0 lands closer to that baseline range.
  This suggests picking the right EPOCH (via early stopping) may matter
  as much as the instability itself -- worth testing directly against
  their exact `ligtas_synthetic_dataset_v2/` before concluding anything
  stronger.

**Decided:**
- Scoped and run as one bounded experiment; results reported as-is, no
  further tuning attempted per instruction.
- No decision made here on adopting `generate_dataset_new_plus_eps.py`
  into the official `generate_dataset.py` -- that sign-off is still the
  2026-09-04 entry's open item, unaffected by this session.

**Still open:**
- The val_loss-vs-hidden-field-val_MAE checkpoint-selection discrepancy
  between this experiment's script and `train_crn.py`'s own convention
  (see Changed above) is unreconciled -- needs a team decision on which
  convention `train_crn.py` itself should use, and whether early
  stopping should become a permanent, committed CLI feature there
  instead of living in an uncommitted scratch script.
- The underlying epoch-to-epoch instability is NOT diagnosed further by
  this session -- it adds evidence (5 seeds instead of 1) but doesn't
  test any of the entry-below's untried ideas (more epochs, GroupNorm,
  the val R² per-batch-averaging issue).
- The 5-seed script itself is not committed -- if this capability
  (seeded reruns + early stopping) is wanted again, it should be written
  into `train_crn.py` or a new tracked script, with review, rather than
  re-run from scratch each time.
- Did not test this exact methodology (5 seeds, early-stopping-by-
  val_loss) against the entry-below's `ligtas_synthetic_dataset_v2/` --
  the two datasets should be close (same generator, same recent
  parameter state) but were generated separately and are not guaranteed
  bit-identical (different RNG draws). A same-dataset comparison would
  be needed to properly settle whether early stopping resolves, or just
  masks, the instability question raised below.
- All the usual parameter-citation blockers (481/600nm eps, mua_water
  wiring, denat_width, denat_amplitude sweep, 970nm gap) are unaffected
  by this session and still open.

**Context to feed next session:**
- Dataset: `ligtas_dataset_new_plus_eps/` (400 samples, 300/50/50,
  generation seed 42 default), built from `generate_dataset_new_plus_eps.py`
  at its current (post-4f879dd) state -- regenerate if PARAMS change
  again, don't assume it matches the entry-below's `_v2` dataset exactly.
  Reported result: **val R² mean=0.8070, std=0.0763; val MAE
  mean=0.0840, std=0.0191 pH**, across seeds 0-4 -- quote the mean+std
  and the seed-0 outlier together, not a single number.
- To reproduce: same `CrudeCRN` (base=16, in_res=out_res=256), same
  hyperparameters (lr=1e-4, tv_weight=0.05, batch=8, n_points=4), seeds
  0-4, patience=10 on val_loss (sparse+TV, not val_MAE), cap 50 epochs.
- If picking up the entry-below's instability investigation, consider
  running this same 5-seed/early-stopping design directly on their
  `ligtas_synthetic_dataset_v2/` (regenerate via their logged command)
  for a true apples-to-apples comparison before drawing conclusions from
  either session's numbers alone.

---

## 2026-09-06 (night) — CRN retrain on updated dataset + instability follow-up (via Claude session)

**⚠️ HANDOFF NOTE: session ended here for the night, not finished. If
nobody has re-tested this by the time you read it, the "still open"
section below is exactly where to pick up.**

**Changed:**
- Regenerated a fresh 400-sample dataset (`ligtas_synthetic_dataset_v2/`,
  local only, gitignored) from the CURRENT state of
  `generate_dataset_new_plus_eps.py` -- i.e. reflecting this session's 5
  parameter citation updates (see the earlier 2026-09-06 entry above/below)
  plus everything from the 2026-09-05 `texture_amplitude` cut.
- Retrained the CRN from scratch (`train_crn.py`, NO architecture or
  hyperparameter changes) on this fresh dataset. Then ran exactly ONE
  stabilization attempt (learning rate halved, 1e-4 -> 5e-5, otherwise
  identical) after validation instability showed up, per explicit
  instruction to try one bounded fix and stop -- not iterate.
- Neither retrain's checkpoint/output folders
  (`crn_outputs_v2_retrain/`, `crn_outputs_v2_lr5e5/`) nor the fresh
  dataset are committed -- all local only, easy to regenerate from the
  commands below if needed.

**Verified (numbers):**
- Linear baseline on the REAL materialized fresh dataset (not just
  self-test's throwaway sampling): **R²=0.679** (300 train samples,
  120,000 sampled pixels) -- matches the candidate file's self-test mean
  of 0.691 within normal seed-to-seed noise. Confirms the dataset
  generated correctly and consistently with expectations.
- **CRN retrain #1 (lr=1e-4, unchanged default), best epoch (30):
  val R²=0.767, val MAE=0.087 pH.** Beats the linear baseline by a real
  margin at its best.
- **But last-10-epoch (41-50) stats tell a different story:**
  val R² mean=0.680 std=0.053 (min 0.573, max 0.758); val MAE mean=0.111
  std=0.016 (min 0.091, max 0.141). Mean val R² (0.680) is essentially
  TIED with the linear baseline (0.679/0.691) -- the CRN's advantage
  over the dumb baseline exists only at its best epoch, not typically.
  About 18 of 50 epochs had a val MAE bump above 0.15 (some to
  0.19-0.27) -- more volatile than the equivalent 50-epoch run on the
  PREVIOUS dataset (before `texture_amplitude` cut + eps/water_fraction
  updates), which had ~86% of epochs in a good 0.08-0.17 band.
- **Stabilization attempt (lr=5e-5) FAILED -- made it WORSE, not
  better:** last-10-epoch val R² mean dropped to 0.605 (std TRIPLED to
  0.165); val MAE mean rose to 0.131 (std DOUBLED to 0.034). Best single
  epoch stayed about the same (R²=0.766 vs 0.767) -- so halving LR didn't
  help the peak and hurt the typical/consistency numbers. Likely
  explanation, NOT confirmed: a smaller LR may need more than 50 epochs
  to converge and this run just hadn't gotten there yet -- pure
  speculation, not tested.
- Per explicit instruction, did NOT attempt a second stabilization fix.
  This instability is being reported honestly, as-is, not resolved.

**Decided:**
- Report the instability plainly in whatever writeup uses these numbers:
  the CRN's best-epoch result is real and worth reporting, but the
  TYPICAL result on this dataset does not yet clearly beat a plain
  linear fit -- that distinction matters and should not be papered over
  by only quoting the best epoch.
- One cheap, bounded fix (halved LR) was tried and explicitly did not
  work -- do not re-try the same fix expecting a different result.

**Still open (pick up here):**
- **The core problem is unresolved.** Root cause of the increased
  volatility on THIS dataset (vs. the previous one) is not identified --
  candidates not yet tested: (a) `texture_amplitude`'s removal took away
  a noise source that may have been incidentally regularizing training,
  (b) the combined scattering/eps/water_fraction shifts changed the
  input distribution enough to matter, (c) something dataset-generation-
  specific to `generate_dataset_new_plus_eps.py` vs. the plain official
  file. Not distinguished from each other yet.
- **Untried ideas for next session, in rough order of cost:**
  1. Re-run the ORIGINAL lr=1e-4 config once more with a different
     random seed to check if THIS specific run's instability was itself
     just an unlucky draw (cheapest check, doesn't require code changes).
  2. Try MORE epochs (e.g. 80-100) at lr=5e-5 to test the "just needed
     longer to converge" theory above -- currently pure speculation.
  3. Revisit the BatchNorm-vs-GroupNorm question shelved back on
     2026-08-31 -- that was set aside because the LR fix resolved
     instability THEN, on the old dataset. It may be worth reopening now
     that a plain LR change isn't fixing it on the new dataset.
  4. Check whether the val R² metric's known per-batch-averaging issue
     (still unfixed, flagged repeatedly since 2026-08-31) is exaggerating
     the apparent volatility -- pool residuals across the whole val set
     once, globally, before concluding the model itself is unstable.
- Exact commands to reproduce, if picking this up fresh:
  ```
  python generate_dataset_new_plus_eps.py --n 400 --out ligtas_synthetic_dataset_v2
  python train_crn.py --data ligtas_synthetic_dataset_v2 --epochs 50 --batch-size 8 --out crn_outputs_v2_retrain
  ```

**Context to feed next session:**
- This is a genuine, not-yet-understood regression in training stability
  relative to the previous dataset -- do not assume it's already fixed,
  and do not assume the earlier LR fix (2026-08-31/09-01) still applies
  unmodified to this dataset; it was tested here and made things worse.
- If someone re-tests this before you read it, check git log / this file
  for a newer entry before repeating the work above.
- All artifacts from tonight (`ligtas_synthetic_dataset_v2/`,
  `crn_outputs_v2_retrain/`, `crn_outputs_v2_lr5e5/`) are LOCAL ONLY,
  not pushed -- regenerate with the commands above rather than looking
  for them in the repo.

---

## 2026-09-06 — Five parameter citation updates + tracker fix 

**Changed:**
- Applied 5 confirmed edits identically to BOTH `generate_dataset.py` (official)
  and `generate_dataset_new_plus_eps.py` (candidate):
  1. `water_fraction`: `0.75` -> **0.732**, `PLACEHOLDER` -> `CITED`
     (Wojtasik-Kalinowska et al. 2016, LWT-Food Sci Technol 67:112-117 --
     mean across 4 dietary treatment groups, n=24 pigs, pork *Longissimus
     dorsi*; groups differed in supplementation, not expected to affect
     baseline water content).
  2. `eps_deoxy`/`eps_oxy`/`eps_met`: **values unchanged**. Status
     `PARTIAL` -> `CITED -- adjacent species (horse), field-standard
     practice`. Citation chain added: Piao et al. (2025) -> Piao et al.
     (2022) -> Tang, Faustman & Hoagland (2004) -> underlying coefficients
     originate from HORSE myoglobin, not pork/beef -- documented as
     standard practice across the meat-color literature (Krzywicki
     1979/1982 and downstream), not a thesis-specific shortcut.
  3. `c_Mb_sd`: value unchanged (0.12). Status -> `BACKCALCULATED --
     justified`. Reasoning documented: a raw SD of 0.005 mg/g across 599
     genetically distinct pigs is biologically implausible (near-zero
     variance for a real trait), so the reported "+/-" is almost
     certainly SE, not SD -- back-calculated SD = SE*sqrt(599) ~= 0.12.
  4. `MB_MOLAR_MASS_G_PER_MOL` (module constant, 17000): value unchanged.
     Comment updated -- now doubly-sourced: standard biochemistry
     reference value, AND independently used by Cross et al. (2018) in
     their own pork myoglobin extraction methodology.
  5. `denat_midpoint`: value unchanged (5.70). Added caveat: the Cross et
     al. (2018) source cohort was explicitly non-PSE by the paper's own
     description ("none displayed the pale, soft, and exudative
     condition") -- reasonable sigmoid-center proxy, but the underlying
     data has no low-pH/high-denaturation tail examples.
- **Found and fixed a tracker bug caused by item 2 above.** Changing
  `eps_deoxy`/`eps_oxy`/`eps_met`'s status to a `CITED` variant silently
  dropped them from `check_params()`'s blocking count, even though 481nm
  and 600nm inside those arrays are STILL unverified provisional
  placeholders -- only the citation *framing* (horse-myoglobin sourcing)
  was resolved, not those two specific wavelength values. A single
  `status` string can't represent "framing settled, 2 of 6 values still
  pending". Fixed by adding a new `pending_wavelengths` field (`[481.0,
  600.0]`) to those three PARAMS entries, and rewriting `check_params()`
  in BOTH files so a parameter with pending wavelengths is flagged as
  blocking (marked `PART`, printed as e.g. `CITED -- ... -- 481/600nm
  STILL PENDING`) regardless of what its own `status` string says.

**Verified (numbers):**
- Blocking parameter count is now **6** (was 8 before any of today's
  edits; briefly, incorrectly, 3 immediately after item 2 above before
  the tracker fix; correctly 6 after the fix) -- `denat_amplitude`,
  `denat_width`, `mua_water`, and `eps_deoxy`/`eps_oxy`/`eps_met` (each
  flagged specifically for their 481/600nm gap, not as a whole).
- Both files re-ran clean after every edit: sign test **PASS** in both.
- `generate_dataset.py`: R² (10-seed mean) = **0.587**; 970nm real-world
  gap = **0.365**.
- `generate_dataset_new_plus_eps.py`: R² (10-seed mean) = **0.691**;
  970nm real-world gap = **0.184**.
- Both changed only marginally from their pre-`water_fraction`-update
  numbers (as expected -- `water_fraction` only multiplies `mua_water`,
  which is near-zero outside 730/970nm, so a 0.75->0.732 change is a
  small nudge, not a structural shift).
- Data integrity: n=20 generated with each, 732,743 meat pixels each,
  **0 NaN / 0 negative / 0 values >1.0** in both.

**Decided:**
- All 5 parameter edits above are adopted in both files, not just
  proposed -- verified working end-to-end after each one.
- The `pending_wavelengths` tracking mechanism is the standing pattern
  going forward for any parameter where citation/framing gets resolved
  before every sub-value does (not unique to eps -- reusable if another
  array-valued parameter hits the same situation).

**Still open:**
- The real remaining blockers, unchanged in substance by this session:
  481nm/600nm digitized myoglobin values (needs Bowen 1949 or equivalent),
  `denat_width` (no citation, confirmed to have a real, sizable effect on
  R² per the 2026-09-05 entry), `mua_water` (values already found per
  earlier entries, still not wired into code), and the `denat_amplitude`
  sweep (plan already agreed, not yet executed/written up).
- The 970nm real-world match, even in the better candidate file, is
  still ~2x off from the real measured value (0.184 gap on a ~0.19
  target) -- "improved" is not "close." Not further investigated this
  session.
- Neither file is presentable/final -- 6 genuine gaps remain, per the
  count above.

**Context to feed next session:**
- Both `generate_dataset.py` and `generate_dataset_new_plus_eps.py` now
  have IDENTICAL treatment of these 5 parameters -- they only differ in
  the scattering model, eps NIR values, `mu_a_baseline`, and (candidate
  only) the `texture_amplitude` cut inherited from the 2026-09-05 entry.
- If you add a citation that resolves framing but not every sub-value
  for some OTHER array parameter, use the same `pending_wavelengths`
  pattern (see `check_params()` docstring in either file) rather than
  overloading `status` -- that's exactly the bug this session fixed.
- Run `check_params()` (or `--selftest`, which calls it) to see the live
  count -- don't assume 6, 7, or 8 without re-running it, since this
  number has changed twice in two days.

---

## 2026-09-05 — Cut `texture_amplitude` (via Claude session) — CORRECTED, see below

**⚠️ Correction within this same entry:** the "Verified" numbers below were
originally computed with a single fixed seed (the same style of number the
2026-09-04 entry's seed-averaging fix, further down, found unreliable by
0.05-0.08). Re-ran with the honest 10-seed method before this entry was
finalized — **the original "barely moves R²" conclusion was wrong.** Left
both the corrected numbers and this note in place rather than rewriting
history quietly.

**Changed:**
- `generate_dataset.py`: removed the `texture_amplitude` parameter from
  `PARAMS` and the multiplicative texture term from `generate_sample()`.
  Cube composition is now `R * illum` (was `R * texture * illum`). Left a
  comment in `PARAMS` pointing here for the decision record.
- No other parameters touched. `mu_a_baseline` and `denat_width` were both
  considered as cut candidates and explicitly rejected — see below.

**Verified (numbers, corrected using 10-seed averaging):**
- Ran a joint `denat_amplitude` (0.2/0.4/0.6/0.8) x `denat_width`
  (0.15/0.22/0.30) grid using the OLD single-seed methodology (seed 0),
  via a throwaway script, not committed. `denat_width` moved R² by
  **0.07-0.10** at every amplitude level, monotonically (narrower width ->
  higher R²). Directionally consistent across all 12 grid points, which
  argues it's a real effect and not pure seed noise, but this predates the
  seed-averaging fix below and has NOT been re-confirmed with it — treat
  as provisional, not final. **Not folded away; still an open parameter.**
- **Original (WRONG) claim:** cutting `texture_amplitude` moved single-seed
  R² by 0.5839 -> 0.5660 (-0.018), read as "barely moves."
- **Corrected, 10-seed-averaged re-test:** WITH `texture_amplitude=0.030`,
  mean R² = **0.5199** (std 0.0294). WITHOUT (the cut as committed), mean
  R² = **0.5896** (std 0.0198, matches the shared `self_test()`'s own
  10-seed output after the merge below). **True difference: +0.0697** —
  about 3x the per-seed noise floor (~0.025), and essentially the same
  size as the scattering-model swap's +0.069 that the 2026-09-04 entry
  flagged as a concerning, decision-worthy change. This is NOT a
  negligible cosmetic removal.
- Mechanism (why, in hindsight): `texture_amplitude` added multiplicative
  pixel noise uncorrelated with pH. That noise was quietly doing some of
  the work of keeping the linear baseline's job hard, similar to
  `sensor_sigma`'s role. Removing it made the task measurably easier for
  a linear model — the opposite of what "cut the uncited cosmetic term"
  was supposed to achieve for free.

**Decided:**
- **Cut `texture_amplitude` anyway, decision made with the corrected
  numbers in hand** (not on the original wrong premise): it is still
  ASSUMED/uncited and doesn't encode the pH mechanism, and the resulting
  mean R² (0.59) is still well under the 0.9 "too easy" alarm. But this is
  now a disclosed trade-off, not a clean win — record it as: fewer uncited
  parameters, at the cost of a real ~0.07 increase in how easy the linear
  baseline finds the task. Whoever owns the parameter track should know
  this before citing "R²=0.59" without this context.
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
- The `denat_amplitude` x `denat_width` grid above needs re-running with
  10-seed averaging (like the texture_amplitude number was) before it's
  trustworthy — it used the same old single-seed style this whole
  correction is about.
- `denat_width` still needs either a literature source or an explicit
  Ch.4 write-up as a second sensitivity axis (not yet written up either
  way).
- Parameterized Data Sheet (Google Sheet) not yet updated to mark
  `texture_amplitude` as CUT, and its R² cell needs the corrected 0.59
  mean (10-seed), not a single-draw number.
- All other open items from prior entries (eps 481/600nm, `mua_water`
  wiring, `water_fraction` citation, `denat_amplitude` sweep writeup,
  failing 970nm check) are unaffected by this session and still open.

**Context to feed next session:**
- `texture_amplitude` no longer exists in `generate_dataset.py`. If
  reconsidering restoring it, use the corrected number above (+0.07 R²,
  not negligible), not the original wrong one.
- Any single-seed R² number from before today should be treated as
  suspect per the 2026-09-04 entry's fix, below — this entry is a live
  example of that caution being justified.
- Do not quote a final linear-baseline R² yet — still moving with the
  usual open blockers (eps 481/600, `mua_water`, `water_fraction`,
  `denat_width`).

---

## 2026-09-04 — Scattering-model comparison

**⚠️ If you're wondering why `generate_dataset.py` shows as modified and
you didn't touch it:** that's expected, see the `self_test()` fix below.
Only that one function changed — nothing else in the file, no parameter
values touched. It's a deliberate correctness fix (same category as the
earlier `/tmp` bug fix), not an accident or unrelated edit.

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
- **Follow-up investigation (same day), isolating and improving on the above:**
  - Built 4 hybrid variants to isolate the changed file's two bundled
    edits: `generate_dataset_eps_only.py`, `generate_dataset_scatter_only.py`
    (both local, untracked).
  - Built `generate_dataset_valuesonly.py` / `generate_dataset_new.py`:
    KEEPS the original 1-term power-law formula unchanged, only refits
    `scatter_a`→8.7436, `scatter_b`→1.6618 (least-squares fit to
    Bergmann et al. 2021's curve, 450-1000nm) — no new formula, no new
    parameters.
  - Built `generate_dataset_new_plus_eps.py`: the above + the eps
    changes from `generate_dataset_changed.py` (myoglobin coefficients
    at 481/600/730/970nm).
  - **Fixed `self_test()` in BOTH `generate_dataset.py` (the shared main
    file) and `generate_dataset_new_plus_eps.py`**: Test 2 now averages
    10 random seeds (mean/std/min/max) instead of one hardcoded seed.
    New helper `_linear_baseline_once(seed)` extracted for reuse.
  - **Retuned `mu_a_baseline` in `generate_dataset_new_plus_eps.py`:
    0.3 → 0.6**, via a full 9-point sweep (0.1-1.0) with 10-seed-averaged
    R² at every point (not the 2-point single-seed check that gave a
    false lead earlier the same day).

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
- **Isolation result: the formula change, not the eps change, drives
  nearly everything.** eps-only barely moved either metric (970nm gap
  0.363→0.354, R² +0.012). Scatter-only (formula) did almost all the
  work (970nm gap →0.202, R² +0.064) — matching "both together" almost
  exactly. The two edits are not two independent trade-off levers; it's
  one change producing both effects.
- **The formula switch is NOT necessary to get its benefit.**
  `generate_dataset_new.py` (old formula, refit values only) matched
  `generate_dataset_changed.py`'s 970nm gap exactly (0.192 vs 0.192) and
  its R² is statistically indistinguishable (0.618 vs 0.604, well within
  the ~0.035 per-seed noise band). The extra 2 parameters and new
  equation in the "changed" version buy nothing beyond what refitting
  the existing formula already achieves.
- **μs' curve fidelity, checked at all 6 bands (not just 970nm):** the
  refit power law tracks Bergmann's real 2-term curve within ±2.7% at
  481/525/573/600/730nm; worst case is 970nm at −5.66% (edge of the
  450-1000nm fit range, as expected). Table in chat log.
- **Adding eps on top of the refit formula ("new + eps") beats
  everything tested, including the original "changed" file:** 970nm gap
  **0.182** (best of all variants), R² 0.615 (statistically tied with
  every other improved variant). Achieved with zero added complexity —
  same 2-knob formula, eps is just an existing array's values.
- **Seed-averaging finding (independent of the scattering decision):**
  the single hardcoded seed in the old `self_test()` sat ~0.05-0.08
  ABOVE the true 10-seed-averaged R² for every configuration tested
  (original: single-seed 0.58 vs true mean 0.527; changed: 0.65 vs
  0.604). Every R² this project has quoted to date was likely inflated
  by the specific seed choice, not just by which parameters were used.
  This is now fixed (see Changed above) — future `--selftest` runs
  report an honest mean ± spread.
- **`mu_a_baseline` retune (full sweep, honest 10-seed R² at each
  point):** R² traces a real, smooth minimum across 0.4-0.7 (confirmed
  by the shape across 9 points, not a 2-point fluke like the earlier
  same-day false lead). **0.6 sits in that minimum (R²=0.605, tied with
  the sweep's best of 0.603) AND improves the 970nm gap substantially
  vs. the prior default of 0.3 (0.182→0.121, ~34% better)** — a rare
  case with no trade-off at all.
- **Final candidate, fully re-verified after the retune:**
  `generate_dataset_new_plus_eps.py` with `mu_a_baseline=0.6`. Sign test
  PASS; R² mean=0.6051, std=0.0363 (10 seeds, matches the sweep
  prediction exactly); 970nm gap=0.121; n=20 generation clean (0 NaN /
  negative / >1.0 across 737,188 pixels).

**Decided:**
- **Recommendation, still pending the parameter owner's actual sign-off:
  adopt `generate_dataset_new_plus_eps.py`'s values into
  `generate_dataset.py`** — refit `scatter_a`/`scatter_b` (keep the
  original formula), the eps updates, and `mu_a_baseline=0.6` — NOT
  `generate_dataset_changed.py`'s 4-knob formula. Reasoning: it matches
  or beats the complex version on every metric tested (real-world match,
  R², data integrity) while adding zero structural complexity and zero
  new parameters to eventually cite. This is stronger than the earlier
  "lean toward" — it's now a dominant choice across every test run, not
  a trade-off — but still needs the parameter track's explicit approval
  before it goes into the real `generate_dataset.py`.
- `self_test()`'s multi-seed averaging fix is a straightforward
  correctness fix (like the earlier `/tmp` fix), applied directly to
  the shared `generate_dataset.py` without a separate sign-off round.

**CLEAR STATEMENT — which file to use, right now:**
Use `generate_dataset_new_plus_eps.py` (with `mu_a_baseline=0.6`,
already set in that file) for anything going forward. Not
`generate_dataset.py` (still has the stale, un-adopted scattering
values) and not `generate_dataset_changed.py` (the complex 4-knob
version — no longer worth using, since the simpler file matches or
beats it on every metric tested). This is a **tested recommendation,
not yet the official file** — nothing has been merged into the real
`generate_dataset.py` yet.

**What still needs to happen, in order:**
1. Parameter/physics track approves adopting `generate_dataset_new_plus_eps.py`'s
   values (this is the only step still needing a human decision, not
   more testing).
2. Once approved, copy those values into `generate_dataset.py` itself
   (or promote this file to be the new `generate_dataset.py`).
3. Regenerate the real dataset and retrain the CRN on it — everything
   generated so far, including all CRN results to date, was built on
   the old values and is not final.

**Still open:**
- **The actual adoption decision** — this comparison recommends
  `generate_dataset_new_plus_eps.py`'s values, but nothing has been
  merged into the real `generate_dataset.py` PARAMS yet. Needs the
  parameter track's confirmation.
- The realism improvement is confirmed at **970nm only** (now backed by
  a full 6-band μs' fidelity check on the formula itself, but the
  eps/absorption side of the realism claim still only has one real
  reference point).
- Not attempted: sweeping `texture_amplitude` similarly (was on the
  original follow-up list; superseded in priority by the scattering
  work above).
- All the usual open items from the parameter-sourcing track (eps at
  481/600nm needing Bowen 1949 properly, `mua_water` wiring,
  `water_fraction` confirmation, `denat_amplitude`/`denat_width`) are
  unaffected by this comparison and still open — see the 2026-09-03
  entry below.

**Context to feed next session:**
- `generate_dataset_new_plus_eps.py` is the fully-tested final
  candidate — local, untracked, not pushed. `mu_a_baseline=0.6` (not
  0.3) in this file specifically; the shared `generate_dataset.py`
  still has 0.3 and the old scattering values until the adoption
  decision above is made.
- `generate_dataset.py`'s `self_test()` IS already fixed to average 10
  seeds — that part went in directly, independent of the scattering
  decision. If you see a single-number R² quoted anywhere for this
  project going forward, treat it as suspect unless it says "mean over
  N seeds."
- `generate_dataset_changed.py`, `_eps_only.py`, `_scatter_only.py`,
  `_valuesonly.py`, `_new.py`, and all `test_*/` folders are throwaway
  local comparison artifacts — only `_new_plus_eps.py` matters going
  forward.

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
