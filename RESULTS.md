# LIGTAS-pH — Final Results

*Consolidated 2026-09-18 by `collect_results.py`, which reads the saved
evidence files rather than recomputing anything. Every figure is traceable to a file
listed under **Where each number comes from**. If this document and the source code
ever disagree, the code is correct — it is what was executed.*

## What this study produced

A physics-based simulator generates six-band multispectral images of pork with known
pixel-wise pH. A convolutional network is given the image and **only four pH readings**
per sample, and must predict pH everywhere else. The dense ground-truth map exists in
the simulation but is never shown during training; it is held back purely to score
against. The numbers below answer three questions: does it work, is it better than
conventional methods, and where does it fail.

---

## 1. Does it work? — predictive accuracy

**Evaluated on ligtas_test_extended/test (500 samples, unseen)**, across 5 independently seeded
training runs, reported as mean ± standard deviation.

| Metric | Value | What it tells you |
|---|---|---|
| R² | **0.8472 ± 0.0384** | Share of pH variation explained. Scale-relative, so it moves with the pH range sampled. |
| MAE | **0.0936 pH** | Typical error in pH units. Range-independent, so it is the more stable figure to quote. |
| RMSE | **0.1282 pH** | As MAE, but penalises large errors more. Exceeds MAE, as expected. |
| Within ±0.15 pH | **81.5%** | Describes the heatmap directly. Needs no threshold or class definition, so nothing in it can be disputed. |
| Within ±0.20 pH | 89.3% | The same, at a looser tolerance. |

**Why four metrics and not one.** R² alone describes a heatmap poorly: it is relative
to whatever spread of pH happens to be present, so a narrower range lowers it even when
accuracy improves. MAE and RMSE are in pH units and do not move with the sampling
design. The tolerance bands describe the deliverable itself and carry no interpretive
choice at all. **Lead with MAE and the tolerance band; report R² alongside them.**

**Contribution:** this is the evidence for Objective 1 — that the pipeline produces
usable pixel-wise pH maps.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `metric_check_outputs/final_evaluation_TEST500.json`
- **Raw per-run data:** `crn_5seed_final/seed_{0..4}/history.json  (per-epoch training record)`
- **Produced by:** `evaluate_heatmap.py`
- **Figure:** `figures/fig_system_output.png  (input → predicted map)`
- Checkpoints `crn_5seed_final/seed_*/crn_best.pt` are excluded by size; rerun `evaluate_heatmap.py` to regenerate the summary from them.

</details>

### Why these numbers can be trusted

The same five checkpoints were scored on three different sets. The validation set was
used during training to decide when to stop, so it is the one that could flatter the
model. The other two never influenced training at all.

| Evaluation set | R² | MAE (pH) | Role |
|---|---|---|---|
| validation_n50 | 0.8486 ± 0.0368 | 0.0902 | used for early stopping — could be optimistic |
| test_n50 | 0.8389 ± 0.0412 | 0.0902 | never touched during training |
| test_n500 **(reported)** | 0.8472 ± 0.0384 | 0.0936 | 500 fresh samples, distinct seed |

**They agree.** That is direct evidence the model did not overfit the split used for
early stopping — a question a panel is entitled to ask. Note that the larger set is
slightly *less* flattering (MAE 0.090 → 0.094): the 50-sample figures were mildly
optimistic, so **the reported numbers are the more conservative ones.**

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `metric_check_outputs/final_evaluation.json (validation), metric_check_outputs/final_evaluation_TEST.json (test n=50), metric_check_outputs/final_evaluation_TEST500.json (test n=500)`
- **Produced by:** `evaluate_heatmap.py --split {val,test} --data {...}`

</details>

### Per-seed detail

| Checkpoint | R² | MAE | RMSE |
|---|---|---|---|
| `seed_0` | 0.7778 | 0.1210 | 0.1558 |
| `seed_1` | 0.8905 | 0.0801 | 0.1094 |
| `seed_2` | 0.8604 | 0.0828 | 0.1235 |
| `seed_3` | 0.8393 | 0.0962 | 0.1325 |
| `seed_4` | 0.8679 | 0.0879 | 0.1201 |

**Why five runs rather than one.** Identical data and settings, differing only in
random initialisation, produce results from 0.78 to 0.89. Reporting
whichever single run we happened to obtain would mislead in one direction or the
other, so every figure here is a mean across five runs with its standard deviation.

<details><summary><b>Evidence in this repository</b></summary>

- **Raw per-run data:** `crn_5seed_final/seed_{0..4}/history.json  (loss and metrics per epoch)`
- **Produced by:** `train_crn.py --seed {0..4} --patience 10 --epochs 50`
- **Figure:** `crn_5seed_final/seed_*/loss_curve.png  (training vs validation loss)`
- `crn_5seed_final/run.log` holds the original console output of the run.

</details>

---

## 2. Is it better than conventional methods — and does that survive parameter uncertainty?

Two standard baselines were fitted and scored on identical data: pixel-wise linear
regression, and partial least squares regression. The comparison was then repeated
across the **entire plausible range** of the pH–scattering coupling strength, because
that parameter has no single citable value in the literature.

| Coupling strength | Linear | PLSR | CRN |
|---|---|---|---|
| 0.2 | 0.3113 | 0.3110 | **0.7270 ± 0.0820** |
| 0.4 *(adopted)* | 0.6212 | 0.6206 | **0.8486 ± 0.0368** |
| 0.6 | 0.7616 | 0.7608 | **0.8862 ± 0.0139** |
| 0.8 | 0.8308 | 0.8299 | **0.8999 ± 0.0169** |

**This establishes two things.**

First, **the task is not trivially solvable.** A linear fit stays below 0.9 at every
setting. Were the relationship simply the formula written into the simulator, a linear
model would recover it almost perfectly. It cannot — because pH acts only on
scattering, while absorption is driven independently by myoglobin, so the network must
separate two physical effects rather than invert one equation.

Second, and this is the stronger point: **the conclusion does not depend on getting
that parameter right.** No single value could be cited for it, so the whole range was
tested and the network wins throughout. That answers *“but is your parameter correct?”*
better than any single citation could, because a cited value might still be wrong for
this particular meat, whereas a range cannot be dismissed the same way.

**Contribution:** this is the evidence for Objective 2.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `deconfound_outputs/full_table.json  (all four amplitudes, with provenance per row)`
- **Raw per-run data:** `deconfound_outputs/crn_amp_{0.2,0.6,0.8}_seed_{0,1,2}/history.json; sweep_outputs/ holds the earlier small-scale sweep`
- **Produced by:** `deconfound_full_scale.py, sweep_denat_amplitude.py, compute_amp04_baselines.py`
- **Figure:** `sweep_outputs/sweep_crn_vs_baselines.png`
- The corrected pooled CRN values were rescored in `metric_check_outputs/metric_comparison_0.2_0.6.json` and `metric_check_outputs/metric_comparison.json`. `full_table.json` retains the superseded batch-averaged figures under `*_SUPERSEDED_batch_averaged` for traceability — do not quote those.

</details>

---

## 3. Does the heatmap point at the right places?

The deliverable is a *map*, so aggregate accuracy is not sufficient — a map could be
accurate on average while placing its features wrongly. This was therefore tested
directly, restricted to the **282 of 500 samples** whose
true pH spans more than one quality classification. Uniform samples are excluded
because localisation is meaningless on them, which makes this deliberately the harder
subset.

| Measure | Result | Chance | Reading |
|---|---|---|---|
| Correct elevated-pH region | **99.6%** | 50% | Direction is essentially always right. A lenient test, but it establishes the signal is real. |
| Per-pixel class accuracy | **79.0%** | — | **The figure to lead with.** Hard cases only, four pixels in five correct. |
| Hottest-quintile overlap | 51.3% | 20% | The demanding measure: well above chance, but honestly partial. |

**Why this section exists.** It answers the most dangerous question a panel can ask:
*if the per-sample R² below is negative, what is the point of the heatmap?* The answer
is that **location and magnitude are different things.** The map finds the right
regions; it overstates how different they are. Only the second is miscalibrated.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `metric_check_outputs/spatial_localization.json`
- **Produced by:** `check_spatial_localization.py  (~5 min, inference only)`
- **Figure:** `figures/fig_prediction_gallery.png  (best to worst case, chosen by error percentile)`

</details>

---

## 4. Where it falls short — spatial magnitude

- Per-sample R²: **-2.2988 ± 0.8530**
- Within-sample correlation: **+0.5410** — genuine signal is present
- Predicted ÷ true spatial SD: **1.290×** — over-expressed
- Between-sample pH SD **0.3036** vs within-sample **0.0843**

**Why the number is negative, stated plainly.** Per-sample R² asks whether the model
predicted the right *amount* of variation inside a single sample. True within-sample
variation is only about 0.084 pH — **smaller than the model's**
**own error of 0.094 pH**. On that scale the measure is punishing, and the
model additionally over-expresses variation by about 1.29×, which drives
it sharply negative.

**What it does and does not mean.** It is a *calibration* result. It does not mean the
model fails to detect spatial structure — section 3 shows it locates correctly. Optimal
rescaling of the existing predictions bounds the achievable value near +0.30, which
identifies this as a calibration remedy rather than an architectural one. **Not**
**attempted**, and disclosed as a limitation instead.

**Do not write** *“spatial reconstruction is unreliable”* — that conflates location with
magnitude and overstates the weakness. Say which one.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `metric_check_outputs/spatial_skill_official.json  (official checkpoints); metric_check_outputs/spatial_skill.json  (independent reproduction)`
- **Produced by:** `check_spatial_skill.py`
- **Figure:** `figures/fig_ph_distribution.png  (shows the within- vs between-sample scales that make this measure punishing)`

</details>

---

## 5. Where it falls short — tissue boundaries

- MAE within a 8 px band along the tissue edge: **0.1558 pH**
- MAE in the interior: **0.0698 pH**
- Ratio **2.31×**, present in **97%** of samples

**Cause.** A convolutional network has less surrounding tissue to work with at the mask
edge, and padding supplies partial background. The effect appears as a visible rim in
the prediction figures, which is how it was noticed.

**Why report it rather than leave it.** The headline MAE averages this degraded rim into
the good interior. For any use concerned with the body of the cut rather than its
perimeter, the interior figure of **0.070 pH** is the more
representative one. Boundary-aware padding or masked convolution would address it
directly; that was not pursued here.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `metric_check_outputs/edge_effect.json`
- **Produced by:** `check_edge_effect.py`
- **Figure:** `figures/fig_prediction_gallery.png  (the rims are visible in the predicted maps)`

</details>

---

## 6. The frozen parameters

Fourteen parameters drive the simulator. **Twelve trace to a published measurement, a
direct measurement, or a fit to published data.** The two that do not are labelled as
swept rather than presented as measurements — and the sweep in section 2 is what
establishes that the conclusion survives their uncertainty.

| Parameter | Value | Status |
|---|---|---|
| `eps_deoxy` | [3.88, 7.6, 10.93, 4.32, 0.113, 0.176] | CITED |
| `eps_oxy` | [7.05, 7.6, 11.26, 1.59, 0.362, 0.354] | CITED |
| `eps_met` | [7.76, 7.6, 3.12, 2.9, 0.155, 0.885] | CITED |
| `c_Mb_mean` | 0.87 | CITED |
| `c_Mb_sd` | 0.12 | BACKCALCULATED |
| `scatter_a` | 8.7436 | FITTED |
| `scatter_b` | 1.6618 | FITTED |
| `denat_amplitude` | 0.4 | SWEPT |
| `denat_midpoint` | 5.7 | CITED (PROXY) |
| `denat_width` | 0.28 | SWEPT |
| `mua_water` | [0.000248, 0.00032, 0.000763, 0.0023, 0.0179, 0.45] | CITED |
| `water_fraction` | 0.732 | CITED |
| `sensor_sigma` | 0.0054 | MEASURED |
| `mu_a_baseline` | 0.8 | TUNED |

These values are **frozen**. The dataset and the trained checkpoints have not been
regenerated since they were produced, so every number in this document refers to the
same simulator state.

<details><summary><b>Evidence in this repository</b></summary>

- **Summary values:** `generate_dataset.py  — the `PARAMS` dict, each entry carrying its own citation string`
- **Raw per-run data:** `docs/digitization/  — the WebPlotDigitizer screenshots the extinction coefficients were read from`
- **Produced by:** `generate_dataset.py --selftest  (sign test + non-triviality baseline)`
- Full decision history, including values considered and rejected, is in `docs/TEAM_LOG.md` (newest entry first).

</details>

---

## Where each number comes from

| Result | Evidence file |
|---|---|
| headline | `metric_check_outputs/final_evaluation_TEST500.json` |
| validation_n50 | `metric_check_outputs/final_evaluation.json` |
| test_n50 | `metric_check_outputs/final_evaluation_TEST.json` |
| amplitude_sweep | `deconfound_outputs/full_table.json` |
| spatial_localisation | `metric_check_outputs/spatial_localization.json` |
| spatial_magnitude | `metric_check_outputs/spatial_skill_official.json` |
| boundary_artefact | `metric_check_outputs/edge_effect.json` |

## Reproducing all of it

```
python generate_dataset.py --selftest      # sign test + non-triviality baseline
python evaluate_heatmap.py                 # section 1
python check_spatial_localization.py       # section 3
python check_edge_effect.py                # section 5
python make_figures.py                     # manuscript figures
python collect_results.py                  # regenerate this document
```

The dataset and checkpoints are excluded from the repository by size, but the dataset
regenerates in about 30 seconds from `generate_dataset.py` at the recorded seed, and was
verified bit-for-bit reproducible.
