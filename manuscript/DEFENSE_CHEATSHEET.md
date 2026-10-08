# LIGTAS-pH: Defense Reference

Every number in Chapters 3 to 5, what it means, where it comes from, and how to answer when asked about it. All values were verified against the evidence files on 2026-10-06.

---

## 1. The one-sentence claim

> We built a physics-based synthetic benchmark and a pipeline that recovers a dense pH map from six spectral bands, trained on only four probe readings per sample. We did not validate against real pork, and we do not claim to have.

Say this before any detail. Almost every hard question is a test of whether the claim is understood. Overstating it loses the room; stating it plainly and owning the limit dissolves most follow-ups.

---

## 2. The headline numbers

| Value | What it means | Evidence |
|---|---|---|
| **R² = 0.847 ± 0.038** | Share of pH variation the model explains, pooled over all pixels of 500 unseen samples | `RESULTS.json` |
| **MAE = 0.094 ± 0.015 pH** | Average size of the error at a pixel, in pH units | `RESULTS.json` |
| **RMSE = 0.128 ± 0.016 pH** | Same, but large errors count more | `RESULTS.json` |
| **81.5% within ±0.15 pH** | Share of pixels inside the tolerance band | `RESULTS.json` |
| **89.3% within ±0.20 pH** | The same at a looser band | `RESULTS.json` |

**What the ± is.** The spread across five independently seeded training runs. It is run-to-run training variability. It is **not** a confidence interval and **not** instrument precision, because no instrument was used.

**Why five seeds.** A single run is not evidence of typical performance. Henderson et al. (2018) show performance varies substantially between runs differing only in random seed. Chapter 2 cites this.

---

## 3. Per-seed results

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R² | 0.778 | 0.890 | 0.860 | 0.839 | 0.868 |

**Seed 0 is the worst and we report it.** It was a consistent low outlier on all three metrics, and the pattern reproduced on a second machine. Excluding it would raise the headline. We did not.

**Seed 3 runs the demonstration** at 0.8393, the closest of the five to the reported mean of 0.8472, and **below** it. Seed 1 scored 0.8905 and is deliberately not used. The selection is a rule in code, `pick_representative_seed()`, not a judgement.

**If asked why per-seed values differ on another machine:** they do not reproduce across hardware. That is why the five checkpoints are committed to the repository. Those files are the evidence for every per-seed number. Aggregate behaviour does reproduce.

---

## 4. The baseline comparison

On the same 500-sample set the headline is measured on:

| Method | R² | MAE (pH) | RMSE (pH) |
|---|---|---|---|
| Linear regression | 0.6504 | 0.1560 | 0.1951 |
| PLSR, 4 components | 0.6500 | 0.1561 | 0.1952 |
| **CRN** | **0.8472** | **0.0936** | **0.1282** |

**Margin +0.197 R². MAE 40% lower. RMSE 34% lower.** Evidence: `metric_check_outputs/baselines_test500.json`.

**Why PLSR and linear are nearly identical.** The six measurements vary largely together. An SVD of the design matrix shows a single component carries **81.0%** of its variance and the first four carry **97.31%**. A four-component model already spans almost the whole space a six-band linear fit can use. Evidence: `check_band_collinearity.py`.

**Robustness.** Table 4.7 repeats the comparison across the swept `denat_amplitude` values 0.2, 0.4, 0.6, 0.8. The CRN leads at every setting, by +0.416, +0.227, +0.125 and +0.069. That table is measured on the validation split, which is stated in the chapter, because the other amplitudes were never evaluated on the 500-sample set.

---

## 5. The hard one: two different R² values

| | Value | Reference mean used |
|---|---|---|
| Pooled R² | **0.847** | the mean of all pixels across all samples |
| Mean per-sample R² | **-2.17** | each sample's own mean |

**Both are correct. Same formula, different reference.**

**Why.** Between-sample pH variation has SD 0.317. Within-sample variation has SD 0.088. Between-sample spread is about **3.6 times larger**. The pooled measure is dominated by the model getting each sample's overall level right. The per-sample measure scores against the much smaller within-sample scale, where even correctly located errors look severe.

**Say it like this:** *The model is reliable about which sample is more acidic, and only partially reliable about the fine pattern inside one sample.*

**Do not say** "spatial reconstruction is unreliable." That conflates location with magnitude and overstates our own weakness.

---

## 6. Location versus magnitude

On the 282 of 500 samples that span more than one quality class:

| Measure | Result | Chance |
|---|---|---|
| Per-pixel class accuracy | 79.0% ± 2.9% | see below |
| Overlap of predicted and true hottest 20% | 51.3% ± 4.3% | 20% |
| Predicted hot region genuinely hotter | 99.6% ± 0.5% | 50% |

On the full 500-sample set, class accuracy is **87.7%** against **62.6%** for a classifier that always guesses the most common class, a lift of **25.1 points**. The 79.0% figure is lower because it is restricted to the harder multi-class samples.

**If asked whether your class boundaries are right.** They are midpoints between the anchors of the cited scale, which gives example values rather than limits. The literature reports no agreed criterion. The result was recomputed under four conventions:

| Boundaries | Accuracy | Majority baseline | Lift |
|---|---|---|---|
| adopted (Fig. 1 midpoints) (5.4 / 5.8) | 87.7% | 62.6% | **+25.1** |
| literature A (5.5 / 6.1) | 82.0% | 54.0% | **+28.1** |
| literature B (5.5 / 6.2) | 81.5% | 63.5% | **+18.0** |
| cited anchors as limits (5.2 / 6.0) | 90.5% | 56.7% | **+33.8** |

The lift stays between +18 and +34 points whichever convention is used, so the conclusion does not depend on where the lines are drawn. Note also that defining PSE as below 5.2 would leave only 0.4% of pixels in that class, because the generator clips at 5.2. Evidence: `metric_check_outputs/class_thresholds.json`.

**If asked how well it detects PSE.** Report it before they ask. Recall by class, five-seed mean: **PSE 29.0%, normal 82.1%, DFD 96.4%**, with PSE ranging 0.5% to 62.3% across seeds. The demo seed is at the low end. The cause is the compression in section 7: predictions are pulled toward the centre, and the lowest pH pixels cross the PSE boundary first. The 87.7% overall figure is carried by DFD, which is 62.6% of pixels. **But the calibration fixes it.** Applying the level correction, the one fitted from probe readings alone, raises PSE recall to **69.3%** and overall accuracy to **89.9%**, costing 0.6 points of DFD recall. The demonstration checkpoint goes from 5.4% to **68.7%**. Say the uncorrected and corrected figures in the same breath: the low number is a calibration effect, not a failure to detect PSE. Evidence: metric_check_outputs/class_recall.json

**Predicted maps correlate with ground truth at r = 0.536 ± 0.056.** Genuine spatial structure is recovered. The problem is magnitude calibration, not absence of signal.

---

## 7. Calibration, and why under-fitting is the safe problem

Two miscalibrations act in **opposite directions**:

- **Between samples:** predictions are **compressed**, spanning about 0.78 of the true range. Recovery slope 1.285 ± 0.130.
- **Within a sample:** variation is **over-expressed**, about 1.28 times the true spatial SD. Recovery scale 0.779 ± 0.056, fitted on the validation split.

A post-hoc correction, fitted on validation and applied to the held-out 500:

| Correction | R² | MAE | RMSE | Per-sample R² | % samples positive |
|---|---|---|---|---|---|
| None | 0.847 | 0.094 | 0.128 | -2.170 | 29.5% |
| **Level only (deployable)** | **0.904** | **0.075** | **0.102** | -0.946 | 39.7% |
| Texture only | 0.870 | 0.087 | 0.118 | -1.616 | 38.0% |
| Level and texture | 0.927 | 0.067 | 0.089 | -0.393 | 52.9% |

**The level half is deployable**, and this was tested rather than asserted. Fitting it from the mean of the four probe readings instead of the true dense mean gives R² 0.9044 against 0.9043, a difference of 0.0001. It accounts for **71.6%** of the gain. The texture half needs a dense ground-truth map and is **not** deployable; it is reported as a bound, not a result.

**If asked whether the model is overfitting:** it is not. It is **under-fitting**, and the evidence is the compression toward the middle of the range, which is regression to the mean. That follows from the design stated in Chapter 3: a deliberately small network, four supervised points, a strong smoothness prior.

**Why that is good news:** compression **costs** accuracy rather than flattering it. The uncorrected 0.847 **understates** what the predictions contain. Overfitting would mean the reported number is better than reality. We have the opposite.

**We did not retrain.** The correction reads the committed checkpoints only.

---

## 8. The pH range, and why R² moves

The generator draws each sample's mean pH uniformly across **5.35 to 6.45** and clips the field to **5.2 to 6.8**.

**The range itself is defensible, and this is worth knowing.** A published benchmark of pork loin ultimate pH reports a range of roughly 5.34 to 6.48, mean near 5.78, SD near 0.23. **Our sampled range of 5.35 to 6.45 corresponds closely to that observed range.** The range matches; the *shape* deliberately does not, for the reason below. Not yet in the manuscript — the primary source needs checking before it goes into Chapter 3.

**This is coverage sampling, not representative sampling.** Uniform draws exercise the model across the whole PSE-to-DFD span, which is how a measurement method is characterised across its operating range. Drawing from the real population distribution would concentrate data near the mean and leave both extremes barely represented.

**The consequence, disclosed:** 43% of samples have a mean pH above 6.0, the upper anchor of the reference scale cited in Chapter 2. The benchmark's class proportions do not match commercial pork.

**If asked what happens on a realistic range:**

| | Full range | Restricted to mean ≤ 6.0 |
|---|---|---|
| Samples | 500 | 284 |
| True SD of pH | 0.330 | 0.218 |
| **R²** | 0.847 | **0.705** |
| **MAE** | 0.094 | **0.085** |
| **RMSE** | 0.128 | **0.117** |

**R² falls but MAE and RMSE improve.** The model is slightly more accurate on the narrower range. R² drops only because narrowing removes variance from its denominator. This is why absolute error in pH units is reported alongside R² throughout. Evidence: `metric_check_outputs/range_restriction.json`.

**Class boundaries.** We classify at **5.4 and 5.8 pH**. The cited scale gives anchor values (5.2, 5.6, 6.0) rather than limits, so our boundaries are the midpoints between consecutive anchors. This is stated in Chapter 4 as our interpolation, not a published boundary.

---

## 9. The question most likely to end a defense

> *"You wrote the formula that made the data, then trained a model to recover it. Isn't that circular?"*

**This is a computer science question, not a physics one, which is why your panel is likely to ask it.**

**Answer:** pH enters the forward model **only through scattering**. Myoglobin redox state is generated **independently of pH** and enters **only through absorption**. To recover pH the model has to separate two physical effects using spectral shape across all six bands. It cannot invert a single relationship.

**Supporting evidence:** a pixel-wise linear fit on raw band values reaches only **0.6509 ± 0.0180** in-sample across ten draws, well under the 0.9 ceiling the study imposes on itself. If the mapping were trivial, a linear model would solve it.

**Watch the three linear-baseline numbers.** They answer different questions and are not interchangeable:

| Value | What it is |
|---|---|
| 0.6509 | in-sample self-test, 10 throwaway draws |
| 0.6504 | held out, 500-sample set (Table 4.6) |
| 0.6212 | held out, validation split (Table 4.7 sweep) |

---

## 10. "Is 0.094 pH accurate enough?"

**Start by conceding the gap, because it is real.** No published standard specifies a required pH accuracy for meat inspection. We looked. Do not invent one.

**Then give the three anchors that do exist.**

**(a) The quality boundaries disagree by more than our error — the strongest point.** Published PSE cutoffs range from ultimate pH < 5.7 to < 5.9, and DFD cutoffs from > 6.1 to > 6.3. The field disagrees about where the lines sit by roughly 0.2 pH. **Our error of 0.094 is less than half that disagreement**, so the instrument is more precise than the decision it informs. We already demonstrate this internally: section 6's four threshold conventions all preserve the conclusion.

**(b) A probe beats us at one point, and we say so.** Commercial meat pH meters specify ±0.01 to ±0.05 pH. We are less accurate than a probe at a single location, and more useful across a surface: 65,536 estimates without contact. **This is why Chapter 1 positions the system as screening, not replacement** — it tells an inspector where to put the probe.

**(c) Relative to the variation it must resolve.** Pork loin ultimate pH varies across the population with a standard deviation near 0.23. Our error is about **40% of one standard deviation** of the spread we would need to distinguish.

**The spoken answer:**

> There is no published accuracy standard for pH in meat inspection, and we do not claim to meet one. What we can say is what our error means against three documented references. The published PSE and DFD boundaries disagree between sources by about 0.2 pH, and our error is less than half that. A calibrated probe achieves 0.01 to 0.05 pH, so we are less accurate at one point but produce a full map without contact, which is why we position this as screening. And population ultimate pH varies with a standard deviation near 0.23, so our error is roughly 40% of the variation we would need to resolve. Establishing a formal acceptability threshold is listed in Chapter 5 as future work.

**Do not claim:** parity with a laboratory probe; that any standard has been met; that 0.094 is "good enough" as a settled fact. The claim is that it is *characterised* against three references, and that the threshold question is open.

**Citation strength, so you know what you are standing on:** the boundary ranges and the population spread come from meat-science and extension literature and are solid enough to state verbally. The probe figures come from manufacturer specifications — factual, but a datasheet, so cite it as commercial practice rather than as a study. **None of these three is in the manuscript yet.** If the team wants them in Chapter 5, the primary sources need checking first, to the same standard as Table 3.1.

**The better answer, if you can get it.** One sentence from the consulting food technologist — *"±0.1 pH is adequate for screening purposes"* — outranks all three anchors, because it is a judgement from someone qualified rather than an inference we assembled. Ask before the defense.

---

## 11. Training behaviour

- Initial sparse-point training loss **0.097**, falling to between **0.009 and 0.020** at the selected checkpoint.
- Early stopping selected epochs **9, 14, 21, 10, 18** across the five seeds, with patience 10. Every run terminated exactly ten epochs after its best, which is confirmable in `crn_5seed_final/run.log`.
- Validation-to-training loss ratio at the selected checkpoints: **1.15 to 1.89, mean 1.61**. Ratios in this range indicate the training partition was not overfitted at the point of selection.
- Worst divergence after the selected epoch: validation loss rose from **0.0210 to a peak of 0.4610**, **22 times** the checkpoint, settling to 0.1161 by the final epoch. Figure 4.1 plots that peak.

**Checkpoint selection used validation loss only.** Neither term of the loss reads the dense pH map, so the map the model is scored against played no part in which checkpoint was saved. Verifiable at `train_crn.py:178-180`.

---

## 12. Model and compute

| Property | Value |
|---|---|
| Trainable parameters | 118,113 |
| Checkpoint size on disk | 0.50 MB |
| Hardware | CPU only, no GPU anywhere in the study |

All four verified by direct measurement. The network was deliberately kept small to establish feasibility, and it is one instantiation of the pipeline rather than a fixed requirement.

---

## 13. Generator parameters

Fourteen parameters across eleven rows of Table 3.1. Ten are cited, measured or fitted. An eleventh, `c_Mb_sd`, is back-calculated. **Three have no citable source and are treated differently:**

- **`denat_amplitude`** , swept across 0.2 to 0.8, and **covered**: Chapter 4 reports performance at every setting, so the conclusion does not depend on the value adopted.
- **`denat_width`** , swept, but **not covered** by the baseline comparison. Disclosed as a limitation. Do not claim the sweep covers it.
- **`mu_a_baseline`** , **tuned, not cited.** Adopted at 0.8 because it improved the 970 nm match, which means the 970 nm check is **not independent of it**. Do not present that check as free-standing validation of this parameter.

**The equine substitution.** Extinction coefficients are horse myoglobin; pork-specific values are not published at these six wavelengths. The substitution follows the Krzywicki (1979) tradition still used today, including Piao et al. (2025). Species differences in myoglobin absorption are documented, so it is adopted as a disclosed assumption, not an equivalence. It appears in Chapter 2, as Assumption 1 in Chapter 3, and in the Chapter 5 limitations.

---

## 14. Limitations, in the order they are likely to come up

1. **No real-pork validation.** The principal limitation. Every number describes a synthetic benchmark. The only external comparison point is the 970 nm water absorption feature.
2. **Mean per-sample R² is negative** at -2.17. Explained in section 5 above.
3. **The pH distribution is not representative** of commercial pork. Section 8.
4. **Equine extinction coefficients** used for pork. Section 13.
5. **Per-seed results do not reproduce across machines.** Aggregate behaviour does.
6. **The five theoretical assumptions are declared, not tested.**
7. **Boundary artifact:** error within an 8-pixel band at the tissue edge is 0.156 pH against 0.070 in the interior, a ratio of 2.31, present in 97% of samples.

---

## 15. Things not to say

- **"Our prototype."** It is a pipeline. "Prototype" implies hardware integration, which does not exist.
- **"Spatial reconstruction is unreliable."** Location is reliable; magnitude is over-expressed. These are different.
- **"The model achieves 0.93."** That figure belongs to Yao et al. (2019), on real hyperspectral data, as a point measurement. Not comparable.
- **"Validated on pork."** Nothing was validated on pork.
- **Anything about lighting design, sensor synchronization or image capture.** No acquisition hardware was built or tested.
- **"R² of 0.705 means the model got worse."** It means the range got narrower. Absolute error improved.

---

## 16. How to check any number

| To verify | Read or run |
|---|---|
| Headline and per-seed metrics | `RESULTS.json` |
| Calibration table | `metric_check_outputs/calibration.json` |
| Baselines on the headline set | `metric_check_outputs/baselines_test500.json` |
| Range restriction | `metric_check_outputs/range_restriction.json` |
| Band collinearity | `python check_band_collinearity.py` |
| Amplitude sweep | `deconfound_outputs/full_table.json` |
| Training curves | `crn_5seed_final/seed_*/history.json` |
| Which sample and seed each figure used | `figures/figure_manifest.json` |
| Generator parameters | `generate_dataset.py`, PARAMS block |

**If a number in the paper disagrees with a number in a file, the file is right and the paper is wrong.**
