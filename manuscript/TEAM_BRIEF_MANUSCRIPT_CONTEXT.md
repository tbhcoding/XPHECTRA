# LIGTAS-pH: Internal Review Notes

**For the project team and anyone reviewing Chapters 3 to 5.**

These notes exist so that a reviewer can disagree from an informed position.
Chapters 3 to 5 are not settled. Several numbers in earlier drafts turned out to
be wrong, and they were caught by checking the draft against the files rather
than by rereading the draft. This document records what the current numbers are,
where each one can be verified, which decisions were deliberate, and which
questions are still open.

Disagreement is wanted. What is not useful is disagreement based on a
half-remembered number. Section 9 lists how to check any claim here against the
repository.

---

## 1. What the study is, in five sentences

The multispectral camera intended to supply training data failed before
collection began. With the adviser's approval the study was rebuilt around a
physics-based synthetic benchmark: a generator produces six-band images of
simulated pork tissue from published optical constants, and a small
convolutional network is trained to recover a dense per-pixel pH map from those
six bands. The network is supervised on only four pH readings per sample,
reflecting what a real probe could realistically provide, and the dense map is
withheld during training and used only for scoring. No real pork was imaged at
any point. The contribution is a working software pipeline and a benchmark whose
every parameter has a declared evidentiary status, not a validated instrument.

---

## 2. Working rules

Agreed by the team. These apply to anything written into the chapters.

1. **Nothing enters the chapters that cannot be traced.** Every value must come
   from a file in this repository or a cited source. Where a value is missing or
   a procedure was not carried out, the gap is stated as a gap. It is not
   estimated, approximated, or filled from a comparable study.
2. **Reporting restraint.** If a claim is in the chapters, it has to be
   defensible under questioning. If it cannot be defended, it does not go in,
   even if it is true.
3. **Minimum sufficient detail.** See section 7 for why this matters more than
   it sounds.
4. **House style.** No em dashes. Avoid padding phrases that carry no
   information: "this study aims to", "it is important to note",
   "furthermore", "moreover". Prefer the plain verb.

---

## 3. The current numbers

Every figure below was read out of a file in this repository. The source column
is where to verify it.

### Headline result

| Metric | Value | Source |
|---|---|---|
| Pooled R-squared | 0.847 +/- 0.038 | `RESULTS.json`, `headline.r2` |
| Mean absolute error | 0.094 +/- 0.015 pH | `RESULTS.json`, `headline.mae_ph` |
| Root mean square error | 0.128 +/- 0.016 pH | `RESULTS.json`, `headline.rmse_ph` |
| Pixels within +/-0.15 pH | 81.51 +/- 6.75 % | `RESULTS.json` |
| Pixels within +/-0.20 pH | 89.25 +/- 4.05 % | `RESULTS.json` |

Measured on 500 unseen samples (`ligtas_test_extended/test`), across 5 training
seeds. The +/- is the standard deviation across those 5 seeds. It is run-to-run
training variability, not a confidence interval and not measurement error.

### Per-seed R-squared

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R-squared | 0.778 | 0.890 | 0.860 | 0.839 | 0.868 |

Seed 0 is a consistent low outlier on all three metrics, reproduced on a second
machine. It is reported, not excluded.

### Network against the classical baselines

Across the `denat_amplitude` sweep, from `figures/figure_manifest.json`:

| Amplitude | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|
| Linear regression | 0.311 | 0.621 | 0.762 | 0.831 |
| PLSR | 0.311 | 0.621 | 0.761 | 0.830 |
| Network | 0.727 | 0.849 | 0.886 | 0.900 |

The network leads at every setting, and the margin is largest where the physical
effect is weakest. Linear and PLSR agree to within 0.001 because the six bands
vary largely together: a single component carries **81.0%** of the design
matrix variance and the first four carry **97.31%** (`check_band_collinearity.py`,
output in `metric_check_outputs/band_collinearity.json`). Do not cite the
condition number of 8.0 as evidence of collinearity; a value that low argues
against it.

### Model size

118,113 trainable parameters, 0.45 MB on disk, roughly 31 ms per 256x256 sample
on CPU. No GPU was used anywhere in the study.

---

## 4. Decisions already made, and why

Reopen any of these, but start from the reasoning rather than from scratch.

**The reported number is the 5-seed mean, not the best seed.** Seed 1 reached
0.890. Reporting it would be seed cherry-picking. The mean with its spread is
the honest summary.

**The demonstration runs seed 3, not seed 1.** Seed 3 scored 0.8393, the closest
of the five to the reported 0.8472 mean, and it is below that mean. This is
deliberate: the demonstration cannot show more than the thesis claims. Selection
is by rule in `pick_representative_seed()`, not by eye.

**The demonstration shows only the prediction, with no chooser.** Earlier
versions showed a true, predicted and error triptych. That view needs the hidden
ground-truth map, which a real deployment would never have, and it invites the
question "if you already know the true pH, what is the model for?" The triptych
belongs in Chapter 4 as a diagnostic, not in the demonstration. A seed chooser
would likewise imply the seed is a user setting; it is an artifact of training.

**Figure 4.2 shows the median sample, not the best.** `sample_383`, ranked 26th
of 50 by per-sample R-squared, with a per-sample R-squared of **-0.428**. Chosen
by that rule.

**"Pipeline", never "prototype".** On the consultant's advice: "prototype"
implies hardware integration, which this study does not have. The word
"prototype" appears zero times in the chapters.

**"CRN" or "convolutional regression network", never "Crude".** The class in
code is still `CrudeCRN`; the manuscript does not use that name.

---

## 5. Open questions

Ranked by how much a good answer would improve the thesis.

1. **Is the per-sample R-squared framing the best one available?** This is the
   hardest number in the study to present (see section 6, item 1). A framing
   that is clearer without being softer would be the single most valuable
   improvement anyone could contribute.
2. **Is the +/-0.15 pH tolerance the right threshold?** It came from the domain
   consultant. +/-0.20 is also reported. If a published grading threshold for
   pork pH exists, citing it would convert a judgement call into a sourced one.
3. **`denat_width` has not been swept in the baseline comparison.** It is swept
   in the generator, but Chapter 4's comparison covers `denat_amplitude` only.
   This is disclosed as a limitation. Running it would close the gap. A
   baselines-only probe takes minutes; the full network sweep takes 5 to 7 hours.
4. **`mu_a_baseline` is tuned and uncited.** Adopted at 0.8 because it improved
   agreement with the external 970 nm reference. The override is disclosed. A
   citation would be better than a disclosure.
5. **The eight references marked [VERIFY] in Chapter 2: closed.** Completed
   in `9338d2d` from the team's own source files. Neither chapter file contains
   a `[VERIFY]` marker any more. Previously listed here as the last open item;
   it is not.
6. **The domain-consultation record is not in the repository.** Chapter 3 lists
   input validation by domain consultation as a validation step and says it is
   documented separately. The document exists on paper but nothing in the
   repository points to it, and the +/-0.15 pH tolerance band that produces the
   81.5% figure rests on it. It should be photographed or scanned and checked in
   the way `docs/digitization/nhsi_tray_species_annotation_cube01.png` was, with
   Chapter 3 naming that file. Until then the claim cannot be followed up by
   anyone reading the repository alone.
7. **Chapter 3 citations need a final check.** Seven sources carry DOIs. Three
   were checked against live sources (Thennadil 2008; Rudin, Osher and Fatemi
   1992; Wold 2001). The remaining four are on arXiv or in an open-access
   journal and are definitely free, but their page numbers have not been
   confirmed against the published versions: Ronneberger 2015, Kingma and Ba
   2015, Henderson 2018, Hodson 2022.

---

## 6. Known weaknesses, most dangerous first

These are disclosed on purpose. Removing a disclosure is not a fix.

1. **Mean per-sample R-squared is -2.17 on the 500-sample set, and negative for
   the large majority of individual samples.** This is real and it is in the
   chapters. It coexists with the pooled 0.847 because between-sample pH
   variation (SD 0.304) is about 3.6 times larger than within-sample variation
   (SD 0.084). The pooled measure is dominated by the model getting each
   sample's overall level right; the per-sample measure scores against the much
   smaller within-sample scale, where even correctly located errors look severe.
   Plain reading: **the model is reliable about which samples are more acidic,
   and only partially reliable about the fine pattern inside one sample.**
2. **No real-pork validation.** The principal limitation. Every number describes
   behaviour on a synthetic benchmark. The only external comparison point is the
   970 nm water absorption feature.
3. **Equine myoglobin extinction coefficients are used for pork.** Pork-specific
   values are not published at these wavelengths. The substitution is standard
   in tissue optics, declared as Assumption 1, and listed in Chapter 5 as
   untested.
4. **Per-seed results are not reproducible across machines.** Aggregate
   behaviour replicates, including seed 0's weakness; individual seed values do
   not. This is why the mean and spread are reported rather than a single run.
5. **The five theoretical assumptions are declared, not tested.**
6. **A boundary artifact exists at mask edges**, from convolutional context loss
   where the tissue mask ends.

---

## 7. Detail deliberately left out of Chapter 3

Anyone revising Chapter 3 may be tempted to restore these. Read the reason
first.

The panel is computer science. They have little physics or biology background.
The guiding principle, from the consultant: detail that answers no question the
panel will ask still gives them something to pull on. Each row below was removed
because it costs more in exposure than it adds in completeness.

| Left out | Reason |
|---|---|
| Both Kubelka-Munk against Beer-Lambert comparisons | Raising an alternative invites "why not Monte Carlo? Why not full radiative transfer?" The question does not arise unless we raise it. |
| The K and S to mu_a, mu_s' conversion factors | Unit bookkeeping. Invites "where does the 0.75 come from?" |
| The ln(10) decadic-to-Napierian conversion | Invites "what is Napierian?" and answers nothing. |
| Oxy, deoxy and met myoglobin fraction glossary | Redox chemistry. "Published extinction coefficients for myoglobin" is sufficient for the argument being made. |
| Logistic function, denaturation midpoint, transition width | Invites "why a logistic curve?" The mechanism sentence was kept, since that is the part a reader needs. |
| Physiological bias initialisation | Implementation trivia with no bearing on any result. |
| Hardware requirements table | Processor, memory, storage, graphics card. Answers nothing. The computational characteristics table was kept, because it is the evidence for Objective 3. |

**The one passage that must not be cut** is the two-pathway separation: pH enters
the forward model through scattering only, and myoglobin redox state is generated
independently and enters through absorption only. That paragraph is the whole
answer to the objection most likely to land, which is a computer science
objection rather than a physics one: "You wrote the formula that made the data,
then trained a model to recover it. Isn't that circular?"

---

## 8. Running the demonstration

The repository is complete: `crn_demo_ui.py`, `crn_model.py`, `RESULTS.json`
and the five trained checkpoints are all committed. The dataset is not. It is
940 MB and gitignored, so it has to be regenerated locally the first time.
This was tested end to end from an empty directory.

    pip install -r requirements.txt
    python generate_dataset.py --n 500 --seed 777 --all-test --out ligtas_test_extended
    python crn_demo_ui.py

The second step takes about 40 seconds and writes roughly 980 MB. The third
opens a local page at http://127.0.0.1:7860.

**Use those exact flags.** `--n 500 --seed 777 --all-test` regenerates the
set bit-for-bit, and it is the set the headline R-squared is measured on. A
different seed produces a valid dataset that is not the one in the thesis, and
the R-squared shown on screen would then describe a different set than the
samples displayed.

**opencv must install successfully.** `generate_dataset.py` imports it inside a
try/except, so generation appears to succeed without it, but it then skips
writing the `*_rgb.png` preview files and the demonstration fails at startup
with a FileNotFoundError. If opencv will not install, fix that before anything
else.

**If pip pulls a CUDA build of torch**, force the CPU wheel:
`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`.
No GPU was used anywhere in the study.

**Expect the figures on screen to match the thesis exactly.** The demonstration
loads the committed seed-3 weights rather than retraining, so it shows 0.8393
against a reported mean of 0.8472. Anyone who retrains will get different
per-seed values, because per-seed results do not reproduce across machines,
and the same aggregate behaviour.

---

## 9. How to verify any claim

| To check | Run or read |
|---|---|
| Headline metrics | `RESULTS.json` |
| Per-seed metrics | `RESULTS.json`, `per_seed` |
| Which sample and seed each figure used | `figures/figure_manifest.json` |
| Band collinearity | `python check_band_collinearity.py` |
| Baseline sweep table | `deconfound_outputs/full_table.json` |
| Loss curves | `crn_5seed_final/seed_3/history.json` |
| Generator parameters and their provenance | `generate_dataset.py` PARAMS block, and Table 3.1 |
| Figures, regenerated from scratch | `python make_figures.py` |
| The dataset itself | not committed, about 1.7 GB; regenerates bit-for-bit via `python generate_dataset.py` |

If a number in the chapters disagrees with a number in a file, **the file wins
and the chapter is wrong.** Errors already caught this way: a stale 90.2% that
should have been 89.3%; a validation-to-training loss ratio reported as 1.15
that was actually a range of 1.15 to 1.89 with a mean of 1.61; and an SVD figure
of 97.46% that was written into a draft before any script had computed it. The
real value is 97.31%. A second pattern, found by independent audit on
2026-10-06: a correct number wrapped in a claim about what it means that was
never checked against the code. Four cases, all now corrected. Check the claim,
not just the digits.

---

## 10. Four things that get confused

**The two R-squared values.** Pooled R-squared is 0.847. Mean per-sample
R-squared is -2.17. Same formula, different reference mean. Both are correct.
Quoting one as though it refuted the other is the most common mistake made about
this project, including by us in earlier drafts.

**The seed spread against measurement error.** The +/- 0.038 is the spread across
5 training runs. It says nothing about instrument precision, which was never
measured because no instrument was used.

**The three evaluation sets.** Validation n=50 (R-squared 0.849, the only set
that influenced training), test n=50 (0.839), test n=500 (0.847, the primary
reported figure). The demonstration and the headline both use the 500 set, so
the on-screen number and the samples shown come from the same evaluation.

**Whether the dense pH map leaked into training.** It did not. Both terms of the
loss avoid it: the data term uses only the 4 supervised points, and the
smoothness term uses only the prediction itself. The dense map determines the
tissue mask and the final score, nothing else.
