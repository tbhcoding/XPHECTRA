# What changed, and why

A record of the edits made on 5 and 6 October 2026, so the changes can be reviewed without rereading the working session. Nothing here has been committed. Every item is reversible.

Read this first, then `DEFENSE_CHEATSHEET.md` for the values themselves.

---

## Chapters 3 to 5

**Three wrong numbers in Table 4.3.** Seed 1 R² read 0.891, the evidence says 0.890. Seed 2 RMSE read 0.124, the evidence says 0.123. Seed 3 RMSE read 0.133, the evidence says 0.132. All three had been rounded from a four-decimal intermediate instead of the raw value. Both evidence files agree, so the chapter was wrong.

**A promise that could not be kept.** Chapters 4 and 5 stated in three places that a calibration correction could raise per-sample R² to about +0.30. That figure came from an oracle rescaling that uses each sample's own true standard deviation, which is unavailable at prediction time. The achievable correction, fitted properly, reaches -0.393. The claim is now reported as a measured result rather than a promise, and the oracle figure is kept but labelled as unattainable.

**The calibration result, added.** A new section in Chapter 4 with Table 4.5. It separates the two miscalibrations, which act in opposite directions, and reports what correcting each recovers. The deployable half raises R² from 0.847 to 0.904. This also establishes that the model is under-fitting rather than over-fitting, which means the headline understates rather than overstates performance.

**Baselines on the headline set, added.** Table 4.6. Previously linear regression and PLSR were reported only on the 50-sample validation split, while the headline was measured on 500 samples. The two were not comparable. They are now, and the network leads by 0.197 R² with 40 percent lower mean absolute error.

**The pH range.** The sampling design is now stated as a coverage design rather than a representativeness claim, and the chapter discloses that 43 percent of samples exceed the upper anchor of the reference scale cited in Chapter 2. Chapter 4 reports what the metrics become when restricted to that scale: R² falls to 0.705 while mean absolute error improves to 0.085. A matching limitation was added to Chapter 5.

**Class boundaries, now stated.** Chapter 4 previously said only that boundaries were "interpolated from published reference anchors." It now states that they are 5.4 and 5.8, and that these are midpoints between the cited anchors of 5.2, 5.6 and 6.0.

**Smaller corrections.** Table 3.2 said the model occupies 0.45 MB on disk; the checkpoint is 0.50 MB, so the row now says so. Table 3.1's caption explains that eleven rows cover fourteen parameters. Chapter 3 now states the generated pH range, which it previously never did. Table 4.7's note discloses that three of its four amplitudes rest on three seeds rather than five.

---

## Chapters 1 and 2

These were drafted from the Google Docs version. The LaTeX version is the one that matters, so treat the text below as changes to apply rather than a replacement.

**The citation numbering was broken.** Chapter 1's bracketed numbers pointed into a different reference list than the one in the document. Reference 12 was an olive-oil spectrofluorimeter where the pork muscle should have been; reference 8, a Monte Carlo photon simulation, was standing in for four separate meat-quality definitions. Every citation is now author-year, which cannot silently break, and must be converted back to the project's numbered style once the master list is merged.

**Ten sources were cited but never listed.** Including Cross 2018, Hale and Querry 1973, Tang 2004 and Bowen 1949, which are the sources behind the generator's own parameters. They are now in the list. Eight entries are marked `[VERIFY]` because the full details could not be confirmed and must come from the team's source files.

**Specific Objective 3 was not measurable.** It described a method rather than an outcome and repeated Objective 1. It now reads as a demonstrable result and maps to what Chapter 4 reports. Objectives 1 and 2 are unchanged.

**The General Objective promised hardware that does not exist.** Part (b), the design specification of an optical acquisition device, was removed at the team's instruction, along with a matching claim in the Significance section.

**Claims that could not be supported.** A reported coefficient of determination of 0.93 was removed; it belonged to a different study on real hyperspectral data and invited an unfavourable comparison. An author attribution was corrected from Cai to Yao. The stated pH range of "5. to 6.5" was corrected to 5.2 to 6.8, which is what the generator produces. A claim of experience in embedded system development was replaced with what the study actually involved. An internal editing legend, left over from drafting, was removed from the top of Chapter 2.

**Sections added to Chapter 2**, at the team's request: the provenance of every optical parameter, the pork-to-horse myoglobin substitution with its justification and its known limitation, the three parameters that have no citable source, and the argument that a synthetic benchmark is the only setting in which a pixel-wise prediction can be scored against pixel-wise truth.

**The Project Dictionary** is now conceptual and operational for each term, with citations added where there were none.

---

## New files

| File | What it is |
|---|---|
| `manuscript/chapters_1_to_2.md` and `.pdf` | the revised Chapters 1 and 2 |
| `manuscript/DEFENSE_CHEATSHEET.md` and `.pdf` | every value in Chapters 3 to 5, explained, with the answers to the likely questions |
| `manuscript/WHAT_CHANGED.md` | this file |
| `check_calibration.py` | measures the miscalibration and what correcting it recovers |
| `compute_test500_baselines.py` | runs the baselines on the 500-sample set |
| `check_range_restriction.py` | measures the effect of restricting to the cited pH range |
| three JSON files in `metric_check_outputs/` | the evidence those three scripts produced |

No model was retrained. No dataset was regenerated. No parameter was changed. All three scripts read the committed checkpoints and write nothing but their own output.

---

## Current state and what still needs a person

1. **Figures: done.** All nine figures now resolve. Chapter 3 gained two new diagrams drawn by the team, the forward optical model (Figure 3.1) and the convolutional regression network (Figure 3.3), and the pipeline diagram (Figure 3.5) was corrected so its first box reads "generation" rather than "generation/acquisition". Chapter 3 figures were renumbered 3.1 to 3.5 to make room; there are no in-text references to figure numbers, so only captions changed. No placeholders remain in either chapter file.
2. **Eight `[VERIFY]` references: done.** The reference list was completed in `9338d2d`. Neither chapter file contains a `[VERIFY]` marker any more.
3. **Someone should read Chapters 1 and 2.** They were substantially rewritten and nobody has checked that no intended meaning was lost.

## One decision to make

The pH-distribution limitation was added on the judgement that disclosing it is safer than leaving it discoverable, since Figure 3.4 already plots the distribution and the measurement favours the study. It does, however, put a weakness in writing that was not there before. It can be removed in full if the team prefers.
