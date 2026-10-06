# Chapter 1: every addition and change

Exact text, so each item can be checked against the original PDF. Nothing is committed.

Three categories:

- **KEPT**, was in your PDF, unchanged
- **CHANGED**, was in your PDF, wording altered, with the reason
- **ADDED**, not in your PDF

---

## Project Dictionary

### KEPT, original wording and original bracket citation

These were restored verbatim after I wrongly replaced them. Citations [5], [7], [8], [9], [12] and [17] are yours and untouched.

| Entry | Citation |
|---|---|
| Convolutional Regression Network (CRN) | [8] |
| DFD (Dark, Firm, Dry) Meat | [8] |
| Hyperspectral Imaging (HSI) | [5] |
| Isosbestic Point | [9] |
| Longissimus thoracis et lumborum (LTL) | [12] |
| Multispectral Imaging (MSI) | [7] |
| PSE (Pale, Soft, Exudative) Meat | [8] |
| Pork Meat | [12] |
| Spatial Mapping | [17] |

Also kept with original wording, no citation in your PDF and none added, because these are your own terms: **Denaturation Amplitude**, **Literature-Constrained Synthetic Benchmark**, **Non-Triviality Baseline**.

### CHANGED

**pH Level**, citation [8] kept, one phrase removed.

- Yours: "where a rise from the normal range (5.4–5.8) to higher levels (**6.0+**) indicates microbial spoilage and protein degradation"
- Now: "where a rise from the normal range (5.4 to 5.8) indicates protein degradation and the onset of spoilage"
- Also: "Used to determine whether the meat is PSE, **RSE, RFN,** or DFD" became "PSE, **normal**, or DFD"

**Reason:** the pipeline classifies DFD at pH ≥ 5.8, not 6.0. Leaving "6.0+" in the dictionary means a panelist finds the system calling pH 5.9 "DFD" while your own definition says the problem starts at 6.0. The RSE and RFN classes are also never used anywhere in the study; only three classes are reported.

**LIGTAS-pH**, reworded.

- Yours: "The intended handheld **prototype** system comprising an integrating dome, a monochrome global shutter camera, a six-wavelength narrowband LED array, and an embedded edge device running a trained CRN"
- Now: "The system proposed by this study, comprising a physics-based synthetic data generator, a trained Convolutional Regression Network, and a two-dimensional pH heatmap output. What was developed and evaluated in this study is its software pipeline."

**Reason:** "prototype" was ruled out on the consultant's advice, and the entry listed four hardware components that do not exist.

**Non-invasive Multispectral Imaging (MSI) System**, citation [7] kept, second half reworded.

- Yours: "...**intended to be realized in this study** as the optical acquisition module of LIGTAS-pH, consisting of an integrating dome, a monochrome global shutter camera, and six narrowband LEDs"
- Now: "...**No such system was implemented in this study.** What the study simulates is the six-channel reflectance image stack that such a system would produce, which serves as the input to the regression model."

**Reason:** "intended to be realized in this study" states something that did not happen.

### ADDED, citation where your PDF had none

| Entry | Citation added | DOI |
|---|---|---|
| Kubelka-Munk Theory | Thennadil, 2008 | 10.1364/JOSAA.25.001480 |
| Partial Least Squares Regression | Wold, Sjostrom and Eriksson, 2001 | 10.1016/S0169-7439(01)00155-1 |
| Sparse-to-Dense Regression | Hu et al., 2023 | 10.1109/TPAMI.2022.3229090 |
| Sparse-to-Dense Regression | Rudin, Osher and Fatemi, 1992 | 10.1016/0167-2789(92)90242-F |

All four are free to access. Thennadil has an open accepted manuscript at strathprints.strath.ac.uk/13710/.

### REMOVED

**Global Shutter Camera** [4] and **Narrowband LED** [14]. Both define hardware components that were never built and are referenced nowhere else in the study. Say the word if either should come back.

---

## Objectives

**CHANGED, Specific Objective 3.**

- Yours: "To establish a software pipeline in which a CRN is trained on four simulated pH readings per sample and evaluated against the withheld full pH map."
- Now: "To demonstrate that the complete pipeline operates end to end, from data generation through training to inference and heatmap output."

**Reason:** the original described a method rather than a result, so there was nothing a panel could mark as achieved. It also repeated Objective 1. Objectives 1 and 2 are unchanged.

**CHANGED, General Objective.** Part (b), "the design specification of a low-cost portable optical acquisition device intended for future physical deployment", removed at your instruction. Nothing in Chapters 3 to 5 delivered it.

---

## Background of the Problem

**CHANGED.** "Cai et al. [2019]" corrected to "Yao et al. (2019)". Cai is the second author of that paper.

**REMOVED.** "with a coefficient of determination of 0.93". That figure belongs to a different study using real hyperspectral data for a point measurement, and quoting it invites a direct comparison against your 0.847 that is not like for like. The study itself is still cited.

**KEPT unchanged:** the WHO foodborne disease figures, the pH 5.4 to 5.8 statement, Barbin et al. 2012, Tang et al. 2023, and the whole pivot paragraph describing the hardware failure.

**CHANGED.** The Metro Manila wet market sentence. The claim could not be traced to any source; published studies with comparable findings are located in Negros Occidental and Dasmarinas, Cavite. The sentence now states the general concern without the unsupported specific. If you hold the original source, the stronger wording can be restored.

---

## Significance of the Study

**CHANGED, preamble.** "the anticipated value of LIGTAS-pH **upon completion of physical hardware validation**" became a statement distinguishing what the study contributes now from what the approach could offer if later validated. The original implied hardware completion was planned.

**CHANGED, Researchers bullet.** Yours claimed experience in "image preprocessing, machine learning regression, and **embedded system development**", and in "**image calibration, ROI mapping**". None of that happened once the rig failed. It now claims physics-based simulation, sparse-to-dense regression, validation under limited supervision, parameter provenance discipline, and adapting a research design when an instrument becomes unavailable.

**CHANGED, Electronics engineers bullet.** Yours said the study "provides a practical framework for integrating a multispectral imaging module with synchronized LED illumination, a monochrome camera, and embedded processing" and could "serve as a guide for developing other low-cost, portable, field-oriented optical sensing systems". It now claims only the two things the study actually establishes: the computational budget any future device must meet, and the wavelength requirement. It ends with an explicit statement that no acquisition hardware was built or tested.

**KEPT unchanged:** Meat inspectors, Meat processing facilities, Public health, Regulatory agencies, Computer science, Future researchers, and Camarines Sur Polytechnic Colleges.

---

## Scope and Limitations

**CHANGED.** "a target range of **5. to 6.5**" became "a generated range of **5.2 to 6.8**". The generator clips to 5.2 and 6.8; this was verified in the code and measured across both datasets. Both ends of the original were wrong.

**CHANGED.** "Except where explicitly stated otherwise, references in this section to acquisition or measurement describe the system's intended design rather than a procedure carried out in this study" became "Where this section describes acquisition or measurement, it describes the design that was intended. No image was captured and no physical measurement was taken in this study."

**CHANGED.** "evaluated across a disclosed range of assumed pH-scattering coupling strengths" became a plain-language sentence saying the link between pH and scattering has no single published value, so the comparison is repeated across its plausible range.

**CHANGED.** The sparse supervision paragraph was shortened and made plainer, ending "This is stated because the output map looks equally confident everywhere, and it is not."

**ADDED.** Tolerance-band accuracy to the list of evaluation metrics. It was omitted, despite being the source of the 81.5% headline figure.

**KEPT unchanged:** the pork loin scoping paragraph, the physics-based synthetic dataset paragraph with its no-invented-parameters statement, and the closing paragraph ending "Physical validation on real pork remains a necessary direction for future work."

---

## A note on the pork loin question

The restriction is not supported by claiming the parameters were measured on loin. They were not, uniformly:

| Parameter | Measured on |
|---|---|
| Extinction coefficients, three of them | equine myoglobin |
| Scattering coefficients | porcine muscle, not loin-specific |
| Water absorption | pure water, species-independent |
| Sensor noise | a bare tray, not tissue |
| Myoglobin concentration | pork loin, 599 pigs |
| Ultimate pH midpoint | pork loin, same population |
| Water fraction | pork longissimus dorsi |

The Scope therefore keeps your original reasoning, that pork loin is the established reference muscle in spectral-based meat quality research. That is accurate and does not overreach.

If a panelist asks how the study can claim a loin restriction when some parameters are equine, the distinction is between quantities that differ between cuts and quantities that do not. How much myoglobin a muscle contains, what its ultimate pH is, and how much water it holds all differ between cuts, and all three are taken from loin measurements. What myoglobin absorbs at a given wavelength is a property of the molecule and does not differ between cuts; it is taken from the best available measurement, which is equine, and that substitution is disclosed in Chapter 2, in Assumption 1 of Chapter 3, and in the Chapter 5 limitations.
