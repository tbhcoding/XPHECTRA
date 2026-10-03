# CHAPTER 3
# METHODOLOGY

This chapter presents the methodology used to develop and evaluate LIGTAS-pH. It covers the research design, the pivot from a planned physical data-collection protocol to a literature-constrained synthetic dataset, the physics-based generation pipeline that replaced it, the instruments used, the model architecture, and the training and evaluation procedures applied to assess the system.

## Research Design

This study adopts a computational, simulation-based research design in which a physics-grounded synthetic dataset is used in place of directly collected experimental data. The pivot was necessitated by the failure of the multispectral imaging hardware before data collection could begin, and was approved by the thesis adviser.

Every parameter governing the synthetic data generator is either traced to a specific published source, reported as a literature-informed sensitivity sweep where no single citable value exists, or explicitly disclosed as a tuned, uncited nuisance parameter. No parameter value in the generator is invented without disclosure.

The claims this study can support are claims about the software pipeline's behaviour under the simulated conditions described in this chapter. This study does not claim, and does not describe, real-pork experimental validation of predictive accuracy.

## Synthetic Data Generation Pipeline

The synthetic dataset is produced by a forward physical model. A randomly generated spatial pH field and an independently generated myoglobin redox state drive two separate optical pathways: pH-driven scattering and myoglobin-driven absorption. The two pathways combine through the Kubelka-Munk two-flux radiative transfer model into six-band reflectance, after which a sensor noise model calibrated against a real hyperspectral tissue cube produces the final image cube.

The myoglobin state is deliberately generated uncorrelated with pH. This matters for the validity of the benchmark: if the two were linked, the network could recover pH from overall brightness alone, and the task would test nothing. Keeping them independent forces the network to separate two physical effects rather than invert a single relationship.

**Figure 3.1.** Schematic of the biophysical forward model.

## Parameters

**Table 3.1.** Generator parameters and their evidentiary status.

| Parameter | Value | Status | Source |
|---|---|---|---|
| eps_deoxy / oxy / met | 6-band arrays | CITED | Tang 2004, Bowen 1949 |
| c_Mb_mean | 0.87 mg/g | CITED | Cross et al. 2018 |
| c_Mb_sd | 0.12 mg/g | BACK-CALCULATED | Same source |
| scatter_a, scatter_b | 8.7436, 1.6618 | FITTED | Jacques 2013, Bergmann 2021 |
| denat_amplitude | 0.4 | SWEPT | No citable value; see Chapter 4 sweep |
| denat_midpoint | 5.70 | CITED (proxy) | Cross et al. 2018 |
| denat_width | 0.28 | SWEPT | No citation found |
| mua_water | 6-band array | CITED | Hale and Querry 1973 |
| water_fraction | 0.732 | CITED | Wojtasik-Kalinowska 2016 |
| sensor_sigma | 0.0054 | MEASURED | Real NHSI cube |
| mu_a_baseline | 0.8 cm⁻¹ | TUNED | Uncited, disclosed |

Of the fourteen parameters, ten are cited, measured, or fitted to published data, and an eleventh (c_Mb_sd) is back-calculated from a cited source. The remaining three are not traceable to a published measurement and are labelled as such in the code rather than presented as measurements. Their status differs, and the study treats them differently:

`denat_amplitude` is swept. Chapter 4 reports the model's performance across the full plausible range of this parameter, so the conclusion does not depend on which value within that range is correct.

`denat_width` is also swept, but it was not included in the baseline comparison reported in Chapter 4. Its effect was examined only against the linear baseline, using a single-seed measurement method that was later found unreliable. This is reported as a limitation rather than as a parameter the sweep covers.

`mu_a_baseline` is tuned and uncited. It was adopted at 0.8 because that value improved agreement with the external 970 nm reference, which means the 970 nm comparison is not independent of it and cannot serve as free-standing validation of this parameter.

A literature-suggested value for `denat_amplitude` (2.19) was tested and rejected. At that value the linear baseline reached an R² of approximately 0.94, which breaks the non-triviality requirement this study imposes on itself, and the 970 nm agreement worsened as well. The adopted value of 0.4 is instead supported by the sweep reported in Chapter 4.

Sample pH was drawn uniformly across 5.35 to 6.45 rather than narrowed to match published PSE and DFD reference figures. Narrowing the range was tested and found to reduce the variance available for the model to explain without improving the underlying model.

## Dataset Composition and Scale

Each synthetic sample consists of a 256 by 256 pixel six-band reflectance cube, a corresponding dense pH map retained solely for evaluation, a tissue mask, and four sparse pH measurements drawn from the hidden field to simulate physical probe readings.

A frozen dataset of 400 samples was generated at a fixed random seed and partitioned into 300 training, 50 validation, and 50 test samples. A further 500 samples were generated under a separate random seed and withheld entirely from training. Using a different seed matters: the seed that produced the frozen dataset would reproduce its own draws, so a set generated with it could overlap the training data. The 500-sample set is therefore an independent draw rather than a further partition of the same generation, and it is the set on which the primary results in Chapter 4 are reported.

**Figure 3.2.** Composition of one synthetic sample: the six spectral bands, an RGB composite, the hidden dense pH map with the four sparse supervision points marked, and the tissue mask.

**Figure 3.3.** Distribution of pH between and within samples across the frozen dataset.

## Model Architecture

Spatial pH prediction is performed by a compact convolutional regression network (CrudeCRN), a small U-Net with two downsampling and two upsampling stages, batch normalisation, and skip connections between corresponding encoder and decoder levels. The network accepts a six-band multispectral cube and outputs a dense per-pixel pH prediction at the same spatial resolution. The output layer's bias is initialised to the physiological pH prior so that training begins near the correct operating range rather than spending early epochs correcting a large constant offset.

**Figure 3.4.** CrudeCRN: a compact U-Net with skip connections.

The architecture was deliberately kept small to establish feasibility within this study's timeframe. It is one instantiation of the broader LIGTAS-pH pipeline rather than a fixed requirement. The training and evaluation methodology described in this chapter, including sparse-point supervision, the smoothness prior, and the held-out evaluation protocol, applies to any model that produces a dense pH map from six-band input, regardless of backbone. Backbone selection for eventual deployment remains an engineering decision for whoever builds the hardware, who would need to weigh inference speed and model size against accuracy on the target device.

## Training

**Table 3.2.** Training configuration.

| Setting | Value |
|---|---|
| Optimizer and learning rate | Adam, 1 × 10⁻⁴ (lowered from 1 × 10⁻³ after diagnosing instability) |
| Batch size and epochs | 8, maximum 50, early stopping with patience 10 |
| Loss | Sparse-point MSE + 0.05 × total-variation smoothness prior |
| Supervision | 4 sparse pH points per sample |
| Seeds | 5, independently trained |

Checkpoint selection uses validation loss only, which is the sparse-point error plus the smoothness term. Neither term reads the dense pH map, so the map the model is scored against never influences which checkpoint is saved.

An early instability in validation performance was traced to the learning rate rather than to batch normalisation, which was the first suspected cause. A separate error in how the R² metric was aggregated, found later, had also overstated how severe that instability appeared. Both are corrected in the current code.

## Instruments

The pipeline runs entirely in software on a single machine. No specialised hardware is required.

**Table 3.3.** Software requirements.

| Component | Version |
|---|---|
| Python | 3.11.5 |
| PyTorch | 2.13.0 (CPU build) |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.1 |
| scikit-learn | for the PLSR baseline |
| Gradio | 6.28.0, for the demonstration interface |

**Table 3.4.** Computational characteristics of the trained model.

| Property | Value |
|---|---|
| Trainable parameters | 118,113 |
| Model size | 0.45 MB |
| Inference time | approximately 31 ms per 256 × 256 sample, CPU only |
| GPU | not used and not required |

These figures are reported because they bear on the third objective. A model of this size and speed imposes no unusual demand on the computing hardware a future device would need.

## System Pipeline

The complete LIGTAS-pH software pipeline proceeds from synthetic data generation, through model training, to inference and heatmap production. The sequence is designed so that it applies unchanged once physical multispectral data becomes available, with only the data source changing rather than the pipeline itself.

**Figure 3.5.** Generation through inference, demonstrated running end to end in Chapter 4.

## Evaluation Protocol

Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and the coefficient of determination (R²).

Three evaluation sets are used: the frozen dataset's validation split of 50 samples, which is used for early stopping; its test split of 50 samples; and the separately generated set of 500 samples. Because R² depends on the amount of variance available to explain rather than on absolute accuracy, agreement across all three sets is treated as the relevant evidence against overfitting rather than any single figure. Results on the 500-sample set are reported as the primary figures in Chapter 4, being the largest set that never influenced training in any way.

Accuracy is also reported as the proportion of pixels falling within fixed tolerance bands of ground truth. These bands depend on no class definition and no class balance, so they describe the predicted map directly and carry no interpretive choice.

A linear regression baseline and a Partial Least Squares Regression baseline with four components, both fitted directly on raw six-band reflectance, are reported alongside the network at every configuration tested. A configuration in which the linear baseline explains the large majority of pH variance is treated as insufficiently challenging and excluded from consideration. This non-triviality requirement is applied as a hard constraint on parameter selection.

A single pooled R² conflates two different questions: how accurately the model estimates each sample's overall pH level, and how accurately it expresses the magnitude of spatial variation within a sample. Both are evaluated and reported separately in Chapter 4, alongside a direct measurement of spatial localisation accuracy.

## Input Validation

No formal published protocol exists for validating this particular combination of modelling choices. The input parameters and key methodological decisions, including the use of equine myoglobin extinction data and the overall acceptability of a physics-based synthetic dataset for this application, were therefore reviewed in consultation with a food technologist with relevant domain expertise. The consultant confirmed these choices are reasonable for the study's purpose, while noting that no standardised certification process exists for this use case.

This consultation is documented separately. It is treated as a supporting check on plausibility, not as a substitute for the literature citations in Table 3.1 or for physical validation on real pork.

## Theoretical Framework

### Meat Quality and Post-Mortem pH Changes

This study is grounded in the established relationship between post-mortem biochemical change and pork quality. Anaerobic glycolysis after slaughter converts glycogen to lactic acid, progressively lowering tissue pH, which affects colour, texture, protein stability, and water-holding capacity, and can produce pale, soft, exudative (PSE) conditions under rapid decline [4, 7]. Because pH varies spatially across a sample and a point probe cannot capture this variation, a spatially resolved, non-invasive measurement approach is motivated independently of the specific acquisition method used [1].

### Optical and Spectral Behaviour of Meat Tissue

As tissue pH falls, light scattering increases, producing paler meat at lower pH [4, 5]. Myoglobin redox state independently governs wavelength-dependent absorption [8]. The synthetic data generator models these two pathways separately and combines them only at the point of computing reflectance, consistent with how the literature describes them as physically distinct mechanisms.

### Light Propagation and Multispectral Radiometry

Light incident on pork tissue is partially absorbed by chromophores and partially scattered by tissue microstructure, and the diffusely reflected fraction carries information about both processes [18]. The six wavelengths used in this study (481, 525, 573, 600, 730, and 970 nm) were selected on the basis of myoglobin absorption features, the isosbestic point near 525 nm, and near-infrared moisture and scattering behaviour, as described in Chapter 2.

### Deep Learning and Spatial pH Prediction

A convolutional neural network is used to learn spatial and spectral patterns directly from multispectral cubes, given the nonlinear relationship between tissue optical properties and biological structure [22, 24]. The network is trained under sparse supervision, using four ground-truth points per sample, which reflects the practical constraint that a real deployment could obtain only a small number of physical probe readings per sample.

### Simulation-Based Modelling

Where real data is unavailable, physics-based simulation is treated as a legitimate methodological choice rather than a compromise, provided every parameter's evidentiary status is disclosed. The specific radiative-transfer formalism implemented here is Kubelka-Munk theory rather than simpler Beer-Lambert attenuation, and the model's physical parameters are assembled from multiple independently published sources, a practice established in comparable tissue-optics simulation work. Supporting literature for both points is developed in Chapter 2.

### Synthesis

Taken together, these foundations support the claim under test: that pH-driven changes in pork tissue optics are in principle detectable through multispectral diffuse reflectance and predictable by a convolutional regression network trained on sparse ground truth. This study tests that claim using a literature-constrained synthetic benchmark rather than real pork measurements, and its conclusions should be read accordingly.

---

# CHAPTER 4
# RESULTS

This chapter presents the trained model's performance on the synthetic benchmark described in Chapter 3, organised according to the study's three objectives: the accuracy of pixel-wise pH prediction, comparison against classical regression baselines across the disclosed parameter range, and demonstration of the pipeline as a working foundation for future physical deployment.

## Validation of the Synthetic Benchmark

Two checks were applied to the generated dataset before any model was trained.

The first confirmed the physical direction of the forward model. Simulated reflectance decreased monotonically with increasing pH across all six bands, consistent with the reduction in light scattering that accompanies decreasing protein denaturation. This sign test passed.

The second established that the inverse problem was non-trivial. A pixel-wise linear regression fitted to raw band values attained an R² of 0.6509 ± 0.0180 across ten independent draws, comfortably below the 0.9 ceiling adopted for this study. The mapping from reflectance to pH could not be recovered by linear means alone.

## Training Behaviour

All five training runs converged. Sparse-point training loss decreased from approximately 0.097 at initialisation to between 0.009 and 0.020 at the selected checkpoint.

Validation loss decreased in parallel during early epochs before diverging. Early stopping on validation loss consistently selected a checkpoint before that divergence, at epochs 9, 14, 21, 10, and 18 across the five seeds. The divergence that followed was pronounced. In the most marked case, validation loss rose from 0.0210 at the selected epoch to 0.1161 by the time training terminated, a factor of 5.5.

At the selected checkpoints the ratio of validation to training loss ranged from 1.15 to 1.89, with a mean of 1.61. Ratios in this range indicate the model had not overfit the training partition at the point of selection.

**Figure 4.1.** Training and validation loss for seed 3, with the early-stopping checkpoint marked. Seed 3 was selected as the seed whose held-out performance is closest to the five-seed mean, rather than for the appearance of its curve.

## Predictive Performance

Performance was evaluated on all three sets described in Chapter 3, and all three agreed, which is the evidence against overfitting to any one selection split. The 500-sample set is reported as the primary figure throughout this chapter.

**Table 4.1.** Headline predictive accuracy, 500 unseen test samples, five seeds.

| Metric | Result |
|---|---|
| R² | 0.847 ± 0.038 |
| Mean absolute error | 0.094 ± 0.015 pH units |
| Root mean square error | 0.128 ± 0.016 pH units |

**Table 4.2.** Agreement across the three evaluation sets.

| Evaluation set | R² | MAE (pH) | Role |
|---|---|---|---|
| Validation, n = 50 | 0.849 | 0.090 | Used for early stopping |
| Test, n = 50 | 0.839 | 0.090 | Never used during training |
| Test, n = 500 | 0.847 | 0.094 | Primary reported figure |

Validation is the only set that influenced training and is therefore the one that could be optimistic. It is not inflated relative to the other two.

**Table 4.3.** Per-seed results on the 500-sample test set.

| Seed | R² | MAE (pH) | RMSE (pH) |
|---|---|---|---|
| 0 | 0.778 | 0.121 | 0.156 |
| 1 | 0.891 | 0.080 | 0.109 |
| 2 | 0.860 | 0.083 | 0.124 |
| 3 | 0.839 | 0.096 | 0.133 |
| 4 | 0.868 | 0.088 | 0.120 |
| Mean ± SD | 0.847 ± 0.038 | 0.094 | 0.128 |

Seed 0 was a consistent low outlier across all three metrics, a pattern replicated in independent runs including on a second machine. It is reported rather than excluded, and the aggregate statistics reflect its influence. All figures in this chapter are means across the five seeds; no single run is reported in isolation.

## Spatial Behaviour of the Predicted Maps

A pooled R² does not indicate whether the predicted heatmap is usable at the pixel level. Measured directly, 81.5% of pixels fell within ±0.15 pH of ground truth and 89.3% within ±0.20 pH.

As supporting evidence, predicted pixels were assigned to PSE, normal, or DFD quality classes using boundaries interpolated from published reference anchors (Sristi et al., 2025). Pixels landed in the correct class 87.7% of the time, against 62.6% for a classifier that always guessed the most common class, a lift of 25.1 points. This figure is reported as supporting evidence only, since the class boundaries are interpolated rather than directly published.

Neither pooled nor per-sample R² indicates whether the model locates elevated pH correctly, since both measure the magnitude of predicted variation rather than its position. Localisation was therefore measured directly, restricted to the 282 of 500 test samples that genuinely spanned more than one quality class.

**Table 4.4.** Spatial localisation on multi-class samples.

| Measure | Result | Chance level |
|---|---|---|
| Per-pixel class accuracy | 79.0% ± 2.9% | not applicable |
| Overlap of predicted and true hottest 20% | 51.3% ± 4.3% | 20% |
| Predicted hot region genuinely hotter | 99.6% ± 0.5% | 50% |

The overlap measure is the demanding one, since it requires the predicted hot region to coincide with the true hot region rather than merely point in the right direction. At 51.3% against a 20% chance level it is well above chance while showing that pixel-level precision remains partial. The 99.6% figure tests direction alone and is reported for completeness rather than as the headline localisation result.

**Figure 4.2.** Hidden true pH map, CRN prediction, and absolute error for one held-out sample. The sample shown is the median case by per-sample R² within its split, selected by that rule rather than by inspection.

Mean per-sample R² was −2.17 on the 500-sample test set, and negative on the large majority of individual samples for every seed. A negative value indicates that the predicted map for a given sample scored worse, on this magnitude-sensitive measure, than a flat map drawn at that sample's own average pH.

This coexists with the pooled and localisation results because between-sample pH variation (SD = 0.304) was approximately 3.6 times larger than within-sample variation (SD = 0.084). Pooled R² is dominated by the model's reliable estimation of each sample's overall level, while per-sample R² measures accuracy against the much smaller within-sample scale, on which even correctly located errors appear severe.

Diagnostics confirm this is a magnitude-calibration problem rather than an absence of spatial signal. Predicted maps correlated with ground truth at r = 0.541 ± 0.069, indicating genuine spatial structure was recovered, while the ratio of predicted to true spatial standard deviation was 1.290 ± 0.109, an over-amplification of approximately 29%. Optimal rescaling of the existing predictions would raise mean per-sample R² to approximately +0.297 ± 0.071. That figure is an upper bound on what a calibration correction could achieve, not a result already obtained.

## Comparison with Baseline Methods

To confirm that the task required more than a simple relationship between raw reflectance and pH, model performance was compared against a linear regression baseline and a Partial Least Squares Regression baseline with four components, both fitted and evaluated on the same held-out pixels.

The convolutional regression network outperformed both linear regression (R² = 0.6212) and PLSR (R² = 0.6206) by a margin of 0.227.

PLSR converged to almost exactly the same solution as ordinary linear regression. A singular value decomposition of the six-band design matrix explains why: the top four components carry 97.31% of its variance, and the matrix has a condition number of 8.0. The six bands are strongly collinear, so a four-component PLSR already spans nearly the whole space a six-band linear fit can use.

Because `denat_amplitude` has no established literature value, the comparison was repeated across the full disclosed range of plausible values at matched sample scale.

**Table 4.5.** Network advantage over classical baselines across the denat_amplitude sweep.

| denat_amplitude | Linear | PLSR | CRN (mean ± SD) | Margin |
|---|---|---|---|---|
| 0.2 | 0.3113 | 0.3110 | 0.7270 ± 0.0820 | +0.416 |
| 0.4 (adopted) | 0.6212 | 0.6206 | 0.8486 ± 0.0368 | +0.227 |
| 0.6 | 0.7616 | 0.7608 | 0.8862 ± 0.0139 | +0.125 |
| 0.8 | 0.8308 | 0.8299 | 0.8999 ± 0.0169 | +0.069 |

Values in this table are measured on the validation split of each amplitude's own dataset, using the same split for all three methods so the comparison is like-for-like. The amplitudes other than 0.4 were never evaluated on the 500-sample set, so validation is the only scale on which all four points are comparable. This is why the network column reads 0.8486 at the adopted setting while the headline in Table 4.1 is 0.847.

**Figure 4.3.** Network against baselines across the sweep, with error bars on the network series.

The network outperformed both baselines at every tested amplitude, by margins each exceeding the 0.05 threshold this study adopts. The margin narrowed as amplitude increased because the underlying pH-scattering relationship became more linearly separable, and at 0.8 the linear baseline alone reached 0.83, close to the non-triviality ceiling of 0.9. This narrowing pattern, rather than the size of the margin at any single setting, was the basis for retaining 0.4 as the operating value.

## Prototype Demonstration

The third objective was addressed by assembling the trained pipeline into a working demonstration interface, which accepts a six-band sample and returns a predicted pH map.

**Figure 4.4.** The prototype interface: a held-out sample is selected, and the predicted pH map is produced from its six spectral bands.

The interface loads one fixed checkpoint and performs inference on samples drawn from the 500-sample held-out set, the same set on which Table 4.1 is reported. The checkpoint used is seed 3, whose held-out R² of 0.839 is the closest of the five to the reported mean of 0.847. Selecting on that basis rather than taking the best-performing seed (0.891) means the interface does not demonstrate accuracy higher than the study reports.

The prototype produces a prediction in approximately 31 ms on CPU. It reads no ground-truth pH at any point; the dense map is used only to determine which pixels are tissue.

Running the complete pipeline in this way demonstrates that the software components of LIGTAS-pH function as a coherent system from raw input to final deliverable, with no step requiring manual intervention or data unavailable from the pipeline itself. It does not demonstrate operation on physically acquired data, which remains future work. Because the training and evaluation methodology is independent of the backbone, that transition would require substituting the data source rather than redesigning the pipeline.

---

# CHAPTER 5
# SUMMARY, FINDINGS, CONCLUSIONS, AND RECOMMENDATIONS

## Summary

This study examined whether a convolutional regression network, trained on four sparse pH readings per sample together with a spatial smoothness prior, could recover a useful spatially resolved estimate of pork pH from six-band multispectral reflectance.

The multispectral imaging hardware originally intended to supply training data failed before collection began. With the adviser's approval, the study was restructured around a physics-based synthetic benchmark in which every generator parameter is traced to a published source or explicitly disclosed as swept or tuned. The forward model drives pH through scattering and myoglobin redox state through absorption as independent pathways, combining them only at the point of computing reflectance, so that the resulting task cannot be solved by brightness alone.

A frozen dataset of 400 samples was generated and partitioned into training, validation, and test splits, with a further 500 samples generated under a separate seed and withheld entirely from training. Five independently seeded networks were trained under sparse supervision and evaluated on all three sets, alongside linear regression and PLSR baselines fitted on the same data. The comparison was repeated across the full plausible range of the study's least-certain physical parameter. The trained pipeline was then assembled into a working demonstration interface.

## Findings

**1. The synthetic benchmark behaves as a physics-based model should, and poses a non-trivial problem.** Simulated reflectance fell monotonically with rising pH across all six bands, matching the known scattering mechanism. A pixel-wise linear fit on raw band values reached only 0.6509 ± 0.0180 over ten draws, well below the 0.9 ceiling adopted for this study, confirming that the mapping from reflectance to pH cannot be recovered by linear means.

**2. The network predicted pixel-wise pH accurately on data it had never seen.** On 500 unseen samples across five independently trained seeds, it achieved an R² of 0.847 ± 0.038, a mean absolute error of 0.094 ± 0.015 pH units, and a root mean square error of 0.128 ± 0.016. At the pixel level, 81.5% of predictions fell within ±0.15 pH of ground truth and 89.3% within ±0.20 pH. Results agreed across all three evaluation sets, with the validation split showing no inflation relative to the two sets that never influenced training.

**3. The network outperformed both classical baselines, and that advantage did not depend on the study's least-certain parameter.** At the operating value it exceeded linear regression (0.6212) and PLSR (0.6206) by 0.227. Repeating the comparison across the full disclosed range of denat_amplitude showed the network ahead at every setting, by margins from 0.069 to 0.416. The two baselines converged to nearly the same solution because the six bands are strongly collinear, with the top four components of the design matrix carrying 97.31% of its variance.

**4. The model locates spatial variation reliably but does not yet calibrate its magnitude.** On the 282 samples spanning more than one quality class, per-pixel class accuracy was 79.0% ± 2.9% and the predicted hottest 20% of pixels overlapped the true hottest 20% in 51.3% ± 4.3% of cases, against a 20% chance level. Mean per-sample R² was nonetheless negative at −2.17, because within-sample pH variation is roughly 3.6 times smaller than between-sample variation, and the model over-amplifies spatial variation by approximately 29%. Predicted maps correlated with ground truth at r = 0.541 ± 0.069, confirming that genuine spatial structure was recovered. Optimal rescaling would raise per-sample R² to approximately +0.297, which bounds what a calibration correction could achieve.

**5. The complete software pipeline runs end to end.** Data generation, training, inference, and heatmap production operate as one sequence with no manual intervention, and the trained model was assembled into a working demonstration interface producing a pH map in approximately 31 ms on CPU at 118,113 parameters.

## Conclusions

**1.** A physics-based synthetic benchmark, with every parameter either cited or explicitly disclosed, provides a workable substitute for physically collected data when hardware is unavailable, provided the claims drawn from it are confined to behaviour under the simulated conditions. The benchmark constructed here is internally consistent with the optical mechanisms described in the literature and is not solvable by trivial means.

**2.** A convolutional regression network trained on four sparse points per sample can recover a dense pixel-wise pH field from six-band reflectance to within approximately 0.09 pH on this benchmark, and does so with a substantial and robustly confirmed advantage over classical regression methods. Because that advantage holds across the entire plausible range of the study's least-certain parameter rather than at one chosen value, the conclusion does not rest on a parameter the literature cannot settle.

**3.** The model's spatial capability is better described as reliable localisation with imperfect magnitude calibration than as a single level of spatial accuracy. It identifies where pH is elevated within a sample; it overstates how much. The diagnosis is specific enough to bound what a correction could achieve, which makes this a defined limitation rather than an open failure.

**4.** The software pipeline functions as a coherent system and is structured so that the transition to physical data would require substituting the data source rather than redesigning the pipeline. This establishes a working foundation for future hardware integration. It does not establish validated predictive accuracy on real pork, which the study neither claims nor attempted.

## Recommendations

**1. Physical validation on real pork, once suitable imaging hardware becomes available.** This is the necessary step to move beyond a feasibility demonstration. Because the model's weights were learned on simulated reflectance, the realistic path is fine-tuning from the current weights on real samples with laboratory-measured pH, followed by re-validation, rather than training from scratch.

**2. A calibration correction for spatial magnitude.** The diagnosis in Finding 4 identifies a consistent over-amplification rather than an absence of signal, and bounds the achievable improvement at approximately +0.30 per-sample R². This is the most tractable improvement available without new data.

**3. Sourcing or further constraining denat_width and mu_a_baseline.** Both remain outside the evidence base: denat_width was not covered by the baseline comparison, and mu_a_baseline was tuned against the same external reference it would otherwise help validate. A published measurement for either would remove a disclosed weakness.

**4. Extending the output for end users.** The current interface returns a continuous pH map. A view that classifies regions into quality categories and reports their proportions would be more directly usable by inspection personnel. This was not implemented because the class boundaries available to this study are interpolated from reference anchors rather than taken from a published standard, and a classified view would make those boundaries load-bearing in a way the present reporting does not.

## Limitations

**No real-pork validation.** This is the study's principal limitation. Every result reported here describes behaviour on a synthetic benchmark. The only external comparison available was at 970 nm, the single wavelength where this study's band design overlaps an independently collected hyperspectral dataset, and there the simulated reflectance read approximately 22 to 34% above the real reference. That comparison is exploratory, covers one wavelength, and is not independent of the tuned mu_a_baseline parameter, so it should not be read as validation.

**Elevated error at tissue boundaries.** Prediction error within an 8-pixel band along the tissue edge was 0.156 pH against 0.070 pH in the interior, a ratio of 2.31, present in 97% of samples. A convolutional network has less surrounding tissue to draw on at the mask edge. For applications concerned with the body of a cut rather than its perimeter, the interior figure is the more representative one.

**Two parameters outside the sweep's coverage.** denat_width is swept in the generator but was not included in the baseline comparison reported in Chapter 4, and mu_a_baseline is tuned and uncited. Neither is hidden, but neither carries the same evidentiary support as the parameters in Table 3.1 marked CITED or MEASURED.

**Corrections made during verification.** Two errors affecting reported figures were identified and corrected during the project team's own review before external assessment. An error in metric aggregation, in which validation R² had been computed per mini-batch and averaged rather than pooled as the baselines were, understated the headline figure; the corrected value is reported throughout. The rationale originally given for generating a 500-sample evaluation set, that a larger set would narrow the reported error bars, was found not to hold, since those bars reflect variance across training seeds rather than evaluation-sample size; the set is retained for the independence reason given in Chapter 3.
