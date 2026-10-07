# CHAPTER 3
# METHODOLOGY

This chapter presents the methodology used to develop and evaluate LIGTAS-pH. It covers the research design, the mathematical models underlying the synthetic data generator and the prediction network, the materials and procedures used, the evaluation methods applied, and the theoretical foundations on which the study rests.

## Research Design

This study adopts a computational, simulation-based research design in which a physics-grounded synthetic dataset is used in place of directly collected experimental data. The pivot was necessitated by the failure of the multispectral imaging hardware before data collection could begin, and was approved by the thesis adviser.

Every parameter governing the synthetic data generator is either traced to a specific published source, reported as a literature-informed sensitivity sweep where no single citable value exists, or explicitly disclosed as a tuned, uncited nuisance parameter. No parameter value in the generator is invented without disclosure.

The claims this study can support are claims about the software pipeline's behaviour under the simulated conditions described in this chapter. This study does not claim, and does not describe, real-pork experimental validation of predictive accuracy.

## Theorems, Algorithms, and Mathematical Models

The synthetic data generator and the prediction network each rest on established formalisms. This section presents the models actually implemented, in the form in which they were implemented.

### Kubelka-Munk Radiative Transfer Model

Diffuse reflectance from the simulated tissue is computed using the Kubelka-Munk two-flux model, which describes light transport through a scattering and absorbing medium in terms of two opposed diffuse fluxes. For a semi-infinite medium, the diffuse reflectance is

> R_inf = 1 + (K / S) - sqrt[ (K / S)² + 2 (K / S) ]  (3.1)

where K is the absorption coefficient and S the scattering coefficient of the medium. Each is derived from the absorption and scattering terms defined below. (Thennadil, 2008)

### The Forward Optical Model

The generator produces each sample's reflectance from two separate physical pathways. pH acts on one of them and not the other, which is what prevents the prediction task from reducing to the inversion of a single formula.

Scattering follows a wavelength power law, scaled by a pH-dependent denaturation factor:

> mu_s'(lambda, pH) = a (lambda / 500)^(-b) × [ 1 + A · sigma( (pH - m) / w ) ]  (3.2)

where a and b set the wavelength dependence and the bracketed term applies the pH effect, using the denaturation parameters reported in Table 3.1. That term decreases as pH rises, encoding the established mechanism that lower pH produces greater protein denaturation, greater scattering, and therefore paler meat.

Absorption is driven by myoglobin redox state, water, and a baseline term, with no pH term at all:

> mu_a(lambda) = ln(10) · c_Mb · [ f_d eps_d + f_o eps_o + f_m eps_m ] + w_f · mu_a,water + mu_a,base  (3.3)

where c_Mb is the myoglobin concentration, the bracketed term combines published extinction coefficients for myoglobin at each wavelength, w_f is the water fraction, and mu_a,base a uniform baseline. All coefficients are taken from the sources listed in Table 3.1.

The separation of Equations 3.2 and 3.3 is what makes the benchmark non-trivial. pH enters the forward model through scattering only. Myoglobin redox state is generated independently of pH and enters through absorption only. A model attempting to recover pH must therefore separate two physical effects using spectral shape across all six bands, rather than invert a single relationship or read overall brightness.

![](figures/fig_forward_model.png)
**Figure 3.1.** The forward optical model. pH acts only on scattering and myoglobin redox state only on absorption; the two pathways meet for the first time at the Kubelka-Munk step.

![](figures/fig_dataset_sample.png)
**Figure 3.2.** Composition of one synthetic sample: the six spectral bands, an RGB composite, the hidden dense pH map with the four sparse supervision points marked, and the tissue mask.

### Convolutional Regression Network Architecture

Spatial pH prediction is performed by a compact convolutional regression network, referred to here as the CRN, implemented as a small U-Net with two downsampling and two upsampling stages and skip connections between corresponding encoder and decoder levels. (Ronneberger, Fischer and Brox, 2015)

The network accepts a six-band multispectral cube and returns a dense per-pixel pH prediction at the same spatial resolution. An encoder-decoder structure is required because the output is a map rather than a single value. A classification backbone returns one prediction per image, which does not satisfy the study's objective.

![](figures/fig_crn_architecture.png)
**Figure 3.3.** The convolutional regression network: a compact U-Net with skip connections.

### Main Features of the Convolutional Regression Network

**Encoder-decoder structure.** The encoder reduces spatial resolution while increasing channel depth, so that units in the deeper layers respond to larger regions of the input. The decoder reverses this, restoring full resolution so that a prediction is produced for every pixel.

**Skip connections.** Each decoder level receives, in addition to the upsampled features from below, the encoder features at the same resolution. Without them, fine spatial detail lost during downsampling cannot be recovered, and the predicted map would be smooth but imprecise.

**Convolution blocks.** Every block applies a 3 by 3 convolution, batch normalisation, and a rectified linear activation, twice. Downsampling uses 2 by 2 max pooling and upsampling uses 2 by 2 transposed convolution.

**Compact size.** The network has 118,113 trainable parameters and its saved checkpoint occupies 0.50 MB. It was deliberately kept small to establish feasibility within the study's timeframe, and is one instantiation of the pipeline rather than a fixed requirement. The training and evaluation methodology described below applies to any model producing a dense pH map from six-band input, regardless of backbone.

### Sparse Supervision and the Smoothness Prior

The network is trained on four pH readings per sample rather than the dense map. The dense map exists in the simulation but is withheld during training and used only for scoring. This reflects the practical constraint that a physical deployment could obtain only a small number of probe readings per sample.

Four points alone do not determine a 256 by 256 field. The problem is made well-posed by adding a total-variation smoothness prior, which penalises abrupt pixel-to-pixel change and so propagates information from the supervised points into the surrounding field. (Rudin, Osher and Fatemi, 1992) The training objective is

> L = (1 / P) · sum over p of ( y_hat_p - y_p )² + lambda · TV( y_hat )  (3.4)

where P is the number of supervised points per sample, y_p the pH at point p, y_hat_p the prediction there, and lambda the weight on the smoothness term. Total variation is computed over the full predicted grid as

> TV( y_hat ) = mean | y_hat(i+1, j) - y_hat(i, j) | + mean | y_hat(i, j+1) - y_hat(i, j) |  (3.5)

Neither term in Equation 3.4 reads the dense pH map. The first uses only the four supervised points, the second only the prediction itself. This matters for the integrity of the evaluation: the map the model is scored against never influences training, and never influences which checkpoint is saved.

## Material and Evaluation Methods

### Instruments

#### Dataset

Each synthetic sample consists of a 256 by 256 pixel six-band reflectance cube, a dense pH map retained solely for evaluation, a tissue mask, and four sparse pH measurements drawn from the hidden field. The six wavelengths are 481, 525, 573, 600, 730, and 970 nm.

**Table 3.1.** Generator parameters and their evidentiary status. Eleven rows cover fourteen parameters: the extinction-coefficient row covers three and the scattering row covers two.

| Parameter | Value | Status | Source |
|---|---|---|---|
| eps_deoxy / oxy / met | 6-band arrays | CITED | Tang 2004, Bowen 1949 |
| c_Mb_mean | 0.87 mg/g | CITED | Cross et al. 2018 |
| c_Mb_sd | 0.12 mg/g | BACK-CALC. | Cross et al. 2018 |
| scatter_a, scatter_b | 8.7436, 1.6618 | FITTED | Jacques 2013, Bergmann 2021 |
| denat_amplitude | 0.4 | SWEPT | No citable value; see Chapter 4 |
| denat_midpoint | 5.70 | CITED (proxy) | Cross et al. 2018 |
| denat_width | 0.28 | SWEPT | No citation found |
| mua_water | 6-band array | CITED | Hale and Querry 1973 |
| water_fraction | 0.732 | CITED | Wojtasik-Kalinowska 2016 |
| sensor_sigma | 0.0054 | MEASURED | Wang, Tang, Li and Chen 2026 |
| mu_a_baseline | 0.8 cm⁻¹ | TUNED | Uncited, disclosed |

Of the fourteen parameters, ten are cited, measured, or fitted to published data, and an eleventh is back-calculated from a cited source. The back-calculation concerns `c_Mb_sd`: the source reports 0.87 ± 0.005 mg/g without labelling the second term. A standard deviation of 0.005 mg/g across 599 animals would be implausibly tight for a biological trait, so the reported figure is almost certainly a standard error. Recovering the standard deviation as 0.005 × sqrt(599) gives approximately 0.12 mg/g, a coefficient of variation near 14%, which is biologically plausible. Three are not traceable to a published measurement, and the study treats them differently according to how well each is covered by other evidence.

The parameter `denat_amplitude` is swept. Chapter 4 reports performance across the full plausible range of this parameter, so the conclusion does not depend on which value within that range is correct.

The parameter `denat_width` is also swept in the generator, at 0.20, 0.28, 0.40 and 0.50, giving transitions that span 0.88 to 2.20 pH units from end to end. The operating value is 0.28. It was not carried through the baseline comparison reported in Chapter 4, and its effect was examined only against the linear baseline using a single-seed measurement method later found unreliable.

The two parameters were treated differently because they do not carry the same risk. In Equation 3.2 the denaturation amplitude sets how large the pH effect on scattering is: at an amplitude of zero the bracketed term is constant, pH no longer influences reflectance, and the prediction task has no signal to recover. The width governs only how steeply that same change is distributed across the pH axis, so at any width the full range of the effect is still traversed somewhere in the domain. Amplitude is therefore the parameter whose uncertainty could invalidate the conclusion, and it is the one carried through the full comparison. The width remains reported as a limitation rather than as a parameter that comparison covers.

The parameter `mu_a_baseline` is tuned and uncited. It was adopted at 0.8 because that value improved agreement with the external 970 nm reference, which means that comparison is not independent of it and cannot serve as free-standing validation of this parameter.

A literature-suggested value for `denat_amplitude` of 2.19 was tested and rejected. At that value the linear baseline reached an R² of approximately 0.94, breaking the non-triviality requirement the study imposes on itself, and agreement at 970 nm worsened. The adopted value of 0.4 is instead supported by the sweep reported in Chapter 4.

Each sample's mean pH was drawn uniformly across 5.35 to 6.45 rather than from the distribution observed in commercial pork, and the resulting field was clipped to 5.2 to 6.8. This is a coverage design rather than a claim of representativeness. Uniform sampling exercises the model across the whole span from PSE to DFD, which is how a measurement method is normally characterised across its operating range, whereas drawing from the population distribution would concentrate the data near the mean and leave both extremes barely represented. The consequence is that the benchmark's class proportions do not match those of commercial pork. Measured on the 500-sample test set, 43% of samples have a mean pH above 6.0, the upper anchor of the reference scale cited in Chapter 2. Chapter 4 reports what the headline metrics become when evaluation is restricted to the range that scale covers.

![](figures/fig_ph_distribution.png)
**Figure 3.4.** Distribution of pH between and within samples across the frozen dataset.

#### Computational Characteristics

**Table 3.2.** Computational characteristics of the trained network.

| Property | Value |
|---|---|
| Trainable parameters | 118,113 |
| Checkpoint size on disk | 0.50 MB |
| Hardware used for inference | CPU only |

These figures bear directly on the third objective. A network of this size imposes no unusual demand on the computing hardware a future device would require. Inference time is not reported, because it depends on the processor used and no timing measurement is committed as evidence.

#### Software

**Table 3.3.** Software requirements.

| Component | Version |
|---|---|
| Python | 3.11.5 |
| PyTorch | 2.13.0, CPU build |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.1 |
| scikit-learn | 1.9.1, for the PLSR baseline |
| OpenCV | 5.0.0.93, writes the RGB composites |
| Gradio | 6.28.0, for the inference interface |
| Pillow | 12.3.0, image handling for the interface |

Two further packages are pinned in the project's requirements file but are deliberately omitted from this table. The h5py library is needed only to read the external hyperspectral reference cubes, which are not redistributable and are not required to reproduce any result reported here, and reportlab and pypdf are used only to typeset this document. Neither is part of the system this table describes.

### Procedures

#### Data Generation

For each sample, a spatial pH field and a myoglobin redox state are generated independently. Each sample's mean pH is drawn uniformly from 5.35 to 6.45, and the resulting field is clipped to 5.2 to 6.8. Equations 3.2 and 3.3 convert these into optical coefficients, Equation 3.1 converts the coefficients into six-band reflectance, and a sensor noise model produces the final image. The noise standard deviation was measured from a real hyperspectral cube as a fraction of signal and is applied as an absolute value, which across the six bands is between two and four times the measured relative level. The benchmark is therefore noisier than the instrument it was measured from, which makes the reported accuracy conservative. The image is divided into quadrants and one pixel within the tissue mask is then drawn at random from each, giving four supervised points whose true pH is recorded. Drawing one per quadrant rather than four at random across the whole image guarantees that the supervised points are spread over the sample, which makes the reconstruction task easier than uniform random sampling would.

#### Dataset Composition

A frozen dataset of 400 samples was generated at a fixed random seed and partitioned into 300 training, 50 validation, and 50 test samples.

A further 500 samples were generated under a separate random seed and withheld entirely from training. Using a different seed matters here: the seed that produced the frozen dataset would reproduce its own draws, so a set generated with it could overlap the training data. The 500-sample set is therefore an independent draw rather than a further partition of the same generation, and it is the set on which the primary results in Chapter 4 are reported.

#### Model Training

**Table 3.4.** Training configuration.

| Setting | Value |
|---|---|
| Optimizer | Adam (Kingma and Ba, 2015) |
| Learning rate | 1 × 10⁻⁴, lowered from 1 × 10⁻³ after diagnosing instability |
| Batch size | 8 |
| Epochs | Maximum 50, early stopping with patience 10 |
| Loss | Equation 3.4, with lambda = 0.05 |
| Supervision | 4 sparse pH points per sample |
| Seeds | 5, independently trained |

Checkpoint selection uses validation loss only. Because neither term of Equation 3.4 reads the dense pH map, the map the network is evaluated against plays no part in which checkpoint is saved.

Five networks were trained from different random initialisations, and all reported results are the mean and standard deviation across those five runs. A single training run is one draw from a distribution of outcomes, and reporting the best of several would overstate what a replication would obtain. (Henderson et al., 2018)

An early instability in validation performance was traced to the learning rate rather than to batch normalisation, which was the first suspected cause. A separate error in how the coefficient of determination was aggregated, found later, had also overstated how severe that instability appeared. Both are corrected in the current code.

#### Model Inference

At inference the network receives only the six-band cube. It does not receive the four sparse points, which exist for training supervision only, and it does not receive any ground-truth pH. The output is a pH value for every pixel, masked to the tissue region for display.

![](figures/fig_pipeline.png)
**Figure 3.5.** The complete pipeline from data generation through training to inference and heatmap output.

#### Validation Procedures

Four checks were applied, each testing a different property of the benchmark or the modelling choices behind it.

**Sign test.** Simulated reflectance must fall as pH rises, in every band, since lower pH produces greater scattering and paler tissue. This is a pass or fail check on whether the forward model has the physical mechanism in the correct direction.

**Non-triviality check.** A pixel-wise linear regression fitted to raw band values must perform poorly. If a linear model could recover pH from the simulated reflectance, the network would demonstrate nothing. The threshold adopted is that linear R² must stay well below 0.9, applied as a hard constraint on parameter selection.

**Input validation by domain consultation.** No formal published protocol exists for validating this particular combination of modelling choices. The input parameters and key methodological decisions, including the use of equine myoglobin extinction data and the overall acceptability of a physics-based synthetic dataset for this application, were reviewed in consultation with a food technologist with relevant domain expertise. The consultant confirmed these choices are reasonable for the study's purpose, while noting that no standardised certification process exists for this use case. The consultation is documented separately and treated as a supporting check on plausibility, not as a substitute for the citations in Table 3.1 or for physical validation on real pork.

**External reference comparison.** Simulated reflectance at 970 nm was compared against a real hyperspectral reference sample from an independently collected dataset (Wang et al., 2026), this being the only wavelength at which the two imaging systems overlap. The result is reported in Chapter 5 as a limitation rather than as validation, for reasons given there.

### Evaluation Methods

Let y_i denote the true pH at pixel i, y_hat_i the predicted value, and n the number of pixels in the set under evaluation.

#### Mean Absolute Error

> MAE = (1 / n) · sum over i of | y_i - y_hat_i |  (3.6)

Mean absolute error expresses the typical error in pH units. It treats all deviations equally and does not change with the spread of pH present in the evaluation set, which makes it the most stable of the three error measures used here. (Hodson, 2022)

#### Root Mean Square Error

> RMSE = sqrt[ (1 / n) · sum over i of ( y_i - y_hat_i )² ]  (3.7)

Root mean square error penalises large deviations more heavily than small ones. A value substantially above the mean absolute error indicates that errors are unevenly distributed.

#### Coefficient of Determination

> R² = 1 - [ sum over i of ( y_i - y_hat_i )² ] / [ sum over i of ( y_i - y_bar )² ]  (3.8)

where y_bar is the mean of all pixels in the evaluation set. This pooled form answers how much of the total pH variation present across the whole set the model explains. A value of 1 indicates perfect prediction, and 0 indicates performance equal to predicting the overall mean everywhere.

The coefficient of determination is relative to the variance available to explain, so it moves with the sampling design rather than with accuracy alone. For this reason it is reported alongside mean absolute error and root mean square error rather than on its own.

#### Per-Sample Coefficient of Determination

> R²_sample = 1 - [ sum over i in S of ( y_i - y_hat_i )² ] / [ sum over i in S of ( y_i - y_bar_S )² ]  (3.9)

where S is the set of pixels in one sample and y_bar_S is the mean pH of that sample alone. Equation 3.9 differs from Equation 3.8 only in the reference mean, but it asks a different question: not whether the model estimates the overall pH level correctly, but whether it expresses the magnitude of variation within a single sample correctly. The two are reported separately in Chapter 4 because a single pooled figure conflates them.

#### Tolerance Band Accuracy

> A(t) = ( count of pixels where | y_i - y_hat_i | <= t ) / n × 100%  (3.10)

The proportion of pixels falling within a fixed tolerance of ground truth. This measure describes the predicted map directly, depends on no class definition and no class balance, and is reported at t = 0.05, 0.10, 0.15, and 0.20 pH.

#### Baseline Comparison Methods

Two classical regression methods are fitted directly on raw six-band reflectance and evaluated on the same held-out pixels as the network.

Linear regression fits a least-squares hyperplane from the six band values to pH.

Partial least squares regression with four components projects the six bands onto a smaller set of latent variables chosen to maximise covariance with the target, then regresses on those. (Wold, Sjostrom and Eriksson, 2001)

Both are reported alongside the network at every parameter setting tested. A configuration in which the linear baseline explains the large majority of pH variance is treated as insufficiently challenging and excluded from consideration.

#### Calibration Assessment

Two forms of magnitude miscalibration are measured separately, because they act in opposite directions and only one of them could be fitted in a physical deployment.

The first acts between samples. Regressing each sample's true mean pH on its predicted mean gives

> y_bar_S = alpha + beta · y_hat_bar_S  (3.11)

where beta greater than one indicates that predicted sample means are compressed toward the centre of the range. Fitting Equation 3.11 requires only an estimate of each calibration sample's mean pH, which sparse probe readings can supply. The correction reported in Chapter 4 is fitted from the mean of the four probe readings rather than from the true dense mean, so no quantity unavailable to a deployment enters the fit.

The second acts within a sample. The ratio of true to predicted within-sample standard deviation,

> gamma = SD( y_S ) / SD( y_hat_S )  (3.12)

is applied to each prediction's deviation from its own mean. Fitting Equation 3.12 requires the within-sample standard deviation of true pH, which a deployment holding only sparse probe readings does not have.

Both corrections are fitted on the validation split and applied to the 500-sample test set, which influenced neither training, nor checkpoint selection, nor the correction parameters. The network is not retrained and the saved checkpoints are read only.

## Theoretical Framework

### Meat Quality and Post-Mortem pH Changes

This study is grounded in the established relationship between post-mortem biochemical change and pork quality. Anaerobic glycolysis after slaughter converts glycogen to lactic acid, progressively lowering tissue pH, which affects colour, texture, protein stability, and water-holding capacity, and can produce pale, soft, exudative conditions under rapid decline [4, 7]. Because pH varies spatially across a sample and a point probe cannot capture that variation, a spatially resolved, non-invasive measurement approach is motivated independently of the acquisition method used [1].

### Optical and Spectral Behavior of Meat Tissue

As tissue pH falls, light scattering increases, producing paler meat at lower pH [4, 5]. Myoglobin redox state independently governs wavelength-dependent absorption [8]. The generator models these two pathways separately, in Equations 3.2 and 3.3, and combines them only at the point of computing reflectance, consistent with how the literature describes them as physically distinct mechanisms.

### Light Propagation and Multispectral Radiometry

Light incident on pork tissue is partially absorbed by chromophores and partially scattered by tissue microstructure, and the diffusely reflected fraction carries information about both processes [18]. The six wavelengths used were selected on the basis of myoglobin absorption features, the isosbestic point near 525 nm, and near-infrared moisture and scattering behaviour, as developed in Chapter 2.

### Deep Learning and Spatial pH Prediction

A convolutional neural network is used to learn spatial and spectral patterns directly from multispectral cubes, given the nonlinear relationship between tissue optical properties and biological structure [22, 24]. The network is trained under sparse supervision, using four ground-truth points per sample, reflecting the practical constraint that a real deployment could obtain only a small number of physical probe readings per sample.

### Simulation-Based Modeling

Where real data is unavailable, physics-based simulation is treated as a legitimate methodological choice rather than a compromise, provided every parameter's evidentiary status is disclosed. The model's physical parameters are assembled from multiple independently published sources, a practice established in comparable tissue-optics simulation work. Supporting literature for both points is developed in Chapter 2.

### Theoretical Assumptions

The following assumptions underlie the synthetic benchmark. They are stated explicitly so that the scope of the study's conclusions is clear.

1. Published extinction coefficients for myoglobin, and published absorption coefficients for water, apply to pork tissue at the six wavelengths used. The myoglobin values are equine, a substitution common in the tissue-optics literature and reviewed with the domain consultant.

2. The Kubelka-Munk two-flux model adequately describes diffuse reflectance from a semi-infinite scattering medium of this kind. The model does not represent surface specular reflection or subsurface layering. A radial illumination falloff is applied in the sensor model, but no other illumination geometry is represented.

3. pH-driven protein denaturation is the dominant pH-dependent influence on light scattering across the range modelled. Other pH-dependent effects on tissue optics are not represented.

4. Four probe readings per sample reflect a realistic constraint on what a physical deployment could obtain. One reading is drawn at random from within each quadrant of the tissue mask, so the four are spread across the sample rather than placed uniformly at random.

5. The simulated tissue is muscle throughout. Fat, connective tissue, and bone are not modelled.

## Notes

The following sources are cited in this chapter. The first group supplies the generator parameters listed in Table 3.1 and the reference scale the quality classes are drawn from; the second supports the models and methods described in the text. Renumber to match the project's citation style when merging with the master reference list, where entries shared with Chapter 2 will appear once.

**Sources of the generator parameters in Table 3.1**

Bergmann, F., Foschum, F., Marzel, L. and Kienle, A. (2021). Ex vivo determination of broadband absorption and effective scattering coefficients of porcine tissue. *Photonics*, 8(9), 365. doi:10.3390/photonics8090365

Bowen, W. J. (1949). The absorption spectra and extinction coefficients of myoglobin. *Journal of Biological Chemistry*, 179, 235-245. PMID 18119239.

Cross, A. J., King, D. A., Shackelford, S. D., Wheeler, T. L., Nonneman, D. J., Keel, B. N. and Rohrer, G. A. (2018). Genome-wide association of myoglobin concentrations in pork loins. *Meat and Muscle Biology*, 2(1), 189-196. doi:10.22175/mmb2017.08.0042

Hale, G. M. and Querry, M. R. (1973). Optical constants of water in the 200 nm to 200 micrometre wavelength region. *Applied Optics*, 12(3), 555-563. doi:10.1364/AO.12.000555. Values taken from the tabulated data file at omlc.org/spectra/water/data/hale73.dat.

Jacques, S. L. (2013). Optical properties of biological tissues: a review. *Physics in Medicine and Biology*, 58(11), R37-R61. doi:10.1088/0031-9155/58/11/R37

Tang, J., Faustman, C. and Hoagland, T. A. (2004). Krzywicki revisited: equations for spectrophotometric determination of myoglobin redox forms in aqueous meat extracts. *Journal of Food Science*, 69(9), C717-C720. doi:10.1111/j.1365-2621.2004.tb09922.x

Wang, M., Tang, J., Li, S. and Chen, W. (2026). NHSI-meat-overtime: a near-infrared hyperspectral dataset of five meat types over storage time. *IET Conference Proceedings*, CP987, 208-212. doi:10.1049/icp.2026.2763

Wojtasik-Kalinowska, I., Guzek, D., Gorska-Horczyczak, E., Glabska, D., Brodowska, M., Sun, D.-W. and Wierzbicka, A. (2016). Volatile compounds and fatty acids profile in Longissimus dorsi muscle from pigs fed with feed containing bioactive components. *LWT - Food Science and Technology*, 67, 112-117. doi:10.1016/j.lwt.2015.11.023

Sristi, P. R., Das, N. R., Akhter, A., Kaniya, N. M. and Hashem, M. A. (2025). Relation among meat pH, color and tenderness: a review. *Meat Research*, 5(3). doi:10.55002/mr.5.3.117

**Sources for the models and methods**

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1). Preprint freely available at arXiv:1709.06560.

Hodson, T. O. (2022). Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not. *Geoscientific Model Development*, 15(14), 5481-5487. doi:10.5194/gmd-15-5481-2022. Open access.

Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *3rd International Conference on Learning Representations*. Freely available at arXiv:1412.6980.

Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)*, 234-241. doi:10.1007/978-3-319-24574-4_28. Preprint freely available at arXiv:1505.04597.

Rudin, L. I., Osher, S. and Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1-4), 259-268. doi:10.1016/0167-2789(92)90242-F.

Thennadil, S. N. (2008). Relationship between the Kubelka-Munk scattering and radiative transfer coefficients. *Journal of the Optical Society of America A*, 25(7), 1480-1485. doi:10.1364/JOSAA.25.001480. Accepted manuscript freely available at strathprints.strath.ac.uk/13710/.

Wold, S., Sjostrom, M. and Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130. doi:10.1016/S0169-7439(01)00155-1.

---

# CHAPTER 4
# RESULTS

This chapter presents the trained network's performance on the synthetic benchmark described in Chapter 3, organised according to the study's three objectives: the accuracy of pixel-wise pH prediction, comparison against classical regression baselines across the disclosed parameter range, and demonstration of the pipeline operating end to end.

## Validation of the Synthetic Benchmark

Two checks were applied to the generated dataset before any network was trained.

The first confirmed the physical direction of the forward model. Simulated reflectance decreased monotonically with increasing pH across all six bands, consistent with the reduction in light scattering that accompanies decreasing protein denaturation. This sign test passed.

The second established that the inverse problem was non-trivial. A pixel-wise linear regression fitted to raw band values attained an in-sample R² of 0.6509 ± 0.0180 across ten independent draws, comfortably below the 0.9 ceiling adopted for this study. This is the generator's own self-test, fitted and scored on the same throwaway draw. Held out, the same linear model reaches 0.6504 on the 500-sample test set, reported in Table 4.6, and 0.6212 on the smaller validation split used for the amplitude sweep in Table 4.7. The three figures answer different questions and are not interchangeable. The mapping from reflectance to pH could not be recovered by linear means alone.

## Training Behaviour

All five training runs converged. Sparse-point training loss decreased from approximately 0.097 at initialisation to between 0.009 and 0.020 at the selected checkpoint.

Validation loss decreased in parallel during early epochs before diverging. Early stopping on validation loss consistently selected a checkpoint before that divergence, at epochs 9, 14, 21, 10, and 18 across the five seeds. The divergence that followed was pronounced. In the most marked case, validation loss rose from 0.0210 at the selected epoch to a peak of 0.4610 in the following epoch, twenty-two times the value at the checkpoint, and had settled to 0.1161 by the time training terminated. Figure 4.1 shows that peak; the selected checkpoint precedes it.

At the selected checkpoints the ratio of validation to training loss ranged from 1.15 to 1.89, with a mean of 1.61. Ratios in this range indicate the network had not overfit the training partition at the point of selection.

![](figures/fig_loss_curves.png)
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
| 1 | 0.890 | 0.080 | 0.109 |
| 2 | 0.860 | 0.083 | 0.123 |
| 3 | 0.839 | 0.096 | 0.132 |
| 4 | 0.868 | 0.088 | 0.120 |
| Mean ± SD | 0.847 ± 0.038 | 0.094 ± 0.015 | 0.128 ± 0.016 |

Seed 0 was a consistent low outlier across all three metrics, a pattern replicated in independent runs including on a second machine. It is reported rather than excluded, and the aggregate statistics reflect its influence.

The coefficient of determination is scale-relative, so its value depends on the spread of pH present in the set it is measured on. Restricting the 500-sample set to the 284 samples whose mean pH falls at or below 6.0, the upper anchor of the reference scale cited in Chapter 2, reduces the standard deviation of true pH from 0.330 to 0.218 and the pooled R² from 0.847 to 0.705. Over the same restriction mean absolute error improves from 0.094 to 0.085 pH units and root mean square error from 0.128 to 0.117. The model is therefore slightly more accurate in absolute terms on the narrower range, and the lower R² reflects the smaller variance available to explain rather than worse prediction. This is why absolute error in pH units is reported alongside R² throughout this chapter.

## Spatial Behaviour of the Predicted Maps

A pooled coefficient of determination does not indicate whether the predicted heatmap is usable at the pixel level. Measured by Equation 3.10, 81.5% of pixels fell within ±0.15 pH of ground truth and 89.3% within ±0.20 pH.

As supporting evidence, predicted pixels were assigned to PSE, normal, or DFD quality classes. The boundaries used were 5.4 and 5.8 pH. The reference scale cited in Chapter 2 gives anchor values for the three conditions, at 5.2, 5.6 and 6.0, rather than the limits between them (Sristi et al., 2025), so the boundaries applied here are the midpoints between consecutive anchors. Pixels landed in the correct class 87.7% of the time, against 62.6% for a classifier that always guessed the most common class, a lift of 25.1 points. Accuracy at the level of the whole image is carried by the most common class. Recall per class, averaged over the five seeds, was 29.0% for PSE, 82.1% for normal and 96.4% for DFD, and PSE recall varied from 0.5% to 62.3% between seeds. The model therefore identifies elevated pH far more reliably than depressed pH. This follows from the compression reported in the previous section: predictions are pulled toward the centre of the range, which moves the lowest pH pixels across the PSE boundary before it moves any other class. Applying the level correction reported in the following section, fitted from the four probe readings alone, raises PSE recall from 29.0% to 69.3% and overall class accuracy from 87.7% to 89.9%, at the cost of 0.6 points of DFD recall. It also narrows the between-seed spread in PSE recall, from 0.5 to 62.3% before correction to 57.6 to 75.7% after it. For the checkpoint used in the demonstration the recovery is from 5.4% to 68.7%. The deficit is therefore a consequence of the magnitude miscalibration rather than an inability to detect the condition. The figure is reported as supporting evidence only, since the class boundaries are interpolated rather than directly published. The pork quality literature reports no agreed criterion, so the result was recomputed under three further conventions: PSE below 5.5 with DFD above 6.1, the same with DFD above 6.2, and the anchors of the cited scale, 5.2 and 6.0, applied directly as limits. Accuracy ranged from 81.5% to 90.5% and the lift over the majority-class baseline from 18.0 to 33.8 points. The advantage over guessing the largest class therefore does not depend on where the boundaries are placed.

Neither Equation 3.8 nor Equation 3.9 indicates whether the model locates elevated pH correctly, since both measure the magnitude of predicted variation rather than its position. Localisation was therefore measured directly, restricted to the 282 of 500 test samples that genuinely spanned more than one quality class.

**Table 4.4.** Spatial localisation on multi-class samples.

| Measure | Result | Chance level |
|---|---|---|
| Per-pixel class accuracy | 79.0% ± 2.9% | not applicable |
| Overlap of predicted and true hottest 20% | 51.3% ± 4.3% | 20% |
| Predicted hot region genuinely hotter | 99.6% ± 0.5% | 50% |

The overlap measure is the demanding one, since it requires the predicted hot region to coincide with the true hot region rather than merely point in the right direction. At 51.3% against a 20% chance level it is well above chance while showing that pixel-level precision remains partial. The 99.6% figure tests direction alone and is reported for completeness rather than as the headline localisation result.

![](figures/fig_heatmap_example.png)
**Figure 4.2.** Hidden true pH map, network prediction, and absolute error for one held-out sample. The sample shown is the median case by per-sample coefficient of determination within its split, selected by that rule rather than by inspection.

Mean per-sample R², computed by Equation 3.9, was -2.17 on the 500-sample test set and negative on the large majority of individual samples for every seed. A negative value indicates that the predicted map for a given sample scored worse, on this magnitude-sensitive measure, than a flat map drawn at that sample's own average pH.

This coexists with the pooled and localisation results because between-sample pH variation, with a standard deviation of 0.317, was approximately 3.6 times larger than within-sample variation at 0.088. Equation 3.8 is accordingly dominated by the model's reliable estimation of each sample's overall level, while Equation 3.9 measures accuracy against the much smaller within-sample scale, on which even correctly located errors appear severe.

Diagnostics confirm this is a magnitude-calibration problem rather than an absence of spatial signal. Predicted maps correlated with ground truth at r = 0.536 ± 0.056, indicating genuine spatial structure was recovered, while the ratio of predicted to true spatial standard deviation was 1.253 ± 0.097, an over-amplification of approximately 25%. The two components of this miscalibration, and what correcting each recovers, are reported in the next section.

## Calibration of the Predicted Maps

The magnitude error resolves into two separate miscalibrations acting in opposite directions. Measured by Equation 3.11, predicted sample means are compressed toward the centre of the range, with a recovery slope of 1.285 ± 0.130, equivalent to the predictions spanning about 0.78 of the true between-sample range. Measured by Equation 3.12, variation within a sample is over-expressed, with a recovery scale of 0.779 ± 0.056 on the validation split where it was fitted, equivalent to about 1.28 times the true within-sample standard deviation there. Reporting only the second would describe the error incompletely.

Compression toward the centre of the training range is the expected behaviour of an under-fitted model, and it follows from the design stated in Chapter 3: a deliberately small network supervised on four points per sample under a strong smoothness prior. The direction matters for interpretation. Compression costs accuracy rather than flattering it, so the uncorrected figures in Table 4.1 understate what the predictions contain rather than overstating it.

**Table 4.5.** Effect of post-hoc calibration, fitted on the validation split and applied to the 500-sample test set. Values are five-seed means.

| Correction | R² | MAE (pH) | RMSE (pH) | Per-sample R² | Samples with positive per-sample R² |
|---|---|---|---|---|---|
| None | 0.847 | 0.094 | 0.128 | -2.170 | 29.5% |
| Level only | 0.904 | 0.075 | 0.102 | -0.946 | 39.7% |
| Texture only | 0.870 | 0.087 | 0.118 | -1.616 | 38.0% |
| Level and texture | 0.927 | 0.067 | 0.089 | -0.393 | 52.9% |

The level correction accounts for 71.6% of the total gain in R², and it is the component a physical deployment could fit. It was fitted twice: once from each calibration sample's true dense mean, and once from the mean of its four probe readings, which is all a deployment would have. The two agree to within 0.0001 in R² (0.9043 against 0.9044), so the correction does not depend on information a deployment lacks. On the level correction alone, R² rises from 0.847 to 0.904 and mean absolute error falls from 0.094 to 0.076 pH units.

The texture correction requires the within-sample standard deviation of true pH, which sparse probe readings cannot provide. Its contribution is reported as a diagnostic bound rather than as a deployable result, and the two are not combined into a single headline figure.

Per-sample R² improves from -2.170 to -0.393 under the full correction, and the share of samples with a positive value rises from 29.5% to 52.9%. It does not become positive on average. A per-sample oracle rescaling, which uses each sample's own true standard deviation, would reach approximately +0.297, but that quantity is unavailable at prediction time and the figure is therefore not attainable in deployment. The correction reported here is fitted on a separate split and reaches -0.393.

## Comparison with Baseline Methods

To confirm that the task required more than a simple relationship between raw reflectance and pH, the network was compared against the two baselines described in Chapter 3, both fitted and evaluated on the same held-out pixels.

On the 500-sample held-out set, the same set the headline in Table 4.1 is measured on, the network outperformed both baselines by a margin of 0.197 in R² over linear regression and 0.197 over partial least squares.

**Table 4.6.** Network against the classical baselines on the 500-sample held-out set. Baselines were fitted on the training split and scored on 200,000 tissue pixels drawn from the test set at 400 pixels per sample, the sampling convention used for every baseline figure in this study. The network is scored on all 18.4 million tissue pixels of the same samples. The pixel sets differ in size but are drawn from the same images, and the baseline sample is unbiased with respect to them.

| Method | R² | MAE (pH) | RMSE (pH) |
|---|---|---|---|
| Linear regression | 0.6504 | 0.1560 | 0.1951 |
| Partial least squares, four components | 0.6500 | 0.1561 | 0.1952 |
| Convolutional regression network | 0.8472 | 0.0936 | 0.1282 |

The network reduced mean absolute error from 0.1560 to 0.0936 pH units against the stronger of the two baselines, a reduction of 40%, and root mean square error from 0.1951 to 0.1282, a reduction of 34%.

Partial least squares regression converged to almost exactly the same solution as ordinary linear regression. A singular value decomposition of the six-band design matrix explains why. A single component carries 81.0% of its variance and the first four carry 97.31%, so the six measurements vary largely together and a four-component model already spans nearly the whole space a six-band linear fit can use.

Because `denat_amplitude` has no established literature value, the comparison was repeated across the full disclosed range of plausible values at matched sample scale.

**Table 4.7.** Network advantage over classical baselines across the denat_amplitude sweep.

| denat_amplitude | Linear | PLSR | CRN (mean ± SD) | Margin |
|---|---|---|---|---|
| 0.2 | 0.3113 | 0.3110 | 0.7270 ± 0.0820 | +0.416 |
| 0.4 (adopted) | 0.6212 | 0.6206 | 0.8486 ± 0.0368 | +0.227 |
| 0.6 | 0.7616 | 0.7608 | 0.8862 ± 0.0139 | +0.125 |
| 0.8 | 0.8308 | 0.8299 | 0.8999 ± 0.0169 | +0.069 |

Values in this table are measured on the validation split of each amplitude's own dataset, using the same split for all three methods so the comparison is like-for-like. The network column rests on five seeds at the adopted value of 0.4 and on three seeds at each of the other three amplitudes. The amplitudes other than 0.4 were never evaluated on the 500-sample set, so validation is the only scale on which all four points are comparable. This is why the network column reads 0.8486 at the adopted setting while the headline in Table 4.1 is 0.847.

![](figures/fig_sweep_comparison.png)
**Figure 4.3.** Network against baselines across the sweep, with error bars on the network series.

The network outperformed both baselines at every tested amplitude. The margin over the stronger baseline ranged from 0.069 at an amplitude of 0.8, where the task is most linearly separable, to 0.416 at 0.2, where it is least. The margin narrowed as amplitude increased because the underlying pH-scattering relationship became more linearly separable, and at 0.8 the linear baseline alone reached 0.83, close to the non-triviality ceiling of 0.9. This narrowing pattern, rather than the size of the margin at any single setting, was the basis for retaining 0.4 as the operating value.

## Pipeline Demonstration

The third objective was addressed by running the complete software pipeline end to end and exposing its inference stage through an interface that accepts a six-band sample and returns a predicted pH map.

![](figures/fig_system_output.png)
**Figure 4.4.** The inference stage of the pipeline: a six-band sample is supplied, and a predicted pH map is produced from spectral reflectance alone.

The interface loads one fixed checkpoint and performs inference on samples drawn from the 500-sample held-out set, the same set on which Table 4.1 is reported. The checkpoint used is seed 3, whose held-out R² of 0.839 is the closest of the five to the reported mean of 0.847. Selecting on that basis rather than taking the best-performing seed, which reached 0.891, means the interface does not demonstrate accuracy higher than the study reports.

Inference runs on CPU without a graphics unit. No ground-truth pH is read at any point; the dense map is used only to determine which pixels are tissue.

Every result reported in this chapter was produced by running the pipeline in full. The frozen dataset was generated by the generator described in Chapter 3, the network was trained on it under the sparse-supervision protocol of Equation 3.4, and the resulting checkpoints were used for inference on held-out samples. No step required manual intervention or data unavailable from the pipeline itself.

This demonstrates that the software components of LIGTAS-pH function as a coherent system from input to final output. It does not demonstrate operation on physically acquired data, which remains future work. Because the training and evaluation methodology is independent of the network backbone, that transition would require substituting the data source rather than redesigning the pipeline.

## Notes

Every figure reported in this chapter is produced by a script in the repository and stored in a file named below, so each can be regenerated and checked independently.

| Reported in | Evidence file | Produced by |
|---|---|---|
| Tables 4.1 to 4.3, tolerance bands | `RESULTS.json`, `metric_check_outputs/final_evaluation_TEST500.json` | `evaluate_heatmap.py`, `collect_results.py` |
| Training behaviour, Figure 4.1 | `crn_5seed_final/seed_*/history.json` | `train_crn.py` |
| Table 4.4, localisation | `metric_check_outputs/spatial_localization.json` | `check_spatial_localization.py` |
| Per-sample R-squared, amplitude ratio, correlation | `metric_check_outputs/spatial_skill_official.json` | `check_spatial_skill.py` |
| Table 4.5, calibration | `metric_check_outputs/calibration.json`, `calibration_probe.json` | `check_calibration.py`, `check_calibration_probe.py` |
| Table 4.6, baselines on the 500-sample set | `metric_check_outputs/baselines_test500.json` | `compute_test500_baselines.py` |
| Table 4.7, amplitude sweep | `deconfound_outputs/full_table.json` | `deconfound_full_scale.py` |
| Band variance shares | `metric_check_outputs/band_collinearity.json` | `check_band_collinearity.py` |
| Range restriction | `metric_check_outputs/range_restriction.json` | `check_range_restriction.py` |
| Class boundaries across conventions | `metric_check_outputs/class_thresholds.json` | `check_class_thresholds.py` |
| Boundary artifact | `metric_check_outputs/edge_effect.json` | `check_edge_effect.py` |

Sources cited in this chapter appear in the reference list for Chapter 3.

---

# CHAPTER 5
# SUMMARY, FINDINGS, CONCLUSIONS, AND RECOMMENDATIONS

## Summary

This study examined whether a convolutional regression network, trained on four sparse pH readings per sample together with a spatial smoothness prior, could recover a useful spatially resolved estimate of pork pH from six-band multispectral reflectance.

The multispectral imaging hardware originally intended to supply training data failed before collection began. With the adviser's approval, the study was restructured around a physics-based synthetic benchmark in which every generator parameter is traced to a published source or explicitly disclosed as swept or tuned. The forward model drives pH through scattering and myoglobin redox state through absorption as independent pathways, combining them only at the point of computing reflectance, so that the resulting task cannot be solved by brightness alone.

A frozen dataset of 400 samples was generated and partitioned into training, validation, and test splits, with a further 500 samples generated under a separate seed and withheld entirely from training. Five independently seeded networks were trained under sparse supervision and evaluated on all three sets, alongside linear regression and partial least squares regression baselines fitted on the same data. The comparison was repeated across the full plausible range of the study's least-certain physical parameter, and the complete pipeline was then run end to end through to inference.

## Findings

**1. The synthetic benchmark behaves as a physics-based model should, and poses a non-trivial problem.** Simulated reflectance fell monotonically with rising pH across all six bands, matching the known scattering mechanism. A pixel-wise linear fit on raw band values reached only 0.6509 ± 0.0180 over ten draws, well below the 0.9 ceiling adopted for this study, confirming that the mapping from reflectance to pH cannot be recovered by linear means.

**2. The network predicted pixel-wise pH accurately on data it had never seen.** On 500 unseen samples across five independently trained seeds, it achieved an R² of 0.847 ± 0.038, a mean absolute error of 0.094 ± 0.015 pH units, and a root mean square error of 0.128 ± 0.016. At the pixel level, 81.5% of predictions fell within ±0.15 pH of ground truth and 89.3% within ±0.20 pH. Results agreed across all three evaluation sets, with the validation split showing no inflation relative to the two sets that never influenced training.

**3. The network outperformed both classical baselines, and that advantage did not depend on the study's least-certain parameter.** On the 500-sample held-out set, the set the headline is measured on, it reached an R² of 0.8472 against 0.6504 for linear regression and 0.6500 for partial least squares, a margin of 0.197, and reduced mean absolute error from 0.1560 to 0.0936 pH units. Repeating the comparison across the full disclosed range of denat_amplitude showed the network ahead at every setting, by margins from 0.069 to 0.416. The two baselines converged to nearly the same solution because the six measurements vary largely together: a single component of the design matrix carries 81.0% of its variance and the first four carry 97.31%.

**4. The model locates spatial variation reliably but does not yet calibrate its magnitude.** On the 282 samples spanning more than one quality class, per-pixel class accuracy was 79.0% ± 2.9% and the predicted hottest 20% of pixels overlapped the true hottest 20% in 51.3% ± 4.3% of cases, against a 20% chance level. Mean per-sample R² was nonetheless negative at -2.17, because within-sample pH variation is roughly 3.6 times smaller than between-sample variation, and the model over-expresses variation within a sample by about 25% while compressing differences between samples to about 0.78 of their true range. Predicted maps correlated with ground truth at r = 0.536 ± 0.056, confirming that genuine spatial structure was recovered. A post-hoc correction fitted on the validation split and applied to the held-out 500 samples, using only each calibration sample's mean pH, raised R² from 0.847 to 0.904 and reduced mean absolute error from 0.094 to 0.075 pH units. Correcting the within-sample component as well reaches R² 0.927 and per-sample R² -0.393, but it requires dense ground truth and is not deployable.

**5. The complete software pipeline runs end to end.** Data generation, training, inference, and heatmap output operate as one sequence with no manual intervention, producing a pH map on CPU, without a graphics unit, from a network of 118,113 parameters.

## Conclusions

**1.** A physics-based synthetic benchmark, with every parameter either cited or explicitly disclosed, provides a workable substitute for physically collected data when hardware is unavailable, provided the claims drawn from it are confined to behaviour under the simulated conditions. The benchmark constructed here is internally consistent with the optical mechanisms described in the literature and is not solvable by trivial means.

**2.** A convolutional regression network trained on four sparse points per sample can recover a dense pixel-wise pH field from six-band reflectance to within approximately 0.09 pH on this benchmark, with a substantial and robustly confirmed advantage over classical regression methods. Because that advantage holds across the entire plausible range of the study's least-certain parameter rather than at one chosen value, the conclusion does not rest on a parameter the literature cannot settle.

**3.** The model's spatial capability is better described as reliable localisation with imperfect magnitude calibration than as a single level of spatial accuracy. It identifies where pH is elevated within a sample, overstates how much variation there is inside a sample, and understates how far samples differ from one another. Both components were measured, and the one a deployment could fit was corrected and reported, which makes this a characterised limitation rather than an open failure.

**4.** The software pipeline functions as a coherent system and is structured so that the transition to physical data would require substituting the data source rather than redesigning the pipeline. This establishes a working foundation for future hardware integration. It does not establish validated predictive accuracy on real pork, which the study neither claims nor attempted.

## Recommendations

**1. Physical validation on real pork, once suitable imaging hardware becomes available.** This is the necessary step to move beyond a feasibility demonstration. Because the network's weights were learned on simulated reflectance, the realistic path is fine-tuning from the current weights on real samples with laboratory-measured pH, followed by re-validation, rather than training from scratch.

**2. Fitting the level calibration on physical samples.** The deployable half of the calibration correction raised R² from 0.847 to 0.904 on the synthetic benchmark using only the mean pH of each calibration sample. Confirming that the same correction holds on real tissue requires a modest set of physical samples with probe readings rather than new imaging hardware, which makes it the most tractable improvement available. The within-sample component cannot be fitted without a dense reference map and is not proposed.

**3. Sourcing or further constraining denat_width and mu_a_baseline.** Both remain outside the evidence base. The first was not covered by the baseline comparison, and the second was tuned against the same external reference it would otherwise help validate. A published measurement for either would remove a disclosed weakness.

**4. Extending the output for end users.** The pipeline currently returns a continuous pH map. A view that classifies regions into quality categories and reports their proportions would be more directly usable by inspection personnel. This was not implemented because the class boundaries available to this study are interpolated from reference anchors rather than taken from a published standard, and a classified view would make those boundaries load-bearing in a way the present reporting does not.

**5. Establishing a tolerance benchmark from the literature.** Accuracy is reported here against fixed tolerance bands chosen for their interpretability rather than against a published acceptability threshold for pork pH measurement. Locating such a threshold would let future work state not only how accurate the system is, but whether that accuracy is sufficient for the intended use.

## Limitations

**No real-pork validation.** This is the study's principal limitation. Every result reported here describes behaviour on a synthetic benchmark. The only external comparison available was at 970 nm, the single wavelength where this study's band design overlaps an independently collected hyperspectral dataset, and there the simulated reflectance read approximately 22 to 34% above the real reference. That comparison is exploratory, covers one wavelength, and is not independent of the tuned mu_a_baseline parameter, so it should not be read as validation.

**Elevated error at tissue boundaries.** Prediction error within an 8-pixel band along the tissue edge was 0.156 pH against 0.070 pH in the interior, a ratio of 2.31, present in 97% of samples. A convolutional network has less surrounding tissue to draw on at the mask edge. For applications concerned with the body of a cut rather than its perimeter, the interior figure is the more representative one.

**Two parameters outside the sweep's coverage.** The parameter denat_width is swept in the generator but was not included in the baseline comparison reported in Chapter 4, and mu_a_baseline is tuned and uncited. Neither is hidden, but neither carries the same evidentiary support as the parameters in Table 3.1 marked CITED or MEASURED.

**The benchmark's pH distribution is not representative of commercial pork.** Sample mean pH was drawn uniformly across 5.35 to 6.45 so that the model would be exercised across the full span from PSE to DFD. The result is that 43% of samples exceed 6.0, the upper anchor of the reference scale cited in Chapter 2. The benchmark therefore characterises the method across its operating range rather than estimating performance on a representative population, and the headline coefficient of determination should be read with that in mind. Chapter 4 reports the metrics restricted to the cited range.

**Assumptions stated but not tested.** The five assumptions listed in Chapter 3 are declared rather than verified within this study. The use of equine myoglobin extinction coefficients and the exclusion of fat and connective tissue from the simulated samples both constrain how far these results should be read as describing real pork.

**Corrections made during verification.** Two errors affecting reported figures were identified and corrected during the project team's own review before external assessment. An error in metric aggregation, in which validation R² had been computed per mini-batch and averaged rather than pooled as the baselines were, understated the headline figure; the corrected value is reported throughout. The rationale originally given for generating a 500-sample evaluation set, that a larger set would narrow the reported error bars, was found not to hold, since those bars reflect variance across training seeds rather than evaluation-sample size. The set is retained for the independence reason given in Chapter 3.
