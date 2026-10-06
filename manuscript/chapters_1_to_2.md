# CHAPTER 1
# INTRODUCTION

This chapter presents the background of the study, the statement of the problem, the objectives, the significance, the scope and limitations, and the project dictionary. It sets out the limitations of current pork quality assessment and the approach this study takes in response.

## Background of the Problem

Meat quality assessment is a central concern of food safety systems. The World Health Organization's first global estimates of foodborne disease, using 2010 as the reference year, attribute 600 million illnesses and 420,000 deaths annually to contaminated food, with the burden falling disproportionately on low- and middle-income countries (World Health Organization, 2015). In the Philippines, where pork is the principal source of animal protein, meat safety remains a recurring concern, and inspection under the Philippine Meat Inspection Code (Republic Act No. 9296) continues to rely substantially on visual and tactile assessment. Laboratory methods such as spear-tip pH probes are valid but destructive, slow, and confined to a single point of measurement.

The pH of meat is an established indicator of post-mortem quality. A normal decline to approximately 5.4 to 5.8 indicates sound muscle-to-meat conversion, while departures from that range indicate defects such as pale, soft, exudative (PSE) or dark, firm, dry (DFD) conditions (Sristi et al., 2025). Non-destructive optical methods, and hyperspectral imaging (HSI) in particular, have been used to predict pork quality attributes including pH, colour, and drip loss (Qiao et al., 2006; Xie et al., 2024). Yao et al. (2019) demonstrated non-invasive pH monitoring of meat using a low-cost portable hyperspectral scanner, though the measurement remained a point reading rather than a spatial one. Full hyperspectral systems remain costly and computationally demanding, which limits their deployment outside controlled settings. Multispectral imaging (MSI) offers a more economical alternative by recording a limited set of wavelengths selected to carry the information the task requires (Bandara et al., 2018; Ma et al., 2022).

Deep learning methods, and convolutional neural networks in particular, have outperformed conventional chemometric models in extracting spectral features for food quality prediction (Xiao et al., 2025; Lun et al., 2025). Compact architectures have made inference on constrained devices practical without reliance on remote computation (Zhang et al., 2022).

Beyond predicting a single average value, imaging-based approaches can reveal how pH is distributed across a sample. Conventional probes, whether spear-tip or flat-surface, measure only at the point of contact, which makes heterogeneity within a single cut difficult to assess (Yao et al., 2019). pH does vary across a muscle, according to local glycogen depletion, the rate of post-mortem temperature decline, and fibre type composition (Spires et al., 2023). This variation matters for grading, since one probe reading may miss a localized PSE or DFD region that affects the usability of the cut. Barbin et al. (2012) applied pixel-wise regression coefficients from partial least squares models to near-infrared hyperspectral images of pork longissimus dorsi, producing two-dimensional maps of pH, drip loss, and colour across the sample surface. Tang et al. (2023) similarly noted that conventional spectroscopic instruments return only a sample average, and demonstrated pixel-level prediction and visualization of pork quality traits by hyperspectral imaging. These results establish the practical value of a pH map over a single reading, which is the motivation for the LIGTAS-pH design.

This study concerns LIGTAS-pH, a multispectral imaging system intended to estimate and visualize the spatial distribution of pH in pork non-invasively. In its intended configuration the system comprises an optical acquisition module with an integrating dome, a monochrome camera, and six narrowband LED sources at wavelengths selected for sensitivity to pH-related spectral change in pork tissue. That module was intended to supply multispectral image stacks to a Convolutional Regression Network (CRN), which maps spectral reflectance to continuous pH values and produces a two-dimensional heatmap.

The imaging hardware became inoperable before data acquisition began, and a replacement could not be obtained within the timeframe of this study. On the recommendation of the thesis adviser, the study was redirected toward the development and evaluation of the underlying software pipeline, comprising a physics-based synthetic data generation process and a convolutional regression architecture, using a literature-constrained synthetic benchmark in place of physically acquired multispectral data. The chapters that follow describe both the originally intended hardware configuration and the methodology actually carried out under this constraint.

## Statement of the Problem

Post-mortem pH is a primary indicator of pork quality, reflecting the biochemical changes that follow slaughter and affecting colour, water-holding capacity, texture, and economic value. The conventional measurement uses a probe, which returns a single point reading and requires physical penetration of the meat. Neither property suits the assessment of variation across a cut. Multispectral imaging is a non-invasive alternative capable of measuring across the sample surface.

Despite advances in reflectance-based pH estimation, there remains no portable, cost-effective system suited to the pork loin that provides continuous, spatially resolved pH assessment. There is also no published benchmark against which such a system's software can be developed and evaluated when physical acquisition is unavailable.

## Objectives of the Study

### General Objective

To design and develop LIGTAS-pH, a multispectral imaging-based system for pixel-wise spatial pH estimation of pork, comprising a physics-based synthetic data generation pipeline and a Convolutional Regression Network developed and evaluated on a literature-constrained synthetic benchmark.

### Specific Objectives

1. To develop a physics-based synthetic dataset and generate pixel-wise spatial pH distribution maps for unseen synthetic samples using the trained Convolutional Regression Network.

2. To evaluate the predictive accuracy of the Convolutional Regression Network against linear regression and Partial Least Squares Regression (PLSR) baselines.

3. To demonstrate that the complete pipeline operates end to end, from data generation through training to inference and heatmap output.

## Significance of the Study

LIGTAS-pH is intended to provide value to those involved in meat quality assessment, food safety, computer science, and public health. Because the output is a predicted two-dimensional pH map rather than a single reading, its significance lies in representing pH variation across the meat surface.

This study is grounded in a synthetic, literature-constrained benchmark. The significance described below therefore distinguishes two things: what the study contributes now, as a documented and evaluated software methodology, and what the approach could offer if it were later validated against real pork. The second is stated as a prospect, not as a result of this study.

**Meat inspectors and quality assurance personnel.** This study is significant to meat inspectors and quality assurance personnel because LIGTAS-pH is intended, upon physical implementation, to provide a non-contact screening tool for identifying localized pH variations in pork meat. Traditional pH probes provide only point-based measurements, while the proposed system aims to generate a two-dimensional pH heatmap that shows which areas of the pork sample may have lower, normal, or higher pH values. This can help inspectors and quality control staff decide which meat samples or specific regions require closer inspection or confirmatory testing.

**Meat processing facilities and slaughterhouses.** For meat processing facilities and slaughterhouses, LIGTAS-pH may help improve quality control by providing a faster and less destructive way of assessing pork pH distribution. Since pH affects meat colour, texture, water-holding capacity, and quality defects such as PSE and DFD, the generated pH map can help personnel monitor product consistency and identify abnormal areas in a meat sample. The system is not intended to replace laboratory testing, but it may serve as an additional decision-support tool in routine quality assessment.

**Public health and food safety.** This study may contribute to public health and food safety by supporting earlier detection of pork quality irregularities through non-invasive pH mapping. Since the system does not require repeated probe insertion during screening, it may reduce unnecessary physical contact with the meat and lower the risk of cross-contamination compared with repeated invasive probing. However, pH mapping alone cannot fully determine microbial spoilage or food safety. LIGTAS-pH should therefore be viewed as a screening tool that provides additional evidence for meat quality assessment and helps identify samples that may require further microbial or laboratory confirmation.

**National Meat Inspection Service, Department of Agriculture, and regulatory agencies.** Government and regulatory agencies such as the National Meat Inspection Service and the Department of Agriculture may benefit from this study as a possible basis for future modernization of meat inspection support tools. The proposed system demonstrates how imaging and machine learning can be used to supplement conventional visual and probe-based inspection methods. If further developed and validated, the system may assist in field screening, documentation, and data-driven assessment of pork quality, while still operating within the limits of existing official inspection and laboratory procedures.

**Researchers.** The study provides experience in physics-based simulation of optical measurement, sparse-to-dense regression, model validation under limited supervision, and the disciplined documentation of parameter provenance. It also provides direct experience of adapting a research design when a planned instrument becomes unavailable, and of reporting the resulting limitations explicitly.

**Computer science and artificial intelligence community.** This study contributes to the computer science and artificial intelligence community by applying deep learning regression to a practical food quality assessment problem. Instead of focusing only on classification, such as fresh or spoiled, this study addresses continuous spatial regression, where the model predicts numerical pH values across the meat surface. It may serve as a reference for future work involving multispectral imaging, sparse-to-dense prediction, physics-based synthetic benchmarks, and heatmap generation for biological or agricultural samples.

**Electronics engineers and system developers.** For electronics engineers and system developers, this study establishes two constraints that any future acquisition device for this task would have to meet. The first is the computational budget. Chapter 3 reports the trained network's size and its inference time on a processor without a graphics unit, which bounds the processing hardware such a device would need to carry. The second is the wavelength requirement. Chapter 2 sets out why each of the six bands was selected and what quantity each is expected to carry, which determines the illumination the device would have to provide. No acquisition hardware was built or tested in this study, and no claim is made about lighting design, sensor synchronization, or image capture.

**Future researchers.** This study may serve as a methodological reference for future researchers who intend to improve or expand non-invasive meat quality assessment. Future studies may build on this work by validating the pipeline on real pork, testing other pork cuts, including other meat species, adding microbial or chemical indicators, or comparing different deep learning architectures on the same benchmark. Since this study is limited to pork loin and pH mapping, future researchers can extend the system toward more comprehensive meat quality and safety assessment.

**Camarines Sur Polytechnic Colleges.** For Camarines Sur Polytechnic Colleges, this study demonstrates the institution's capability to produce applied research that combines computer science, food safety, and artificial intelligence. The development of LIGTAS-pH may encourage more interdisciplinary research projects and support the institution's role in developing locally relevant technological solutions for agriculture, food quality, and public health.

## Scope and Limitations

This research designs, develops, and evaluates the LIGTAS-pH software pipeline, a physics-based, synthetic-data-driven simulation of the intended non-invasive multispectral imaging system, for estimating and visualizing the spatial distribution of pH in pork loin (longissimus thoracis et lumborum).

The system was originally intended to comprise an optical acquisition module consisting of an integrating dome, a monochrome global shutter camera, and an LED array with six narrowband sources at 481, 525, 573, 600, 730, and 970 nm, selected to capture biochemical markers associated with pH variation in pork tissue. This hardware became inoperable before data acquisition. The present study accordingly employs a literature-constrained synthetic dataset generated from a physics-based forward model based on Kubelka-Munk radiative transfer theory, as detailed in Chapter 3. Where this section describes acquisition or measurement, it describes the design that was intended. No image was captured and no physical measurement was taken in this study.

The Convolutional Regression Network is trained to convert six-band multispectral image cubes into two-dimensional pH heatmaps across a generated range of 5.2 to 6.8 pH, and is evaluated using mean absolute error, root mean square error, the coefficient of determination, and the proportion of pixels falling within fixed tolerance bands, against the synthetic benchmark's withheld ground-truth field. To confirm that the task requires more than a simple linear relationship between reflectance and pH, performance is additionally compared against linear regression and partial least squares regression baselines. Because the strength of the link between pH and light scattering has no single published value, that comparison is repeated across the full range of plausible values for it, so that the conclusion does not depend on the one value adopted.

The study is scoped to pork loin (longissimus thoracis et lumborum), which restricts applicability to other cuts, meat species, or biological samples. Pork loin is used as the sole modelled sample type because of its established role as the reference muscle in spectral-based meat quality research, which is where the published tissue measurements this study draws on were made. Extension to other cuts would require additional data and is identified as future work. The study models six discrete wavelengths, which limits spectral resolution relative to hyperspectral systems. The output is restricted to spatial pH estimation and provides no other measure of quality such as microbial load, chemical composition, or fat content.

This study relies entirely on a physics-based synthetic dataset, constructed so that every generator parameter either traces to a published source or is explicitly disclosed as a swept or tuned value with no literature basis. No parameter value is adopted without disclosure of its evidentiary status, and no claim of experimentally validated accuracy on real pork tissue is made on the basis of this dataset.

The model predicts a value at every pixel, but only four pixels per sample carry a ground-truth label during training. Every other predicted value is an inference the network makes under a smoothness constraint, not a value it was taught directly. This is stated because the output map looks equally confident everywhere, and it is not.

Because ground truth is drawn from a synthetic pH field rather than physically measured samples, the performance reported in Chapter 4 describes feasibility under simulated conditions and should not be interpreted as validated predictive accuracy on real pork tissue. Physical validation on real pork remains a necessary direction for future work.

## Project Dictionary

The Project Dictionary contains the technical terms that define the conceptual and operation of this study:

**Convolutional Regression Network (CRN).** A specialized deep learning architecture that modifies a standard CNN to predict continuous numerical values (pH) for each pixel rather than discrete classification labels [8], used in this study to process stacked multispectral images and generate spatial pH heatmaps of pork samples.

**Denaturation Amplitude (denat_amplitude).** The parameter governing the magnitude of pH-driven increase in tissue light scattering within this study's synthetic data generator. Because no single published value exists for this parameter, it is evaluated across a disclosed sweep of plausible values (0.2 to 0.8) rather than fixed to one number, and model performance is reported at each value to verify that findings are not an artifact of the specific value used for primary training.

**DFD (Dark, Firm, Dry) Meat.** A quality condition caused by limited pre-slaughter glycogen reserves, resulting in an abnormally high ultimate pH that leads to reduced light scattering and a dark, translucent appearance [8], represented in this study as regions with unusually high predicted pH values of the meat.

**Hyperspectral Imaging (HSI).** It refers to an imaging technique that can cover and acquire information in the electromagnetic spectrum to process it and obtain a detailed spectral information about the spectrum for every pixel in an image [5], used in this study as a comparative reference; HSI is not directly implemented.

**Isosbestic Point.** A specific wavelength where the optical absorbance of myoglobin remains constant regardless of its oxygenation state, used as an internal baseline for normalization [9], applied in this study as a reference band for stabilizing spectral measurements.

**Kubelka-Munk Theory.** A two-flux radiative transfer model describing diffuse reflectance from a scattering and absorbing medium as a function of its absorption and scattering coefficients (Thennadil, 2008). Used in this study as the physical basis of the synthetic data generator, converting simulated tissue absorption (myoglobin- and water-driven) and scattering (pH-driven) into six-band synthetic reflectance.

**LIGTAS-pH (Light Imaging for Grading Tissue Acidity Spatially - pH).** The system proposed by this study, comprising a physics-based synthetic data generator, a trained Convolutional Regression Network, and a two-dimensional pH heatmap output. What was developed and evaluated in this study is its software pipeline.

**Literature-Constrained Synthetic Benchmark.** The dataset and evaluation framework used in this study in place of physically acquired multispectral data, generated entirely from a physics-based forward model whose parameters are drawn from published sources wherever available, and explicitly disclosed as swept or tuned values where no citable source exists.

**Longissimus thoracis et lumborum (LTL).** Located on the vertebrae and supporting body movement [12], used as the sole meat sample type modelled in this study.

**Multispectral Imaging (MSI).** A non-destructive analytical technique that captures spatially registered image data across a targeted selection of discrete optical wavelengths to extract both physical and chemical properties of a sample [7], treated in this study as a simpler and more cost-effective alternative to Hyperspectral Imaging (HSI), simulated in this study using six narrowband wavelengths to generate synthetic reflectance inputs for the model.

**Non-invasive Multispectral Imaging (MSI) System.** A hardware implementation of MSI designed to perform optical measurements without physical contact with or alteration of the specimen [7]. In this study the system is represented in software: its six-channel reflectance output is simulated rather than captured, and that simulated output is what the regression model receives.

**Non-Triviality Baseline.** A simple linear regression fit directly on raw six-band reflectance, used to verify that the pH-prediction task is not solvable by a trivial model. The Convolutional Regression Network's performance is required to exceed this baseline by a stated margin, across the disclosed denat_amplitude sweep, for its use to be considered scientifically justified.

**Partial Least Squares Regression (PLSR).** A classical chemometric regression method that models the target variable using a reduced set of latent components derived from correlated predictor variables (Wold, Sjostrom and Eriksson, 2001). Used in this study as a second baseline alongside ordinary linear regression, consistent with its standard use for comparison in the spectral-imaging meat-quality literature.

**PSE (Pale, Soft, Exudative) Meat.** A severe quality defect caused by an abnormally rapid post-mortem pH decline, resulting in excessive protein denaturation, increased light scattering, and poor moisture retention [8], interpreted in this study as low-pH regions of the meat.

**pH Level.** A quantitative measure of acidity or alkalinity in meat, where a rise from the normal range (5.4 to 5.8) indicates protein degradation and the onset of spoilage, typically changing within 24 hours post-mortem as a result of the conversion of glycogen into lactate during postmortem glycolysis [8]. Used to determine whether the meat is PSE, normal, or DFD.

**Pork Meat.** The edible skeletal muscle tissue derived from domesticated pigs (Sus scrofa domesticus), whose postmortem quality is commonly evaluated through physicochemical parameters such as pH, color, and water-holding capacity [12], used in this study to define the target application of the LIGTAS-pH system for non-invasive pH estimation across pork muscle surfaces.

**Sparse-to-Dense Regression.** The prediction task addressed by this study's Convolutional Regression Network, in which a small number of labeled ground-truth points (four per sample) are used to train a model that outputs a prediction at every pixel of the sample surface (Hu et al., 2023), with a total-variation smoothness prior making the underdetermined problem well-posed (Rudin, Osher and Fatemi, 1992).

**Spatial Mapping.** The process of generating a two-dimensional, high-resolution heatmap [17], produced as the final output of the CRN model and visually representing the localized pH values across the entire surface of the meat cut.

---

# CHAPTER 2
# RELATED LITERATURE AND STUDIES

This chapter reviews literature and studies relevant to the system. It covers meat quality assessment and pH dynamics, the optical behaviour of meat tissue, spectral imaging technologies, synthetic data generation under parameter uncertainty, deep learning approaches to spatial-spectral regression, and the evaluation of such models.

## Meat Quality and pH Dynamics

### Pork Quality Evaluation and the Significance of pH in the Longissimus Muscle

Yao et al. (2019) identified pork pH as a principal measure of freshness and quality, noting that conventional measurement by meat infusion and pH meter is invasive, slow, and limited to a single point. They reviewed portable instruments, including the smartphone-based spectrometer of Das et al. (2016) and the low-cost spectrofluorimeter of Hossain et al. (2017), which permit rapid non-destructive analysis but likewise measure at one location. This limitation motivates the spatially resolved approach taken in the present study.

Pork quality is commonly assessed by grading and by physicochemical testing. The Bureau of Agriculture and Fisheries Standards (2025) defines carcass grading as an assessment of the whole carcass measured at specific anatomical locations, among them the longissimus dorsi or loin eye, where the colour of lean and the loin eye area serve as quality indicators. Anatomically this muscle corresponds to the longissimus thoracis et lumborum (LTL) in the thoracic and lumbar regions. Beyond grading, the industry assesses quality by measuring pH (Jess et al., 2025). Pork loins are widely used for this purpose, and processors measure both initial and ultimate pH to assess product quality. The standing of the LTL in carcass evaluation and in industry quality control has established it as the reference muscle for pork quality traits including pH.

Cross et al. (2018) report a population mean ultimate pH of 5.70 across 599 pigs sampled from pork loin, the figure this study adopts as the midpoint of its denaturation-scattering model. That cohort was explicitly screened to exclude PSE-affected carcasses, a limitation carried forward into how the midpoint is disclosed and used in Chapter 3.

### Biochemical and Optical Principles in Meat Science

Dorleku et al. (2025) examined the prediction of pork quality from pH measured at specific temperatures, finding that measurement during cooling from 39 °C to 31 °C was more consistent than the conventional practice of measuring at fixed times of 45 minutes and 24 hours post-mortem. The study also reported that rapid pH decline at higher temperatures produces protein denaturation leading to PSE meat. The method remains invasive, which is the limitation the present study seeks to address.

**Figure 2.1.** Relation between meat pH and colour (Sristi et al., 2025).

Figure 2.1 shows the phenotypic consequence of abnormal post-mortem acidification and the difference in appearance between normal meat, pale PSE tissue, and dark DFD tissue (Sristi et al., 2025). Xie et al. (2024) showed that hyperspectral imaging combined with wavelet transform supports non-destructive biochemical analysis of pork during storage, decomposing spectral signals to obtain features that predict stages of rigor mortis, ageing, and spoilage with reported accuracy above 95 percent. Taken together, these studies indicate that biochemical parameters such as pH and optical measurements carry complementary information about meat quality.

**Figure 2.2.** Visualization of spatial pH distribution obtained by hyperspectral imaging (Jia et al., 2017).

Figure 2.2 illustrates the case for spatial rather than point assessment, showing the localized and heterogeneous distribution of quality indicators within a single cut.

Han et al. (2024) reviewed the chemical and structural determinants of meat colour across six variables: myoglobin content, muscle structure, lipid oxidation, Maillard reactions, additives, and packaging. They reported that high myoglobin content at an appropriate pH produces the bright red colour associated with freshness, whereas low pH and high oxidation produce brown metmyoglobin and darker pigments. The review emphasises that meat colour results from complex physicochemical relationships and that objective high-resolution measurement is required to monitor it.

**Table 2.1.** Chemical states and conditions of fresh meat colour (American Meat Science Association, 2023).

| Chemical form of myoglobin | Compound on free binding site | Iron state | Colour | Other conditions |
|---|---|---|---|---|
| Deoxymyoglobin | None | Fe²⁺ (ferrous) | Purple | Very low oxygen, as in intact muscle or vacuum packaging |
| Oxymyoglobin | O₂ | Fe²⁺ (ferrous) | Red | Exposure to air, as in film overwrap tray |
| Carboxymyoglobin | CO | Fe²⁺ (ferrous) | Red | Oxygen concentration dependent |
| Metmyoglobin | H₂O | Fe³⁺ (ferric) | Brown | 1 to 3 percent oxygen, as in a leaking vacuum package |

Spires et al. (2023) examined post-mortem metabolism and pork quality across genetic lines under electrical stimulation, using 150 pigs from three genetic populations with varying stimulation levels, quantifying glycogen and lactate and measuring myosin heavy chain isoform abundance. High-voltage stimulation increased the rate of pH decline and lactic acid formation, producing paler and less firm meat with greater drip loss, though ultimate pH alone did not determine the quality outcome. Genetic lines differed in fibre type distribution, which further affected metabolic response and colour. The study indicates that pH decline is a principal factor but not the sole explanation of quality variation, and that local differences in glycogen depletion and fibre type produce spatial variation within a muscle.

Piao et al. (2025) adapted Krzywicki's equations to obtain more precise estimates of the oxy-, deoxy-, and metmyoglobin redox forms at meat surfaces, using tissue optics, the Lambert-Beer relation, and diffuse photon remission models to address cases where the original equations returned implausible or negative values. Forty-four beef samples across several muscles were monitored over seven days with daily spectral reflectance measurement. The adapted algorithm resolved metmyoglobin formation and colour degradation more clearly than the original. The authors identify sensitivity to baseline settings, the assumption of wavelength-independent scattering, and the inability of optical spectroscopy to distinguish myoglobin from haemoglobin as limitations.

## Non-Invasive Technology for Meat Inspection

Optical assessment of meat quality rests on how light of a given wavelength propagates through tissue and is absorbed and scattered (Duong et al., 2023). In turbid biological media such as pork, propagation is governed principally by the absorption coefficient and the scattering coefficient (Jacques, 2013).

Diffuse reflectance from such a medium is described more accurately by Kubelka-Munk theory, a two-flux radiative transfer model relating reflectance to the ratio of the absorption coefficient and the reduced scattering coefficient, than by simple exponential attenuation with depth, which assumes negligible scattering (Thennadil, 2008). Absorption is governed by chemical chromophores, principally myoglobin, while scattering is governed by the physical microstructure of the muscle fibre lattice.

**Figure 2.3.** Monte Carlo simulation of photon fluence distribution in pork tissue at 650, 700, 750, 800, 850, and 900 nm (Duong et al., 2023).

Figure 2.3 shows the complexity of these optical interactions. Incident light does not simply reflect from the surface: photons penetrate the tissue and scatter radially, interacting with its biochemical structure before emerging as diffuse reflectance (Duong et al., 2023). It is this emergent diffuse component that carries internal physicochemical information, whereas specular reflection is surface glare (Trinderup, 2015). Separating the two is the purpose of an integrating dome in the intended optical configuration (Bandara et al., 2018).

Post-mortem pH alters the biochemistry of meat in ways that directly affect its scattering and absorption (Huff-Lonergan and Lonergan, 2005). As pH approaches the isoelectric point of the major muscle proteins, protein denaturation causes the myofibrillar lattice to shrink. Water is lost from the cells into the extracellular space, reducing water-holding capacity and substantially increasing achromatic light scattering (Bandara et al., 2018). Meat at lower pH therefore exhibits higher diffuse reflectance across the visible spectrum and a paler appearance. This mechanism, pH to denaturation to scattering, is the pathway through which pH enters the forward model described in Chapter 3.

### Selection of Spectral Bands

The six bands at 481, 525, 573, 600, 730, and 970 nm were selected from the optical properties of pork tissue and their relation to pH-influenced quality indicators. In the visible region, meat colour is governed principally by the relative concentrations of the myoglobin derivatives, each with a distinct absorption spectrum, which reflectance models can resolve from specific bands (Piao et al., 2025).

**Figure 2.4.** SHAP wavelength importance for pH prediction (Wang, Ma, Li and Zhang, 2026).

The three most informative wavelengths, 481, 573, and 600 nm, were identified as the principal optical drivers of pork pH prediction by the Shapley additive explanations analysis of Wang, Ma, Li and Zhang (2026), shown in Figure 2.4. Three further bands at 525, 730, and 970 nm were included on the basis of established requirements in the meat science literature, serving as internal reference, structural baseline, and moisture indicator respectively.

**Figure 2.5.** Reflectance spectra of the three myoglobin forms, obtained according to AMSA recommendations (Piao et al., 2025).

The visible colour of pork is determined by the absorption characteristics of the heme group in myoglobin, which differ between redox forms as shown in Figure 2.5. The Soret band between 400 and 450 nm corresponds to the strongest absorption of the porphyrin ring, and its trailing edge is sensitive to the deoxygenated state, which supports selection of 481 nm (Krzywicki, 1979; Piao et al., 2025). The Q-bands between 500 and 600 nm comprise the alpha and beta absorption features: deoxymyoglobin presents a single broad trough near 556 nm, while oxygenation splits this into a double minimum near 542 and 580 nm. The 573 nm band lies within this region and tracks that transition. The convergence of all three redox curves at 525 nm provides a reference baseline against which pH-driven change at the active wavelengths, including the metmyoglobin shift at 600 nm, can be interpreted (Krzywicki, 1979; Piao et al., 2025).

1. **481 nm.** Near the edge of the Soret absorption band and sensitive to deoxymyoglobin concentration (Wang, Ma, Li and Zhang, 2026).
2. **573 nm.** Within the Q-band, the region of the myoglobin absorption spectrum sensitive to oxygenation state, permitting the deoxymyoglobin to oxymyoglobin transition to be monitored (Wang, Ma, Li and Zhang, 2026).
3. **600 nm.** Sensitive to conversion of the heme pigment to metmyoglobin. The structural integrity of the heme group responds to local pH, so pH-induced discoloration is expressed at this wavelength (Wang, Ma, Li and Zhang, 2026).
4. **525 nm.** The absorbance of myoglobin here is independent of redox state (Krzywicki, 1979; Piao et al., 2025), which makes it an internal baseline against which the active wavelengths can be interpreted.
5. **730 nm.** Outside the regions of dominant pigment absorption, serving as a structural reference where reflectance is governed by scattering from the myofibrillar lattice rather than by chemical absorption (Duong et al., 2023; Piao et al., 2025).
6. **970 nm.** Corresponds to an absorption feature of water, carrying information on moisture and water-holding capacity, both of which are affected by pH-induced protein transitions (Kamruzzaman et al., 2015; Kandpal et al., 2013).

**Figure 2.6.** Feature wavelength selection for monitoring water-holding capacity in red meat (Kamruzzaman et al., 2015).

### Provenance of the Optical Parameters Used in This Study

Because the present study generates its data rather than acquiring it, the optical constants entering the forward model determine the realism of the benchmark, and each is drawn from a specific published source.

The reduced scattering coefficient follows the wavelength-dependent power law of Jacques (2013), refit toward the porcine-specific curve reported by Bergmann et al. (2021). Absorption is computed from water content using the absorption coefficients of Hale and Querry (1973) and the water fraction reported for pork longissimus muscle by Wojtasik-Kalinowska et al. (2016). Myoglobin concentration is taken from the 599-pig population reported by Cross et al. (2018).

The extinction coefficients separating the three myoglobin redox forms were digitized directly from the visible-region spectra of Tang, Faustman and Hoagland (2004) and the near-infrared spectra of Bowen (1949), rather than interpolated between tabulated columns. Linear interpolation between Tang's tabulated values was found to misestimate reflectance at 573 nm by 9 to 14 percent, owing to spectral curvature in that region that a straight line cannot capture.

These extinction coefficients are equine. Pork-specific values are not published at the six wavelengths used. The substitution follows established practice in meat science: the Krzywicki (1979) equations, derived from horse myoglobin absorption spectra, are applied to other species and remain the basis of current quantification methods, including the modified formulation of Piao et al. (2025). The practice is nonetheless not exact, since species differences in myoglobin absorption, including shifts in the position of isosbestic points, are documented in the comparative literature. The substitution is therefore adopted as a disclosed assumption rather than as an equivalence, and is listed among the theoretical assumptions in Chapter 3 and among the limitations in Chapter 5.

Not every parameter can be traced to a published value. Three are disclosed as having no citable source. The denaturation amplitude, which governs how strongly pH drives scattering, has no single reported figure and is therefore swept across its plausible range with results reported at every setting rather than fixed to one number. The denaturation width is likewise swept and is disclosed as unsourced. A baseline absorption term is tuned rather than cited. Chapter 3 states the evidentiary status of each parameter individually, and Chapter 4 reports performance across the swept range so that the conclusions can be checked against the uncertainty that remains.

## Spectral Imaging Technologies: From HSI to MSI

Hyperspectral imaging acquires continuous spectral and spatial information across hundreds of contiguous bands (Kamruzzaman et al., 2015), producing a three-dimensional hypercube that yields dense physicochemical information from biological tissue under controlled conditions (Barbin et al., 2012). HSI systems have been used in meat science to map and predict post-mortem pH decline, total volatile basic nitrogen accumulation, and intramuscular marbling (Qiao et al., 2006).

**Figure 2.7.** A typical HSI system and its output reflectance curves (Ma et al., 2022).

As shown in Figure 2.7, such a system combines a broadband illumination source, an objective lens, a spectrograph that disperses reflected signal across wavelengths, a sensitive camera, a translation stage for line scanning, and a computer for acquisition (Ma et al., 2022; Feng et al., 2017). The spectrograph records hundreds of contiguous bands at every spatial point, in contrast to the three broad channels of a conventional camera, producing a continuous reflectance curve at each pixel.

An example of a publicly documented hyperspectral meat dataset is NHSI-meat-overtime (Wang, Tang, Li and Chen, 2026), which images five meat types: chicken, beef, mutton, pork, and salmon. This study uses it as an external reference point for the 970 nm band, the only band overlapping the LIGTAS-pH selection and the only publicly available dataset offering such an overlap.

### Limitations of HSI for Real-Time Deployment

Although precise, HSI faces hardware and algorithmic obstacles to industrial adoption (Feng et al., 2017). Acquiring full hypercubes produces large volumes of redundant and collinear data, which constrains real-time processing (Wang, Ma, Li and Zhang, 2026; Kamruzzaman et al., 2015). Conventional architectures also depend on a spectrograph, a line-scanning translation stage, and high-intensity illumination, which are costly and impractical in environments such as slaughterhouses or wet markets (Bandara et al., 2018; Kandpal et al., 2013).

### The Shift to MSI

In response, optical food assessment has moved toward using a subset of discrete wavelengths selected for their informativeness (Bandara et al., 2018; Wang, Ma, Li and Zhang, 2026). Feature selection methods identify the most informative bands while discarding the rest, reducing dimensionality and permitting real-time analysis (Jia et al., 2017). This reduction allows a low-cost LED illumination array to replace a spectrograph and an embedded processor to replace a workstation, while retaining predictive accuracy (Ma et al., 2022; Zhang et al., 2022).

**Figure 2.8.** A typical MSI system and its block diagram (Bandara et al., 2018).

A modern MSI system replaces the spectrograph with targeted illumination (Bandara et al., 2018). The architecture shown in Figure 2.8 is based on a light-tight chamber with an integrating hemisphere in which discrete narrowband LEDs are flashed in sequence to deliver uniform diffuse illumination. A single camera captures a two-dimensional reflectance image at each wavelength, and the resulting images are stacked to form a reduced multispectral cube.

## Synthetic Data Generation and Parameter Uncertainty

This study anticipates the question of whether a simulation-derived dataset is an acceptable substitute for physically collected data, given the absence of real pork samples. Precedent exists in the optical simulation literature. Quintana-Quintana et al. (2025) construct a Monte Carlo tissue-light transport simulation for hyperspectral microscopy by assembling absorption and scattering parameters for four tissue types, each drawn from a separate published source. This is structurally the approach taken in Chapter 3, where each of fourteen physical parameters is independently sourced, cited, or explicitly disclosed as swept or tuned.

The limitation of the approach is equally clear and is stated rather than minimised: a model calibrated from published constants reproduces the behaviour those constants encode, and agreement with the published literature is not evidence of agreement with physical measurement. Quintana-Quintana et al. (2025) acknowledge the same constraint, validating their simulation against published optical properties rather than against independent measurement of the tissues modelled. The single external comparison available to this study is the 970 nm water absorption feature, discussed in Chapter 4.

A second consideration is specific to the task addressed here. The evaluation reported in Chapter 4 scores a dense, pixel-wise prediction against a dense reference field. No physical procedure yields such a reference for pork. Probe measurement is destructive and returns a value at a single point, and even sub-sectional destructive sampling after imaging would return one value per section rather than one per pixel. Published work producing pixel-level prediction maps validates them against sample-level or point-level reference values (Barbin et al., 2012; Tang et al., 2023), because no finer reference exists. A synthetic benchmark is therefore not only a substitute for data that could not be collected, but the only setting in which a pixel-wise prediction can be scored against a pixel-wise ground truth. The cost of that advantage is that the reference is simulated, so the result establishes the feasibility of the method rather than its accuracy on tissue.

## pH Measurement

**Figure 2.9.** RGB to monochrome conversion of multispectral image bands (Bandara et al., 2018).

Bandara et al. (2018) collected spectral data using a low-cost configuration built on a smartphone camera, capturing RGB images at each selected wavelength band and converting them to monochrome to form a multispectral dataset, as shown in Figure 2.9. The authors observed that conventional RGB to greyscale conversion, which applies weights derived from human visual response, is unsuited to spectral analysis because it over-weights the green channel and under-represents response at other wavelengths. Conversion methods preserving the dominant sensor response were used instead to approximate reflectance intensity. A general limitation of RGB-based systems is that channel sensitivity varies across wavelengths, which distorts the signal. Systems acquiring monochrome images at each wavelength avoid this conversion bias and measure per-wavelength reflectance more directly, which is the configuration the present study simulates.

## Deep Learning and Convolutional Regression Networks

### Chemometrics and Deep Learning

Classical chemometric and linear methods, including partial least squares regression (Wold et al., 2001) and support vector machines, have long been used for spectral analysis in the agrifood sector (Wang, Ma, Li and Zhang, 2026; Xu et al., 2025). These methods are mathematically transparent and perform well on small datasets, but depend on manual feature extraction and are sensitive to non-linear scattering from the meat surface, environmental noise, and baseline drift. Deep learning frameworks address these constraints by inferring hierarchical features directly from the data and modelling non-linear relationships between scattering and pH without manual feature engineering (Xiao et al., 2025; Lun et al., 2025).

### Convolutional Neural Networks in Spectroscopy

Convolutional neural networks are the dominant approach to spectral-spatial image analysis (Tang et al., 2023). A convolution applies a learned kernel across the input:

> y(i, j) = sum over m, n of x(i + m, j + n) · w(m, n) + b  (2.1)

where x is the input feature map, w the convolution kernel, b the bias term, and y(i, j) the output at spatial location (i, j) (Ronneberger et al., 2015; Tang et al., 2023).

In optical meat assessment, one-dimensional CNNs are applied along the spectral dimension to detect absorption features corresponding to myoglobin redox states and water content (Wang, Ma, Li and Zhang, 2026; Jiang et al., 2021), while two-dimensional CNNs are applied across the spatial dimensions to extract morphological and textural features such as structural shrinkage from protein denaturation and the distribution of intramuscular fat (Tang et al., 2023; Xing et al., 2023). Dual-branch architectures combine the two to improve robustness against background interference (Xu et al., 2025).

**Figure 2.10.** Preprocessing and spectral enhancement pipeline preceding CNN regression (Jiang et al., 2021).

Figure 2.10 shows a representative preprocessing workflow: radiometric calibration against dark and white references, spatial registration and region-of-interest extraction to isolate tissue from background (Huang et al., 2023; Rana et al., 2024; Xu et al., 2024), and spectral enhancement by denoising and augmentation before the data reach the regression model (Zhang et al., 2020). Jiang et al. (2021) report that a fully convolutional regression approach converges faster and to a lower error than support vector machine and partial least squares models on near-infrared data, which they attribute to the capacity of convolutional layers to capture structure in spectral sequences that conventional algorithms handle poorly in the presence of redundancy and noise.

### Convolutional Regression Networks for Spatial Mapping

The objective of LIGTAS-pH is not a single averaged pH value for a whole cut but the variation of pH across the tissue. A Convolutional Regression Network addresses this by terminating in a regression output rather than a classification layer, producing a continuous value (Wang, Ma, Li and Zhang, 2026; Xun et al., 2024). The model is trained by minimising prediction error, conventionally

> L = (1 / n) · sum over i of ( y_i - y_hat_i )²  (2.2)

where y_i is the true pH, y_hat_i the predicted pH, and n the number of supervised values. Mean squared and mean absolute error are the losses conventionally used for regression heads of this kind (Xiao et al., 2025; Lun et al., 2025).

At inference the network receives a three-dimensional array of two spatial dimensions and six spectral channels, and returns a pH estimate at every pixel. These predictions form a two-dimensional heatmap in which regions of abnormal pH, including PSE and DFD zones, can be located directly (Tang et al., 2023; Xing et al., 2023).

Because stochastic training can converge differently depending on random weight initialisation and batch ordering, a single training run is not sufficient evidence of typical performance. Henderson et al. (2018) demonstrate this for deep reinforcement learning agents, showing that reported performance varies substantially across runs differing only in random seed, and argue for reporting variability across multiple seeds rather than a single result. This study accordingly reports the mean and standard deviation across five independently seeded runs.

Early training runs in this study exhibited pronounced validation instability. Batch normalization at small batch size was the first suspected cause, which Wu and He (2018) make a reasonable suspicion: they document that its effectiveness degrades as batch size decreases, because its normalization statistics are estimated from fewer samples. An isolated investigation was nonetheless carried out before any change to the architecture, and it traced the instability to the learning rate rather than to batch normalization. Lowering the rate was the remedy, and no architectural change was made. A separate error in how the coefficient of determination was aggregated, found later, had additionally overstated how severe the instability appeared. Both are described in Chapter 3.

### Deep Learning Architectures for Sparse Supervision

Hu et al. (2023) surveyed deep learning approaches to dense prediction from extremely sparse ground truth, reviewing more than one hundred techniques including convolutional encoder-decoder designs, multi-scale feature fusion, and masked loss functions that exclude unlabelled pixels during training. Explicit handling of sparse ground truth through validity masks and multi-scale fusion improved accuracy, whereas naive treatment of missing pixels produced biased learning. Although the benchmarks concerned LiDAR depth completion, the principles extend to any setting requiring a dense spatial map from a limited number of measurements, which is the problem this study addresses.

Norelyaqine et al. (2023) examined encoder-decoder CNNs for full-resolution spatial prediction, comparing U-Net variants with ResNet50, VGG, DenseNet, and Xception backbones. The analysis emphasised skip connections and transposed convolutions for recovering fine spatial detail lost during downsampling. The tasks were binary segmentation, but the findings on spatial feature extraction and full-resolution reconstruction inform the design of encoder-decoder networks for continuous regression, including the architecture adopted here (Ronneberger et al., 2015).

Xiao et al. (2025) reviewed deep learning regression for continuous food quality attributes including pH, moisture, fat, and protein from near-infrared and hyperspectral data, reporting that CNN-based regression consistently achieved greater precision than conventional chemometric methods, which handle non-linear spectral relationships poorly. The review identified spectral-spatial CNNs, transfer learning, and linear-activation regression heads with mean squared or mean absolute error loss as the principal approaches. It noted that most work addresses one-dimensional spectral data and that the production of two-dimensional pixel-wise regression maps visualizing chemical distribution remains comparatively unexplored.

Lun et al. (2025) reviewed deep learning architectures combined with spectroscopic techniques for food quality determination, comparing CNNs, capsule networks, long short-term memory models, and hybrids against conventional chemometric methods for predicting continuous chemical properties. The authors noted that convolution and pooling extract spatial features effectively, that residual connections are important to the performance of deeper networks, and that linear activation at the output layer is necessary for regression. They likewise observed that the field has concentrated on one-dimensional spectral regression and that prediction of two-dimensional spatial distributions remains underexplored.

## Evaluation of Algorithm

### Dataset and Performance Evaluation

Elangovan et al. (2024) assessed regression models using standard performance metrics for accuracy and generalization. The coefficient of determination and root mean square error of calibration were used to evaluate performance on the training set, while the coefficient of determination of prediction, root mean square error of prediction, and mean absolute error were used on the test data. Higher values of the coefficient of determination indicate that the model accounts for more of the variance, while lower error values indicate greater precision. Comparing these metrics across models allowed the authors to establish the advantage of their proposed model in predicting pork quality indicators.

The choice between mean absolute error and root mean square error is not arbitrary. Hodson (2022) sets out the conditions under which each is appropriate, noting that root mean square error penalises large errors more heavily and that the two should be interpreted together rather than treated as interchangeable. The present study reports both, along with the coefficient of determination and tolerance-band accuracy, as described in Chapter 3.

**Table 2.2.** Standard statistical metrics for evaluating spatial-spectral regression models.

| Metric | Significance | Target |
|---|---|---|
| Coefficient of determination | The proportion of variance in the measured quality parameter accounted for by the model. Evaluated on both calibration and prediction sets. | Above 0.80 considered good; above 0.90 considered excellent |
| Root mean square error | The average deviation between predicted and reference values on unseen data, penalising large errors more heavily. | As close to zero as possible |
| Mean absolute error | The average magnitude of prediction error without squaring, giving a direct linear measure of average inaccuracy. | As close to zero as possible |

The thresholds in Table 2.2 are those conventionally applied in the hyperspectral and multispectral literature to models predicting a single value per sample from physically acquired data. They are reported here as context rather than as the acceptance criterion for this study, whose predictions are pixel-wise and whose data are synthetic.

## Synthesis of the State of the Art

Post-mortem pH is an established marker of pork quality. The decline in muscle pH after slaughter initiates a sequence of quality changes, among them PSE meat, colour loss, and reduced water retention (Dorleku et al., 2025; Jess et al., 2025; Spires et al., 2023). Measurement, however, generally requires puncturing the meat and returns a reading from one location (Yao et al., 2019). That spatial limitation has directed the field toward non-invasive alternatives.

Hyperspectral and multispectral imaging have gained ground as such alternatives, with classification accuracies above 95 percent reported for biochemical quality staging (Xie et al., 2024). Monte Carlo simulation has characterised wavelength-dependent penetration in pork tissue (Duong et al., 2023), and spectral modelling has advanced to the point that myoglobin redox states can be estimated from surface reflectance (Piao et al., 2025). Together these establish that meaningful chemistry can be read from the surface without contact.

Obtaining usable spectral data requires preprocessing. Band registration ensures spatial alignment across wavelengths (Huang et al., 2023; Rana et al., 2024), spectral noise is commonly addressed by autoencoder-based denoising (Zhang et al., 2020), and region-of-interest segmentation isolates informative tissue (Xu et al., 2024; Cervantes-Sanchez et al., 2021). For continuous quality parameters, CNN-based models have outperformed classical chemometric methods (Xiao et al., 2025; Lun et al., 2025). Encoder-decoder designs with skip connections are effective where spatial structure matters (Norelyaqine et al., 2023), and compact architectures have made deployment outside the laboratory more realistic (Zhang et al., 2022; Elangovan et al., 2024).

The persistent constraint is data. Annotated spectral datasets for meat quality are small, and their collection is slow and costly. What remains absent is a method for producing pixel-wise, spatially continuous maps of pH from multispectral images. Most existing work stops at classification or point prediction, and full two-dimensional chemical mapping of a property such as pH is largely unaddressed (Xiao et al., 2025; Lun et al., 2025).

## Gap Bridged by the Study

The reviewed literature establishes each component of the proposed system individually. Post-mortem pH is established as the principal biochemical indicator of pork quality (Dorleku et al., 2025; Jess et al., 2025; Spires et al., 2023); surface reflectance is established as carrying sufficient spectral-chemical information for non-invasive assessment (Xie et al., 2024; Piao et al., 2025; Duong et al., 2023); preprocessing methods for noise suppression and spatial alignment are established (Huang et al., 2023; Rana et al., 2024; Zhang et al., 2020; Xu et al., 2024; Cervantes-Sanchez et al., 2021); and compact architectures are established as capable of accurate spectral regression on constrained hardware (Norelyaqine et al., 2023; Zhang et al., 2022). The gap lies not in any component but in their integration.

No reported system produces pixel-wise, spatially continuous estimates of pH across a pork sample from a portable, non-contact configuration. Existing non-invasive methods, including portable spectrometers (Das et al., 2016; Hossain et al., 2017) and low-cost hyperspectral scanners (Yao et al., 2019), measure at a point, leaving the spatial gradient produced by non-uniform post-mortem glycolysis unmeasured (Dorleku et al., 2025; Spires et al., 2023). Full hyperspectral systems achieve high spatial resolution (Xie et al., 2024) but are too costly and computationally demanding for portable deployment. Multispectral imaging is the feasible alternative, but existing MSI work on meat quality extends only to classification rather than continuous spatial regression (Xu et al., 2024; Elangovan et al., 2024). Sparse-to-dense regression with masked loss, which would permit dense pH maps to be recovered from a small number of probe measurements (Hu et al., 2023; Norelyaqine et al., 2023), has not been applied in this domain, and pixel-wise two-dimensional prediction of food quality attributes remains underused (Xiao et al., 2025; Lun et al., 2025).

LIGTAS-pH addresses this gap by combining tissue-optics-informed wavelength selection (Duong et al., 2023), alignment of sparse pH supervision with corresponding image regions, sparse-to-dense regression under a total-variation smoothness prior (Hu et al., 2023; Rudin et al., 1992), and a compact U-Net-based CRN (Ronneberger et al., 2015). Because the imaging hardware required to collect real pork data became unavailable, the pipeline is evaluated on a literature-constrained synthetic benchmark, with overfitting addressed through early stopping on held-out validation loss, the smoothness prior, and multi-seed reporting (Henderson et al., 2018), and with performance reported using mean absolute error, root mean square error, and the coefficient of determination across held-out and unseen test sets.

---

# NOTES

The list below covers Chapters 1 and 2. Entries marked **[NEW]** are additions to the existing Chapter 2 reference list and must be merged into it alphabetically, after which the whole list is renumbered and the author-year citations above are converted to the project's numbered style.

American Meat Science Association. (2023). *Chemistry of fresh meat color.* https://meatscience.org/docs/default-source/publications-resources/factsheets/color_chemistry_resource1_7.28.23_final.pdf

Bandara, C., Prabhath, G. W. K., Dissanayake, D. W. S. C. B. and Herath, V. R. (2018). A multispectral imaging system to assess meat quality. *IEEE Region 10 Humanitarian Technology Conference (R10-HTC).* doi:10.1109/R10-HTC.2018.8629858

Barbin, D. F., ElMasry, G., Sun, D.-W. and Allen, P. (2012). Predicting quality and sensory attributes of pork using near-infrared hyperspectral imaging. *Analytica Chimica Acta*, 719, 30-42. doi:10.1016/j.aca.2012.01.004

**[NEW]** Bergmann, F., Foschum, F., Marzel, L. and Kienle, A. (2021). Ex vivo determination of broadband absorption and effective scattering coefficients of porcine tissue. *Photonics*, 8(9), 365. doi:10.3390/photonics8090365. Open access.

**[NEW]** Bowen, W. J. (1949). The absorption spectra and extinction coefficients of myoglobin. *Journal of Biological Chemistry*, 179, 235-245. PMID 18119239. Open access at jbc.org. A DOI of 10.1016/S0021-9258(18)56832-0 is listed for this article in the project's parameter sheet and should be confirmed against the publisher record before use.

Bureau of Agriculture and Fisheries Standards. (2025). *Philippine national standard: pork carcass grading.* Department of Agriculture, Philippines. https://bafs.da.gov.ph/wp-content/uploads/2025/12/IG-Pork-carcass-grading.pdf

Cervantes-Sanchez, F., Maktabi, M., Köhler, H., Sucher, R., Rayes, N., Avina-Cervantes, J. G., Cruz-Aceves, I. and Chalopin, C. (2021). Automatic tissue segmentation of hyperspectral images in liver and head neck surgeries using machine learning. *Artificial Intelligence Surgery.* doi:10.20517/ais.2021.05

**[NEW]** Cross, A. J., King, D. A., Shackelford, S. D., Wheeler, T. L., Nonneman, D. J., Keel, B. N. and Rohrer, G. A. (2018). Genome-wide association of myoglobin concentrations in pork loins. *Meat and Muscle Biology*, 2(1), 189-196. doi:10.22175/mmb2017.08.0042. Open access.

Das, A. J., Wahi, A., Kothari, I. and Raskar, R. (2016). Ultra-portable, wireless smartphone spectrometer for rapid, non-destructive testing of fruit ripeness. *Scientific Reports*, 6, 32504. doi:10.1038/srep32504

Dorleku, J. B., Tayengwa, T., Bohrer, B. M. and Juárez, M. (2025). Measuring pH of pork at specific temperatures postmortem to predict quality traits. *Meat and Muscle Biology*, 9(1). doi:10.22175/mmb.18532

Duong, H. T., Nguyen, M. K., Nguyen Thi, M. N., Ta Ngoc, M. C., Nguyen, A. X., Tran, A. T. and Ha, Q. T. (2023). Investigation of light propagation in pork tissue applied in multispectral imaging technique by Monte Carlo simulation method, 5, 36-28. doi:10.29007/spmn

Elangovan, P., Dhurairajan, V., Nath, M. K., Yogarajah, P. and Condell, J. (2024). A novel approach for meat quality assessment using an ensemble of compact convolutional neural networks. *Applied Sciences*, 14(14), 5979. doi:10.3390/app14145979

Feng, C.-H., Makino, Y., Oshita, S. and García Martín, J. F. (2017). Hyperspectral imaging and multispectral imaging as the novel techniques for detecting defects in raw and processed meat products. *Food Control*, 84, 165-176. doi:10.1016/j.foodcont.2017.07.013

**[NEW]** Hale, G. M. and Querry, M. R. (1973). Optical constants of water in the 200 nm to 200 µm wavelength region. *Applied Optics*, 12(3), 555-563. doi:10.1364/AO.12.000555. The values used in this study were taken from the tabulated data file hosted at omlc.org/spectra/water/data/hale73.dat rather than from the article text.

Han, J., Wang, Y., Wang, Y., Hao, S., Zhang, K., Tian, J. and Jin, Y. (2024). Effect of changes in the structure of myoglobin on the color of meat products. *Food Materials Research*, 4(1). doi:10.48130/fmr-0024-0003

**[NEW]** Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1). Preprint at arXiv:1709.06560

**[NEW]** Hodson, T. O. (2022). Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not. *Geoscientific Model Development*, 15(14), 5481-5487. doi:10.5194/gmd-15-5481-2022

Hossain, M. A., Canning, J., Cook, K., Ast, S. and Jamalipour, A. (2017). Photo- and thermal degradation of olive oil measured using an optical fibre smartphone spectrofluorimeter, 10323, 1032310. doi:10.1117/12.2265580

Hu, J., Bao, C., Ozay, M., Fan, C., Gao, Q., Liu, H. and Lam, T. L. (2023). Deep depth completion from extremely sparse data: a survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence.* doi:10.1109/TPAMI.2022.3229090

Huang, F., Chen, Y., Wang, X., Wang, S. and Wu, X. (2023). Spectral clustering super-resolution imaging based on multispectral camera array. *IEEE Transactions on Image Processing*, 32, 1257-1271. doi:10.1109/tip.2023.3242589

Huff-Lonergan, E. and Lonergan, S. M. (2005). Mechanisms of water-holding capacity of meat: the role of postmortem biochemical and structural changes. *Meat Science*, 71(1), 194-204. doi:10.1016/j.meatsci.2005.04.022

Jacques, S. L. (2013). Optical properties of biological tissues: a review. *Physics in Medicine and Biology*, 58(11), R37-R61. doi:10.1088/0031-9155/58/11/r37

Jess, C. E., Johnson, L. G., Prusa, K. J., Lonergan, S. M. and Huff-Lonergan, E. J. (2025). Fresh pork loin pH influences meat quality and the presence of desmin degradation products. *Meat and Muscle Biology*, 9(1). doi:10.22175/mmb.18426

Jia, B., Yoon, S.-C., Zhuang, H., Wang, W. and Li, C. (2017). Prediction of pH of fresh chicken breast fillets by VNIR hyperspectral imaging. *Journal of Food Engineering*, 208, 57-65. doi:10.1016/j.jfoodeng.2017.03.023

Jiang, D., Hu, G., Qi, G. and Mazur, N. (2021). A fully convolutional neural network-based regression approach for effective chemical composition analysis using near-infrared spectroscopy in cloud. *Journal of Artificial Intelligence and Technology*, 1(1), 74-82.

Kamruzzaman, M., Makino, Y. and Oshita, S. (2015). Hyperspectral imaging for real-time monitoring of water holding capacity in red meat. *LWT*, 66, 685-691. doi:10.1016/j.lwt.2015.11.021

Kandpal, L., Lee, H., Kim, M., Mo, C. and Cho, B.-K. (2013). Hyperspectral reflectance imaging technique for visualization of moisture distribution in cooked chicken breast. *Sensors*, 13(10), 13289-13300. doi:10.3390/s131013289

**[NEW]** Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *3rd International Conference on Learning Representations.* arXiv:1412.6980

Krzywicki, K. (1979). Assessment of relative content of myoglobin, oxymyoglobin and metmyoglobin at the surface of beef. *Meat Science*, 3(1), 1-10. doi:10.1016/0309-1740(79)90019-6

Lun, Z., Wu, X., Dong, J. and Wu, B. (2025). Deep learning-enhanced spectroscopic technologies for food quality assessment. *Foods*, 14(13), 2350. doi:10.3390/foods14132350

Ma, C., Yu, M., Chen, F. and Lin, H. (2022). An efficient and portable LED multispectral imaging system and its application to human tongue detection. *Applied Sciences*, 12(7), 3552. doi:10.3390/app12073552

Norelyaqine, A., Azmi, R. and Saadane, A. (2023). Architecture of deep convolutional encoder-decoder networks for building footprint semantic segmentation. *Scientific Programming*, 2023, 1-15. doi:10.1155/2023/8552624

Piao, D., Ramanathan, R., Denzer, M. L., Pfeiffer, M. and Mafi, G. (2025). Modified Krzywicki's equations to quantify myoglobin forms on meat surfaces. *Meat and Muscle Biology*, 9(1). doi:10.22175/mmb.18338

Qiao, J., Wang, N., Ngadi, M. O., Gunenc, A., Monroy, M., Gariépy, C. and Prasher, S. O. (2006). Prediction of drip-loss, pH, and color for pork using a hyperspectral imaging technique. *Meat Science*, 76(1), 1-8. doi:10.1016/j.meatsci.2006.06.031

**[NEW]** Quintana-Quintana, L., Witteveen, M., Dashtbozorg, B., Ortega, S., Ruers, T. J. M., Sterenborg, H. J. C. M. and Callico, G. M. (2025). Exploring the role of sample thickness for hyperspectral microscopy tissue discrimination through Monte Carlo simulations. *Biomedical Optics Express*, 16(11), 4644-4661. doi:10.1364/BOE.563094. Open access.

Rana, S., Gerbino, S., Crimaldi, M., Cirillo, V., Carillo, P., Sarghini, F. and Maggio, A. (2024). Comprehensive evaluation of multispectral image registration strategies in heterogenous agriculture environment. *Journal of Imaging*, 10(3), 61. doi:10.3390/jimaging10030061

**[NEW]** Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)*, 234-241. doi:10.1007/978-3-319-24574-4_28. Preprint at arXiv:1505.04597

**[NEW]** Rudin, L. I., Osher, S. and Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1-4), 259-268. doi:10.1016/0167-2789(92)90242-F

Spires, M. D., Bodmer, J. S., Beline, M., Wicks, J. C., Zumbaugh, M. D., Shi, T. H., Reichert, B. T., Schinckel, A. P., Grant, A. L. and Gerrard, D. E. (2023). Postmortem metabolism and pork quality development are affected by electrical stimulation across three genetic lines. *Animals*, 13(16), 2599. doi:10.3390/ani13162599

Sristi, P. R., Das, N. R., Akhter, A., Kaniya, N. M. and Hashem, M. A. (2025). Relation among meat pH, color and tenderness: a review. *Meat Research*, 5(3). doi:10.55002/mr.5.3.117

Tang, X., Rao, L., Xie, L., Yan, M., Chen, Z., Liu, S., Chen, L., et al. (2023). Quantification and visualization of meat quality traits in pork using hyperspectral imaging. *Meat Science*, 196, 109052. doi:10.1016/j.meatsci.2022.109052

**[NEW]** Tang, J., Faustman, C. and Hoagland, T. A. (2004). Krzywicki revisited: equations for spectrophotometric determination of myoglobin redox forms in aqueous meat extracts. *Journal of Food Science*, 69(9), C717-C720. doi:10.1111/j.1365-2621.2004.tb09922.x

**[NEW]** Thennadil, S. N. (2008). Relationship between the Kubelka-Munk scattering and radiative transfer coefficients. *Journal of the Optical Society of America A*, 25(7), 1480-1485. doi:10.1364/JOSAA.25.001480

Trinderup, C. H. (2015). *Multispectral imaging of meat quality: colour and texture.* DTU Compute PHD-2014 No. 358.

**[NEW]** Wang, M., Tang, J., Li, S. and Chen, W. (2026). NHSI-meat-overtime: a near-infrared hyperspectral dataset of five meat types over storage time. *IET Conference Proceedings*, CP987, 208-212. doi:10.1049/icp.2026.2763

Wang, M., Ma, J., Li, S. and Zhang, W. (2026). Rapid and non-destructive pork quality prediction using an interpretable deep learning-ensemble model. *Future Foods*, 13, 101000. doi:10.1016/j.fufo.2026.101000

**[NEW]** Wojtasik-Kalinowska, I., Guzek, D., Gorska-Horczyczak, E., Glabska, D., Brodowska, M., Sun, D.-W. and Wierzbicka, A. (2016). Volatile compounds and fatty acids profile in Longissimus dorsi muscle from pigs fed with feed containing bioactive components. *LWT - Food Science and Technology*, 67, 112-117. doi:10.1016/j.lwt.2015.11.023

**[NEW]** Wold, S., Sjöström, M. and Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130. doi:10.1016/S0169-7439(01)00155-1

**[NEW]** World Health Organization. (2015). *WHO estimates of the global burden of foodborne diseases: foodborne disease burden epidemiology reference group 2007-2015.* World Health Organization, Geneva. https://www.who.int/publications/i/item/9789241565165

**[NEW]** Wu, Y. and He, K. (2018). Group normalization. *European Conference on Computer Vision (ECCV)*, 3-19. doi:10.1007/978-3-030-01261-8_1. Preprint at arXiv:1803.08494

Xiao, Y., Zhou, L., Zhao, Y., Qi, H., Pu, Y. and Zhang, C. (2025). Deep learning-based regression of food quality attributes using near-infrared spectroscopy and hyperspectral imaging: a review. *Food Chemistry*, 493(Pt 4), 145932. doi:10.1016/j.foodchem.2025.145932

Xie, A., Zhang, Y., Wu, H. and Chen, M. (2024). Monitoring the aging and edible safety of pork in postmortem storage based on HSI and wavelet transform. *Foods*, 13(12), 1903. doi:10.3390/foods13121903

Xing, X., Zhou, Y., Wang, S., Ma, H., Zhang, Y. and Li, X. (2023). Visualization and prediction of TVB-N content in chilled pork by HSI. *Food Science.*

Xu, Z., Chen, S., Li, J., Yao, H., Zhou, P., Chen, S., Zhao, H., Bai, Y. and Zhao, D. (2025). Research on rapid and non-destructive detection model for pork freshness based on dual-branch hyperspectral feature extraction network combined with machine learning. *Journal of Food Composition and Analysis*, 148, 108506. doi:10.1016/j.jfca.2025.108506

Xu, Z., Han, Y., Zhao, D., Li, K., Li, J., Dong, J., Shi, W., Zhao, H. and Bai, Y. (2024). Research progress on quality detection of livestock and poultry meat based on machine vision, hyperspectral and multi-source information fusion technologies. *Foods*, 13(3), 469. doi:10.3390/foods13030469

Xun, Z., Wang, X., Xue, H., Zhang, Q., Yang, W., Zhang, H., Li, M., Jia, S., Qu, J. and Wang, X. (2024). Deep machine learning identified fish flesh using multispectral imaging. *Current Research in Food Science*, 9, 100784. doi:10.1016/j.crfs.2024.100784

Yao, X., Cai, F., Zhu, P., Fang, H., Li, J. and He, S. (2019). Non-invasive and rapid pH monitoring for meat quality assessment using a low-cost portable hyperspectral scanner. *Meat Science*, 152, 73-80. doi:10.1016/j.meatsci.2019.02.017

Zhang, C., Zhou, L., Zhao, Y., Zhu, S., Liu, F. and He, Y. (2020). Noise reduction in the spectral domain of hyperspectral images using denoising autoencoder methods. *Chemometrics and Intelligent Laboratory Systems*, 203, 104063. doi:10.1016/j.chemolab.2020.104063

Zhang, J., Yu, X., Lei, X. and Wu, C. (2022). A novel CapsNet neural network based on MobileNetV2 structure for robot image classification. *Frontiers in Neurorobotics*, 16, 1007939. doi:10.3389/fnbot.2022.1007939

## Entries to check against the LaTeX source

The following are in the existing reference list but are not cited in the revised text above. They are flagged, not removed, because they may still be cited elsewhere in the LaTeX version of the paper. Check each before deciding: Lu et al. (2021), implantable catheter oximeter; Mäkelä et al. (2020), hyperspectral near infrared image calibration; Ning et al. (2026), multi-task residual 1D CNN; Shaikh et al. (2021), hyperspectral calibration with a low-cost reference; Wu et al. (2012), time series hyperspectral imaging of beef; Zurowietz and Nattkemper (2020), interactive visualization for feature localization.

Nothing in the existing list has been deleted. Where a source is cited in the LaTeX but absent from the revised text above, the citation in the LaTeX stands.

## Statements that could not be sourced

**Microbial contamination of raw meat in Metro Manila wet markets.** The claim as previously written attributed a specific finding to a reference that does not support it. Published studies locate comparable findings in Negros Occidental and in Dasmariñas, Cavite, not Metro Manila. The sentence has been rewritten in the revised Chapter 1 to state the general concern without the unsupported specific. If the team holds the original source, the stronger wording can be restored with that citation.
