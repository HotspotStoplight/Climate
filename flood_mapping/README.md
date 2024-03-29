# Climate Modeling for HotspotStoplight

## Repo Overview
This repo contains the code for the climate modeling for HotspotStoplight. The code is written in Python and relies primarily on Google Earth Engine via the `geemap` package. Dependencies are managed by `poetry` and installation instructions are in the `SETUP.md` file. 

## Flood Modeling
Flood modeling is the first module that we are working for HotspotStoplight. The code is organized in the `data` folder. Data are stored in the `/inputs` and `/outputs` subdirectories, while the `/src` folder contains an ETL pipeline to process Earth Engine data, train a random forest model, and apply the model to predict flood risk for a given area (in this first use case, for Costa Rica).

## Attribution
The Sentinel-1 workflow was [adapted from UN-Spider](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step)

Code to import geoboundaries via python was adapted from [`pygeoboundaries`](https://github.com/ibhalin/pygeoboundaries?tab=readme-ov-file)

# Writeup

## Introduction

Part of a broader project to get climate hazards data at 30m resolution (give or take) for San Jose, Costa Rica

"Flood hazard maps are designed to indicate the probability of flooding over space and serve as a critical decision-making tool for a range of end users including building/infrastructure developers and disaster response planners. However, such maps do not exist for much of the developing world due to the extremely high data and computational requirements of the engineering hydraulic models that have traditionally been used in their production." (Sampson et al., 2015)

### Flood Probability Mapping

"Below we have identified six key challenges that need to be solved to enable a global flood model to be built. The bracketed colors in the list below associate each challenge with a region of the methodological flowchart (Figure 1).

    Global terrain data (green).
    Extreme flow generation (blue).
    Global river network and geometry (yellow).
    Flood defenses (purple).
    Computational hydraulic engine (orange).
    Automation framework (red).
" (Sampson et al., 2015)

Of these, terrain, river network, and automation are most relevant to us. We don't have a map of flood defenses, and we are not using a computational hydraulic engine. However, the machine learning model might allow us to compensate.

#### Overview

All processing is carried out with the Google Earth Engine Python API. Data are stored in Google Cloud.

#### Flood Labels
Known flood events for a given country are pulled from the [EMDAT International Disaster Database](https://doc.emdat.be/). Following [UN-Spider's recommended practices](https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step), we use Sentinel-1 SAR data to derive flooded pixel labels.

#### Conditioning Factors
We assemble conditioning factors from the [Google Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog) and the [Google Earth Engine Community Catalog](https://gee-community-catalog.org/). We base our layers on the following two papers:

- Xinxiang Lei, Wei Chen, Mahdi Panahi, Fatemeh Falah, Omid Rahmati, Evelyn Uuemaa, Zahra Kalantari, Carla Sofia Santos Ferreira, Fatemeh Rezaie, John P. Tiefenbacher, Saro Lee, Huiyuan Bian, Urban flood modeling using deep-learning approaches in Seoul, South Korea, Journal of Hydrology, Volume 601, 2021, 126684, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2021.126684.
- Kalantar, Bahareh, Naonori Ueda, Vahideh Saeidi, Saeid Janizadeh, Fariborz Shabani, Kourosh Ahmadi, and Farzin Shabani. 2021. "Deep Neural Network Utilizing Remote Sensing Datasets for Flood Hazard Susceptibility Mapping in Brisbane, Australia" Remote Sensing 13, no. 13: 2638. https://doi.org/10.3390/rs13132638 

Accordingly, we use:
- DEM
- Slope
- the Global Human Settelemt Layer (GHSL)
- Flow Direction
- Stream Distance Proximity
- Flow Accumulation
- Stream Power Index
- Topographic Position Index
- Compound Topographic Index
- Terrain Position Index
- Planform Curvature
- Tangential Curvature
- Aspect
- Terrain Ruggedness Index
- Maximum precipitation in the previous year

Following Hawker et al. (2022), we use FABDEM (Copernicus 30m DEM with buildings and forests removed) as our DEM, deriving slope from it as well.

#### Model Training
Currently, we are using an out-of-the-box random forest model implemented in Google Earth Engine. For a given country, we train the model on a stratified sample of 100,000 pixels from all known flood events within the bounding box. We then test and validate on a 60/20/20 split of the sample.

#### Validation
We are in the process of coming up with a method of formally validating our results. For the moment, we have simply spot-checked our results against several data sources, including:
- This map of flood risk zones across the European Union: https://discomap.eea.europa.eu/floodsviewer/
- This map of flood risk zones across Nicaragua: https://servidormapas.ineter.gob.ni/mapstore/#/viewer/openlayers/57
- This global 250m resolution floodplains map: https://gee-community-catalog.org/projects/gfplain250/

From Sampson et al. (2015): "The global model is shown to capture between two thirds and three quarters of the area determined to be at risk in the benchmark data without generating excessive false positive predictions. When aggregated to ∼1 km, mean absolute error in flooded fraction falls to ∼5%."

Sampson et al. (2015) also give extensive explanation of validation metrics, e.g., hit rate, false alarm ratio, and critical success rate (aka the "fit" statistic).

They benchmark their data on [Alberta State flood models](https://www.arcgis.com/apps/mapviewer/index.html?panel=gallery&suggestField=true&layers=b3c6842b559b4297825c369c53a89249) and [UK data afor the Severn and Thames catchments](https://flood-map-for-planning.naturalresources.wales/) from the Environmental Agency of England and Wales. [In another paper](https://www.fathom.global/insight/fathom-us-model-iowa-flood-center/), they validated on [Iowa state flood from the Iowa Flood Center](https://ifis.iowafloodcenter.org/ifis/).

*Follow this process for validation.*

[(More on validation from Fathom)](https://www.fathom.global/insight/fathom-us-model-iowa-flood-center/)

### Vulnerability
To assess risk, we calcualte vulnerability as a function of flood probability, population density, and relative wealth (derived from the Human Development Index and gridded GDP PPP):

![Vulnerability Calculation](https://github.com/HotspotStoplight/Climate/blob/main/flood_mapping/public/vulnerability_calc.png)

We follow Smith et al. (2019) in using the High Resolution Settlement Layer (HRSL) for population density, and use HDI and PPP from the World Bank.

## Results
Find an accuracy ranging from 77% to 97%, with a true positive rate of between 77% and 99%, and a false positive rate from 0.6% to 20%.

## Discussion
Model is good! Needs some fine-tuning for sure but has a lot of potential to advance things. Validation is super important. Better historic training data would be good--deriving things from SAR is 1) already a model and 2) limited to dates from 2014 onwards (and sometimes later in some countries). Also not a perfectly representative sampling of knowing flood events.

### Model Applicability

### Limitations

#### Model Uncertainty
- Deriving flood labels from SAR data; already uncertain
- Constructing a model on top of global datasets, which are themselves uncertain
- Training the model on a limited sample of known flood events (doesn't go back before 2016)
- Some datasets are low resolution (e.g., 25km precip data)

## Conclusions

## References

Facebook Connectivity Lab and Center for International Earth Science Information Network - CIESIN - Columbia University. 2016. High Resolution Settlement Layer (HRSL). Source imagery for HRSL Copyright 2016 DigitalGlobe. Accessed DAY MONTH YEAR. Data shared under: Creative Commons Attribution International.

Hawker, Laurence, Peter Uhe, Luntadila Paulo, Jeison Sosa, James Savage, Christopher Sampson, and Jeffrey Neal. "A 30m global map of elevation with
forests and buildings removed." Environmental Research Letters (2022).

Kalantar, Bahareh, Naonori Ueda, Vahideh Saeidi, Saeid Janizadeh, Fariborz Shabani, Kourosh Ahmadi, and Farzin Shabani. 2021. "Deep Neural Network Utilizing Remote Sensing Datasets for Flood Hazard Susceptibility Mapping in Brisbane, Australia" Remote Sensing 13, no. 13: 2638. https://doi.org/10.3390/rs13132638 

Smith, A., Bates, P.D., Wing, O. et al. New estimates of flood exposure in developing countries using high-resolution population data. Nat Commun 10, 1814 (2019). https://doi.org/10.1038/s41467-019-09282-y

Xinxiang Lei, Wei Chen, Mahdi Panahi, Fatemeh Falah, Omid Rahmati, Evelyn Uuemaa, Zahra Kalantari, Carla Sofia Santos Ferreira, Fatemeh Rezaie, John P. Tiefenbacher, Saro Lee, Huiyuan Bian, Urban flood modeling using deep-learning approaches in Seoul, South Korea, Journal of Hydrology, Volume 601, 2021, 126684, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2021.126684.

___


*Things to read:*

Yamazaki, D., Kanae, S., Kim, H. & Oki, T. A physically based description of floodplain inundation dynamics in a global river routing model. Water Resour. Res. 47,W04501 (2011).

Pappenberger, F., Dutra, E., Wetterhall, F. & Cloke, H. L. Deriving global flood hazard maps of fluvial floods through a physical model cascade. Hydrol. Earth Syst. Sci. 16, 4143–4156 (2012).

Ward, P. J. et al. Assessing flood risk at the global scale: model setup, results, and sensitivity. Environ. Res. Lett. 8, 044019 (2013).

Alfieri, L. et al. Advances in pan-European flood hazard mapping. Hydrol. Process. 28, 4067–4077 (2014).

Sampson, C. C. et al. A high-resolution global flood hazard model. Water Resour. Res. 51, 7358–7381 (2015).

Dottori, F. et al. Development and evaluation of a framework for global flood hazard mapping. Adv. Water Resour. 94, 87–102 (2016).

Alfieri, L. et al. Global projections of river flood risk in a warmer world. Earths Future 5, 171–182 (2017).

Wing, O. E. et al. Validation of a 30 m resolution flood hazard model of the conterminous United States. Water Resour. Res. 53, 7968–7986 (2017).