# Subseasonal Rodeo

### **Predicting subseasonal temperature & precipitation 3-4 and 5-6 weeks ahead for Western USA across 514 geographical regions**

### Introduction

Water and fire managers in Western USA rely on subseasonal (2-6 weeks ahead) forecasts to allocate water resources, manage wildfires, and prepare for droughts and other weather extremes [1]. Physical PDE based models typically dominate shorter term weather forecasts, however these become more chaotic in nature for subseasonal time frames. The abundant availability of meterological records (https://www.noaa.gov/) and computational resources allows for statistical and machine learning approaches to be employed to improve the skill of the longer term forecasts. In colaboration with the US Government and leading US Universities, Topcoder hosted a competitive challenge running through 2020 (Subseasonal Rodeo) to incentivize community engagement and stimulate development in this important area.

This repo outlines a hybrid spatial-temporal deep learning approach to the subseasonal forecasting task.

### Purpose

The objective of this modelling effort is to forecast 2 week average temperature, and 2 week total precipitation for 3-4 and 5-6 weeks forecast horizons - subseasonal forecasts - across 514 regional geographies in Western USA. This resuts in four modelling challenges - 

- temp34
- temp56
- prec34
- prec56

The geographical regions lie in the Western contiguous USA (not including Hawaii & Alaska), bounded by latitudes 25N to 50N and longitudes 125W to 93W, at a 1° by 1° resolution. A sample prediction across these geographical regions is shown below. The scoring metric is RMSE averaged across all 514 regions, for a given forecast horizon.

#### USA Regional Temperature Plot
![USA Regional Temperature Plot](subseasonal_forecasting/plotting/usa_regional_temperature_plot.png)

#### Data

Data is sourced from NOAA - the National Oceanic and Atmospheric Administration - with data available for all geographical regions from 1979 to 2020. These data sources are updated daily, as such forecasts can be made 3-4 and 5-6 weeks ahead with minimal data lag. Where available at a different lat lon resolution, the data is interpolated to a 1° by 1° resolution to fit the target prediction grid.

The following data sources are used in this study as meterological input features - 

- Temperature - target and feature variable
- Precipitation - target and feature variable
- Relative humidity at surface (rhum) - feature variable
- Geopotential height (hgt10) - feature variable
- Potential evaporation rate (pevpr) - feature variable
- Precipitable water (pr_wtr) - feature variable
- Pressure (pres) - feature variable
- Sea level pressure (slp) - feature variable

Target variables are sourced using subseasonal_forecasting/download/(get_gt.py, get_target_data.py); feature variables are sourced using subseasonal_forecasting/download/get_feature_data_(2019, 2020).sh.

### Spatial Temporal Modelling Approach

#### Processing

All datasets were downloaded, interpolated to 1° by 1° resolution for the 514 required geographies, converted to 2 week averages (or totals) and merged on unique start date and region lat lon.

Cyclic features for date fields are added - with cosine and sin components taken for month of year and day of year.

A unique region_id string was added to represent the region uniquely, resulting in a 514 class categorical feature.

All 8 continuous weather features were standardized by subtracting mean and dividing by std of training dataset, before being scaled to [0,1] using a MinMaxScaler.

#### Spatial Temporal Model Diagram
![Spatial Temporal Model Diagram](subseasonal_forecasting/plotting/spatial_temporal_model_diagram.png)





###### [1] *Improving Subseasonal Forecasting in the Western U.S. with Machine Learning - https://arxiv.org/pdf/1809.07394.pdf*
