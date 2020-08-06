# Subseasonal Rodeo

#### **Predicting subseasonal temperature & precipitation 3-4 and 5-6 weeks ahead for Western USA across 514 geographical regions**

#### Introduction

Water and fire managers in Western USA rely on subseasonal (2-6 weeks ahead) forecasts to allocate water resources, manage wildfires, and prepare for droughts and other weather extremes [1]. Physical PDE based models typically dominate shorter term weather forecasts, however these become more chaotic in nature for subseasonal time frames. The abundant availability of meterological records (https://www.noaa.gov/) and computational resources allows for statistical and machine learning approaches to be employed to improve the skill of the longer term forecasts. In colaboration with the US Government and leading US Universities, Topcoder hosted a competitive challenge running through 2020 (Subseasonal Rodeo) to incentivize community engagement and stimulate development in this important area.

This repo outlines a hybrid spatial-temporal deep learning approach to the subseasonal forecasting task.

#### Purpose

The objective of this modelling effort is to forecast 2 week average temperature, and 2 week total precipitation for 3-4 and 5-6 weeks forecast horizons - subseasonal forecasts - across 514 regional geographies in Western USA. This resuts in four modelling challenges - 

- temp34
- temp56
- prec34
- prec56

The geographical regions lie in the Western contiguous USA (not including Hawaii & Alaska), bounded by latitudes 25N to 50N and longitudes 125W to 93W, at a 1◦ by 1◦ resolution. A sample prediction across these geographical regions is shown below. The scoring metric is RMSE averaged across all 514 regions, for a given forecast horizon.

#### USA Regional Temperature Plot
![USA Regional Temperature Plot](subseasonal_forecasting/plotting/usa_regional_temperature_plot.png)

#### Data




#### Spatial Temporal Model Diagram
![Spatial Temporal Model Diagram](subseasonal_forecasting/plotting/spatial_temporal_model_diagram.png)





###### [1] *Improving Subseasonal Forecasting in the Western U.S. with Machine Learning - https://arxiv.org/pdf/1809.07394.pdf*
