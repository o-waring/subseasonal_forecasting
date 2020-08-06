### Folder Structure - subseasonal_forecasting

```
├── subseasonal_forecasting
│   ├── data
│   │   ├── ground_truth
│   │   ├── prediction
│   │   ├── standardization
│   │   ├── training
│   │   ├── sample_output.csv
│   │   └── target_points.csv
│   ├── download
│   │   ├── __init__.py
│   │   ├── get_feature_data_2019.sh
│   │   ├── get_feature_data_2020.sh
│   │   └── get_gt.py
│   ├── models
│   │   ├── prec34_model.hdf5
│   │   ├── prec56_model.hdf5
│   │   ├── tmp34_model.hdf5
│   │   └── tmp56_model.hdf5
│   ├── plotting
│   │   ├── global_local_region_tensor.png
│   │   ├── global_region_tensor.png
│   │   ├── spatial_temporal_model_diagram.png
│   │   ├── usa.r
│   │   └── usa_regional_temperature_plot.png
│   ├── predict
│   │   ├── __init__.py
│   │   ├── interpolate_feature_data.py
│   │   └── prediction_processing.py
│   ├── processing
│   │   ├── __init__.py
│   │   ├── inputs_processing.py
│   │   └── preprocessing.py
│   ├── tests
│   │   ├── __init__.py
│   ├── train
│   │   ├── 3_spatial_temporal_model_emb_prec34.ipynb
│   │   ├── 3_spatial_temporal_model_emb_prec56.ipynb
│   │   ├── 3_spatial_temporal_model_emb_tmp34.ipynb
│   │   └── 3_spatial_temporal_model_emb_tmp56.ipynb
│   ├── utils
│   │   ├── __init__.py
│   │   └── load_functions.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── predict.py
│   └── README.py
├── README.md
└── .gitignore
```
