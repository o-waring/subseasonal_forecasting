import joblib
from glob import glob
import numpy as np
import pandas as pd
import sklearn
import os

# Processing feature dataset functions


# Loading functions
from processing.load_functions import load_tar_datasets, load_column_names, load_feature_datasets, load_locations, \
    load_standardizers

# Prediction functions

# Get params #TODO make some of these user input OR based on update run/full run/check to see if they are already updated
params = {'download_feature_data': False}

if __name__ == "__main__":
    # ---1---Download, process & interpolate feature datasets
    # Download 2019 & 2020 feature data
    # if params['download_feature_data']:
    #     os.system("sh get_feature_data_2019.sh")
    #     os.system("sh get_feature_data_2020.sh")
    #
    # # Process & Interpolate feature datasets
    # os.system("python3 src/prediction/process_interpolate_feature_data.py")
    locs = load_locations()
    print(locs)
    # print('\n',feature_means,'\n', feature_stds,'\n', feature_scaler,'\n', prec_scaler,'\n', tmp_scaler)
