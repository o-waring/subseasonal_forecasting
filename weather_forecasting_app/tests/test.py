import joblib
from glob import glob
import numpy as np
import pandas as pd
import sklearn

from utils.load_functions import load_tar_datasets, load_column_names, load_feature_datasets, load_locations, load_standardizers

if __name__ == "__main__":

    prec_df = load_tar_datasets(pred_var='temp34')

    locs = load_locations()

    col_names = load_column_names(features_only=True)

    # feat_df = load_feature_datasets() #TODO

    feature_means, feature_stds, feature_scaler, prec_scaler, tmp_scaler = load_standardizers()

    print('\n',feature_means,'\n', feature_stds,'\n', feature_scaler,'\n', prec_scaler,'\n', tmp_scaler)