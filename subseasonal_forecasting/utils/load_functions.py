import joblib
from glob import glob
import numpy as np
import pandas as pd


def load_tar_datasets(pred_var, file_dir='data/ground_truth/'):
    """ Assumes all datasets are downloaded """
    assert pred_var in ['temp34', 'prec34'], "pred_var should be either temp34 or prec34"
    all_files = glob(file_dir + "gt*.csv")
    assert len(all_files) > 0, "No files found - check directory"
    all_files.sort()
    file_list = []
    for i, filename in enumerate(all_files):
        if i == 0:
            tp_df = pd.read_csv(filename, index_col=None, header=0)
            tp_df['region_id'] = list(zip(tp_df['lat'], tp_df['lon']))
            tp_df = tp_df[['lat', 'lon', 'region_id', pred_var]].copy()
            tp_df.columns = ['lat', 'lon', 'region_id'] + [str(col) + '_' + str(filename[6:-4]) for col in
                                                           tp_df.columns[3:]]
            file_list = [tp_df]
        else:
            tp_df = pd.read_csv(filename, index_col=None, header=0)
            tp_df = tp_df[[pred_var]].copy()
            tp_df.columns = [str(col) + '_' + str(filename[6:-4]) for col in tp_df.columns]
            file_list.append(tp_df)

    frame = pd.concat(file_list, axis=1, ignore_index=False)
    return frame


def load_column_names(features_only=False):
    """ Loads column names for processed input array """
    col_names = pd.read_csv('data/standardization/column_names.csv')
    full_col_names = list(col_names['0'])
    col_names = list(col_names['0'])
    col_names.remove('tmp2m'), col_names.remove('precip')
    if features_only:
        return col_names
    else:
        return full_col_names


def load_feature_datasets(filepath='data/prediction_inputs_raw/processed_features.npy'):
    features_df = np.load(filepath, allow_pickle=True)
    features_df = pd.DataFrame(features_df)
    col_names = load_column_names(features_only=True)
    features_df.columns = col_names
    return features_df


def load_locations(filepath='data/target_points.csv'):
    locations_df = pd.read_csv(filepath)
    locations_df['region_id'] = list(zip(locations_df['lat'], locations_df['lon']))
    return locations_df


def load_standardizers(rootdir='data/standardization/'):
    feature_means = np.array(np.load(rootdir + 'feature_means.npy'))
    feature_stds = np.array(np.load(rootdir + 'feature_stds.npy'))
    feature_scaler = joblib.load(rootdir + 'all_feature_scaler.pkl', "r")
    prec_scaler = joblib.load(rootdir + 'prec_scaler.pkl', "r")
    tmp_scaler = joblib.load(rootdir + 'temp_scaler.pkl', "r")
    return feature_means, feature_stds, feature_scaler, prec_scaler, tmp_scaler
