import joblib
from glob import glob
import numpy as np
import pandas as pd
import sklearn
import os
import tensorflow as tf

# Processing feature dataset functions

# TODO sensibly name and place these files
# Loading functions
from utils.load_functions import load_locations, load_column_names, load_feature_datasets, load_tar_datasets, \
    load_standardizers
# Prediction functions
from predict.interpolate_feature_data import run_interpolation_pipeline
from predict.prediction_processing import process_tar_data, get_prediction_data, prepare_single_spatial_temporal_region, \
    generate_all_region_input, PreprocessTemporalSpatialDataPrediction

# Get params #TODO make some of these user input OR based on update run/full run/check to see if they are already updated
# TODO put inside a constants file
params = {'download_feature_data': False,
          'process_feature_data': False}

if __name__ == "__main__":
    # ---1--- Download, interpolate, process & join data
    print('---1--- Downloading, interpolating, processing & joining data')

    # -1.1- Download 2019 & 2020 feature data
    if params['download_feature_data']:
        os.system("sh download/get_feature_data_2019.sh")
        os.system("sh download/get_feature_data_2020.sh")

    # -1.2- Process & Interpolate feature data
    if params['process_feature_data']:
        run_interpolation_pipeline()

    # -1.3- Load processed feature data
    df = load_feature_datasets()

    # -1.4- Download & process temperature & precipitation data, and join
    pred_input = get_prediction_data(pred_date='2020-07-21', latest_data_date='2020-07-05', data=df)

    # ---2--- Prepare spatial temporal model inputs
    print('---2--- Preparing spatial temporal model inputs')

    # -2.1- Load target locations, standardizers & column names
    locations = load_locations()
    feature_means, feature_stds, feature_scaler, prec_scaler, tmp_scaler = load_standardizers()
    full_col_names = load_column_names()

    # Process prediction input into global padded tensor scaled form
    y = PreprocessTemporalSpatialDataPrediction(scaler=feature_scaler, data_mean=feature_means, data_std=feature_stds,
                                                data=np.array(pred_input), locations=locations,
                                                col_names=full_col_names,
                                                num_regions=514, num_features=15, max_sg=5)
    y.preprocess_pipeline()
    input_tensor = y.global_region_tensor
    rgn_id_vocab, rgn_id_to_int, int_to_rgn_id, target_region_ids = y.get_region_ids()

    # ---3--- Prepare input batches
    print('---3--- Preparing input batches')
    pred_model_input = generate_all_region_input(input_tensor, target_region_ids, rgn_id_to_int)

    spatial_input_b1 = pred_model_input[0][0:256, :, :, :, :]
    temporal_input_b1 = pred_model_input[1][0:256, :, :]
    reg_emb_input_b1 = pred_model_input[2][0:256, :, :]
    spatial_input_b2 = pred_model_input[0][256:512, :, :, :, :]
    temporal_input_b2 = pred_model_input[1][256:512, :, :]
    reg_emb_input_b2 = pred_model_input[2][256:512, :, :]
    spatial_input_b3 = pred_model_input[0][258:, :, :, :, :]
    temporal_input_b3 = pred_model_input[1][258:, :, :]
    reg_emb_input_b3 = pred_model_input[2][258::, :]


    def batch_gen(spatial_data, temporal_data, reg_emb_data):
        yield ({"spatial_input": spatial_data, "temporal_input": temporal_data, "region_id_input": reg_emb_data})


    BATCH_SIZE, SEQ_LEN, SPATIAL_WIDTH, SPATIAL_FEATURES, TEMPORAL_FEATURES = 256, 26, 11, 8, 4
    b1_generator = tf.data.Dataset.from_generator(
        generator=lambda: batch_gen(spatial_input_b1, temporal_input_b1, reg_emb_input_b1),
        output_types=({"spatial_input": np.float16, "temporal_input": np.float16, "region_id_input": np.int16}),
        output_shapes=({"spatial_input": [BATCH_SIZE, SEQ_LEN, SPATIAL_WIDTH, SPATIAL_WIDTH, SPATIAL_FEATURES],
                        "temporal_input": [BATCH_SIZE, SEQ_LEN, TEMPORAL_FEATURES],
                        "region_id_input": [BATCH_SIZE, SEQ_LEN, 1]}))
    b2_generator = tf.data.Dataset.from_generator(
        generator=lambda: batch_gen(spatial_input_b2, temporal_input_b2, reg_emb_input_b2),
        output_types=({"spatial_input": np.float16, "temporal_input": np.float16, "region_id_input": np.int16}),
        output_shapes=({"spatial_input": [BATCH_SIZE, SEQ_LEN, SPATIAL_WIDTH, SPATIAL_WIDTH, SPATIAL_FEATURES],
                        "temporal_input": [BATCH_SIZE, SEQ_LEN, TEMPORAL_FEATURES],
                        "region_id_input": [BATCH_SIZE, SEQ_LEN, 1]}))
    b3_generator = tf.data.Dataset.from_generator(
        generator=lambda: batch_gen(spatial_input_b3, temporal_input_b3, reg_emb_input_b3),
        output_types=({"spatial_input": np.float16, "temporal_input": np.float16, "region_id_input": np.int16}),
        output_shapes=({"spatial_input": [BATCH_SIZE, SEQ_LEN, SPATIAL_WIDTH, SPATIAL_WIDTH, SPATIAL_FEATURES],
                        "temporal_input": [BATCH_SIZE, SEQ_LEN, TEMPORAL_FEATURES],
                        "region_id_input": [BATCH_SIZE, SEQ_LEN, 1]}))
