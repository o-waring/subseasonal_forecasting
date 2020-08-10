"""
Spatial-Temporal Models - temp34 Model
***Note this script is adapted from a colab notebook to demonstrate required code for model construction in Tensorflow 2.0
This script will not run as standalone***
"""

import numpy as np
import datetime
from math import floor
import tensorflow as tf
print("tf version:",tf.__version__)
print(tf.test.gpu_device_name())
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Conv2D, LSTM, Flatten, Reshape, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError,nmae
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.load_functions import load_locations, load_column_names

## Load Data ##
global_region_tensor = np.load('../data/prediction/global_region_tensor_scaled_sg5.npy', allow_pickle=True)
col_names = load_column_names()
target_regions = load_locations()

## Get Params ##
# Rodeo Track Params
SPRINT_WEEK = 21
TAR_VAR = 'tmp'
TAR_WEEK = '34'
PRED_TAR = TAR_VAR+TAR_WEEK
TAR_INDEX = {'tmp':2, 'prec':3}[TAR_VAR]
SEED = 112358

# Data Params
MAX_SG = int(filename.split('_')[-1].split('.')[0].split('sg')[1]) #maximum spatial granularity of local region from input data
NUM_TIMESTEPS = global_region_tensor.shape[0]
NUM_REGIONS = len(target_regions)
NUM_FEATURES = global_region_tensor.shape[3]
TEMPORAL_FEATURES = 4 #NUM_FEATURES-1 #remove date field
SPATIAL_FEATURES = NUM_FEATURES-7 #remove date field, lat, lon, 4*time cyclic fields
PADDED_LAT = global_region_tensor.shape[1]
PADDED_LON = global_region_tensor.shape[2]
TRAIN_SPLIT = NUM_TIMESTEPS - 2*1008
VALID_SPLIT = NUM_TIMESTEPS - 1008

# Tunable Processing Params
SEQ_LEN = 26
STEP = 7
SG = 5
SG = min(SG,MAX_SG) #check spatial granularity is less than maximum allowed by data
# define target region ids
target_region_ids = [(locations['lat'].max()-region[0]+MAX_SG,region[1]-locations['lon'].min()+MAX_SG) for region in target_regions]
SPATIAL_WIDTH = 2*SG+1
BATCH_SIZE = 256
STEPS_PER_EPOCH = floor((NUM_REGIONS*TRAIN_SPLIT - ({'34':2,'56':4}[TAR_WEEK]*14 + 1) - SEQ_LEN*STEP)/BATCH_SIZE)
BUFFER_SIZE = 15000
PREFETCH_SIZE = 5
EPOCHS = 50

### Model Params
# Architecture
CONV2D_LAYERS = 1
CONV1D_LAYERS = 0
LSTM_LAYERS = 2

# Embedding of region id
RGN_ID_EMB_DIM = 8 # from 512 regions

## Layer Params
# Spatial conv layers
SPTL_CONV_FILTERS = 16
SPTL_KERNEL_SIZE = 2
SPTL_STRIDE = 1
SPTL_PADDING = 'same'
SPTL_OUTPUT_DIM = 24 #12
# Conv1D time dimensionality reduction layers
CONV1D_FILTERS = 16
# LSTM sequencing layers
LSTM1_UNITS = 64
LSTM2_UNITS = 64
DROPOUT_RATE = 0.3
REC_DROPOUT_RATE = 0 #else break GPU requirements https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#used-in-the-notebooks_1
FC1_UNITS = 128
FC2_UNITS = 128
MODEL_NAME = PRED_TAR+'_sp-tm_sg'+str(SG)+'_seq'+str(SEQ_LEN)+'_stp'+str(STEP)+'_emb'+str(RGN_ID_EMB_DIM)
TIMESTAMP = '{date:%Y_%m_%d_%H_%M}'.format(date = datetime.datetime.now())

# Define region id mapping
rgn_id_vocab = [str(region) for region in target_regions]
rgn_id_to_int = {rgn_id:i for i, rgn_id in enumerate(rgn_id_vocab)}
int_to_rgn_id = {i:rgn_id for i, rgn_id in enumerate(rgn_id_vocab)}

print("Prediction target: ",PRED_TAR,
      "\nEpochs: ",EPOCHS,
      "\nBatch size: ",BATCH_SIZE,
      "\nSteps per epoch: ",STEPS_PER_EPOCH,
      # "\nValidation steps: ",VALIDATION_STEPS,
      "\nSequence length: ", SEQ_LEN,
      "\nSpatial granularity: ", SG,
      "\nMax spatial granularity: ", MAX_SG,
      "\nModel name: ", MODEL_NAME)

## Transform Input Sequences ##

def window_single_spatial_temporal_target(dataset, target_region, target_index, seq_len, sg, step, tar_var=TAR_VAR,
                                          tar_week=TAR_WEEK, region_emb=True):
    """  """
    # Set future target prediction var based on model scenario
    assert (tar_var in ['tmp', 'prec']) & (tar_week in ['34', '56'])
    # Number of timesteps target is after training history
    target_size = {'34': 2, '56': 4}[tar_week] * 14 + 1
    # Column index of target variable
    target_col = {'tmp': 3, 'prec': 4}[tar_var]

    # Total length of history
    history_size = seq_len * step

    # Crop local region
    local_region = dataset[:, (target_region[0] - sg):(target_region[0] + sg + 1),
                   (target_region[1] - sg):(target_region[1] + sg + 1), :]

    # Set training indices
    start_index = target_index - history_size - target_size
    end_index = target_index - target_size
    #     print(start_index, end_index, target_index)
    assert (start_index > 0) & (end_index > 0), "Timestep {} history out of range".format(target_index)

    indices = range(start_index, end_index, step)
    # Spatial data
    spatial_data = local_region[indices, :, :, 3:-4].astype(np.float16)
    # Temporal data
    if region_emb:
        local_reg_id = str((int(local_region[0, sg, sg, 1]), int(local_region[0, sg, sg, 2])))
        local_reg_id = rgn_id_to_int[local_reg_id]
        temporal_data = local_region[indices, sg, sg, -4:].astype(np.float16)
        region_embedding = np.repeat(local_reg_id, seq_len).reshape(seq_len, 1).astype(np.int16)
        # print(temporal_data.shape, region_embedding.shape)
    elif not region_emb:
        temporal_data = local_region[indices, sg, sg, 1:].astype(np.float16)
    # Target variable
    target_data = np.array(local_region[target_index, sg, sg, target_col]).astype(np.float16)

    return np.array(spatial_data), np.array(temporal_data), np.array(region_embedding), np.array(target_data)


def generate_input_batch(dataset, batch_size, sg, seq_len, step, tar_var, tar_week):
    """ """
    # Set future target prediction var based on model scenario
    assert (tar_var in ['tmp', 'prec']) & (tar_week in ['34', '56'])
    # Number of timesteps target is after training history
    target_size = {'34': 2, '56': 4}[tar_week] * 14 + 1
    # Column index of target variable
    target_col = {'tmp': 3, 'prec': 4}[tar_var]

    # Initialize lists
    spatial_data, temporal_data, reg_emb_data, target_data = [], [], [], []

    # Set minimum time index
    min_time_index = target_size + seq_len * step + 1

    # Generate windowed batch (from random indices and regions)
    for batch in range(batch_size):
        # Generate random region & timestep
        rand_region_idx = np.random.choice(NUM_REGIONS, 1, replace=False)[0]
        rand_region = target_region_ids[rand_region_idx]
        rand_date_idx = np.random.choice(range(min_time_index, NUM_TIMESTEPS), 1, replace=False)[0]
        #         print("Batch number:",batch,"Start Region:",rand_region_idx,"\nStart Region LatLon:",rand_region,"\nStart Timestep:", rand_date_idx)
        # Generate single window
        spt, tmp, remb, tar = window_single_spatial_temporal_target(dataset, rand_region, rand_date_idx, seq_len, sg,
                                                                    step, tar_var, tar_week)
        spatial_data.append(spt)
        temporal_data.append(tmp)
        reg_emb_data.append(remb)
        target_data.append(tar)

    spatial_data = np.stack(spatial_data)
    temporal_data = np.stack(temporal_data)
    reg_emb_data = np.stack(reg_emb_data)
    target_data = np.stack(target_data)

    yield ({"spatial_input": spatial_data, "temporal_input": temporal_data, "region_id_input": reg_emb_data},
           {"target_output": target_data})

## Initiate Generator ##
dataset_generator = tf.data.Dataset.from_generator(
    generator=lambda: generate_input_batch(dataset=global_region_tensor, batch_size=BATCH_SIZE, sg=SG, seq_len=SEQ_LEN, step=STEP, tar_var=TAR_VAR, tar_week=TAR_WEEK),
    output_types=({"spatial_input":np.float16,"temporal_input":np.float16,"region_id_input":np.int16},{"target_output":np.float16}),
    output_shapes=({"spatial_input":[BATCH_SIZE,SEQ_LEN,SPATIAL_WIDTH,SPATIAL_WIDTH,SPATIAL_FEATURES], \
                    "temporal_input":[BATCH_SIZE,SEQ_LEN,TEMPORAL_FEATURES],
                    "region_id_input":[BATCH_SIZE,SEQ_LEN,1]}, \
                    {"target_output":[BATCH_SIZE,]}))
dataset_generator = dataset_generator.prefetch(PREFETCH_SIZE).repeat()
# Test
for batch in dataset_generator.repeat().take(1):
    print(batch[0]['spatial_input'].shape, batch[0]['temporal_input'].shape, batch[0]['region_id_input'].shape, batch[1]['target_output'].shape)

## Build Model ##

def build_sptl_tmpl_model(seq_len=SEQ_LEN, sptl_width=SPATIAL_WIDTH, sptl_features=SPATIAL_FEATURES,
                          tmpl_features=TEMPORAL_FEATURES, batch_size=BATCH_SIZE,
                          sptl_conv_filters=SPTL_CONV_FILTERS, sptl_kernel_size=SPTL_KERNEL_SIZE,
                          sptl_stride=SPTL_STRIDE, sptl_padding=SPTL_PADDING, sptl_output_dim=SPTL_OUTPUT_DIM,
                          rgn_id_vocab=NUM_REGIONS, rgn_id_emb_dim=RGN_ID_EMB_DIM, lstm1_units=LSTM1_UNITS,
                          lstm2_units=LSTM2_UNITS, dropout_rate=DROPOUT_RATE, model_name=MODEL_NAME,
                          print_summary=True):
    # Spatial Model Input
    spatial_input = Input(shape=[seq_len, sptl_width, sptl_width, sptl_features], batch_size=batch_size,
                          name='spatial_input')

    spatial_model = TimeDistributed(Conv2D(filters=sptl_conv_filters, kernel_size=(sptl_kernel_size, sptl_kernel_size),
                                           strides=(sptl_stride, sptl_stride), padding=sptl_padding,
                                           name='sptl_conv2d'), name='td_sptl_conv2d')(spatial_input)

    spatial_model = Flatten(name='sptl_flatten')(spatial_model)

    spatial_model = Reshape(target_shape=(seq_len, -1), name='sptl_reshape')(spatial_model)

    spatial_output = TimeDistributed(Dense(units=sptl_output_dim, activation='relu', name='sptl_emb_out'),
                                     name='td_sptl_emb_out')(spatial_model)

    # Temporal Model Input
    temporal_input = Input(shape=(seq_len, tmpl_features), batch_size=batch_size, name='temporal_input')

    # Region ID Input
    region_id_input = Input(shape=(seq_len, 1), batch_size=batch_size, name='region_id_input')

    region_id_emb = Reshape((seq_len,), input_shape=(seq_len, 1), name='region_id_reshape')(region_id_input)

    region_id_emb = Embedding(input_dim=rgn_id_vocab, input_length=seq_len, mask_zero=False, batch_size=batch_size,
                              output_dim=rgn_id_emb_dim, name='durations_emb_layer')(region_id_emb)

    # Concat Inputs
    concat = K.concatenate([spatial_output, temporal_input, region_id_emb], axis=-1)

    # LSTM Architecture
    lstm_model = LSTM(units=lstm1_units, return_sequences=True, name='lstm_1')(concat)

    lstm_model = LSTM(units=lstm2_units, return_sequences=False, name='lstm_2')(lstm_model)

    model_output = Dense(units=1, name='target_output')(lstm_model)

    model = Model(inputs=[spatial_input, temporal_input, region_id_input], outputs=[model_output], name=model_name)

    if print_summary:
        print(model.summary())

    return (model)

spatial_temporal_model = build_sptl_tmpl_model()


def train_spatial_temporal_model(model, dataset_generator, opt='adam', epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                                 include_tb=False):  # validation_data, val_steps = VALIDATION_STEPS,

    ## Early stopping
    earlystopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, restore_best_weights=True)  # val_loss

    # Automatically save latest best model to file
    filepath = repo_path + "models/model_saves/" + PRED_TAR + '/' + RUN_ID + ".hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    # Set callbacks
    callbacks_list = [checkpoint, earlystopping]

    # Include tensorboard
    if include_tb:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())
        callbacks_list.extend([tensorboard_cb])

    # Optimizers
    optimizers = {'adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)}

    model.compile(loss='mean_absolute_error', optimizer=optimizers[opt],
                  metrics=[mae, RootMeanSquaredError(), Huber()])

    # Fit model #x = [spatial_train, temporal_train_x], y = temporal_train_y,
    history = model.fit(dataset_generator, epochs=epochs, use_multiprocessing=True,
                        # validation_data = validation_data, validation_steps = val_steps,
                        steps_per_epoch=steps_per_epoch, verbose=1, callbacks=callbacks_list)
    return (history)

## Train ##
history = train_spatial_temporal_model(model = spatial_temporal_model, dataset_generator = dataset_generator,
                                       opt = 'adam', steps_per_epoch=500, include_tb=False)

