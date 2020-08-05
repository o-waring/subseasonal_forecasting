import numpy as np
import pandas as pd
import datetime

from utils.load_functions import load_tar_datasets, load_column_names


def process_tar_data(pred_var, weeks, col):
    assert col in ['tmp','prec']
    assert pred_var in ['temp34','prec34']
    tmp = load_tar_datasets(pred_var)
    tmp_list = []
    for rgn in tmp['region_id'].unique():
        region_tmp = pd.DataFrame(tmp[tmp['region_id']==rgn].iloc[0,-weeks:].values, columns=[col])
        region_tmp['lat'] = rgn[0]
        region_tmp['lon'] = rgn[1]
        tmp_list.append(region_tmp)
    tmp_df = pd.concat(tmp_list, axis=0, ignore_index=False)
    tmp_df = tmp_df.reset_index(drop=True)
    return tmp_df


def get_prediction_data(pred_date, latest_data_date, data, seq_len=26):
    """ Get input data for current target week from features dataset.
        Note start date for these is two weeks behind current date.
        Note usually data is not available for latest day - in which case take latest
            available data date for last data point in input sequence.
        Get temp/prec data using get_gt.py method, and join to features datasets.
        Note - for a given date, these interpolate and 2 week average the values 3/4 weeks ahead of date.
        Therefore we process differently - these start dates are 28 days behind current prediction date. """
    # Features data
    prediction_date = datetime.datetime.strptime(pred_date,'%Y-%m-%d')
    last_start_date = prediction_date - datetime.timedelta(days=14)
    last_available_start_date = datetime.datetime.strptime(latest_data_date,'%Y-%m-%d')
    pred_date_index = [last_available_start_date] + [last_start_date - datetime.timedelta(days=x) for x in range(14,14*seq_len,14)]
    data = data[data.start_date.isin(pred_date_index)].reset_index(drop=True)
    # Temp/prec data
    tmp = process_tar_data('temp34', seq_len, 'tmp')
    prec = process_tar_data('prec34', seq_len, 'prec')
    # Join all
    data['precip'] = prec['prec']
    data['tmp2m'] = tmp['tmp']
    full_col_names = load_column_names()
    data = data[full_col_names]
    return data


def prepare_single_spatial_temporal_region(dataset, target_region, rgn_id_to_int, seq_len, sg, region_emb=True):
    """  """
    # Crop local region
    local_region = dataset[:, (target_region[0] - sg):(target_region[0] + sg + 1),
                   (target_region[1] - sg):(target_region[1] + sg + 1), :]
    # Spatial data
    spatial_data = local_region[:, :, :, 3:-4].astype(np.float16)
    # Temporal data
    if region_emb:
        local_reg_id = str((int(local_region[0, sg, sg, 1]), int(local_region[0, sg, sg, 2])))
        local_reg_id = rgn_id_to_int[local_reg_id]
        temporal_data = local_region[:, sg, sg, -4:].astype(np.float16)
        region_embedding = np.repeat(local_reg_id, seq_len).reshape(seq_len, 1).astype(np.int16)
        return np.array(spatial_data), np.array(temporal_data), np.array(region_embedding)
    # print(temporal_data.shape, region_embedding.shape)
    else:
        temporal_data = local_region[:, sg, sg, 1:].astype(np.float16)
        return np.array(spatial_data), np.array(temporal_data)


def generate_all_region_input(input_tensor, target_region_ids, rgn_id_to_int):
    spatial_data, temporal_data, reg_emb_data = [], [], []
    for rgn in target_region_ids:
        # Generate single window
        spt, tmp, remb = prepare_single_spatial_temporal_region(input_tensor, rgn, rgn_id_to_int, seq_len=26, sg=5)
        spatial_data.append(spt)
        temporal_data.append(tmp)
        reg_emb_data.append(remb)
    spatial_data = np.stack(spatial_data)
    temporal_data = np.stack(temporal_data)
    reg_emb_data = np.stack(reg_emb_data)

    return spatial_data, temporal_data, reg_emb_data


class PreprocessTemporalSpatialDataPrediction:
    """
    Class for conducting preprocessing pipeline for temporal spatial data for model prediction
    Standardizes feature fields, and scales all categorical feature fields using existing scaler, means, stds
    Transforms dataset to spatial form [timesteps,lat,lon,features]
    Creates missing regions to enable above matrix transformation (filling in missing value with 0 - note this aligns well with [0,1] scaling)
    """

    def __init__(self, scaler, data_mean, data_std, data: np.array, locations: np.array, col_names: list,
                 num_regions: int, num_features: int, max_sg: int = 5):
        self.scaler = scaler
        self.data_mean = data_mean
        self.data_std = data_std
        self.data = data
        self.locations = locations
        self.regions = self.locations['region_id'].unique()
        self.latmin, self.latmax, self.lonmin, self.lonmax = self.locations['lat'].min(), self.locations['lat'].max(), \
                                                             self.locations['lon'].min(), self.locations['lon'].max()
        self.bin_width, self.bin_height = self.lonmax - self.lonmin + 1, self.latmax - self.latmin + 1
        self.col_names = col_names
        self.num_regions = num_regions
        self.num_features = num_features
        self.weather_features = 8
        self.cyclical_features = 4
        self.num_timesteps = self.data.reshape(self.num_regions, -1, self.num_features).shape[1]
        self.max_sg = max_sg

    def standardize_and_scale_data(self):
        """ Standarize features using mean, std from train set; then scale to 0-1 scale """
        # Ensure dataset is order by start date, by region (lat, then lon)
        #         self.data = self.data.sort_values(by=['start_date', 'lat','lon']).reset_index(drop=True)
        # Reshape all regions together
        self.data = np.array(self.data).reshape(-1, self.num_features)
        # Extract train_split - note we only standardize using mean and std from train set
        TRAIN_SPLIT = self.num_timesteps  # - 2*1008*self.num_regions

        # Start Date - need this for indexing/grouping by region
        date = self.data[:, 0].reshape((-1, 1))

        # Keep lat/lon from scaling
        lat_lon = self.data[:, 1:3].astype(np.float16)

        # Standardize feature fields using training means & stds
        features = self.data[:, 3:-4].astype(np.float32)
        features = ((features - self.data_mean) / self.data_std).astype(np.float16)

        # Deal with cyclical features separately - these are already scaled
        cyclical_features = self.data[:, -4:].astype(np.float16)

        # All features - stack and scale
        all_features = np.hstack((features, cyclical_features))

        # Scale feature data to 0-1 scale using training scaler
        scaler = self.scaler
        all_features = scaler.fit_transform(all_features)

        # Recombine & Reshape
        self.data = np.hstack((date, lat_lon, all_features))
        self.data = self.data.reshape(-1, self.num_features)

    def process_datetime(self, dt_fmt: str = '%Y-%m-%d', datetimecol='start_date'):
        # print("Parsing datetime fields \n")
        def lookup(s):
            """
            This is an extremely fast approach to datetime parsing.
            """
            dates = {date: pd.to_datetime(date, format=dt_fmt) for date in s.unique()}
            return s.map(dates)

        self.data[datetimecol] = lookup(self.data[datetimecol])
        self.all_dates = np.unique(self.data["start_date"].dt.strftime('%Y-%m-%d'))

    def get_missing_regions(self):
        """ Find regions in lat-lon box which are not modelled geographic regions  """
        self.all_region_ids = [(lat, lon) for lat in range(self.latmin, self.latmax + 1) for lon in
                               range(self.lonmin, self.lonmax + 1)]
        self.num_total_regions = len(set(self.all_region_ids))
        self.missing_regions = list(set(self.all_region_ids) - set(list(self.regions)))
        self.missing_regions.sort()

    def mask_missing_regions(self, mask_value=0):
        """ Create masked data for missing region - zero pad (0,1) scaled features """
        self.get_missing_regions()
        masked_rgn_lst = []
        for rgn in self.missing_regions:
            date_col = self.data.reshape(self.num_regions, -1, self.num_features)[0, :, 0].reshape(self.num_timesteps,
                                                                                                   1)  # take same dates
            lat_col = np.array([rgn[0]] * self.num_timesteps).reshape(self.num_timesteps,
                                                                      1)  # take lat of current region
            lon_col = np.array([rgn[1]] * self.num_timesteps).reshape(self.num_timesteps,
                                                                      1)  # take lon of current region
            feature_cols = np.array([mask_value] * self.num_timesteps * (self.weather_features)).reshape(
                self.num_timesteps, self.weather_features)  # mask weather features
            cyclical_cols = self.data.reshape(self.num_regions, -1, self.num_features)[0, :,
                            -self.cyclical_features:].reshape(self.num_timesteps,
                                                              self.cyclical_features)  # take time features
            masked_rgn = np.hstack((date_col, lat_col, lon_col, feature_cols, cyclical_cols))
            masked_rgn_lst.append(masked_rgn)
        masked_rgns = np.concatenate(masked_rgn_lst)
        self.masked_rgns_df = pd.DataFrame(masked_rgns)
        self.masked_rgns_df.columns = self.col_names
        self.masked_rgns_df['region_id'] = list(
            zip(self.masked_rgns_df['lat'].astype(int), self.masked_rgns_df['lon'].astype(int)))
        self.masked_rgns_df['model_region'] = False

    def convert_rgns_data_to_df(self):
        """ Convert to df of shape (num_timesteps*num_regions, num_features). Ordered by region by timestep """
        self.data = pd.DataFrame(self.data.reshape(-1, self.num_features))
        self.data.columns = self.col_names
        # Create unique region id
        self.data['region_id'] = list(zip(self.data['lat'].astype(int), self.data['lon'].astype(int)))
        self.data['model_region'] = True

    def join_masked_regions(self):
        """ Join masked regions df to rgns data df, and sort """
        self.data = pd.concat([self.data, self.masked_rgns_df])
        # Convert date to datetime
        self.process_datetime()
        # Sort
        self.data = self.data.sort_values(by=['lat', 'lon', 'start_date']).reset_index(drop=True)
        assert self.num_total_regions == len(self.data['region_id'].unique())

    def get_global_region_tensor(self, save=False):
        print("Generating global spatial grid \n")
        spatial_tw_list = []
        #         self.all_dates = np.sort(self.data['start_date'].unique())
        # For every timestep, create binsize*binsize global spatial grid of demand
        for counter, time_window in enumerate(self.all_dates):
            mask = self.data['start_date'] == np.datetime64(time_window)
            pvt_current_time_window = np.flipud(np.array(self.data[mask]).reshape(self.bin_height, self.bin_width, -1))
            spatial_tw_list.append(pvt_current_time_window)
        # Store global_regions_tensor - stack as tensor: for bin_index [num_timesteps, bin_width , bin_height, number of channels]
        self.global_region_tensor = np.stack(spatial_tw_list).reshape(-1, self.bin_height, self.bin_width,
                                                                      pvt_current_time_window.shape[2])
        self.global_region_tensor = self.global_region_tensor[:, :, :, :-2]

        # Pad outer edge with max spatial granularity
        npad = ((0, 0), (self.max_sg, self.max_sg), (self.max_sg, self.max_sg),
                (0, 0))  # pad (before, after) on region height/width dimensions only
        self.global_region_tensor = np.pad(self.global_region_tensor, pad_width=npad, mode='constant',
                                           constant_values=0)
        # Save to disk
        if save:
            print("Saving global spatial tensor \n")
            np.save('data/processed/spatial_temporal/global_region_tensor_scaled_sg' + str(self.max_sg),
                    self.global_region_tensor)

    def get_target_regions(self):
        self.target_regions = self.locations['region_id'].unique()
        self.target_region_ids = [(self.locations['lat'].max() - region[0] + self.max_sg,
                                   region[1] - self.locations['lon'].min() + self.max_sg) for region in
                                  self.target_regions]

    def get_region_ids(self):
        self.get_target_regions()
        self.rgn_id_vocab = [str(region) for region in self.target_regions]
        self.rgn_id_to_int = {rgn_id: i for i, rgn_id in enumerate(self.rgn_id_vocab)}
        self.int_to_rgn_id = {i: rgn_id for i, rgn_id in enumerate(self.rgn_id_vocab)}
        return (self.rgn_id_vocab, self.rgn_id_to_int, self.int_to_rgn_id, self.target_region_ids)

    def preprocess_pipeline(self):
        self.standardize_and_scale_data()
        self.mask_missing_regions()
        self.convert_rgns_data_to_df()
        self.join_masked_regions()
        self.get_global_region_tensor()
        del self.data