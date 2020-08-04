"""
 Script - interpolate_feature_data.py
 Overview - extracts relevant lat/lon regional data from netCDF4 datasets for weather feature datasets - namely:
 hgt, pres, pevpr, slp, rhum, pr_wtr. See https://arxiv.org/pdf/1809.07394.pdf Appendix A for data sources.
 Note this are downloaded for 2019 and 2020 using get_feature_data_2019(2020).sh

 Lat/lon regions defined by target_points.csv; these prescribe Western USA box regions of size 1 lat, 1 lon
 Loaded datasets for weather features come at a different scale, so these are interpolated to the unit lat lon scale
 required for this modelling exercise.
 Two week averages are taken from a given start date.
 All datasets are downloaded and processed for 2019 and 2020 (to date), interpolated, 2-week averaged, and merged,
 with date based features added. Full pipeline saves processed_features.npy to disc.

 """

import datetime
import netCDF4
from glob import glob
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
from utils.load_functions import load_locations


## Functions

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''

    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr, \
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def interpolate_lat_lon(grid_x, grid_y, points, values):
    grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    return (grid_z1[:, :, 0])


def extract_region_data(dataset_name, dataset_dict, buffer=2):
    intp_data_list = []
    grid_x, grid_y = np.mgrid[lat_max:lat_min:23j, lon_min:lon_max:31j]
    for year in ['2019', '2020']:
        print("-",year)
        nc = netCDF4.Dataset(file_path + dataset_dict[dataset_name]['fp_prfx'] + '.' + year + '.nc')

        lat_south = np.argwhere((np.array(nc['lat']) >= dataset_dict[dataset_name]['lat_min'] - buffer) & \
                                (np.array(nc['lat']) <= dataset_dict[dataset_name]['lat_max'] + buffer)).min()
        lat_north = np.argwhere((np.array(nc['lat']) >= dataset_dict[dataset_name]['lat_min'] - buffer) & \
                                (np.array(nc['lat']) <= dataset_dict[dataset_name]['lat_max'] + buffer)).max()
        lon_east = np.argwhere((np.array(nc['lon']) >= dataset_dict[dataset_name]['lon_min'] - buffer) & \
                               (np.array(nc['lon']) <= dataset_dict[dataset_name]['lon_max'] + buffer)).min()
        lon_west = np.argwhere((np.array(nc['lon']) >= dataset_dict[dataset_name]['lon_min'] - buffer) & \
                               (np.array(nc['lon']) <= dataset_dict[dataset_name]['lon_max'] + buffer)).max()

        lat_list = [lat for lat in np.array(nc['lat']) if (lat >= dataset_dict[dataset_name]['lat_min'] - buffer) & \
                    (lat <= dataset_dict[dataset_name]['lat_max'] + buffer)]
        lon_list = [lon for lon in np.array(nc['lon']) if (lon >= dataset_dict[dataset_name]['lon_min'] - buffer) & \
                    (lon <= dataset_dict[dataset_name]['lon_max'] + buffer)]

        # get lat lon grid from current dataset
        points = np.array([(lat, lon) for lat in lat_list for lon in lon_list])

        # If multiple levels in dataset, get required level
        if 'level' in nc.variables.keys():
            level_idx = np.argwhere((np.array(nc['level']) == dataset_dict[dataset_name]['level'])).max()
            # Return data from all time steps for lat lon range at given level
            arr = np.array(nc[dataset_name][:, level_idx, lat_south:lat_north + 1, lon_east:lon_west + 1])
            intp_data = []

        else:
            # Return data from all time steps for lat lon range
            arr = np.array(nc[dataset_name][:, lat_south:lat_north + 1, lon_east:lon_west + 1])

        # Interpolate data to required lat lon grid for each time step, stack and return
        intp_data = []
        # Interpolate lat and lon to our 1 by 1 grid for each time step
        for time_step in range(arr.shape[0]):
            intp_data_tmp = interpolate_lat_lon(grid_x, grid_y, points, values=arr[time_step].reshape(-1, 1))
            intp_data.append(intp_data_tmp)

        # Stack all timesteps
        intp_data = np.stack(intp_data)
        # Return dataset for each year
        intp_data_list.append(intp_data)

    # Stack all years
    intp_data = np.concatenate(intp_data_list)

    return (intp_data)


def get_2wk_lat_lon_data(df, var, events_per_day=4):
    base = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
    latlon_list = []
    for region in target_points['region_id'].unique():
        # Extract Rodeo target lat lon regions only
        lat, lon = region[0], region[1]
        lat_adj, lon_adj = lat - lat_min, lon - lon_min
        # Get 2week data
        s = pd.DataFrame(df[:, lat_adj, lon_adj])
        s = s.groupby(s.index // events_per_day).mean()  # Convert 6hour timeline to daily
        s[var] = s.rolling(window=14).mean().shift(-13)  # Get 2 week average
        s['start_date'] = [base + datetime.timedelta(days=x) for x in range(np.int(df.shape[0] / events_per_day))]
        s['lat'] = lat
        s['lon'] = lon
        s.drop(0, inplace=True, axis=1)
        latlon_list.append(s)
    out_df = pd.concat(latlon_list, ignore_index=True)
    return (out_df)


# if __name__ == "__main__":
def run_interpolation_pipeline():
    # Define target points, and lat lon ranges
    print('--1--Defining lat lon ranges from target grid points')
    global lat_min, lon_min, lat_max, lon_max, file_path, target_points
    target_points = load_locations()
    lat_min = target_points['lat'].min()
    lon_min = target_points['lon'].min()
    lat_max = target_points['lat'].max()
    lon_max = target_points['lon'].max()

    file_path = 'data/prediction_inputs/'
    files = glob(file_path + '*')
    print('\n--2--Interpolating feature datasets:')
    print('\nFiles to Interpolate: ')
    print(*files, sep='\n')

    dataset_dict = {
        'hgt': {'fp_prfx': 'hgt', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max,
                'level': 10},
        'pres': {'fp_prfx': 'pres.sfc.gauss', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min,
                 'lon_max': lon_max, 'level': None},
        'pevpr': {'fp_prfx': 'pevpr.sfc.gauss', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min,
                  'lon_max': lon_max, 'level': None},
        'slp': {'fp_prfx': 'slp', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max,
                'level': None},
        'rhum': {'fp_prfx': 'rhum.sig995', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min,
                 'lon_max': lon_max, 'level': None},
        'pr_wtr': {'fp_prfx': 'pr_wtr.eatm', 'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min,
                   'lon_max': lon_max, 'level': None}
    }
    for dataset in dataset_dict.keys():
        print('Interpolating dataset', dataset)
        dataset_dict[dataset]['data'] = extract_region_data(dataset_name=dataset, dataset_dict=dataset_dict, buffer=5)
        print("-",dataset, "done")

    print('\n--3--Converting to 2 week averages')
    pres = get_2wk_lat_lon_data(df=dataset_dict['pres']['data'], var='pres')
    hgt = get_2wk_lat_lon_data(df=dataset_dict['hgt']['data'], var='hgt', events_per_day=1)
    pevpr = get_2wk_lat_lon_data(df=dataset_dict['pevpr']['data'], var='pevpr')
    slp = get_2wk_lat_lon_data(df=dataset_dict['slp']['data'], var='slp')
    rhum = get_2wk_lat_lon_data(df=dataset_dict['rhum']['data'], var='rhum', events_per_day=1)
    pr_wtr = get_2wk_lat_lon_data(df=dataset_dict['pr_wtr']['data'], var='pr_wtr')

    print('\n--4--Combining datasets')
    df = pres[['start_date', 'lat', 'lon', 'pres']].copy()
    df = df.merge(hgt, left_on=['lon', 'lat', 'start_date'], right_on=['lon', 'lat', 'start_date'], how='inner')
    df = df.merge(rhum, left_on=['lon', 'lat', 'start_date'], right_on=['lon', 'lat', 'start_date'], how='inner')
    df = df.merge(slp, left_on=['lon', 'lat', 'start_date'], right_on=['lon', 'lat', 'start_date'], how='inner')
    df = df.merge(pr_wtr, left_on=['lon', 'lat', 'start_date'], right_on=['lon', 'lat', 'start_date'], how='inner')
    df = df.merge(pevpr, left_on=['lon', 'lat', 'start_date'], right_on=['lon', 'lat', 'start_date'], how='inner')
    df['month'] = df['start_date'].dt.month
    df['month_sin'] = np.sin(df['month'] * (2. * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2. * np.pi / 12))
    df['dayofyear'] = df['start_date'].dt.dayofyear
    df['dayofyear_sin'] = np.sin(df['dayofyear'] * (2. * np.pi / 366))
    df['dayofyear_cos'] = np.cos(df['dayofyear'] * (2. * np.pi / 366))
    df.drop('month', inplace=True, axis=1)
    df.drop('dayofyear', inplace=True, axis=1)

    print('\n--5--Cleaning NaN attributed weather values (large negative values set as dflt for NaNs)')
    pevpr_mean = df[df['pevpr'] > 0]['pevpr'].mean()
    df.loc[df['pevpr'] < 0, 'pevpr'] = pevpr_mean

    print('\n--6--Saving merged dataset to file')
    np.save('data/prediction_inputs/processed_features', np.array(df))
