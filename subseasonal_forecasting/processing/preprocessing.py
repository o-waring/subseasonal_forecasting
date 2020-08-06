import pandas as pd
import numpy as np
import datetime

from utils.load_functions import load_tar_datasets, load_column_names
from processing.preprocessing import PreprocessTemporalSpatialData, process_series

# from google.colab import drive
# drive.mount('/content/drive')
# repo_path = "/content/drive/My Drive/repos/subseasonal_rodeo/"

if __name__ == "__main__":

    repo_path = "../"

    # Series datasets
    pres = process_series('data/training_inputs_raw/gt-contest_pres.sfc.gauss-14d-1948-2018.h5',repo_path) # Pressure
    hgt = process_series('data/training_inputs_raw/gt-contest_wind_hgt_10-14d-1948-2018.h5',repo_path) # Wind @10m geopotential height
    rhum = process_series('data/training_inputs_raw/gt-contest_rhum.sig995-14d-1948-2018.h5',repo_path) # Relative humidity
    slp = process_series('data/training_inputs_raw/gt-contest_slp-14d-1948-2018.h5',repo_path) # Sea level pressure
    prec = process_series('data/training_inputs_raw/gt-contest_precip-14d-1948-2018.h5',repo_path) # Precipitation
    prwtr = process_series('data/training_inputs_raw/gt-contest_pr_wtr.eatm-14d-1948-2018.h5',repo_path) # Precipitable water
    pevpr = process_series('data/training_inputs_raw/gt-contest_pevpr.sfc.gauss-14d-1948-2018.h5',repo_path) # Potential evaporation

    # Tmp2m datasets
    temp = pd.read_hdf(repo_path+'data/gt-contest_tmp2m-14d-1979-2018.h5')
    temp['start_date'] = pd.to_datetime(temp['start_date'], unit='ns')
    temp.drop(['tmp2m_sqd',	'tmp2m_std'], axis=1, inplace=True)

    # Tmp2m & Precipitation Climatology & Anomaly
    df = temp.merge(prec, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['day'] = df['start_date'].dt.day
    df['dayofyear'] = df['start_date'].dt.dayofyear

    # Merge all into single df
    # Spatial temporal datasets
    df = df.merge(pres, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del pres
    df = df.merge(hgt, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del hgt
    df = df.merge(rhum, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del rhum
    df = df.merge(slp, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del slp
    df = df.merge(prwtr, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del prwtr
    df = df.merge(pevpr, left_on = ['lon','lat','start_date'], right_on = ['lon','lat','start_date'], how = 'inner')
    del pevpr

    # Add cyclical date features
    df['month_sin'] = np.sin(df['month']*(2.*np.pi/12))
    df['month_cos'] = np.cos(df['month']*(2.*np.pi/12))
    df['dayofyear_sin'] = np.sin(df['dayofyear']*(2.*np.pi/366))
    df['dayofyear_cos'] = np.cos(df['dayofyear']*(2.*np.pi/366))

    # Sort by date as primary sorting key so we can split for train/validation/test easily
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('start_date')))
    df = df.loc[:, cols]
    df = df.sort_values(by=['start_date', 'lat','lon']).reset_index(drop=True)
    df = df.drop(['year','month','day','dayofyear'], axis=1)
    for col in df.columns[1:]:
        df[col] = df[col].astype('float32')

    # Sort by date as primary sorting key so we can split for train/validation/test easily
    df = df.sort_values(by=['lat','lon', 'start_date']).reset_index(drop=True)

    ## Save Unscaled Merged Dataset & Column Names
    # pd.DataFrame(df.columns.values).to_csv(repo_path+'data/processed/spatial_temporal/column_names.csv')
    # np.save(repo_path+'data/processed/spatial_temporal/full_data_unscaled_1979-2018', np.array(df))
    # df = np.array(np.load(repo_path+'data/processed/spatial_temporal/full_data_unscaled_1979-2018.npy',allow_pickle=True))

    ## Conduct Preprocessing -- standardize, scale, transform to spatial 2D, mask missing regions, outer pad spatial tensor

    col_names = pd.read_csv(repo_path+'data/standardization/column_names.csv')
    col_names = list(col_names['0'].values)

    locations = pd.read_csv(repo_path+'data/standarization/target_points.csv')
    locations['region_id'] = list(zip(locations['lat'], locations['lon']))

    y = PreprocessTemporalSpatialData(df, locations, col_names, num_regions=514, num_features=15, max_sg=5)

    y.preprocess_pipeline()

    np.save(repo_path+'data/processed/full_data_scaled_1979-2018', df)