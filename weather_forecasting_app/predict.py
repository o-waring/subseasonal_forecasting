import joblib
from glob import glob
import numpy as np
import pandas as pd

from src.get_data.load_functions import load_tar_datasets

if __name__ == "__main__":

    prec_df = load_tar_datasets(pred_var='temp34', file_dir='data/')

    print(prec_df.head())