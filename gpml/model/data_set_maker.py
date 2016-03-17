import pandas as pd
from . import configer


def check_and_save_to_hdf(df, file_name):
    if pd.isnull(df.values).any():
        raise RuntimeError('df contains NaNs - %s' % file_name)
    df.to_hdf(file_name, key='table', append=False)


def make_data_set():

    check_and_save_to_hdf(basic_data_set,
                          config.basic_data_set_file_name)


def make_submission_set():
    config = configer.from_json('model/config_mmlm2016.json')
    check_and_save_to_hdf(submission_games,
                          config.submission_data_set_file_name)
