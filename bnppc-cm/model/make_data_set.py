from gpml.data_set import data_set_maker
from . import configer
import pandas as pd


def split_local_train_and_test_data():
    config = configer.from_json('model/config.json')

    data_set_maker.split_and_save_local_data(
        config.data_frames['train'], config.local_test_size,
        config.local_file_names['train'], config.local_file_names['test']
    )


def extract_transform_load():
    # May need to do this before the local split
    config = configer.from_json('model/config.json')
    config.open_local_train_and_test_files()
    train = config.local_data_frames['train']

    data_set = data_set_maker.normalise_num_columns(train, ['ID', 'target'])
    data_set = data_set_maker.fill_nans_in_num_columns_with(data_set, 0)
    data_set = data_set_maker.fill_nans_in_str_columns_with(data_set, '-')
    data_set = data_set_maker.dummy_encode_str_columns(data_set)

    # Normalise data (floats)
    # Replace NaNs... (floats and categoricals)
    # One hot encoding
