from gpml.data_set import data_set_maker
from . import configer
import pandas as pd


def split_local_train_and_test_data():
    print('Splitting data set in a local training and testing sets')
    config = configer.from_json('model/config.json')
    config.open_data_sets()

    data_set_maker.split_and_save_local_data(
        config.data_set_frames['training_data_set'],
        config.local_test_size,
        config.local_data_set_names['local_training_data_set'],
        config.local_data_set_names['local_testing_data_set']
    )


def extract_transform_load():
    print('Making a data set.')

    config = configer.from_json('model/config.json')
    train = config.data_frames['train']
    test = config.data_frames['test']
    num_train = train.shape[0]

    # Make one big data set for encoding
    data_set = pd.concat((train, test), axis=0, ignore_index=True)
    # Feature v22 has 15348 unique values
    data_set = data_set.drop('v22', axis=1)

    data_set = data_set_maker.normalise_num_columns(data_set, ['ID', 'target'])
    data_set = data_set_maker.fill_nans_in_num_columns_with(data_set, 0)
    data_set = data_set_maker.fill_nans_in_str_columns_with(data_set, '-')
    data_set = data_set_maker.dummy_encode_str_columns(data_set)

    train = data_set.iloc[:num_train]
    test = data_set.iloc[num_train:]

    training_file_name = config.data_set_names['training_data_set']
    testing_file_name = config.data_set_names['testing_data_set']
    data_set_maker.check_and_save_to_hdf(train, training_file_name)
    data_set_maker.check_and_save_to_hdf(test, testing_file_name)

    print('Final Training data shape: %i, %i' % train.shape)
    print('Final Testing data shape:  %i, %i' % test.shape)
    print('Finished.')
