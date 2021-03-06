from . import santander_configer
from gpml.data_set import data_set_maker
import pandas as pd


def run(project_dir):
    config = get_config(project_dir)
    extract_transform(config)
    data_set_maker.split_evaluation_train_and_test_data(config)


def get_config(project_dir):
    return santander_configer.from_json(
        'santander/model/config.json', project_dir)


def extract_transform(config):
    train = config.data_frames['train']
    test = config.data_frames['test']
    num_train = train.shape[0]

    print('Train shape: (%i, %i)' % train.shape)
    print('Test shape: (%i, %i)' % test.shape)

    data_set = pd.concat((train, test), axis=0, ignore_index=True)
    meta_data = data_set[config.meta_columns]
    data_set = data_set.drop(config.meta_columns, axis=1)
    data_set = data_set.drop(config.columns_to_remove, axis=1)
    data_set = data_set_maker.drop_columns_with_zero_variance(data_set)
    data_set = data_set_maker.normalise_num_columns(data_set, [config.y_label])

    # Features to add:
    #  - Normalise
    #  - One Hot
    #  - Factorise
    #  - Clip
    #  - Feature interactions - 2 and 3 level
    #  - KNN features
    #  - PCA
    #  - Sum zeros/nans
    #  - Rounding to remove noise?
    #  - Bin continous features
    #  - Baysean coding
    #  - Baysean coding of interactions
    #  - MRMR feature selection
    #  - TNSE - 2D
    #  - Replace with target mean - ignoring current row.

    data_set = pd.concat([data_set, meta_data], axis=1)
    train = data_set.iloc[:num_train]
    test = data_set.iloc[num_train:].drop(config.y_label, axis=1)
    training_file_name = config.data_set_names['training_data_set']
    testing_file_name = config.data_set_names['testing_data_set']
    data_set_maker.check_and_save_to_hdf(train, training_file_name, [])
    data_set_maker.check_and_save_to_hdf(test, testing_file_name, [])

    print('Final Training data shape: (%i, %i)' % train.shape)
    print('Final Testing data shape:  (%i, %i)' % test.shape)
    print('Finished.')
