from gpml.data_set import data_set_maker
from . import configer
import pandas as pd


def split_local_train_and_test_data():
    print('\nSplitting data set in a local training and testing sets')
    config = configer.from_json('model/config.json')
    config.open_data_sets()

    data_set_maker.split_and_save_local_data(
        config.data_set_frames['training_data_set'],
        config.local_test_size,
        config.local_data_set_names['local_training_data_set'],
        config.local_data_set_names['local_testing_data_set']
    )


def extract_transform_load():
    print('\nMaking a data set.')

    config = configer.from_json('model/config.json')
    meta_columns = ['ID', 'target']
    train = config.data_frames['train']
    test = config.data_frames['test']
    num_train = train.shape[0]

    # Make one big data set for encoding
    data_set = pd.concat((train, test), axis=0, ignore_index=True)
    data_set = data_set.drop(config.columns_to_remove, axis=1)
    # data_set = data_set_maker.fill_nans_in_str_columns_with(data_set, '-')

    categorical_columns = data_set_maker.get_str_columns(data_set)
    data_set, categoricals = data_set_maker.seperate_categoricals(
        data_set, categorical_columns)

    data_set = factorise_categoricals(data_set, categoricals, na_sentinel=-1)
    data_set = data_set_maker.normalise_num_columns(data_set, meta_columns)
    # data_set = one_hot_encode_categoricals(
    #     data_set, categoricals, config.columns_to_not_one_hot)
    data_set = data_set_maker.fill_nans_in_num_columns_with(
        data_set, -999, meta_columns)

    train = data_set.iloc[:num_train]
    test = data_set.iloc[num_train:]
    training_file_name = config.data_set_names['training_data_set']
    testing_file_name = config.data_set_names['testing_data_set']
    data_set_maker.check_and_save_to_hdf(train, training_file_name, [])
    data_set_maker.check_and_save_to_hdf(test, testing_file_name, meta_columns)

    print('Final Training data shape: %i, %i' % train.shape)
    print('Final Testing data shape:  %i, %i' % test.shape)
    print('Finished.')


def one_hot_encode_categoricals(
        data_set, categoricals, exclude):
    # exclude = [x for x in exclude if x in data_set.columns]
    categoricals = categoricals.drop(exclude, axis=1)
    dummies = data_set_maker.dummy_encode(categoricals)
    dummies = data_set_maker.scale_dummy_columns(dummies)
    data_set = pd.concat([data_set, dummies], axis=1)
    return data_set


def factorise_categoricals(data_set, categoricals, na_sentinel):
    factorised = data_set_maker.factorise(categoricals, na_sentinel)
    data_set = pd.concat([data_set, factorised], axis=1)
    return data_set
