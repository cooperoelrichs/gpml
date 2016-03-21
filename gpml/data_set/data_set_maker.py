import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


def check_and_save_to_hdf(df, file_name):
    if pd.isnull(df.values).any():
        raise RuntimeError('df contains NaNs - %s' % file_name)
    print('Saving HDF: %s' % file_name)
    save_hdf(df, file_name)


def save_hdf(df, file_name):
    df.to_hdf(file_name, key='table', append=False)


def save_csv(df, file_name):
    df.to_csv(file_name, sep=',')


def split_and_save_local_data(data, test_size,
                              train_file_name, test_file_name):
    train, test = train_test_split(data, test_size=test_size,
                                   random_state=99)

    print('Local Train shape: %i, %i' % train.shape)
    print('Local Test shape:  %i, %i' % test.shape)

    check_and_save_to_hdf(train, train_file_name)
    check_and_save_to_hdf(test, test_file_name)


def get_num_columns(df):
    return df.select_dtypes(include=['number']).columns


def get_str_columns(df):
    """
    This will get all 'object' columns.
    TODO: add Pandas Doc reference
    """
    return df.select_dtypes(include=['object']).columns


def normalise_num_columns(df, non_feature_columns):
    num_columns = get_num_columns(df)
    feature_num_columns = num_columns.difference(non_feature_columns)
    df[feature_num_columns] = df[feature_num_columns].apply(
        lambda x: (x - x.mean()) / x.std())
    return df


def fill_nans_in_num_columns_with(df, this):
    num_columns = get_num_columns(df)
    df = fill_nans_in_these_columns_with(df, num_columns, this)
    return df


def fill_nans_in_str_columns_with(df, this):
    str_columns = get_str_columns(df)
    df = fill_nans_in_these_columns_with(df, str_columns, this)
    return df


def fill_nans_in_these_columns_with(df, columns, this):
    df[columns] = df[columns].fillna(this)
    return df


def dummy_encode_str_columns(df):
    str_columns = get_str_columns(df)

    # This doesn't makes a dummy for all values, which will make the output
    # non-unique.
    # Sparse causes the hdf wright/read to fail.
    dummies = pd.get_dummies(df[str_columns], prefix=df[str_columns].columns,
                             sparse=False)

    # memory_usage = dummies.memory_usage(index=True) / 1024 ^ 3
    # print('Memory size of dummies (GB): %0.3f' % memory_usage)

    df = pd.concat([df[df.columns.difference(str_columns)], dummies], axis=1)
    return df


def make_data_set(config):
    pass
    # check_and_save_to_hdf(data_set, config.data_set_file_name)


def make_submission_set(config):
    pass
    # check_and_save_to_hdf(submission, config.submission_data_set_file_name)
