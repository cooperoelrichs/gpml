import pandas as pd
from sklearn.cross_validation import train_test_split


def check_and_save_to_hdf(df, file_name):
    if pd.isnull(df.values).any():
        raise RuntimeError('df contains NaNs - %s' % file_name)
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

    save_csv(train, train_file_name)
    save_csv(test, test_file_name)


def make_data_set(config):
    pass
    # check_and_save_to_hdf(data_set, config.data_set_file_name)


def make_submission_set(config):
    pass
    # check_and_save_to_hdf(submission, config.submission_data_set_file_name)
