import pandas as pd
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
    Get all columns of dtype str by assuming that all 'object' columns
    are str columns.

    str cannot be selected directly (see link).
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html
    """
    return df.select_dtypes(include=['object']).columns


def normalise_num_columns(df, non_feature_columns):
    num_columns = get_num_columns(df)
    feature_num_columns = num_columns.difference(non_feature_columns)

    df[feature_num_columns] = df[feature_num_columns].apply(
        lambda x: (x - x.mean()) / x.std()
    )
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


def get_columns_with_na(df):
    columns = df.columns[df.isnull().any(axis=0)]
    return columns


def get_dummies(df, drop_first, dummy_na):
    return pd.get_dummies(
        df,
        prefix=df.columns,
        sparse=False,
        drop_first=drop_first,
        dummy_na=dummy_na
    )


def dummy_encode_str_columns(df):
    """
    Dummy encode str columns, assuming na is important.

    1. Drop one category from each column.
    2. Assume na is important, and choose it as the dropped value if it
       is present in a columns.
    3. If na is not present in a column then drop the first category.
    """
    str_columns = get_str_columns(df)
    str_columns_with_na = get_columns_with_na(df[str_columns])
    str_columns_without_na = str_columns.difference(str_columns_with_na)

    # Drop na instead of the first category
    dummies_dropping_na = get_dummies(
        df[str_columns_with_na], drop_first=False, dummy_na=False)

    # Drop first
    dummies_dropping_first = get_dummies(
        df[str_columns_without_na], drop_first=True, dummy_na=False)

    # memory_usage = dummies.memory_usage(index=True) / 1024 ^ 3
    # print('Memory size of dummies (GB): %0.3f' % memory_usage)

    df = pd.concat(
        [
            df[df.columns.difference(str_columns)],
            dummies_dropping_na,
            dummies_dropping_first
        ],
        axis=1
    )
    return df


def scale_dummy_columns(df, dummy_columns):
    """
    Scale dummy columns to be centred around 0, with a range of (1, -1).

    This method assumed dummy columns are encoded by (1, 0).
    This method maintains the on/off nature of dummy columns rather than
    shifting the mean or scaling by the standard deviation.
    """
    df[dummy_columns] = df[dummy_columns].apply(
        lambda x: (x - 0.5) * 2
    )
    return df


def make_data_set(config):
    pass
    # check_and_save_to_hdf(data_set, config.data_set_file_name)


def make_submission_set(config):
    pass
    # check_and_save_to_hdf(submission, config.submission_data_set_file_name)
