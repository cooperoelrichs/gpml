import pandas as pd
import numpy as np
import tables
import os
from sklearn.cross_validation import train_test_split
from .knn_linear_features import NearestNeighbourLinearFeatures


def drop_columns_with_zero_variance(df):
    columns_to_drop = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            columns_to_drop.append(column)

    print('Dropping %i columns with zero variance.' % len(columns_to_drop))
    df = df.drop(columns_to_drop, axis=1)
    return df


def add_knn_linear_features(
        X_train, X_test, y_train, num_new_features=15, max_group_size=5):
    for X in [X_train, X_test]:
        for i in [0, 1, 2, 3]:
            new_column_name = 'v22-%i' % (i + 1)

            X[new_column_name] = X['v22'].fillna('@@@@').apply(
                lambda x: '@' * (4 - len(str(x))) + str(x)
            ).apply(
                lambda x: ord(x[i])
            )

    # drop_list=[
    #     'v91','v1', 'v8', 'v10', 'v15', 'v17',
    #     'v25', 'v29', 'v34', 'v41', 'v46', 'v54',
    #     'v64', 'v67', 'v97', 'v105', 'v111', 'v122'
    # ]

    # refcols = list(X.columns)
    # for elt in refcols:
    #     if X[elt].dtype == 'O':
    #         X[elt], temp = pd.factorize(X[elt])

    a = NearestNeighbourLinearFeatures(
        n_neighbours=num_new_features,
        max_elts=max_group_size,
        verbose=False,
        random_state=12
    )
    a.fit(X_train, y_train)
    X_train = a.transform(X_train)
    X_test = a.transform(X_test)
    return X_train, X_test


def check_and_save_to_hdf(df, file_name, to_ignore):
    if pd.isnull(df.drop(to_ignore, axis=1).values).any():
        raise RuntimeError(
            'df contains NaNs in these columns: %s'
            % str(get_columns_with_na(df.drop(to_ignore, axis=1)))
        )
    print('Saving HDF: %s' % file_name)
    save_hdf(df, file_name)


def check_and_save_array_to_hdf(x, file_name):
    if np.isnan(x).any():
        raise RuntimeError('Array contains NaNs')
    print('Saving HDF: %s' % file_name)
    save_array_to_hdf(x, file_name)


def save_array_to_hdf(x, file_name):
    name = os.path.splitext(os.path.basename(file_name))[0]

    f = tables.openFile(file_name, 'w')
    atom = tables.Atom.from_dtype(x.dtype)
    ds = f.createCArray(f.root, name, atom, x.shape)
    ds[:] = x
    f.close()


def save_hdf(df, file_name):
    df.to_hdf(file_name, key='table', append=False)


def save_csv(df, file_name):
    df.to_csv(file_name, sep=',')


def split_and_save_evaluation_data(
        data, test_size, train_file_name, test_file_name, labels):
    train, test = train_test_split(
        data, test_size=test_size, random_state=1, stratify=labels)

    print('Local Train shape: %i, %i' % train.shape)
    print('Local Test shape:  %i, %i' % test.shape)

    check_and_save_to_hdf(train, train_file_name, [])
    check_and_save_to_hdf(test, test_file_name, [])


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


def normalise_num_columns(df, to_ignore):
    num_columns = get_num_columns(df)
    feature_num_columns = num_columns.difference(to_ignore)
    print('Normalising %i numerical columns' % len(feature_num_columns))
    df[feature_num_columns] = df[feature_num_columns].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    return df


def fill_nans_in_num_columns_with(df, this, meta_columns):
    num_columns = get_num_columns(df).difference(meta_columns)
    print('Filling Nans with %s, currrent data range is: %0.2f - %0.2f'
          % (
              str(this),
              df[num_columns].min(axis=0).min(),
              df[num_columns].max(axis=0).max()
          ))
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


def seperate_categoricals(df, categorical_columns):
    if categorical_columns.size == 0:
        print('Note: There are no categorical columns!')
    categoricals = df[categorical_columns]
    remaining = df.drop(categorical_columns, axis=1)
    return remaining, categoricals


def dummy_encode(df):
    """
    Dummy encode df.

    1. Drop one category from each column.
    2. Assume na is important, and choose it as the dropped value if it
       is present in a columns.
    3. If na is not present in a column then drop the first category.
    """
    columns_with_na = get_columns_with_na(df)
    columns_without_na = df.columns.difference(columns_with_na)

    # Drop na instead of the first category
    dummies_dropping_na = get_dummies(
        df[columns_with_na], drop_first=False, dummy_na=False)

    # Drop first
    dummies_dropping_first = get_dummies(
        df[columns_without_na], drop_first=True, dummy_na=False)

    # memory_usage = dummies.memory_usage(index=True) / 1024 ^ 3
    # print('Memory size of dummies (GB): %0.3f' % memory_usage)

    dummies = pd.concat(
        [
            dummies_dropping_na,
            dummies_dropping_first
        ],
        axis=1
    )
    return dummies


def factorise(df, na_sentinel):
    df = df.copy()  # Don't modify in place.
    for column_name in df.columns:
        factorised, uniques = pd.factorize(
            df[column_name], na_sentinel=na_sentinel)
        df[column_name] = factorised
    return df


def scale_dummy_columns(df):
    """
    Scale dummy columns to be centred around 0, with a range of (1, -1).

    This method assumed dummy columns are encoded by (1, 0).
    This method maintains the on/off nature of dummy columns rather than
    shifting the mean or scaling by the standard deviation.
    """
    df = df.apply(
        lambda x: (x - 0.5) * 2
    )
    return df


def split_evaluation_train_and_test_data(config):
    print('\nSplitting the data set for local evaluation')
    config.open_data_sets()
    split_and_save_evaluation_data(
        config.data_set_frames['training_data_set'],
        config.evaluation_test_size,
        config.evaluation_data_set_names['evaluation_training_data_set'],
        config.evaluation_data_set_names['evaluation_testing_data_set'],
        config.data_set_frames['training_data_set'][config.y_label]
    )
