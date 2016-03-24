from gpml.model import model_maker
from . import configer


def get_x(df, not_x_labels):
    feature_names = df.columns.difference(not_x_labels)
    return df[feature_names]


def get_y(df, y_label):
    return df[y_label]


def get_x_and_y(df, not_x_labels, y_label):
    return get_x(df, not_x_labels), get_y(df, y_label)


def get_xs_and_ys(local_train, local_test, not_x_labels, y_label):
    X_train, y_train = get_x_and_y(
        local_train, not_x_labels, y_label)
    X_test, y_test = get_x_and_y(
        local_test, not_x_labels, y_label)
    return X_train, y_train, X_test, y_test


def train_and_validate_model():
    print('Train and validate a model aginst local data.')
    config = configer.from_json('model/config.json')

    config.open_local_data_sets()
    X_train_local, y_train_local, X_test_local, y_test_local = get_xs_and_ys(
        config.local_data_set_frames['local_training_data_set'],
        config.local_data_set_frames['local_testing_data_set'],
        config.not_x_labels, config.y_label
    )

    print('\nModel building and cross validation.')

    # Fit C parameter
    # Fit regularisation type
    # ... whats next?
    #  - Forums?
    #  - Feature building?
    #  - Other model types (SVM)?
    #  - Ensemble?

    lr = model_maker.basic_lr()
    kf_results = model_maker.kfolds_evaluation(
        X_train_local, y_train_local, lr)
    kf_results.print_results()
    kf_results.print_mean_results()

    print('\nLocal validation.')
    results = model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        lr
    )

    model_maker.dump_lr_model(lr, config.model_dump_file_name, results)
    print('Finished.\n')


def make_a_submission():
    print('Generating a submission.')
    config = configer.from_json('model/config.json')
    config.open_data_sets()
    X_train, y_train, X_submission, _ = get_xs_and_ys(
        config.data_set_frames['training_data_set'],
        config.data_set_frames['testing_data_set'],
        config.not_x_labels, config.y_label
    )

    lr = model_maker.empty_lr()
    lr, _ = model_maker.load_lr_model(lr, config.model_dump_file_name)
    id_column = config.data_set_frames['testing_data_set']['ID']

    model_maker.make_and_save_submission(
        X_train, y_train,
        X_submission, id_column,
        lr, config.submission_file_name
    )
    print('Finished.\n')
