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
    lr = model_maker.basic_lr()
    kf_results = model_maker.kfolds_evaluation(
        X_train_local, y_train_local, lr)
    kf_results.print_results()
    kf_results.print_mean_results()

    print('\nLocal validation.')
    model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        lr
    )

    # I should be saving the model here, and generating a submission later

    print('\nGenerate a submission.')
    config.open_data_sets()
    X_train, y_train, X_submission, _ = get_xs_and_ys(
        config.data_set_frames['training_data_set'],
        config.data_set_frames['testing_data_set'],
        config.not_x_labels, config.y_label
    )
    id_column = config.data_set_frames['testing_data_set']['ID']
    model_maker.make_and_save_submission(
        X_train, y_train,
        X_submission, id_column,
        lr, config.submission_file_name
    )

    print('Finished.')
