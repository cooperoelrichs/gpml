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
    print('\nTrain and validate a model aginst local data.')
    config = configer.from_json('model/config.json')

    config.open_local_data_sets()
    X_train_local, y_train_local, X_test_local, y_test_local = get_xs_and_ys(
        config.local_data_set_frames['local_training_data_set'],
        config.local_data_set_frames['local_testing_data_set'],
        config.not_x_labels, config.y_label
    )

    print('\nModel building and cross validation.')
    model = model_maker.basic_lr()
    param_grid = config.parameter_grids[type(model).__name__]
    model = model_maker.do_grid_search(
        model, param_grid, X_train_local, y_train_local)

    # ... whats next?
    #  - Forums?
    #  - Feature engineerings?
    #  - Other model types (SVM)?
    #  - Ensemble?

    kf_results = model_maker.kfolds_evaluation(
        X_train_local, y_train_local, model)
    kf_results.print_mean_results()

    print('\nLocal validation.')
    results = model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        model
    )

    model_maker.dump_model(model, config.model_dump_file_name, results)
    print('Finished.')


def make_a_submission():
    print('Generating a submission.')
    config = configer.from_json('model/config.json')
    model, _ = model_maker.load_model(config.model_dump_file_name)

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
        model, config.submission_file_name
    )
    print('Finished.\n')
