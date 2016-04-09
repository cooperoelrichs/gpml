from gpml.model import model_maker
from . import configer
from .model_setup import (LRModelSetup, SVCModelSetup,
                          SGDCModelSetup, XGBModelSetup)


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


def train_and_validate_sgdc():
    model_name = 'SGDClassifier'
    print('\n%s Model.' % model_name)
    config = configer.from_json('model/config.json')
    param_grid = config.parameter_grids[model_name]
    model_setup = SGDCModelSetup(param_grid)
    train_and_validate_simple_model(model_name, model_setup, config)


def train_and_validate_svc():
    model_name = 'SVC'
    print('\n%s Model.' % model_name)
    config = configer.from_json('model/config.json')
    param_grid = config.parameter_grids[model_name]
    model_setup = SVCModelSetup(param_grid)
    train_and_validate_simple_model(model_name, model_setup, config)


def train_and_validate_lr():
    model_name = 'LogisticRegression'
    print('\n%s Model.' % model_name)
    config = configer.from_json('model/config.json')
    param_grid = config.parameter_grids[model_name]
    model_setup = LRModelSetup(param_grid)
    train_and_validate_simple_model(model_name, model_setup, config)


def train_and_validate_xgb():
    model_name = 'XGBClassifier'
    print('\n%s Model.' % model_name)
    config = configer.from_json('model/config.json')
    param_grid = config.parameter_grids[model_name]
    model_setup = XGBModelSetup(param_grid)
    model, results = train_and_validate(model_setup, config)
    model_setup.plot_stuff(model, config.model_dump_dir)
    model_setup.dump_model(
        model, results, config.model_dump_file_names[model_name])


# def train_and_validate_simple_model(model_name, model_setup, config):
#     model, results = train_and_validate(model_setup, config)
#     model_json = model_setup.dump_model_to_json_obj(model)
#     model_maker.dump_model(
#         model_json, config.model_dump_file_names[model_name], results)


def train_and_validate(model_setup, config):
    print('\nTrain and validate a model aginst local data.')
    model = model_setup.model

    config.open_local_data_sets()
    X_train_local, y_train_local, X_test_local, y_test_local = get_xs_and_ys(
        config.local_data_set_frames['local_training_data_set'],
        config.local_data_set_frames['local_testing_data_set'],
        config.not_x_labels, config.y_label
    )

    print('\nModel building and cross validation.')
    # --1. Submission.--
    # 2. Optimise remotly.
    # 3. New type of model.
    # 4. Ensemble.
    # 5. Feature engineer?

    model = model_maker.do_grid_search(
        model, model_setup.parameter_grid, X_train_local, y_train_local)

    kf_results = model_maker.kfolds_evaluation(
        X_train_local, y_train_local, model)
    kf_results.print_mean_results()

    print('\nLocal validation.')
    results = model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        model
    )

    print('Finished.')
    return model, results


def make_sgdc_submission():
    model_setup = SGDCModelSetup(None)
    make_a_submission(model_setup)


def make_xgb_submission():
    model_setup = XGBModelSetup(None)
    make_a_submission(model_setup)


def make_a_submission(model_setup):
    print('\n%s submission.' % model_setup.name)
    print('Loading model')
    config = configer.from_json('model/config.json')
    model, results = model_setup.load_model(
        config.model_dump_file_names[model_setup.name])

    print('\nValidating loaded model.')
    config.open_local_data_sets()
    X_train_local, y_train_local, X_test_local, y_test_local = get_xs_and_ys(
        config.local_data_set_frames['local_training_data_set'],
        config.local_data_set_frames['local_testing_data_set'],
        config.not_x_labels, config.y_label
    )

    _ = model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        model
    )

    print('\nTraining on the entire data set, and making predictions.')
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
    print('Finished.')
