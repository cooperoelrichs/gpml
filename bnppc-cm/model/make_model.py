from gpml.model import model_maker
from . import configer
from .model_setup import LRModelSetup


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


def train_and_validate_svc_model():
    pass


def train_and_validate_lr_model():
    config = configer.from_json('model/config.json')
    model = model_maker.basic_lr()
    param_grid = config.parameter_grids[type(model).__name__]
    model_setup = LRModelSetup(model, param_grid)
    train_and_validate_model(model_setup, config)


def train_and_validate_model(model_setup, config):
    print('\nTrain and validate a model aginst local data.')
    model = model_setup.model

    config.open_local_data_sets()
    X_train_local, y_train_local, X_test_local, y_test_local = get_xs_and_ys(
        config.local_data_set_frames['local_training_data_set'],
        config.local_data_set_frames['local_testing_data_set'],
        config.not_x_labels, config.y_label
    )

    print('\nModel building and cross validation.')
    model = model_maker.do_grid_search(
        model, model_setup.parameter_grid, X_train_local, y_train_local)

    # ... whats next?
    #  - Other model types (SVM, ElasticNet, SGDClassifier)?
    #  - Forums?
    #  - Feature engineering?
    #  - Ensemble? - http://mlwave.com/kaggle-ensembling-guide/
    #  - Remove outliers? - http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

    kf_results = model_maker.kfolds_evaluation(
        X_train_local, y_train_local, model)
    kf_results.print_mean_results()

    print('\nLocal validation.')
    results = model_maker.evaluate_model(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        model
    )

    model_maker.dump_model(
        model_setup.dump_model_to_json_obj(model),
        config.model_dump_file_name, results
    )
    print('Finished.')


def make_a_submission():
    print('\nGenerating a submission.')
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
    print('Finished.')
