from gpml.model import model_maker
from . import distracted_driver_configer
from gpml.model.model_setup import (
    LRModelSetup,
    # SVCModelSetup,
    # SGDCModelSetup,
    XGBModelSetup,
    # ETCModelSetup,
    # RFCModelSetup,
    #
    # AvgEnsSetup,
    # LREnsSetup,
    # ETCEnsSetup,
    # XGBCEnsSetup
)


def run(project_dir):
    config = get_config(project_dir)

    # train_and_validate_lr(config)
    train_and_validate_xgb(config)

    # make_lr_submission(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'santander/model/config.json', project_dir)


def train_and_validate_lr(config):
    train_and_validate_generic_model(LRModelSetup, config)


def train_and_validate_xgb(config):
    raise RunTimeError('Add balencing weights')
    train_and_validate_generic_model(XGBModelSetup, config)


def train_and_validate_generic_model(model_setup_class, config):
    model_setup = model_setup_class(config)
    model, results = train_and_validate(model_setup, config)
    print('Dumping model.')
    model_setup.dump_model(
        model, results, config.model_dump_file_names[model_setup.name])
    print('Finished.')


def train_and_validate(model_setup, config):
    print('\nTrain and validate a model aginst eval data.')
    model = model_setup.model
    config.open_evaluation_data_sets()

    print('\nModel building and cross validation.')
    # model = model_maker.do_grid_search(
    #     model, model_setup.parameter_grid, X_train_local, y_train_local)
    model_maker.evaluate_model_using_kfolds(model, model_setup, config)

    print('\nLocal validation.')
    results = model_maker.evaluate_model_against_evaluation_data(
        model, model_setup, config)
    return model, results


def make_lr_submission(config):
    make_generic_submission(LRModelSetup, config)


def make_generic_submission(model_setup_class, config):
    model_setup = model_setup_class(config)
    print('\nMaking %s submission.' % model_setup.name)
    make_a_submission(model_setup, config)


def make_a_submission(model_setup, config):
    print('Loading model')
    model, results = model_setup.load_model(
        config.model_dump_file_names[model_setup.name])
    model_maker.print_result_from_dict(results)
    model_setup.model = model

    print('\nValidating loaded model.')
    # config.open_local_data_sets()
    # evaluate_model_against_local_data(model_setup, config)

    model_maker.train_make_and_save_submission(model_setup, config)
    print('Finished.')
