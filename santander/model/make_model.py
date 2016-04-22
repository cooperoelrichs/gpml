from gpml.model import model_maker
from . import santander_configer
from gpml.model.model_setup import (
    LRModelSetup,
    # SVCModelSetup,
    # SGDCModelSetup,
    # XGBModelSetup,
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
    train_and_validate_lr(config)


def get_config(project_dir):
    return santander_configer.from_json(
        'santander/model/config.json', project_dir)


def train_and_validate_lr(config):
    train_and_validate_generic_model(LRModelSetup, config)


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
