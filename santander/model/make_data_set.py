from . import santander_configer as configer
from gpml.data_set import data_set_maker
import pandas as pd


def run(project_dir):
    extract_transform(project_dir)
    # split_evaluation_train_and_test_data(project_dir)


def extract_transform(project_dir):
    config = configer.from_json('santander/model/config.json', project_dir)
    train = config.data_frames['train']
    test = config.data_frames['test']
    num_train = train.shape[0]

    print('Train shape: (%i, %i)' % train.shape)
    print('Test shape: (%i, %i)' % test.shape)

    # Make one big data set for encoding
    data_set = pd.concat((train, test), axis=0, ignore_index=True)
    data_set = data_set.drop(config.columns_to_remove, axis=1)

    raise RuntimeError('We should actually do something...')


def split_evaluation_train_and_test_data(
        project_dir='/Users/coelrichs/Projects/Kaggle/'):
    print('\nSplitting data set into evaluation and full data sets')
    config = configer.from_json('model/config.json', project_dir)
    config.open_data_sets()

    data_set_maker.split_and_save_evaluation_data(
        config.data_set_frames['training_data_set'],
        config.evaluation_test_size,
        config.evaluation_data_set_names['evaluation_training_data_set'],
        config.evaluation_data_set_names['evaluation_testing_data_set']
    )
