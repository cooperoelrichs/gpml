from . import distracted_driver_configer
from gpml.data_set import data_set_maker
import pandas as pd


def run(project_dir):
    config = get_config(project_dir)
    extract_transform(config)
    # data_set_maker.split_evaluation_train_and_test_data(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def extract_transform(config):
    image_list = pd.read_csv(config.driver_imgs_list)
    print_persons(image_list)


def print_persons(image_list):
    person_counts = image_list['subject'].value_counts().sort_index()
    print('Person counts:')
    for person in person_counts.index:
        print('    %s - %i' % (person, person_counts[person]))
