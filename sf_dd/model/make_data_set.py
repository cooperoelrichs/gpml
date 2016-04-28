import pandas as pd
from . import distracted_driver_configer
from sklearn.cross_validation import train_test_split


def run(project_dir):
    config = get_config(project_dir)
    split_evaluation_train_and_test_data(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def split_evaluation_train_and_test_data(config):
    # Create evaluation train and test data sets with different people
    # in each.

    image_list = pd.read_csv(config.driver_imgs_list)
    print_persons(image_list)
    persons = image_list['subject'].unique()
    train_persons, test_persons = train_test_split(
        persons, test_size=config.evaluation_test_size,
        random_state=1
    )

    train = image_list[image_list['subject'].isin(train_persons)]
    test = image_list[image_list['subject'].isin(test_persons)]

    print('Local Train shape: %i, %i' % train.shape)
    print('Local Test shape:  %i, %i' % test.shape)

    train.to_csv(config.evaluation_imgs_list['train'])
    test.to_csv(config.evaluation_imgs_list['test'])


def print_persons(image_list):
    person_image_counts = image_list['subject'].value_counts().sort_index()
    print('Image count by person:')
    for person in person_image_counts.index:
        print('    %s %i' % (person, person_image_counts[person]))
