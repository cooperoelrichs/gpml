import os
import pandas as pd
import numpy as np
import cv2
import tables
from sklearn.cross_validation import train_test_split

from . import distracted_driver_configer
from gpml.data_set import data_set_maker


def run(project_dir):
    config = get_config(project_dir)
    extract_transform(config)
    # split_evaluation_data(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def extract_transform(config):
    # TODO
    # 1. Create DF linking person_id, image_id, image, and class.
    # 2. Write this DF as a single large HDF5.
    # 3. Normalise images.

    training_image_list = pd.read_csv(config.driver_imgs_list).apply(
        lambda r: os.path.join(r['classname'], r['img']),
        axis=1
    )
    testing_image_list = pd.read_csv(config.sample_submission)['img']

    for image_list, directory, output in [
        [training_image_list,
         config.image_dirs['train'],
         config.data_sets['training_images']],
        [testing_image_list,
         config.image_dirs['test'],
         config.data_sets['testing_images']]
    ]:
        # image_list = image_list[0:9]
        images = load_images(config, directory, image_list)
        data_set_maker.check_and_save_array_to_hdf(images, output)


def load_images(config, directory, image_list):
    print('Loading %i images' % len(image_list))
    x, y = config.image_size
    images = np.empty((len(image_list), x, y), dtype='uint8')

    for i, img_name in image_list.iteritems():
        image_path = os.path.join(directory, img_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError('Image reading failed: %s' % image_path)

        image = cv2.resize(image, config.image_size)
        # image = image.transpose(2, 0, 1)
        # image = np.reshape(image, (1, num_features)

        images[i, :, :] = image

    return images


def split_evaluation_data(config):
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
