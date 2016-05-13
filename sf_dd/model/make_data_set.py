import os
import random
import pandas as pd
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split

from . import distracted_driver_configer
from gpml.data_set import data_set_maker


def run(project_dir):
    config = get_config(project_dir)
    extract_transform(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def extract_transform(config):
    # TODO
    # 1. Create DF linking person_id, image_id, image, and class.
    # 2. Write this DF as a single large HDF5.
    # 3. Normalise images.

    random.seed(1)  # Seed the random rotations

    training_image_list = pd.read_csv(config.driver_imgs_list).apply(
        lambda r: os.path.join(r['classname'], r['img']),
        axis=1
    )
    testing_image_list = pd.read_csv(config.sample_submission)['img']

    for image_list, image_dir, ds_file, il_file in [
        [
            training_image_list,
            config.image_dirs['train'],
            config.data_sets['training_images'],
            config.image_lists['training_images']
        ],
        [
            testing_image_list,
            config.image_dirs['test'],
            config.data_sets['testing_images'],
            config.image_lists['testing_images']
        ]
    ]:

        image_list = image_list.sort_values(inplace=False)
        image_list = image_list.reset_index(drop=True, inplace=False)

        images = load_and_transform_images(
            image_dir, image_list,
            (config.image_size[1], config.image_size[2]),
            cv2.IMREAD_GRAYSCALE
        )
        images = normalise_images(images)

        pd.DataFrame({'img': image_list}).to_csv(il_file, index_label='i')
        data_set_maker.check_and_save_array_to_hdf(images, ds_file)


def load_and_transform_images(directory, image_list, image_size, colour_flag):
    print('Loading %i images' % len(image_list))
    x, y = image_size
    channels = channels_from_colours(colour_flag)
    images = np.empty((len(image_list), channels, x, y), dtype='uint8')

    for i, img_name in image_list.iteritems():
        image_path = os.path.join(directory, img_name)
        image = cv2.imread(image_path, colour_flag)
        if image is None:
            raise RuntimeError('Image reading failed: %s' % image_path)

        image = random_rotation(image, (-10, 10))
        image_size_rev = (image_size[1], image_size[0])  # cols, rows
        image = cv2.resize(image, image_size_rev, cv2.INTER_LINEAR)

        if channels == 1:
            # Add an empty dimension to gray scale images
            image = np.expand_dims(image, axis=0)
        images[i, :, :, :] = image

    return images


def random_rotation(image, bounds):
    rotation = random.uniform(bounds[0], bounds[1])
    rotation_matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), rotation, 1
    )
    image = cv2.warpAffine(
        image, rotation_matrix, (image.shape[1], image.shape[0])
    )
    return image


def channels_from_colours(colour_flag):
    if colour_flag == cv2.IMREAD_GRAYSCALE:
        channels = 1
    elif colour_flag == cv2.IMREAD_COLOUR:
        channels = 3
    else:
        raise RuntimeError('Colour flag not suported.')
    return channels


def normalise_images(images):
    if images.dtype == 'uint8':
        images = images.astype('float32')
        images /= 255  # 8 bit images
    else:
        raise RuntimeError('Image dtype, %s, not supported' % images.dtype)

    # Skip this while we are trying to match the ZFT model.
    # images -= images.mean()
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
