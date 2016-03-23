import json
import pandas as pd
import os


def from_json(config_file):
    """Create Configer from json file."""
    with open(config_file) as f:
        config = json.load(f)
    return Configer(config)


class Configer(object):
    """Class for parsing and storing config."""

    def __init__(self, config):
        self.project_dir = config['project_dir']
        self.data_dir = self.project_dir + config['data_dir']
        self.submission_file_name = (self.data_dir +
                                     config['submission_file_name'])
        self.model_dump_file_name = (self.data_dir +
                                     config['model_dump_file_name'])

        self.file_names = self.add_dir_to_names(
            config['file_names'], self.data_dir)
        self.data_set_names = self.add_dir_to_names(
            config['data_set_names'], self.data_dir)
        self.local_data_set_names = self.add_dir_to_names(
            config['local_data_set_names'], self.data_dir)

        self.data_frames = self.open_data_files(self.file_names, 'csv')
        self.local_test_size = config['local_test_size']

        self.not_x_labels = config['not_x_labels']
        self.y_label = config['y_label']

    def add_dir_to_names(self, names, dir):
        names = dict([(name, dir + file_name)
                      for name, file_name in names.items()
                      ])
        return names

    def open_data_files(self, file_names, file_type):
        frames = {}
        for name, file_name in file_names.items():
            print('Reading: %s' % name)
            if file_type == 'csv':
                frames[name] = pd.read_csv(file_name)
            elif file_type == 'hdf':
                frames[name] = pd.read_hdf(file_name, key='table')
            else:
                raise RuntimeError('File type not supported: %s' % file_type)

        return frames

    def open_data_sets(self):
        data_set_frames = self.open_data_files(self.data_set_names, 'hdf')
        self.data_set_frames = data_set_frames

    def open_local_data_sets(self):
        local_data_set_frames = self.open_data_files(
            self.local_data_set_names, 'hdf')
        self.local_data_set_frames = local_data_set_frames

        # If Mutable
        # def __init__(self):
        #     self._name = ''
        # @property
        # def name(self):
        #     return self._name
        # @name.setter
        # def name(self, value):
        #     self._name = value
