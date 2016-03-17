import json
import pandas as pd


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

        self.data_set_file_name = (
            self.data_dir + config['data_set_file_name'])
        self.submission_data_set_file_name = (
            self.data_dir + config['submission_data_set_file_name'])

        self.local_file_names = config['local_file_names']
        self.file_names = config['file_names']
        self.data_frames = self.open_data_files(self.data_dir, self.file_names)

        self.local_test_size = config['local_test_size']

    def open_data_files(self, data_dir, file_names):
        frames = {}
        for name, file_name in file_names.items():
            print('loading: %s' % name)
            frames[name] = pd.read_csv(data_dir + file_name)

        return frames

    def open_local_train_and_test_files(self):
        local_data_frames = self.open_data_files(self.data_dir,
                                                 self.local_file_names)
        self.local_data_frames = local_data_frames

        # If Mutable
        # def __init__(self):
        #     self._name = ''
        # @property
        # def name(self):
        #     return self._name
        # @name.setter
        # def name(self, value):
        #     self._name = value
