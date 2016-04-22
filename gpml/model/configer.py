import json
import pandas as pd
import os


def from_json(configer_variant_class, config_file, project_dir):
    """Create Configer from json file."""
    with open(config_file) as f:
        config = json.load(f)
    return configer_variant_class(config, project_dir)


class ConfigerBase(object):
    """Class for parsing and storing config."""

    def __init__(self, config, project_dir):
        self.project_dir = project_dir
        self.data_dir = self.project_dir + config['data_dir']
        self.model_dump_dir = self.data_dir + config['model_dump_dir']
        self.submission_dir = self.data_dir + config['submission_dir']

        self.maybe_create_dirs([self.model_dump_dir, self.submission_dir])

        self.model_dump_file_names = self.add_dir_to_names(
            config['model_dump_file_names'], self.model_dump_dir)
        self.submission_file_names = self.add_dir_to_names(
            config['submission_file_names'], self.submission_dir)

        self.file_names = self.add_dir_to_names(
            config['file_names'], self.data_dir)
        self.data_set_names = self.add_dir_to_names(
            config['data_set_names'], self.data_dir)
        self.evaluation_data_set_names = self.add_dir_to_names(
            config['evaluation_data_set_names'], self.data_dir)

        self.data_frames = self.open_data_files(self.file_names, 'csv')
        self.evaluation_test_size = config['evaluation_test_size']

        self.meta_columns = config['meta_columns']
        self.y_label = config['y_label']
        self.columns_to_remove = config['columns_to_remove']
        self.columns_to_not_one_hot = config['columns_to_not_one_hot']

        self.parameter_grids = config['parameter_grids']
        self.fitting_parameters = config['fitting_parameters']
        self.model_parameters = config['model_parameters']

        self.model_averaging_weights = config['model_averaging_weights']

    def maybe_create_dirs(self, dirs):
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)  # mkdir -p

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

    def open_evaluation_data_sets(self):
        evaluation_data_set_frames = self.open_data_files(
            self.evaluation_data_set_names, 'hdf')
        self.evaluation_data_set_frames = evaluation_data_set_frames

        # If Mutable
        # def __init__(self):
        #     self._name = ''
        # @property
        # def name(self):
        #     return self._name
        # @name.setter
        # def name(self, value):
        #     self._name = value
