import json


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
        self.basic_data_set_file_name = (self.data_dir +
                                         config['basic_data_set_file_name'])
        self.file_names = config['file_names']

        # If Mutable
        # def __init__(self):
        #     self._name = ''
        # @property
        # def name(self):
        #     return self._name
        # @name.setter
        # def name(self, value):
        #     self._name = value
