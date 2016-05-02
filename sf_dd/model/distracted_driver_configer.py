from gpml.model import configer


def from_json(config_file, project_dir):
    return configer.from_json(
        DistractedDriverConfiger, config_file, project_dir)


class DistractedDriverConfiger(configer.ConfigerBase):
    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)

        self.driver_imgs_list = self.data_dir + config['driver_imgs_list']
        self.sample_submission = self.data_dir + config['sample_submission']
        self.evaluation_imgs_list = self.add_dir_to_names(
            config['evaluation_imgs_list'], self.data_dir)
        self.image_dirs = self.add_dir_to_names(
            config['image_dirs'], self.data_dir)
        self.image_size = tuple(config['image_size'])
        self.data_sets = self.add_dir_to_names(
            config['data_sets'], self.data_dir)
