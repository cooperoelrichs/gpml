from gpml.model import configer


def from_json(config_file, project_dir):
    return configer.from_json(
        DistractedDriverConfiger, config_file, project_dir)


class DistractedDriverConfiger(configer.ConfigerBase):
    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)

        self.driver_imgs_list = self.data_dir + config['driver_imgs_list']
        self.evaluation_imgs_list = self.add_dir_to_names(
            config['evaluation_imgs_list'], self.data_dir)
