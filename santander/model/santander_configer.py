from gpml.model import configer


def from_json(config_file, project_dir):
    return configer.from_json(SantanderConfiger, config_file, project_dir)


class SantanderConfiger(configer.ConfigerBase):
    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
