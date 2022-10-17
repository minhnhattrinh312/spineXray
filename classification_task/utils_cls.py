import yaml
def load_config(path_config):
    with open(path_config, 'r') as fp:
        cfg = yaml.safe_load(fp)
    return cfg
config = load_config("classification_task/config.yaml")