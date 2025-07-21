import yaml
import os

def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("SPORTSORACLE_CONFIG", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) 