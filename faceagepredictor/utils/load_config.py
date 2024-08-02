import os

import yaml


def load_config() -> dict:
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    config_path = os.path.join(project_dir, "config", "config.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
