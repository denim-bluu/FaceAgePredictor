from typing import Dict

import yaml

import os


def load_config() -> Dict:
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    config_path = os.path.join(project_dir, "config", "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
