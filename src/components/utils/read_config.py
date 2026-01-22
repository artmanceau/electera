import json
import os
from pathlib import Path

# TODO : Adapt to have config in S3


class ConfigReader:

    @staticmethod
    def _read_config(config_path, config_class):
        file_path = os.path.join(
            Path(__file__).parent.parent.parent.parent, config_path
        )
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return config_class(**config_dict)
