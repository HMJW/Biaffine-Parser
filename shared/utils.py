import json
from argparse import Namespace
import os


def get_config(config_filepath):
    with open(config_filepath, "r") as config_file:
        conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
    return conf

def save_config(config, config_filepath):
    with open(config_filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=4)