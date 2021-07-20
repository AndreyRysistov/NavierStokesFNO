import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.callbacks.tensor_board.log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp, "logs/")
    config.callbacks.checkpoint.dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp, "checkpoints/")
    config.graphics.dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()), config.exp, "graphics/")
    create_dirs([config.callbacks.tensor_board.log_dir, config.callbacks.checkpoint.dir, config.graphics.dir])
    return config