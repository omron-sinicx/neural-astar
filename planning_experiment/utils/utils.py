"""Utility functions.
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X

Small parts of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

import argparse
import glob
import random
import re
from collections import OrderedDict

import gin
import numpy as np
import torch
import torch.nn as nn
from planning_experiment.utils import get_mechanism


def set_global_seeds(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def get_latest_run_id(log_path):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    """
    max_run_id = 0

    loc = log_path + "[0-9]*"
    file_name = log_path.split("/")[-1][:-1]

    for path in glob.glob(loc):
        file = path.split("/")[-1].split("_")
        ext = file[-1]
        ext_file_name = "_".join(file[:-1])

        if (ext_file_name == file_name and ext.isdigit()
                and int(ext) > max_run_id
            ):
            max_run_id = int(ext)

    return max_run_id


def print_row(width, items):
    """ Prints the given items.  """
    def fmt_item(x):
        if isinstance(x, np.ndarray):
            assert x.ndim == 0
            x = x.item()
        if isinstance(x, float):
            rep = "%.3f" % x
        else:
            rep = str(x)
        return rep.ljust(width)

    print(" | ".join(fmt_item(item) for item in items))


def print_stats(info):
    """Prints performance statistics output from Runner."""
    print_row(15, ["Loss", "% Optimal", "% Success", "% Expansion"])
    print_row(
        15,
        [
            info["avg_loss"], info["avg_optimal"], info["avg_success"],
            info["avg_exp"]
        ],
    )
    return info


def load_state_dict(model, state_dict, multi_gpu):

    if multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" not in k:
                k = "module." + k
            else:
                k = k.replace("features.module.", "module.features.")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)


def gin_parse_args():
    """
    Load training configurations via gin-config
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gin-files",
        "-gf",
        type=str,
        required=True,
        nargs="+",
        help="gin config file(s)",
    )
    parser.add_argument("--gin-bindings",
                        "-gb",
                        type=str,
                        nargs="+",
                        help="gin extra binding(s)")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="planner to run (optional)",
        choices=[
            "NeuralAstar",
            "VanillaAstar",
            "BBAstar",
            "SAIL",
            "SAILGPU",
        ],
    )
    parser.add_argument(
        "--pretrained-path",
        "-p",
        type=str,
        help="path to pretrained model (optional)",
    )
    parser.add_argument(
        "--datafile",
        "-d",
        type=str,
        help="path to .npz data file (optional)",
    )
    parser.add_argument(
        "--logdir",
        "-w",
        type=str,
        help="logdir name (optional)",
    )
    parser.add_argument(
        "--use-multi-gpu",
        action='store_true',
        help="enable multi-gpu training (optional)",
    )

    args = parser.parse_args()
    gin_files = args.gin_files
    gin_bindings = [] if args.gin_bindings is None else args.gin_bindings

    gin.parse_config_files_and_bindings(gin_files, gin_bindings)

    if args.model is not None:
        planner = "@{}Planner".format(args.model)
        with gin.unlock_config():
            gin.bind_parameter("Runner.planner",
                               gin.config.parse_value(planner))

    if args.datafile is not None:
        datafile = args.datafile
        with gin.unlock_config():
            gin.bind_parameter("train.datafile", datafile)

    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path
        with gin.unlock_config():
            gin.bind_parameter("Runner.pretrained_path", pretrained_path)

    if args.logdir is not None:
        logdir = args.logdir
        with gin.unlock_config():
            gin.bind_parameter("train.logdir", logdir)

    if str(gin.query_parameter("Runner.mechanism") == "auto"):
        datafile = str(gin.query_parameter("train.datafile"))
        with gin.unlock_config():
            gin.bind_parameter("Runner.mechanism", get_mechanism(datafile))

    if args.use_multi_gpu:
        print("multi-gpu training enabled")
        with gin.unlock_config():
            gin.bind_parameter("Runner.multi_gpu", True)


def gin_get_config_dict():
    """
    Convert gin config string to parameters dictionary
    
    Returns:
        config_dict (dict): dictionary of gin configurations
    """

    gin_operative_config_str = gin.operative_config_str()
    # merge entries broken into multiple lines
    gin_operative_config_str = re.sub('\\\\\n', '', gin_operative_config_str)
    config = re.split("\n", gin_operative_config_str)
    config_dict = {}
    macro_keys = []
    for line in (x for x in config if x != ""):
        if line[0] == "#":
            continue

        # format keys and values
        k, v = re.split("=", line)
        k, v = re.sub("\s", "", k), re.sub("\s", "", v)

        # convert values to proper types
        tmp = gin.config.parse_value(v)
        config_dict[k] = (str(tmp) if type(tmp)
                          == gin.config.ConfigurableReference else tmp)

        # handle macros
        if v[0] == "%":
            macro_keys.append(v[1:])
            config_dict[k] = config_dict[v[1:]]

    # remove macros
    for k in macro_keys:
        config_dict.pop(k)

    return config_dict


def configure_logdirname(save_directory, model_name, important_parameters):
    save_directory += "/" + model_name + "/"
    for param in important_parameters:
        v = str(gin.query_parameter(param))
        if v[0] == "%":
            v = str(gin.query_parameter(v))
        save_directory += "{}_{}_".format(re.split("\.", param)[1], v)
    save_directory += "{:05d}".format(get_latest_run_id(save_directory) + 1)

    return save_directory