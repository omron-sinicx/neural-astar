"""Train a planner model for MP/TiledMP/CSM datasets
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
Parts of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

from __future__ import print_function

import os
import time
import re
import json

import numpy as np

import gin
import torch

from planning_experiment.utils import (
    print_row,
    print_stats,
    set_global_seeds,
    gin_parse_args,
    gin_get_config_dict,
    create_dataloader,
    configure_logdirname,
)
from planning_experiment.runner import Runner
from planning_experiment.planners import *


@gin.configurable
def train(
    datafile=gin.REQUIRED,
    logdir=gin.REQUIRED,
    save_intermediate=False,
    seed=1993,
    batch_size=gin.REQUIRED,
    num_epochs=gin.REQUIRED,
    important_parameters_path="config/important_parameters.json",
):

    # set random seed
    print("Using seed: {}".format(seed))
    set_global_seeds(seed)

    # save directory
    model_name = re.sub("Planner|\@", "",
                        gin.query_parameter("Runner.planner").__repr__())
    save_directory = os.path.join('log', logdir, datafile.split("/")[-1][:-4])
    important_parameters = json.load(open(important_parameters_path))
    save_directory = configure_logdirname(save_directory, model_name,
                                          important_parameters[model_name])
    print("save directory:", save_directory)
    os.makedirs(save_directory, exist_ok=True)

    # Create DataLoaders
    trainloader = create_dataloader(datafile,
                                    "train",
                                    batch_size,
                                    shuffle=True)
    validloader = create_dataloader(datafile,
                                    "valid",
                                    batch_size,
                                    shuffle=False)
    testloader = create_dataloader(datafile, "test", batch_size, shuffle=False)

    # Create runner
    runner = Runner()

    # Create logger
    gin_config_dict = gin_get_config_dict()
    with open("{}/config.json".format(save_directory), "w") as fp:
        json.dump(gin_config_dict, fp)

    # Print header
    col_width = 5
    print(
        "\n      |             Train             |              Valid            |"
    )
    print_row(
        col_width,
        [
            "Epoch",
            "Loss",
            "%Opt",
            "%Suc",
            "%Exp",
            "Loss",
            "%Opt",
            "%Suc",
            "%Exp",
            "W",
            "dW",
            "LR (x100)",
            "Time",
            "Best",
        ],
    )

    # Train and evaluate the model
    tr_total_loss, tr_total_optimal, tr_total_success, tr_total_exp = [], [], [], []
    v_total_loss, v_total_optimal, v_total_success, v_total_exp = [], [], [], []
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train the model
        tr_info = runner.train(trainloader)

        # Compute validation stats and save the best model
        v_info = runner.validate(validloader)
        time_duration = time.time() - start_time

        # Print epoch logs
        print_row(
            col_width,
            [
                epoch + 1,
                tr_info["avg_loss"],
                tr_info["avg_optimal"],
                tr_info["avg_success"],
                tr_info["avg_exp"],
                v_info["avg_loss"],
                v_info["avg_optimal"],
                v_info["avg_success"],
                v_info["avg_exp"],
                tr_info["weight_norm"],
                tr_info["grad_norm"],
                tr_info["lr"] * 100.,
                time_duration,
                "!" if v_info["is_best"] else " ",
            ],
        )

        # Keep track of metrics:
        tr_total_loss.append(tr_info["avg_loss"])
        tr_total_optimal.append(tr_info["avg_optimal"])
        tr_total_success.append(tr_info["avg_success"])
        tr_total_exp.append(tr_info["avg_exp"])
        v_total_loss.append(v_info["avg_loss"])
        v_total_optimal.append(v_info["avg_optimal"])
        v_total_success.append(v_info["avg_success"])
        v_total_exp.append(v_info["avg_exp"])

        # Save intermediate model.
        if save_intermediate:
            torch.save(
                {
                    "model": runner.planner.state_dict(),
                    "best_state_dict": runner.best_state_dict,
                    "tr_total_loss": tr_total_loss,
                    "tr_total_optimal": tr_total_optimal,
                    "tr_total_success": tr_total_success,
                    "tr_total_exp": tr_total_exp,
                    "v_total_loss": v_total_loss,
                    "v_total_optimal": v_total_optimal,
                    "v_total_success": v_total_success,
                    "v_total_exp": v_total_exp,
                },
                save_directory + "/e" + str(epoch) + ".pth",
            )

    # Test accuracy
    print("\nFinal test performance:")
    t_final_info = runner.test(testloader)
    print_stats(t_final_info)
    np.savez_compressed(save_directory + "/final",
                        **t_final_info["predictions"])

    print("\nBest test performance:")
    t_best_info = runner.test(testloader, load_best=True)
    print_stats(t_best_info)
    np.savez_compressed(save_directory + "/best", **t_best_info["predictions"])

    # Save the final trained model
    torch.save(
        {
            "model": runner.planner.model.state_dict(),
            "best_state_dict": runner.best_state_dict,
            "last_state_dict": runner.last_state_dict,
            "tr_total_loss": tr_total_loss,
            "tr_total_optimal": tr_total_optimal,
            "tr_total_success": tr_total_success,
            "tr_total_exp": tr_total_exp,
            "v_total_loss": v_total_loss,
            "v_total_optimal": v_total_optimal,
            "v_total_success": v_total_success,
            "v_total_exp": v_total_exp,
            "t_final_loss": t_final_info["avg_loss"],
            "t_final_optimal": t_final_info["avg_optimal"],
            "t_final_success": t_final_info["avg_success"],
            "t_final_exp": t_final_info["avg_exp"],
            "t_best_loss": t_best_info["avg_loss"],
            "t_best_optimal": t_best_info["avg_optimal"],
            "t_best_success": t_best_info["avg_success"],
            "t_best_exp": t_best_info["avg_exp"],
        },
        save_directory + "/data.pth",
    )


if __name__ == "__main__":

    # apply gin configurations
    gin_parse_args()

    train()
