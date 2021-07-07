"""Visualizng results for SDD
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gin

from planning_experiment.utils.SDD import SDD
from planning_experiment.utils import get_mechanism
from planning_experiment.planners import NeuralAstar, BBAstar

# Change this path if necessary
LOGDIR = "log/icml2021_sdd"


def main(args):
    ms = int(args.max_steps)

    with gin.unlock_config():
        gin.parse_config_files_and_bindings([
            "config/macros.gin",
            "config/neural_astar.gin",
            "config/bb_astar.gin",
        ], [])

    hardness = args.hardness
    test_scene = args.test_scene
    os.makedirs("figures/sdd_%0.2f_%04d_%s" % (hardness, ms, test_scene),
                exist_ok=True)

    # Load models
    na = NeuralAstar(get_mechanism("moore"),
                     encoder_input="rgb+",
                     learn_obstacles=True,
                     ignore_obstacles=True,
                     Tmax=1.0,
                     encoder_arch="UnetMlt").cuda()
    na.load_state_dict(
        torch.load(
            f"{LOGDIR}/{test_scene}_moore_{hardness:0.2f}_{ms:04d}/NeuralAstar_UnetMlt_4/data.pth"
        ))
    bba = BBAstar(get_mechanism("moore"),
                  encoder_input="rgb+",
                  learn_obstacles=True,
                  ignore_obstacles=True,
                  Tmax=1.0,
                  encoder_arch="UnetMlt").cuda()
    bba.load_state_dict(
        torch.load(
            f"{LOGDIR}/{test_scene}_moore_{hardness:0.2f}_{ms:04d}/BBAstar_UnetMlt_4/data.pth"
        ))

    # Load data
    test_dataset = SDD("data/sdd/s064_0.5_128_%03d" % ms,
                       is_train=False,
                       test_scene=test_scene)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Visualize results
    for i, samples in tqdm(enumerate(dataloader)):
        images = samples["image"].cuda()
        start_images = samples["start_image"].cuda()
        goal_images = samples["goal_image"].cuda()
        traj_images = samples["traj_image"].cuda()

        histories_na, paths_na, pred_costs_na = na(images, start_images,
                                                   goal_images)
        histories_bba, paths_bba, pred_costs_bba = bba(images, start_images,
                                                       goal_images)

        fig, ax = plt.subplots(1, 5, figsize=[12, 4])
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        ax[0].imshow(images[0].permute(1, 2, 0).cpu())
        start_loc = np.argwhere(start_images[0].squeeze().cpu().numpy())[0]
        goal_loc = np.argwhere(goal_images[0].squeeze().cpu().numpy())[0]
        ax[0].plot(start_loc[1], start_loc[0], "ro")
        ax[0].plot(goal_loc[1], goal_loc[0], "ro")
        ax[1].imshow(images[0].permute(1, 2, 0).cpu())
        pts = np.argwhere(traj_images[0].squeeze().cpu().numpy())
        ax[1].plot(pts[:, 1], pts[:, 0], "ro", markersize=5)
        ax[2].imshow(images[0].permute(1, 2, 0).cpu())
        pts = np.argwhere(paths_bba[0].squeeze().cpu().numpy())
        ax[2].plot(pts[:, 1], pts[:, 0], "ro", markersize=5)
        ax[3].imshow(images[0].permute(1, 2, 0).cpu())
        pts = np.argwhere(paths_na[0].squeeze().cpu().numpy())
        ax[3].plot(pts[:, 1], pts[:, 0], "ro", markersize=5)
        ax[4].imshow(images[0].permute(1, 2, 0).cpu())
        ax[4].imshow(pred_costs_na[0].detach().squeeze().cpu(),
                     alpha=.8,
                     cmap="BuGn")
        fig.tight_layout()
        plt.savefig("figures/sdd_%0.2f_%04d_%s/result_%08d.pdf" %
                    (hardness, ms, test_scene, i))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", "-ms", type=int, default=300)
    parser.add_argument("--hardness", "-hd", type=float, default=1.0)
    parser.add_argument("--test-scene", "-ts", type=str, default="video0")

    args = parser.parse_args()
    main(args)