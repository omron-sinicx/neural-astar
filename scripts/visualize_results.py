"""Visualizng results for MP/TiledMP/CSM datasets
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import os
import re

from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
torch.backends.cudnn.benchmark = True

from planning_experiment.utils.visualization import retrieve_and_visualize_results, get_query_configs

# Change these paths if necessary
MP_LOGDIR = "/data/log/plexp1_032"
TILEDMP_LOGDIR = "log/icml2021_tiledmp"
CSM_LOGDIR = "log/icml2021_csm"


# Drawing function
def draw(query_configs, logdir, N=50):
    assert N > 1
    map_name = re.split("/", logdir)[-1]
    config_files = glob(os.path.join(logdir, "**/config.json"), recursive=True)
    os.makedirs(os.path.join("figures", map_name), exist_ok=True)
    rv_input, rv, path_lens, num_exps, pc = retrieve_and_visualize_results(
        query_configs, config_files, N, rootdir=".")

    num_methods = len(rv)
    for t in tqdm(range(N), desc=logdir):
        fig, ax = plt.subplots(1, num_methods + 1, figsize=[15, 3])
        for i in range(num_methods):
            ax[i].imshow(rv[i][t])
            ax[i].set_title("H:{:4d} \nP:{:3d}".format(
                num_exps[i][t].int().item(),
                path_lens[i][t].item(),
                fontsize=5),
                            fontsize=14)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        ax[-1].imshow(rv_input[t])
        ax[-1].imshow(pc[-1][t], cmap="BuGn", alpha=.8)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        fig.tight_layout()
        plt.savefig(
            os.path.join("figures", map_name, "result_{:04d}.pdf".format(t)))
        plt.close()


def main():

    # Base config
    va_config = {
        "#planner_name": "VanillaAstar",
        "#planner_abb": "VA",
        "#planner_params": {
            "g_ratio": 0.5,
            "Tmax": 1.0
        },
        "#planner_ginfiles": ["macros.gin", "vanilla_astar.gin"],
        "Runner.planner": "@VanillaAstarPlanner",
        "VanillaAstar.g_ratio": 0.5,
    }
    query_configs = [va_config] + get_query_configs()

    # Exp 1
    query_configs[-3]["BBAstarPlanner.dilate_gt"] = False
    query_configs[-2]["NeuralAstarPlanner.dilate_gt"] = False
    query_configs[-1]["NeuralAstarPlanner.dilate_gt"] = False

    for logdir in glob(f"{MP_LOGDIR}/*moore*"):
        draw(query_configs, logdir, N=100)

    # Exp 2 and 3
    query_configs[-3]["BBAstarPlanner.dilate_gt"] = True
    query_configs[-2]["NeuralAstarPlanner.dilate_gt"] = True
    query_configs[-1]["NeuralAstarPlanner.dilate_gt"] = True

    for logdir in glob(f"{TILEDMP_LOGDIR}/all_064_moore_c16"):
        draw(query_configs, logdir, N=50)

    for logdir in glob(f"{CSM_LOGDIR}/mixed_064_moore_c16"):
        draw(query_configs, logdir, N=50)


if __name__ == "__main__":
    main()