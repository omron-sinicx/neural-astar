"""Summarize quantitative results for SDD
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from glob import glob

# Change this path if necessary
LOGDIR = "log/icml2021_sdd"


def show_scores(max_steps=300, max_hardness=1.0):
    print("Intra-scene scenarios")
    for method in ["BBAstar_UnetMlt_4", "NeuralAstar_UnetMlt_4"]:
        cd = np.concatenate([
            np.load(x)["test_cd"] for x in glob(
                f"{LOGDIR}/video0*{max_hardness:0.2f}_{max_steps:04d}/{method}/loss.npz"
            )
        ])
        hardness = np.concatenate([
            np.load(x)["test_lr"] for x in glob(
                f"{LOGDIR}/video0*{max_hardness:0.2f}_{max_steps:04d}/{method}/loss.npz"
            )
        ])
        cd = cd[hardness <= max_hardness]
        score_bs = bs.bootstrap(cd, stat_func=bs_stats.mean)
        print(
            method, "%0.1f (%0.1f, %0.1f)" %
            (score_bs.value, score_bs.lower_bound, score_bs.upper_bound))

    print("Inter-scene scenarios")
    for method in ["BBAstar_UnetMlt_4", "NeuralAstar_UnetMlt_4"]:
        cd = np.concatenate([
            np.load(x)["test_cd"] for x in glob(
                f"{LOGDIR}/*{max_hardness:0.2f}_{max_steps:04d}/{method}/loss.npz"
            ) if "video0" not in x
        ])
        hardness = np.concatenate([
            np.load(x)["test_lr"] for x in glob(
                f"{LOGDIR}/*{max_hardness:0.2f}_{max_steps:04d}/{method}/loss.npz"
            ) if "video0" not in x
        ])
        cd = cd[hardness <= max_hardness]
        score_bs = bs.bootstrap(cd, stat_func=bs_stats.mean)
        print(
            method, "%0.1f (%0.1f, %0.1f)" %
            (score_bs.value, score_bs.lower_bound, score_bs.upper_bound))


# main result
show_scores(300, 1.0)