"""Summarize quantitative results for MP/TiledMP/CSM datasets
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

# Change these paths if necessary
MP_LOGDIR = "log/icml2021_mp"
TILEDMP_LOGDIR = "log/icml2021_tiledmp"
CSM_LOGDIR = "log/icml2021_csm"

from pprint import pprint
from glob import glob

from planning_experiment.utils.visualization import retrieve_and_print_metrics, get_query_configs

# MP Dataset
query_configs = get_query_configs(use_sailgpu=False)
query_configs[-3]["BBAstarPlanner.dilate_gt"] = False
query_configs[-2]["NeuralAstarPlanner.dilate_gt"] = False
query_configs[-1]["NeuralAstarPlanner.dilate_gt"] = False

config_files = glob(f"{MP_LOGDIR}/*moore*/**/config.json", recursive=True)
scores_moore = retrieve_and_print_metrics(query_configs,
                                          config_files,
                                          retrieve_all=True)
print("---MP Dataset---")
pprint(scores_moore)

# Tiled MP
query_configs = get_query_configs(use_sailgpu=True)
config_files = glob(f"{TILEDMP_LOGDIR}/*moore*/**/config.json", recursive=True)
scores_moore = retrieve_and_print_metrics(query_configs, config_files)
print("---Tiled MP Dataset---")
pprint(scores_moore)

# CSM
query_configs = get_query_configs(use_sailgpu=True)
config_files = glob(f"{CSM_LOGDIR}/*moore*/**/config.json", recursive=True)
scores_moore = retrieve_and_print_metrics(query_configs, config_files)
print("---CSM Dataset---")
pprint(scores_moore)