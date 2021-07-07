"""Visualization functions.
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import sys
import os
import re
import json
import numpy as np
from glob import glob

import gin
import torch
import pandas as pd
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from .metrics import compute_opt_exp
from .mechanism import get_mechanism
from .data import create_dataloader
from ..planners import *

NPZ_FILE = "best.npz"
DICT_KEY = "best_state_dict"


def compute_bsmean_cbound(pred_dists, rel_exps, opt_dists, masks):

    opt1, exp = [], []
    for i in range(len(pred_dists)):
        o1, e = compute_opt_exp(pred_dists[i:i + 1], rel_exps[i:i + 1],
                                opt_dists[i:i + 1], masks[i:i + 1])
        if (len(o1) > 0):
            opt1.append(o1.mean())
            exp.append(np.maximum(1 - e, 0).mean())
    opt1 = np.array(opt1)
    exp = np.array(exp)
    opt1_bounds = bs.bootstrap(
        opt1, stat_func=bs_stats.mean)  # use subopt score instead of opt
    exp_bounds = bs.bootstrap(exp, stat_func=bs_stats.mean)
    EPS = 1e-10
    hmean_value = 2. / (1. / (opt1 * 1. + EPS) + 1. / (exp + EPS))
    hmean_bounds = bs.bootstrap(hmean_value, stat_func=bs_stats.mean)

    scores = np.array([
        [opt1_bounds.value, opt1_bounds.lower_bound, opt1_bounds.upper_bound],
        [exp_bounds.value, exp_bounds.lower_bound, exp_bounds.upper_bound],
        [
            hmean_bounds.value, hmean_bounds.lower_bound,
            hmean_bounds.upper_bound
        ],
    ])
    return scores


def print_metrics(best_npz, method_name):
    printed_metrics = [method_name]
    for s in [[0, 15]]:
        masks = best_npz["masks"][:, s[0]:s[1]].max(1)
        scores = compute_bsmean_cbound(best_npz["pred_dists"],
                                       best_npz["rel_exps"],
                                       best_npz["opt_dists"], masks)
        for i in range(3):
            printed_metrics.append("{:0.1f} ({:0.1f}, {:0.1f})".format(
                scores[i, 0] * 100, scores[i, 1] * 100, scores[i, 2] * 100))
    return printed_metrics


def retrieve_data(query_config,
                  config_files,
                  retrieve_all=False,
                  retrieve_weights=False):

    best_npz = {"pred_dists": [], "opt_dists": [], "rel_exps": [], "masks": []}

    for config_file in config_files:
        ref_config = json.load(open(config_file))
        query_data = set([
            "{}_{}".format(k, v)
            for (k, v) in zip(query_config.keys(), query_config.values())
            if k[0] != "#"
        ])
        ref_data = set([
            "{}_{}".format(k, v)
            for (k, v) in zip(ref_config.keys(), ref_config.values())
        ])
        if query_data.issubset(ref_data):
            tmp = np.load(
                os.path.join(re.sub("config.json", NPZ_FILE, config_file)))
            if retrieve_all:
                best_npz["pred_dists"].append(tmp["pred_dists"])
                best_npz["opt_dists"].append(tmp["opt_dists"])
                best_npz["rel_exps"].append(tmp["rel_exps"])
                best_npz["masks"].append(tmp["masks"])
            else:
                best_npz = tmp

            if retrieve_weights:
                best_weights = torch.load(
                    os.path.join(re.sub("config.json", "data.pth",
                                        config_file)))[DICT_KEY]
            else:
                best_weights = None

    if best_npz["pred_dists"] == []:
        raise ValueError("query not found: {}".format(query_config))

    elif retrieve_all:
        best_npz["pred_dists"] = np.concatenate(best_npz["pred_dists"])
        best_npz["opt_dists"] = np.concatenate(best_npz["opt_dists"])
        best_npz["rel_exps"] = np.concatenate(best_npz["rel_exps"])
        best_npz["masks"] = np.concatenate(best_npz["masks"])

    return best_npz, best_weights


def retrieve_and_print_metrics(query_configs,
                               config_files,
                               retrieve_all=False):
    printed_metrics = []
    for query_config in query_configs:
        best_npz, _ = retrieve_data(query_config, config_files, retrieve_all)
        printed_metrics.append(
            print_metrics(best_npz, query_config["#planner_abb"]))
    printed_metrics = np.vstack(printed_metrics)
    printed_metrics = pd.DataFrame(printed_metrics[:, 1:],
                                   index=printed_metrics[:, 0],
                                   columns=[
                                       "Opt",
                                       "RExp",
                                       "HMean",
                                   ])
    return printed_metrics


def retrieve_and_visualize_results(query_configs,
                                   config_files,
                                   batch_size,
                                   rootdir="."):

    best_npz_va, _ = retrieve_data(query_configs[0], config_files)
    best_npz_vbf, _ = retrieve_data(query_configs[1], config_files)
    best_npz_na, _ = retrieve_data(query_configs[-1], config_files)
    exps_va = best_npz_va["rel_exps"]
    rel_exps_vbf = best_npz_vbf["rel_exps"]
    rel_exps_na = best_npz_na["rel_exps"]
    rel_exps = exps_va * (rel_exps_vbf - rel_exps_na)
    rel_exps_vbf[np.isnan(rel_exps_vbf)] = 9999
    rel_exps[rel_exps_vbf > 1] = 0
    rel_exps_vct = rel_exps.reshape(len(rel_exps), -1)
    idx = np.nanargmax(rel_exps_vct, 1)
    start_maps = np.zeros_like(rel_exps_vct)
    start_maps[range(len(idx)), idx] = 1
    start_maps = start_maps.reshape(rel_exps.shape)
    start_maps = torch.from_numpy(start_maps).float()

    # retrieve dataloader
    datafile = json.load(open(config_files[0]))['train.datafile']
    datafile = glob("{}/**/{}".format(rootdir, datafile), recursive=True)[0]
    dataloader = create_dataloader(datafile, 'test', len(rel_exps_vct))

    # select good samples
    val = np.nanmax(rel_exps_vct, 1)
    sample_ids = np.argsort(-val)[:batch_size]
    data = next(iter(dataloader))
    map_designs, goal_maps, opt_policies, opt_dists = data
    map_designs = map_designs.unsqueeze(1)
    map_designs = map_designs[sample_ids].cuda()
    start_maps = start_maps[sample_ids].cuda()
    goal_maps = goal_maps[sample_ids].cuda()
    opt_dists = opt_dists[sample_ids].cuda()

    # retrieve planners
    planners = []
    for query_config in query_configs:
        _, best_weights = retrieve_data(query_config,
                                        config_files,
                                        retrieve_weights=True)

        planner_cls = getattr(sys.modules[__name__],
                              query_config["#planner_name"])
        with gin.unlock_config():
            gin_files = [
                glob("{}/**/{}".format(rootdir, x), recursive=True)[0]
                for x in query_config["#planner_ginfiles"]
            ]
            gin.parse_config_files_and_bindings(gin_files, [])
        planner = planner_cls(get_mechanism(config_files[0]),
                              **query_config["#planner_params"])
        if (query_config["#planner_name"] == "SAIL"):
            planner.qnet.load_state_dict(best_weights)
        elif (query_config["#planner_name"] != "VanillaAstar"):
            planner.load_state_dict(best_weights)
        planners.append(planner)

    # run planners
    result_volumes_input, result_volumes, path_lens, num_exps, pred_costs_all = [], [], [], [], []
    for planner in planners:
        planner = planner.cuda()
        planner.eval()
        if "SAIL" in planner.__repr__():
            planner.qnet.eval()
            outputs = planner(map_designs, start_maps, goal_maps, opt_dists)
            histories, paths, _ = outputs
        elif "VanillaAstar" in planner.__repr__():
            outputs = planner(map_designs, start_maps, goal_maps)
            histories, paths = outputs
        elif "MMPAstar" in planner.__repr__():
            outputs = planner(map_designs, start_maps, goal_maps)
            pred_costs, obstacle_maps = outputs
            histories, paths = planner.perform_astar(-1 * pred_costs,
                                                     obstacle_maps, start_maps,
                                                     goal_maps)
        else:
            outputs = planner(map_designs, start_maps, goal_maps)
            histories, paths, pred_costs = outputs
            pred_costs_all.append(pred_costs.cpu().detach().squeeze())

        path_lens.append(paths.sum((1, 2, 3)))
        num_exps.append(histories.sum((1, 2, 3)))

        rv_in, rv = create_result_volume(map_designs.cpu(), start_maps.cpu(),
                                         goal_maps.cpu(), histories.cpu(),
                                         paths.cpu())
        result_volumes_input = rv_in
        result_volumes.append(rv)

    return result_volumes_input, result_volumes, path_lens, num_exps, pred_costs_all


def create_result_volume(map_designs,
                         start_maps,
                         goal_maps,
                         histories=None,
                         paths=None):
    result_volume_input = torch.ones_like(start_maps).expand(
        -1, 3, -1, -1).permute(0, 2, 3, 1) * .8
    result_volume_input *= map_designs.permute(0, 2, 3, 1)
    s_idx = start_maps.long()[:, 0]  # .squeeze()
    result_volume_input[s_idx == 1, :] = torch.tensor([1., 0,
                                                       0]).type_as(map_designs)
    g_idx = goal_maps.long()[:, 0]  # .squeeze()
    result_volume_input[g_idx == 1, :] = torch.tensor([1., 0,
                                                       0]).type_as(map_designs)
    result_volume = result_volume_input.clone()
    if (histories is not None):
        h_idx = histories.long()[:, 0]  # .squeeze()
        result_volume[h_idx == 1, :] = torch.tensor([.2, .8,
                                                     0]).type_as(map_designs)
    if (paths is not None):
        p_idx = paths.long()[:, 0]  # .squeeze()
        result_volume[p_idx == 1, :] = torch.tensor([1., 0,
                                                     0]).type_as(map_designs)

    return result_volume_input, result_volume


def get_query_configs(use_sailgpu=False):
    na_config = {
        '#planner_name': 'NeuralAstar',
        '#planner_abb': 'Neural A*',
        '#planner_params': {
            'g_ratio': 0.5,
            'Tmax': 1.0,
            'encoder_depth': 4,
            'encoder_backbone': 'vgg16_bn',
            'encoder_arch': 'Unet',
        },
        '#planner_ginfiles': ['macros.gin', 'neural_astar.gin'],
        'Runner.planner': '@NeuralAstarPlanner',
        'NeuralAstar.g_ratio': 0.5,
        'NeuralAstar.encoder_depth': 4,
        'NeuralAstar.encoder_arch': 'Unet',
        'NeuralAstar.encoder_backbone': 'vgg16_bn',
        'NeuralAstarPlanner.dilate_gt': True,
        'NeuralAstar.detach_g': True,
        'Runner.scheduler_params': None,
        'Runner.lr': 0.001,
    }
    nbf_config = {
        '#planner_name': 'NeuralAstar',
        '#planner_abb': 'Neural BF',
        '#planner_params': {
            'g_ratio': 0.0,
            'Tmax': 1.0,
            'encoder_depth': 4,
            'encoder_backbone': 'vgg16_bn',
            'encoder_arch': 'Unet',
        },
        '#planner_ginfiles': ['macros.gin', 'neural_astar.gin'],
        'Runner.planner': '@NeuralAstarPlanner',
        'NeuralAstar.g_ratio': 0.0,
        'NeuralAstar.encoder_depth': 4,
        'NeuralAstar.encoder_backbone': 'vgg16_bn',
        'NeuralAstar.encoder_arch': 'Unet',
        'NeuralAstarPlanner.dilate_gt': True,
        'NeuralAstar.detach_g': True,
        'Runner.scheduler_params': None,
        'Runner.lr': 0.001,
    }
    vbf_config = {
        '#planner_name': 'VanillaAstar',
        '#planner_abb': 'VBF',
        '#planner_params': {
            'g_ratio': 0.0,
            'Tmax': 1.0
        },
        '#planner_ginfiles': ['macros.gin', 'vanilla_astar.gin'],
        'Runner.planner': '@VanillaAstarPlanner',
        'VanillaAstar.g_ratio': 0.0,
    }
    vwa_config = {
        '#planner_name': 'VanillaAstar',
        '#planner_abb': 'VWA',
        '#planner_params': {
            'g_ratio': 0.2,
            'Tmax': 1.0
        },
        '#planner_ginfiles': ['macros.gin', 'vanilla_astar.gin'],
        'Runner.planner': '@VanillaAstarPlanner',
        'VanillaAstar.g_ratio': 0.2,
    }
    if use_sailgpu:
        sail_config = {
            '#planner_name': 'SAILGPU',
            '#planner_abb': 'SAIL',
            '#planner_params': {
                'beta0': 0.0,
                'Tmax': 1.0,
                'Nmax': 300,
                'cdist_factor': 2,
            },
            '#planner_ginfiles': ['macros.gin', 'sail_gpu.gin'],
            'Runner.planner': '@SAILGPUPlanner',
            'SAILGPU.Nmax': 300,
            'SAILGPU.beta0': 0.0,
            'SAILGPU.cdist_factor': 2,
        }
        sl_config = {
            '#planner_name': 'SAILGPU',
            '#planner_abb': 'SAIL-SL',
            '#planner_params': {
                'beta0': 1.0,
                'Tmax': 1.0,
                'Nmax': 300,
                'cdist_factor': 2,
            },
            '#planner_ginfiles': ['macros.gin', 'sail_gpu.gin'],
            'Runner.planner': '@SAILGPUPlanner',
            'SAILGPU.Nmax': 300,
            'SAILGPU.beta0': 1.0,
            'SAILGPU.cdist_factor': 2,
        }
    else:
        sail_config = {
            '#planner_name': 'SAIL',
            '#planner_abb': 'SAIL',
            '#planner_params': {
                'beta0': 0.0,
                'Tmax': 1.0,
                'Nmax': 300,
            },
            '#planner_ginfiles': ['macros.gin', 'sail.gin'],
            'Runner.planner': '@SAILPlanner',
            'SAIL.Nmax': 300,
            'SAIL.beta0': 0.0,
        }
        sl_config = {
            '#planner_name': 'SAIL',
            '#planner_abb': 'SAIL-SL',
            '#planner_params': {
                'beta0': 1.0,
                'Tmax': 1.0,
                'Nmax': 300,
            },
            '#planner_ginfiles': ['macros.gin', 'sail.gin'],
            'Runner.planner': '@SAILPlanner',
            'SAIL.Nmax': 300,
            'SAIL.beta0': 1.0,
        }
    bba_config = {
        '#planner_name': 'BBAstar',
        '#planner_abb': 'BB-A*',
        '#planner_params': {
            'g_ratio': 0.5,
            'Tmax': 1.0,
            'encoder_depth': 4,
        },
        '#planner_ginfiles': ['macros.gin', 'bb_astar.gin'],
        'Runner.planner': '@BBAstarPlanner',
        'BBAstar.bbastar_lambda': 20.,
    }
    query_configs = [
        vbf_config, vwa_config, sail_config, sl_config, bba_config, nbf_config,
        na_config
    ]

    return query_configs