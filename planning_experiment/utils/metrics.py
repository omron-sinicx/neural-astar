"""Evaluation functions.
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import numpy as np


def compute_opt_suc_exp(pred_dists, rel_exps, opt_dists, masks):

    wall_dist = np.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt = (diff_dists == 0)[masks]
    suc = 1.0 - (pred_dists == wall_dist)[masks]
    exp = (np.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])
    if len(exp) == 0:
        exp = np.ones_like(pred_dists)[masks]  ## workaround

    return opt, suc, exp


def compute_mean_metrics(pred_dists, rel_exps, opt_dists, masks):

    opt, suc, exp = compute_opt_suc_exp(pred_dists, rel_exps, opt_dists, masks)

    return opt.mean(), suc.mean(), exp.mean()


def compute_opt_exp(pred_dists, rel_exps, opt_dists, masks):

    wall_dist = np.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt1 = (diff_dists == 0)[(pred_dists != wall_dist) & masks]
    exp = (np.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])

    return opt1, exp
