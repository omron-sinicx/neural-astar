"""Base class of imitation learning planners.
Author: Ryo Yonetani, Mohammadamin Barekatain
Affiliation: OMRON SINIC X
"""

import numpy as np

import gin

from ._base import _PlannerBaseClass

from ..utils.data import (
    get_hard_medium_easy_masks,
    _sample_onehot,
)

from ..utils.metrics import compute_mean_metrics


@gin.configurable
class _ILPlanner(_PlannerBaseClass):
    def __init__(self, mechanism, loss_fn, sample_from, model_cls):
        super().__init__(mechanism, model_cls)
        self.loss_fn = loss_fn
        self.sample_from = sample_from

    def _forward(
        self,
        map_designs,
        goal_maps,
        opt_policies_CPU,
        start_maps_CPU,
        goal_maps_CPU,
        opt_dists_CPU,
        device,
    ):
        raise NotImplementedError()

    def train_self(
        self,
        map_designs,
        goal_maps,
        opt_policies,
        opt_dists,
        map_designs_CPU,
        goal_maps_CPU,
        opt_policies_CPU,
        opt_dists_CPU,
        device,
    ):
        # randomly select a start point for each map
        masks = get_hard_medium_easy_masks(opt_dists_CPU, reduce_dim=True)
        if self.sample_from == -1:
            masks = np.concatenate(masks, axis=1).max(axis=1, keepdims=True)
        else:
            masks = masks[self.sample_from]
        start_maps_CPU = _sample_onehot(masks)
        rel_exps_maps = np.empty_like(opt_dists_CPU)
        loss, pred_dists, rel_exps = self._forward(
            map_designs,
            goal_maps,
            opt_policies_CPU,
            start_maps_CPU,
            goal_maps_CPU,
            opt_dists_CPU,
            device,
        )

        pred_dist_maps = np.empty_like(opt_dists_CPU)
        pred_dist_maps[:] = np.NAN
        masks = start_maps_CPU.astype(bool)
        pred_dist_maps[masks] = pred_dists[:]
        rel_exps_maps[masks] = rel_exps[:]

        p_opt, p_suc, p_exp = compute_mean_metrics(pred_dist_maps,
                                                   rel_exps_maps,
                                                   opt_dists_CPU, masks)

        return loss, p_opt, p_suc, p_exp

    def eval_self(
        self,
        map_designs,
        goal_maps,
        opt_policies,
        opt_dists,
        map_designs_CPU,
        goal_maps_CPU,
        opt_policies_CPU,
        opt_dists_CPU,
        device,
        num_eval_points,
    ):

        masks = get_hard_medium_easy_masks(opt_dists_CPU, False,
                                           num_eval_points)
        masks = np.concatenate(masks, axis=1)
        pred_dist_maps = np.empty_like(opt_dists_CPU)
        pred_dist_maps[:] = np.NAN
        loss_tot = 0.0
        rel_exps_maps = np.empty_like(opt_dists_CPU)
        rel_exps_maps[:] = np.NAN

        for i in range(masks.shape[1]):
            loss, pred_dists, rel_exps = self._forward(
                map_designs,
                goal_maps,
                opt_policies_CPU,
                masks[:, i],
                goal_maps_CPU,
                opt_dists_CPU,
                device,
            )
            loss_tot += loss
            pred_dist_maps[masks[:, i]] = pred_dists[:]
            rel_exps_maps[masks[:, i]] = rel_exps[:]
        loss_tot /= masks.shape[1]

        p_opt, p_suc, p_exp = compute_mean_metrics(
            pred_dist_maps,
            rel_exps_maps,
            opt_dists_CPU,
            masks.max(axis=1),
        )

        return loss_tot, p_opt, p_suc, p_exp, pred_dist_maps, rel_exps_maps, masks
