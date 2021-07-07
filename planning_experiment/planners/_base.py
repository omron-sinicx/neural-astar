"""Base class for all planners
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import abc
from ._il_utils import DifferentiableAstar


class _PlannerBaseClass(abc.ABC):
    def __init__(self, mechanism, model_cls):
        self.model = model_cls(mechanism)
        self.astar_ref = DifferentiableAstar(
            mechanism,
            g_ratio=0.5,
            Tmax=1,
        )
        self.mechanism = mechanism

    def loss_fn(self):
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
        raise NotImplementedError()

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
        raise NotImplementedError()
