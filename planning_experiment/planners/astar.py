"""A*-based planners
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""


import sys
import numpy as np

import gin
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from ._il_planner import _ILPlanner
from ._il_utils import (
    DifferentiableAstar,
    BBAstarFunc,
    Unet,
    UnetMlt,
    dilate_opt_trajs,
)
from ..utils.data import get_opt_trajs


@gin.configurable(blacklist=["mechanism"])
class NeuralAstar(nn.Module):
    """
    Implementation of Neural A* Search
    """
    def __init__(
        self,
        mechanism,
        encoder_input,
        encoder_arch,
        encoder_backbone,
        encoder_depth,
        ignore_obstacles,
        learn_obstacles,
        g_ratio,
        Tmax,
        detach_g,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.astar = DifferentiableAstar(
            mechanism=mechanism,
            g_ratio=g_ratio,
            Tmax=Tmax,
            detach_g=detach_g,
        )
        self.encoder_input = encoder_input
        encoder = getattr(sys.modules[__name__], encoder_arch)
        self.encoder = encoder(len(self.encoder_input), encoder_backbone,
                               encoder_depth)
        self.ignore_obstacles = ignore_obstacles
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print('WARNING: learn_obstacles has been set to True')

    def forward(self, map_designs, start_maps, goal_maps):
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        pred_cost_maps = self.encoder(
            inputs, map_designs if not self.ignore_obstacles else None)
        obstacles_maps = map_designs if not self.learn_obstacles else torch.ones_like(
            map_designs)

        histories, paths = self.astar(pred_cost_maps, start_maps, goal_maps,
                                      obstacles_maps)

        return histories, paths, pred_cost_maps


@gin.configurable(blacklist=["mechanism"])
class VanillaAstar(nn.Module):
    """
    Implementation of Vanilla A* Search
    """
    def __init__(
        self,
        mechanism,
        g_ratio,
        Tmax,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.astar = DifferentiableAstar(
            mechanism=mechanism,
            g_ratio=g_ratio,
            Tmax=Tmax,
        )

    def forward(self, map_designs, start_maps, goal_maps):
        obstacles_maps = map_designs

        histories, paths = self.astar(map_designs, start_maps, goal_maps,
                                      obstacles_maps)

        return histories, paths


@gin.configurable(blacklist=["mechanism"])
class BBAstar(nn.Module):
    """
    Implementation of Black-box A* Search
    """
    def __init__(
        self,
        mechanism,
        encoder_input,
        encoder_arch,
        encoder_backbone,
        encoder_depth,
        ignore_obstacles,
        learn_obstacles,
        g_ratio,
        Tmax,
        detach_g,
        bbastar_lambda,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.astar = DifferentiableAstar(
            mechanism=mechanism,
            g_ratio=g_ratio,
            Tmax=Tmax,
            detach_g=detach_g,
        )
        self.bbastar = BBAstarFunc.apply
        self.encoder_input = encoder_input
        encoder = getattr(sys.modules[__name__], encoder_arch)
        self.encoder = encoder(len(self.encoder_input), encoder_backbone,
                               encoder_depth)
        self.ignore_obstacles = ignore_obstacles
        self.learn_obstacles = learn_obstacles
        self.bbastar_lambda = bbastar_lambda

        if self.learn_obstacles:
            print('WARNING: learn_obstacles has been set to True')

    def forward(self, map_designs, start_maps, goal_maps):
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        pred_cost_maps = self.encoder(
            inputs, map_designs if not self.ignore_obstacles else None)
        obstacles_maps = map_designs if not self.learn_obstacles else torch.ones_like(
            map_designs)

        with torch.no_grad():
            histories, paths = self.astar(pred_cost_maps, start_maps,
                                          goal_maps, obstacles_maps)

        histories = self.bbastar(pred_cost_maps, start_maps, goal_maps,
                                 obstacles_maps, histories, self.astar,
                                 self.bbastar_lambda)

        return histories, paths, pred_cost_maps


@gin.configurable
class NeuralAstarPlanner(_ILPlanner):
    def __init__(self,
                 mechanism,
                 loss_fn,
                 sample_from,
                 dilate_gt,
                 skip_exp_when_training=True,
                 output_exp_instead_of_rel_exp=False,
                 model_cls=NeuralAstar):
        super().__init__(mechanism, loss_fn, sample_from, model_cls)
        self.dilate_gt = dilate_gt
        self.skip_exp_when_training = skip_exp_when_training
        self.output_exp_instead_of_rel_exp = output_exp_instead_of_rel_exp

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
        start_maps = torch.from_numpy(start_maps_CPU).float().to(device)

        outputs = self.model.forward(map_designs, start_maps, goal_maps)

        # compute optimal trajectory
        opt_trajs_CPU = get_opt_trajs(start_maps_CPU, goal_maps_CPU,
                                      opt_policies_CPU, self.mechanism)
        opt_trajs = torch.from_numpy(opt_trajs_CPU).float().to(device)
        if self.dilate_gt:
            opt_trajs_ = dilate_opt_trajs(opt_trajs, map_designs,
                                          self.mechanism)
        else:
            opt_trajs_ = opt_trajs
        loss = self.loss_fn(outputs[0], opt_trajs_)

        wall_dist = np.min(opt_dists_CPU)
        with torch.no_grad():
            pred_dists = -(outputs[1].sum(dim=(1, 2, 3)) - 1)
            arrived = (outputs[1] * start_maps).sum(dim=(1, 2, 3))
            not_passed_through_obstacles = (outputs[1] *
                                            (1 - map_designs)).sum(
                                                dim=(1, 2, 3)) == 0
            arrived = arrived * not_passed_through_obstacles
            pred_dists = pred_dists * arrived + wall_dist * (1.0 - arrived)
            pred_dists = pred_dists.cpu().data.numpy()

            # relative number of expansions
            pred_exps = outputs[0].cpu().data.numpy().sum((1, 2, 3))
            if (self.model.training & self.skip_exp_when_training) is not True:
                astar_outputs = self.astar_ref(map_designs, start_maps,
                                               goal_maps, map_designs)
                exps = astar_outputs[0].cpu().data.numpy().sum((1, 2, 3))
            else:
                exps = pred_exps
            rel_exps = pred_exps / exps

            if self.output_exp_instead_of_rel_exp:
                rel_exps = pred_exps

        return loss, pred_dists, rel_exps


@gin.configurable
class VanillaAstarPlanner(NeuralAstarPlanner):
    def __init__(self,
                 mechanism,
                 loss_fn,
                 sample_from,
                 dilate_gt,
                 skip_exp_when_training=True,
                 output_exp_instead_of_rel_exp=False,
                 model_cls=VanillaAstar):
        super().__init__(mechanism, loss_fn, sample_from, dilate_gt,
                         skip_exp_when_training, output_exp_instead_of_rel_exp,
                         model_cls)


@gin.configurable
class BBAstarPlanner(NeuralAstarPlanner):
    def __init__(self,
                 mechanism,
                 loss_fn,
                 sample_from,
                 dilate_gt,
                 skip_exp_when_training=True,
                 output_exp_instead_of_rel_exp=False,
                 model_cls=BBAstar):
        super().__init__(mechanism, loss_fn, sample_from, dilate_gt,
                         skip_exp_when_training, output_exp_instead_of_rel_exp,
                         model_cls)
