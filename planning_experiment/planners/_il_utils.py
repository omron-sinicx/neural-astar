"""Utility functions for imitation learning planners.
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X

Small parts of this script has been copied from https://github.com/martius-lab/blackbox-backprop/blob/master/blackbox_backprop/shortest_path.py
"""

import math

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def _st_softmax_noexp(val):
    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y


def expand(x, neighbor_filter, padding=1):
    x = x.unsqueeze(0)
    num_samples = x.shape[1]
    y = F.conv2d(x, neighbor_filter, padding=padding,
                 groups=num_samples).squeeze()
    y = y.squeeze(0)
    return y


def backtrack(start_maps, goal_maps, parents, current_t):
    num_samples = start_maps.shape[0]
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    for t in range(current_t):
        path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
        loc = parents[range(num_samples), loc]
    return path_maps


class DifferentiableAstar(nn.Module):
    """
    Implementation based on https://rosettacode.org/wiki/A*_search_algorithm
    """
    def __init__(self,
                 mechanism,
                 g_ratio,
                 Tmax,
                 detach_g=True,
                 verbose=False):

        super().__init__()

        neighbor_filter = mechanism.get_neighbor_filter()
        self.neighbor_filter = nn.Parameter(neighbor_filter,
                                            requires_grad=False)
        self.get_heuristic = mechanism.get_heuristic

        self.g_ratio = g_ratio
        assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
        self.Tmax = Tmax
        self.detach_g = detach_g
        self.verbose = verbose

    def forward(self, cost_maps, start_maps, goal_maps, obstacles_maps):
        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4

        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0]

        num_samples = start_maps.shape[0]
        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples,
                                                  0)
        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)

        h = self.get_heuristic(goal_maps)
        h = h + cost_maps
        g = torch.zeros_like(start_maps)

        parents = (
            torch.ones_like(start_maps).reshape(num_samples, -1) *
            goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1])

        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):

            # select the node that minimizes cost
            f = self.g_ratio * g + (1 - self.g_ratio) * h
            f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1]))
            f_exp = f_exp * open_maps
            selected_node_maps = _st_softmax_noexp(f_exp)

            # break if arriving at the goal
            dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2),
                                                                keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()
            if torch.all(is_unsolved == 0):
                if self.verbose:
                    print("All problems solved at", t)
                break

            histories = histories + selected_node_maps
            histories = torch.clamp(histories, 0, 1)
            open_maps = open_maps - is_unsolved * selected_node_maps
            open_maps = torch.clamp(open_maps, 0, 1)

            # open neighboring nodes, add them to the openlist if they satisfy certain requirements
            neighbor_nodes = expand(selected_node_maps, neighbor_filter)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            # update g if one of the following conditions is met
            # 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
            # 2) neighbor is in the open list but g < g2
            g2 = expand((g + cost_maps) * selected_node_maps, neighbor_filter)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            if self.detach_g:
                g = g.detach()

            # update open maps
            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()

            # for backtracking
            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)

        # backtracking
        path_maps = backtrack(start_maps, goal_maps, parents, t)
        return histories.unsqueeze(1), path_maps.unsqueeze(1)


class Unet(nn.Module):

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def __init__(self, input_dim, encoder_backbone, encoder_depth):
        super().__init__()
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.model = smp.Unet(
            encoder_name=encoder_backbone,
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    def forward(self, x, map_designs):
        y = torch.sigmoid(self.model(x))
        if map_designs is not None:
            y = y * map_designs + torch.ones_like(y) * (1 - map_designs)
        return y


class UnetMlt(nn.Module):

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def __init__(self, input_dim, encoder_backbone, encoder_depth):
        super().__init__()
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.model = smp.Unet(
            encoder_name=encoder_backbone,
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )
        self.const = nn.Parameter(torch.ones(1) * 10.)

    def forward(self, x, map_designs):
        y = torch.sigmoid(self.model(x))
        if map_designs is not None:
            y = y * map_designs + torch.ones_like(y) * (1 - map_designs)
        return y * self.const


@gin.configurable
def MSELoss(history, opt_trajs):
    loss = nn.MSELoss()(history, opt_trajs)
    return loss


@gin.configurable("L1")
def L1Loss(history, opt_trajs):
    loss = nn.L1Loss()(history, opt_trajs)
    return loss


def get_min(val):
    y = val.reshape(val.shape[0], -1)
    min_val, ind = y.min(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return min_val, y_hard


def sample_onehot(binmaps):
    n_samples = len(binmaps)
    binmaps_n = binmaps * torch.rand_like(binmaps)
    binmaps_vct = binmaps_n.reshape(n_samples, -1)
    ind = binmaps_vct.max(dim=-1)[1]
    onehots = torch.zeros_like(binmaps_vct)
    onehots[range(n_samples), ind] = 1
    onehots = onehots.reshape_as(binmaps_n)

    return onehots


class sail_samples:
    feat = []
    oracle_q = []

    def append(self, f, q):
        self.feat.append(f)
        self.oracle_q.append(q)


def dilate_opt_trajs(opt_trajs, map_designs, mechanism):
    neighbor_filter = mechanism.get_neighbor_filter()
    neighbor_filter = neighbor_filter.type_as(opt_trajs)
    num_samples = len(opt_trajs)
    neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples, 0)
    ot_conv = expand(opt_trajs.squeeze(), neighbor_filter)
    ot_conv = torch.clamp(ot_conv.reshape_as(opt_trajs), 0, 1)
    ot_conv = ot_conv * map_designs
    return ot_conv


class BBAstarFunc(torch.autograd.Function):
    """
    Implementation of Blackbox Backprop for A* Search
    """
    @staticmethod
    def forward(ctx,
                cost_maps,
                start_maps,
                goal_maps,
                obstacles_maps,
                histories,
                astar_instance,
                lambda_val=5.0):
        ctx.lambda_val = lambda_val
        ctx.astar = astar_instance
        ctx.save_for_backward(cost_maps, start_maps, goal_maps, obstacles_maps)
        ctx.histories = histories
        return ctx.histories

    @staticmethod
    def backward(ctx, grad_output):
        cost_maps, start_maps, goal_maps, obstacles_maps = ctx.saved_variables
        cost_prime = cost_maps + ctx.lambda_val * grad_output
        cost_prime[cost_prime < 0.0] = 0.0  # instead of max(weights, 0.0)
        with torch.no_grad():
            better_histories, better_paths = ctx.astar(cost_prime, start_maps,
                                                       goal_maps,
                                                       obstacles_maps)
        gradient = -(ctx.histories - better_histories) / ctx.lambda_val
        return gradient, None, None, None, None, None, None