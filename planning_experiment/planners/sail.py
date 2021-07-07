"""SAIL planners (computing cdist on CPU).
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import numpy as np

import gin
import torch
import torch.nn as nn

from ._il_planner import _ILPlanner
from ._il_utils import (
    expand,
    backtrack,
    get_min,
    sample_onehot,
    sail_samples,
)
from ..utils.data import get_opt_trajs


@gin.configurable(blacklist=["mechanism"])
class SAIL(nn.Module):
    """
    Implementation of the SAIL planner (https://arxiv.org/abs/1707.03034).
    """
    def __init__(self,
                 mechanism,
                 beta0,
                 num_skips,
                 Tmax=1.,
                 Nmax=60,
                 verbose=False):
        super().__init__()

        neighbor_filter = mechanism.get_neighbor_filter()
        self.neighbor_filter = nn.Parameter(neighbor_filter,
                                            requires_grad=False)
        self.beta0 = beta0
        self.num_skips = num_skips
        self.Tmax = Tmax
        self.Nmax = Nmax
        self.verbose = verbose

        self.qnet = nn.Sequential(
            nn.Linear(17, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, map_designs, start_maps, goal_maps, opt_dists):
        assert map_designs.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4

        map_designs = map_designs[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        opt_dists = -opt_dists[:, 0]

        obstacles_maps = map_designs

        num_samples = start_maps.shape[0]
        size = start_maps.shape[-1]

        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples,
                                                  0)

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)

        grid = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
        pos = torch.stack(grid, dim=0).type(start_maps.type())
        pos_expand = pos.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
        goal_loc = torch.einsum("kij, bij -> bk", pos, goal_maps)
        goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

        # search-based features
        x_v = pos_expand.reshape(num_samples, 2, size, size)
        x_g = goal_loc_expand.reshape(num_samples, 2, 1, 1)
        x_g = x_g.expand(num_samples, 2, size, size)
        g_v = torch.zeros_like(start_maps).unsqueeze(1)
        h_euc = torch.sqrt(((pos_expand - goal_loc_expand)**2).sum(1))
        h_euc = h_euc.reshape_as(start_maps).unsqueeze(1)
        h_man = torch.abs(pos_expand - goal_loc_expand).sum(1)
        h_man = h_man.reshape_as(start_maps).unsqueeze(1)
        d_tree = torch.zeros_like(start_maps).unsqueeze(1)
        feat = torch.cat((g_v, d_tree, x_v, x_g, h_euc, h_man), 1)

        # environment-based feature
        obs_expand = obstacles_maps.reshape(num_samples, -1).unsqueeze(1)
        obs_expand = pos_expand * (1 - obs_expand) + 100000 * obs_expand
        # HACK: don't compute cdist on GPU as it's a memory eater
        pop = pos_expand.permute(0, 2, 1).to('cpu')
        oep = obs_expand.permute(0, 2, 1).to('cpu')
        for i in [[0, 1], [1, 2], [0, 2]]:
            cdist = torch.cdist(pop[..., i[0]:i[1]], oep[..., i[0]:i[1]])
            d2o_val, d2o_loc = cdist.min(dim=2)
            d2o_val = d2o_val.reshape_as(start_maps).unsqueeze(1)
            d2o_loc = (pos.reshape(2, -1).T[d2o_loc].permute(0, 2, 1).reshape(
                num_samples, 2, size, size))
            feat = torch.cat((feat, d2o_val.type_as(pos_expand),
                              d2o_loc.type_as(pos_expand)), 1)
        feat = feat.permute(0, 2, 3, 1)  # TODO: need a more efficient approach

        # NOTE: feat[..., 0] -> g_v and feat[..., 1] -> d_tree, which will be updated during search
        g_val = 0

        parents = (
            torch.ones_like(start_maps).reshape(num_samples, -1) *
            goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1])

        samples = sail_samples()
        Tmax = int(self.Tmax * size * size)
        Nmax = int(self.Nmax)
        for t in range(Tmax):
            g_val = g_val + 1
            qnet_out = self.qnet(feat).squeeze().detach()

            # store training samples
            if np.mod(t, self.num_skips) == 0:
                oh = sample_onehot(open_maps)
                samples.append((feat * oh.unsqueeze(-1)).sum((1, 2)),
                               (opt_dists * oh).sum((1, 2)))

            if self.qnet.training:
                h = self.beta0 * opt_dists + (1 - self.beta0) * qnet_out
            else:
                h = qnet_out

            h = h * open_maps + 100000 * (1 - open_maps
                                          )  # mask unopen locations
            v, selected_node_maps = get_min(h)

            # break if arriving at the goal
            dist_to_goal = torch.norm(selected_node_maps - goal_maps,
                                      dim=(1, 2),
                                      keepdim=True)
            is_unsolved = (dist_to_goal > 1e-8).float()
            if torch.all(is_unsolved == 0):
                if self.verbose:
                    print("All problems solved at", t)
                break
            open_maps = torch.clamp(
                open_maps - is_unsolved * selected_node_maps, 0, 1)
            histories = torch.clamp(histories + selected_node_maps, 0, 1)
            neighbors = expand(selected_node_maps, neighbor_filter)
            neighbors = neighbors * obstacles_maps * (1 - histories)
            g_map = feat[..., 0].clone()
            d_map = feat[..., 1].clone()
            feat[..., 0] = g_map * (1 - neighbors) + g_val * neighbors
            d_val = (expand(
                (feat[..., 1] + 1) * selected_node_maps, neighbor_filter) *
                     obstacles_maps)
            d_val = ((1 - histories) + histories * (g_val < g_map)) * d_val
            feat[..., 1] = d_map * (1 - neighbors) + d_val * neighbors
            open_maps = torch.clamp(open_maps + neighbors, 0, 1)

            # for backtracking
            neighbors = neighbors.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * neighbors + parents * (1 - neighbors)

        # backtracking
        path_maps = backtrack(start_maps, goal_maps, parents, t)

        feat, oracle_q = samples.feat, samples.oracle_q
        if (len(feat) > Nmax):
            feat = feat[0:-1:int(len(feat) / Nmax)][:Nmax]
            oracle_q = oracle_q[0:-1:int(len(oracle_q) / Nmax)][:Nmax]
        if self.training:
            qnet_out = self.qnet(torch.cat(feat))
        else:
            with torch.no_grad():
                qnet_out = self.qnet(torch.cat(feat))

        oracle_q = torch.cat(oracle_q).unsqueeze(1)
        training_pairs = [qnet_out, oracle_q]

        return histories.unsqueeze(1), path_maps.unsqueeze(1), training_pairs


@gin.configurable
class SAILPlanner(_ILPlanner):
    def __init__(self,
                 mechanism,
                 loss_fn,
                 sample_from,
                 skip_exp_when_training=True,
                 model_cls=SAIL):
        super().__init__(mechanism, loss_fn, sample_from, model_cls)
        # HACK: expose model.sail instead of the model itself as self.model
        # so that data parallelism can be done properly
        self.sail = self.model
        self.model = self.sail.qnet
        self.skip_exp_when_training = skip_exp_when_training

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
        self.sail = self.sail.to(device)
        start_maps = torch.from_numpy(start_maps_CPU).float().to(device)
        opt_dists = torch.from_numpy(opt_dists_CPU).to(device)

        outputs = self.sail.forward(map_designs, start_maps, goal_maps,
                                    opt_dists)

        # compute optimal trajectory
        opt_trajs_CPU = get_opt_trajs(start_maps_CPU, goal_maps_CPU,
                                      opt_policies_CPU, self.mechanism)
        opt_trajs = torch.from_numpy(opt_trajs_CPU).float().to(device)
        loss = self.loss_fn(outputs[2][0], outputs[2][1])

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

            # # relative number of expansions
            pred_exps = outputs[0].cpu().data.numpy().sum((1, 2, 3))
            if (self.model.training & self.skip_exp_when_training) is not True:
                astar_outputs = self.astar_ref(map_designs, start_maps,
                                               goal_maps, map_designs)
                exps = astar_outputs[0].cpu().data.numpy().sum((1, 2, 3))
            else:
                exps = pred_exps
            rel_exps = pred_exps / exps

        return loss, pred_dists, rel_exps
