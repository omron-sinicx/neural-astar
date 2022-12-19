"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from neural_astar.planner.astar import VanillaAstar
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torch.nn.modules.loss import _Loss
from torchvision.utils import make_grid
from glob import glob


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


class PlannerModule(pl.LightningModule):
    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config

    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(self.planner.parameters(), self.config.params.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories, opt_trajs)
        self.log("metrics/train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories, opt_trajs)
        va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
        pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        p_opt = (pathlen_astar == pathlen_model).mean()

        exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

        h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

        self.log("metrics/val_loss", loss)
        self.log("metrics/p_opt", p_opt)
        self.log("metrics/p_exp", p_exp)
        self.log("metrics/h_mean", h_mean)

        return loss


EPS = 1e-10


@dataclass
class Metrics:
    p_opt: float
    p_exp: float
    h_mean: float

    def __repr__(self):
        return f"optimality: {self.p_opt:0.3f}, efficiency: {self.p_exp:0.3f}, h_mean: {self.h_mean:0.3f}"


def run_planner(
    batch: Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor],
    planner: nn.Module,
    criterion: _Loss,
) -> Tuple[torch.tensor, AstarOutput]:
    """
    Run planner on a given batch

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]): input batch
        planner (nn.Module): planner
        criterion (_Loss): loss function

    Returns:
        Tuple[torch.tensor, AstarOutput]: computed loss + planner output
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_designs, start_maps, goal_maps, opt_trajs = batch
    map_designs = map_designs.to(device)
    start_maps = start_maps.to(device)
    goal_maps = goal_maps.to(device)
    opt_trajs = opt_trajs.to(device)
    planner_outputs = planner(map_designs, start_maps, goal_maps)
    loss = criterion(planner_outputs.histories, opt_trajs)

    return loss, planner_outputs


def calc_metrics(na_outputs: AstarOutput, va_outputs: AstarOutput) -> Metrics:
    """
    Calculate opt, exp, and hmean metrics for problem instances each with a single starting point

    Args:
        na_outputs (AstarOutput): outputs from Neural A*
        va_outputs (AstarOutput): outputs from vanilla A*

    Returns:
        Metrics: opt, exp, and hmean values
    """
    pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    pathlen_na = na_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    p_opt = (pathlen_astar == pathlen_na).mean()

    exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    exp_na = na_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

    h_mean = 2.0 / (1.0 / (p_opt + EPS) + 1.0 / (p_exp + EPS))

    return Metrics(p_opt, p_exp, h_mean)


def calc_metrics_from_multiple_results(
    na_outputs_list: Sequence[AstarOutput], va_outputs_list: Sequence[AstarOutput]
) -> Metrics:
    """
    Calculate opt, exp, and hmean metrics for problem instances each with multiple starting points

    Args:
        na_outputs (Sequence[AstarOutput]): Sequence of outputs from Neural A*
        va_outputs (Sequence[AstarOutput]): Sequence of outputs from vanilla A*

    Returns:
        Metrics: opt, exp, and hmean values
    """
    p_opt_list, p_exp_list = [], []
    for na_outputs, va_outputs in zip(na_outputs_list, va_outputs_list):
        pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        pathlen_na = na_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        p_opt_list.append(pathlen_astar == pathlen_na)

        exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        exp_na = na_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        p_exp_list.append(np.maximum((exp_astar - exp_na) / exp_astar, 0.0))
    p_opt = np.vstack(p_opt_list).mean(0)
    p_exp = np.vstack(p_exp_list).mean(0)
    h_mean = 2.0 / (1.0 / (p_opt + EPS) + 1.0 / (p_exp + EPS))

    return Metrics(p_opt.mean(), p_exp.mean(), h_mean.mean())


def visualize_results(
    map_designs: torch.tensor, planner_outputs: Union[AstarOutput, dict], scale: int = 1
) -> np.ndarray:
    """
    Create a visualization of search results

    Args:
        map_designs (torch.tensor): input maps
        planner_outputs (AstarOutput): outputs from planner
        scale (int): scale factor to enlarge output images. Default to 1.

    Returns:
        np.ndarray: visualized results
    """

    if type(planner_outputs) == dict:
        histories = planner_outputs["histories"]
        paths = planner_outputs["paths"]
    else:
        histories = planner_outputs.histories
        paths = planner_outputs.paths
    results = make_grid(map_designs).permute(1, 2, 0)
    h = make_grid(histories).permute(1, 2, 0)
    p = make_grid(paths).permute(1, 2, 0).float()
    results[h[..., 0] == 1] = torch.tensor([0.2, 0.8, 0])
    results[p[..., 0] == 1] = torch.tensor([1.0, 0.0, 0])

    results = ((results.numpy()) * 255.0).astype("uint8")

    if scale > 1:
        results = Image.fromarray(results).resize(
            [x * scale for x in results.shape[:2]], resample=Image.NEAREST
        )
        results = np.asarray(results)

    return results


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
