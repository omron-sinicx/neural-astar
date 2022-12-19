"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from neural_astar.planner.astar import VanillaAstar
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torchvision.utils import make_grid


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
        # va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
        # pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        # pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
        # p_opt = (pathlen_astar == pathlen_model).mean()

        # exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        # exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
        # p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

        # h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

        self.log("metrics/val_loss", loss)
        # self.log("metrics/p_opt", p_opt)
        # self.log("metrics/p_exp", p_exp)
        # self.log("metrics/h_mean", h_mean)

        return loss


def visualize_results(
    map_designs: torch.tensor, planner_outputs: AstarOutput, scale: int = 1
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
