"""Neural A* search
Author: Ryo Yonetani
Affiliation: OSX
"""

from typing import Optional
import torch
import torch.nn as nn

from . import encoder
from .differentiable_astar import DifferentiableAstar, AstarOutput


class NeuralAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
        Tmax: float = 0.25,
        encoder_input: str = "m+",
        encoder_arch: float = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: Optional[float] = None,
    ):
        """
        Neural A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            Tmax (float, optional): how much of the map the model explores during training. Defaults to 0.25.
            encoder_input (str, optional): input format. Set "m+" to use the concatenation of map_design and (start_map + goal_map). Set "m" to use map_design only. Defaults to "m+".
            encoder_backbone (str, optional): encoder architecture. Defaults to "vgg16_bn".
            encoder_depth (int, optional): depth of the encoder. Defaults to 4.
            learn_obstacles (bool, optional): if the obstacle is invisible to the model. Defaults to False.
            const (Optional[float], optional): learnable weight to be multiplied for h(v). Defaults to None.

        Examples:
            >>> planner = NeuralAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths
        """
        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        self.encoder_input = encoder_input
        encoder_arch = getattr(encoder, encoder_arch)
        self.encoder = encoder_arch(len(self.encoder_input), encoder_depth,
                                    const)
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")

    def forward(self,
                map_designs: torch.tensor,
                start_maps: torch.tensor,
                goal_maps: torch.tensor,
                store_intermediate_results: bool = False) -> AstarOutput:
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        pred_cost_maps = self.encoder(inputs)
        obstacles_maps = map_designs if not self.learn_obstacles else torch.ones_like(
            map_designs)

        astar_outputs = self.astar(pred_cost_maps, start_maps, goal_maps,
                                   obstacles_maps, store_intermediate_results)

        return astar_outputs


class VanillaAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
    ):
        """
        Vanilla A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.

        Examples:
            >>> planner = VanillaAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths
        """

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=1.0,
        )

    def forward(self,
                map_designs: torch.tensor,
                start_maps: torch.tensor,
                goal_maps: torch.tensor,
                store_intermediate_results: bool = False) -> AstarOutput:
        obstacles_maps = map_designs

        astar_outputs = self.astar(map_designs, start_maps, goal_maps,
                                   obstacles_maps, store_intermediate_results)

        return astar_outputs