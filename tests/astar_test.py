import pytest
import torch
import torch.nn as nn


@pytest.fixture
def setup():
    map_designs = torch.ones((8, 1, 32, 32))
    start_maps = torch.zeros((8, 1, 32, 32))
    start_maps[:, :, 0, 0] = 1
    goal_maps = torch.zeros((8, 1, 32, 32))
    goal_maps[:, :, -1, -1] = 1

    return map_designs, start_maps, goal_maps


def test_neural_astar(setup):
    from neural_astar.planner import NeuralAstar

    map_designs, start_maps, goal_maps = setup
    planner = NeuralAstar()
    output = planner(map_designs, start_maps, goal_maps)


def test_vanilla_astar(setup):
    from neural_astar.planner import VanillaAstar

    map_designs, start_maps, goal_maps = setup
    planner = VanillaAstar()
    output = planner(map_designs, start_maps, goal_maps)


def test_pq_astar(setup):
    from neural_astar.planner import NeuralAstar

    map_designs, start_maps, goal_maps = setup
    planner = NeuralAstar(use_differentiable_astar=False)
    output = planner(map_designs, start_maps, goal_maps)
