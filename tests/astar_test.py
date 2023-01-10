import pytest
import torch


@pytest.fixture
def setup():
    map_designs = torch.ones((8, 1, 64, 64))
    map_designs[:, :, 24:48, 24:48] = 0
    start_maps = torch.zeros((8, 1, 64, 64))
    start_maps[:, :, 0, 0] = 1
    goal_maps = torch.zeros((8, 1, 64, 64))
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
    from neural_astar.planner import VanillaAstar

    map_designs, start_maps, goal_maps = setup
    planner = VanillaAstar(use_differentiable_astar=True)
    output = planner(map_designs, start_maps, goal_maps)
    planner_pq = VanillaAstar(use_differentiable_astar=False)
    output_pq = planner_pq(map_designs, start_maps, goal_maps)
    torch.allclose(output.histories, output_pq.histories)
    torch.allclose(output.paths, output_pq.paths)
