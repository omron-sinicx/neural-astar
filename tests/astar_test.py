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
    assert torch.allclose(output.histories, output_pq.histories)
    assert torch.allclose(output.paths, output_pq.paths)


def test_astar_on_rectangle(setup):
    from neural_astar.planner import NeuralAstar

    map_designs, start_maps, goal_maps = setup
    map_designs = torch.concat((map_designs, map_designs), -1)
    start_maps = torch.concat((start_maps, torch.zeros_like(start_maps)), -1)
    goal_maps = torch.concat((torch.zeros_like(goal_maps), goal_maps), -1)
    planner = NeuralAstar()
    output = planner(map_designs, start_maps, goal_maps)
