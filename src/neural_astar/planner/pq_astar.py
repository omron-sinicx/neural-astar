import numpy as np
from pqdict import pqdict


def get_neighbor_indices(idx, H, W, mechanism="moore"):
    neighbor_indices = []
    if idx % W - 1 >= 0:
        neighbor_indices.append(idx - 1)
    if idx % W + 1 < W:
        neighbor_indices.append(idx + 1)
    if idx // W - 1 >= 0:
        neighbor_indices.append(idx - W)
    if idx // W + 1 < H:
        neighbor_indices.append(idx + W)
    if mechanism == "moore":
        if (idx % W - 1 >= 0) & (idx // W - 1 >= 0):
            neighbor_indices.append(idx - W - 1)
        if (idx % W + 1 < W) & (idx // W - 1 >= 0):
            neighbor_indices.append(idx - W + 1)
        if (idx % W - 1 >= 0) & (idx // W + 1 < H):
            neighbor_indices.append(idx + W - 1)
        if (idx % W + 1 < W) & (idx // W + 1 < H):
            neighbor_indices.append(idx + W + 1)
    return np.array(neighbor_indices)


def get_moore_heuristics(idx, goal_idx, W):
    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    dxdy = np.abs(loc - goal_loc)
    h = dxdy.sum() - dxdy.min()
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return h + 0.001 * euc


def get_history(close_list, W):
    return np.array([[idx % W, idx // W] for idx in close_list.keys()])


def backtrack(parent_list, goal_idx, W):
    current_idx = goal_idx
    path = []
    while current_idx != None:
        path.append([current_idx % W, current_idx // W])
        current_idx = parent_list[current_idx]
    return np.array(path)


def pq_astar(map_design, pred_cost, start_map, goal_map, mechanism="moore", weight=0.5):
    H, W = map_design.shape
    start_idx = np.argwhere(start_map.flatten()).item()
    goal_idx = np.argwhere(goal_map.flatten()).item()
    map_design_vct = map_design.flatten()
    pred_cost_vct = pred_cost.flatten()
    open_list = pqdict()
    close_list = pqdict()
    open_list.additem(start_idx, 0)
    parent_list = dict()
    parent_list[start_idx] = None

    num_steps = 0
    while goal_idx not in close_list:
        if len(open_list) == 0:
            print("goal not found")
            return None
        num_steps += 1
        v_idx, v_cost = open_list.popitem()
        close_list.additem(v_idx, v_cost)
        for n_idx in get_neighbor_indices(v_idx, H, W, mechanism):
            if (
                (map_design_vct[n_idx] == 1)
                & (n_idx not in open_list)
                & (n_idx not in close_list)
            ):
                # fnew = v_cost + pred_cost_vct[n_idx]+ get_moore_heuristics(n_idx, goal_idx, W) - get_moore_heuristics(v_idx, goal_idx, W)
                fnew = (
                    v_cost
                    - (1 - weight) * get_moore_heuristics(v_idx, goal_idx, W)
                    + weight * pred_cost_vct[n_idx]
                    + (1 - weight) * get_moore_heuristics(n_idx, goal_idx, W)
                )
                open_list.additem(n_idx, fnew)
                parent_list[n_idx] = v_idx

    path = backtrack(parent_list, goal_idx, W)
    history = get_history(close_list, W)
    return history, path
