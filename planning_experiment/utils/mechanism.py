"""Kernel mechanisms implementations.
Most of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

from __future__ import print_function

import abc

import gin
import numpy as np
import torch


def _get_expanded_locs(goal_maps):
    num_samples, size = goal_maps.shape[0], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

    return loc_expand, goal_loc_expand


class Mechanism(abc.ABC):
    """Base class for maze transition mechanisms."""
    def __init__(self, num_actions, num_orient, action_to_move=None):
        self.num_actions = num_actions
        self.num_orient = num_orient
        self.action_to_move = action_to_move

    def next_loc(self, current_loc, one_hot_action):
        move = self.action_to_move[np.argmax(one_hot_action)]
        return tuple(np.add(current_loc, move))

    @abc.abstractmethod
    def neighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes next states for each action."""

    @abc.abstractmethod
    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        """Computes previous states for each action."""

    @abc.abstractmethod
    def print_policy(self, maze, goal, policy):
        """Prints the given policy."""

    @abc.abstractstaticmethod
    def get_heuristic(goal_maps):
        """Compute heuristic function given current mechanism"""


@gin.configurable("news")
class NorthEastWestSouth(Mechanism):
    """
    In NEWS, the agent can move North, East, West, or South.
    """
    def __init__(self):
        action_to_move = [(0, -1, 0), (0, 0, +1), (0, 0, -1), (0, +1, 0)]
        super().__init__(num_actions=4,
                         num_orient=1,
                         action_to_move=action_to_move)

    def _north(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (maze[p_y - 1][p_x] != 0.0):
            return p_orient, p_y - 1, p_x
        return p_orient, p_y, p_x

    def _east(self, maze, p_orient, p_y, p_x):
        if (p_x < (maze.shape[1] - 1)) and (maze[p_y][p_x + 1] != 0.0):
            return p_orient, p_y, p_x + 1
        return p_orient, p_y, p_x

    def _west(self, maze, p_orient, p_y, p_x):
        if (p_x > 0) and (maze[p_y][p_x - 1] != 0.0):
            return p_orient, p_y, p_x - 1
        return p_orient, p_y, p_x

    def _south(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (maze[p_y + 1][p_x] != 0.0):
            return p_orient, p_y + 1, p_x
        return p_orient, p_y, p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._north(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._south(maze, p_orient, p_y, p_x),
        ]

    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._south(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._north(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        action2str = ["↑", "→", "←", "↓"]
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print(u"\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")
            print("")

    @staticmethod
    def get_neighbor_filter():
        neighbor_filter = torch.zeros(1, 1, 3, 3)
        neighbor_filter[0, 0, 0, 1] = 1
        neighbor_filter[0, 0, 1, 0] = 1
        neighbor_filter[0, 0, 1, 2] = 1
        neighbor_filter[0, 0, 2, 1] = 1

        return neighbor_filter

    @staticmethod
    def get_heuristic(goal_maps, tb_factor=0.001):
        loc_expand, goal_loc_expand = _get_expanded_locs(goal_maps)
        # manhattan distance
        h = torch.abs(loc_expand - goal_loc_expand).sum(dim=1)
        euc = torch.sqrt(((loc_expand - goal_loc_expand)**2).sum(1))
        h = (h + tb_factor * euc).reshape_as(goal_maps)

        return h


@gin.configurable("moore")
class Moore(Mechanism):
    """
    In Moore, the agent can move to any of the eight cells in its Moore
    neighborhood.
    """

    _ACTION_TO_MOVE = [
        (0, -1, 0),
        (0, 0, +1),
        (0, 0, -1),
        (0, +1, 0),
        (0, -1, +1),
        (0, -1, -1),
        (0, +1, +1),
        (0, +1, -1),
    ]

    def __init__(self):
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        super().__init__(num_actions=8,
                         num_orient=1,
                         action_to_move=action_to_move)

    def _north(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (maze[p_y - 1][p_x] != 0.0):
            return p_orient, p_y - 1, p_x
        return p_orient, p_y, p_x

    def _east(self, maze, p_orient, p_y, p_x):
        if (p_x < (maze.shape[1] - 1)) and (maze[p_y][p_x + 1] != 0.0):
            return p_orient, p_y, p_x + 1
        return p_orient, p_y, p_x

    def _west(self, maze, p_orient, p_y, p_x):
        if (p_x > 0) and (maze[p_y][p_x - 1] != 0.0):
            return p_orient, p_y, p_x - 1
        return p_orient, p_y, p_x

    def _south(self, maze, p_orient, p_y, p_x):
        if (p_y < (maze.shape[0] - 1)) and (maze[p_y + 1][p_x] != 0.0):
            return p_orient, p_y + 1, p_x
        return p_orient, p_y, p_x

    def _northeast(self, maze, p_orient, p_y, p_x):
        if ((p_y > 0) and (p_x < (maze.shape[1] - 1))
                and (maze[p_y - 1][p_x + 1] != 0.0)):
            return p_orient, p_y - 1, p_x + 1
        return p_orient, p_y, p_x

    def _northwest(self, maze, p_orient, p_y, p_x):
        if (p_y > 0) and (p_x > 0) and (maze[p_y - 1][p_x - 1] != 0.0):
            return p_orient, p_y - 1, p_x - 1
        return p_orient, p_y, p_x

    def _southeast(self, maze, p_orient, p_y, p_x):
        if ((p_y < (maze.shape[0] - 1)) and (p_x < (maze.shape[1] - 1))
                and (maze[p_y + 1][p_x + 1] != 0.0)):
            return p_orient, p_y + 1, p_x + 1
        return p_orient, p_y, p_x

    def _southwest(self, maze, p_orient, p_y, p_x):
        if ((p_y < (maze.shape[0] - 1)) and (p_x > 0)
                and (maze[p_y + 1][p_x - 1] != 0.0)):
            return p_orient, p_y + 1, p_x - 1
        return p_orient, p_y, p_x

    def neighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._north(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._south(maze, p_orient, p_y, p_x),
            self._northeast(maze, p_orient, p_y, p_x),
            self._northwest(maze, p_orient, p_y, p_x),
            self._southeast(maze, p_orient, p_y, p_x),
            self._southwest(maze, p_orient, p_y, p_x),
        ]

    def invneighbors_func(self, maze, p_orient, p_y, p_x):
        return [
            self._south(maze, p_orient, p_y, p_x),
            self._west(maze, p_orient, p_y, p_x),
            self._east(maze, p_orient, p_y, p_x),
            self._north(maze, p_orient, p_y, p_x),
            self._southwest(maze, p_orient, p_y, p_x),
            self._southeast(maze, p_orient, p_y, p_x),
            self._northwest(maze, p_orient, p_y, p_x),
            self._northeast(maze, p_orient, p_y, p_x),
        ]

    def print_policy(self, maze, goal, policy):
        action2str = ["↑", "→", "←", "↓", "↗", "↖", "↘", "↙"]
        for o in range(self.num_orient):
            for y in range(policy.shape[1]):
                for x in range(policy.shape[2]):
                    if (o, y, x) == goal:
                        print("!", end="")
                    elif maze[y][x] == 0.0:
                        print(u"\u2588", end="")
                    else:
                        a = policy[o][y][x]
                        print(action2str[a], end="")
                print("")
            print("")

    @staticmethod
    def get_neighbor_filter():
        neighbor_filter = torch.ones(1, 1, 3, 3)
        neighbor_filter[0, 0, 1, 1] = 0

        return neighbor_filter

    @staticmethod
    def get_heuristic(goal_maps, tb_factor=0.001):
        loc_expand, goal_loc_expand = _get_expanded_locs(goal_maps)
        # chebyshev distance
        dxdy = torch.abs(loc_expand - goal_loc_expand)
        h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]
        euc = torch.sqrt(((loc_expand - goal_loc_expand)**2).sum(1))
        h = (h + tb_factor * euc).reshape_as(goal_maps)

        return h


def get_mechanism(mechanism, verbose=False):
    if ("news" in mechanism):
        if verbose:
            print("Using NEWS Drive")
        return NorthEastWestSouth()
    elif ("moore" in mechanism):
        if verbose:
            print("Using Moore Drive")
        return Moore()
    else:
        raise ValueError("Unsupported mechanism: %s" % mechanism)
