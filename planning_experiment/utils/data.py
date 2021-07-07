"""Dataset utils
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
Part of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

from __future__ import print_function

import numpy as np
import torch
import torch.utils.data as data

from .mechanism import Mechanism

TEST_RANDOM_SEED = 2020
NUM_POINTS_PER_MAP = 5


def create_dataloader(datafile: str,
                      dataset_type: str,
                      batch_size: int,
                      shuffle: bool = False):
    """
    Creates a maze DataLoader.
    Args:
      datafile (str): Path to the dataset
      dataset_type (str): One of "train", "valid", or "test"
      batch_size (int): The batch size
      shuffle (bool): Whether to shuffle the data
    """
    dataset = MazeDataset(datafile, dataset_type)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=0)


class MazeDataset(data.Dataset):
    def __init__(self, filename: str, dataset_type: str):
        """
        Args:
          filename (str): Dataset filename (must be .npz format).
          dataset_type (str): One of "train", "valid", or "test".
        """
        assert filename.endswith("npz")  # Must be .npz format
        self.filename = filename
        self.dataset_type = dataset_type  # train, valid, test

        self.mazes, self.goal_maps, self.opt_policies, self.opt_dists = self._process(
            filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(self, filename: str):
        """
        Data format: list, [train data, test data]
        """
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]
            mazes = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]
            opt_dists = f["arr_" + str(idx + 3)]

        # Set proper datatypes
        mazes = mazes.astype(np.float32)
        goal_maps = goal_maps.astype(np.float32)
        opt_policies = opt_policies.astype(np.float32)
        opt_dists = opt_dists.astype(np.float32)

        # Print number of samples
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(mazes.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(mazes.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(mazes.shape[0]))
        print("\tSize: {}x{}".format(mazes.shape[1], mazes.shape[2]))
        return mazes, goal_maps, opt_policies, opt_dists

    def __getitem__(self, index: int):
        maze = self.mazes[index]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]

        return maze, goal_map, opt_policy, opt_dist

    def __len__(self):
        return self.mazes.shape[0]


def get_opt_trajs(start_maps: np.ndarray, goal_maps: np.ndarray,
                  opt_policies: np.ndarray, mechanism: Mechanism):

    opt_trajs = np.zeros_like(start_maps)
    opt_policies = opt_policies.transpose((0, 2, 3, 4, 1))

    for i in range(len(opt_trajs)):
        current_loc = tuple(np.array(np.nonzero(start_maps[i])).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_maps[i])).squeeze())

        while goal_loc != current_loc:
            opt_trajs[i][current_loc] = 1.0
            next_loc = mechanism.next_loc(current_loc,
                                          opt_policies[i][current_loc])
            assert (
                opt_trajs[i][next_loc] == 0.0
            ), "Revisiting the same position while following the optimal policy"
            current_loc = next_loc

        opt_trajs[i][current_loc] = 1.0

    return opt_trajs


def get_hard_medium_easy_masks(opt_dists_CPU: np.ndarray,
                               reduce_dim: bool = True,
                               num_points_per_map: int = 5):
    # make sure the selected nodes are random but fixed
    np.random.seed(TEST_RANDOM_SEED)
    # impossible distance
    wall_dist = np.min(opt_dists_CPU)

    n_samples = opt_dists_CPU.shape[0]
    od_vct = opt_dists_CPU.reshape(n_samples, -1)
    od_nan = od_vct.copy()
    od_nan[od_nan == wall_dist] = np.nan
    od_min = np.nanmin(od_nan, axis=1, keepdims=True)
    thes = od_min.dot(np.array([[1.0, 0.85, 0.70, 0.55]])).astype("int").T
    thes = thes.reshape(4, n_samples, 1, 1, 1)

    masks_list = []
    for i in range(3):
        binmaps = ((thes[i] <= opt_dists_CPU) &
                   (opt_dists_CPU < thes[i + 1])) * 1.0
        binmaps = np.repeat(binmaps, num_points_per_map, 0)
        masks = _sample_onehot(binmaps)
        masks = masks.reshape(n_samples, num_points_per_map,
                              *opt_dists_CPU.shape[1:])
        if reduce_dim:
            masks = masks.max(axis=1)
        masks_list.append(masks.astype(bool))
    return masks_list


def _sample_onehot(binmaps):
    n_samples = len(binmaps)
    binmaps_n = binmaps * np.random.rand(*binmaps.shape)

    binmaps_vct = binmaps_n.reshape(n_samples, -1)
    ind = binmaps_vct.argmax(axis=-1)
    onehots = np.zeros_like(binmaps_vct)
    onehots[range(n_samples), ind] = 1
    onehots = onehots.reshape(binmaps_n.shape).astype("bool")

    return onehots
