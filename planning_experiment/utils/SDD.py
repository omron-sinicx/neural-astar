"""Dataset utils for SDD
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torch


class SDD(Dataset):
    def __init__(self,
                 dirname,
                 is_train=True,
                 test_scene='gates',
                 load_hardness=False,
                 load_label=False):
        self.data = sorted(glob('%s/*/*/*.npz' % dirname))
        if (is_train):
            self.data = [x for x in self.data if test_scene not in x]
        else:
            self.data = [x for x in self.data if test_scene in x]
        self.load_hardness = load_hardness
        self.load_label = load_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        s = np.load(self.data[idx])
        if (self.load_hardness):
            return s['length_ratio']
        if (self.load_label):
            return s['label']

        sample = {
            'image': s['image'],
            'traj_image': s['traj_image'],
            'start_image': s['start_image'],
            'goal_image': s['goal_image'],
            'label': s['label'],
            'length_ratio': s['length_ratio']
            # 'traj': s['traj'],
        }

        sample = CustomToTensor()(sample)

        return sample


class CustomToTensor(object):
    def __call__(self, sample):
        image, traj_image, start_image, goal_image, label, length_ratio = \
        sample['image'], sample['traj_image'], sample['start_image'], sample['goal_image'], sample['label'], sample['length_ratio']
        image = torch.from_numpy(np.array(image)).type(torch.float32).permute(
            2, 0, 1) / 255.
        traj_image = torch.from_numpy(traj_image).type(torch.float32)
        start_image = torch.from_numpy(start_image).type(torch.float32)
        goal_image = torch.from_numpy(goal_image).type(torch.float32)
        sample = {
            'image': image,
            'traj_image': traj_image.unsqueeze(0),
            'start_image': start_image.unsqueeze(0),
            'goal_image': goal_image.unsqueeze(0),
            'label': label,
            'length_ratio': length_ratio,
        }

        return sample
