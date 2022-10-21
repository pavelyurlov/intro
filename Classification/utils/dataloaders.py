import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

import cv2


class MyCaltech256Dataset(Dataset):
    def __init__(self, subsample, root_dir, transform=None):
        self.subsample = pd.read_csv(subsample, sep=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.subsample)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.subsample.iloc[idx, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        label = self.subsample.iloc[idx, 1]

        return img, label


def create_dataset(split_path, root_dir, transform):
    """
    :param split_path: path to file with lists of image files
    :param root_dir: path to image folder
    :param transform:
    :return: instance of MyCaltech256Dataset class
    """
    return MyCaltech256Dataset(
        subsample=split_path,
        root_dir=root_dir,
        transform=transform
    )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2, seed=None):
    """
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param seed:
    :return:
    """
    my_worker = None
    my_generator = None
    if seed:
        my_worker = _seed_worker
        my_generator = torch.Generator()
        my_generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=my_worker,
        generator=my_generator
    )


def get_class_names(classes_folder):
    return sorted(os.listdir(classes_folder))
