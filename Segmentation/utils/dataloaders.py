import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

import cv2

from utils.augmentations import transform_to_tensor


COLOURS = np.array([
    [150,  95,  60],  # 0 vista
    [195, 195, 195],  # 1 building
    [ 88,  88,  88],  # 2 road
    [  0, 120,   0],  # 3 tree
    [255, 255,   0],  # 4 cable
], dtype=np.uint8)
COLOURS_GREEN = tuple(COLOURS[:, 1])


def get_class_index_mask(mask, colours=COLOURS_GREEN):
    for i, colour in enumerate(colours):
        mask = torch.where(mask == colour, i, mask)
    return mask.long()


class MySegmentationDataset(Dataset):
    def __init__(self, idxs_path, root_dir, transform=None):
        self.idxs = list(pd.read_csv(idxs_path, header=None)[0])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                'images/Img_{}.jpeg'.format(self.idxs[idx]))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root_dir,
                                 'masks/Mask_{}.png'.format(self.idxs[idx]))
        mask = cv2.imread(mask_path)[:, :, 1]

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        mask = get_class_index_mask(mask)

        return img, mask


def create_dataset(idxs_path, root_dir, transform):
    """
    :param idxs_path: path to file with lists of image files
    :param root_dir: path to image folder
    :param transform:
    :return: instance of MySegmentationDataset class
    """
    return MySegmentationDataset(
        idxs_path=idxs_path,
        root_dir=root_dir,
        transform=transform
    )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=2, seed=None):
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


def get_class_weights(train_idxs, root_dir, batch_size, num_workers, num_classes):
    dataset = create_dataset(train_idxs, root_dir, transform_to_tensor())
    dataloader = create_dataloader(dataset, batch_size, num_workers=num_workers)

    classes_pixels = torch.zeros(num_classes)
    for _, masks in dataloader:
        for i in range(len(classes_pixels)):
            classes_pixels[i] += torch.sum(masks == i)

    classes_loss_weights = torch.sum(classes_pixels) / classes_pixels
    return classes_loss_weights
