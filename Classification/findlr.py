import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.optim import lr_scheduler

import cv2
import pickle

import torchvision
import timm
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import datetime
import time
import tqdm
import copy
import random

import argparse

from models.efficientnet_b0 import get_efficientnet_b0
from models.resnet18 import get_resnet18
from utils.augmentations import *
from utils.dataloaders import *
from utils.training import *
from utils.metrics import *
from utils.visualise import *

from torch_lr_finder import LRFinder


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['efficientnet_b0', 'resnet18'],
                        default='resnet18', help='model')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--img-folder', type=str, help='path to folder with folders of image categories')
    parser.add_argument('--splits-folder', type=str, help='path to folder with train-val-test splits')
    parser.add_argument('--no-augment', action='store_true', help='use test augmentations for train')
    parser.add_argument('--rand-aug', action='store_true', help='apply RandAugment')
    parser.add_argument('--rand-aug-n', type=int, default=5, help='number of RandAugment transformations')
    parser.add_argument('--rand-aug-m', type=int, default=5, help='magnitude of RandAugment transformations')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2, help='number of dataloader workers')
    parser.add_argument('--seed', type=int, default=0, help='global seed')
    parser.add_argument('--det', action='store_true', help='use deterministic algorithms if available')
    parser.add_argument('--optimiser', type=str, choices=['Adam', 'SGD'],
                        default='Adam', help='optimiser')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='LRFinder min lr')
    parser.add_argument('--max-lr', type=float, default=1, help='LRFinder max lr')
    parser.add_argument('--iters', type=int, default=100, help='number of iterations to train LRFinder')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing parameter')
    parser.add_argument('--out-dir', type=str, default='./runs', help='path to save results')
    parser.add_argument('--no-cuda', action='store_true', help='do not use cuda even if available')

    return parser.parse_args()


def main(opt):
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.det:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True

    assert os.path.isdir(opt.img_folder), 'Provide valid path to image folder'
    assert os.path.isdir(opt.splits_folder), 'Provide valid path to splits folder'

    class_names = get_class_names(opt.img_folder)

    model = None
    if opt.model == 'efficientnet_b0':
        model = get_efficientnet_b0(num_classes=len(class_names))
    elif opt.model == 'resnet18':
        model = get_resnet18(num_classes=len(class_names))

    if opt.weights:
        assert os.path.exists(opt.weights), 'Provide valid path to model state dict'
        checkpoint = torch.load(opt.weights, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    transform = get_train_transforms()
    if opt.no_augment:
        transform = get_test_transforms()
    elif opt.rand_aug:
        transform = get_train_transforms_rand_aug(n_transformations=opt.rand_aug_n,
                                                  magnitude=opt.rand_aug_m)

    dataset = create_dataset(os.path.join(opt.splits_folder,
                                          'train_lst.txt'),
                             os.path.join(opt.img_folder, '..'),
                             transform)

    dataloader = create_dataloader(dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.workers,
                                   seed=opt.seed)

    optimiser = None
    if opt.optimiser == 'Adam':
        optimiser = optim.Adam(model.parameters())
    elif opt.optimiser == 'SGD':
        optimiser = optim.SGD(model.parameters(),
                              lr=opt.min_lr,
                              momentum=0.9)

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-findlr'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    device = torch.device('cpu')
    if not opt.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    print('Using {}'.format(device))

    assert 0.0 <= opt.label_smoothing <= 1.0
    criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)

    lr_finder = LRFinder(model, optimiser, criterion, device)
    lr_finder.range_test(dataloader, start_lr=opt.min_lr, end_lr=opt.max_lr, num_iter=opt.iters)

    lr_finder.plot()
    f, ax = plt.subplots(figsize=(8, 6))
    lr_finder.plot(ax=ax)
    ax.set_title('{} from {} to {}'.format(opt.optimiser, opt.min_lr, opt.max_lr))
    f.savefig(os.path.join(dir_to_save, 'lr-search-{}-{}:{}.png'.format(opt.optimiser,
                                                                        opt.min_lr,
                                                                        opt.max_lr)))

    print('Results saved to {} folder'.format(dir_to_save))


if __name__ == '__main__':
    options = parse_options()
    main(options)
