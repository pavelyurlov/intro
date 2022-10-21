import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.optim import lr_scheduler

import pickle

import torchvision
from torchvision import models
import segmentation_models_pytorch as smp

import albumentations as A
import cv2

import matplotlib.pyplot as plt
import datetime
import time
import tqdm
import copy
import random

import argparse

from models.lraspp_mobilenet_v3_large import get_lraspp_mobilenet_v3_large
from models.deeplabv3 import get_deeplab_v3_plus
from utils.augmentations import *
from utils.dataloaders import *
from utils.training import *
from utils.metrics import *
from utils.visualise import *

from torch_lr_finder import LRFinder


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['lraspp', 'deeplabv3+'],
                        default='deeplabv3+', help='model')
    parser.add_argument('--n-classes', type=int, default=5, help='number of classes')
    parser.add_argument('--classes-equal', action='store_true', help='do not add class weights for BCE')
    parser.add_argument('--dice', action='store_true', help='add Dice Loss to criterion')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--data-folder', type=str, help='path to folder with data')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n-workers', type=int, default=2, help='number of dataloader workers')
    parser.add_argument('--no-augment', action='store_true', help='use test augmentations for train')
    parser.add_argument('--seed', type=int, default=0, help='global seed')
    parser.add_argument('--det', action='store_true', help='use deterministic algorithms if available')
    parser.add_argument('--optimiser', type=str, choices=['Adam', 'SGD'],
                        default='Adam', help='optimiser')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='LRFinder min lr')
    parser.add_argument('--max-lr', type=float, default=1, help='LRFinder max lr')
    parser.add_argument('--iters', type=int, default=100, help='number of iterations to train LRFinder')
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

    assert os.path.isdir(opt.data_folder), 'Provide valid path to dataset folder'

    model = None
    if opt.model == 'lraspp':
        model = get_lraspp_mobilenet_v3_large(num_classes=opt.n_classes)
    elif opt.model == 'deeplabv3+':
        model = get_deeplab_v3_plus(num_classes=opt.n_classes)

    if opt.weights:
        assert os.path.exists(opt.weights), 'Provide valid path to model state dict'
        checkpoint = torch.load(opt.weights, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    transform = get_train_transforms()
    if opt.no_augment:
        transform['train'] = get_test_transforms()

    dataset = create_dataset(os.path.join(opt.data_folder,
                                          'lists/train_lst.txt'),
                             opt.data_folder,
                             transform)

    dataloader = create_dataloader(dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.n_workers,
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

    class_weights = None
    if not opt.classes_equal:
        class_weights = get_class_weights(
            os.path.join(opt.data_folder, 'lists/train_lst.txt'),
            opt.data_folder,
            opt.batch_size,
            opt.n_workers,
            opt.n_classes
        ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if opt.dice:
        criterion = DiceLossPlusCE(weight=class_weights)

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
