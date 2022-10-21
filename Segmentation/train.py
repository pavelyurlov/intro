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


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['lraspp', 'deeplabv3+'],
                        default='deeplabv3+', help='model')
    parser.add_argument('--n-classes', type=int, default=5, help='number of classes')
    parser.add_argument('--classes-equal', action='store_true', help='do not add class weights for cross entropy')
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
    parser.add_argument('--lr', type=float, default=None, help='specify initial lr')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cyclic', 'one-cycle', None],
                        default=None, help='specify lr scheduler type')
    parser.add_argument('--scheduler-step', type=int, default=5, help='lr scheduler step')
    parser.add_argument('--scheduler-gamma', type=float, default=0.9, help='lr scheduler gamma')
    parser.add_argument('--min-lr', type=float, default=1e-4, help='cyclic scheduler min lr')
    parser.add_argument('--max-lr', type=float, default=1e-1, help='cyclic or one cycle scheduler max lr')
    parser.add_argument('--step-size-up', type=int, default=5, help='cyclic scheduler number of increasing epochs')
    parser.add_argument('--step-size-down', type=int, default=5, help='cyclic scheduler number of decreasing epochs')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--no-tensorboard', action='store_true', help='plot metrics and images to tensorboard')
    parser.add_argument('--log-interval', type=int, default=10, help='tensorboard log interval during training')
    parser.add_argument('--no-visual', action='store_true', help='do not draw predictions on val')
    parser.add_argument('--no-verbose', action='store_true', help='do not print out short summary after each epoch')
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

    transform = {
        'train': get_train_transforms(),
        'val': get_test_transforms()
    }
    if opt.no_augment:
        transform['train'] = get_test_transforms()

    datasets = {
        k: create_dataset(os.path.join(opt.data_folder,
                                       'lists/{}_lst.txt'.format(k)),
                          opt.data_folder,
                          transform[k])
        for k in transform.keys()
    }

    dataloaders = {
        k: create_dataloader(datasets[k],
                             batch_size=opt.batch_size,
                             shuffle=True,
                             num_workers=opt.n_workers,
                             seed=opt.seed)
        for k in datasets.keys()
    }

    optimiser = None
    if opt.optimiser == 'Adam':
        optimiser = optim.Adam(model.parameters(),
                               lr=1e-3 if opt.lr is None else opt.lr)
    elif opt.optimiser == 'SGD':
        optimiser = optim.SGD(model.parameters(),
                              lr=1e-2 if opt.lr is None else opt.lr,
                              momentum=0.9)

    scheduler = None
    if opt.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimiser,
                                        step_size=opt.scheduler_step,
                                        gamma=opt.scheduler_gamma)
    elif opt.scheduler == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimiser,
                                          base_lr=opt.min_lr,
                                          max_lr=opt.max_lr,
                                          step_size_up=opt.step_size_up * len(dataloaders['train']),
                                          step_size_down=opt.step_size_down * len(dataloaders['train']))
    elif opt.scheduler == 'one-cycle':
        scheduler = lr_scheduler.OneCycleLR(optimiser,
                                            max_lr=opt.max_lr,
                                            epochs=opt.epochs,
                                            steps_per_epoch=len(dataloaders['train']))

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-train'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    summary_writers = {}
    if not opt.no_tensorboard:
        summary_writers['train'] = summary.create_file_writer(os.path.join(dir_to_save,
                                                                           'tensorboard_logs/train'))
        summary_writers['val'] = summary.create_file_writer(os.path.join(dir_to_save,
                                                                         'tensorboard_logs/val'))

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

    history = train(model=model,
                    device=device,
                    criterion=criterion,
                    dataloaders=dataloaders,
                    optimiser=optimiser,
                    scheduler=scheduler,
                    scheduler_type=opt.scheduler,
                    num_epochs=opt.epochs,
                    verbose=not opt.no_verbose,
                    use_tensorboard=not opt.no_tensorboard,
                    log_interval=opt.log_interval,
                    summary_writers=summary_writers,
                    draw_predictions=not opt.no_visual,
                    dir_to_save=dir_to_save)


if __name__ == '__main__':
    options = parse_options()
    main(options)
