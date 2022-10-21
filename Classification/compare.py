import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import onnxruntime

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
from utils.metrics import *
from utils.visualise import *
from utils.onnx import *


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['efficientnet_b0', 'resnet18'],
                        default='resnet18', help='model')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--weights-onnx', type=str, default=None, help='path to ONNX model')
    parser.add_argument('--img-folder', type=str, help='path to folder with folders of image categories')
    parser.add_argument('--splits-folder', type=str, help='path to folder with train-val-test splits')
    parser.add_argument('--subsample', type=str, choices=['train', 'val', 'test'], default='val',
                        help='sample to compare on')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2, help='number of dataloader workers')
    parser.add_argument('--no-visual', action='store_true', help='do not draw predictions')
    parser.add_argument('--draw-batch', action='store_true', help='plot only one batch')
    parser.add_argument('--out-dir', type=str, default='./runs', help='path to save results')

    return parser.parse_args()


def main(opt):
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
    else:
        print('No weights provided for validation')

    assert os.path.exists(opt.weights_onnx), 'Provide valid path to ONNX model'
    ort_session = onnxruntime.InferenceSession(opt.weights_onnx)

    transform = get_test_transforms()

    dataset = create_dataset(os.path.join(opt.splits_folder,
                                          '{}_lst.txt'.format(opt.subsample)),
                             os.path.join(opt.img_folder, '..'),
                             transform)

    dataloader = create_dataloader(dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.workers)

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-compare'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    metrics = compare(model,
                      ort_session,
                      dataloader,
                      draw_predictions=not opt.no_visual,
                      draw_all=not opt.draw_batch,
                      class_names=class_names,
                      path_to_save=dir_to_save)

    with open(os.path.join(dir_to_save, 'metrics.pickle'), 'wb') as f:
        pickle.dump(metrics, f)

    print('Results saved to {} folder.'.format(dir_to_save))
    print('Summary on {}:'.format(opt.subsample))
    for k, v in metrics.items():
        print(k)
        for kk, vv in v.items():
            print('    {} {:.4f}'.format(kk, vv))


if __name__ == '__main__':
    options = parse_options()
    main(options)
