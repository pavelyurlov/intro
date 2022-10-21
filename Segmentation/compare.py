import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import onnxruntime

import pickle

import torchvision
from torchvision import models
import segmentation_models_pytorch as smp

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
from utils.metrics import *
from utils.visualise import *
from utils.onnx import *


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lraspp', 'deeplabv3+'],
                        default='deeplabv3+', help='model')
    parser.add_argument('--classes-equal', action='store_true', help='do not add class weights for cross entropy')
    parser.add_argument('--dice', action='store_true', help='add Dice Loss to criterion')
    parser.add_argument('--n-classes', type=int, default=5, help='number of classes')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--weights-onnx', type=str, default=None, help='path to ONNX model')
    parser.add_argument('--data-folder', type=str, help='path to folder with data')
    parser.add_argument('--subsample', type=str, choices=['train', 'val'], default='val',
                        help='sample to compare on')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n-workers', type=int, default=2, help='number of dataloader workers')
    parser.add_argument('--no-visual', action='store_true', help='do not draw predictions')
    parser.add_argument('--draw-batch', action='store_true', help='plot only one batch')
    parser.add_argument('--out-dir', type=str, default='./runs', help='path to save results')

    return parser.parse_args()


def main(opt):
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
    else:
        print('No weights provided for validation')

    assert os.path.exists(opt.weights_onnx), 'Provide valid path to ONNX model'
    ort_session = onnxruntime.InferenceSession(opt.weights_onnx)

    transform = get_test_transforms()

    dataset = create_dataset(os.path.join(opt.data_folder,
                                          'lists/{}_lst.txt'.format(opt.subsample)),
                             opt.data_folder,
                             transform)

    dataloader = create_dataloader(dataset,
                                   batch_size=opt.batch_size,
                                   num_workers=opt.n_workers)

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-compare'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    class_weights = None
    if not opt.classes_equal:
        class_weights = get_class_weights(
            os.path.join(opt.data_folder, 'lists/train_lst.txt'),
            opt.data_folder,
            opt.batch_size,
            opt.n_workers,
            opt.n_classes
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if opt.dice:
        criterion = DiceLossPlusCE(weight=class_weights)

    metrics = compare(model,
                      ort_session,
                      dataloader,
                      criterion,
                      draw_predictions=not opt.no_visual,
                      draw_all=not opt.draw_batch,
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
