import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import torch.onnx
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
from utils.dataloaders import *
from utils.onnx import *


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['efficientnet_b0', 'resnet18'],
                        default='resnet18', help='model')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--img-size', type=int, default=224, help='image width and height')
    parser.add_argument('--opset-version', type=int, default=16, help='opset version')
    parser.add_argument('--img-folder', type=str, help='path to folder with folders of image categories')
    parser.add_argument('--out-dir', type=str, default='./runs', help='path to save results')

    return parser.parse_args()


def main(opt):
    assert os.path.isdir(opt.img_folder), 'Provide valid path to image folder'
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
        print('No model weights provided')

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-export'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    path_to_save = os.path.join(dir_to_save, 'model.onnx')
    export(model,
           (3, opt.img_size, opt.img_size),
           opt.opset_version,
           path_to_save)

    print('Exported to ONNX and saved to {} folder'.format(dir_to_save))

    torch_x = torch.randn(1, 3, opt.img_size, opt.img_size)
    torch_y = model(torch_x)

    ort = onnxruntime.InferenceSession(path_to_save)
    onnx_x = {ort.get_inputs()[0].name: to_numpy(torch_x)}
    onnx_y = ort.run(None, onnx_x)[0]

    print('pyTorch and ONNX all close on random input?')
    np.testing.assert_allclose(onnx_y, to_numpy(torch_y), rtol=1e-03, atol=1e-05)
    print('Yes')


if __name__ == '__main__':
    options = parse_options()
    main(options)
