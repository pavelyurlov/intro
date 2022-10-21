import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import torch.onnx
import onnxruntime


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
from utils.dataloaders import *
from utils.onnx import *


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lraspp', 'deeplabv3+'],
                        default='deeplabv3+', help='model')
    parser.add_argument('--weights', type=str, default=None, help='path to model state dict')
    parser.add_argument('--n-classes', type=int, default=5, help='number of classes')
    parser.add_argument('--img-height', type=int, default=528, help='image height')
    parser.add_argument('--img-width', type=int, default=928, help='image width')
    parser.add_argument('--opset-version', type=int, default=16, help='opset version')
    parser.add_argument('--out-dir', type=str, default='./runs', help='path to save results')

    return parser.parse_args()


def main(opt):
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
        print('No model weights provided')

    dir_specify = len(os.listdir(opt.out_dir)) if os.path.isdir(opt.out_dir) else 0
    dir_to_save = os.path.join(opt.out_dir, 'run{:02d}-export'.format(dir_specify + 1))
    os.makedirs(dir_to_save)

    path_to_save = os.path.join(dir_to_save, 'model.onnx')
    export(model,
           (3, opt.img_height, opt.img_width),
           opt.opset_version,
           path_to_save)

    print('Exported to ONNX and saved to {} folder'.format(dir_to_save))

    torch_x = torch.randn(1, 3, opt.img_height, opt.img_width)
    torch_y = model(torch_x)  # ['out']

    ort = onnxruntime.InferenceSession(path_to_save)
    onnx_x = {ort.get_inputs()[0].name: to_numpy(torch_x)}
    onnx_y = ort.run(None, onnx_x)[0]

    print('pyTorch and ONNX all close on random input?')
    np.testing.assert_allclose(onnx_y, to_numpy(torch_y), rtol=1e-03, atol=1e-05)
    print('Yes')


if __name__ == '__main__':
    options = parse_options()
    main(options)
