import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiceLossPlusCE(nn.Module):
    def __init__(self, weight=None):
        super(DiceLossPlusCE, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, eps=1e-7):
        ce_loss = self.CE(inputs, targets)

        num_classes = inputs.size(1)

        true_one_hot = torch.eye(num_classes)[targets].permute(0, 3, 1, 2).type(inputs.type())
        probas = F.softmax(inputs, dim=1)
        dims = (0,) + tuple(range(2, true_one_hot.ndimension()))
        intersection = torch.sum(probas * true_one_hot, dims)
        cardinality = torch.sum(probas + true_one_hot, dims)
        dice_loss = 1 - (2. * intersection / (cardinality + eps)).mean()

        return ce_loss + dice_loss


def calculate_metrics(gt: torch.Tensor, pred: torch.Tensor, num_classes, eps=1e-5):
    """
    :param gt:
    :param pred:
    :param num_classes:
    :param eps:
    :return: accuracy, precision, recall, IoU
    """
    accuracy = (gt == pred).float().mean().item()

    precision, recall, iou = 0.0, 0.0, 0.0
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c

        tp = (pred_c & gt_c).sum().float().item()
        fp = (pred_c & ~gt_c).sum().float().item()
        fn = (~pred_c & gt_c).sum().float().item()

        precision += tp / (tp + fp + eps)
        recall += tp / (tp + fn + eps)
        iou += tp / (tp + fp + fn + eps)

    precision /= num_classes
    recall /= num_classes
    iou /= num_classes

    return np.array([accuracy, precision, recall, iou])
