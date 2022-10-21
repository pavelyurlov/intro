import torch


def calculate_metrics(gt: torch.Tensor, pred: torch.Tensor, num_classes, eps=1e-5):
    """
    :param gt: tensor of ground truth classes
    :param pred: tensor of predicted classes
    :param num_classes:
    :param eps:
    :return: accuracy, precision, recall, F1
    """
    accuracy = (gt == pred).float().mean().item()

    global_precision, global_recall, global_f1 = 0.0, 0.0, 0.0
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c

        tp = (pred_c & gt_c).sum().float().item()
        fp = (pred_c & ~gt_c).sum().float().item()
        fn = (~pred_c & gt_c).sum().float().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        global_precision += precision
        global_recall += recall
        global_f1 += f1

    global_precision /= num_classes
    global_recall /= num_classes
    global_f1 /= num_classes

    return [accuracy, global_precision, global_recall, global_f1]
