import torch
import torch.onnx
import onnxruntime
import numpy as np
import os
import tqdm

from utils.visualise import display_and_compare_batch_predictions
from utils.metrics import calculate_metrics


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export(model_torch,
           input_shape,
           opset_version,
           path_to_save):
    model_torch.cpu().eval()

    batch_size = 1
    x = torch.randn(batch_size, *input_shape, requires_grad=True)

    with torch.no_grad():
        torch_out = model_torch(x)

    torch.onnx.export(model_torch,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      path_to_save,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=opset_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def compare(model_torch,
            ort_session,
            dataloader,
            draw_predictions,
            draw_all,
            class_names,
            path_to_save):
    """

    :param model_torch:
    :param ort_session:
    :param dataloader:
    :param draw_predictions:
    :param draw_all:
    :param class_names:
    :param path_to_save: folder if draw_all
    :return:
    """
    model_torch.eval()

    classes_true = []
    classes_pred_torch = []
    classes_pred_onnx = []

    outputs_torch = []
    outputs_onnx = []

    for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloader)):
        with torch.no_grad():
            outs_torch = model_torch(inputs)
        outputs_torch.append(outs_torch)

        inputs_onnx = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        outs_onnx = ort_session.run(None, inputs_onnx)[0]
        outputs_onnx.append(outs_onnx)

        _, preds_torch = torch.max(outs_torch, 1)
        preds_onnx = np.argmax(outs_onnx, 1)

        classes_true.extend(labels)
        classes_pred_torch.extend(preds_torch)
        classes_pred_onnx.extend(preds_onnx)

        if draw_all:
            display_and_compare_batch_predictions(inputs,
                                                  labels,
                                                  preds_torch,
                                                  preds_onnx,
                                                  'pyTorch',
                                                  'ONNX',
                                                  class_names,
                                                  path_to_save=os.path.join(path_to_save,
                                                                            'batch-{:03d}.png'.format(i + 1)),
                                                  max_batch_size=len(inputs))

        elif draw_predictions:
            draw_predictions = False
            if os.path.isdir(path_to_save):
                path_to_save = os.path.join(path_to_save, 'batch.png')
            fig = display_and_compare_batch_predictions(inputs,
                                                        labels,
                                                        preds_torch,
                                                        preds_onnx,
                                                        'pyTorch',
                                                        'ONNX',
                                                        class_names,
                                                        path_to_save)

    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_torch = calculate_metrics(
        torch.tensor(classes_true), torch.tensor(classes_pred_torch), len(class_names)
    )
    metrics_onnx = calculate_metrics(
        torch.tensor(classes_true), torch.tensor(classes_pred_onnx), len(class_names)
    )
    metrics = {
        'torch': dict(zip(metric_names, metrics_torch)),
        'onnx': dict(zip(metric_names, metrics_onnx))
    }
    return metrics

