import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import os
import pickle
import time
import copy

from tensorflow import summary
from utils.metrics import calculate_metrics
from utils.visualise import display_batch_predictions, plot_to_image


def train_one_epoch(model,
                    criterion,
                    optimiser,
                    scheduler,
                    scheduler_type,
                    dataloader,
                    device,
                    use_tensorboard,
                    global_iter,
                    log_interval,
                    summary_writer):
    model.to(device)
    model.train()

    global_iter = global_iter
    running_loss = 0.0
    running_metrics = np.zeros(4)

    cnt_total = 0

    for inputs, masks in dataloader:
        global_iter += 1
        cnt_batch = len(inputs)
        cnt_total += cnt_batch

        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs)  # ['out']
        loss = criterion(outputs, masks)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_value = loss.item()
        running_loss += loss_value * cnt_batch
        running_metrics += cnt_batch * calculate_metrics(
            masks, torch.argmax(outputs, 1), outputs.size(1)
        )

        if use_tensorboard and global_iter % log_interval == 0:
            with summary_writer.as_default():
                summary.scalar('loss', loss_value, step=global_iter)

        if scheduler_type == 'cyclic' or scheduler_type == 'one-cycle':
            scheduler.step()

    epoch_loss = running_loss / cnt_total
    epoch_metrics = running_metrics / cnt_total

    if use_tensorboard:
        with summary_writer.as_default():
            summary.scalar('pixel accuracy', epoch_metrics[0], step=global_iter)
            summary.scalar('mean class precision', epoch_metrics[1], step=global_iter)
            summary.scalar('mean class recall', epoch_metrics[2], step=global_iter)
            summary.scalar('mean class IoU', epoch_metrics[3], step=global_iter)

    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'IoU']
    epoch_stats = dict(zip(metric_names, [epoch_loss, *epoch_metrics]))

    return epoch_stats, global_iter


def evaluate(model,
             criterion,
             dataloader,
             device,
             use_tensorboard,
             global_iter,
             summary_writer,
             draw_predictions,
             draw_all,
             path_to_save):
    """
    :param model:
    :param criterion:
    :param dataloader:
    :param device:
    :param use_tensorboard:
    :param global_iter:
    :param summary_writer:
    :param draw_predictions:
    :param draw_all:
    :param path_to_save: folder if draw_all
    :return:
    """
    model.to(device)
    model.eval()

    running_loss = 0.0
    running_metrics = np.zeros(4)

    cnt_total = 0

    for i, (inputs, masks) in enumerate(dataloader):
        cnt_batch = len(inputs)
        cnt_total += cnt_batch

        inputs = inputs.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs = model(inputs)  # ['out']
        loss = criterion(outputs, masks)

        loss_value = loss.item()
        running_loss += loss_value * cnt_batch
        running_metrics += cnt_batch * calculate_metrics(
            masks, torch.argmax(outputs, 1), outputs.size(1)
        )

        if draw_all:
            display_batch_predictions(inputs,
                                      masks,
                                      torch.argmax(outputs, 1),
                                      path_to_save=os.path.join(path_to_save,
                                                                'batch-{:03d}.png'.format(i + 1)),
                                      max_batch_size=len(inputs))

        elif draw_predictions:
            draw_predictions = False
            if os.path.isdir(path_to_save):
                path_to_save = os.path.join(path_to_save, 'batch.png')
            fig = display_batch_predictions(inputs,
                                            masks,
                                            torch.argmax(outputs, 1),
                                            path_to_save)
            if use_tensorboard:
                with summary_writer.as_default():
                    summary.image('Images', plot_to_image(fig), step=global_iter)

    epoch_loss = running_loss / cnt_total
    epoch_metrics = running_metrics / cnt_total

    if use_tensorboard:
        with summary_writer.as_default():
            summary.scalar('loss', epoch_loss, step=global_iter)
            summary.scalar('pixel accuracy', epoch_metrics[0], step=global_iter)
            summary.scalar('mean class precision', epoch_metrics[1], step=global_iter)
            summary.scalar('mean class recall', epoch_metrics[2], step=global_iter)
            summary.scalar('mean class IoU', epoch_metrics[3], step=global_iter)

    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'IoU']
    epoch_stats = dict(zip(metric_names, [epoch_loss, *epoch_metrics]))

    return epoch_stats


def train(model,
          device,
          criterion,
          dataloaders,
          optimiser,
          scheduler,
          scheduler_type,
          num_epochs,
          verbose,
          use_tensorboard,
          log_interval,
          summary_writers,
          draw_predictions,
          dir_to_save):
    since = time.time()
    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'IoU']

    history = {
        'best_val_iou': 0.0,
        'train': {k: [] for k in metric_names},
        'val': {k: [] for k in metric_names},
        'optimiser': None,
        'scheduler': None,
    }

    best_model_parameters = copy.deepcopy(model.state_dict())
    best_iou = 0.0

    global_iter = 0

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 15)

        # train
        epoch_start = time.time()
        epoch_stats, global_iter = train_one_epoch(model,
                                                   criterion,
                                                   optimiser,
                                                   scheduler,
                                                   scheduler_type,
                                                   dataloaders['train'],
                                                   device,
                                                   use_tensorboard,
                                                   global_iter,
                                                   log_interval,
                                                   summary_writers['train'])
        if scheduler_type == 'step':
            scheduler.step()

        for k, v in epoch_stats.items():
            history['train'][k].append(v)
        if verbose:
            print('Train    ({:.2f}s)'.format(time.time() - epoch_start))
            for k, v in epoch_stats.items():
                print('  {} {:.4f}'.format(k, v), end='')
            print()

        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # validation
        epoch_start = time.time()
        epoch_stats = evaluate(model,
                               criterion,
                               dataloaders['val'],
                               device,
                               use_tensorboard,
                               global_iter,
                               summary_writers['val'],
                               draw_predictions,
                               False,
                               path_to_save=os.path.join(
                                   dir_to_save,
                                   'val-epoch-{:03d}.png'.format(epoch + 1)
                               ))
        for k, v in epoch_stats.items():
            history['val'][k].append(v)
        if verbose:
            print('Val    ({:.2f}s)'.format(time.time() - epoch_start))
            for k, v in epoch_stats.items():
                print('  {} {:.4f}'.format(k, v), end='')
            print()

        if epoch_stats['IoU'] > best_iou:
            best_iou = epoch_stats['IoU']
            best_model_parameters = copy.deepcopy(model.state_dict())

        if device.type != 'cpu':
            torch.cuda.empty_cache()

    time_elapsed = int(time.time() - since)
    print('Training complete in {}m {}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val IoU {:.4f}'.format(best_iou))

    torch.save(model.state_dict(), os.path.join(dir_to_save, 'last.pth'))
    torch.save(best_model_parameters, os.path.join(dir_to_save, 'best.pth'))
    print('Results saved to {}'.format(dir_to_save))

    model.load_state_dict(best_model_parameters)

    history['best_val_iou'] = best_iou
    history['optimiser'] = optimiser
    history['scheduler'] = scheduler
    history['scheduler-type'] = scheduler_type

    with open(os.path.join(dir_to_save, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    return history
