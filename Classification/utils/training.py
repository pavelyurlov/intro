import torch
import torch.nn as nn
import torch.optim as optim

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
    running_corrects = 0
    cnt_total = 0

    for inputs, labels in dataloader:
        global_iter += 1
        cnt_batch = len(inputs)
        cnt_total += cnt_batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_value = loss.item()
        running_loss += loss_value * cnt_batch

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        if use_tensorboard and global_iter % log_interval == 0:
            with summary_writer.as_default():
                summary.scalar('Loss', loss_value, step=global_iter)

        if scheduler_type == 'cyclic' or scheduler_type == 'one-cycle':
            scheduler.step()

    accuracy = running_corrects.float().item() / cnt_total
    if use_tensorboard:
        with summary_writer.as_default():
            summary.scalar('Accuracy', accuracy, step=global_iter)

    return running_loss / cnt_total, accuracy, global_iter


def evaluate(model,
             dataloader,
             device,
             use_tensorboard,
             global_iter,
             summary_writer,
             draw_predictions,
             draw_all,
             class_names,
             path_to_save):
    """
    :param model:
    :param dataloader:
    :param device:
    :param use_tensorboard:
    :param global_iter:
    :param summary_writer:
    :param draw_predictions:
    :param draw_all:
    :param class_names:
    :param path_to_save: folder if draw_all
    :return:
    """
    model.to(device)
    model.eval()

    classes_true = []
    classes_pred = []

    for i, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        classes_true.extend(labels)
        classes_pred.extend(preds)

        if draw_all:
            display_batch_predictions(inputs,
                                      classes_true,
                                      classes_pred,
                                      class_names,
                                      path_to_save=os.path.join(path_to_save,
                                                                'batch-{:03d}.png'.format(i + 1)),
                                      max_batch_size=len(inputs))

        elif draw_predictions:
            draw_predictions = False
            if os.path.isdir(path_to_save):
                path_to_save = os.path.join(path_to_save, 'batch.png')
            fig = display_batch_predictions(inputs,
                                            classes_true,
                                            classes_pred,
                                            class_names,
                                            path_to_save)
            if use_tensorboard:
                with summary_writer.as_default():
                    summary.image('Images', plot_to_image(fig), step=global_iter)

    accuracy, precision, recall, f1 = calculate_metrics(
        torch.tensor(classes_true), torch.tensor(classes_pred), len(class_names)
    )

    if use_tensorboard:
        with summary_writer.as_default():
            summary.scalar('Accuracy', accuracy, step=global_iter)
            summary.scalar('Precision', precision, step=global_iter)
            summary.scalar('Recall', recall, step=global_iter)
            summary.scalar('F1', f1, step=global_iter)

    return accuracy, precision, recall, f1


def train(model,
          device,
          dataloaders,
          optimiser,
          criterion,
          scheduler,
          scheduler_type,
          num_epochs,
          verbose,
          use_tensorboard,
          log_interval,
          summary_writers,
          draw_predictions,
          class_names,
          dir_to_save):
    since = time.time()
    history = {
        'best_acc': 0.0,
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'optimiser': None,
        'scheduler': None,
    }

    best_model_parameters = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    global_iter = 0

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 15)

        # train
        epoch_start = time.time()
        epoch_loss, epoch_acc, global_iter = train_one_epoch(model,
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
        if verbose:
            print('Train loss {:.4f} accuracy {:.4f}    ({:.2f}s)'.format(
                epoch_loss, epoch_acc, time.time() - epoch_start))
        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # validation
        epoch_start = time.time()
        accuracy, precision, recall, f1 = evaluate(model,
                                                   dataloaders['val'],
                                                   device,
                                                   use_tensorboard,
                                                   global_iter,
                                                   summary_writers['val'],
                                                   draw_predictions,
                                                   False,
                                                   class_names,
                                                   path_to_save=os.path.join(
                                                       dir_to_save,
                                                       'val-epoch-{:03d}.png'.format(epoch + 1)
                                                   ))
        if verbose:
            print('Val accuracy {:.4f} precision {:.4f}    ({:.2f}s)'.format(
                accuracy, precision, time.time() - epoch_start))
            print('    recall {:.4f} f1 {:.4f}\n'.format(
                recall, f1))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_parameters = copy.deepcopy(model.state_dict())

        if device.type != 'cpu':
            torch.cuda.empty_cache()

        history['loss'].append(epoch_loss)
        history['accuracy'].append(accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

    time_elapsed = int(time.time() - since)
    print('Training complete in {}m {}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val accuracy {:.4f}'.format(best_accuracy))

    torch.save(model.state_dict(), os.path.join(dir_to_save, 'last.pth'))
    torch.save(best_model_parameters, os.path.join(dir_to_save, 'best.pth'))
    print('Best and last model state dicts saved to {}\n'.format(dir_to_save))

    model.load_state_dict(best_model_parameters)

    history['best-acc'] = best_accuracy
    history['optimiser'] = optimiser
    history['scheduler'] = scheduler
    history['scheduler-type'] = scheduler_type

    with open(os.path.join(dir_to_save, 'history.pickle'), 'wb') as f:
        pickle.dump(history, f)

    return history
