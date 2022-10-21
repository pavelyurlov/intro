import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
import io


MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)


def denormalise_image(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0) * np.array(STDS) + np.array(MEANS)
    img = np.clip(img, 0, 1)
    return img


def save_image(img, path_to_save):
    img = denormalise_image(img)
    plt.imsave(fname=path_to_save, arr=img)


def save_confusion_matrix(classes_true,
                          classes_pred,
                          figsize,
                          path_to_save):
    f, ax = plt.subplots(figsize=(figsize, figsize))
    ConfusionMatrixDisplay.from_predictions(classes_true, classes_pred, ax=ax)
    f.savefig(path_to_save)


def display_batch_predictions(images,
                              classes_true,
                              classes_pred,
                              class_names,
                              path_to_save,
                              max_batch_size=16):
    batch_size = min(len(images), max_batch_size)
    n_rows = int(np.ceil(batch_size / 4))
    n_cols = min(4, batch_size)
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    # gs = gridspec.GridSpec(n_rows, n_cols)
    for i in range(batch_size):
        # ax = plt.subplot(gs[i // n_cols, i % n_cols])
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(denormalise_image(images[i]))
        ax.axis('off')
        y, z = classes_true[i].item(), classes_pred[i].item()
        title = 'Ground truth {}\nPrediction {}'.format(
            class_names[y], class_names[z])
        ax.set_title(title, color=['red', 'green'][y == z])
    fig.savefig(path_to_save, bbox_inches='tight')
    return fig


def display_and_compare_batch_predictions(images,
                                          classes_true,
                                          classes_pred_first,
                                          classes_pred_second,
                                          name_first,
                                          name_second,
                                          class_names,
                                          path_to_save,
                                          max_batch_size=16):
    batch_size = min(len(images), max_batch_size)
    n_rows = int(np.ceil(batch_size / 4))
    n_cols = min(4, batch_size)
    fig = plt.figure(figsize=(4 * n_cols, 4.5 * n_rows))
    # gs = gridspec.GridSpec(n_rows, n_cols)
    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        # ax = plt.subplot(gs[i // n_cols, i % n_cols])
        ax.imshow(denormalise_image(images[i]))
        ax.axis('off')
        y = classes_true[i].item()
        z_first = classes_pred_first[i].item()
        z_second = classes_pred_second[i].item()
        title = 'Ground truth {}\n{} prediction {}\n{} prediction {}'.format(class_names[y],
                                                                             name_first, class_names[z_first],
                                                                             name_second, class_names[z_second])
        ax.set_title(title, color=['red', 'green'][z_first == z_second])
    fig.savefig(path_to_save, bbox_inches='tight')
    return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    #plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
