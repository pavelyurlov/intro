import numpy as np
import torch
import matplotlib.pyplot as plt
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


def squeeze_mask(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    return img


def save_image(img, path_to_save):
    img = denormalise_image(img)
    plt.imsave(fname=path_to_save, arr=img)


def display_batch_predictions(images,
                              gts,
                              preds,
                              path_to_save,
                              max_batch_size=4):
    batch_size = min(len(images), max_batch_size)
    n_rows = 3
    n_cols = min(4, batch_size)
    fig = plt.figure(figsize=(4 * n_cols, 5 * n_rows))

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(denormalise_image(images[i]))
        ax.axis('off')
        ax.set_title('Image')

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, n_cols + i + 1)
        ax.imshow(squeeze_mask(gts[i]))
        ax.axis('off')
        ax.set_title('Ground truth')

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, 2 * n_cols + i + 1)
        ax.imshow(squeeze_mask(preds[i]))
        ax.axis('off')
        ax.set_title('Prediction')

    fig.savefig(path_to_save, bbox_inches='tight')
    return fig


def display_and_compare_batch_predictions(images,
                                         gts,
                                         preds_first,
                                         preds_second,
                                         name_first,
                                         name_second,
                                         path_to_save,
                                         max_batch_size=4):
    batch_size = min(len(images), max_batch_size)
    n_rows = 4
    n_cols = min(4, batch_size)
    fig = plt.figure(figsize=(4 * n_cols, 5 * n_rows))

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(denormalise_image(images[i]))
        ax.axis('off')
        ax.set_title('Image')

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, n_cols + i + 1)
        ax.imshow(squeeze_mask(gts[i]))
        ax.axis('off')
        ax.set_title('Ground truth')

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, 2 * n_cols + i + 1)
        ax.imshow(squeeze_mask(preds_first[i]))
        ax.axis('off')
        ax.set_title('{} prediction'.format(name_first))

    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, 3 * n_cols + i + 1)
        ax.imshow(squeeze_mask(preds_second[i]))
        ax.axis('off')
        ax.set_title('{} prediction'.format(name_second))

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
