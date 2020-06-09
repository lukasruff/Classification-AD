import torch
import numpy as np

from utils.visualization.plot_fun import plot_images_grid


def plot_extreme_samples(data, scores, labels, idx, xp_path, n=32, prefix='', suffix=''):
    """
    Plots the n extreme (most anomalous and most normal) train and test set samples.

    :param data: The data of type torch.Tensor or np.ndarray.
    :param scores: iterable with scores.
    :param labels: iterable with labels.
    :param idx: iterable with indices.
    :param xp_path: export path for image files.
    :param n: number of extreme samples to plot.
    :param prefix: option to add some prefix to filenames.
    :param suffix: option to add some suffix to filenames.
    """

    if not (torch.is_tensor(data)):
        data = torch.tensor(data)  # convert to torch.Tensor if np.ndarray
    if data.dim() == 3:  # single-channel images with B x H x W
        data = data.unsqueeze(1)  # add channel C dimension
    if data.dim() == 4 and not data.size(1) in (1, 3):
        data = data.permute(0, 3, 1, 2)  # Convert from (B x H x W x C) to (B x C x H x W)

    # Overall data
    idx_sorted = idx[np.argsort(scores)]  # by score, from lowest to highest
    X_low = data[idx_sorted[:n], ...]
    X_high = data[idx_sorted[-n:], ...]
    plot_images_grid(X_low, xp_path=xp_path, filename= prefix + 'all_low' + suffix, padding=2)
    plot_images_grid(X_high, xp_path=xp_path, filename=prefix + 'all_high' + suffix, padding=2)

    # Normal samples (within-class scoring)
    idx_sorted = idx[labels == 0][np.argsort(scores[labels == 0])]  # by score, from lowest to highest
    X_low = data[idx_sorted[:n], ...]
    X_high = data[idx_sorted[-n:], ...]
    plot_images_grid(X_low, xp_path=xp_path, filename=prefix + 'normal_low' + suffix, padding=2)
    plot_images_grid(X_high, xp_path=xp_path, filename=prefix + 'normal_high' + suffix, padding=2)

    # Outlier samples (out-of-class scoring)
    if np.sum(labels) > 0:
        idx_sorted = idx[labels == 1][np.argsort(scores[labels == 1])]  # by score, from lowest to highest
        X_low = data[idx_sorted[:n], ...]
        X_high = data[idx_sorted[-n:], ...]
        plot_images_grid(X_low, xp_path=xp_path, filename=prefix + 'outlier_low' + suffix, padding=2)
        plot_images_grid(X_high, xp_path=xp_path, filename=prefix + 'outlier_high' + suffix, padding=2)
