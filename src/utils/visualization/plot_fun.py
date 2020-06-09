import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.utils import make_grid


def plot_images_grid(x: torch.tensor, xp_path, filename, title='', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(xp_path + '/' + filename, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


def plot_dist(x, xp_path, filename, target=None, title='', axlabel=None, legendlabel=None):
    """
    Plot the univariate distribution (histogram and kde) of values in x.

    :param x: Data as a series, 1d-array, or list.
    :param xp_path: Export path for the plot as string.
    :param filename: Filename as string.
    :param target: Target as series, 1d-array, or list that categorize subplots. Optional.
    :param title: Title for the plot as string. Optional.
    :param axlabel: Name for the support axis label as string. Optional.
    :param legendlabel: Legend label for the relevant component of the plot as string. Optional.
    """

    # Set plot parameters
    sns.set()
    sns.set_style('white')
    sns.set_palette('colorblind')

    # Convert data to pandas DataFrame and set legend labels
    if target is not None:
        data = {'x': list(x), 'target': list(target)}
        columns = ['x', 'target']
        unique_targets = list(set(target))
        if legendlabel is not None:
            assert len(legendlabel) == len(unique_targets)
            label = legendlabel
        else:
            label = None
    else:
        data = {'x': list(x)}
        columns = ['x']
        if legendlabel is not None:
            assert len(legendlabel) == 1
            label = legendlabel
        else:
            label = None

    dataset = pd.DataFrame(data=data, columns=columns)

    if target is not None:
        # sort dataframe by target
        df_list = [dataset.loc[dataset['target'] == val] for val in unique_targets]
        for i, df in enumerate(df_list):
            sns.distplot(df[['x']], norm_hist=True, axlabel=axlabel, label=label[i])
    else:
        sns.distplot(dataset, norm_hist=True, axlabel=axlabel, label=label)

    if not (title == ''):
        plt.title(title)
    plt.legend()

    plt.savefig(xp_path + '/' + filename, bbox_inches='tight')
    plt.clf()


def plot_line(x, xp_path, filename, title='', xlabel='Epochs', ylabel='Values', legendlabel=None, log_scale=False):
    """
    Draw a line plot with grouping options.

    :param x: Data as a series, 1d-array, or list.
    :param xp_path: Export path for the plot as string.
    :param filename: Filename as string.
    :param title: Title for the plot as string. Optional.
    :param xlabel: Label for x-axis as string. Optional.
    :param ylabel: Label for y-axis as string. Optional.
    :param legendlabel: String or list of strings with data series legend labels. Optional.
    :param log_scale: Boolean to set y-axis to log-scale.
    """

    # Set plot parameters
    sns.set()
    sns.set_style('whitegrid')

    # Convert data to pandas DataFrame and set legend labels
    data = {
        'x': [],
        'y': [],
        'label': []
    }

    if isinstance(x, list):
        n_series = len(x)

        if legendlabel is None:
            legendlabel = ['series ' + str(i + 1) for i in range(n_series)]
        else:
            assert len(legendlabel) == n_series

        for i, series in enumerate(x):
            data['x'].extend(list(range(1, len(x[i]) + 1)))
            data['y'].extend(list(x[i]))
            data['label'].extend([legendlabel[i]] * len(x[i]))
    else:
        if legendlabel is None:
            legendlabel = ['series 1']
        else:
            assert len(legendlabel) == 1

        data['x'].extend(list(range(1, len(x) + 1)))
        data['y'].extend(list(x))
        data['label'].extend(legendlabel * len(x))

    df = pd.DataFrame(data, columns=['x', 'y', 'label'])

    sns.lineplot(x='x', y='y', hue='label', data=df, palette='colorblind')

    if log_scale:
        plt.yscale('symlog')
        plt.grid(True, axis='both')
    else:
        plt.grid(False, axis='x')
        plt.grid(True, axis='y')

    # Add title, axis labels, and legend
    if not (title == ''):
        plt.title(title)
    plt.legend(legendlabel, title=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(xp_path + '/' + filename, bbox_inches='tight')
    plt.clf()
