import click
import torch
import logging
import random
import numpy as np

from classifier import Classifier
from datasets.main import load_dataset
from utils.config import Config
from utils.visualization.plot_fun import plot_dist


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'emnist', 'cifar10', 'cifar100', 'imagenet1k']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'imagenet_WideResNet', 'toy_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None, help='Model file path (default: None).')
@click.option('--rep_dim', type=int, default=64, help='Final layer dimensionality.')
@click.option('--bias_terms', type=bool, default=True, help='Option to include bias terms in the network.')
@click.option('--objective', type=click.Choice(['hsc', 'deepSAD', 'bce', 'focal']),
              default='hsc', help='Set specific type of classification objective to use.')
@click.option('--hsc_norm', type=click.Choice(['l1', 'l2', 'l2_squared', 'l2_squared_linear']),
              default='l2_squared_linear', help='Set specific norm to use with HSC.')
@click.option('--focal_gamma', type=float, default=2.0, help='Focal loss hyperparameter gamma. Default=2.0')
@click.option('--outlier_exposure', type=bool, default=False,
              help='Apply outlier exposure using oe_dataset_name. Doubles the specified batch_size.')
@click.option('--oe_dataset_name', type=click.Choice(['emnist', 'tinyimages', 'cifar100', 'imagenet22k', 'noise']),
              default='tinyimages', help='Choose the dataset to use as outlier exposure.')
@click.option('--oe_size', type=int, default=79302016,
              help='Size of the outlier exposure dataset (option to train on subsets).')
@click.option('--oe_n_classes', type=int, default=-1,
              help='Number of classes in the outlier exposure dataset.'
                   'If -1, all classes.'
                   'If > 1, the specified number of classes will be sampled at random.')
@click.option('--blur_oe', type=bool, default=False, help='Option to blur (Gaussian filter) OE samples.')
@click.option('--blur_std', type=float, default=1.0, help='Gaussian blurring filter standard deviation. default=1.0')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for training.')
@click.option('--lr', type=float, default=0.001, help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=0.5e-6, help='Weight decay (L2 penalty) hyperparameter.')
@click.option('--data_augmentation', type=bool, default=False,
              help='Apply data augmentation (random flipping, rotation, and translation) for training.')
@click.option('--data_normalization', type=bool, default=False, help='Normalize data wrt dataset sample mean and std.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, rep_dim, bias_terms, objective, hsc_norm,
         focal_gamma, outlier_exposure, oe_dataset_name, oe_size, oe_n_classes, blur_oe, blur_std, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, data_augmentation, data_normalization,
         num_threads, n_jobs_dataloader, normal_class):
    """
    A binary classification model.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % cfg.settings['normal_class'])
    logger.info('Apply outlier exposure: %s' % cfg.settings['outlier_exposure'])
    if outlier_exposure:
        logger.info('Outlier exposure dataset: %s' % cfg.settings['oe_dataset_name'])
        logger.info('Size of OE dataset: %d' % cfg.settings['oe_size'])
        if cfg.settings['oe_dataset_name'] in ['emnist', 'cifar100']:
            logger.info('Number of classes in OE dataset: %d' % cfg.settings['oe_n_classes'])
        logger.info('Blur OE samples with a Gaussian filter: %s' % cfg.settings['blur_oe'])
        if cfg.settings['blur_oe']:
            logger.info('Gaussian blur filter StdDev: %g' % cfg.settings['blur_std'])
    logger.info('Network: %s' % net_name)
    logger.info('Representation dimensionality: %d' % cfg.settings['rep_dim'])
    logger.info('Include bias terms into the network: %s' % cfg.settings['bias_terms'])
    logger.info('Use data augmentation: %s' % cfg.settings['data_augmentation'])
    logger.info('Normalize data: %s' % cfg.settings['data_normalization'])

    # Print model configuration
    logger.info('Objective: %s' % cfg.settings['objective'])
    if cfg.settings['objective'] == 'hsc':
        logger.info('HSC norm: %s' % cfg.settings['hsc_norm'])
    if cfg.settings['objective'] == 'focal':
        logger.info('Focal loss gamma: %g' % cfg.settings['focal_gamma'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        np_random_state = np.random.RandomState(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])
    else:
        cfg.settings['seed'] = None

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        torch.cuda.set_device(device)
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name=dataset_name, data_path=data_path, normal_class=cfg.settings['normal_class'],
                           data_augmentation=cfg.settings['data_augmentation'],
                           normalize=cfg.settings['data_normalization'], seed=cfg.settings['seed'])
    # Load outlier exposure dataset if specified
    if outlier_exposure:
        oe_dataset = load_dataset(dataset_name=cfg.settings['oe_dataset_name'], data_path=data_path,
                                  normal_class=cfg.settings['normal_class'],
                                  data_augmentation=cfg.settings['data_augmentation'],
                                  normalize=cfg.settings['data_normalization'], seed=cfg.settings['seed'],
                                  outlier_exposure=cfg.settings['outlier_exposure'], oe_size=cfg.settings['oe_size'],
                                  oe_n_classes=cfg.settings['oe_n_classes'], blur_oe=cfg.settings['blur_oe'],
                                  blur_std=cfg.settings['blur_std'])
    else:
        oe_dataset = None

    # Initialize Classifier model and set neural network
    classifier = Classifier(cfg.settings['objective'], cfg.settings['hsc_norm'], cfg.settings['focal_gamma'])
    if cfg.settings['objective'] in ['bce', 'focal']:
        net_name = net_name + '_classifier'
    classifier.set_network(net_name, rep_dim=cfg.settings['rep_dim'], bias_terms=cfg.settings['bias_terms'])

    # If specified, load model
    if load_model:
        classifier.load_model(model_path=load_model, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    classifier.train(dataset, oe_dataset,
                     optimizer_name=cfg.settings['optimizer_name'],
                     lr=cfg.settings['lr'],
                     n_epochs=cfg.settings['n_epochs'],
                     lr_milestones=cfg.settings['lr_milestone'],
                     batch_size=cfg.settings['batch_size'],
                     weight_decay=cfg.settings['weight_decay'],
                     device=device,
                     n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    classifier.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Get scores
    train_idx, train_labels, train_scores = zip(*classifier.results['train_scores'])
    train_idx, train_labels, train_scores = np.array(train_idx), np.array(train_labels), np.array(train_scores)
    test_idx, test_labels, test_scores = zip(*classifier.results['test_scores'])
    test_idx, test_labels, test_scores = np.array(test_idx), np.array(test_labels), np.array(test_scores)

    # Plot score distributions
    plot_dist(x=train_scores, xp_path=xp_path, filename='train_scores',
              title='Distribution of anomaly scores (train set)', axlabel='Anomaly Score', legendlabel=['normal'])
    plot_dist(x=test_scores, xp_path=xp_path, filename='test_scores', target=test_labels,
              title='Distribution of anomaly scores (test set)', axlabel='Anomaly Score',
              legendlabel=['normal', 'outlier'])

    # Save results, model, and configuration
    classifier.save_results(export_json=xp_path + '/results.json')
    classifier.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
