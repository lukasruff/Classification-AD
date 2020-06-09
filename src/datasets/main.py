from .mnist import MNIST_Dataset
from .emnist import EMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .cifar100 import CIFAR100_Dataset
from .tinyimages import TinyImages_Dataset
from .imagenet1k import ImageNet1K_Dataset
from .imagenet22k import ImageNet22K_Dataset


def load_dataset(dataset_name, data_path, normal_class, data_augmentation: bool = False, normalize: bool = False,
                 seed=None, outlier_exposure: bool = False, oe_size: int = 79302016, oe_n_classes: int = -1,
                 blur_oe: bool = False, blur_std: float = 1.0):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'emnist', 'cifar10', 'cifar100', 'tinyimages', 'imagenet1k', 'imagenet22k')
    assert dataset_name in implemented_datasets

    # Set default number of OE classes if oe_n_classes == -1
    if oe_n_classes == -1:
        if dataset_name == 'emnist':
            oe_n_classes = 26
        if dataset_name == 'cifar100':
            oe_n_classes = 100

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'emnist':
        dataset = EMNIST_Dataset(root=data_path,
                                 normal_class=normal_class,
                                 outlier_exposure=outlier_exposure,
                                 oe_n_classes=oe_n_classes,
                                 blur_oe=blur_oe,
                                 blur_std=blur_std,
                                 seed=seed)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  data_augmentation=data_augmentation,
                                  normalize=normalize)

    if dataset_name == 'cifar100':
        dataset = CIFAR100_Dataset(root=data_path,
                                   normal_class=normal_class,
                                   data_augmentation=data_augmentation,
                                   normalize=normalize,
                                   outlier_exposure=outlier_exposure,
                                   oe_n_classes=oe_n_classes,
                                   seed=seed)

    if dataset_name == 'tinyimages':
        dataset = TinyImages_Dataset(root=data_path,
                                     data_augmentation=data_augmentation,
                                     normalize=normalize,
                                     size=oe_size,
                                     blur_oe=blur_oe,
                                     blur_std=blur_std,
                                     seed=seed)

    if dataset_name == 'imagenet1k':
        dataset = ImageNet1K_Dataset(root=data_path,
                                     normal_class=normal_class,
                                     data_augmentation=data_augmentation,
                                     normalize=normalize)

    if dataset_name == 'imagenet22k':
        dataset = ImageNet22K_Dataset(root=data_path,
                                      data_augmentation=data_augmentation,
                                      normalize=normalize,
                                      size=oe_size,
                                      blur_oe=blur_oe,
                                      blur_std=blur_std,
                                      seed=seed)

    return dataset
