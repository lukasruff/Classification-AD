from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import EMNIST
from base.torchvision_dataset import TorchvisionDataset
from PIL.ImageFilter import GaussianBlur

import numpy as np
import torch
import torchvision.transforms as transforms
import random


class EMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, split: str = 'letters', normal_class: int = 1, outlier_exposure: bool = False,
                 oe_n_classes: int = 26, blur_oe: bool = False, blur_std: float = 1.0, seed: int = 0):
        super().__init__(root)

        self.image_size = (1, 28, 28)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shuffle = True
        self.split = split
        random.seed(seed)  # set seed

        if outlier_exposure:
            self.normal_classes = None
            self.outlier_classes = list(range(1, 27))
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, oe_n_classes))
        else:
            # Define normal and outlier classes
            self.normal_classes = tuple([normal_class])
            self.outlier_classes = list(range(1, 27))
            self.outlier_classes.remove(normal_class)
            self.outlier_classes = tuple(self.outlier_classes)

        # EMNIST preprocessing: feature scaling to [0, 1]
        transform = []
        if blur_oe:
            transform += [transforms.Lambda(lambda x: x.filter(GaussianBlur(radius=blur_std)))]
        transform += [transforms.ToTensor()]
        transform = transforms.Compose(transform)
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyEMNIST(root=self.root, split=self.split, train=True, transform=transform,
                             target_transform=target_transform, download=True)

        if outlier_exposure:
            idx = np.argwhere(np.isin(train_set.targets.cpu().data.numpy(), self.known_outlier_classes))
            idx = idx.flatten().tolist()
            train_set.semi_targets[idx] = -1 * torch.ones(len(idx)).long()  # set outlier exposure labels

            # Subset train_set to selected classes
            self.train_set = Subset(train_set, idx)
            self.train_set.shuffle_idxs = False
            self.test_set = None
        else:
            # Subset train_set to normal_classes
            idx = np.argwhere(np.isin(train_set.targets.cpu().data.numpy(), self.normal_classes))
            idx = idx.flatten().tolist()
            train_set.semi_targets[idx] = torch.zeros(len(idx)).long()
            self.train_set = Subset(train_set, idx)

            # Get test set
            self.test_set = MyEMNIST(root=self.root, split=self.split, train=False, transform=transform,
                                     target_transform=target_transform, download=True)


class MyEMNIST(EMNIST):
    """
    Torchvision EMNIST class with additional targets for the outlier exposure setting and patch of __getitem__ method
    to also return the outlier exposure target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyEMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)
        self.shuffle_idxs = False

    def __getitem__(self, index):
        """Override the original method of the EMNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
