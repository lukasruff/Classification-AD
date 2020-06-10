from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity
from base.torchvision_dataset import TorchvisionDataset
from PIL.ImageFilter import GaussianBlur

import numpy as np
import torch
import torchvision.transforms as transforms
import random
import os


class TinyImages_Dataset(TorchvisionDataset):

    def __init__(self, root: str, data_augmentation: bool = True, normalize: bool = False, size: int = 79302016,
                 blur_oe: bool = False, blur_std: float = 1.0, seed: int = 0):
        super().__init__(root)

        self.image_size = (3, 32, 32)

        self.n_classes = 1  # only class 1: outlier since 80 Million Tiny Images is used for outlier exposure
        self.shuffle = False
        self.size = size

        # TinyImages preprocessing: feature scaling to [0, 1] and data augmentation if specified
        transform = [transforms.ToTensor(),
                     transforms.ToPILImage()]
        if data_augmentation:
            transform += [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(32, padding=4)]
        else:
            transform += [transforms.CenterCrop(32)]
        if blur_oe:
            transform += [transforms.Lambda(lambda x: x.filter(GaussianBlur(radius=blur_std)))]
        transform += [transforms.ToTensor()]
        if data_augmentation:
            transform += [transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))]
        if normalize:
            # CIFAR-10 mean and std
            transform += [transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))]
        transform = transforms.Compose(transform)

        # Get dataset
        self.train_set = TinyImages(root=self.root, size=self.size, transform=transform, download=True, seed=seed)
        self.test_set = None


class TinyImages(VisionDataset):
    """`80 Million Tiny Images <https://groups.csail.mit.edu/vision/TinyImages>`_ Dataset.

    VisionDataset class with additional targets for the outlier exposure setting and patch of __getitem__ method
    to also return the outlier exposure target as well as the index of a data sample.

    Args:
        root (string): Root directory of dataset where ``tiny_images.bin`` file exists or will be saved to
            if download is set to True.
        size (int, optional): Set the dataset sample size. Default = 79302016 (full dataset).
        exclude_cifar (bool, optional): If true, exclude the CIFAR samples from the 80 million tiny images dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If
            dataset is already downloaded, it is not downloaded again.
        seed (int, optional): Seed for drawing dataset sample if size is not full dataset
    """
    url = 'http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin'
    filename = 'tiny_images.bin'

    def __init__(self, root, size: int = 79302016, exclude_cifar=True, transform=None, download=False, seed: int = 0):

        super(TinyImages, self).__init__(root)
        self.size = size
        self.exclude_cifar = exclude_cifar
        self.transform = transform

        # Draw random permutation of indices of self.size if not full dataset
        self.shuffle_idxs = True
        if self.size < 79302016:
            random.seed(seed)
            self.idxs = random.sample(range(79302016), self.size)  # set seed to have a fair comparison across models
        else:
            self.idxs = list(range(79302016))

        if download:
            self.download()

        data_file = open(os.path.join(root, self.filename), 'rb')

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order='F')

        self.load_image = load_image
        self.offset = 0  # offset index

        if exclude_cifar:
            self.cifar_idxs = []
            with open(os.path.join(root, '80mn_cifar_idxs.txt'), 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        index = (index + self.offset) % self.size
        index = self.idxs[index]

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302016)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 1, -1, index

    def __len__(self):
        return 79302016

    def _check_integrity(self):
        root = self.root
        filename = self.filename
        fpath = os.path.join(root, filename)
        return check_integrity(fpath)

    def download(self):

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename)
