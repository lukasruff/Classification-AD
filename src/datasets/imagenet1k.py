from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset

import numpy as np
import torch
import torchvision.transforms as transforms


class ImageNet1K_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, data_augmentation: bool = False, normalize: bool = False):
        super().__init__(root)

        classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
                   'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
                   'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner',
                   'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']

        self.image_size = (3, 224, 224)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 30))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        # ImageNet preprocessing: feature scaling to [0, 1], data normalization, and data augmentation
        train_transform = [transforms.Resize(256)]
        test_transform = [transforms.Resize(256),
                          transforms.CenterCrop(224)]
        if data_augmentation:
            # only augment training data
            train_transform += [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomCrop(224)]
        else:
            train_transform += [transforms.CenterCrop(224)]
        train_transform += [transforms.ToTensor()]
        test_transform += [transforms.ToTensor()]
        if data_augmentation:
            train_transform += [transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))]
        if normalize:
            train_transform += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            test_transform += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyImageNet1K(root=self.root + '/imagenet1k/one_class_train', transform=train_transform,
                                 target_transform=target_transform)

        # Subset train_set to normal_classes
        idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
        idx = idx.flatten().tolist()
        train_set.semi_targets[idx] = torch.zeros(len(idx)).long()
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyImageNet1K(root=self.root + '/imagenet1k/one_class_test', transform=test_transform,
                                     target_transform=target_transform)


class MyImageNet1K(ImageFolder):
    """
    Torchvision ImageFolder class with additional targets for the outlier exposure setting and patch of __getitem__
    method to also return the outlier exposure target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyImageNet1K, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        """Override the original method of the ImageFolder class.
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        semi_target = int(self.semi_targets[index])

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, semi_target, index
