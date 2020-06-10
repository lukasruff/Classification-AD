from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
from PIL.ImageFilter import GaussianBlur

import torch
import torchvision.transforms as transforms
import random


class ImageNet22K_Dataset(TorchvisionDataset):

    def __init__(self, root: str, data_augmentation: bool = True, normalize: bool = False, size: int = 14155519,
                 blur_oe: bool = False, blur_std: float = 1.0, seed: int = 0):
        super().__init__(root)

        self.image_size = (3, 224, 224)

        self.n_classes = 1  # only class 1: outlier since ImageNet22K is used for outlier exposure
        self.shuffle = False
        self.size = size

        # ImageNet preprocessing: feature scaling to [0, 1], data normalization, and data augmentation
        transform = [transforms.Resize(256)]
        if data_augmentation:
            transform += [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomCrop(224)]
        else:
            transform += [transforms.CenterCrop(224)]
        if blur_oe:
            transform += [transforms.Lambda(lambda x: x.filter(GaussianBlur(radius=blur_std)))]
        transform += [transforms.ToTensor()]
        if data_augmentation:
            transform += [transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))]
        if normalize:
            transform += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        transform = transforms.Compose(transform)

        # Get dataset
        self.train_set = MyImageNet22K(root=self.root + '/fall11_whole_extracted', size=self.size, transform=transform,
                                       seed=seed)
        self.test_set = None


class MyImageNet22K(ImageFolder):
    """
    Torchvision ImageFolder class with additional targets for the outlier exposure setting and patch of __getitem__
    method to also return the outlier exposure target as well as the index of a data sample.

    Args:
        root (string): Root directory ``fall11_whole_extracted`` of the ImageNet22K dataset.
        size (int, optional): Set the dataset sample size. Default = 14155519 (full dataset; excluding ImageNet1K).
        exclude_imagenet1k (bool, optional): If true, exclude the ImageNet1K samples from the ImageNet22K dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        seed (int, optional): Seed for drawing dataset sample if size is not full dataset
    """
    imagenet1k_pairs = [('acorn', 'n12267677'),
                        ('airliner', 'n02690373'),
                        ('ambulance', 'n02701002'),
                        ('american_alligator', 'n01698640'),
                        ('banjo', 'n02787622'),
                        ('barn', 'n02793495'),
                        ('bikini', 'n02837789'),
                        ('digital_clock', 'n03196217'),
                        ('dragonfly', 'n02268443'),
                        ('dumbbell', 'n03255030'),
                        ('forklift', 'n03384352'),
                        ('goblet', 'n03443371'),
                        ('grand_piano', 'n03452741'),
                        ('hotdog', 'n07697537'),
                        ('hourglass', 'n03544143'),
                        ('manhole_cover', 'n03717622'),
                        ('mosque', 'n03788195'),
                        ('nail', 'n03804744'),
                        ('parking_meter', 'n03891332'),
                        ('pillow', 'n03938244'),
                        ('revolver', 'n04086273'),
                        ('rotary_dial_telephone', 'n03187595'),
                        ('schooner', 'n04147183'),
                        ('snowmobile', 'n04252077'),
                        ('soccer_ball', 'n04254680'),
                        ('stingray', 'n01498041'),
                        ('strawberry', 'n07745940'),
                        ('tank', 'n04389033'),
                        ('toaster', 'n04442312'),
                        ('volcano', 'n09472597')]
    imagenet1k_labels = [label for name, label in imagenet1k_pairs]

    def __init__(self, size: int = 14155519, exclude_imagenet1k=True, seed: int = 0, *args, **kwargs):

        super(MyImageNet22K, self).__init__(*args, **kwargs)
        self.size = size
        self.exclude_imagenet1k = exclude_imagenet1k

        if exclude_imagenet1k:
            imagenet1k_idxs = tuple([self.class_to_idx.get(label) for label in self.imagenet1k_labels])
            self.samples = [s for s in self.samples if s[1] not in imagenet1k_idxs]  # s = ('<path>', idx) pair
            self.targets = [s[1] for s in self.samples]
            self.imgs = self.samples

            for label in self.imagenet1k_labels:
                try:
                    self.classes.remove(label)
                    del self.class_to_idx[label]
                except:
                    pass

        # Draw random permutation of indices of self.size if not full dataset
        self.shuffle_idxs = True
        if self.size < 14155519:
            random.seed(seed)  # set seed to have a fair comparison across models
            self.idxs = random.sample(range(len(self.samples)), self.size)
        else:
            self.idxs = list(range(len(self.samples)))

        self.offset = 0  # offset index

    def __getitem__(self, index):
        """Override the original method of the ImageFolder class.
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        index = (index + self.offset) % self.size
        index = self.idxs[index]

        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 1, -1, index
