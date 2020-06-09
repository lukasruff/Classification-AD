from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

        self.image_size = None  # tuple with the size of an image from the dataset (e.g. (1, 28, 28) for MNIST)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0,
                pin_memory: bool = False) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return train_loader, test_loader
