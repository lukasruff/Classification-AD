from torch.utils.data import Dataset, DataLoader
from base.base_dataset import BaseADDataset


class Array_Dataset(BaseADDataset):

    def __init__(self, X_train, y_train, y_semi_train, X_test=None, y_test=None, y_semi_test=None):
        super().__init__(root='')

        self.shuffle = True

        # Get train set
        self.train_set = MyTensorDataset(X_train, y_train, y_semi_train)

        # Get test set
        if X_test is None:
            self.test_set = None
        else:
            self.test_set = MyTensorDataset(X_test, y_test, y_semi_test)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0,
                pin_memory: bool = False) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return train_loader, test_loader


class MyTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors: Triple of (X, target, semi_target) tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert len(tensors) == 3

        self.tensors = tensors
        self.shuffle_idxs = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (x, target, semi_target, index)
        """
        return self.tensors[0][index], self.tensors[1][index], self.tensors[2][index], index

    def __len__(self):
        return self.tensors[0].size(0)
