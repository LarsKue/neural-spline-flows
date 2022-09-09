
import torch
from torch.utils.data import Dataset


class UnlabeledDataset(Dataset):
    def __init__(self, data: Dataset):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, y = self.data.__getitem__(item)
        return x


class SingleTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = torch.as_tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
