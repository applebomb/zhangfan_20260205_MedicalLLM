import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """基础数据集类，可供未来不同任务继承"""
    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError
