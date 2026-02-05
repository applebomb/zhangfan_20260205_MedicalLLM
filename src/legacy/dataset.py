import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MedicalDataset(Dataset):
    """医疗序列数据集"""
    def __init__(self, file_path, maxlen=128, pad_id=0):
        # 加载数据 (List[Dict])
        self.data = torch.load(file_path)
        self.maxlen = maxlen
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        
        # 截断或填充序列
        # 因为是预测下一个疾病，输入是 x[0:n-1], 目标是 x[1:n]
        if len(input_ids) > self.maxlen:
            # 如果太长，随机切一段或者直接截取最后一段
            input_ids = input_ids[:self.maxlen]
        
        # 转换为 Tensor
        x_raw = torch.tensor(input_ids, dtype=torch.long)
        
        # 构造输入和目标 (Shifted)
        # 输入: x[0 : len-1]
        # 目标: x[1 : len]
        x = x_raw[:-1]
        y = x_raw[1:]
        
        # 填充到固定长度 maxlen-1 (因为取了相邻对)
        pad_len = (self.maxlen - 1) - len(x)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=self.pad_id)
            # 目标填充 -1，以便 CrossEntropyLoss ignore_index=-1 忽略
            y = F.pad(y, (0, pad_len), value=-1)
        else:
            # 如果正好或更长，也要保证长度一致
            x = x[:self.maxlen-1]
            y = y[:self.maxlen-1]
            
        return x, y

def get_dataloader(file_path, batch_size=4, maxlen=128, pad_id=0, shuffle=True):
    dataset = MedicalDataset(file_path, maxlen, pad_id)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
