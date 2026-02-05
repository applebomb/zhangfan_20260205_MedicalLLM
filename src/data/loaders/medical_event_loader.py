import torch
import torch.nn.functional as F
from ..dataset import BaseDataset

class MedicalEventDataset(BaseDataset):
    """专门针对医疗事件序列预测的任务适配器"""
    def __init__(self, file_path, maxlen=128, pad_id=0):
        super().__init__(file_path)
        self.maxlen = maxlen
        self.pad_id = pad_id

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        
        if len(input_ids) > self.maxlen:
            input_ids = input_ids[:self.maxlen]
        
        x_raw = torch.tensor(input_ids, dtype=torch.long)
        
        # Shifted sequences for next-token prediction
        x = x_raw[:-1]
        y = x_raw[1:]
        
        pad_len = (self.maxlen - 1) - len(x)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=self.pad_id)
            y = F.pad(y, (0, pad_len), value=-1) # ignore_index=-1
        else:
            x = x[:self.maxlen-1]
            y = y[:self.maxlen-1]
            
        return x, y

def get_medical_dataloader(file_path, batch_size=4, maxlen=128, pad_id=0, shuffle=True):
    dataset = MedicalEventDataset(file_path, maxlen, pad_id)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
