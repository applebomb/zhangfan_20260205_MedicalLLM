"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-06
"""

import torch
import torch.nn.functional as F
from ..dataset import BaseDataset

class MedicalEventTimeDataset(BaseDataset):
    """
    专门针对带有时间信息的医疗事件序列预测的数据集适配器
    返回: x (input_ids), y (target_ids), ages_x (当前年龄), time_gaps_y (距离下次事件的时间), mask_loss (是否忽略Loss)
    """
    def __init__(self, file_path, maxlen=128, pad_id=0, ignore_loss_ids=None):
        super().__init__(file_path)
        self.maxlen = maxlen
        self.pad_id = pad_id
        self.ignore_loss_ids = ignore_loss_ids if ignore_loss_ids is not None else []

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        ages = item['ages']
        
        # 截断
        if len(input_ids) > self.maxlen:
            input_ids = input_ids[:self.maxlen]
            ages = ages[:self.maxlen]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ages = torch.tensor(ages, dtype=torch.float)
        
        # 计算时间间隔 (Time Gaps)
        # time_gaps[i] = ages[i+1] - ages[i]
        # 最后一个元素的 time_gap 可以设为 0 或者特定值，但通常 y 的最后一位是 ignore_index
        time_gaps = torch.zeros_like(ages)
        if len(ages) > 1:
            time_gaps[:-1] = ages[1:] - ages[0:-1]
        
        # 构造训练对
        x = input_ids[:-1]
        y = input_ids[1:]
        
        ages_x = ages[:-1]
        time_gaps_y = time_gaps[:-1]
        
        # 确定哪些 token 不参与 Loss 计算 (如 MALE, FEMALE)
        mask_loss = torch.tensor([tid.item() in self.ignore_loss_ids for tid in y], dtype=torch.bool)

        # 填充 (Padding)
        pad_len = (self.maxlen - 1) - len(x)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=self.pad_id)
            y = F.pad(y, (0, pad_len), value=self.pad_id) # 这里 y 也用 pad_id，Loss 会处理 ignore_index
            ages_x = F.pad(ages_x, (0, pad_len), value=0.0)
            time_gaps_y = F.pad(time_gaps_y, (0, pad_len), value=0.0)
            mask_loss = F.pad(mask_loss, (0, pad_len), value=False)
        else:
            x = x[:self.maxlen-1]
            y = y[:self.maxlen-1]
            ages_x = ages_x[:self.maxlen-1]
            time_gaps_y = time_gaps_y[:self.maxlen-1]
            mask_loss = mask_loss[:self.maxlen-1]
            
        return x, y, ages_x, time_gaps_y, mask_loss

def get_medical_time_dataloader(file_path, batch_size=4, maxlen=128, pad_id=0, ignore_loss_ids=None, shuffle=True):
    dataset = MedicalEventTimeDataset(file_path, maxlen, pad_id, ignore_loss_ids)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
