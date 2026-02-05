import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    模型基类，提供共性功能：
    1. 统一的权重初始化
    2. 参数量统计
    3. 基础保存/加载接口
    """
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def print_num_params(self):
        print(f"Model parameters: {self.get_num_params()/1e3:.2f}K")
