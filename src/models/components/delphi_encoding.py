"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-06
"""

import torch
import torch.nn as nn
import math

class DelphiAgeEncoding(nn.Module):
    """
    Delphi-2M 年龄编码 (Age Encoding)
    基于正弦/余弦的连续时间编码，并经过可学习的线性变换。
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 线性变换，用于混合 Sin/Cos 的结果
        self.linear_proj = nn.Linear(d_model, d_model)
        
        # 频率计算逻辑: lowest frequency is given by 1/365
        # 这里的 d_model 必须是偶数，以便拼接 sin 和 cos
        half_dim = d_model // 2
        
        # 频率分布：基准刻度是 1 天 (更符合 Transformer 标准)
        # 之前的 365.0 导致频率过低，难以区分天级别的差异
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, ages):
        """
        ages: [Batch, Seq_Len] (单位: 天)
        """
        # ages: [Batch, Seq_Len, 1]
        # inv_freq: [Half_Dim]
        # sinusoid_inp: [Batch, Seq_Len, Half_Dim]
        sinusoid_inp = torch.einsum("bi,j->bij", ages.float(), self.inv_freq)
        
        # 生成 Sin 和 Cos
        emb_sin = torch.sin(sinusoid_inp)
        emb_cos = torch.cos(sinusoid_inp)
        
        # 拼接 -> [Batch, Seq_Len, d_model]
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        
        # 经过线性层
        return self.linear_proj(emb)
