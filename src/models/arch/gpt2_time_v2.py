"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from ..components.layers import Block
from ..components.delphi_encoding import DelphiAgeEncoding
from ..components.loss_func.delphi_loss import DelphiLoss

class MedicalGPT2ModelV2(BaseModel):
    """
    医疗序列 GPT-2 模型 (V2 Delphi-style)
    使用 DelphiAgeEncoding 替代传统的 WPE，并包含双预测头和 DelphiLoss。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.maxlen, config.n_embd),
            age_emb = DelphiAgeEncoding(config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                Block(config.n_embd, config.n_head, config.maxlen, config.dropout) 
                for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Delphi-2M 的核心：Logits 代表发生率 (Rate)
        # 所有的预测共用这一组 Logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 损失函数
        pad_id = getattr(config, 'pad_id', 0)
        time_loss_weight = getattr(config, 'time_loss_weight', 0.01)
        self.criterion = DelphiLoss(ignore_index=pad_id, time_loss_weight=time_loss_weight)

        # 初始化权重
        self.apply(self._init_weights)
        self.print_num_params()

    def forward(self, idx, targets=None, ages=None, time_gaps=None, mask_loss=None):
        """
        idx: [Batch, Seq]
        targets: [Batch, Seq]
        ages: [Batch, Seq] (患者当前的年龄，单位：天)
        time_gaps: [Batch, Seq] (距离下个事件的时间间隔)
        mask_loss: [Batch, Seq] (布尔掩码，True 表示不计入 Loss)
        """
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        
        # Token Embedding
        tok_emb = self.transformer.wte(idx) # [B, T, C]
        # Position Embedding
        pos_emb = self.transformer.wpe(pos) # [1, T, C]
        
        # Age Embedding (Delphi Continuous Encoding)
        if ages is None:
            # 如果没传年龄，回退到 0 (虽然不理想，但保证兼容)
            ages = torch.zeros((b, t), device=device)
        
        age_emb = self.transformer.age_emb(ages) # [B, T, C]
        
        # 融合 Embedding: Token + Position + Age
        x = self.transformer.drop(tok_emb + pos_emb + age_emb)
        
        # Transformer Blocks
        for block in self.transformer.h:
            x = block(x)
        
        # LayerNorm
        x = self.transformer.ln_f(x)
        
        # Logits (Log-Rates)
        logits = self.lm_head(x) # [B, T, Vocab_Size]

        loss = None
        loss_cls = None
        loss_time = None
        
        if targets is not None and time_gaps is not None:
            loss, loss_cls, loss_time = self.criterion(logits, targets, time_gaps, mask_loss)

        return logits, loss, loss_cls, loss_time

    @torch.no_grad()
    def generate(self, idx, current_age, max_new_tokens, temperature=1.0, top_k=None):
        """
        带时间的生成逻辑
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.maxlen else idx[:, -self.config.maxlen:]
            age_cond = current_age if current_age.size(1) <= self.config.maxlen else current_age[:, -self.config.maxlen:]
            
            logits, _, _, _ = self(idx_cond, ages=age_cond)
            
            # 预测下一个事件
            next_event_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_event_logits, min(top_k, next_event_logits.size(-1)))
                next_event_logits[next_event_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_event_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 预测时间间隔 (Delta T)
            # 基于指数分布的性质，平均等待时间是 1/lambda_total
            lambdas = torch.exp(next_event_logits)
            lambda_total = lambdas.sum(dim=-1, keepdim=True)
            
            # 采样 Delta T (使用指数分布采样: -log(U) / lambda)
            u = torch.rand_like(lambda_total)
            delta_t = -torch.log(u + 1e-8) / (lambda_total + 1e-8)
            
            # 更新状态
            idx = torch.cat((idx, idx_next), dim=1)
            next_age = current_age[:, -1:] + delta_t
            current_age = torch.cat((current_age, next_age), dim=1)
            
            # 如果生成了 [END]，则停止
            # if idx_next.item() == self.tokenizer.token2id["[END]"]:
            #     break
                
        return idx, current_age
