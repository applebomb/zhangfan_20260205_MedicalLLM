"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DelphiLoss(nn.Module):
    """
    Delphi-2M 损失函数，结合了分类损失 (CrossEntropy) 和 时间损失 (Exponential NLL)
    """
    def __init__(self, ignore_index=0, time_loss_weight=0.01):
        super().__init__()
        self.ignore_index = ignore_index # 对应 [PAD] (non-informative)
        self.time_loss_weight = time_loss_weight

    def forward(self, logits, target_ids, target_time_gaps, mask_loss_tokens=None):
        """
        logits: [Batch, Seq, Vocab_Size]
        target_ids: [Batch, Seq] (下一个 Token 的 ID)
        target_time_gaps: [Batch, Seq] (距离下一个 Token 的时间间隔, 单位: 天)
        mask_loss_tokens: [Batch, Seq] (布尔掩码, True 表示该位置不需要计算 Loss, 如性别)
        """
        
        # 计算 Lambda_total 的对数 (使用 logsumexp 保证稳定性)
        # log(sum(exp(logits)))
        log_lambda_total = torch.logsumexp(logits, dim=-1)
        
        # 计算 Lambda_total
        lambda_total = torch.exp(log_lambda_total)

        # --- Part 1: Classification Loss (Cross Entropy) ---
        # 计算分类损失，即预测“下一个是哪个事件”
        # 使用 ignore_index 处理 [PAD]
        loss_cls = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_ids.view(-1), 
            ignore_index=self.ignore_index,
            reduction='none'
        )
        loss_cls = loss_cls.view(target_ids.shape)

        # --- Part 2: Time Loss (Exponential NLL) ---
        # 计算时间损失，即预测“下一个事件何时发生”
        # L_time = -log(lambda_total) + lambda_total * T_observed
        # 针对 T_observed=0 的情况，添加一个微小的偏移量 (如 30 分钟 = 1/48 天) 
        # 否则模型会尝试让 lambda_total 趋于无穷大
        safe_time_gaps = target_time_gaps + (1.0 / 48.0) 
        loss_time = -log_lambda_total + (lambda_total * safe_time_gaps)

        # --- Part 3: 总 Loss 聚合 ---
        # 应用有效掩码 (Valid Mask)
        # 1. 忽略 [PAD] (target_ids == ignore_index)
        # 2. 忽略 mask_loss_tokens (如性别等不应作为预测目标的 token)
        
        valid_mask = (target_ids != self.ignore_index)
        if mask_loss_tokens is not None:
            valid_mask = valid_mask & (~mask_loss_tokens)
        
        valid_mask = valid_mask.float()
        
        # 只在有效位置计算 loss
        masked_loss_cls = loss_cls * valid_mask
        masked_loss_time = loss_time * valid_mask
        
        eps = 1e-8
        total_valid = valid_mask.sum() + eps
        
        mean_loss_cls = masked_loss_cls.sum() / total_valid
        mean_loss_time = masked_loss_time.sum() / total_valid
        
        # 使用权重平衡两个任务 (默认 0.01 缩小 time loss 规模)
        total_loss = mean_loss_cls + (self.time_loss_weight * mean_loss_time)
        
        return total_loss, mean_loss_cls, mean_loss_time
