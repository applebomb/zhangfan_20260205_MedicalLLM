"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-04
Copyright (c) 2026 61316173 @qq.com. All Rights Reserved.

NOTICE:  All information contained herein is, and remains
the property of the author. The intellectual and technical concepts
contained herein are proprietary to the author and are protected
by trade secret or copyright law. Dissemination of this information
or reproduction of this material is strictly forbidden unless prior
written permission is obtained from the author.
"""

import torch
from tokenizer import MedicalTokenizer

def evaluate(model, dataloader, device, tokenizer):
    """
    评估模型在验证集上的表现
    返回: 平均 Loss 和 准确率 (忽略填充部分)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            total_loss += loss.item()
            
            # 计算准确率 (排除 target 为 -1 的 padding 部分)
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -1)
            correct = (preds == y) & mask
            
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy

def show_predictions(model, dataloader, device, tokenizer, num_samples=1):
    """
    直观展示预测结果：原序列 -> 真实下一个 -> 预测下一个
    """
    model.eval()
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        logits, _ = model(x)
        preds = torch.argmax(logits, dim=-1)
    
    for i in range(min(num_samples, x.size(0))):
        # 找到非 padding 的长度
        mask = (y[i] != -1)
        valid_len = mask.sum().item()
        
        # 为了展示，我们只取前几个 token
        display_len = min(valid_len, 10)
        
        input_tokens = tokenizer.decode(x[i][:display_len].tolist())
        target_tokens = tokenizer.decode(y[i][:display_len].tolist())
        pred_tokens = tokenizer.decode(preds[i][:display_len].tolist())
        
        print(f"\n样本 {i+1} 预测对比 (前 {display_len} 个有效位置):")
        print(f"输入序列: {' '.join(input_tokens)}")
        print(f"真实后续: {' '.join(target_tokens)}")
        print(f"预测后续: {' '.join(pred_tokens)}")
        
        # 计算该样本的简单错误率
        error_rate = 1 - (preds[i][:valid_len] == y[i][:valid_len]).float().mean().item()
        print(f"该序列实际错误率: {error_rate:.2%}")
