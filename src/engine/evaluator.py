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

def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -1)
            correct = (preds == y) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

def show_predictions(model, dataloader, device, tokenizer, num_samples=1):
    model.eval()
    # 随机取一个 batch
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        logits, _ = model(x)
        preds = torch.argmax(logits, dim=-1)
    
    for i in range(min(num_samples, x.size(0))):
        mask = (y[i] != -1)
        valid_len = mask.sum().item()
        display_len = min(valid_len, 10)
        
        input_tokens = tokenizer.decode(x[i][:display_len].tolist())
        target_tokens = tokenizer.decode(y[i][:display_len].tolist())
        pred_tokens = tokenizer.decode(preds[i][:display_len].tolist())
        
        print(f"\n[Sample Prediction] Input: {' '.join(input_tokens)}")
        print(f"Target: {' '.join(target_tokens)} | Pred: {' '.join(pred_tokens)}")
