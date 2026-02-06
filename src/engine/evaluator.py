"""
 @Author : zhangfan
 @email  : 61316173 @qq.com @Date   : 2026-02-04
"""

import torch

def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    # 针对 V2 的额外统计
    total_loss_cls = 0
    total_loss_time = 0
    
    pad_id = tokenizer.token2id.get("[PAD]", 0)
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5: # V2
                x, y, ages, time_gaps, mask_loss = [t.to(device) for t in batch]
                logits, loss, loss_cls, loss_time = model(x, y, ages, time_gaps, mask_loss)
                if loss_cls is not None: total_loss_cls += loss_cls.item()
                if loss_time is not None: total_loss_time += loss_time.item()
                # V2 预测头是 log-rates，需要 argmax 获得类别
                preds = torch.argmax(logits, dim=-1)
                ignore_idx = pad_id
            else: # V1
                x, y = batch[0].to(device), batch[1].to(device)
                logits, loss = model(x, y)
                preds = torch.argmax(logits, dim=-1)
                ignore_idx = -1
                
            total_loss += loss.item()
            
            mask = (y != ignore_idx)
            correct = (preds == y) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {
        'loss': avg_loss,
        'acc': accuracy,
        'loss_cls': total_loss_cls / len(dataloader),
        'loss_time': total_loss_time / len(dataloader)
    }

def show_predictions(model, dataloader, device, tokenizer, num_samples=1):
    model.eval()
    # 随机取一个 batch
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        if len(batch) == 5: # V2
            x, y, ages, time_gaps, mask_loss = [t.to(device) for t in batch]
            logits, _, _, _ = model(x, ages=ages)
            preds = torch.argmax(logits, dim=-1)
            ignore_idx = tokenizer.token2id.get("[PAD]", 0)
            
            # 时间预测 (Delta T)
            lambdas = torch.exp(logits)
            lambda_total = lambdas.sum(dim=-1)
            pred_time_gaps = 1.0 / (lambda_total + 1e-8)
        else: # V1
            x, y = batch[0].to(device), batch[1].to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            ignore_idx = -1
            pred_time_gaps = None
    
    for i in range(min(num_samples, x.size(0))):
        mask = (y[i] != ignore_idx)
        valid_len = mask.sum().item()
        display_len = min(valid_len, 10)
        
        input_tokens = tokenizer.decode(x[i][:display_len].tolist())
        target_tokens = tokenizer.decode(y[i][:display_len].tolist())
        pred_tokens = tokenizer.decode(preds[i][:display_len].tolist())
        
        print(f"\n[Sample Prediction] Input: {' '.join(input_tokens)}")
        print(f"Target: {' '.join(target_tokens)} | Pred: {' '.join(pred_tokens)}")
        
        if pred_time_gaps is not None:
            target_time = time_gaps[i][:display_len].tolist()
            pred_time = pred_time_gaps[i][:display_len].tolist()
            
            time_comp = []
            errors = []
            for t, p in zip(target_time[:5], pred_time[:5]):
                diff = p - t
                abs_err = abs(diff)
                errors.append(abs_err)
                
                # 定义评价
                if abs_err < 1.0: status = "准"
                elif diff > 0: status = f"迟{abs_err:.1f}天"
                else: status = f"早{abs_err:.1f}天"
                
                time_comp.append(f"<{t:.1f} vs {p:.1f}天:{status}>")
            
            mae = sum(errors) / len(errors) if errors else 0
            print(f"时间预测对比 (实际 vs 预测): {' '.join(time_comp)} | 平均绝对误差: {mae:.2f}天")