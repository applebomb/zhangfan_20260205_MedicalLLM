import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from eval import evaluate, show_predictions
import os

def train_model(model, train_loader, val_loader, config, device, tokenizer, epochs=10, lr=1e-3):
    """
    训练模型的主函数
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 初始化 TensorBoard SummaryWriter
    log_dir = "./runs/medical_gpt2_v1"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志将保存至: {log_dir}")

    model.to(device)
    
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 记录 Step Loss
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device, tokenizer)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        print(f"\n--- Epoch {epoch+1} 结束 ---")
        print(f"平均训练 Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"验证集 Loss: {val_loss:.4f} | 验证集准确率: {val_acc:.2%}")
        
        # 每轮打印一些预测示例
        show_predictions(model, val_loader, device, tokenizer)
        print("-" * 30)

    writer.close()
    return model
