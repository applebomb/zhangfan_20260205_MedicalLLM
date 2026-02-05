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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from .evaluator import evaluate, show_predictions

class Trainer:
    """通用训练引擎"""
    def __init__(self, model, config, device, tokenizer):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # 从配置读取超参
        self.lr = getattr(config.train, 'lr', 1e-3)
        self.epochs = getattr(config.train, 'epochs', 10)
        self.exp_name = getattr(config, 'exp_name', 'default_exp')
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # TensorBoard
        self.log_dir = os.path.join("runs", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.model.to(self.device)

    def train(self, train_loader, val_loader):
        global_step = 0
        print(f"开始训练实验: {self.exp_name}")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                logits, loss = self.model(x, y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
            
            # 验证
            val_loss, val_acc = evaluate(self.model, val_loader, self.device, self.tokenizer)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            # 预测示例展示
            show_predictions(self.model, val_loader, self.device, self.tokenizer)

        self.writer.close()
        self.save_checkpoint()

    def save_checkpoint(self):
        save_dir = os.path.join("checkpoints", self.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "final_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")
