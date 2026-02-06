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
import os
from tokenizer import MedicalTokenizer
from model_base_202602_ver1 import MedicalGPT2Model, GPT2Config
from dataset import get_dataloader
from train import train_model

def main():
    # 1. 配置路径
    vocab_path = "./data_processed/vocab.json"
    train_path = "./data_processed/train.pt"
    val_path = "./data_processed/val.pt"
    
    # 2. 检查数据
    if not os.path.exists(vocab_path):
        print(f"错误: 词表文件不存在 {vocab_path}，请先运行数据预处理。")
        return
    
    # 3. 加载 Tokenizer 并获取词表大小
    tokenizer = MedicalTokenizer.from_json(vocab_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"词表已加载，Size: {vocab_size}")

    # 4. 模型配置 (一切从简)
    config = GPT2Config(
        vocab_size=vocab_size,
        maxlen=128,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.1
    )
    
    # 5. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = MedicalGPT2Model(config)
    
    # 6. 准备数据加载器
    # 假设 pad_id 在 MedicalTokenizer 中定义为 0 ("[PAD]")
    pad_id = tokenizer.token2id.get("[PAD]", 0)
    
    train_loader = get_dataloader(train_path, batch_size=4, maxlen=config.maxlen, pad_id=pad_id)
    val_loader = get_dataloader(val_path, batch_size=4, maxlen=config.maxlen, pad_id=pad_id)
    
    # 7. 开始训练
    print("开始训练基础模型...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        tokenizer=tokenizer,
        epochs=50,  # 只有一个人，多跑几轮看看 loss 下降情况
        lr=5e-4
    )

    # 8. 保存最终模型
    save_path = "./checkpoints/medical_gpt2_v1_final.pth"
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    main()
