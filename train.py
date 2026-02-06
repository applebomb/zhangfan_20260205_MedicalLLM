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
import argparse
import os
from src.utils.config_utils import load_config
from src.utils.logger import setup_logger # 还没写，先占位或简化
from src.models.arch.registry import get_model
from src.data.loaders.medical_event_loader import get_medical_dataloader
from src.data.loaders.medical_event_time_loader import get_medical_time_dataloader
from src.engine.trainer import Trainer
from src.tokenizer import MedicalTokenizer

def main():
    parser = argparse.ArgumentParser(description="Medical GPT-2 Training")
    parser.add_argument("--config", type=str, default="configs/v1_baseline.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    
    # 2. 环境准备
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)
    print(f"Using device: {device}")

    # 3. 加载 Tokenizer
    tokenizer = MedicalTokenizer.from_json(config.data.vocab_path)
    vocab_size = tokenizer.get_vocab_size()
    # 动态将词表大小和序列长度注入模型配置
    config.model.vocab_size = vocab_size
    config.model.maxlen = config.data.maxlen
    
    # 获取 pad_id 和 ignore_loss_ids
    pad_id = tokenizer.token2id.get("[PAD]", 0)
    ignore_loss_ids = tokenizer.get_ignore_loss_ids()
    config.model.pad_id = pad_id
    config.model.time_loss_weight = getattr(config.train, 'time_loss_weight', 0.01)

    # 4. 构建模型
    model = get_model(config)
    
    # 5. 准备数据加载器
    use_time = getattr(config.data, 'use_time', False)
    
    if use_time:
        train_loader = get_medical_time_dataloader(
            config.data.train_path, 
            batch_size=config.train.batch_size, 
            maxlen=config.data.maxlen, 
            pad_id=pad_id,
            ignore_loss_ids=ignore_loss_ids
        )
        val_loader = get_medical_time_dataloader(
            config.data.val_path, 
            batch_size=config.train.batch_size, 
            maxlen=config.data.maxlen, 
            pad_id=pad_id,
            ignore_loss_ids=ignore_loss_ids
        )
    else:
        train_loader = get_medical_dataloader(
            config.data.train_path, 
            batch_size=config.train.batch_size, 
            maxlen=config.data.maxlen, 
            pad_id=pad_id
        )
        val_loader = get_medical_dataloader(
            config.data.val_path, 
            batch_size=config.train.batch_size, 
            maxlen=config.data.maxlen, 
            pad_id=pad_id
        )

    # 6. 启动训练引擎
    trainer = Trainer(model, config, device, tokenizer)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
