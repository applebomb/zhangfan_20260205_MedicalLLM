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
import pandas as pd
import random
import os
from tokenizer import MedicalTokenizer

# 1. 加载 Tokenizer
vocab_path = "./data_processed/vocab.json"
if not os.path.exists(vocab_path):
    # 如果不存在，可能需要先运行 build_vocab.py，但用户说确保存在
    # 这里简单处理，如果不存在则报错
    raise FileNotFoundError(f"词表文件未找到: {vocab_path}")

tokenizer = MedicalTokenizer.from_json(vocab_path)

# 2. 读取原始数据
data_path = "./data_raw/processed_merged_data.xlsx"
df = pd.read_excel(data_path)

# 确保按就诊天数排序
df = df.sort_values(by="就诊天数")

# 提取代码和年龄
# 假设所有数据属于同一个病人 P001
codes = df["诊断编码"].astype(str).tolist()
ages = df["就诊天数"].astype(float).tolist()

# 转化为 ID
input_ids = tokenizer.encode(codes)

processed_data = [{
    "id": "P001",
    "input_ids": input_ids,
    "ages": ages
}]

# 3. 构造训练和测试数据集 (因为只有一个病人，两者相同)
train_data = processed_data
val_data = processed_data

# 4. 持久化保存
os.makedirs("./data_processed", exist_ok=True)
torch.save(train_data, "./data_processed/train.pt")
torch.save(val_data,   "./data_processed/val.pt")

print(f"处理完成：训练集 {len(train_data)} 人, 验证集 {len(val_data)} 人")

# 5. 验证过程
print("\n--- 验证过程 ---")
sample = train_data[0]
print(f"病人 ID: {sample['id']}")
print(f"原始序列长度: {len(codes)}")
print(f"编码后的 ID 序列 (前10个): {sample['input_ids'][:10]}")
print(f"对应的解码序列 (前10个): {tokenizer.decode(sample['input_ids'][:10])}")
print(f"年龄序列 (前10个): {sample['ages'][:10]}")

# 检查是否有 [UNK]
decoded_full = tokenizer.decode(sample['input_ids'])
if "[UNK]" in decoded_full:
    unk_count = decoded_full.count("[UNK]")
    print(f"警告: 发现 {unk_count} 个 [UNK] 标记，请检查词表是否完整。")
else:
    print("验证通过: 所有编码均在词表中。")

# --- 直观验证 .pt 文件内容 ---
print("\n--- .pt 文件内容抽检 ---")
loaded_train = torch.load("./data_processed/train.pt")
print(f"训练集对象类型: {type(loaded_train)}")
print(f"总病人数: {len(loaded_train)}")

if len(loaded_train) > 0:
    sample_patient = loaded_train[0]
    print(f"\n第一个病人的键值结构: {list(sample_patient.keys())}")
    print(f"id: {sample_patient['id']}")
    print(f"input_ids 长度: {len(sample_patient['input_ids'])}")
    print(f"input_ids 示例 (前5个): {sample_patient['input_ids'][:5]}")
    print(f"ages 长度: {len(sample_patient['ages'])}")
    print(f"ages 示例 (前5个): {sample_patient['ages'][:5]}")

