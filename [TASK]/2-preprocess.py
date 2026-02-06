帮我将 数据集处理成可以准备送入模型的形式

- 数据文件是 ./data_raw/processed_merged_data.xlsx
里面 就诊天数 对应 年龄（出生后多少天）， 诊断编码 是需要进行 tokenizer 的 对象

- 因为只有一个 病人， 所以构造的训练和测试数据集相同，主要是跑通

- 处理好以后，写一个 验证过程，在终端我看看对不对，包括 编码 和 序列

- 处理完以后是方便送入模型的状态

- 你的处理 py 写在 ./src/preproess.py 这个文件


这是给你的参考，但是需要根据上面的任务进行修订，里面细节有出入：

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
import random
from tokenizer import MedicalTokenizer # 引用你之前写的类  tokenizer 这个py 在./src/下面

# 1. 加载 Tokenizer (确保 vocab.json 存在)
tokenizer = MedicalTokenizer.from_json("./data_processed/vocab.json")

# 2. 模拟读取你的原始数据 (这里假设你已经解析成了 list)
# 格式: (patient_id, [(Event_Str, Age_Float), ...])
raw_patients = [
    ("P001", [("MALE", 0.0), ("E11", 45.2), ("DEATH", 78.0)]),
    ("P002", [("FEMALE", 0.0), ("I10", 50.1), ("[END]", 50.1)]),
    # ... 你的56个病人
]

processed_data = []

for pid, events in raw_patients:
    codes = [e for e in events]
    ages  = [e[1] for e in events]
    
    # 转化为 ID
    input_ids = tokenizer.encode(codes)
    
    # 存入字典
    processed_data.append({
        "id": pid,
        "input_ids": input_ids,
        "ages": ages
    })

# 3. 划分数据集 (Delphi 论文采用 80% 训练, 20% 验证)
random.shuffle(processed_data)
split_idx = int(len(processed_data) * 0.8)

train_data = processed_data[:split_idx]
val_data   = processed_data[split_idx:]

# 4. 持久化保存 (Serialization)
# 以后训练直接 load 这个文件，速度极快
torch.save(train_data, "./data_processed/train.pt")
torch.save(val_data,   "./data_processed/val.pt")

print(f"处理完成：训练集 {len(train_data)} 人, 验证集 {len(val_data)} 人")
