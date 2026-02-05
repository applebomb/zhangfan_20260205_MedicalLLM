import torch
import pandas as pd
import os
import argparse
from src.tokenizer import MedicalTokenizer

def main():
    parser = argparse.ArgumentParser(description="Medical Data Preprocessing")
    parser.add_argument("--version", type=str, default="v1", help="Data version (e.g., v1)")
    args = parser.parse_args()

    # 路径定义
    raw_dir = f"data/{args.version}/raw"
    processed_dir = f"data/{args.version}/processed"
    os.makedirs(processed_dir, exist_ok=True)

    vocab_path = os.path.join(processed_dir, "vocab.json")
    data_path = os.path.join(raw_dir, "processed_merged_data.xlsx")

    # 1. 加载 Tokenizer
    # 如果 vocab 不存在，先从 raw 数据构建 (简单逻辑)
    if not os.path.exists(vocab_path):
        print(f"Vocab not found at {vocab_path}, building from {data_path}...")
        df_raw = pd.read_excel(data_path)
        tokenizer = MedicalTokenizer()
        tokenizer.build_vocab_from_samples(df_raw["诊断编码"].astype(str).tolist())
        tokenizer.save_json(vocab_path)
    else:
        tokenizer = MedicalTokenizer.from_json(vocab_path)

    # 2. 读取原始数据
    df = pd.read_excel(data_path)
    df = df.sort_values(by="就诊天数")

    # 提取代码和年龄
    codes = df["诊断编码"].astype(str).tolist()
    ages = df["就诊天数"].astype(float).tolist()

    # 转化为 ID
    input_ids = tokenizer.encode(codes)

    processed_data = [{
        "id": "P001",
        "input_ids": input_ids,
        "ages": ages
    }]

    # 3. 构造训练和测试数据集
    train_data = processed_data
    val_data = processed_data

    # 4. 持久化保存
    torch.save(train_data, os.path.join(processed_dir, "train.pt"))
    torch.save(val_data,   os.path.join(processed_dir, "val.pt"))

    print(f"Preprocessing complete for version {args.version}.")
    print(f"Saved to {processed_dir}")

if __name__ == "__main__":
    main()