import pandas as pd
import os
from tokenizer import MedicalTokenizer

def build_medical_vocab():
    # 路径设置
    data_path = os.path.join("data_raw", "processed_merged_data.xlsx")
    vocab_save_path = os.path.join("data_processed", "vocab.json")
    
    # 确保目标目录存在
    os.makedirs("data_processed", exist_ok=True)
    
    print(f"正在读取数据: {data_path}")
    # 读取 Excel 文件
    df = pd.read_excel(data_path)
    
    # 提取 "诊断编码" 列，去掉空值并转为字符串
    if "诊断编码" not in df.columns:
        print(f"错误: Excel 文件中未找到 '诊断编码' 列。可用列: {df.columns.tolist()}")
        return
    
    icd_codes = df["诊断编码"].dropna().astype(str).unique().tolist()
    
    print(f"提取到 {len(icd_codes)} 个唯一的 ICD 编码。")
    
    # 实例化并构建词表
    tokenizer = MedicalTokenizer()
    tokenizer.build_from_data(icd_codes)
    
    # 保存词表
    tokenizer.save_json(vocab_save_path)
    print(f"词表构建完成并保存至: {vocab_save_path}")

if __name__ == "__main__":
    build_medical_vocab()
