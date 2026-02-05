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

import json
import os

class MedicalTokenizer:
    def __init__(self):
        # 1. 定义特殊 Token (保留位)
        # 根据需求，区分 "non-informative padding" 和 "no-event padding"
        self.special_tokens = {
            "[PAD]": 0,       # 非信息填充 (用于Batch对齐，不参与计算)
            "[NO_EVENT]": 1,  # 无事件填充 (用于时间校准，参与 Loss 计算)
            "[END]": 2,       # 序列结束标记
            "DEATH": 3,       # 死亡事件
            "MALE": 4,        # 性别 (只作为输入 Context，不预测)
            "FEMALE": 5,      # 性别 (只作为输入 Context，不预测)
            "[UNK]": 6        # 未知标记
        }
        
        # 初始化映射表
        self.token2id = self.special_tokens.copy()
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.icd_start_id = len(self.special_tokens) # ICD 编码从这里开始
        
        # 记录哪些 token 不需要计算 Loss
        # 注意：[NO_EVENT] 是需要计算 Loss 的
        self.ignore_loss_tokens = {"[PAD]", "MALE", "FEMALE"}

    def build_vocab_from_samples(self, icd_codes):
        """
        icd_codes: list of strings, e.g., ["E11", "I21", "J11", ...]
        """
        unique_icds = sorted(list(set(icd_codes)))
        
        current_id = len(self.token2id)
        for code in unique_icds:
            if code not in self.token2id:
                self.token2id[code] = current_id
                self.id2token[current_id] = code
                current_id += 1
        
        print(f"词表构建完成。总大小: {len(self.token2id)}。包含 ICD 数量: {len(unique_icds)}")

    def build_from_data(self, icd_codes):
        """Alias for build_vocab_from_samples"""
        return self.build_vocab_from_samples(icd_codes)

    def encode(self, token_list):
        """将字符列表转化为 ID 列表"""
        ids = []
        for t in token_list:
            if t in self.token2id:
                ids.append(self.token2id[t])
            else:
                # 遇到未知编码时使用 [UNK]
                ids.append(self.token2id["[UNK]"])
        return ids

    def decode(self, id_list):
        """将 ID 列表转回字符"""
        return [self.id2token.get(i, "[UNK]") for i in id_list]

    def get_vocab_size(self):
        return len(self.token2id)

    def save_json(self, path="vocab.json"):
        """保存词表为 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, indent=2, ensure_ascii=False)
        print(f"词表已保存至: {path}")

    def save_vocab(self, path="vocab.json"):
        """Alias for save_json"""
        return self.save_json(path)

    def load_vocab(self, path="vocab.json"):
        """从 JSON 文件加载词表到当前实例"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found at {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
            
        self.token2id = token2id
        self.id2token = {int(v): k for k, v in token2id.items()}

    @classmethod
    def from_json(cls, path="vocab.json"):
        """从 JSON 文件加载词表并返回新实例"""
        tokenizer = cls()
        tokenizer.load_vocab(path)
        return tokenizer

    def get_ignore_loss_ids(self):
        """获取需要忽略 Loss 的 Token ID 列表"""
        return [self.token2id[t] for t in self.ignore_loss_tokens if t in self.token2id]

if __name__ == "__main__":
    # 简单的测试脚本
    tokenizer = MedicalTokenizer()
    
    # 模拟从 Excel 读取的数据
    sample_icds = ["E11.9", "I10.x00", "J44.900", "E11.9", "M54.500"]
    tokenizer.build_vocab_from_samples(sample_icds)
    
    test_seq = ["MALE", "E11.9", "[NO_EVENT]", "I10.x00", "[END]"]
    encoded = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(encoded)
    
    print(f"输入: {test_seq}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    print(f"词表大小: {tokenizer.get_vocab_size()}")
    
    # 测试保存和加载
    vocab_path = "vocab_test.json"
    tokenizer.save_json(vocab_path)
    new_tokenizer = MedicalTokenizer.from_json(vocab_path)
    print(f"加载后词表大小: {new_tokenizer.get_vocab_size()}")
    
    # 清理测试文件
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
