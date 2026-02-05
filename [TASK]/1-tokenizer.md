目标是写出 对这个项目进行 tokenizer.py 的工具函数

写LLM 开始阶段的 tokenizer.py

- 文件命名为 tokenizer.py， 写在 ./src/ 下面

- 文件实现对 ./data_raw/processed_merged_data.xlsx 文件 诊断编码 进行编码的功能，并加入以下token :
"[PAD]": 0,       # 非信息填充 (用于Batch对齐，不参与计算)
"[NO_EVENT]": 1,  # 无事件填充 (用于时间校准，参与 Loss 计算)
"[END]": 2,       # 序列结束标记
"DEATH": 3,       # 死亡事件
"MALE": 4,        # 性别 (只作为输入 Context，不预测)
"FEMALE": 5       # 性别 (只作为输入 Context，不预测)
"[UNK]": 6

附加信息：
3. 如何在其他地方使用？
这样设计后，你的调用逻辑会非常清晰：
场景 A：你需要创建一个新的词表（第一次跑）
# 在 build_vocab.py 中
from tokenizer import MedicalTokenizer

tokenizer = MedicalTokenizer()
tokenizer.build_from_data(my_56_samples)  # 你自己写的构建函数
tokenizer.save_json("./vocab.json")       # 保存下来
场景 B：你需要训练或预测（日常使用）
# 在 train.py 或 inference.py 中
from tokenizer import MedicalTokenizer

# 一行代码完成加载，不用关心文件读取细节
tokenizer = MedicalTokenizer.from_json("./vocab.json") 

ids = tokenizer.encode(["E11", "I21"])
总结
Delphi-2M 作为一个基于 nanoGPT 修改的模型，其工程实现应遵循 Transformer 社区的标准规范。将 IO（文件读取）逻辑包含在 tokenizer.py 的类定义中，能确保你的模型在不同阶段（训练 vs 丹麦数据验证）始终使用严格一致的词表映射。



以下是你参考的实现，你可以更完善它，也可以保持简单：

"""
import json

class MedicalTokenizer:
    def __init__(self):
        # 1. 定义特殊 Token (保留位)
        # 根据论文 [1]，我们需要区分 "non-informative padding" 和 "no-event padding"
        self.special_tokens = {
            "[PAD]": 0,       # 非信息填充 (用于Batch对齐，不参与计算)
            "[NO_EVENT]": 1,  # 无事件填充 (用于时间校准，参与 Loss 计算)
            "[END]": 2,       # 序列结束标记
            "DEATH": 3,       # 死亡事件
            "MALE": 4,        # 性别 (只作为输入 Context，不预测)
            "FEMALE": 5       # 性别 (只作为输入 Context，不预测)
            "[UNK]":6 # 不预测
        }
        
        # 初始化映射表
        self.token2id = self.special_tokens.copy()
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.icd_start_id = len(self.special_tokens) # ICD 编码从这里开始
        
        # 记录哪些 token 不需要计算 Loss (Logits设为 -Inf) [2]
        # 注意：[NO_EVENT] 是需要计算 Loss 的 [3]
        self.ignore_loss_tokens = {"[PAD]", "MALE", "FEMALE"}

    def build_vocab_from_samples(self, patient_records):
        """
        patient_records: list of strings, e.g., ["E11", "I21", "J11", ...]
        """
        unique_icds = sorted(list(set(patient_records)))
        
        current_id = self.icd_start_id
        for code in unique_icds:
            if code not in self.token2id:
                self.token2id[code] = current_id
                self.id2token[current_id] = code
                current_id += 1
        
        print(f"词表构建完成。总大小: {len(self.token2id)}。包含 ICD 数量: {len(unique_icds)}")

    def encode(self, token_list):
        """将字符列表转化为 ID 列表"""
        ids = []
        for t in token_list:
            if t in self.token2id:
                ids.append(self.token2id[t])
            else:
                # 遇到未知编码时的处理 (通常在测试集出现)
                print(f"Warning: Unknown token '{t}', using [PAD]")
                ids.append(self.token2id["[PAD]"])
        return ids

    def decode(self, id_list):
        """将 ID 列表转回字符 (用于 Debug 可视化)"""
        return [self.id2token.get(i, "[UNK]") for i in id_list]

    def get_vocab_size(self):
        return len(self.token2id)

    def save_vocab(self, path="vocab.json"):
        with open(path, 'w') as f:
            json.dump(self.token2id, f, indent=2)

    def load_vocab(self, path="vocab.json"):
        with open(path, 'r') as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}

# --- 模拟你的使用场景 ---

# 1. 实例化
tokenizer = MedicalTokenizer()

# 2. 模拟你的 56 条原始 ICD 数据 (去重后的部分展示)
my_sample_icds = ["E11", "I10", "G43", "J11", "M54", "K52"] 

# 3. 构建词表
tokenizer.build_vocab_from_samples(my_sample_icds)

# 4. 准备一个完整的训练序列 (模拟论文 Figure 1d 的结构)
# 包含：性别 + 早期病史 + No-event填充 + 晚期病史 + END
input_sequence = [
    "MALE", 
    "E11", 
    "[NO_EVENT]", 
    "[NO_EVENT]", 
    "I10", 
    "[END]"
]

# 5. 编码
input_ids = tokenizer.encode(input_sequence)

# 6. 解码 (验证)
decoded = tokenizer.decode(input_ids)

print("-" * 30)
print(f"输入序列: {input_sequence}")
print(f"编码结果 (Tensor input): {input_ids}")
print(f"解码回看: {decoded}")
print("-" * 30)

# 7. 调试技巧：查看哪些 ID 是不需要算 Loss 的
ignore_ids = [tokenizer.token2id[t] for t in tokenizer.ignore_loss_tokens]
print(f"Loss Mask IDs (这些ID预测时权重为0): {ignore_ids}")
"""
