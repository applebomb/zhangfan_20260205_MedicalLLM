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

任务是写出第一个 基础模型
注意，不要改动除你任务以外的任何文件，包括删除、修改，专心做好自己的任务


- 基础模型是 gpt2
- 模型层数和 架构、embedding宽度、maxlen ....  由里面config 变量 方便调整
- 目标是 跑通预测流程，所以 2层 2个head .... 就够了，一切从简
- 训练和测试数据集在 ./data_processed/train.pt ./data_processed/val.pt 
- 词表是 ./data_processed/vocab.json, 处理词表 的 文件在./src/tokenizer.py

- 第一版主要是跑通 gpt2 的架构， 所以 对于 train.pt 里面的 ages 可以忽略
使用 input_ids 来预测 下一个 发生的 疾病， 最后结果 需要 用 tokenizer.py 以及 vocab.json 进行解码

- 每轮输出 loss, 并评价loss 实际错误发生率

- 要观察loss 是否在下降，你看看是不是 使用 summery 写log

注意，不要改动除你任务以外的任何文件，包括删除、修改，专心做好自己的任务

将每段代码按逻辑分段 处理好，加上注释；

- ./src/train_model.py 作为入口点
- ./src/train.py 作为训练模型的模块
- ./src/dataset.py 用来处理数据送入
- ./src/eval.py 作为验证输出
- ./src/model_base_202602_ver1.py 为模型