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

from .gpt2_v1 import MedicalGPT2ModelV1
from .gpt2_time_v2 import MedicalGPT2ModelV2

def get_model(config):
    """根据配置中的 model.type 实例化对应的模型"""
    model_type = config.model.type
    
    if model_type == "gpt2_v1":
        return MedicalGPT2ModelV1(config.model)
    elif model_type == "gpt2_time_v2":
        return MedicalGPT2ModelV2(config.model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
