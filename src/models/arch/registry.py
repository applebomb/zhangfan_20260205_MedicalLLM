from .gpt2_v1 import MedicalGPT2ModelV1

def get_model(config):
    """根据配置中的 model.type 实例化对应的模型"""
    model_type = config.model.type
    
    if model_type == "gpt2_v1":
        return MedicalGPT2ModelV1(config.model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
