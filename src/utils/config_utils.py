import yaml
from types import SimpleNamespace

def load_config(config_path):
    """加载 YAML 配置文件并转换为对象形式"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 将字典转换为可通过 . 访问的对象 (递归)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(config_dict)
