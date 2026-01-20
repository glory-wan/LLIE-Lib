# 应用到你的BaseModel
import torch.nn as nn


class BaseModel(nn.Module):
    """基础模型类"""
    _registry = {}  # 模型注册表

    def __init_subclass__(cls, **kwargs):
        """自动注册所有子类"""
        super().__init_subclass__(**kwargs)
        BaseModel._registry[cls.__name__] = cls

    @classmethod
    def get_model_class(cls, model_name):
        """根据名称获取模型类"""
        return cls._registry.get(model_name, cls)

    @classmethod
    def create_model(cls, model_name, config):
        """创建模型实例"""
        model_class = cls.get_model_class(model_name)
        return model_class(config)


class ZeroDCE(BaseModel):
    """Zero-DCE模型"""

    def __init__(self, config):
        super().__init__()
        print(f"创建ZeroDCE模型，配置: {config}")


class SCI(BaseModel):
    """SCI模型"""

    def __init__(self, config):
        super().__init__()
        print(f"创建SCI模型，配置: {config}")


# 查看所有注册的模型
print("注册的模型:", list(BaseModel._registry.keys()))
# 输出: 注册的模型: ['ZeroDCE', 'SCI']

# 通过名称创建模型
config = {'device': 'cuda'}
model1 = BaseModel.create_model('ZeroDCE', config)
model2 = BaseModel.create_model('SCI', config)


# 动态添加新模型
class NewModel(BaseModel):
    """新模型（自动注册）"""

    def __init__(self, config):
        super().__init__()
        print(f"创建NewModel，配置: {config}")


print("更新后的注册表:", list(BaseModel._registry.keys()))
# 输出: 注册的模型: ['ZeroDCE', 'SCI', 'NewModel']