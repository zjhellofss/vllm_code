import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig, Qwen2Config, Qwen2ForCausalLM # 假设是Qwen2，如果是Qwen3请保持原样
# 如果您的环境确实有 Qwen3，请使用您原来的导入：
# from transformers import AutoTokenizer, AutoConfig, PretrainedConfig, Qwen3Config, Qwen3ForCausalLM
from huggingface_hub import hf_hub_download
import os

# 为了代码能跑通，这里做一个别名处理，如果您确实有Qwen3Config请忽略这几行
try:
    Qwen3Config
except NameError:
    # 假设使用 Qwen2 作为替代（因为目前 HuggingFace 主要支持 Qwen2）
    print("Warning: Qwen3Config not found, aliasing Qwen2Config for demonstration.")
    from transformers import Qwen2Config as Qwen3Config
    from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM

class MedusaConfig(PretrainedConfig):
    """Medusa模型的配置类，用于定义模型结构参数"""
    model_type = "medusa"

    def __init__(self, base_model_name_or_path="Qwen/Qwen2-1.5B", medusa_num_heads=2, medusa_num_layers=1, **kwargs):
        # 基础模型路径
        self.base_model_name_or_path = base_model_name_or_path
        # Medusa头数量（预测头数量）
        self.medusa_num_heads = medusa_num_heads
        # Medusa层数（每个预测头的层数）
        self.medusa_num_layers = medusa_num_layers
        super().__init__(**kwargs)

class ResBlock(nn.Module):
    """残差块模块，包含线性层和SiLU激活函数"""
    def __init__(self, hidden_size):
        super().__init__()
        # 线性变换层，权重初始化为0
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        # SiLU激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        # 残差连接：输入 + 激活(线性变换(输入))
        return x + self.act(self.linear(x))

class MedusaModelQwen3(Qwen3ForCausalLM):
    """基于Qwen3的Medusa模型，用于加速推理的多头预测模型"""
    def __init__(self, config):
        # 1. 修复：必须显式调用父类初始化，且不能写在注释行里
        super().__init__(config)
        
        # 从配置中获取Medusa参数
        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        
        # 基础模型路径
        base_model_name_or_path = getattr(config, "base_model_name_or_path", "Qwen/Qwen2-1.5B")
        
        # 模型维度参数
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Medusa配置
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        
        # 加载tokenizer (通常建议在模型外部加载，但保持您的逻辑)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        
        # 构建Medusa多头预测层
        # 每个头包含：多个残差块 + 线性输出层
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),  # 多层残差块
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),  # 词汇表输出层
                )
                for _ in range(medusa_num_heads)  # 创建多个预测头
            ]
        )

    @property
    def base_model(self):
        """返回基础模型（自身）"""
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """从预训练模型加载，支持加载Medusa头权重"""
        try:
            # 尝试直接加载配置
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs, config=config
            )
        except:
            # 如果失败，使用Medusa配置加载
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            # 加载基础模型配置
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = config.medusa_num_heads
            base_model_config.medusa_num_layers = config.medusa_num_layers
            
            # 加载基础模型
            model = super().from_pretrained(
                config.base_model_name_or_path, *args, **kwargs, config=base_model_config
            )
            
            # 加载Medusa头权重
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            
            # 2. 修复：else 语句和逻辑分行
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path  # 本地文件
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")  # 从Hub下载
            
            # 加载权重到Medusa头
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model

    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """前向传播"""
        if not medusa_forward:
            # 普通模式：使用基础模型的前向传播
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        
        # Medusa模式：使用推理模式加速
        with torch.inference_mode():
            # 获取基础模型的隐藏状态
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            
            # 如果需要输出原始logits
            orig = None
            if output_orig:
                orig = self.lm_head(outputs[0])  # 基础模型的输出层
            
            # 复制隐藏状态用于Medusa头计算
            hidden_states = outputs[0].clone()
            
            # 计算所有Medusa头的logits
            medusa_logits = []
            for i in range(self.medusa):
                medusa_logits.append(self.medusa_head[i](hidden_states))
            
            # 3. 修复：return 语句分行
            if output_orig:
                # 返回：Medusa logits, 模型输出, 原始logits
                return torch.stack(medusa_logits, dim=0), outputs, orig
            
            # 返回：Medusa logits
            return torch.stack(medusa_logits, dim=0)

# 测试用例
if __name__ == "__main__":
    # 自定义模型权重名称或者路径
    # 注意：这里使用 Qwen2-1.5B 作为示例，因为 Qwen3 可能不存在
    base_model_name_or_path = "Qwen/Qwen2-1.5B-Instruct" 
    
    try:
        # 加载基础配置并设置Medusa参数
        base_config = Qwen3Config.from_pretrained(base_model_name_or_path)
        base_config.medusa_num_heads = 4  # 设置4个预测头
        base_config.medusa_num_layers = 1  # 每个头1层
        
        # 初始化模型
        model = MedusaModelQwen3(base_config)
        model.eval()  # 设置为评估模式
        
        # 获取tokenizer
        tokenizer = model.get_tokenizer()
        
        # 测试输入
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # 普通前向传播测试
        outputs_normal = model(**inputs)
        print("Normal output logits shape:", outputs_normal.logits.shape)
        
        # Medusa前向传播测试
        medusa_logits = model(**inputs, medusa_forward=True)
        print("Medusa logits shape:", medusa_logits.shape)
    except Exception as e:
        print(f"Run failed: {e}")
        print("Tip: Ensure you have access to the model on HuggingFace Hub and transformers installed.")