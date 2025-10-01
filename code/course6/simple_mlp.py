from transformers import PretrainedConfig


class MLPConfig(PretrainedConfig):
    model_type = "llama"  # 自定义类型，vLLM 不支持也无所谓

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = 1024
        self.pad_token_id = -1


from transformers import PreTrainedModel
import torch.nn as nn


class MLPModel(PreTrainedModel):
    config_class = MLPConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.input_dim,
            padding_idx=config.pad_token_id,
        )
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = self.fc1(self.embed_tokens(x))
        x = self.relu(x)
        return self.fc2(x)


config = MLPConfig(input_dim=128, hidden_dim=256, output_dim=10)
model = MLPModel(config)

# 保存为 Hugging Face 格式
model.save_pretrained("/home/test_fss/code/vllm_learn/code/course6/mlp_model")

from transformers import AutoTokenizer

# 用个最小的 tokenizer 占位
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("/home/test_fss/code/vllm_learn/code/course6/mlp_model")
