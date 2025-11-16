from vllm import LLM
import torch

model_id = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitblas"
)