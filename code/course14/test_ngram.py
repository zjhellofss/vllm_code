# from vllm import LLM, SamplingParams

# prompts = [
#     "The future of AI is",
# ]
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(
#     model="/root/vllm_learn/qwen3",
#     speculative_config={
#         "method": "ngram",
#         "num_speculative_tokens": 5,
#         "prompt_lookup_max": 4,
#     },
# )
# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



import os
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

prompts = [
    "Hello, my name is",
    "The president of the The president United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config = {
    "method": "ngram",
    "prompt_lookup_max": 5,      # 在历史上下文中查找匹配时，N-Gram 的最小长度
    "prompt_lookup_min": 3,      # 限制 N-Gram 查找的最大长度
    "num_speculative_tokens": 3, # 在每个解码步骤为每个请求动态推测 3 个令牌
}

llm = LLM(
    model="/root/vllm_learn/qwen3",
    speculative_config=speculative_config,enforce_eager=True
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")