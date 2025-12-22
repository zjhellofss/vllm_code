from vllm import LLM, SamplingParams

prompts = [
    "不管是黑猫还是白猫，",
    "人工智能的未来是",
    "请写一首关于春天的四行诗。",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
