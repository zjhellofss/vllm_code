from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_DIR = "/root/vllm_learn/Meta-Llama-3.1-8B-Instruct-W8A8-Dynamic-Per-Token"

print("=" * 60)
print("Loading Quantized Model")
print("=" * 60)

print("\n📦 Loading model...")
# Reduce max_model_len to fit in available GPU memory
llm = LLM(
    MODEL_DIR,
    max_model_len=8192,           # Reduced from 131072 to fit in GPU memory
    gpu_memory_utilization=0.95,  # Increase GPU memory utilization
)

# Load tokenizer with BOS token
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

print("\n" + "=" * 60)
print("Generation Results")
print("=" * 60)

prompts = [
    "My name is",
    "The capital of France is",
    "Hello, I am",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n[Test {i}]")
    print(f"Prompt: {prompt}")
    print("-" * 40)
    output = llm.generate(prompt, sampling_params=sampling_params)
    print(f"Output: {prompt}{output[0].outputs[0].text}\n")

print("=" * 60)
print("✅ Inference completed successfully!")
print("=" * 60)
