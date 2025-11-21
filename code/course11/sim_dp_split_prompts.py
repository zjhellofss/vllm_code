def get_chunk(prompts, dp_size, global_dp_rank):
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    chunk = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(chunk) == 0:
        chunk = ["Placeholder"]
    return chunk

if __name__ == "__main__":
    prompts = [f"prompt_{i}" for i in range(10)]  # 10 条示例 prompt
    dp_size = 4

    print(f"All prompts ({len(prompts)}): {prompts}\n")
    for rank in range(dp_size):
        assigned = get_chunk(prompts, dp_size, rank)
        print(f"rank {rank} -> assigned {len(assigned)} prompts: {assigned}")