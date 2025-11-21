# split_multiproc_demo.py
# 使用 multiprocessing 启动多个进程模拟 DP ranks（CPU 可运行）

from multiprocessing import Process
from time import sleep

def worker(prompts, dp_size, global_dp_rank):
    """子进程函数：按给定公式取出当前 rank 的 prompts 并打印"""
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    chunk = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(chunk) == 0:
        chunk = ["Placeholder"]
    print(f"[PID {__import__('os').getpid()}] DP rank {global_dp_rank} -> {len(chunk)} prompts: {chunk}", flush=True)

if __name__ == "__main__":
    # 示例数据
    prompts = [f"prompt_{i}" for i in range(7)]  # 7 条 prompt，故会产生 remainder 情况
    dp_size = 4  # 分成 4 个 rank（会出现某些 rank 被分到 0 条）
    procs = []

    # 在单节点模拟启动 dp_size 个进程，每个进程被视作一个 global_dp_rank
    for rank in range(dp_size):
        p = Process(target=worker, args=(prompts, dp_size, rank))
        p.start()
        procs.append(p)

    # 等待所有子进程结束，单个子进程最长等 5 秒
    timeout_seconds = 5
    for p in procs:
        p.join(timeout=timeout_seconds)
        if p.exitcode is None:
            print(f"Process {p.pid} didn't exit within {timeout_seconds}s, killing it.")
            p.kill()