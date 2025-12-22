import torch
import time

device = 'cuda'
N = 2

# 1. 初始状态：x 为全 1
x = torch.ones(N, N, device=device)
y = torch.ones(N, N, device=device)
z = torch.empty(N, N, device=device)

# 捕获图
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

for _ in range(3):
    temp = torch.mm(x, y)
torch.cuda.synchronize()  # 确保预热彻底完成


with torch.cuda.stream(stream):
    # 捕获：z = x * y (即 1 * 1)
    with torch.cuda.graph(graph):
        z = torch.mm(x, y)
torch.cuda.synchronize()

print("--- 初始捕获完成 ---")
print(f"初始 x 地址: {x.data_ptr()}, 初始 z 结果:\n{z}")

print("\n--- 场景 1: 原地修改内容 (x.fill_(2)) ---")
x.fill_(2.0)  # 地址没变，内容变了
graph.replay()
print(f"x 地址: {x.data_ptr()}, z 结果 (预期应该是 4):\n{z}")

print("\n--- 场景 2: 改变变量地址 (x = torch.full...) ---")
x = torch.full((N, N), 5.0, device=device)

graph.replay()

print(f"图重放后的 z 结果:")
print(z)
