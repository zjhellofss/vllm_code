import torch
device = 'cuda'

# 为不同形状创建不同的图
shapes = [512, 1024, 2048]
graphs = {}
stream = torch.cuda.Stream()
for size in shapes:
    # 为每个形状分配独立的张量
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    z = torch.empty(size, size, device=device)

    # 创建独立的图
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(5):
            z = torch.mm(x, y) + torch.sin(x)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        z = torch.mm(x, y) + torch.sin(x)
    torch.cuda.synchronize()

    # 存储图及其对应的张量
    graphs[size] = {
        'graph': graph,
        'x': x,
        'y': y,
        'z': z
    }

# 使用时根据形状选择对应的图


def run_with_shape(size):
    g = graphs[size]
    g['x'].normal_()  # 更新数据
    g['graph'].replay()  # 重放对应的图
    return g['z']


run_with_shape(512)