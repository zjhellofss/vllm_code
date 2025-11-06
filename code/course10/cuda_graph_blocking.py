import torch
import time

assert torch.cuda.is_available(), "CUDA not available"
device = 'cuda'
N = 1024


x_host = torch.randn(N, N)  
y_host = torch.randn(N, N)

x_cuda = torch.empty(N, N, device=device)
y_cuda = torch.empty(N, N, device=device)
z_cuda = torch.empty(N, N, device=device)

stream = torch.cuda.Stream()
graph = torch.cuda.CUDAGraph()


with torch.cuda.stream(stream):
    for _ in range(5):
        x_cuda.copy_(x_host)  
        y_cuda.copy_(y_host)
        z_cuda = torch.mm(x_cuda, y_cuda) + torch.sin(x_cuda)
torch.cuda.synchronize()

with torch.cuda.graph(graph, stream=stream):
  x_cuda.copy_(x_host)  
  y_cuda.copy_(y_host)
  z_cuda = torch.mm(x_cuda, y_cuda) + torch.sin(x_cuda)



