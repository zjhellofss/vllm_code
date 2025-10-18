# import time
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def matrix_multiply(A, B):
#     A_shape = A.shape
#     B_shape = B.shape
#     rows_A = A_shape[0]
#     cols_A = A_shape[1]
#     rows_B = B_shape[0]
#     cols_B = B_shape[1]
#     assert cols_A == rows_B
#     C = np.zeros((rows_A, cols_B))
#     for i in range(rows_A):
#         for j in range(cols_B):
#             for k in range(rows_B):
#                 C[i][j] += A[i][k] * B[k][j]
#     return C
#
#
# def matrix_multiply_blocked(A, B, BLOCK_SIZE):
#     M, K = A.shape
#     K, N = B.shape
#     C = np.zeros((M, N), dtype=np.float32)
#     for m in range(0, M, BLOCK_SIZE):
#         for n in range(0, N, BLOCK_SIZE):
#             acc = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
#             for k in range(0, K, BLOCK_SIZE):
#                 a = A[m: m + BLOCK_SIZE, k: k + BLOCK_SIZE]
#                 b = B[k: k + BLOCK_SIZE, n: n + BLOCK_SIZE]
#                 acc += np.dot(a, b)
#             C[m: m + BLOCK_SIZE, n: n + BLOCK_SIZE] = acc
#     return C
#
#
# def benchmark_matrix_multiplication(sizes, BLOCK_SIZE):
#     times_native = []
#     times_naive = []
#     times_blocked = []
#
#     for size in sizes:
#         print('size:{}'.format(size))
#         a = np.random.randn(size, size)
#         b = np.random.randn(size, size)
#
#         # Naive matrix multiplication
#         t1 = time.time()
#         c2 = matrix_multiply(a, b)
#         t2 = time.time()
#         times_naive.append(t2 - t1)
#
#         # Blocked matrix multiplication
#         t1 = time.time()
#         c3 = matrix_multiply_blocked(a, b, BLOCK_SIZE)
#         t2 = time.time()
#         times_blocked.append(t2 - t1)
#
#     return times_native, times_naive, times_blocked
#
#
# if __name__ == '__main__':
#     sizes = [64, 128, 256, 512, 1024]  # Different matrix sizes to benchmark
#     BLOCK_SIZE = 32
#
#     _, times_naive, times_blocked = benchmark_matrix_multiplication(sizes, BLOCK_SIZE)
#
#     # Plotting the results
#     plt.figure(figsize=(10, 6))
#     plt.plot(sizes, times_naive, label='Naive (matrix_multiply)', marker='o')
#     plt.plot(sizes, times_blocked, label='Blocked (matrix_multiply_blocked)', marker='o')
#
#     plt.xlabel('Matrix Size')
#     plt.ylabel('Time (seconds)')
#     plt.title('Matrix Multiplication Performance Comparison')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_linear_kernel_fwd(
        x_ptr,  # 输入数据矩阵首元素指针
        w_ptr,  # 权重矩阵首元素指针
        z_ptr,  # 输出结果地址
        M, N, K,  # Matrix dimensions
        BLOCK_SIZE_M: tl.constexpr = 128,  # 块大小
        BLOCK_SIZE_N: tl.constexpr = 128,
        BLOCK_SIZE_K: tl.constexpr = 64,
):
    # 对于每个triton block的二维坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 一个triton block的处理范围（在M,N轴上)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]  # 形状为 (1, BLOCK_SIZE_N)。

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 在k轴上进行一个分块归约
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        # tl.load()加载了一个块
        # 加载的是一个范围(offs_m，xk)

        # x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k k是现在处理块的一个起始地址，
        # 加一个范围表示当前在k轴上的处理范围
        # k等于8的时候，tl.arange(0, BLOCK_SIZE_K)[None, :] 表示0,4，两者相加就表示(8,12)的这么一个范围值

        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)
        x = x.to(tl.float16)

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        # tl.load加载的是(w_k,offs_n)
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
        w = w.to(tl.float16)
        # z += x@w
        z = tl.dot(x, w, acc=z)
    # 一个triton block计算的结果大小是block_m×block_n
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    tl.store(z_ptr + z_offset, z, mask=z_mask)


@torch.no_grad()
def fused_ffn(
        x,
        weight,
):

    out_shape_0 = x.shape[:-1]
    x = x.view((-1, x.shape[-1]))
    M, K = x.shape
    N = weight.shape[1]

    # Allocates output.
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # 2D launch kernel where each block gets its own program.
    # 配置网格中（二维），每个维度上的triton block数量
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    _fused_linear_kernel_fwd[grid](
        x,
        weight,
        z,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return z.view((*out_shape_0, N))


if __name__ == '__main__':
    batch_size = 64
    sequence_length = 128
    hidden_dim = 1280

    # 假设权重矩阵 weight 的形状为 [hidden_dim, output_dim]
    output_dim = 2560

    x = torch.randn((batch_size, sequence_length, hidden_dim), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)

    ## warm up
    for i in range(5):
        golden = x@weight
        output = fused_ffn(x, weight)
        x = torch.randn((batch_size, sequence_length, hidden_dim), device='cuda', dtype=torch.float16)
        weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)

    repeat_time = 5
    import time 
    times_torch = []
    times_triton = []
    for i in range(repeat_time):
        # 重新生成输入
        x = torch.randn((batch_size, sequence_length, hidden_dim), device='cuda', dtype=torch.float16)
        weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()

        t1 = time.time()
        output = fused_ffn(x, weight)
        torch.cuda.synchronize()
        t2 = time.time()
        print('triton time:{}'.format(t2 - t1))
        times_triton.append(t2 - t1)

        t1 = time.time()
        golden = x@weight
        torch.cuda.synchronize()
        t2 = time.time()
        times_torch.append(t2-t1)
        print('pytorch time:{}'.format(t2 - t1))

    import matplotlib.pyplot as plt

    # 将时间从秒转换为毫秒
    times_torch_ms = [t * 1000 for t in times_torch]
    times_triton_ms = [t * 1000 for t in times_triton]

    sizes = [i for i in range(repeat_time)]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_torch_ms, label='torch (matrix_multiply)', marker='o')
    plt.plot(sizes, times_triton_ms, label='triton (matrix_multiply)', marker='o')

    plt.xlabel('Run Index')
    plt.ylabel('Time (milliseconds)')
    plt.title('Matrix Multiplication Performance Comparison (Torch vs Triton)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('cc.png')