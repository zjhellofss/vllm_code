import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    X_ptr,   # 输入张量 X 的指针
    Y_ptr,   # 输入张量 Y 的指针
    Z_ptr,   # 输出张量 Z 的指针
    N,       # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 块大小，编译时常量
):
    # 1. 计算当前 block 的起始位置和线程索引
    pid = tl.program_id(0)                    # 当前 block ID
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N                        # 防止越界访问

    # 2. 加载数据
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # 3. 执行向量加法
    z = x + y

    # 4. 存储结果
    tl.store(Z_ptr + offsets, z, mask=mask)


def vector_add_torch(x: torch.Tensor, y: torch.Tensor):
    """
    使用 Triton 实现的向量加法 (Z = X + Y)
    支持任意大小的 1D 张量
    """
    assert x.is_cuda and y.is_cuda, "输入张量必须在 GPU 上"
    assert x.shape == y.shape, "输入张量形状必须相同"
    N = x.numel()  # 元素总数
    assert N > 0, "张量不能为空"

    # 输出张量
    z = torch.empty_like(x)

    # 定义块大小（通常为 2 的幂，如 1024）
    BLOCK_SIZE = 1024

    # 计算需要多少个 block
    num_blocks = triton.cdiv(N, BLOCK_SIZE)  # ceil(N / BLOCK_SIZE)

    # 启动 kernel
    vector_add_kernel[(num_blocks,)](
        X_ptr=x,
        Y_ptr=y,
        Z_ptr=z,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return z


# =========================
# 示例：测试函数
# =========================
if __name__ == "__main__":
    # 创建测试数据（在 GPU 上）
    a = torch.randn(131072, device='cuda')
    b = torch.randn(131072, device='cuda')

    # 调用自定义 Triton kernel
    c_triton = vector_add_torch(a, b)

    # 对比 PyTorch 原生结果
    c_torch = a + b

    # 验证是否一致
    print("Max diff:", torch.max(torch.abs(c_triton - c_torch)))
    print("Correct:", torch.allclose(c_triton, c_torch, atol=1e-6))
