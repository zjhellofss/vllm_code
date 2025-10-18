import torch
import triton
import triton.language as tl


# ================================
# Baseline: Naive PyTorch
# ================================
def naive_softmax(x):
    x_max = x.max(dim=1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


# ================================
# Triton V1: One Row Per Program
# ================================
@triton.jit
def softmax_kernel_v1(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax_v1(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel_v1[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y



if __name__ == "__main__":
    torch.manual_seed(42)

    shapes = [
        (8, 512),
        (3, 123),
        (17, 64),
        (1000, 256),  # 大 batch 测试 grid-stride 优势
        (2048, 2048),
    ]

    for shape in shapes:
        print(f"\n{'='*60}")
        print(f"Testing shape: {shape}")
        print('='*60)

        x = torch.randn(*shape, device='cuda')

        y_torch = naive_softmax(x)
        y_triton_v1 = triton_softmax_v1(x)

        y_builtin = torch.softmax(x, dim=1)

        atol = 1e-6
        print("✅ Correctness Check:")
        print(f"  Torch vs Builtin:      {torch.allclose(y_torch, y_builtin, atol=atol)}")
        print(f"  Triton V1 vs Builtin:  {torch.allclose(y_triton_v1, y_builtin, atol=atol)}")

        sample_row = 0
        print(f"\n>>> Sample Output Row {sample_row} (first 5 elements):")
        print(f"  Torch:   {y_torch[sample_row, :5].cpu().numpy()}")
        print(f"  TritonV1:{y_triton_v1[sample_row, :5].cpu().numpy()}")
    try:
        from triton.testing import do_bench
        print(f"\n{'='*60}")
        print("⏱️  Performance Benchmark (shape=1024x1024)")
        print('='*60)

        x_large = torch.randn(8192, 8192, device='cuda')

        ms_torch = do_bench(lambda: naive_softmax(x_large))
        ms_v1 = do_bench(lambda: triton_softmax_v1(x_large))

        print(f"PyTorch:   {ms_torch:.3f} ms")
        print(f"Triton V1: {ms_v1:.3f} ms")
        print(f"Speedup V1 over Torch: {ms_torch/ms_v1:.2f}x")

    except Exception as e:
        print(f"Benchmark skipped: {e}")
