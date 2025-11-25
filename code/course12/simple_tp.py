import numpy as np

def gelu(x):
    """GeLU 激活函数 (使用 tanh 近似)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# 1. 初始化参数
np.random.seed(42)
B = 4   # Batch size
H = 8   # Hidden dimension (输入维度)
D = 4   # Output dimension (输出维度)

# 创建输入矩阵 X (B, H) 和 权重矩阵 A (H, D)
X = np.random.randn(B, H)
A = np.random.randn(H, D)

# ==========================================
# 方法 1: 单机基准计算 (不拆分)
# ==========================================
# 直接计算 Y = GeLU(X @ A)
target_output = gelu(np.dot(X, A))
print(f"基准输出形状: {target_output.shape}")


# ==========================================
# 方法 2: 模拟行拆分并行 (Row Parallelism)
# ==========================================
print("\n--- 开始模拟行拆分并行 ---")

# 假设我们有 2 个设备 (GPU)，将 A 按行切分，将 X 按列切分
# split_size = H // 2

# [切分权重矩阵 A] -> A1, A2
# A 的形状是 (H, D)，按行切分 (axis=0)
A1 = A[:H//2, :]
A2 = A[H//2:, :]
print(f"设备1 权重 A1 形状: {A1.shape}")
print(f"设备2 权重 A2 形状: {A2.shape}")

# [切分输入矩阵 X] -> X1, X2
# X 的形状是 (B, H)，为了配合 A 的行切分，X 需要按列切分 (axis=1)
X1 = X[:, :H//2]
X2 = X[:, H//2:]
print(f"设备1 输入 X1 形状: {X1.shape}")
print(f"设备2 输入 X2 形状: {X2.shape}")

# [并行计算]
# 每个设备独立计算自己的部分积: Xi * Ai
# 结果形状均为 (B, D)
Y1_partial = np.dot(X1, A1) 
Y2_partial = np.dot(X2, A2)

# [同步点 / All-Reduce]
# 在 GeLU 之前，必须将各设备的部分结果相加
# 对应公式: XA = X1A1 + X2A2
Y_combined = Y1_partial + Y2_partial

# [应用非线性激活函数]
# 聚合后才能执行 GeLU
parallel_output = gelu(Y_combined)

# ==========================================
# 验证结果
# ==========================================
# 检查并行计算结果与基准结果是否一致
is_close = np.allclose(target_output, parallel_output)
print(f"\n结果验证: {'成功' if is_close else '失败'}")
print(f"两者误差 (Max Diff): {np.max(np.abs(target_output - parallel_output))}")

# 演示如果直接在局部做 GeLU 再相加是错误的 (数学原理验证)
wrong_output = gelu(Y1_partial) + gelu(Y2_partial)
print(f"\n错误做法 (先 GeLU 后聚合) 误差: {np.max(np.abs(target_output - wrong_output))}")
print("结论: GeLU(X1A1 + X2A2) != GeLU(X1A1) + GeLU(X2A2)")