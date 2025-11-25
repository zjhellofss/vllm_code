import numpy as np

# ==========================================
# 0. 初始化
# ==========================================
np.random.seed(42)
Vocab = 10
Hidden = 4
B, S = 2, 2

# 完整的 Embedding 表 (模拟 Ground Truth)
# 形状: (10, 4)
E_full = np.random.randn(Vocab, Hidden)

# 输入 Token IDs (Batch=2, Seq=2)
# 包含落在两个 GPU 范围内的 ID
# 1, 3 -> GPU 1 (0-4)
# 6, 8 -> GPU 2 (5-9)
Input_IDs = np.array([
    [1, 6], 
    [3, 8]
])

print("输入 IDs:\n", Input_IDs)

# ==========================================
# 1. 单机基准 (Lookup)
# ==========================================
# Numpy 的高级索引模拟 Embedding Lookup
Output_Ref = E_full[Input_IDs] 
print(f"\n基准输出形状: {Output_Ref.shape}") # (2, 2, 4)


# ==========================================
# 2. 并行模拟 (Parallel Embedding)
# ==========================================
print("\n--- 开始并行模拟 ---")

# [切分权重] 按词汇维度切分 (Row Parallel in terms of Matrix, 
# 但通常称为 Vocab Parallel)
# Split Size = 5
V_per_gpu = Vocab // 2

# GPU 1: 负责 ID 0-4
E_gpu1 = E_full[:V_per_gpu, :] # (5, 4)
range_start_1 = 0
range_end_1 = V_per_gpu

# GPU 2: 负责 ID 5-9
E_gpu2 = E_full[V_per_gpu:, :] # (5, 4)
range_start_2 = V_per_gpu
range_end_2 = Vocab

def parallel_embedding_forward(input_ids, local_weight, start_idx, end_idx):
    """
    每个 GPU 独立执行的 Forward 函数
    """
    # 1. 创建掩码: 找出哪些 ID 属于当前 GPU
    mask = (input_ids >= start_idx) & (input_ids < end_idx)
    
    # 2. 将全局 ID 映射为本地 ID (Offset)
    # 例如 GPU 2 负责 5-9，ID=6 对应的本地索引是 1
    local_ids = input_ids - start_idx
    
    # 3. 为了避免索引越界，将不属于自己的 ID 置为 0 
    safe_ids = np.where(mask, local_ids, 0)
    
    # 4. 查表 (Lookup)
    local_output = local_weight[safe_ids]
    
    mask_expanded = mask[:, :, np.newaxis]
    
    # 5. 只有属于自己的 ID 保留 Lookup 结果，其他的变成 0.0
    final_local_output = local_output * mask_expanded
    
    return final_local_output

# --- GPU 1 计算 ---
Out_gpu1 = parallel_embedding_forward(Input_IDs, E_gpu1, range_start_1, range_end_1)
print("\nGPU 1 输出 (部分为0):")
print(Out_gpu1[0]) # 看第一行: [Vector(ID=1), Vector(0.0)]

# --- GPU 2 计算 ---
Out_gpu2 = parallel_embedding_forward(Input_IDs, E_gpu2, range_start_2, range_end_2)
print("\nGPU 2 输出 (部分为0):")
print(Out_gpu2[0]) # 看第一行: [Vector(0.0), Vector(ID=6)]


# ==========================================
# 3. 同步聚合 (All-Reduce)
# ==========================================
Output_Fused = Out_gpu1 + Out_gpu2

print(f"\n验证结果: {np.allclose(Output_Ref, Output_Fused)}")
print("逻辑: Embedding(ID) = Embedding_GPU1(ID) + Embedding_GPU2(ID)")
print("      其中一个必然是 0向量，另一个是真实向量")