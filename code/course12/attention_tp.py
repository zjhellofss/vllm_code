import numpy as np

if __name__ == '__main__':
    # 参数设置
    bsz = 4
    seq_len = 16
    hidden_dim = 128
    num_heads = 8
    head_dim = hidden_dim // num_heads

    np.random.seed(42)  # 设置随机种子，保证结果可复现

    # 随机初始化权重矩阵
    wq_weight = np.random.randn(hidden_dim, hidden_dim)
    wk_weight = np.random.randn(hidden_dim, hidden_dim)
    wv_weight = np.random.randn(hidden_dim, hidden_dim)
    wo_weight = np.random.randn(hidden_dim, hidden_dim)

    inputs = np.random.randn(bsz, seq_len, hidden_dim)

    print("===== 标准版本的多头注意力计算 =====")
    # 1. 标准版本 - 不分割计算
    q = np.matmul(inputs, wq_weight)  # [bsz, seq_len, hidden_dim]
    k = np.matmul(inputs, wk_weight)  # [bsz, seq_len, hidden_dim]
    v = np.matmul(inputs, wv_weight)  # [bsz, seq_len, hidden_dim]

    # 重塑为多头形式
    q = q.reshape(bsz, seq_len, num_heads, head_dim)  # [bsz, seq_len, num_heads, head_dim]
    k = k.reshape(bsz, seq_len, num_heads, head_dim)  # [bsz, seq_len, num_heads, head_dim]
    v = v.reshape(bsz, seq_len, num_heads, head_dim)  # [bsz, seq_len, num_heads, head_dim]

    # 调整维度顺序
    q = np.transpose(q, (0, 2, 1, 3))  # [bsz, num_heads, seq_len, head_dim]
    k = np.transpose(k, (0, 2, 1, 3))  # [bsz, num_heads, seq_len, head_dim]
    v = np.transpose(v, (0, 2, 1, 3))  # [bsz, num_heads, seq_len, head_dim]

    # 注意力分数计算
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(head_dim)  # [bsz, num_heads, seq_len, seq_len]

    # 应用softmax
    attn_probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_probs = attn_probs / np.sum(attn_probs, axis=-1, keepdims=True)  # [bsz, num_heads, seq_len, seq_len]

    # 注意力输出计算
    attn_output = np.matmul(attn_probs, v)  # [bsz, num_heads, seq_len, head_dim]

    # 恢复原始维度
    attn_output = np.transpose(attn_output, (0, 2, 1, 3))  # [bsz, seq_len, num_heads, head_dim]
    attn_output = attn_output.reshape(bsz, seq_len, hidden_dim)  # [bsz, seq_len, hidden_dim]

    # 最终输出投影
    output = np.matmul(attn_output, wo_weight)  # [bsz, seq_len, hidden_dim]

    print("===== 张量并行版本的多头注意力计算 =====")
    # 2. 张量并行版本 - 按头切分
    # 每个并行组处理一半的头
    heads_per_gpu = num_heads // 2

    # GPU 1处理前half_heads个头
    # 按列切分QKV权重 - 每个GPU负责一半头的权重
    wq_sub1 = wq_weight[:, :hidden_dim // 2]  # 前half_heads个头的权重
    wk_sub1 = wk_weight[:, :hidden_dim // 2]
    wv_sub1 = wv_weight[:, :hidden_dim // 2]
    wo_sub1 = wo_weight[:hidden_dim // 2, :]
    # GPU 1上的计算
    q1 = np.matmul(inputs, wq_sub1)  # [bsz, seq_len, hidden_dim//2]
    k1 = np.matmul(inputs, wk_sub1)
    v1 = np.matmul(inputs, wv_sub1)

    # 重塑为多头形式
    q1 = q1.reshape(bsz, seq_len, heads_per_gpu, head_dim)
    k1 = k1.reshape(bsz, seq_len, heads_per_gpu, head_dim)
    v1 = v1.reshape(bsz, seq_len, heads_per_gpu, head_dim)

    # 调整维度顺序
    q1 = np.transpose(q1, (0, 2, 1, 3))  # [bsz, heads_per_gpu, seq_len, head_dim]
    k1 = np.transpose(k1, (0, 2, 1, 3))
    v1 = np.transpose(v1, (0, 2, 1, 3))

    # 计算注意力分数
    scores1 = np.matmul(q1, np.transpose(k1, (0, 1, 3, 2))) / np.sqrt(head_dim)

    # 应用softmax
    attn_probs1 = np.exp(scores1 - np.max(scores1, axis=-1, keepdims=True))
    attn_probs1 = attn_probs1 / np.sum(attn_probs1, axis=-1, keepdims=True)

    # 注意力输出计算
    attn_output1 = np.matmul(attn_probs1, v1)  # [bsz, heads_per_gpu, seq_len, head_dim]

    # 恢复原始维度
    attn_output1 = np.transpose(attn_output1, (0, 2, 1, 3))  # [bsz, seq_len, heads_per_gpu, head_dim]
    attn_output1 = attn_output1.reshape(bsz, seq_len, hidden_dim // 2)  # [bsz, seq_len, hidden_dim//2]

    # GPU 2处理后half_heads个头
    wq_sub2 = wq_weight[:, hidden_dim // 2:]  # 后half_heads个头的权重
    wk_sub2 = wk_weight[:, hidden_dim // 2:]
    wv_sub2 = wv_weight[:, hidden_dim // 2:]
    wo_sub2 = wo_weight[hidden_dim // 2:, :]

    # GPU 2上的计算
    q2 = np.matmul(inputs, wq_sub2)
    k2 = np.matmul(inputs, wk_sub2)
    v2 = np.matmul(inputs, wv_sub2)

    # 重塑为多头形式
    q2 = q2.reshape(bsz, seq_len, heads_per_gpu, head_dim)
    k2 = k2.reshape(bsz, seq_len, heads_per_gpu, head_dim)
    v2 = v2.reshape(bsz, seq_len, heads_per_gpu, head_dim)

    # 调整维度顺序
    q2 = np.transpose(q2, (0, 2, 1, 3))  # [bsz, heads_per_gpu, seq_len, head_dim]
    k2 = np.transpose(k2, (0, 2, 1, 3))
    v2 = np.transpose(v2, (0, 2, 1, 3))

    # 计算注意力分数
    scores2 = np.matmul(q2, np.transpose(k2, (0, 1, 3, 2))) / np.sqrt(head_dim)

    # 应用softmax
    attn_probs2 = np.exp(scores2 - np.max(scores2, axis=-1, keepdims=True))
    attn_probs2 = attn_probs2 / np.sum(attn_probs2, axis=-1, keepdims=True)

    # 注意力输出计算
    attn_output2 = np.matmul(attn_probs2, v2)  # [bsz, heads_per_gpu, seq_len, head_dim]

    # 恢复原始维度
    attn_output2 = np.transpose(attn_output2, (0, 2, 1, 3))  # [bsz, seq_len, heads_per_gpu, head_dim]
    attn_output2 = attn_output2.reshape(bsz, seq_len, hidden_dim // 2)  # [bsz, seq_len, hidden_dim//2]

    # 合并结果 (相当于在head维度上concatenate)
    output_tp_parallel = attn_output1 @ wo_sub1 + attn_output2 @ wo_sub2

    print(np.mean(np.abs(output - output_tp_parallel)))