#!/usr/bin/env bash
# 1. 查看可用模型
curl -s --noproxy '*' http://127.0.0.1:13333/v1/models | jq .

# 2. 走 chat/completions 生成文本
curl -s --noproxy '*' http://127.0.0.1:13335/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' | jq -r '.choices[0].message.content'
