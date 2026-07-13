#!/bin/bash
# Triton 按需加载入口脚本
#
# 支持两种配置方式（优先级从高到低）：
#   1. TRITON_MODELS_FILE：指向一个文本文件，每行一个模型名，支持 # 注释
#   2. TRITON_MODELS：英文逗号分隔的模型名
# 留空时加载模型仓库中的所有模型（兼容旧行为）。
set -e

ARGS=(
  tritonserver
  --model-repository=/models
  --log-verbose=1
  --exit-on-error=false
  --strict-readiness=false
)

# 收集需要加载的模型名
MODELS=()

if [ -n "$TRITON_MODELS_FILE" ] && [ -f "$TRITON_MODELS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"          # 去掉注释
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -n "$line" ] && MODELS+=("$line")
  done < "$TRITON_MODELS_FILE"
elif [ -n "$TRITON_MODELS" ]; then
  IFS=',' read -ra RAW_MODELS <<< "$TRITON_MODELS"
  for m in "${RAW_MODELS[@]}"; do
    m=$(echo "$m" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -n "$m" ] && MODELS+=("$m")
  done
fi

if [ ${#MODELS[@]} -gt 0 ]; then
  ARGS+=(--model-control-mode=explicit)

  # CUSTOM_LABELS 模型是前端显示类别名依赖的公共服务，始终确保它被加载
  UNIQUE_MODELS=$(printf "%s\n" "${MODELS[@]}" CUSTOM_LABELS | awk '!seen[$0]++')
  for m in $UNIQUE_MODELS; do
    ARGS+=(--load-model="$m")
  done
fi

exec "${ARGS[@]}"
