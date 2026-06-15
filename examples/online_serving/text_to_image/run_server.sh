#!/bin/bash
# Qwen-Image online serving startup script with inter-request cache support

MODEL="${MODEL:-/mnt/shared/models/Qwen-Image}"
PORT="${PORT:-8091}"
CACHE_BACKEND="${CACHE_BACKEND:-inter_request}"
PERSISTENT_CACHE_DIR="${PERSISTENT_CACHE_DIR:-./persistent_cache}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

echo "Starting Qwen-Image server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Cache backend: $CACHE_BACKEND"
echo "Persistent cache dir: $PERSISTENT_CACHE_DIR"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

CACHE_CONFIG="{\"inter_request_max_entries\":100,\"inter_request_max_memory_gb\":4.0,\"inter_request_persistent_cache_dir\":\"${PERSISTENT_CACHE_DIR}\"}"

ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --cache-backend "$CACHE_BACKEND" \
    --cache-config "$CACHE_CONFIG" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
