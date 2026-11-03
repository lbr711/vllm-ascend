#!/bin/bash
set -euo pipefail
rm -rf ~/ascend/log/debug/* 2>/dev/null || true
rm -rf .torchair_cache 2>/dev/null || true

# From qwen30b_a3b_vllm_union_centralized/env.json (motor_engine_union_env)
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-0:0}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1200}"

export HCCL_IF_IP=$POD_IP
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --data-parallel-size) DATA_PARALLEL_SIZE="$2"; shift 2 ;;
        --data-parallel-size-local) DATA_PARALLEL_SIZE_LOCAL="$2"; shift 2 ;;
        --data-parallel-start-rank) DATA_PARALLEL_START_RANK="$2"; shift 2 ;;
        --data-parallel-address) DATA_PARALLEL_ADDRESS="$2"; shift 2 ;;
        --data-parallel-rpc-port) DATA_PARALLEL_RPC_PORT="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
done

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set in boot.sh}"
WEIGHT_MOUNT_PATH="${WEIGHT_MOUNT_PATH:?WEIGHT_MOUNT_PATH must be set in boot.sh}"
SNAPSHOT_META="${SNAPSHOT_META:?SNAPSHOT_META must be set in boot.sh}"

echo "=========================================="
echo "ROLE=$ROLE POD_IP=$POD_IP PORT=$PORT"
echo "DP=$DATA_PARALLEL_SIZE local=$DATA_PARALLEL_SIZE_LOCAL start=${DATA_PARALLEL_START_RANK:-0}"
echo "DP_ADDR=$DATA_PARALLEL_ADDRESS RPC=$DATA_PARALLEL_RPC_PORT TP=$TENSOR_PARALLEL_SIZE"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME WEIGHT_MOUNT_PATH=$WEIGHT_MOUNT_PATH SNAPSHOT_META=$SNAPSHOT_META"
echo "=========================================="

COMMON_ARGS=(
    --host "$POD_IP"
    --port "$PORT"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL"
    --data-parallel-address "$DATA_PARALLEL_ADDRESS"
    --data-parallel-rpc-port "$DATA_PARALLEL_RPC_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --quantization ascend
    --served-model-name "$SERVED_MODEL_NAME"
    --max-num-seqs 24
    --max-model-len 2048
    --max-num-batched-tokens 2048
    --trust-remote-code
    --gpu-memory-utilization 0.9
    --compilation-config '{"cudagraph_capture_sizes":[4,8,16,24],"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --snapshot-config "{\"snapshot_metadata\": \"${SNAPSHOT_META}\", \"enable_auto_checkpoint\": true}"
)

if [ "$ROLE" = "master" ]; then
    vllm serve "$WEIGHT_MOUNT_PATH" "${COMMON_ARGS[@]}"
elif [ "$ROLE" = "slave" ]; then
    vllm serve "$WEIGHT_MOUNT_PATH" --headless --data-parallel-start-rank "$DATA_PARALLEL_START_RANK" "${COMMON_ARGS[@]}"
else
    echo "[ERROR] Unknown ROLE: $ROLE"
    exit 1
fi
