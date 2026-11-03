#!/bin/bash
source ~/.bashrc 2>/dev/null || true
set -euo pipefail

export ROLE=master
export IS_MASTER=1
export snapshot_enable="${snapshot_enable:-true}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-W8A8}"
export WEIGHT_MOUNT_PATH="${WEIGHT_MOUNT_PATH:-/weight}"
export SNAPSHOT_META="${SNAPSHOT_META:-/home/code/config/snapshot_metadata.json}"
PORT="${PORT:-9000}"

cp -rf /code /home
cd /home/code

# Apply patches and wait for the rank table before starting the engines.
bash /home/code/patches/apply_site_packages_patches.sh \
    /home/code/patches/vllm_snapshot_0.30.0_site_packages_v1.patch \
    /home/code/patches/vllm_ascend_snapshot_0.30.0rc2_site_packages_v3.patch
python3 /home/code/utils/rank_table_checker.py
sleep 10
DATA_PARALLEL_ADDRESS=$(python3 /home/code/utils/get_master_ip.py)

# Keep the existing snapshot metadata watcher.
if [[ "$snapshot_enable" == "true" ]]; then
    nohup python3 -u /home/code/utils/watch_snapshot_config.py \
        --snapshot-metadata-path "$SNAPSHOT_META" > /dev/stdout 2>&1 &
fi

echo "ROLE=$ROLE POD_IP=$POD_IP PORT=$PORT"
echo "DP_ADDR=$DATA_PARALLEL_ADDRESS"

# snapshot fixed env
export HCCL_IF_IP=$POD_IP
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"

# model env
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-0:0}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1200}"

exec vllm serve "$WEIGHT_MOUNT_PATH" \
    --host "$POD_IP" \
    --port "$PORT" \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-address "$DATA_PARALLEL_ADDRESS" \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 4 \
    --quantization ascend \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-num-seqs 24 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_capture_sizes":[4,8,16,24],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --snapshot-config "{\"snapshot_metadata\": \"${SNAPSHOT_META}\", \"enable_auto_checkpoint\": true}"
