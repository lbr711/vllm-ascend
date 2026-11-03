#!/bin/bash
source ~/.bashrc 2>/dev/null || true

cp -rf /code /home
cd /home/code
export snapshot_enable="${snapshot_enable:-true}"
export SERVED_MODEL_NAME="Qwen3-30B-A3B-W8A8"
export WEIGHT_MOUNT_PATH="/weight"
export SNAPSHOT_META="/home/code/config/snapshot_metadata.json"

# patch补丁应用
bash /home/code/patches/apply_site_packages_patches.sh /home/code/patches/vllm_snapshot_0.30.0_site_packages_v1.patch /home/code/patches/vllm_ascend_snapshot_0.30.0rc2_site_packages_v3.patch

# 等待ranktable就绪
echo "[$(date "+%Y-%m-%d %H:%M:%S")]------------>Waiting for global rank table completed"
python3 /home/code/utils/rank_table_checker.py
echo "[$(date "+%Y-%m-%d %H:%M:%S")]------------>MA global rank table is completed"

# 获取master IP
sleep 10
DP_MASTER_IP=$(python3 /home/code/utils/get_master_ip.py)
echo "DP_MASTER_IP=${DP_MASTER_IP}"

export IS_MASTER=1
export ROLE=master
echo "IS_MASTER=${IS_MASTER} ROLE=${ROLE}"

# snapshot metadata配置文件刷新
if [[ "$snapshot_enable" == "true" ]]; then
  nohup python3 -u /home/code/utils/watch_snapshot_config.py --snapshot-metadata-path "${SNAPSHOT_META}" > /dev/stdout 2>&1 &
fi

# 启动进程
bash /home/code/utils/run_engineserver.sh \
    --port 9000 --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-start-rank 0 \
    --data-parallel-address "${DP_MASTER_IP}" --data-parallel-rpc-port 13389 --tensor-parallel-size 4
