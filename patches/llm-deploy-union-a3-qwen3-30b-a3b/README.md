# llm-deploy-adapter

项目为 A2机型 + vllm-ascend + patch方式 + Qwen3-30B-A3B-W8A8模型 + 显存快照 + 单机部署。

镜像：https://www.hiascend.com/developer/ascendhub/detail/f1690465f39847a8b0a1f9e5b36a03c4
* A2 机型：mindie-motor:3.2.0b6-vllm_ascend0.30.0rc2-a2-ubuntu22.04-py3.12
* A3 机型：mindie-motor:3.2.0b6-vllm_ascend0.30.0rc2-a3-openeuler24.03-py3.12

```shell
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie-motor:3.2.0b6-vllm_ascend0.30.0rc2-a2-openeuler24.03-py3.12

docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie-motor:3.2.0b6-vllm_ascend0.30.0rc2-a3-openeuler24.03-py3.12
```

启动命令： 
```shell
# 单容器 DP4 x TP4, 需要 16 个可见 NPU 逻辑设备.
bash /code/master.sh
```

两个节点分别运行 DP2 x TP4 时 (全局 DP4, 每个容器需要 8 个逻辑设备):

```shell
# master 节点: 先把 master.sh 的 --data-parallel-size-local 4 改为 2.
bash /code/master.sh
# slave 节点
bash /code/slave.sh
```

两个入口各自完成补丁应用、rank table 等待和引擎启动, 不再调用其它启动 shell.
均要求设置 POD_IP. 总 DP 数、本地 DP 数、TP 数和 RPC 端口直接在各脚本的 vllm serve 命令中修改.
master 默认 DP4/local4/TP4, slave 为 DP4/local2/TP4, RPC 端口均为 13389.
沿用原逻辑: snapshot_enable 只控制 metadata watcher, 引擎仍传入 snapshot-config.
健康检查仅用于提供 HTTP 服务的 master, headless slave 不提供该接口.

健康检查，就绪探针：
```shell
bash /home/code/health_check.sh
```

服务验证：
```json
curl http://{ip}:9000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen3-30B-A3B-W8A8","messages":[{"role":"user","content":[{"type":"text","text":"你好，请介绍一下自己"}]}],"max_tokens":128,"stream":false}'
```

模型权重大小：50GB
显存快照大小：80GB
A2节点总资源：NPU 8卡、CPU 190核（可用170核）、内存2006339MiB 2006GB（可用1900GB）
扣除模型全和显存快照：1900-130 = 1770GB = 1,688,003MiB
