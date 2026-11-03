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
bash /code/boot.sh
```

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