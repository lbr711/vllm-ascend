# Snapshot migration to vLLM 0.28

## Branches and scope

| Repository | Snapshot source | Target base | New branch |
| --- | --- | --- | --- |
| vLLM | `snapshot_0.26.0`, `fb46598f49` | `v0.28.0`, `2cf0a6915c` | `snapshot_0.28.0` |
| vLLM-Ascend | `snapshot_0.26.0rc1`, `5d6f0b9c2` | `993782efc842308f646dcf80a562d45288f265d3` | `snapshot_0.28.0_993782ef` |

The previous 0.28 branches are retained as
`snapshot_0.28.0_backup_20260924` and
`snapshot_0.28.0_993782ef_backup_20260924`. The 0.26 and 0.29 branches
are unchanged. The migration reuses snapshot adaptations already developed
against the same Ascend base, then adjusts and tests them against vLLM 0.28.
The unrelated 0.29 Kimi import compatibility change is not included.

Checkpoint creation requires an idle service with no active requests.
This is not migration of in-flight inference.

## Architecture and restore changes

| Area | Adaptation |
| --- | --- |
| API server | vLLM 0.28 still uses `entrypoints/openai/api_server.py` and `entrypoints/serve/utils/server_utils.py`. Snapshot routes, shared monitor and optional sentinel are integrated there, not in the 0.29 launcher layout. |
| DP ports | Carry the source branch's separate snapshot ports, DP-port exclusions and allocation/consumption logs. Preserve upstream 0.28's reserved DP-master port range handling. Probing does not hold sockets open and cannot guarantee availability indefinitely. |
| Engine lifecycle | Preserve transport reconnection, worker resume, scheduler identity refresh and post-worker connector rebuild ordering. Backend tensor/graph restoration remains in Ascend. |
| Runner selection | Use separate V1 and V2 reset paths selected by `use_v2_model_runner`; do not apply V1 InputBatch fields to V2. |
| V2 request/block state | Reset staged request/input state and slot mappings; rebuild block-table layout/address tensors through the original initializer. Keep shared RequestState references. |
| V2 PCP | Recreate the PCP manager through its cold-start factory and update model-state/speculator references. |
| V2 drafter | Use `get_draft_model`, restore its checkpoint and reset speculator buffers/constants. Do not access `sample_src_positions`, which is absent in vLLM 0.28. |
| V2 graph/prefetch | Recreate target/draft ACL Graph managers before normal capture; rebuild enabled KV prefetch runtime and its reference. Do not discard reusable on-disk compilation artifacts. |
| Metadata builders | Reset target, draft and runner-owned builders. DSA PCP also resets its nested global-cache builder, which is not in runner attention groups. |
| DCP constants | Regenerate the indexer's replicated column indices and SFA's rank-order tensor from the rebuilt DCP group. Per-request outputs overwritten before reads are not unnecessarily reset. |
| Weight switching | The old full-o-projection dictionary is replaced by WeightSwitchState. After local parameters are restored, regenerate contiguous transposed gather inputs and repeated scales; discard old async handles. The next collective fills gather outputs. Refresh the switch configuration's group reference. |
| Embedding TP | Refresh the cached group used by embedding all-gather/reduce-scatter. Its temporary tensors are fully overwritten by forward, so no extra clearing is required. |
| KV block zeroing | Rebuild initialized V1/V2 zeroer metadata after restore; these objects hold device addresses/strides/page sizes. Do not enable a zeroer that cold start did not create. |
| Derived weights | Persist derived/prolog weights whose source projections are removed; do not rederive from deleted weights. Preserve checkpoint H2D copy strategies for supported packed W4A8 layouts. |
| KV C8 | Persist `fak_descale_reciprocal` and `quant_kscale` under snapshot configuration; arithmetic on Parameters does not automatically register the resulting tensors. |
| MoE/all-to-all | Adapt hooks to routed-expert, communication-method and dispatcher owners. Restore expert maps/masks and refresh group/native communication references. Reset MegaMoE symmetric-buffer state. |
| Compiled kernels | Invalidate loaded Triton JIT/launcher caches, including wrapped kernels, before first use. Preserve compiled-code disk caches and stable compiler-side indices. This targets the pinned Triton-Ascend API, not arbitrary versions. |
| KV connectors | Retain role-specific StoreConnector and Hybrid Mooncake restoration, refreshed TP references, and prepare-before-rebuild ordering. Respect independent store TransferEngine ownership. |

## Removed or unnecessary resets

- The old SFA `o_proj_full_aclnn_input_params` state no longer exists; its
  replacement is covered through WeightSwitchState rather than an old-field reset.
- Removed `AttentionMaskBuilder.mla_mask` and 0.29-only speculator fields are
  not accessed.
- Request-built metadata, collective outputs and buffers fully overwritten
  before reading are not treated as checkpoint-owned fixed tensors.
- Cached `ep_group` fields in the inspected DeepSeek V4/GLM5Next model classes
  are used to derive sizes during construction, not by their forward paths;
  the active MoE communication owners are refreshed separately.

## Verification and limits

CPU/mocked-device tests run against the local vLLM 0.28 source:

- vLLM snapshot/lifecycle/transport/network tests: 48 passed.
- Sentinel tests: 18 passed. Snapshot CLI cases: 4 passed.
- Ascend runner, checkpoint, communication, quantization, attention/CP,
  embedding, rotary and MoE regression selection: 450 passed, 11 skipped,
  27 subtests passed.
- Connector tests are run separately from TransferEngine tests because a
  connector test module installs process-global mocks that contaminate combined
  collection. Connector selection: 148 passed, 145 subtests passed.
  TransferEngine tests alone: 3 passed.
- Ruff checks on changed Python files and `git diff --check` passed.

These checks do not execute NPU kernels, HCCL teardown or CRIU and do not prove
post-restore numerical equivalence. Hardware validation is still required:

1. GLM-5.2 W8A8C8/W4A8C8 and DeepSeek V4 Flash/MTP on the intended A2/A3 image.
2. V1/V2 selection, DSA CP/DCP/PCP, MegaMoE and the actual all-to-all backend.
3. Multiple deterministic before/after requests, draft acceptance rate and graph
   recapture without unexpected recompilation.
4. Centralized/distributed DP and P/D pooled-KV reuse with the chosen backend.

Dynamic EPLB and arbitrary multimodal/model-specific state are not claimed as
validated by this migration. An idle snapshot does not by itself establish
native backend handle validity; that remains part of hardware testing.
