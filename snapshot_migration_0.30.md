# Snapshot migration to vLLM 0.30.0

## Pairing

| Repository | Source snapshot tip | New baseline | New branch |
| --- | --- | --- | --- |
| vLLM | `snapshot_0.29.0` / `7b099ccdbc` | `v0.30.0` / `ced6857afa` | `snapshot_0.30.0` |
| vLLM-Ascend | `snapshot_0.29.0_a71b766ce` / `3dcd09908` | `2bb3f4471` | `snapshot_0.30.0_2bb3f447` |

The source branches are unchanged. This migration carries the snapshot delta,
not the differences between the upstream release branches. The new Ascend base
also pins `v0.30.0` in `.github/vllm-release-tag.commit`.

Snapshot creation assumes an idle service without external inference requests.
Create a new snapshot with this pairing; this is not an upgrade of a previously
created container image/checkpoint.

## Retained lifecycle

- Snapshot configuration, suspend/resume/device-unlock APIs, shared monitor and
  optional automatic-checkpoint sentinel.
- EngineCore/client transport reconnect, DP coordination-group rebuild, worker
  communication-group rebuild, connector identity and handshake refresh.
- Reserved restore-port conflict avoidance and port-selection diagnostics.
- Target/draft checkpoints, packed NZ copies, persistent derived weights,
  module/implementation restore hooks, runner/builder/block-table resets,
  in-process Triton launcher invalidation and ACL Graph recapture.
- Both Model Runner V1 and V2. Snapshot does not override runner selection.
- Existing long-prompt/per-layer diagnostics and the latest V1 block-table and
  dummy query-start-offset fixes. Historical raw diagnostic logs are not added.

## Baseline changes reviewed

| Area | Migration action |
| --- | --- |
| EngineClient | Preserve the new renderer import alongside snapshot imports; retain upstream request/renderer behavior. |
| V2 PCP | Remove the deleted `req_states` factory argument. Reattach `kv_cache_config`, model-state and speculator references to the new PCP manager, as initialization now does. |
| V2 speculative decoding | Reset the new shared draft-prefill/confidence buffers; recreate an enabled online acceptance estimator before recapture. The estimator is not model checkpoint state. |
| V2 graph manager | Preserve the new `ubatch_runner` reference when recreating the graph manager; this does not certify DBO recovery. |
| KPool indexer metadata | Upstream replaced individual buffers with a per-slot-mapping-address dictionary. Clear that dictionary before recapture, so each draft step rebuilds its own metadata storage. |
| DCP | Keep the new draft/global-to-local metadata preparation methods alongside the snapshot reset hook; do not replace them with the old implementation. |
| MegaMoE | Keep upstream sleep/HCCL teardown and refresh methods. Snapshot retains its separate symmetric-buffer invalidation after group reconstruction. |
| DSA/RoPE | Preserve the new layer-scoped RoPE lookup and YaRN constructor arguments. Snapshot persistence/rebuild remains on existing owners. |
| SFA C8 | Preserve the new distinction between main cache dtype and indexer cache dtype; do not restore removed dtype attributes. |
| Model loading | New NZ/Triton warmup threads join before `load_model` returns. Resume does not invoke model loading or restart these threads. |
| Communication | Preserve upstream ETP/Engram group teardown additions. Snapshot still invokes the original group destruction/initialization entry points. |
| Connectors | Preserve the new pool layout/PP/PCP bookkeeping and existing prepare-before-rebuild ordering; do not replace cold-start connector construction. |

No new fallback path or feature-disable workaround is introduced. New restore
adaptations are confined to snapshot hooks and lifecycle helpers.

## Validation

CPU/mock tests are run with the two local source trees explicitly on
`PYTHONPATH`, one process, and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
The environment has an older installed vLLM, so running without this source
path does not validate this pairing.

- vLLM snapshot/API/sentinel/transport suites: 38 passed.
- vLLM port allocation and snapshot CLI cases: 10 passed (local socket tests
  require execution outside the socket-restricted sandbox).
- Ascend runner/worker/DSA/KPool lifecycle suites: 44 passed, including the
  acceptance-estimator enabled/disabled cases.
- Ascend checkpoint, trace, MoE, block-table, DCP and distributed suites:
  64 passed, 11 skipped.
- Focused connector snapshot/reset/rebuild suites: 10 passed.
- Rotary/MLA/W4A8/W8A8/KV-C8 and updated V2 lifecycle suites: 102 passed.
- Changed Python files compile; `git diff --check` passes.

These tests do not execute NPU kernels, CRIU, native HCCL teardown or graph
recapture. They are not proof of hardware recovery or numerical equivalence.
Ruff is not installed in the current test environment.

## Outstanding hardware investigation

Latest user report: **DeepSeek V4 now completes restore, but inference accuracy
is abnormal.** Migration does not claim to fix that issue. Keep it separate
from the GLM-5.2 long-prompt accuracy investigation.

On this pairing, collect deterministic before/after requests on the same DP,
including short and long prompts, repeated requests and MTP acceptance rate.
Compare the retained layer summaries to locate the first divergence.

New optional capabilities such as watermarking, dynamic EPLB, microbatch/DBO,
Engram, and additional model-specific state are not certified for snapshot use
by this migration. They require their own lifecycle audit and hardware tests
before enabling them. An idle checkpoint does not by itself validate their
device constants, handles, or background state.

# Incremental PP snapshot support on `snapshot_0.30.0_eee24e2e`

Scope: idle snapshots, unchanged model/quantization/DP/TP/PP topology and layer
partition, with `MooncakeConnectorV1`. The [GLM5.2 PD example](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/GLM5.2.html) uses prefill PP=2
and partition `41,37`; the connector still requires decode PP=1. This change
does not enable unsupported cold-start combinations or migrate active requests.

## Shared checkpoint directory

- PP=1 retains its existing directory, filenames, checkpoint format and restore behavior.
- PP>1 uses `{model}_dp{DP}_tp{TP}_pp{PP}/model_ckpt.{dp}tp{tp}pp{pp}.pth`.
  Draft-model files use the `model_ckpt_drafter` prefix on stages owning a draft.
- Instances with identical weights and topology share completed rank shards;
  different DP/TP/PP ranks do not overwrite one another. Different weight
  revisions or quantization configurations must use different snapshot roots.
- All snapshot writers (PP=1 and PP>1) take an exclusive file lock on
  `{path}.lock` before checking existence, holding it through `torch.save`.
  Concurrent writers of the same shard wait and then skip the completed file;
  different shard paths use independent locks. Lock files are kept, never
  unlinked, so contenders always lock the same inode. Lock failures propagate.
- Trade-off: no atomic-publication/strict-validation switches or hard links.
  The shared filesystem must support cross-node file locks, and all writers
  must use this protocol. A writer crash can still leave a partial checkpoint;
  restore does not acquire this lock. Deployment must ensure successful writes
  before restore and isolate different weights, quantization and layer partitions.

## Inference-path audit and lifecycle actions

| Owner / first consumer | Restore action |
| --- | --- |
| Stage-local embedding, decoder layers, norm/head and last-stage MTP | Separate PP rank shards; existing parameter/derived-tensor restoration applies to the real stage modules, not `PPMissingLayer` placeholders. |
| Worker PP sends and coordinator-retained metadata/tensor sends | Drain native Work and synchronize before checkpoint. Never wait on restored Work handles. |
| V1/V2 `intermediate_tensors`, read by non-first-stage warmup/capture before receiving real input | Zero all payload tensors in place before recapture, including hidden states, residual and optional auxiliary keys. Real requests overwrite their valid slices. |
| V1 sampling and KV-output carry-over | Clear `execute_model_state`, previous sampled tokens, KV connector output and valid sampled-count tensor. Existing draft/request resets remain in place. |
| V2 `PPHandler` sampled-token broadcasts | Abort/destroy its independently owned sibling group before global group and KV transport teardown; recreate after group initialization. Reset receive FIFO and request generations; rebind legacy spec-PP closures only if that protocol was installed. Preserve runtime-restored streams. |
| Per-layer shared experts reading `moe_config` groups | Refresh target and draft layers' group references, not just the last configuration retained in the global MoE registry. Existing expert-map persistence and dispatcher reset remain in use. |
| SFA/DSA metadata, quantized projection payloads, RoPE, block tables and draft buffers | Continue existing builder/module hooks on each stage before recapture; PP does not bypass them or reload weights through the cold-start path. |
| Mooncake worker endpoint and scheduler handshake | EngineCore and every worker resolve their own current local IP once via `get_ip(force=True)` for both PP=1 and PP>1. Only the common master IP, checkpoint path and engine ID cross the resume RPC; unresolved local addresses fail explicitly. Existing endpoint rebuild republishes the `(pp_rank, tp_rank)` handshake entries and clears remote metadata caches. |
| V2 default model-state attention metadata | Rebuilt by `prepare_attn` for dummy capture and real batches before use; stable model configuration and layer indices remain unchanged. |

V2 non-last-stage dummy runs may leave `execute_model_state` without any live
request. Snapshot preparation checks actual request count, not that leftover
warmup object. This avoids rejecting clean snapshots after cold-start capture.

The resume RPC no longer carries `local_ip`. Deploy the matching vLLM and
vLLM-Ascend changes together; mixing old/new resume signatures is unsupported.

Validation: 68 CPU contract tests in `tests/ut/snapshot/test_baseline_restore.py`
passed with one CPU thread. They cover DP/TP/PP filename isolation, reuse of
existing shards, cross-process write serialization (including a visible partial
file), lock error propagation/release, target/draft round-trip, PP=1 gating, V1/V2
runtime reset, independent-group lifetime, per-layer group aliases, and per-node
IP selection. These tests do not validate NPU graph/HCCL execution or shared
filesystem behavior on the deployment. Hardware acceptance still requires
GLM5.2 PP=2 cold-start versus restore checks (short/long prompts and MTP), plus
concurrent snapshot creation by multiple instances using the actual shared path.

# Incremental migration to `snapshot_0.30.0_c6a42a86`

## Pairing and history

- Source: `snapshot_0.30.0_eee24e2e` at `7313261e6`; kept unchanged.
- Ascend baseline: `c6a42a8649cc0179037b063177ab51cb814b7e28`.
- Target: `snapshot_0.30.0_c6a42a86`. The 11 snapshot commits above
  `eee24e2e` were replayed, without merging remote debug/log commits.
- vLLM pairing remains `snapshot_0.30.0` at `8dd67ddead`; this migration
  does not modify that repository or change the resume RPC.
- Build a new snapshot with this pairing. Do not restore an old process image
  using the new Python implementation or reuse weight shards across differing
  models, quantization layouts or layer partitions.

## Consumer-path adaptations

| Owner and first consumer | Restore action / audit result |
| --- | --- |
| SFA indexer Hadamard matrices, consumed by C8 **and new C4** quantization | Rebuild for `enable_sparse_li_quant`, not C8 alone. These are non-persistent buffers, so state_dict is insufficient. |
| TurboQuant `rotation`, `centroids`, `norm_lut`, new shared `inverse_rotation`; consumed by target/draft forward, compression and inverse | Recompute with the original `legacy_hadamard` seed sequence; bypass the cached initializer and refill all tensors in place. Existing aliases and the shared-cache entries remain valid, including layers initialized later. |
| SFA implementation's new ordinary `turboquant` helper | Invoke its restore hook through the implementation's existing lifecycle entry point; DSA already delegates to its helper. |
| PP transport top-k payload, aliased to the model's top-k index buffer | Fill with the invalid index sentinel `-1`, rather than overwriting the module reset with zero. Other intermediate payloads retain zero initialization. No receive/cold-start protocol changes. |
| GQA PCP metadata builder's retained process group and batch context | Rebind the group after communication rebuild and clear per-batch references before capture. DCP implementation group refresh remains inherited from its mixin. |
| V4.1 flat slots, optional DSpark/A5 metadata buffers and compressor source RoPE | Restore invalid-index/zero sentinels; rebind source tables after global RoPE restoration. Retain feature configuration. |
| V4.1 PCP cached RoPE tuples | Clear old full-table aliases and regenerate scratch views before capture; reset hidden restore indices. Both local and nested global builders run their existing hooks. |
| KPool metadata/query caches keyed by slot-mapping addresses | Clear both dictionaries before capture. Tail slot buffers need no extra reset: build copies every valid slot and fills the complete padding slice with `-1` before consumption. |
| DSV4 multimodal sentinel table | Invalidate the ordinary cached stack; its source Parameters are checkpoint-restored and the table is recomputed on first use. This isolated repair does not certify the full multimodal pipeline. |
| V1 C8 MXFP scale-page views | Existing module restore refills the scale cache from restored Parameters in place; new runner tuples retain valid aliases. Upstream per-page reassignment logic is preserved. |
| MoE group registry and V2 PP helper rename | Iterate all config-keyed communication objects as before. No snapshot import references the removed `pp_utils` module. Preserve the new upstream `pp_transport` implementation. |
| Worker and KV lifecycle | Retain group destruction, KV transport destruction, group rebuild, model restore, graph recapture, then endpoint rebuild. Worker initialization invokes the updated upstream group initializer rather than copying its old body. Pool/MultiConnector lifecycle remains unchanged. |

The additional production edits are confined to restore hooks/helpers. They do
not select V2, alter cold-start quantization, recreate runtime-owned streams or
events, or silently switch a failing configuration to another implementation.

## Validation and limits

- 80 CPU contract tests passed in `tests/ut/snapshot/test_baseline_restore.py`,
  including the retained PP, shared-writer, connector and teardown-order tests.
  New cases cover C4/C8 gating, legacy/non-legacy TurboQuant, shared aliases,
  inverse matrices, PP top-k sentinels, PCP/group/RoPE refresh and cache invalidation.
- Ruff checks and formatting passed for the changed Python files; `git diff
  --check` passed. These AST-extracted CPU tests execute production methods but
  do not establish NPU import/ABI, graph capture, HCCL or end-to-end accuracy.
- This is migration of the existing snapshot scope, not certification of all
  newly introduced upstream features. Dynamic EPLB/HIXL weight transfer,
  Engram, sparse KV offload, DSpark PD auxiliary-KV/prefix-pool transport,
  V4.1 A5 graph-side metadata execution and full multimodal execution still
  require dedicated lifecycle/hardware validation before snapshot deployment.
- Hardware regression remains required for the deployed model: V1/V2 as used,
  graph recapture, PP=1 and MooncakeV1 PP>1, short/long prompts, MTP, and pooled
  KV transfer where enabled. No claim that all stale state has been eliminated
  is made from CPU tests alone.
