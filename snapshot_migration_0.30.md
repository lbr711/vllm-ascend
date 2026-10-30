# Snapshot migration to vLLM 0.30.0

## Release v0.30.0 migration (2026-10-30)

- Source: `snapshot_0.30.0rc2` at `a844cd4f6`, baseline `7d825229c`.
- Target: `upstream/releases/v0.30.0` at `66b5a5eaca3dbf97aa4022e0d6170ba7ba53dc9f`.
- New local branch: `snapshot_0.30.0`; the source branch is unchanged.
- All 19 snapshot commits were replayed. Conflicts occurred in KPool metadata,
  MLA weight initialization, MLA CP helpers, pool-worker method insertion and
  MXFP8 expert-weight processing/tests. Keep the release's constructors, layouts
  and RL refresh logic, then attach snapshot hooks to those implementations.
- This release diverges from the rc2 baseline; it is not merely a tag change.

### Additional state adaptation

| Owner / first consumer | Restore action |
| --- | --- |
| MLA PROLOG_V3 private weight dictionary | Rebuild from restored source projections using the baseline in-place refresh; preserve dictionary and tensor identities. |
| MLA strided writer / FlashMLA / DCP decode and prefill workspaces | Restore constant slot-index arrays in place; prolog overwrites the active KV scratch before reading it. |
| KPool metadata builder | Clear captured state tables, scratch and shape caches together with per-step metadata/query caches before recapture. |
| W4A8 MXFP4 and W8A8 MXFP8 per-expert lists | Persist the actual list operands before their original Parameters are removed, only when snapshot is enabled. Preserve the existing MegaMoE list coverage. |
| V1/V2 KV block zeroer | Recreate initialized zeroers after module restore: their device address/stride/page tables cannot survive as stale device contents. Do not create a zeroer where cold start did not initialize one. |

The new `MooncakeConnectorV2` / `MooncakePullConnector` implementation now uses
the existing two-phase snapshot endpoint lifecycle. The Pull worker stops and
joins its receiving thread and executor before releasing registered regions and
all engine references. Rebuild refreshes identity, TP group and ports, registers
the retained cache tensors and starts a new consumer thread with empty remote
metadata/request caches. Scheduler rebuild closes the old control threads and
sockets; P publishes metadata only after EngineCore collects fresh worker
handshakes. Missing registrations and thread-exit failures raise errors rather
than skipping cleanup. No new vLLM resume protocol or V1/Hybrid implementation
changes are required. The temporary V2/Pull snapshot rejection was removed.
Existing MooncakeConnectorV1, Hybrid, Layerwise and store snapshot hooks remain in place.
The new pool layerwise key helpers retain topology-derived host values rather
than introducing a separate transport owner.

### Dependency and validation boundaries

- Target pins include torch 2.10.0, torch-npu 2.10.0.post7,
  triton-ascend 3.6.0.post2, memfabric-hybrid/memcache-hybrid 1.3.0;
  the Dockerfile defaults to CANN 9.2.1 and vLLM v0.30.0.
- Inspected the exact aarch64 `triton-ascend==3.6.0.post2` wheel from the Ascend
  package index. `JITFunction.device_caches` is a defaultdict whose per-device
  entries contain compiled kernels and binders. Autotuner still owns `cache`
  and wraps `fn`; Heuristics wraps `fn`. The existing reset clears in place and
  preserves the defaultdict factory. No dependency installation was performed.
- CPU/mock tests and source inspection do not establish NPU numerical accuracy
  or native library restore compatibility. Test new snapshots on the target
  image; do not reuse snapshots or weight artifacts from the older baseline.
- This review covers migrated lifecycle hooks and the changed state listed
  above, not every newly introduced model/quantization/offload combination.
  V2/Pull transport, NPU graph capture, distributed PD/PP, pooling and
  workload accuracy require separate hardware acceptance.
- Validation: 477 tests passed and 3 skipped in isolated CPU/mock invocations:
  baseline restore (92), worker lifecycle (18), MXFP4 (23), MXFP8 (29),
  V2 restore (11), model-runner restore (13), checkpoint (4), KPool restore (3),
  MLA (210), MLA CP (64 passed / 3 skipped), PROLOG_V3 composition (10).
  Updated hand-built test configs to explicitly set `snapshot_config=None`.
  `git diff --check` and default pre-commit checks passed.
- V2/Pull follow-up validation: 311 CPU/mock tests passed in isolated runs:
  snapshot lifecycle (11), base worker (58), Pull worker (97), connector (6),
  Pull scheduler (48), baseline restore (91). Default pre-commit checks passed.
  This does not replace a multi-node NPU snapshot/restore test.

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

# Incremental migration to `snapshot_0.30.0_56debc19`

## Pairing and history

- Source: `snapshot_0.30.0_c6a42a86` at `f0f7f77bc`; kept unchanged.
- Ascend baseline: `56debc194a42d473f8c456d4850369442d694c8f`.
- Target: `snapshot_0.30.0_56debc19`. The 13 snapshot commits above
  `c6a42a86` were replayed in order; the target baseline remains the first
  parent of the migrated snapshot series.
- vLLM pairing remains `snapshot_0.30.0`; no resume RPC or checkpoint filename
  change is introduced by this migration.

## Consumer-path audit and adaptation

| New baseline area | Restore result |
| --- | --- |
| PCP graph RoPE staging and hidden-restore buffers | Each consumed slice is copied or zeroed before graph execution. Existing graph recapture replaces captured registrations; no checkpoint entry is required. |
| PCP SFA O-projection collective buffers | Input slices are copied and collective output slices are overwritten before use. The retained process-group wrapper is rebuilt by the existing SFA restore hook. |
| Sparse MLA valid-mask and plan buffers | The valid mask is copied for every prepared batch; lengths and the plan are regenerated from current selected indices. Existing reset still clears block tables, lengths, metadata and sinks. |
| V1 static KV graph buffers | The new baseline zeroes static KV buffers after `capture_model`; snapshot restore invokes that same recapture path. |
| Shared-expert overlap prepared inputs/events | They are request-scoped and created by the current forward. No long-lived request object is restored. |
| A5 MegaMoE MXFP4/MXFP8 expert lists | The operator consumes cloned per-expert weight/scale tensors after the source Parameters are disposed. With snapshot enabled, W4A4, W4A8 and W8A8 now register those lists as persistent buffer aliases so checkpoint restore refills the exact tensors consumed by MegaMoE. Snapshot-disabled initialization is unchanged. |
| Mooncake layerwise listener gate | Preserve normal blocking receive when snapshot is disabled and bounded polling when enabled. Rebuild tolerates test/minimal configs without a declared snapshot field; production `VllmConfig` behavior is unchanged. |

No fallback or feature-disable path is added. Hardware validation is still
required for NPU graph recapture, collectives, MooncakeV1 PP and every deployed
quantization path; CPU tests cannot prove numerical equivalence.

# Incremental migration to `snapshot_0.30.0_8233545b`

## Pairing and history

- Source: `snapshot_0.30.0_56debc19` at `800846a5160a77bed7450404d89df96965edf0d2`;
  kept unchanged.
- Ascend baseline: `8233545b5`. It is a descendant of
  `56debc194a42d473f8c456d4850369442d694c8f`.
- Target: `snapshot_0.30.0_8233545b`. The 14 snapshot commits above the old
  baseline were replayed in order.
- The only production conflict was Mooncake store initialization. The result
  retains both the snapshot-owned current-hostname cache and the new baseline's
  scheduler-only SSD setup gate; neither path falls back to the other's role.
- vLLM remains on `snapshot_0.30.0`; this migration does not change its resume
  RPC, checkpoint layout or port protocol.

## Incremental consumer-path audit

| New baseline area | Restore result |
| --- | --- |
| SFA cache grouping and Prolog V3 indexer compatibility | The change selects a preprocessing implementation and does not introduce persistent tensors, request state or native resources. Existing SFA and metadata-builder restore hooks remain the owners of runtime state. |
| DSV4 fixed-K PCP speculative decoding | `_mtp_hidden_buffer` is now allocated before compilation so graph capture and PCP restoration share one stable address. Every target forward copies the complete subsequently consumed token slice before the draft reads it. PCP itself is reconstructed on restore and reconnects model/speculator consumers; no checkpoint entry or replacement allocation is required. |
| PCP draft metadata and adaptive verification | Restore clears `execute_model_state`, request/block-table buffers and speculator state, then reconstructs the PCP manager. New batch replacements are request-local values produced after restore, not persistent state. |
| KVPP speculative decoding | The baseline widens accepted configurations but adds no new long-lived object. Existing restore shuts down the old prefetch executor and recreates `KVPPRuntime` from current cache/configuration state for both V1 and V2. |
| MiniMax M3 fused sparse preparation | Builder-owned context, cumulative-length and start-location buffers are ordinary device tensors, but every active slice is written by `build()` before the operator consumes it. Fused query/index outputs are allocated per invocation, while KV/index caches retain normal block-table ownership. No stale tail is consumed. |
| New MiniMax Triton kernels | Their launcher/autotune caches are process-global native runtime state. The existing restore pass scans loaded vLLM/vLLM-Ascend modules and clears `JITFunction`, `Autotuner` and `Heuristics` caches before graph recapture, covering these kernels without model-specific duplication. |
| Step-3.5/3.7 overlay backing and XLite layout detection | These changes derive cache layout or immutable model-format flags during initialization. Cache contents remain governed by fresh block tables; no new restorable tensor or handle is retained. |
| Static V1 KV-buffer zeroing revert | The target baseline deliberately removes its cold-start post-capture zero pass and its tests. Migration preserves that baseline decision instead of silently changing cold startup. Snapshot restore still invalidates request/block-table state and follows the same graph-capture behavior as this baseline. GLM-5.3-Flash C8, which motivated the reverted upstream change, requires explicit hardware accuracy validation on this baseline. |

No additional snapshot production hook was needed for this baseline delta.
This conclusion is based on the first consumers and overwrite coverage above,
not merely on whether newly added tensors appear in `state_dict`.

## Validation and limits

- Snapshot contract suite: 84 passed.
- Individually isolated affected suites for SFA/CP, MiniMax M3 metadata and
  fused preparation, PCP, fixed-K MTP, DSpark, adaptive verification, V2
  speculator/model runner and Mooncake backend: 918 passed. The one MTP/PCP
  port-allocation case was rerun outside the socket-restricted sandbox and
  passed (75/75 for that file).
- The migrated commit series matches the source series under `git range-diff`;
  the only intentional difference is the Mooncake conflict resolution described
  above.

These are CPU/mock tests. They do not execute NPU kernels, HCCL teardown,
container restore or ACL Graph recapture. Hardware acceptance remains required
for the deployed model/configuration, especially PP/PCP, fixed-K speculation,
KVPP, MiniMax fused kernels and GLM-5.3-Flash C8. A new container snapshot must
be created from this exact pairing; restoring a process image created with the
old baseline is unsupported.

## Incremental migration to `snapshot_0.30.0rc2`

### Pairing and history

- Source: `snapshot_0.30.0_8233545b` at `445cdd1aa`; kept unchanged.
- Target baseline: `upstream/releases/v0.30.0rc2` at
  `7d825229c77025cfb185a010c25819c1214b9450` (fetched for this migration).
- All 17 snapshot commits were replayed without conflicts. `git range-diff`
  reports identical patches, including Triton cache invalidation, PDMix scale
  persistence and force-EPLB cache invalidation.
- vLLM pairing remains the local `snapshot_0.30.0` branch. No vLLM files,
  checkpoint filenames or resume protocol were changed.

### Incremental consumer-path review

| Baseline delta | Lifecycle assessment |
| --- | --- |
| DSV4 MTP hidden buffer returns to lazy allocation | Forward writes the active hidden-state slice before the draft consumes it. Keep existing storage across restore and recapture; do not replace it merely because allocation became lazy. FULL graph/compiler behavior still requires hardware validation. |
| PCP speculative decode sharding restriction and V2 metadata changes | Preserve the release branch's restrictions. Existing restore recreates the PCP manager, reconnects model/speculator references and resets request/speculator state before capture. Do not reintroduce main-branch feature behavior through the snapshot patch. |
| MiniMax M3 TP decode scoring | TP group is obtained at forward time. Local block tables, context lengths, scores and top-k candidates are derived in forward; zero-block ranks initialize neutral candidates. No new model-owned tensor cache is introduced by this delta. |
| MiniMax metadata and idle-DP flags | Active context-length slices are copied in build; cumulative lengths and start positions are recomputed. Dummy-forward flags belong to the forward context, not a persistent model cache. The retained attention-mask builder is already cleared by the metadata restore pass. |
| KVPP speculative method restrictions | Retain release validation and layer-ownership rules. Existing V1/V2 restore rebuilds KVPP runtime; this delta adds no persistent native resource. |

This is an incremental review of the baseline delta, not proof that every model
and hardware combination is free of stale state. NPU restore, graph recapture,
PD transfer, PP and numerical accuracy still require hardware acceptance.
Create new snapshots from the new pairing; do not reuse process images from
the previous baseline.

### Validation

- 86 independent snapshot contract tests passed.
- 293 additional tests passed in isolated processes: checkpoint (4), worker
  lifecycle (16), MiniMax metadata/scoring (88), fused preparation (17), PCP
  manager (76), MTP/PCP (66), V2 restore (11), model-runner restore (13) and
  distributed snapshot (2). MTP/PCP was rerun outside the socket-restricted
  sandbox; the original single failure was `PermissionError` creating a socket.
- Use `PYTHONPATH=/home/liu/vllm-project/vllm` for the paired source tree. The
  installed vLLM in the local virtual environment lacks `IndexerKVDType`.
  Combining these suites in one process also exposes cross-test mock pollution;
  the results above are from separate invocations, without production changes.
- `git diff --check` and default pre-commit checks passed. Manual markdownlint
  reports pre-existing multiple top-level headings in this migration journal.
- No NPU or container end-to-end test was performed locally.
