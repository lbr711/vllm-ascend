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
