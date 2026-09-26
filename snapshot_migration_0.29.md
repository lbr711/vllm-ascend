# Snapshot migration to vLLM 0.29

## Branches and scope

| Repository | Source | Target base | Migration branch |
| --- | --- | --- | --- |
| vLLM | `snapshot_0.26.0`, `e53b517f20` | `v0.29.0`, `98dff2a81d` | `snapshot_0.29.0` |
| vLLM-Ascend | `snapshot_0.26.0rc1`, `e2128e27e` | `a71b766ce6fc0a9669a412e15a5cf53f7f827093` | `snapshot_0.29.0_a71b766ce` |

The Ascend branch reuses the previous migration work on the same Ascend base,
then adapts it to vLLM 0.29. The old deployment branches are unchanged.
Snapshot creation requires an idle service; this is not in-flight request migration.

## Changes that affect restoration

| Area | New structure and adaptation |
| --- | --- |
| API server | Lifecycle, router registration and server entry moved to `entrypoints/launchers`. Mount snapshot routes and start the configured sentinel there, not in the deprecated OpenAI server shim. |
| Worker interface | Preserve the new executor KV layout interfaces and device synchronization while retaining suspend/unlock/resume RPCs. |
| ModelRunner V2 | Do not apply V1 InputBatch resets to V2. Use a separate runtime reset implementation selected by `use_v2_model_runner`. |
| V2 request state | Reset staged writes, request counters, input buffers, block IDs and slot mappings in place. Rebuild block-table address/layout tensors using the original initializer. Keep shared RequestState references. |
| V2 PCP | Recreate the PCP manager through its cold-start factory and reconnect model-state/speculator references. |
| V2 draft model | Obtain the draft model through `get_draft_model`; restore its checkpoint and reset speculator-owned inputs, carried state and algorithm-specific index constants. |
| V2 ACL Graph | Replace target/draft graph managers after releasing old graph resources, then use the normal capture entry. No model reload or explicit compiler-cache invalidation is added here. |
| V2 KV prefetch | Recreate an enabled KV prefetch runtime and reconnect its model-state reference. |
| Attention metadata | Target and draft metadata builders share the existing hook/reset dispatcher. Removed fields such as `AttentionMaskBuilder.mla_mask` are no longer reset. |
| DSA metadata | Reset the graph-stable DSA-CP speculative buffers added after `993782ef`. DeepSeek V4.1 builders reset request metadata, queued device tasks and cached RoPE references. Re-enable every metadata provider after recreating its executor. |
| DSA RoPE | Preserve the new sleep-mode buffer ownership while retaining snapshot-time cache validation and rebuild. Rebuild with the configured original sequence length and YaRN mode. |
| MLA/SFA derived weights | The new implementation removes some source projections. Persist the derived/prolog tensors on their owning modules instead of replaying old derivation against deleted weights. The old SFA full-o-projection path no longer exists. |
| KV C8 | Persist `fak_descale_reciprocal` and `quant_kscale` when snapshot is enabled: arithmetic on Parameters produces ordinary tensors, not registered parameters. |
| MoE expert maps | V2 uses upstream device expert maps. Persist the layer's expert-map/mask/routing buffers with the existing helper, preserving aliases held by ExpertMapManager. V1 CPU maps alone are insufficient. |
| MoE/all-to-all | Carry restore hooks into the new routed-expert, communication-method and dispatcher owners rather than the removed monolithic fused-MoE implementation. |
| Triton | Reset loaded JIT caches in both vLLM and Ascend namespaces, including kernels behind autotuner/heuristics wrappers. Disk compilation caches are retained. This implementation targets the pinned Triton-Ascend API, not arbitrary Triton versions. |
| KV connectors | Restore role-specific StoreConnector and Hybrid Mooncake entries; refresh cached TP group references. Respect the new Mooncake independent-store TransferEngine mode rather than resetting an unrelated global TransferEngine. Preserve the prepare-before-rebuild ordering for shared TransferEngine references. |

## Resource ownership

- Model parameters and persistent derived tensors: checkpoint D2H/H2D path.
- Rebuildable module tensors and implementation state: module lifecycle hooks.
- Runner/request/draft/metadata state: version-specific runner reset, with shared builder hooks.
- Process-global tensors and compiled launchers: global reset and JIT-cache reset.
- Communication groups, transport and connector references: worker/engine lifecycle and connector rebuild.
- ACL Graph handles: graph cleanup and recapture, not tensor checkpointing.

## Verification and limits

CPU/mocked-device validation performed during migration:

- vLLM snapshot configuration/lifecycle/transport suites: 38 passed; four CLI
  snapshot configuration cases additionally passed with an explicit CPU platform.
- Ascend snapshot, attention/CP, rotary, dispatcher and quantization suites:
  261 passed, 11 skipped.
- Focused connector snapshot/rebuild cases: 11 passed (32 subtests).
- Expert-map and pool backend suites: 116 passed (102 subtests).
- Ruff checks on changed Python files and `git diff --check` passed.
- New-baseline snapshot, DSA/DSA-CP/V4.1 metadata, RoPE/sleep ownership,
  ModelRunner V1/V2, MoE/quantization and connector rebuild groups passed.

These tests do not execute NPU kernels, native communication teardown or CRIU.
They do not establish numerical equivalence after restore. An existing unrelated
layerwise-connector reset-flag test failed during broader selection; the focused
snapshot/rebuild cases above passed.

Before deployment, verify cold-start/restore on the actual A2/A3 image with:

1. GLM W8A8C8/W4A8C8, DeepSeek V4 Flash and MTP, with the intended CP/MoE configuration.
2. V2 and any retained V1 deployment, graph capture enabled, multiple identical
   deterministic requests before and after restore, not just one request.
3. Centralized and distributed DP transport readiness, expert-map consistency,
   draft acceptance rate, and no unexpected recompilation.
4. P/D and pooled KV reuse for the configured connector backend.

Dynamic EPLB, additional multimodal/model-specific runtime state, and native
backend resource validity still require feature-specific hardware validation;
the static audit is not a proof that every configuration is covered.
