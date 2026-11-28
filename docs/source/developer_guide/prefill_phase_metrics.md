# Prefill phase coverage metrics

This branch requires the matching vLLM `feat/prefill-phase-utilization` branch
based on v0.30.0. Install both branches together. The implementation lives in
vLLM's independent `vllm.v1.metrics.prefill_phase` module; Ascend only adds the
admission hook to schedulers that override upstream `schedule()`:
`RecomputeScheduler`, `DyntraLBScheduler` and `ProfilingChunkScheduler`.
First-token and termination hooks are inherited from the upstream scheduler.

Add this option to the P engine's existing launch command:

```bash
--prefill-phase-metrics-interval 3
```

Do not use `--disable-log-stats`. The feature is disabled by default; an invalid
interval or an enabled feature with log stats disabled is rejected. There is
no fallback to the sum of request durations or device synchronization.

Query each engine's `/metrics` endpoint:

```promql
100 * vllm:prefill_phase_utilization
```

The gauge has `model_name` and `engine` labels. It reports the last completed
fixed window, starts at zero, and is refreshed during long prefills and idle
periods. Each DP engine has its own window origin and series. A frontend scrape
reads the existing Prometheus multiprocess files; it does not run the
calculation. No coordinator modification is needed to query the engine metric.

Overlapping request intervals count once. A chunked request remains active
until its first token or termination, including waits and preemptions. The
result is phase time coverage, not NPU execution utilization or a throughput
capacity estimate. It does not replace Motor's existing utilization metrics.

The implementation adds CPU lifecycle bookkeeping and one sampler thread per
enabled scheduler. It adds no NPU events or synchronizations. Validate throughput,
TTFT and TPOT with collection disabled and enabled on the target Ascend system
before rollout; CPU unit tests cannot establish production performance.
