# Results

This document gives the results of the study in Simplified Technical English.
The full data is in `NOTEBOOK.md` and `results/`. The claims and thresholds
are in `CLAIM.md`. The thresholds were set before the first test. They did
not change.

## 1. Test configuration

- Primary GPU: NVIDIA A10, 72 SMs, compute capability 8.6 (Ampere).
- Anchor GPU: NVIDIA A100-SXM4-40GB, 108 SMs, compute capability 8.0.
- The SM clock is locked and confirmed under load. A10 boxes lock at
  1200 MHz. One A10 part holds only 1050 MHz because of power sag. The
  A100 locks at 1410 MHz. Constants are per part and are not blended.
- The unit of work is one tile: a 64×64 SGEMM block, or a 128×128
  tensor-core block in the generalization test. The parameter K sets
  the tile time.
- Each quoted p99 comes from 10,000 or more events.
- A float64 reference validates every output. No output is compared with
  another run of the same kernel.
- The device timer has a resolution of 1.024 microseconds on both parts.
- A second A10 reproduced the primary results. The agreement is within 2%.

## 2. Phase 1: tile calibration

- One tile takes 6.7 to 17.8 microseconds for K = 32, 64, and 128, with one
  block per SM.
- When four blocks share one SM, tile times increase and spread. The p99
  tile time is then 57.6 microseconds.

## 3. Phase 2: baseline without preemption

Condition: all 288 residency slots are full.

- An urgent task waits 957 microseconds (median, n = 2000). The wait is
  equal to the remaining queue work (correlation 0.99).
- Stream priority does not decrease the wait. Priorities order pending
  blocks. They do not remove resident blocks.
- One free residency slot decreases the wait to 76 microseconds.

## 4. Phase 3: the yield at tile boundaries

Mechanism: one flag in device memory. Each block reads the flag at each
tile boundary. Each block acts on the flag independently.

- The poll cost is 0.8% at full occupancy. The worst measured cost across
  all code versions is 3.4%. The pre-registered limit is 10%.
- The preemption latency is 59.4 microseconds (median) and 70.7 (p99).
  This value changes between code versions: 54 to 69 microseconds. Quote
  it as a range.
- The urgent task completes in approximately 21 microseconds on the device
  timeline. With endpoints that match the baseline, the gain is
  approximately 30 times.

## 5. Phase 4: the scaling surface

Test: resident blocks 16 to 288, poll period 1 to 8 boundaries, 10,000
events per cell, zero anomalies.

- The cost stays at or below 3.4% in every valid cell. **Claim N1 holds on
  the cost axis.**
- The flag-propagation time stays at 1 to 3 microseconds at every block
  count. It shows no growth. The pre-registered A-N1 test requires more
  than 2× growth. **A-N1 fails. Claim N1 holds on the latency axis.**
- The latency grows 8.4× from 16 to 288 blocks. The cause is tile-time
  growth under SM sharing. The cause is not flag propagation.
- Two cells are not valid and are excluded with cause: the 72-block
  overhead cell (unstable block placement) and the poll-period axis of the
  overhead surface (compiler differences between code variants).

## 6. Phase 5: safety of the yield in the middle of a pipeline

Test: a tile with cp.async double-buffered staging. The yield interrupts
the pipeline while copies are in flight. The urgent tile then uses the
same shared memory. Yield-site logging records the pipeline position of
every claim. Across all runs on both GPUs, 28,859 yields hit a true
buffer conflict.

- **Drain discipline** (one wait instruction before reuse): **0 of 28,859
  conflicting yields corrupt an output.** The one-sided 95% bound is
  3.0 × 10⁻⁴ per 10,000 events.
- **Naive discipline** (no wait): 0 outputs are corrupt. This result
  applies to this pipeline geometry only. The PTX memory model does not
  permit the naive form. The test shows non-occurrence, not permission.
- **Poison control** (one value made incorrect on purpose): 10,000 of
  10,000 outputs are caught. The detector is valid.
- With checks inside the tile, the preemption latency is **17.4
  microseconds (median) and 22.5 (p99)** at full occupancy in the
  reference build. The cost is at most 5.3% across builds and parts.
  This latency is inside the 2.2–25.3 microsecond band that
  ExpertPlex reports on Hopper hardware.
- One real fault was found and removed. A missing barrier in the check
  protocol let warps of one block separate. The block then wrote incorrect
  tiles until the kernel stopped. This fault occurred in approximately half
  of all runs. After the correction, 21 of 21 runs are clean.

## 7. The sm_80 anchor (A100)

Test: one clock-locked hour on an A100. All predictions were registered
before the run.

- The decomposition transfers. The notification floor is one timer
  quantum, flat to 432 blocks. Stage-level handoff is 16.4 microseconds
  (median) and 21.5 (p99) at saturation. The safety signature repeats.
- Two predictions about memory headroom failed. The loaded observation
  delay did not improve on 2.6× the bandwidth: it degraded from 97
  microseconds (A10) to 3.78 milliseconds. The co-residency latency
  growth did not shrink. The tails are architectural, not headroom
  effects.

## 8. Reserved capacity, the alternative design

Test: hold R residency slots free for urgent work instead of yielding.

- Reservation is cheap in throughput: at most 1.5% at R = 32.
- Reservation cliffs in waves. An urgent task of U blocks completes in
  ceil(U/R) waves. At R = 32, U = 16 the urgent task completes in 111
  microseconds (median).
- Reservation is denominated in resource envelopes, not slots. An
  unpinned urgent kernel that needed 79 registers received nothing from
  32 reserved slots (fault 13, kept as a finding).
- The cooperative path completes 16× more urgent work for 1.8× the time
  at approximately 3% cost. It holds the better operating points.

## 9. The scheduler oracle

Test: count every task execution and every urgent execution on the
device. Check exact invariants after each run.

- Approximately 185 million task executions across four runs. Every task
  executed exactly once. Every urgent event executed exactly once. Zero
  invariant violations.
- This closes the blind spot that output checks alone leave open:
  a scheduler that skips or repeats idempotent work.

## 10. The tensor-core generalization

Test: the same scheduling skeleton behind a CUTLASS-class kernel. One
task is a 128×128 fp16 tensor-core tile with a three-stage cp.async
pipeline and 56.8 KB of shared memory. Occupancy is 1 block per SM. No
second block is present to hide any stall.

- Correctness gates pass: float64 check, poison control 10,000 of
  10,000, sanitizers clean, zero register spills.
- The safety signature transfers whole: 0 corrupt outputs in 10,000
  yields per discipline. The oracle holds exactly: 14.3 million tasks
  per arm, each executed exactly once.
- The registered cost prediction for per-stage checkpoints **failed**.
  Measured cost is 13.7 to 14.7% against a predicted 4% and a
  registered 10% limit. The cause is measured by decomposition: at 1
  block per SM, no co-resident work hides the checkpoint's flag read
  (0.87 microseconds, removable by software pipelining) or its barrier
  (0.24 microseconds, not removable).
- Tile-boundary checkpoints stay cheap: 2.8 to 3.4%.
- The trade is now measured on both ends: detection in 9.2 microseconds
  (median) for 14% cost, or 35.8 microseconds for 3% cost.

## 11. Conclusions

1. Claim N1 holds on both pre-registered axes. Bounded tile-level
   preemption operates below Hopper.
2. The preemption bound on Ampere is the worst in-flight tile time, not
   the flag propagation time. This decomposition repeats on sm_80.
3. The latency sequence of the study: 957 microseconds (no preemption) →
   ~66 (boundary yield) → 17.4 (mid-pipeline yield, drain discipline).
4. The dangerous failure mode on Ampere is control divergence in the
   check protocol, not a data race in the pipeline. The data race did
   not occur in 28,859 conflicting tries. The control fault occurred in
   half of all runs before correction.
5. Checkpoint cost follows occupancy. With co-resident blocks the cost
   is at most 5.3%. At 1 block per SM, per-stage checkpoints cost 14%
   and the economical choice moves to tile boundaries at 3%.
6. Cluster hardware is necessary only where a handoff-sensitive
   dependency spans blocks. Without that dependency, one relaxed atomic
   flag and block-local quiescence are sufficient, and the oracle and
   safety gates show it exactly.

## 12. Known limits

- Two GPU models, one vendor. Hopper numbers are cited, not measured.
- One kernel family on the synthetic surfaces. One tensor-core design
  point (one stage depth, one register footprint).
- The timer resolution (1.024 microseconds) sets the floor on the
  propagation readings.
- The naive-discipline result is limited to the measured geometries.
- The mechanism of the loaded observation delay is not isolated.
- The small-R negative reservation cost is an open anomaly. It is
  confounded with per-part clock sag.
- A real model (Phase 6) is optional future work.
