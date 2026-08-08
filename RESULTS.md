# Results

This document gives the results of the study in Simplified Technical English.
The full data is in `NOTEBOOK.md` and `results/`. The claims and thresholds
are in `CLAIM.md`. The thresholds were set before the first test. They did
not change.

## 1. Test configuration

- GPU: NVIDIA A10, 72 SMs, compute capability 8.6 (Ampere).
- The SM clock is locked at 1200 MHz. The lock is confirmed under load.
  Higher lock values do not hold: the 150 W power limit decreases the clock.
- The unit of work is one tile: a 64×64 SGEMM block. The parameter K sets
  the tile time.
- Each quoted p99 comes from 10,000 or more events.
- A float64 reference validates every output. No output is compared with
  another run of the same kernel.
- The device timer has a resolution of 1.024 microseconds.
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
same shared memory. 10,000 forced yields per discipline.

- **Drain discipline** (one wait instruction before reuse): **0 of 10,000
  outputs are corrupt.** The 95% confidence bound is 3.7 × 10⁻⁴.
- **Naive discipline** (no wait): 0 of 10,000 outputs are corrupt. This
  result applies to this pipeline geometry only. The in-flight copies land
  approximately 5 microseconds before the yield point. A wider race window
  is necessary before a stronger claim.
- **Poison control** (one value made incorrect on purpose): 10,000 of
  10,000 outputs are caught. The detector is valid.
- With checks inside the tile, the preemption latency is **18.4
  microseconds (median) and 23.6 (p99)** at full occupancy. The cost is
  5.0%. This latency is inside the 2.2–25.3 microsecond band that
  ExpertPlex reports on Hopper hardware.
- One real fault was found and removed. A missing barrier in the check
  protocol let warps of one block separate. The block then wrote incorrect
  tiles until the kernel stopped. This fault occurred in approximately half
  of all runs. After the correction, 21 of 21 runs are clean.

## 7. Conclusions

1. Claim N1 holds on both pre-registered axes. Bounded tile-level
   preemption operates below Hopper.
2. The preemption bound on Ampere is the worst in-flight tile time, not
   the flag propagation time.
3. The latency sequence of the study: 957 microseconds (no preemption) →
   ~66 (boundary yield) → 18.4 (mid-pipeline yield, drain discipline).
4. The dangerous failure mode on Ampere is control divergence in the check
   protocol, not a data race in the pipeline. The data race did not occur
   in 10,000 tries. The control fault occurred in half of all runs before
   correction.

## 8. Known limits

- One GPU model (A10). One kernel shape. One tile size (K = 64) on the
  scaling surface. One node.
- The timer resolution (1.024 microseconds) sets the floor on the
  propagation readings.
- The naive-discipline result is limited to this pipeline geometry.
- Synthetic tiles only. A real model (Phase 6) is optional future work.
