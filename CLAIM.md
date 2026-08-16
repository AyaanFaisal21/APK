# Claim

**Filed 2026-08-04. Do not edit after data collection begins.** If it turns out to be wrong, it stays here and the writeup says so. Amendments go at the bottom, dated, with the reason.

---

## N1 — the naive claim

> Bounded tile-level preemption works on Ampere without cluster primitives at cost comparable to Hopper, because the preemption bound is dominated by tile execution time (2.2–25.3 μs per ExpertPlex) and the check-propagation epoch is negligible against it.

## A-N1 — the falsifier

> On Ampere, preemption latency is **propagation-bound rather than tile-bound**: the check epoch is comparable to or larger than tile execution time, and it degrades with resident block count in a way Hopper's cluster-local propagation does not. This is why ExpertPlex required thread-block clusters and DSMEM.

N1 is expected to be at least partly wrong. It is stated in its simplest falsifiable form on purpose — the point is to find out *where* it breaks, not to defend it.

---

## Pre-registered thresholds

Fix these now. Moving them after seeing data invalidates the result.

- **Preemption latency** = wall time from the host setting the flag to the last resident worker having switched tasks. Measured on-device with `clock64()`, not host timers. Report the full distribution; the headline is p99.
- **Comparable cost** means steady-state throughput overhead **≤ 10%** with polling enabled versus disabled — chosen against ExpertPlex's reported ~8% decode overhead and 1.12× prefill slowdown. Above 10%, N1 is falsified on the cost axis.
- **Propagation-bound** means the check epoch exceeds **25% of median tile execution time**, or grows by more than **2×** as resident blocks go 16 → 72.
- **Tile durations** are held in the 2–25 μs band to match ExpertPlex's regime. Results outside that band are reported separately, not blended in.
- Correctness after any yield is checked against a **float64 reference**, not against another run of the same kernel.
- Minimum **10,000** preemption events per configuration before any p99 is reported.

## What would make this uninteresting

- Overhead under 1% and latency flat across block counts, i.e. N1 holds trivially. Then write it short: the technique generalizes below Hopper, three weeks not eight, and that is a real finding about hardware requirements.
- The mechanism cannot be built safely at all on `sm_86`. Then the finding is *which* Hopper guarantee is load-bearing, which is the same paper from the other side.

## Scope conditions, stated rather than buried

- Single GPU. No multi-GPU, no NVSHMEM, no communication overlap.
- Synthetic tile workload first; a real model only in Phase 6, if at all.
- One architecture (`sm_86`). Claims about Hopper are cited from ExpertPlex, not measured here.
- Not a reproduction of ExpertPlex. Its code is unreleased and its testbed is 8×H800. This measures the *mechanism* on different hardware, and says so.

---

## Amendments

**2026-08-14 — post-registration constructs, registered before their data
is collected.** Reason: two external review rounds plus a verification
audit found no fault in the measurements but found that two registered
constructs did not measure what their names claimed (first-observer delay
as "check epoch"; the volatile flag protocol), and that the mid-pipeline
safety experiment never opened the race window it claimed to test. The
original thresholds above are unmoved and their registered readings will
be co-reported. The corrective constructs are:

- **A1 — Notification tail.** Per-block observation-delay distributions
  (min/median/p95/max across resident blocks, per event) replace
  first-observer delay as the propagation reading. A visibility
  microbenchmark (poll-only floor variant; loaded variant with matched
  residency, known cadence, DRAM traffic) isolates notification from
  residual tile work. Notification is a non-factor if the across-block
  max delay (floor variant) stays within 3 timer quanta at every block
  count 16–288 with no monotonic growth; it is a factor worth its own
  section if the loaded-variant max grows with block count beyond 10 us
  at 288.
- **A2 — Window-forced safety.** Issue-adjacent checkpoints (poll before
  any cp.async wait) with per-event yield-site logging. The
  drain-causality claim requires the naive discipline to corrupt at
  buffer-0-colliding sites under this geometry AND the drain discipline
  to stay at zero corruption under the identical geometry. Statistic for
  zero-failure cells: one-sided 95% rule-of-three bound (3.0e-4 at
  0/10,000).
- **A3 — Reserved-capacity Pareto.** Urgent p99 versus background
  throughput across reserved residency slots and urgent work sizes;
  cooperative handoff must be compared against spatial headroom before
  any "preferable" language is used. Registered now; not yet run.
- **A4 — A100 anchor (promoted from optional).** Prediction: the
  mechanism decomposition (epoch flat, tile term dominant) holds on
  sm_80; the floor-variant max delay stays within 2 quanta; constants
  are expected to shift and will not be blended with GA102 numbers.
