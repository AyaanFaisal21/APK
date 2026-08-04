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

*(none yet — date and justify each)*
