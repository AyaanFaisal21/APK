# Adaptive Persistent Kernels on Ampere

**The question:** ExpertPlex achieves bounded tile-level preemption inside a persistent kernel using Hopper thread-block clusters, distributed shared memory, and TMA multicast. None of those exist below `sm_90`. What preemption granularity and overhead are achievable without them?

**Status as of 2026-08-04:** **Phases 1–2 complete.** Phase 1: persistent kernel runs on the A10, sanitizer-clean, float64-verified; tile duration tunable via K (6.7–17.8 μs in-band for K ∈ {32, 64, 128} at 1200 MHz; the 150 W cap throttles through higher clock locks — see NOTEBOOK). Phase 2: the drain-time baseline is measured — with all 288 residency slots occupied, an urgent tile waits **957 μs p50 / 1436 μs p99** (= remaining drain, corr 0.9934); with even one slot open, 76 μs; the floor is one co-resident tile (~25 μs). That ~50–90× gap is what Phase 3's yield has to close.

**Hardware:** NVIDIA A10 (GA102, `sm_86`, CC 8.6, 24 GB GDDR6, ~600 GB/s, 72 SMs).

## Navigation

| File | What it holds |
|---|---|
| [`CLAIM.md`](CLAIM.md) | N1 and its falsifier, with pre-registered thresholds. **Read before writing code.** |
| [`BACKGROUND.md`](BACKGROUND.md) | What persistent / mega / adaptive kernels are, and which Hopper primitives the mechanism leans on |
| [`FINDINGS.md`](FINDINGS.md) | Verified literature facts with citations, separated from unverified ones |
| [`ROADMAP.md`](ROADMAP.md) | Six phases, 5–8 weeks, on hardware already in hand |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated entries: prediction before each run, outcome after |
| [`papers/`](papers/) | Nine PDFs plus a reading index (PDFs gitignored) |
| [`src/persistent.cu`](src/persistent.cu) | Phase 1: persistent kernel, fetch-add queue, calibrated synthetic tile, float64-verified |
| [`scripts/`](scripts/) | `lock_clocks.sh` (step zero on any rental) and `calibrate.sh` (K sweep) |

## The shape of the result

Either outcome is publishable as a characterization:

- **N1 holds** — bounded preemption generalizes below Hopper, and every Ampere GPU in the world can do it.
- **N1 fails** — preemption latency on Ampere is propagation-bound rather than tile-bound, which localizes exactly why ExpertPlex needed cluster primitives.

There is no version of this where a correct measurement is uninteresting.

## Standard

Report what fails. The two prior repos ([TransformerOp](../TransformerOp), [Cornfield](../cornfield)) document five attention kernels that lost to SDPA, an op-level 3.6× that became 0.95× end-to-end, and a float4 optimization that bought nothing but exposed a silent-launch-failure bug. Same standard here.
