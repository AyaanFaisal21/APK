# Adaptive Persistent Kernels on Ampere

**The question:** ExpertPlex achieves bounded tile-level preemption inside a persistent kernel using Hopper thread-block clusters, distributed shared memory, and TMA multicast. None of those exist below `sm_90`. What preemption granularity and overhead are achievable without them?

**Status as of 2026-08-04:** **Phases 1–3 complete.** Phase 1: calibrated tile (K ∈ {32, 64, 128} → 6.7–17.8 μs at 1200 MHz; higher clock locks throttle at the 150 W cap). Phase 2: drain-time baseline — saturated residency makes an urgent tile wait **957 μs p50 / 1476 μs p99** (n=2000; = remaining drain, corr 0.9934); one open slot collapses it to 76 μs. Phase 3: the naive independent yield costs **0.80% steady-state overhead** (at K=64, 288 blocks, poll-every-tile; falsifier: 10%) and preempts in **59.4 μs p50 / 70.7 μs p99** over 10k device-timestamped events — urgent e2e 20.5 μs on device-timeline endpoints, ≈47× over baseline (≈30× with matched host-observed endpoints). The latency is the **max in-flight co-resident tile**, confirmed by direct measurement (tile-wall p99 at 4-way residency = 57.6 μs ≈ the observed 58.4 μs spread); the check epoch itself is ~1 μs. **Phase 4: the surface is measured and N1 holds on both pre-registered axes.** Across blocks {16…288} × poll period {1,2,4,8} (10k events/cell, 0 anomalies): overhead ≤ 3.4% worst-case across three builds (falsifier: 10%); the check epoch stays flat at 1–3 μs and *shrinks* 16→72 (A-N1 required >2× growth). Latency does grow 8.4× to saturation — but entirely through the tile term stretching under SM co-residency, not propagation. On Ampere, bounded tile-level preemption generalizes below Hopper, with the bound denominated in contention-stretched tile time. Figure and summary in `results/summary/`; the audit trail (6 subverted problems and counting) is in the NOTEBOOK. Next: Phase 5, the safety question.

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
| [`src/tile.cuh`](src/tile.cuh) | The synthetic tile all three binaries schedule and measure — the calibration unit |
| [`src/persistent.cu`](src/persistent.cu) | Phase 1: persistent kernel, fetch-add queue, K→duration calibration, float64-verified |
| [`src/urgent_baseline.cu`](src/urgent_baseline.cu) | Phase 2: the drain-time baseline across three residency shapes |
| [`src/yield.cu`](src/yield.cu) | Phases 3–4: the yield instrument — latency (dual set modes), overhead, the sweep |
| [`scripts/`](scripts/) | `lock_clocks.sh` (step zero on any rental), `calibrate.sh`, `phase2/3/4.sh` (one per measurement) |
| [`results/summary/`](results/summary/) | Committed surface CSV + figure (raw per-event CSVs stay in gitignored `results/raw/`) |

## The shape of the result

Either outcome is publishable as a characterization:

- **N1 holds** — bounded preemption generalizes below Hopper, and every Ampere GPU in the world can do it.
- **N1 fails** — preemption latency on Ampere is propagation-bound rather than tile-bound, which localizes exactly why ExpertPlex needed cluster primitives.

There is no version of this where a correct measurement is uninteresting.

## Standard

Report what fails. The two prior repos ([TransformerOp](../TransformerOp), [Cornfield](../cornfield)) document five attention kernels that lost to SDPA, an op-level 3.6× that became 0.95× end-to-end, and a float4 optimization that bought nothing but exposed a silent-launch-failure bug. Same standard here.
