# Roadmap

**Definition of done:** a public repo with reproducible code, raw results, a figure set, and a written report answering N1 with numbers — whichever way it goes.

**Total: 5–8 weeks, on the A10 already in hand.** No new hardware, no external artifacts on the critical path.

---

## Phase 1 — Your own persistent kernel (1–2 weeks)

Write a minimal persistent kernel from scratch. Grid fixed at 72 blocks (one per SM), global task queue via atomic fetch-add, synthetic tile work — a small GEMM per task, sized so tile duration lands in the 2–25 μs band.

**Do this before touching MegaQwen.** It is the entire risk strategy: you own every line, iteration is instant, and no external build can block you. MegaQwen enters at Phase 6, after results already exist.

*Done when:* it runs, tasks complete correctly, and tile duration is a tunable parameter.

## Phase 2 — Two workloads (3–5 days)

You need something to preempt *for*. Simplest form that isolates the mechanism: the persistent kernel chews through a long background stream while an **urgent task arrives from the host at random intervals**. Measure how long it waits.

Do not build a serving system. Build the mechanism and measure it.

*Done when:* urgent-task wait time is measurable end to end, with preemption not yet implemented (this is the baseline — how long you wait when the only option is to let the current work drain).

## Phase 3 — The yield, and the two numbers (1 week)

Naive version first: a global flag, each block polling at its own task boundary.

Measure:
- **Preemption latency** — flag set → last worker switched. `clock64()` on-device; you are measuring microseconds and host timers will not resolve it.
- **Steady-state overhead** — throughput with polling on versus off.

Compare against ExpertPlex's ~8% decode overhead and 1.12× prefill slowdown.

*Done when:* both numbers exist with distributions over ≥10k events.

## Phase 4 — The curve (1 week)

Sweep **polling frequency** (every tile / every *k* tiles / adaptive) against **resident block count** (16 → 72). Two inputs, two outputs: overhead and preemption latency.

**This is the paper's figure.** If latency grows with block count, A-N1 is confirmed and you have localized why clusters matter.

*Done when:* the surface is measured and plotted.

## Phase 5 — The safety question (1–2 weeks — this may become the real contribution)

ExpertPlex: *"Independent tile-boundary checks are unsafe because high-performance operations pipeline both warps and CTAs across tiles."*

That is a claim about **TMA-multicast pipelines on Hopper**. Your Ampere kernel pipelines differently — `cp.async`, no multicast, no cross-CTA shared memory. So: is naive independent polling actually unsafe here, or does the absence of cluster-level coupling make it safe by construction?

Test it. Deliberately yield mid-pipeline; check outputs against a float64 reference. If independent checks *are* safe on Ampere, that is an inversion worth writing up — the simpler hardware permits the simpler mechanism, and the cost of Hopper's performance features includes needing a more complex preemption protocol.

*Done when:* you can state, with evidence, whether per-block independent yield is sound on `sm_86` and under what conditions.

## Phase 6 — Realism, optional and last (1–2 weeks)

Port the yield into [MegaQwen](https://github.com/Infatoshi/MegaQwen) (MIT, CC 8.6+, Qwen3-0.6B). Only after Phases 1–5 have produced numbers. If the build fights you, you still have a paper.

Reproduce its own reported figure on the A10 first — that alone is a useful datapoint given the 3090's higher bandwidth.

---

## Kill criteria

- **Phase 1 runs past two weeks** — persistent kernels are harder than budgeted. Drop to a simpler synthetic and re-scope rather than pushing through.
- **Preemption latency is dominated by something unpredicted** (launch configuration, L2 contention, occupancy effects) — that is the finding. Pivot the paper toward it instead of treating it as noise.
- **Overhead under 1% everywhere and latency flat** — N1 confirmed cheaply. Write it short and publish; three weeks, real result.

## Measurement hygiene

Non-negotiable, and the reason anyone will believe the numbers:

- `nvidia-smi -pm 1`, lock SM and memory clocks with `-lgc` / `-lmc`, **read them back to confirm**. If the rental denies this, that fact changes the provider, not the methodology.
- Fix and record the power cap.
- Discard warmup iterations.
- ≥10k preemption events per configuration before quoting p99.
- On-device timing (`clock64()`) for preemption latency; host timestamps for end-to-end. Report both.
- Pin CPU cores, disable frequency scaling.
- Profile with Nsight for the timeline; take timing numbers from unprofiled runs.
- Report full distributions. Means hide the phenomenon under study.
- Raw results flushed to disk per configuration. A crash at hour three must not cost the run.

## What this is not

Not a reproduction of ExpertPlex — its code is unreleased, its testbed is 8×H800, and its models do not fit in 24 GB. This measures the *mechanism* on hardware that lacks the primitives ExpertPlex was built on, and the writeup will say exactly that in the first paragraph.
