# Notebook

One entry per session. **Prediction before the run, outcome after.** Predicting first is what makes a surprise legible later — without it you will reconstruct a story in which you expected whatever happened.

Record subverted problems especially: the small ones that produce plausible wrong numbers. The stale-buffer bug in Cornfield — a kernel whose launch silently failed, passing `allclose` off a recycled buffer and clocking 0.016 ms for doing nothing — cost hours and is now the most useful anecdote in that repo. Those are the entries worth having.

Format:

```
## YYYY-MM-DD — short title

**Doing:** what and why
**Predicted:** the number or behavior expected, before running
**Got:** what actually happened
**Surprised by:** the gap, if any
**Next:** the single next action
```

---

## 2026-08-04 — Project set up, nothing built

**Doing:** assembled the paper corpus, wrote background, filed N1/A-N1 in `CLAIM.md`.

**Predicted:** n/a — no code yet. Phase 1 is ~200 lines of CUDA: fixed 72-block grid, atomic fetch-add task queue, synthetic GEMM tiles tuned to 2–25 μs.

**Got:** nine papers in `papers/`, five documents. Confirmed the A10 is `sm_86`/CC 8.6 — same compute capability as a 3090, so MegaQwen should run; and confirmed ExpertPlex's mechanism depends on `sm_90` clusters, DSMEM, and TMA, which the A10 lacks. That absence is the project.

**Surprised by:** ExpertPlex being a much cleaner match to the question than expected — it names the exact unsafety problem (`"independent tile-boundary checks are unsafe"`) that Phase 5 tests on hardware where the reasoning may not carry.

**Next:** `touch persistent.cu`. Phase 1.

## 2026-08-04 — Phase 1 kernel written, not yet run

**Doing:** wrote `src/persistent.cu` (~300 lines): grid defaults to one block
per SM, global task queue via atomic fetch-add, each task one 64×64 SGEMM tile
with K as the duration knob. Per-tile `clock64()` records, float64 host
reference from day one, and 0xFF-fill on C and the tile log before every pass
so a tile that never runs surfaces as NaN / block=-1 instead of passing off
stale memory (the Cornfield lesson, designed in rather than debugged in).
Plus `scripts/lock_clocks.sh` (step zero: lock, **read back**, record) and
`scripts/calibrate.sh` (K sweep 32→512). Nothing has compiled yet — no nvcc
on the Mac; first build happens on the A10.

**Predicted:** written before first run, on paper only. At 1695 MHz locked:
K=128 tile = 2.10 MFLOP; guessing the naive smem kernel sustains 100–150
GFLOP/s per SM (~25–35% of the SM's ~434 GFLOP/s peak), so **p50 ≈ 14–21 μs**
— upper half of the band. K sweep 32→512 roughly linear: ~4 → ~60+ μs, so
K=512 lands OUT OF BAND and the usable knob range is K≈32–256. With locked
clocks, p99/p50 < 1.10. Queue cost invisible at this granularity: one
atomicAdd per ~15 μs across ≤72 blocks. Verification passes with max rel err
≤ 1e-5 (fp32 accumulation over K≤512 vs float64). Tasks/block spread within
±20% of ideal 32.

**Got:** ran same day on a Lambda `gpu_1x_a10` (driver 580.105.08, CUDA 12.8).
Compiled first try; `compute-sanitizer` clean; every configuration verified
against the float64 reference.

*Subverted problem #1, and the one that matters:* **a clock lock is not a
constant clock.** `-lgc 1695` was accepted and read back as locked — then
under sustained load the GPU sat at the 150 W power cap (which is also
`power.max_limit`, no headroom to raise) and DVFS throttled to 1290 MHz
straight through the "locked" setting. 1350 MHz held ~8 s at 147–150 W, then
dipped to 1335 as the die warmed. **1200 MHz held exactly, ~130 W, ~20 W of
margin, for the full sustained run.** Every cycles→μs conversion made at the
nominal lock would have been silently ~29% optimistic — plausible wrong
numbers, zero errors raised. All measurement below is at `-lgc 1200`, and the
under-load read-back (not the idle one) is now the required evidence.

The sweep, p50 tile duration at 1200 MHz (n=2304 per point, final pass,
warmups discarded):

| K | p50 μs | p99 μs | max rel err | band |
|---|---|---|---|---|
| 32 | 6.67 | 8.30 | 1.3e-06 | in |
| 64 | 10.33 | 13.11 | 3.0e-06 | in |
| 128 | 17.77 | 22.01 | 5.3e-06 | in |
| 256 | 33.60 | 40.57 | 9.1e-06 | out |
| 512 | 78.30 | 81.40 | 1.7e-05 | out |

Queue balanced perfectly: 32/32 tasks per block, all 72 blocks. Aggregate
3.9–4.3 TFLOP/s (≈38% of the A10's fp32 peak at 1200 MHz) from the naive
smem kernel — prediction said 25–35%, so slightly better than guessed. In
clock-independent terms: predicted 23.7–35.6k cycles per K=128 tile, got
21.3k, just under the range.

**Surprised by:** (1) The power-cap throttle above — the prediction was
denominated in a clock the card cannot sustain. (2) p99/p50 = 1.24 at K=128
vs the predicted <1.10 — but the spread *shrinks* as tiles grow (1.04 at
K=512), so it is a fixed-size tail, not proportional jitter; worth knowing
before Phase 3 quotes preemption p99s at this granularity. (3) K=512 is
superlinear (78.3 μs vs ~65 extrapolated from K≤256): per-tile A/B traffic
outgrows L2 reuse. Irrelevant to the band but a reminder that "tile duration
∝ K" has a validity range.

**Next:** Phase 2 — the urgent-task arrival harness and the drain-time
baseline. Knob range established: K ∈ {32, 64, 128} spans 6.7–17.8 μs;
K≈192 should land ~25 μs if the band edge is ever needed.
