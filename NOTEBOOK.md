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

## 2026-08-04 — Phase 2: drain-time baseline, written and predicted

**Doing:** `src/urgent_baseline.cu` — background persistent kernel (tasks
cycle the tile grid, idempotent writes) + single-tile urgent kernel launched
from the host on a max-priority stream at a random point in the drain
(U(0.2, 0.8) of estimated drain, fixed seed). No preemption exists; measuring
what the urgent task pays for that. All cross-kernel timing via
`%globaltimer` (ns, coherent across SMs/kernels) — `clock64()` would be
meaningless across two kernels on different SMs. Three grid shapes, because
"the GPU is busy" is not one condition: `sat` (every residency slot filled —
the true baseline), `sat-1` (exactly one slot open), `1persm` (Phase 1's
72-block shape). Also refactored the tile into `src/tile.cuh`, shared by both
binaries.

**Predicted:** occupancy comes back 3 blocks/SM (register-bound) → 216 slots.
Drain of 20k K=64 tasks ≈ 2.5–3.5 ms. `sat`: urgent e2e ≈ remaining drain at
arrival — p50 ≈ 1.2–1.8 ms, i.e. **~100× the ~10–30 μs tile time**, scaling
~1:1 with remaining tasks; gap (urgent start − bg last exit) within
[−1 tile, +50 μs] — urgent may start slightly *before* full drain as blocks
exit one by one at queue exhaustion. `sat-1`: e2e p50 ≈ 30–80 μs despite
"100% utilization" — one open slot collapses the baseline; gap large
negative. `1persm`: similar, 25–60 μs. Urgent exec: ~10 μs solo, 20–35 μs
when sharing an SM with resident bg blocks. Stream priority does nothing in
`sat` mode (priorities reorder pending blocks; they cannot evict resident
ones).

**Got:** sanitizer clean, both kernels float64-verified in every mode.
Occupancy came back **4 blocks/SM → 288 slots** (predicted 3). 500 trials
per mode, K=64, 20k bg tasks:

| mode | blocks | e2e p50 | e2e p99 | gap p50 | urgent exec p50 |
|---|---|---|---|---|---|
| sat | 288 | **957 μs** | 1436 μs | −23.6 μs | 19.5 μs |
| sat-1 | 287 | 76.3 μs | 95.9 μs | −954 μs | 64.5 μs |
| 1persm | 72 | 26.1 μs | 32.5 μs | −1680 μs | 15.4 μs |

The baseline claim holds exactly: in `sat`, corr(e2e, remaining tasks at
arrival) = **0.9934**, slope 88.1 ns/task = remaining drain across 288
workers. The urgent task pays the whole remaining queue, ~50–90× a tile
time, and stream priority (−5 vs 0, confirmed set) does nothing about it.
Gap distribution is tight around −24 μs: urgent starts about one
co-resident-tile-time before the *last* bg block exits, as predicted.

**Surprised by:** (1) Drain 1.66 ms, not the predicted 2.5–3.5 ms — and the
reason is worth keeping: 4-way co-residency raises per-SM throughput ~1.7×
over one block/SM (1persm drain 2.83 ms for identical work). The naive tile
hides its own latency badly; co-residency does it for free. Consequence:
Phase 3's resident-block sweep (16→72) changes *throughput*, not just
propagation, and overhead percentages must be computed against same-shape
baselines. (2) `sat-1` execution: the open slot let urgent start immediately
(gap −954 μs) but the tile ran **64.5 μs vs 15.4 μs** in 1persm — 4×
slower sharing an SM with 3 bg blocks. "Got scheduled" and "ran at speed"
are different quantities; e2e still 12× better than sat. (3) The three
modes span **26 → 76 → 957 μs** for the same urgent tile on the same "busy"
GPU — residency shape, not utilization, is the entire story. Phase 1's
72-block grid would have produced a baseline 37× too optimistic if used
naively for `sat`.

**Next:** Phase 3 — the naive yield: global flag, per-block polling at task
boundaries, preemption latency (flag set → last worker switched) and
steady-state polling overhead vs this baseline. The number to beat is
957 μs p50 / 1436 μs p99; the floor is one tile (~25 μs co-resident).

## 2026-08-04 — Phase 3: the naive yield, written and predicted

**Doing:** `src/yield.cu` — global device-memory flag, every block polling
independently at its own task boundary (the exact mechanism ExpertPlex calls
unsafe; safety is Phase 5, this is cost and latency). One long-running
kernel hosts all 10k events. Set-instant timestamped two ways that must
agree up to PCIe visibility: `--set dev` (block 0 sets the flag at scheduled
instants, logs `%globaltimer` at the write — same clock as the observers,
zero cross-domain error; primary) and `--set host` (the CLAIM.md wording;
host `steady_clock` mapped onto `%globaltimer` via a 300-rep spin-kernel
calibration, measured idle — stated limitation). Claimer block executes the
urgent tile for e2e comparability with Phase 2. Overhead mode: ABAB
interleaved drains of 200k tasks, poll on/off, 20 reps each.

**Predicted:** overhead: one volatile L2 read per ~25 μs co-resident tile →
**≤ 1.5%**, far under the 10% falsifier. Latency (dev-set, 288 blocks,
K=64): bound is one residual tile + check epoch; last-of-288 residual ≈ a
full co-resident tile → **p50 ≈ 22–28 μs, p99 ≈ 38–50 μs** (tile p99 tail ×
max-of-288 pushes it above the single-tile p99). First-observer delay ≈
tile/288 ≈ sub-μs. Spread ≈ latency (first observer ~immediate, last ≈ full
tile). Host-set latency = dev-set + calibration residual: within **±5 μs**
of dev-set if the idle-measured offset (predicted p50 ≈ 8–15 μs, PCIe write
+ memcpyAsync enqueue) survives load. e2e set→urgent-done ≈ first-observer
+ exec ≈ **30–45 μs vs 957 μs baseline, ~25×**. Anomalies (block missing an
event at 150–400 μs gaps): zero. Verify passes; sanitizer clean.

**Got:** first full-size verify run FAILED with all 9.4M elements NaN —
*subverted problem #2, caught by its own guard.* `cudaMemset`'s fill runs as
a kernel; null-stream work does not order against a non-blocking stream; and
a **saturated persistent grid blocks the fill kernel outright** (Phase 2's
own lesson, biting the harness). The 0xFF fill therefore executed *after*
the drain and erased every result. It passed under compute-sanitizer
(serialized execution) and Phase 2 dodged it by launch timing — a race that
was always there. Fix: explicit `cudaDeviceSynchronize()` between guard
fills and launch, applied to both binaries. Without the NaN guard this would
have been invisible or, worse, a silent stale-data pass.

After the fix: verify PASS (max rel err 2.97e-06), sanitizer clean, 10,000
events per set mode, **0 anomalies, 0 unset**.

**The two numbers (288 blocks, K=64, ~25.4 μs mean co-resident tile):**
- **Polling overhead: 0.80%** (17.905 vs 17.763 ms median drain, ABAB ×20).
  N1's cost axis holds with 12× margin over the 10% falsifier.
- **Preemption latency (dev-set, primary): p50 59.4 / p90 64.5 / p99 70.7 /
  max 81.9 μs.** Decomposition: first observer 1.02 μs p50; the rest is
  observation spread (58.4 μs p50) — i.e. waiting for the *slowest in-flight
  tile*. Even p01 is 51 μs: this is the floor, not a tail.
- Urgent e2e set→done: **20.5 μs p50 / 35.8 p99 vs 957/1436 baseline — 47×.**
  Note e2e < latency: the first observer claims and finishes the urgent tile
  while stragglers are still reaching their boundaries. "Urgent served" and
  "kernel fully redirectable" are different quantities; ExpertPlex's bound
  concerns the latter.

**Host-set vs dev-set:** host-set read p50 49.1 μs with first-observer delay
**−9.7 μs** — physically impossible, so the idle-measured calibration offset
is ~10 μs too large (likely PCIe link-power wake on an idle link; under load
the link is hot). Corrected: 49.1 + 9.7 = 58.8 vs dev 59.4 — **agreement
within 0.6 μs**. The dual-instrument design paid for itself; dev-set is the
primary number, and idle calibrations do not transfer to loaded runs
(subverted problem #3, the plausible-wrong-number kind — host-set alone
would have quietly reported 17% better latency). Cosmetic: the calibration
print shows raw epoch offsets (~1.8e15 μs); print deltas next time.

**Surprised by:** latency is **2.2× the prediction** (59 vs 22–28 μs), and
the reason is a real finding: the "one tile execution" term in the
preemption bound is not the mean tile time but the **max over all in-flight
co-resident tiles**. At 4-way SM sharing, individual tile walls spread far
beyond the 25.4 μs throughput average (the max of 288 residuals sits at
~58 μs, ~2.3× mean). ExpertPlex's 2.2–25.3 μs tile-boundary figure is a
*boundary-interval* statistic; the bound's effective tile term at
saturation is the contention-stretched worst case. Check epoch (~1 μs) is
currently ≪ 25% of tile time — N1 holds at this block count; the scaling
test (does it grow >2× from 16→288 blocks?) is Phase 4's axis.

**Next:** Phase 4 — the surface: polling period × resident block count
(extend the sweep past ROADMAP's 72 to 288, where residency actually
saturates), overhead and latency per cell. Then the A-N1 verdict.

## 2026-08-04 — Audit of Phases 1–3 (pre-Phase-4 checkup)

Full re-read of code, claims, and methodology, plus validation runs.
Original entries above stay as written; errors are amended here, dated,
per the CLAIM.md convention.

**Errors found in the write-ups (measured numbers unaffected):**

1. *Phase 1 "≈38% of fp32 peak" is wrong — it is ≈19%.* The prediction
   contained an arithmetic error carried into the outcome: a K=128 tile is
   2·64·64·128 = **1.05 MFLOP**, not 2.10. The binary always used the
   correct formula (printed GFLOP/s are right); only the notebook's
   efficiency claim doubled it. Corrected: 59 GFLOP/s/SM ≈ 19% of the
   307 GFLOP/s per-SM peak at 1200 MHz — **worse** than the 25–35% guess,
   not "slightly better." The flattering conclusion survived because it
   flattered; that is exactly what this audit is for.
2. *Phase 1 "fixed-size tail" is wrong.* p99/p50 across K = 32…512 is
   1.24, 1.27, 1.24, 1.21, 1.04 — a **proportional ~24% tail through
   K=256** that collapses only at K=512. Not a fixed-size tail, and not
   monotone shrinkage.
3. *Phase 3 "47×" has an endpoint mismatch.* Baseline e2e is host-arrival →
   host-observed completion (includes launch + sync legs); Phase 3 e2e is
   device-set → device-done. Order of magnitude stands; apples-to-apples
   (adding ~10 μs of host legs) is ≈30×. Report both going forward.
4. *Phase 2 baseline p99 was under-sampled.* n=500 → n=2000 on the
   race-fixed binary: p99 1436 → **1476 μs** (+2.8%); p50 957.3 → 957.5
   (stable to 0.02% across sessions — good reproducibility datapoint).
   Verification also re-passes post-race-fix, re-legitimizing Phase 2.

**Validation runs:**

- *Tile-wall distribution at 288 blocks* (`audit_tilewall_288.csv`,
  n=9216): min 10.7 / p50 17.0 / p90 41.9 / **p99 57.6 / max 66.2 μs** —
  heavy-tailed, p99/p50 = 3.4. Phase 3's observation spread p50 (58.4 μs)
  sits on the wall p99 (57.6 μs), as max-of-~288-in-flight predicts. The
  "latency = worst in-flight co-resident tile" interpretation is now
  **measured, not inferred**.
- *Clock under this load class:* 1200 MHz held.

**Caveats carried to Phase 4 (design changes queued):**

- Dev-set setter is phase-locked to block 0's boundaries: block 0's own
  observation is always ≈ one full tile after set, which could prop up the
  floor. Host-set agreement at p50 says the max is usually another block,
  but Phase 4 must log the argmax block id + block 0's own delay to
  quantify it.
- "Saturated" is kernel-shape-specific (registers bind: 4 × 256 threads ×
  ~60 regs). A much smaller urgent kernel might co-schedule even at "sat";
  the 957 μs baseline applies to same-shape urgent work. (The gap = −23.6 μs
  proves this tile could not fit — claim scoped, not falsified.)
- Hygiene gaps vs ROADMAP: no CPU pinning / governor control on the rental
  — affects host-side timestamps only; all Phase 3 headline numbers are
  %globaltimer (wall ns) and SM-clock-independent by construction, which is
  worth stating as a design property. No warmup discard in Phase 3
  overhead/latency modes (medians robust; add warmups anyway). No Nsight
  timeline captured yet — do one profiled (non-timing) run in Phase 4.
- Overhead 0.80% is one cell (K=64, 288 blocks, poll-every-tile); the
  Phase 4 surface scopes it.

**Checked and sound:** barrier logic around `s_task`/`s_flag` (uniform
branches, no divergent `__syncthreads`); missed-event obs semantics (late
blocks logged truthfully); overhead A/B fairness (same grid/residency,
template strips the poll, ABAB interleave); float64 verification math; NaN
and log-sentinel guards; idempotent duplicate tile writes; counter overflow
margins; CLAIM.md untouched since data collection began.

## 2026-08-04 — Phase 4: the surface, written and predicted

**Doing:** sweep resident blocks {16, 36, 72, 144, 216, 288} × poll period
{1, 2, 4, 8 boundaries} — 24 cells, each with 10k dev-set events (+100
warmup, discarded) and a 20-rep ABAB overhead pair (2 warmup pairs
discarded; tasks scaled ~700/block so per-rep wall stays constant). Audit
changes wired in: per-event argmax block and block-0 self-delay now logged,
so the setter phase-lock caveat gets quantified instead of hand-waved.
Adaptive polling dropped from the plan: at 0.80% overhead for poll-every-1
there is nothing for adaptivity to save; the k-sweep exists to show the
latency cost of skipping checks, not to find a sweet spot.

**Predicted:** *Overhead surface ≈ flat and boring*: ≤1% everywhere,
decreasing with k, no strong block-count dependence (one L2 read per k
tiles per block cannot move a compute-bound drain). *Latency vs blocks at
k=1 is the A-N1 verdict and I predict A-N1 fails in an interesting way*:
p50 grows ~13 → ~59 μs from 16 → 288 blocks, but the growth lives entirely
in the tile term — max in-flight wall stretches with SM co-residency
(measured: solo p50 10.3, 4-way wall p99 57.6) — while the check epoch
(first-observer delay, and lat minus max-wall) stays ~1–3 μs at every
block count: <25% of median tile and <2× growth 16→288. So naive
independent polling on Ampere is **tile-bound, not propagation-bound**, and
the degradation with worker count is contention physics, not flag
propagation — the opposite localization from what A-N1 (via ExpertPlex's
Hopper reasoning) expects. *Poll period*: latency p50 grows ≈ +(k−1)/2 ×
mean wall per skipped boundary (at 288 blocks: ~59, ~70, ~95, ~145 μs for
k=1,2,4,8); overhead saved is ≪0.5 points — poll-every-1 strictly
dominates. *Setter*: block 0 is last observer in ≪5% of events at 288
(its one-tile self-delay ~p50 of wall ≈ 17–25 μs, far below the 58 μs
max-of-288), rising at 16 blocks where fewer rivals exist. If instead the
epoch grows with blocks or block 0 dominates the max, my Phase 3 story is
wrong and the notebook says so.

**Got:** *(pending)*

**Surprised by:** *(pending)*

**Next:** run `scripts/phase4.sh`, build the figure, file the N1/A-N1
verdict against the pre-registered thresholds.

**Got:** three instrument battles before the data could be trusted, then a
clean 24-cell surface with **0 anomalies** anywhere.

*Subverted problem #4 — occupancy is not a constant of the source tree.*
The first sweep silently compiled to **3 blocks/SM** (the poll-period
parameter added register pressure), so `--blocks 288` orphaned 72 blocks
that never became resident: 10,000/10,000 events flagged anomalous at 288.
The anomaly counter caught it; Phase 3's binary genuinely was 4/SM and its
numbers stand. Fixes: `__launch_bounds__(THREADS, 4)` makes residency a
compile-time contract; latency mode now refuses grids larger than residency.

*Subverted problem #5 — the pin then manufactured a fake overhead.* Under
the 64-reg cap the POLL variant spilled (36 B stores/tile) while POLL=false
did not; solo-occupancy cells read ~10% "overhead" that was really
asymmetric spill traffic, nearly independent of poll period — the tell.
Templating the poll period and moving thread-0 bookkeeping to shared cut it
to 8 B; and resurrecting the *committed Phase 3 binary* from git showed its
POLL variant had spilled 16 B all along (its 0.82% at 288 blocks reproduces
Phase 3's 0.80% exactly). Conclusion, worth stating in the writeup: **there
is no spill-free poll variant at 4 blocks/SM — the register cost of yield
machinery at the occupancy cliff is part of the mechanism's price on
sm_86**, and it hides under co-residency (≤1% at 288) but shows at solo
occupancy (≤3.4% worst measured across three builds).

*Subverted problem #6 — grid == SM count is a bistable shape.* B=72
overhead swung ±10–12% between runs; raw drains are bimodal (7.1 vs 8.0 ms,
sticky per launch sequence) — block placement sometimes doubles up SMs and
idles others at exactly 1 block/SM. B=72 overhead is excluded with cause;
B<72 and B≥144 are tight (±0.15–2%).

The surface (`results/summary/phase4_surface.csv`, figure alongside),
setter-excluded latency p50/p99 at poll-every-1:
16 → 8.2/11.3 μs · 36 → 10.2/11.3 · 72 → 11.3/12.3 · 144 → 16.4/20.5 ·
216 → 33.8/41.0 · 288 → 68.6/90.1. Latency ≈ ×k for poll-every-k at every
block count (at 288: 68.6 → 133 → 262 → 520 μs). First-observer delay p50
stays 1–3 μs at every cell. Setter-last%: 89.6% at 16 blocks → 0% at ≥144 —
the audit's caveat was real and is now excluded by construction.

**The pre-registered verdict (thresholds from CLAIM.md, unmoved):**
- *Cost axis:* every clean cell across three builds ≤ 3.4%, vs the 10%
  falsifier → **N1 holds**.
- *Propagation-bound test:* check epoch (first-observer) p50 1–3 μs; it
  *shrinks* 16→72 (3.1 → 1.0 μs), nowhere near the >2× growth A-N1
  requires → **A-N1's scaling clause fails; N1 holds**. One honest wrinkle:
  at B=16 the 25%-of-tile clause technically trips (3.07/10.24 = 30%) —
  but the cause is the 1.024 μs `%globaltimer` quantum (all values are
  multiples of it; instrument resolution, discovered this session) plus
  sparse pollers, not propagation, and the clause was written to detect
  propagation. Reported as a threshold-crossing with cause, not hidden.
- *The real Ampere story:* latency grows 8.4× from 16→288 blocks, but the
  growth is entirely the **tile term stretching under SM co-residency**
  (walls: 10.3 μs solo → p99 57.6 μs at 4-way), while the check epoch stays
  flat. ExpertPlex's Hopper reasoning localizes preemption cost in
  propagation; on Ampere it lives in contention. Bounded tile-level
  preemption generalizes below Hopper — with the bound denominated in
  *contention-stretched* tile time.

**Surprised by:** (1) Saturation-point latency is build-sensitive: p50 at
288 blocks spanned 54.3 → 59.4 → 68.6 μs across three code variants of the
same mechanism (shared-memory bookkeeping shifts tile walls). B ≤ 144 cells
reproduce to the quantum across builds. Quote the 288 number as a range
with build caveat, never as a point. (2) The globaltimer quantum (1.024 μs)
— the audit had waved at "ns granularity"; wrong, and it sets the floor on
the epoch measurement. (3) Overhead at high residency is ~0 within noise —
polling is not merely cheap, it is *hidden* by co-resident warps.

**Next:** Phase 5 — the safety question: deliberately yield mid-pipeline on
`sm_86` and check float64 correctness; ExpertPlex's unsafety claim may not
carry to an architecture without cross-CTA pipelining.

## 2026-08-06 — Deep-dive audit #2 (pre-Phase-5), box migration pending

Full re-derivation of every published number plus a fresh methodological
sweep, prompted before Phase 5 work begins. CLAIM.md verified untouched
since the scaffold commit (git history — the pre-registration stands).

**Internal consistency: all cross-phase checks pass.**
- Phase 1 K-sweep decomposes to ~3.0 μs fixed cost + 114–124 ns/K through
  K=256 (the K=512 slope break is the documented L2-reuse falloff).
- Phase 2's drain slope (88.1 ns/task × 288 = 25.4 μs/tile) matches Phase
  4's co-resident walls; 1persm drain (2827 μs) matches Phase 1's solo
  tile prediction (2869 μs) to 1.5%.
- Phase 4 latency scales ×1.94–2.25 per poll-period doubling at every
  block count, and B=16 excl-setter p50 (8.19 μs) sits below one solo
  wall (10.33 μs), exactly where max-of-16 residuals belongs.

**Two disclosure gaps found, now on the record:**
1. *The overhead surface's poll-period axis is not interpretable as poll
   rate.* At B=16/36, P=1 reads LOWER than P=2 (0.91% vs 1.93%) because
   the PE=1 template variant spills 8 B while PE≥2 variants spill 16 B —
   codegen, not polling. Only the P=1 columns (both binaries) are
   apples-to-apples. The cost verdict already uses the worst case across
   all variants (3.4%), so the verdict stands; the axis caveat must
   appear in the writeup.
2. *The check-epoch readings sit at instrument resolution.* Every
   first-observer p50 is 1–3 quanta of the 1.024 μs globaltimer tick.
   The honest A-N1 statement is "epoch ≤ ~3 μs at every block count,
   growth below detectability" — not "the epoch shrinks." The A-N1
   verdict is unchanged under any reading (no >2× growth is visible at
   any resolution), but the earlier phrasing implied more precision than
   the instrument has.

**Judged sound on re-examination:** the latency definition (boundary
observation = redirectable instant), setter exclusion, B=72 exclusion
with cause, endpoint-mismatch disclosure (30× matched / 47× device),
worst-case-across-builds overhead quoting, n≥10k for every quoted p99,
scope conditions in CLAIM.md matching what was actually measured.

**Blocked on hardware (new box 150.136.146.214 unreachable on port 22 —
likely still provisioning or a firewall rule):** the sanitization-pass
rebuild check (`make && make sanitize`), clock lock + under-load
read-back on the new instance, and a cross-instance reproduction
spot-check of the 288-block cell — worth doing regardless, since a
second physical A10 reproducing the surface is a result in itself.

**Next:** when the box answers: rebuild, sanitize, reproduce one latency
+ one overhead cell, remove the README untested-notice. Then Phase 5.

## 2026-08-06 — Sanitization pass verified; cross-instance reproduction

**Doing:** the deferred hardware checks from audit #2, on a fresh
`gpu_1x_a10` (a different physical card from every prior number). The
lossy SSH path was bypassed by having the box clone the public GitHub
repo and run the whole sequence detached under nohup.

**Predicted:** build clean, sanitizer clean (the sanitization pass was
comments plus one printf and one include). Reproduction within the
documented ranges: Phase 1 K=128 p50 within a few percent of 17.77 μs;
288-block latency p50 inside the 54–69 μs build/session range; overhead
at saturation ~0.

**Got:** all of it. Build clean; **compute-sanitizer 0 errors on all
three binaries** — the README untested-notice is retired. Reproduction
on the second card:

| quantity | original | second A10 | delta |
|---|---|---|---|
| Phase 1 K=128 tile p50 / p99 | 17.77 / 22.01 μs | 18.05 / 22.10 μs | +1.6% / +0.4% |
| float64 max rel err (same seed) | 5.307e-06 | 5.307e-06 | bit-identical |
| 288-block latency p50 / p99 | 68.61 / 90.11 μs | 66.56 / 78.85 μs | inside range |
| overhead at saturation, P=1 | −0.10% | −0.00% | ~0 both |
| setter-last fraction at 288 | 0.0% | 0.0% | match |

The tile agreement doubles as the under-load clock evidence: at an
unlocked boost (1695) the tile would read ~12.6 μs; 18.05 μs is only
consistent with the 1200 MHz lock holding.

**Surprised by:** nothing in the numbers — the first fully boring run
this project has had, which after five subverted problems is itself the
news. Operational note for the log: Lambda reuses IPs (stale host key)
and a freshly restarted instance passed small SSH bursts while stalling
bulk transfers for ~15 minutes; clone-from-GitHub + nohup + short-burst
polling is the resilient pattern for flaky rentals.

**Next:** Phase 5 — the safety question (yield mid-pipeline, float64
verdict), per the audit's remaining-work list.

## 2026-08-06 — Phase 5: mid-pipeline yield safety, written and predicted

**Doing:** `src/midyield.cu` — a cp.async double-buffered pipelined tile
(KBP=16, two stages, one commit group per chunk; the sm_80+ staging real
kernels use), persistent queue as before, dev-set events. At a forced
yield the CLAIMER interrupts its own pipeline mid-flight — outstanding
async copies targeting the very smem the urgent tile is about to use —
and runs the urgent tile in that smem under one of three disciplines:
`drain` (cp.async.wait_all + barrier first), `naive` (no wait; the
in-flight group races the urgent staging), `poison` (drain, then corrupt
one staged float — instrument control). The abandoned bg tile restarts
from scratch, so C stays correct by construction in every discipline;
the safety verdict lives entirely in the per-event urgent outputs, all
10k of which are captured and checked against a float64 reference.
Non-claimers only observe — one block per event tests the discipline.
Stage-granularity polling also gives the latency payoff measurement vs
boundary polling, and overhead runs {off, boundary, stage} complete the
cost picture.

**Predicted:** `drain`: **0/10,000 corrupt** (rule-of-three upper bound
3.7e-4) — on sm_86 one wait instruction makes independent mid-pipeline
yield safe, because nothing outside the CTA can hold a reference into
its smem; the ExpertPlex hazard needs cluster-level coupling the
hardware cannot express. `naive`: **20–50% corrupt** — the in-flight
group targets buf0 half the time and lands within the urgent staging
window most of that; nonzero is the load-bearing prediction (it
reproduces the hazard class inside one CTA and proves the instrument
sees real races, not just planted ones). `poison`: **100%**. C passes
in all disciplines. Latency at 288 blocks, stage polling: **p50
20–35 μs** vs ~55–75 μs boundary — mid-tile checks break the
whole-tile floor; first-observer stays 1–2 quanta. Overhead vs poll-off
on the pipelined tile: stage ≤ 3.5%, boundary ≤ 1.5%; same codegen
caveat as Phase 4 (compare within-binary only). Occupancy stays 4/SM
(16.5 KB smem, pinned launch bounds). Risk noted before running: if
`naive` shows ZERO corruption at 10k events, the honest reading is that
the race window is too small at this tile size, not that no discipline
is needed — widen the window (larger KBP) before concluding anything.

**Got:** *(pending)*

**Surprised by:** *(pending)*

**Next:** run `scripts/phase5.sh` on the A10.
