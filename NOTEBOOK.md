# Notebook

One entry per session. The prediction is written before the run. The
result is written after. Faults get a number and stay on the record.

Full narrative versions of these entries are preserved for future agents
outside the repo, and in git history.

## Fault index

Faults that made wrong numbers possible. All were caught by the
instruments this repo requires: float64 references, NaN fills, poison
controls, anomaly counters, and read-back under load.

| # | Fault | Correction |
|---|---|---|
| 1 | A clock lock is not a constant clock. The 150 W cap throttled through a 1695 MHz lock | Lock at 1200 MHz. Read back under load, not at idle |
| 2 | Null-stream memset fills do not order against a non-blocking stream. A saturated grid blocks the fill until drain | Explicit device sync between guard fills and launch |
| 3 | Idle host-to-device clock calibration is ~10 us too large under load | Device-set timestamps are primary. Host-set agrees within 0.6 us after correction |
| 4 | Register pressure silently cut occupancy from 4 to 3 blocks per SM and orphaned 72 blocks | Occupancy pinned with launch bounds. Latency mode refuses grids above residency |
| 5 | A pinned register cap made the poll variant spill and faked a 10% overhead | Poll period is a template constant. Overhead compares are within-binary only |
| 6 | Grid size equal to SM count gives bistable block placement, drains 12% apart | The 72-block overhead cell is excluded with cause |
| 7 | A zero exit code over a lossy link is not delivery. Files "sent" to the box had not arrived | Transfers are checksum-gated and resent until the remote md5 matches |
| 8 | A missing barrier let warps of one block diverge across two barrier sites. Whole tiles went silently corrupt in ~half of runs | Branch conditions use a barrier-protected snapshot of shared state. 21 of 21 runs clean after the fix |

---

## 2026-08-04. Setup

- Corpus of nine papers assembled. N1 and A-N1 registered in `CLAIM.md`.
- Confirmed: the A10 is `sm_86`. ExpertPlex needs `sm_90` clusters, DSMEM,
  and TMA. That absence is the project.

## 2026-08-04. Phase 1: tile calibration

- Predicted: p50 14 to 21 us at K = 128, from 25 to 35% of fp32 peak.
- Result: p50 17.8 us at 1200 MHz. K = 32, 64, 128 give 6.7, 10.3,
  17.8 us. All cells pass the float64 check (max rel err <= 1.7e-05).
- Fault 1 found here. Efficiency is ~19% of peak, not the 38% first
  written (a 2x arithmetic slip, amended in the audit).
- Tail is proportional (~24%) through K = 256. The K = 512 slope break is
  L2 reuse falling off.

## 2026-08-04. Phase 2: baseline

- Predicted: urgent wait equals remaining drain, ~100x one tile.
- Result: at full residency (288 blocks) the urgent task waits 957 us
  (median). Wait tracks remaining queue at correlation 0.9934. One open
  slot cuts it to 76 us. Stream priority changes nothing.
- Occupancy is 4 blocks per SM, not the predicted 3.
- Co-residency raises per-SM throughput 1.7x. "Scheduled" and "fast" are
  different: the sat-1 urgent tile starts at once but runs 4x slower.

## 2026-08-04. Phase 3: boundary yield

- Predicted: latency p50 22 to 28 us (one mean tile). Overhead <= 1.5%.
- Result: overhead 0.80%. Latency p50 59.4 us, p99 70.7 us over 10,000
  device-timestamped events. Urgent completes in ~21 us on the device
  timeline, ~30x over baseline with matched endpoints.
- The 2.2x latency miss is the finding: the bound is the worst in-flight
  co-resident tile, not the mean tile. Confirmed by direct measurement
  (tile-wall p99 at 4-way residency is 57.6 us).
- Faults 2 and 3 found here.

## 2026-08-04. Audit 1

- Two write-up errors amended (efficiency 2x, tail shape). Measured
  numbers unaffected.
- Baseline p99 firmed at n = 2000: 1476 us. p50 stable to 0.02%.
- Endpoint mismatch in the "47x" disclosed. Matched value is ~30x.

## 2026-08-04. Phase 4: the surface

- Predicted: overhead flat and small. Latency grows through the tile
  term. Propagation stays flat. A-N1 fails.
- Result: 24 cells (blocks 16 to 288, poll period 1 to 8), 10,000 events
  per cell, zero anomalies. Worst overhead across three builds: 3.4%
  against the 10% limit. Propagation stays at 1 to 3 us at every block
  count. Latency grows 8.4x to saturation, all of it tile-time growth
  under SM sharing. N1 holds on both axes.
- Faults 4, 5, and 6 found here.
- The device timer ticks at 1.024 us. Propagation readings sit at that
  resolution floor.

## 2026-08-06. Audit 2

- Every published number re-derived. Cross-phase checks agree: drain
  slope, tile walls, and solo predictions match within 1.5%.
- `CLAIM.md` verified untouched since registration (git history).
- Two disclosure gaps recorded: the overhead poll-period axis is
  compiler-confounded (only poll-every-1 columns compare), and the
  propagation readings sit at timer resolution.

## 2026-08-06. Cross-instance reproduction

- A second physical A10 reproduced Phase 1 within 1.6% (float64 max rel
  err bit-identical), the 288-block latency inside its documented range,
  and ~0 overhead at saturation. Sanitizer clean on all binaries.
- The tile agreement doubles as under-load proof of the 1200 MHz lock.

## 2026-08-06. Phase 5: mid-pipeline yield safety

- Instrument: cp.async double-buffered tile. Forced yields land while a
  copy group is outstanding into the shared memory the urgent tile then
  uses. Three disciplines, 10,000 events each, every urgent output
  checked against float64.
- Predicted: drain 0 corrupt. Naive 20 to 50% corrupt. Poison 100%.
  Risk note filed in advance: naive at zero means the window is too
  small at this geometry, not that no discipline is needed.

| discipline | corrupt | worst rel err | bg C |
|---|---|---|---|
| drain | 0 / 10,000 | 1.4e-06 | PASS |
| naive | 0 / 10,000 | 1.4e-06 | PASS |
| poison (control) | 10,000 / 10,000 | 9.9e-01 | PASS |

| poll granularity | lat p50 | lat p99 | overhead |
|---|---|---|---|
| stage (mid-pipeline) | 18.4 us | 23.6 us | 5.0% |
| boundary (whole tile) | 50.2 us | 58.4 us | 2.0% |

- Statement: per-block independent yield is sound on `sm_86` with the
  drain discipline. Stage checks put preemption at 18.4 us median at
  full saturation, inside ExpertPlex's 2.2 to 25.3 us Hopper band, for
  5% cost.
- Naive at 0 of 10,000: the copy group physically lands ~5 us before the
  checkpoint. The claim is scoped to this geometry. Window widening is
  future work.
- Faults 7 and 8 found here. Fault 8 is itself a result: the danger on
  Ampere is control divergence in the check protocol, not the data race
  the Hopper reasoning points at. The race never fired in 10,000 tries.
  The control fault fired in half of all runs before the fix.
- Latency arc of the study: 957 us (none) to ~66 us (boundary) to
  18.4 us (mid-pipeline).

## Next

The writeup. Optionally Phase 6 (port to a real model).

## 2026-08-08. External review response: construct corrections

Two review rounds (recorded in full for future agents outside the repo)
found no fault in the measurements but two faults in the constructs and
several in the claims. Corrections, in the reviewer-approved order.

**Post-registered reanalysis of existing Phase 4 events** (no new data;
`scripts/reanalyze_spread.py`, deterministic, identity-checked on all
240k events):

- Observation spread (last minus first, across blocks) at poll-every-1:
  7.2, 9.2, 9.2, 15.4, 31.7, 67.6 us for 16 to 288 blocks.
- Key fact: 36 to 72 blocks doubles the polling CTAs at constant
  occupancy and moves the spread 1.00x. Growth appears only when
  co-residency rises (1.67x, 2.07x, 2.13x at 2, 3, 4 blocks per SM).
- Spread scales 1.94x to 1.99x per poll-period doubling at 288 blocks.
- Reading: consistent with residual-checkpoint-work domination. Not yet
  causal. The visibility microbenchmark decides.

**Code corrections (fault 9, class: unspecified synchronization):**

- The generation flag was a volatile protocol. CUDA C++ volatile
  carries no inter-thread synchronization guarantee. The flag is now a
  device-scope atomic, relaxed load and relaxed store, with the
  contract documented in `tile.cuh`: the atomic communicates only the
  generation value; nothing is published through it (claims use
  atomicCAS; setGT is host-read after sync). Relaxed is by design:
  release was rejected because no acquire pairing exists to serve.
- The setter's threadfence is removed under the same contract.
- Applied to yield.cu and midyield.cu. SASS comparison and a full
  surface rerun are required before the old numbers are cited as
  current; prediction below.

**New instruments:**

- `--dump-obs`: full per-event, per-block observation matrices to disk
  (the review assumed these survived; only first/last/spread did).
- `visibility` mode: notification latency with tile work removed.
  Floor variant (poll-only) and loaded variant (same 4/SM residency,
  known FMA cadence, DRAM-resident traffic). Self-terminating kernel,
  no host in the loop, fixed seed.

**Predictions, filed before the reruns:**

- Atomic vs volatile surface: every latency cell within one timer
  quantum of the volatile numbers; overhead within 0.3 points. SASS
  for the hot loop differs by at most the load opcode.
- Visibility floor: Dmax across blocks stays at or below 3 timer
  quanta (3.1 us) at every block count 16 to 288. No growth trend.
- Visibility loaded: Dmax p50 at or below 10 us at 288 blocks; if it
  grows with block count beyond that, the propagation story is wrong
  and A-N1 partially revives. This is the falsifiable one.
- Per-block quantiles at 288 blocks, poll-every-1: D50 near half the
  co-resident wall p50, D95 near the wall p99, iid order-statistics
  fit expected to FAIL at SM level (blocks sharing an SM are
  correlated); a good fit would itself be a finding.

**Claim retirements (effective now, ahead of the paper rewrite):**
"generalizes below Hopper" narrows to sm_86/A10 evidence; "for every
Ampere GPU in the world" is retired; "the difficulty lives in control
flow, not data" is withdrawn until the cp.async window is actually
opened; "bounded" splits into bounded checkpoint granularity versus
measured handoff latency; ExpertPlex comparisons become order-of-
magnitude statements, not equivalence claims.

**Next:** on the A10: SASS diff, sanitize, atomic surface rerun with
obs dumps, visibility floor and loaded sweeps. Then the reserved-
capacity Pareto and the cp.async window widening.

## 2026-08-14. Scrutiny audit received; window-forcing built; new box

External verification audit (third scrutiny pass; full report linked in
the briefing) verified every external quote and number, confirmed no
scoop, and found 9 blocking draft defects. Measurement-relevant items
land here; paper-prose items queue for the rewrite.

- Fault 10 (accounting): the mid-pipeline safety claim counted all
  10,000 stage yields as collision geometry; the pipeline gives ~1/4
  (buffer-0 outstanding only at site c=1 of 0..3), and sites were
  unlogged. Fix: per-event yield-site logging (claimSite, one int).
- Fault 11 (explanation): the naive-zero mechanism story ("issued a full
  stage-compute ~5 us before the yield") contradicts the code; the
  outstanding group is issued in the same iteration as the checkpoint.
  Explanation withdrawn; superseded by the window-forcing experiment.
- Barrier comment corrected (audit B9): arrival-count completion is not
  an sm_70 change; divergent __syncthreads is documented UB; the
  observed silent phase-shift is a permitted UB outcome, not documented
  semantics.
- CLAIM.md amendments A1-A4 registered (dated) before their data:
  notification tail, window-forced safety, Pareto, A100 anchor.
- New poll mode `issue`: checkpoint before any cp.async wait. At sites
  c = 0, 1, 2 a buffer-0 group is outstanding at the yield by
  construction; site 3 is buffer-1 only.

Predictions, filed before the window-forcing runs:

- issue + naive: corruption strictly > 0, concentrated at sites 0-2;
  site 3 near zero. Overall 15-60% of claimed events at colliding
  sites. This is the load-bearing prediction: if it stays zero, the
  drain-causality claim cannot be made at this tile geometry at all.
- issue + drain: 0 corrupt. issue + poison: 100%.
- Claim-site distribution approximately uniform over c = 0..3 in both
  stage and issue modes.
- stage + naive rerun with site logging: still 0 corrupt; ~2,500
  events at the one colliding site (c=1), reported precisely per B7.
- Surface and visibility predictions filed 2026-08-08 stand unchanged.

## 2026-08-15. Review-2 batch: results (third A10, us-west)

Ops note first: three stacked faults blocked the new box (instance key
provisioning failed; browser pastes corrupted authorized_keys until a
tr-cleaned rewrite; this Mac's ssh-agent was empty after a reboot, so
the client aborted after the server had accepted the key). Verbose sshd
logging localized it. Fault 12: an agent-empty client failure reads
identically to a server rejection from the client side.

All gates clean: 4x sanitizer 0 errors, all verifies pass, poison
verify gate trips as required, 0 anomalies in every run.

**A-flag verdict (atomic vs volatile):** hot poll kernel SASS is
instruction-identical (4400 = 4400); every poll-1 latency cell
reproduces within 1 quantum on a third physical A10. Large-period
cells differ 2.7 to 3.3% (run variance at 100s of us; the 1-quantum
prediction was mis-specified for those scales). Overhead at solo
occupancy reads ~3.0% (was ~0.9%) because the poll-OFF baseline got
16 instructions FASTER in the atomic build; the audit's whole-kernel
overhead framing is vindicated. Worst cell 3.02%, falsifier 10% intact.
The 72-block cell is sane on this box (+1.3%, no bistability observed).

**A1 verdict (notification tail):**

| blocks | floor Dmax p50/p99 | loaded cadence | loaded Dmax p50/p99 |
|---|---|---|---|
| 16-144 | 1.02 / 1.02 us | 7.0-7.5 us | 7.2-8.2 / 8.2-9.2 us |
| 216 | 1.02 / 1.02 | 8.5 | 21.5 / 23.6 |
| 288 | 1.02 / 1.02 | ~9-10 | 97.3 / 110.6 |

Floor: device-scope notification reaches all 288 independently polling
CTAs within ONE timer quantum, flat at every count. A1's floor
criterion holds maximally: propagation is free on sm_86.
Loaded: the registered growth trigger FIRED (97 us at 288). The tail
is congestion physics on the polling path, not propagation (the floor
proves that); the DRAM-streaming microbench is harsher than the real
tile workload, which sits at 64.5 us at the same count. The two
variants bracket the real kernel. Own section in the paper.

**A2 verdict (window-forced safety):** issue-adjacent naive with
buffer-0 groups outstanding at the yield: **6,376 true-collision
events, 0 corrupt** (sites 0-2 of the 10,000; site split 18/22/24/36%,
stage mode uniform 26/26/24/23%). Poison: 10,000/10,000. The
drain-causality claim CANNOT be made at this geometry: the registered
zero branch is taken. Mechanism reading: the claim path (two barriers +
CAS, >= 1-2 us) exceeds L2-hit cp.async completion (~0.3 us), so
copies land before the urgent tile's first store; the hazard window is
unreachable when sources are L2-resident. The documented lever if
pursued: DRAM-pressured copies (the loaded-visibility result proves
congestion stretches memory operations ~10x). Drain remains free
insurance: SASS-identical, latency-identical (17.4 us p50 both).

**iid prediction WRONG, in the strong direction:** the pooled-F^N
order-statistics model predicts Dmax p50 66.6 us at 288 blocks;
empirical is 64.5 (3%). Cross-block delay correlations are ~0
(same-SM-candidate pair -0.001, neighbor pair +0.020). Straggler
tails at saturation are quantitatively predicted by independent draws;
the predicted SM-level correlation does not exist at measurable size.

**Next:** reserved-capacity Pareto (A3), then the paper rewrite on the
corrected constructs. All summary tables now generate from committed
CSVs by script (scripts/analyze_review2.py).

## 2026-08-15. A3 Pareto: instruments extended, predictions filed

Reserved arm (urgent_baseline): --reserve R holds slots out of the
grid; --urgent-blocks U launches a U-block urgent job, block b computes
tile b (distinct work), job e2e = arrival to last-block completion.
Cooperative arm (yield): --urgent-tiles U makes urgent work a mini-queue
popped by yielding workers; U = 1 reproduces the old claimer exactly.

Predictions, before any run:
- Reserved throughput loss is linear in R: ~0.35% per slot (R/288),
  measured drain rate confirms within 1 point.
- Reserved urgent e2e: U <= R gives 40-90 us (one wave). U > R runs
  ceil(U/R) waves: R=1, U=16 predicted >= 350 us, approaching the
  no-reserve baseline regime.
- Cooperative urgent e2e grows weakly with U (poppers parallelize):
  p50 within 2x from U=1 to U=16; stays under 100 us at U=16.
- Cooperative throughput cost: the ~0-3% poll overhead plus under 1%
  of capacity in urgent work at these arrival rates.
- The Pareto verdict predicted: cooperative dominates every reserved
  point with U > R; reserved wins only the U=1, small-R corner, and
  only on latency, never on throughput. Falsifier: if reserved R=1/U=1
  beats cooperative U=1 by more than 2x on p99, spatial headroom owns
  the small-urgent regime and the paper says so.

## 2026-08-15. A3 Pareto: results

Gates clean (2x sanitizer 0 errors, multi-tile verifies pass). One
instrument fault before valid data:

- Fault 13: the unpinned urgent kernel compiled to 79 registers and
  could not fit the 16,384 registers a 3-way-occupied SM has spare.
  Symptom: urgent jobs waited full drain DESPITE reserved slots, at
  every R, indistinguishable from no reservation. Fix: launch-bounds
  pin to 64 registers. Kept as a result: reservation is denominated in
  resource envelopes, not slot counts. An urgent kernel that outgrows
  the envelope gets nothing from the reservation.

Corrected table (p50/p99 us, 10k trials or events per cell; full table
in results/summary/pareto.csv):

| config | rate loss | e2e p50 | e2e p99 |
|---|---|---|---|
| reserved R=1, U=1 | -5.4% | 95.4 | 113.7 |
| reserved R=1, U=16 | -4.0% | 964.0 | 1338.0 |
| reserved R=8, U=16 | +0.5% | 195.3 | 215.4 |
| reserved R=32, U=16 | +1.5% | 111.3 | 124.9 |
| coop U=1 | +2.9% | 22.5 | 37.9 |
| coop U=16 | +3.4% | 41.0 | 79.9 |

Predictions vs got:

- Wave scaling: confirmed exactly. R=1/U=16 gives 964 us (predicted
  >= 350, ceil(U/R) waves); R=8/U=16 gives 2 waves; latency flattens
  once R >= U.
- Cooperative flat in U: confirmed. 22.5 to 41.0 us p50 from U=1 to 16
  (1.8x for 16x the work); p99 stays under 80 us.
- Falsifier did not fire: cooperative beats reserved R=1/U=1 by 3x on
  p99 (37.9 vs 113.7), not the reverse.
- One prediction INVERTED: the throughput axis. Predicted cooperative
  cheaper; measured reservation cheaper. Removing 1 to 4 blocks from
  the 288-block grid RAISES drain throughput (negative loss, -5.4% at
  R=1): the last co-resident blocks cost more in wall-stretch than
  they contribute. Cooperative pays its ~2.9% poll overhead always.
  The saturated grid is not throughput-optimal; the Pareto verdict is
  therefore a true tradeoff, not dominance: reservation buys cheap
  throughput and 100 us-class latency IF the job fits both the slot
  count and the resource envelope; cooperative handoff buys 22 to 41 us
  latency, flat in urgent size, for ~3% throughput, with no envelope
  constraint and no wave cliff.

A1, A2, A3, and the A-flag rerun are now all answered. The measurement
set for the revised paper is complete.

**Next:** the paper rewrite on corrected constructs.

## 2026-08-15. Environmental audit of the review-2 and A3 batches

Checked on the box that produced them: ECC volatile counts zero, no Xid
or driver errors since boot, no other compute processes, host idle
(load 0.08, memory and disk free), temperature 41 to 49 C throughout.

Observation 14, the one real finding: this A10 is a leakier part than
box 1. The 1200 MHz lock sags to 1035-1080 MHz at the 150 W cap under
peak load (SwPowerCap bit active); box 1 held 1200 at ~130 W. The
under-load read-back was not repeated on this box before its batches:
a protocol lapse, caught by this audit. Bound measured directly: a
sag-proof 1050 MHz lock reproduces the 288-block cell at p50 70.7 us
versus 68.6 (mixed clock) versus 64.5 (recorded); the entire clock
regime shifts timing numbers by at most ~10%, inside the documented
cross-build band. No verdict has less than 3x margin. Rule amended in
practice: the under-load read-back is per-BOX, not per-project.

## 2026-08-15. A100 anchor (A4): plan and predictions, filed before any run

Purpose, ranked: (1) literature comparability. GPreempt, Hummingbird,
LithOS, and MPK all report on A100; one run puts in-kernel handoff on
the same part, from user space, on a stock driver. (2) External
validity: sm_80 GA100 and sm_86 GA102 bracket the Ampere family; if
the decomposition holds on both, "Ampere-class" is earned. (3) The L2
regime probe: A100's 40 MB L2 holds this workload's entire footprint;
GA102's 6 MB did not.

Predictions:
- Occupancy register-bound at 4 blocks/SM; saturation = 432 blocks.
- Timer quantum: measure first; assumed 1.024 us until shown otherwise.
- Floor visibility Dmax <= 2 quanta, flat to 432 blocks.
- Co-residency wall stretch SMALLER than GA102's 2.3x (predict p99
  wall <= 1.6x the throughput mean) on L2 and bandwidth headroom.
- Boundary yield p50 at saturation 45 to 75 us; stage 15 to 25 us.
- Loaded visibility Dmax at saturation <= 25 us (vs 97 on the A10).
  If it explodes anyway, congestion tails are architectural rather
  than bandwidth-bound. Falsifiable either way.
- Safety: naive 0 corrupt (stronger L2 regime), poison 100%.
- Overhead <= 3% in every cell.
- Constants are reported per part and never blended with GA102 numbers.

Budget: ~2 hours on gpu_1x_a100_sxm4. Order: lock hunt with under-load
read-back FIRST (observation 14), quantum measurement, occupancy check,
then calibration, surface subset (4 block counts), visibility, safety
trio, Pareto spot, slack.

## 2026-08-15. A100 anchor (A4): results

Fault 15 first: the initial batch compiled for sm_86 and every kernel
launch failed with no-kernel-image, yet all completion markers printed
green (the runner only gated the build). The clock hunt "verified"
1410 MHz against an idle GPU. Caught by inspecting gate output rather
than trusting markers; runner now hard-aborts on any gate failure.
Rule: a completion marker is not a gate.

Valid run (ARCH=sm_80): all gates enforced and clean. 1410 MHz held
under real load (400 W part). Occupancy 4 blocks/SM, saturation 432,
as predicted. Timer quantum: 1.024 us on GA100 too.

| quantity | A100 (sm_80, 1410 MHz) | A10 (sm_86, 1200 MHz) |
|---|---|---|
| solo tile K=64 p50 | 7.8 us | 10.3 us |
| boundary yield p50/p99 at sat | 55.3 / 64.5 us (432 blk) | 64.5 / 86.0 us (288 blk) |
| stage yield p50/p99 at sat | 16.4 / 21.5 us | 17.4 / 22.5 us |
| first observer p50 | 1.02 us flat | 1.02 us flat |
| floor visibility Dmax | 1.02 us flat to 432 | 1.02 us flat to 288 |
| loaded visibility Dmax p50 at sat | 158.7 us | 97.3 us |
| overhead (p1) | 2.5 to 5.3% | ~0 to 3.0% |
| safety naive / issue-naive / poison | 0 / 0 / 10,000 corrupt | same |
| coop U=1 / U=16 e2e p50 | 15.4 / 34.8 us | 22.5 / 41.0 us |
| reserved R=8 U=16 e2e p50 | 175.1 us | 195.3 us |

Predictions vs got: confirmed - occupancy, floor visibility (1 quantum,
flat), boundary 45-75, stage 15-25, safety signature, per-part
constants. FALSIFIED, both in the informative direction: (1) loaded
visibility did not shrink with 3x bandwidth and 40 MB L2; it GREW
(158.7 vs 97.3 us). (2) Co-residency wall stretch did not drop below
1.6x; p99/p50 is ~3.9, like GA102's ~3.4. Conclusion: straggler tails
under co-residency are architectural contention physics, not bandwidth
famine; memory headroom does not buy them back. Overhead prediction
missed at low blocks (5.3% at 108) but sits far inside the falsifier.

The decomposition transfers across the Ampere family: notification is
one timer quantum everywhere; the bound is the worst in-flight tile;
the hazard signature is unchanged. The constants are per-part and
reported per part. "Ampere-class" is now earned by measurement at both
ends of the family, on the same silicon the comparison literature
reports on: stage-granularity in-kernel handoff at 16.4 us p50 on an
A100, from user space, on a stock driver.

**Next:** the paper rewrite. Every registered construct (A1-A4) is
answered; 15 faults on the record; nothing further blocks prose.

## 2026-08-15. Amendment: A100 loaded-visibility value corrected

The A100 entry above attributed 158.7 us to the 432-block loaded cell.
Recomputation from the retained observation matrix (visl_b432.bin)
gives Dmax p50 3776.5 us, p99 4853.8 us; the 158.7 line in the run log
belongs to 324 blocks. Fault 16, class: log-scrape misattribution,
caught by recomputing from raw data before figure generation. The A10
288 bins match their log lines exactly, validating the log-sourced
rows. Consequence: the falsified bandwidth prediction is stronger than
recorded; under full-occupancy DRAM streaming a straggling block can
stall for milliseconds. visibility.csv now carries a source column
(bin vs log) and the paper cites the bin-computed value.

## 2026-08-15. Review 3 response: revision pass applied

Concessions and upgrades applied to the paper; two review points were
answered with new analysis from retained matrices rather than conceded
(scripts/heldout_maxofn.py, results/summary/heldout_maxofn.csv):

- Floor visibility is an every-event statement: 0 of 19,800 events on
  either part exceeded one observed quantum; stated as elapsed < 2
  quanta under the quantization model (review point 2 asked for the
  conservative form; the data supports the absolute form).
- Held-out independence test (fit half, predict half, four quantiles):
  A100 exact to the quantum at p50/p90/p95/p99; A10 p50 within 5%,
  p99 over-predicted 20% (measured tails LIGHTER than independence;
  the model is a conservative envelope there). Per-block product model
  is identical to pooled, so worker heterogeneity (3x spread in
  per-block medians) does not move the maximum. Pairwise correlations
  in [-0.11, 0.04] over 300 pairs, both parts. Registered prediction
  (SM-level correlation breaks the fit) confirmed wrong.

Conceded and rewritten: abstract floor/loaded split (the contradiction
with the visibility section); every non-structural use of "bound";
drain upgraded from "insurance" to semantically-required-by-PTX with
the timing account demoted to non-manifestation explanation; loaded
causal language withdrawn (mechanism unisolated; falsified prediction
kept); ExpertPlex notification-path equivalence narrowed (our floor is
one-CTA dissemination, not their six-stage epoch); barrier mechanism
softened to consistent-with; hang-to-corruption transformation claim
dropped; symmetric envelope constraint on both Pareto arms; negative
reservation cost demoted to anomaly confounded with per-part sag
(cross-card drains implicate box 3; A100 shows no effect); scope cuts
("like most below Hopper" out; "Ampere family" -> both measured
parts); Concordia precise; Ada-MK cited, MegaQwen footnoted.

Open items carried in Threats: exact per-task execution/generation
oracle (designed, needs one validation run), loaded-visibility
mechanism isolation, small-R anomaly sweep. Tensor-core
generalization stays behind all three.

## 2026-08-15. Review 4 pass: clarity and correctness edits

Editorial standard now explicit: a valid measured result stays; a
limitation enters only if it changes what the result means; one
decisive corrective experiment beats five what-ifs.

Applied (22 edits, all verified): the central condition narrowed to
handoff-SENSITIVE cross-CTA state (the old wording was literally false
of our own system; the queue and flag are shared but
handoff-insensitive by construction, and the paper now says so as a
positive example); the loaded construct renamed to observation delay
(it includes the worker's stalls; the registered name overstated
isolation; numbers unchanged); "flat in urgent size" replaced with the
measured sublinear scaling (16x work = 1.8x p50, 2.1x p99) in all
three locations; the conclusion now splits the cross-CTA obligation
(no counterpart here) from the intra-CTA obligation (remains, and our
faults demonstrate it); the novelty sentence names the actual gap
instead of re-litigating existence; abstract quantile wording matched
to what 5.3 shows; figure float pinned and caption fixed.

Not touched, per the same standard: the max-of-N section, all
numerical results, the A100 anchor, the preregistration material.
FlexSched queued for reading, not cited.

Test queue (two experiments, in order): exact scheduler oracle
(execution counts, generation, claimant), then one CUTLASS-class
pipelined kernel. Profiling and reservation-sweep stay parked as
labeled unknowns.

## 2026-08-15. Full claim audit of the paper draft; fault 17

Every quantitative claim and near-claim in the draft was traced to its
source: committed summary CSVs, retained observation matrices, run-log
records validated against matrices, hardware specifications, or the
externally verified literature facts. Results:

Verified clean (spot list): the latency arc and all per-cell surface
numbers (bin cross-checks summary exactly: 64.51/86.02); the held-out
quantiles; the every-event floor claim (0 of 19,800 over one quantum);
the site splits; the Pareto table; the resource-envelope arithmetic
(65,536 - 3x64x256 = 16,384 < 79x256); the rule-of-three and binomial
statistics; wall-stretch and drain cross-checks; all ExpertPlex,
GPreempt, Hummingbird, LithOS quotes and numbers (audit-verified);
the stream-priority documentation claim.

Fault 17 (class: unreconstructable aggregate): the paper claimed
"46,376 forced yields with verified buffer-conflict geometry." No
committed computation produces that number; the correct total,
computed from the mid-run CSVs (claimed events at colliding sites,
drain + naive, both parts), is 28,859, with zero corruption. The
figure was written into the v2 draft without a script behind it.
Corrected everywhere; computation committed as
scripts/audit_claims.py. Rule reinforced: any number in prose must be
reproducible by a committed script over committed data.

Also caught and fixed in the same audit: "twice" for the control-
divergence failure (one fault, stochastic manifestation across runs);
the surface table's epoch column mixed builds (column removed; the
registered construct is reported in prose with the 288 reference
values computed from the retained matrix: first-obs 1.02/2.05); the
8.4x latency growth belonged to the volatile build while the table
cites the reference build (7.0x; corrected and attributed); "3x
bandwidth" was a spec error (A100-SXM4-40GB 1,555 vs A10 600 GB/s =
2.6x; also caught independently by review 5); "thresholds verbatim"
overstated (condensed; claim file is authority); "never edited"
imprecise (registered sections unmodified; amendments append-only);
stale fault counts.

Data-retention note: the review2 per-cell CSVs were never pulled and
box 3 is terminated; the committed summary and the 288-block matrix
survive, and the matrix reproduces the summary exactly, which is what
made the audit possible. Lesson: pull per-cell raws before releasing
a box, not just summaries.

Review-5 pass applied in the same commit: 2.6x, abstract split
(handoff tails smaller on A100; adversarial test worse), registered-
threshold wording for N1/A-N1, runtime-vs-evaluation contract split,
sm_80-anchor terminology, SOTA removed, six audit-history cuts.

## 2026-08-15. Scheduler oracle: built and predicted (box 5, us-west A10)

The one correctness gap named by three review rounds: idempotent tile
writes make duplicate execution, stale generations, and wrong
ownership invisible to the numerical oracle. Built now as exact
structural invariants, template-isolated (ORACLE parameter) so the
measurement instantiations stay SASS-identical; the run script asserts
the 4400-instruction count of the non-oracle hot kernel.

Invariants checked, bounded runs so they are exact, not statistical:
- execution_count[task] == 1 for every task id below the bound
  (duplicates and misses both counted, first offender reported);
- urgent completions per fired event == urgent-queue size (yield) and
  == 1 (midyield, including through the claimer's abandon-and-restart
  path, which must still produce exactly one background writeback);
- every fired event has a recorded claimant; all events fire.

Runs: yield oracle U=1 and U=16 (10k events each, ~31M-task bounds);
midyield events --oracle for stage/drain and the window-forced
issue/naive (10k each).

Predictions, filed before the runs: every invariant holds exactly in
all four runs -- zero duplicates, zero misses, zero urgent-count
violations, zero unclaimed events. This is falsifiable in the
strongest sense: any single violation is a protocol bug and the
paper's "suffices" claim fails as stated. Also predicted: the
non-oracle SASS guard passes (4400 unchanged), and the per-box
under-load clock check on this new part determines the lock (1200 if
it holds like boxes 1-2, 1050 if it sags like box 3).

## 2026-08-15. Oracle run: two guard firings before valid data

Fault 5's guard earned its keep twice in one session:

- Firing 1 (real): adding the oracle counters as kernel parameters
  shifted the non-oracle hot kernel 4400 -> 4512 instructions. The
  template isolated the code but not the signature. Fix: counters
  moved to device globals set by cudaMemcpyToSymbol; measurement
  variants reference them only under a compile-time-false branch.
- Firing 2 (stale baseline; fault 18): the guard still failed at 4512
  because 4400 was the review-2-era count. Building the pre-oracle
  parent commit on the same box shows the hot kernel moved to 4512 at
  the A3 urgent-queue rework, which ran unguarded. The oracle edit is
  codegen-clean by the strongest test: equality with its parent. The
  Pareto measurements were taken with the 4512 kernel and are
  internally consistent (within-binary comparisons throughout).
  Rule: a guard baseline is re-pinned whenever a feature lands, or the
  guard becomes a historian. Guard now asserts 4512.

The paper's "same 4400 SASS instructions under either protocol"
statement describes the volatile-vs-atomic comparison at its own
commit and remains true as scoped; reproducing it requires that
commit, which the artifact's git history provides.

## 2026-08-15. Oracle verdict: all invariants hold exactly

Third launch clean (guard 4512, lock 1050 on this part -- another sag
unit like box 3, caught by the per-box check). The four runs:

| run | tasks checked | dup | miss | events | violations | unclaimed |
|---|---|---|---|---|---|---|
| yield U=1 | 42,278,900 | 0 | 0 | 10,000 | 0 | 0 |
| yield U=16 | 42,283,728 | 0 | 0 | 10,000 | 0 | 0 |
| midyield stage/drain | 50,971,832 | 0 | 0 | 10,000 | 0 | 0 |
| midyield issue/naive | 49,789,964 | 0 | 0 | 10,000 | 0 | 0 |

~185M bounded task executions, exactly-once everywhere; every event
claimed with correct urgent counts, through the abandon-and-restart
path and the window-forced geometry. Safety counters unchanged
(0/10,000 corrupt both midyield runs). Prediction held exactly.

The sanitizer-gate small config FAILED by completeness (0 events
fired), diagnosed as a sizing artifact: the probe measures ~190x
slower under instrumentation, the bounded run shrinks to 239 tasks and
drains before the first gap elapses. Zero invariant violations, zero
sanitizer errors. Gate note: the run script's oracle gate checked only
that output existed, not that it said PASS; sloppy, harmless here,
tightened for next use.

The idempotence blind spot named by reviews 3 through 6 is closed:
"suffices" now rests on exact structural invariants, not numerics.
