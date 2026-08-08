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
