# Roadmap

**Definition of done:** a public repo with reproducible code, raw
results, a figure set, and a written report that answers N1 with
numbers, in either direction.

## Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | Persistent kernel with a calibrated synthetic tile (2 to 25 us) | Done |
| 2 | Baseline: urgent-task wait with no preemption | Done |
| 3 | Naive yield. Two numbers: latency and overhead | Done |
| 4 | Surface: block count x poll period | Done |
| 5 | Safety: forced mid-pipeline yield, float64-verified | Done |
| 5a | Amendments A1 to A4: site logging, atomic flag, urgent queue, reserved arm | Done |
| 5b | sm_80 anchor on an A100, predictions registered first | Done |
| 5c | Scheduler oracle: exact execution-count invariants | Done |
| 5d | Tensor-core generalization at 1 block/SM | Done |
| 6 | Optional: port the yield into MegaQwen | Not started |
| 7 | Writeup | Draft complete. Freeze items: compile check, artifact packaging, reading queue |

Results are in [`RESULTS.md`](RESULTS.md). Each measurement has one
script in [`scripts/`](scripts/) and one binary in [`src/`](src/).

## Kill criteria (none fired)

- Phase 1 runs past two weeks: re-scope. Did not fire (one day).
- Latency dominated by something unpredicted: pivot to it. Partially
  fired in the useful sense. The tile term, not propagation, dominates,
  and that became the finding.
- Overhead under 1% and latency flat: publish short. Did not fire.

## Measurement rules

These rules made every number in this repo. Keep them.

1. Lock clocks. Read them back under load, not at idle.
2. Record the power cap.
3. Discard warmup iterations and warmup events.
4. Quote no p99 below 10,000 events.
5. Use the device timer for device intervals. Its tick is 1.024 us.
6. Verify every output against a float64 reference. Never verify a
   kernel against another run of itself.
7. Fill output buffers with NaN before each run. A task that does not
   run must fail loudly.
8. Run compute-sanitizer after any kernel edit, before any measurement.
9. Prove the detector detects (a poison control) before trusting a zero.
10. Flush raw results to disk per configuration.
11. Compare overhead within one binary only. Codegen differences
    between variants can exceed the effect measured.

## Phase 6 sketch, if taken

- Port the drain-discipline yield into MegaQwen (CC 8.6, MIT).
- Reproduce its reported decode rate on the A10 first.
- Then measure preemption latency against a decode stream.
- Budget: 1 to 2 weeks. 10 to 20 GPU hours.
