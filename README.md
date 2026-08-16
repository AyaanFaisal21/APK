# Adaptive Persistent Kernels on Ampere

**Question.** ExpertPlex does bounded tile-level preemption inside a
persistent kernel with Hopper primitives: thread-block clusters, DSMEM,
and TMA multicast. These do not exist below `sm_90`. What preemption
granularity, cost, and safety are possible without them?

**Answer.** Bounded preemption works on Ampere. Claim N1 holds on both
pre-registered axes, and the decomposition repeats on `sm_80`. One
scope limit is itself a measured result: checkpoint cost follows
occupancy. The full results are in [`RESULTS.md`](RESULTS.md).

## AI use

AI tools (Claude) assisted with code and documentation. The developer
directed, reviewed, and verified all work. No measured number comes
from a model: every result comes from the instruments in `src/`, gated
by pre-registered thresholds ([`CLAIM.md`](CLAIM.md)), float64
references, sanitizer runs, and reproduction on a second GPU.

**Status.** All measurement phases complete. The paper draft is written.
Phase 6 (real model) is optional.

| Measurement | Result |
|---|---|
| Tile calibration | K = 32 to 128 gives 6.7 to 17.8 us per tile at 1200 MHz |
| Baseline, no preemption | Urgent task waits 957 us (median) at full residency |
| Yield at tile boundaries | 54 to 69 us median latency across builds, at most 3.4% cost |
| Scaling surface, 24 cells | N1 holds. Propagation stays at 1 to 3 us at all block counts |
| Mid-pipeline yield safety | 0 of 28,859 buffer-conflicting yields corrupt. 17.4 us median latency, in the Hopper band |
| sm_80 anchor (A100) | The decomposition transfers. Two memory-headroom predictions failed informatively |
| Reserved capacity | Cheap but cliffs in waves and binds to resource envelopes. Cooperative holds the better points |
| Scheduler oracle | ~185M task executions, each exactly once. Zero violations |
| Tensor-core pipeline | Safety and oracle transfer whole. Per-stage checkpoint cost is 14% at 1 block/SM; tile-boundary cost is 3% |

**Hardware.** NVIDIA A10 (GA102, `sm_86`, 24 GB, 72 SMs), SM clock
locked and verified under load (1200 MHz; one part holds 1050 MHz). One
A100-SXM4-40GB (`sm_80`, 1410 MHz) as the second-architecture anchor. A
second A10 reproduced the primary results within 2%.

## Navigation

| File | Content |
|---|---|
| [`RESULTS.md`](RESULTS.md) | All results, limits, and conclusions |
| [`CLAIM.md`](CLAIM.md) | Claims N1 and A-N1 with pre-registered thresholds. Never edited after registration |
| [`BACKGROUND.md`](BACKGROUND.md) | Definitions and the Hopper primitives in question |
| [`FINDINGS.md`](FINDINGS.md) | Literature facts with citations |
| [`ROADMAP.md`](ROADMAP.md) | Phases, kill criteria, and measurement rules |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated run log: prediction, result, faults found |
| [`papers/`](papers/) | Reading index (PDFs not committed) |
| [`paper/`](paper/) | The paper source and figures |
| [`src/`](src/) | Five binaries, one per measurement. Each file header says how to run it |
| [`scripts/`](scripts/) | One script per measurement, plus `lock_clocks.sh` and the table generators |
| [`results/summary/`](results/summary/) | Committed figures and summary CSVs. Every quoted number traces to one of these files |

## Standard

Report what fails. Two prior repos ([TransformerOp](../TransformerOp),
[Cornfield](../cornfield)) set this standard. This repo keeps it: 18
faults found, all recorded in the NOTEBOOK, none deleted. Registered
predictions that failed are reported as findings, not removed.
