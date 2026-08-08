# Adaptive Persistent Kernels on Ampere

**Question.** ExpertPlex does bounded tile-level preemption inside a
persistent kernel with Hopper primitives: thread-block clusters, DSMEM,
and TMA multicast. These do not exist below `sm_90`. What preemption
granularity, cost, and safety are possible without them?

**Answer.** Bounded preemption works on Ampere. Claim N1 holds on both
pre-registered axes. The full results are in [`RESULTS.md`](RESULTS.md).

## AI use

AI tools (Claude) assisted with code and documentation. The developer
directed, reviewed, and verified all work. No measured number comes
from a model: every result comes from the instruments in `src/`, gated
by pre-registered thresholds ([`CLAIM.md`](CLAIM.md)), float64
references, sanitizer runs, and reproduction on a second GPU.

**Status.** Phases 1 to 5 complete. The writeup remains. Phase 6 (real
model) is optional.

| Phase | Measurement | Result |
|---|---|---|
| 1 | Tile calibration | K = 32 to 128 gives 6.7 to 17.8 us per tile at 1200 MHz |
| 2 | Baseline, no preemption | Urgent task waits 957 us (median) at full residency |
| 3 | Yield at tile boundaries | 59 us median latency, 0.8% cost |
| 4 | Scaling surface, 24 cells | N1 holds. Propagation stays at 1 to 3 us at all block counts |
| 5 | Mid-pipeline yield safety | 0 of 10,000 yields corrupt with the drain discipline. 18.4 us median latency, in the Hopper band |

**Hardware.** NVIDIA A10 (GA102, `sm_86`, 24 GB, 72 SMs). SM clock locked
at 1200 MHz, confirmed under load. A second A10 reproduced the primary
results within 2%.

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
| [`src/`](src/) | Four binaries, one per measurement. Each file header says how to run it |
| [`scripts/`](scripts/) | One script per measurement, plus `lock_clocks.sh` |
| [`results/summary/`](results/summary/) | Committed figure and surface CSV |

## Standard

Report what fails. Two prior repos ([TransformerOp](../TransformerOp),
[Cornfield](../cornfield)) set this standard. This repo keeps it: eight
faults found, all recorded in the NOTEBOOK, none deleted.
