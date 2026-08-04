# Papers

PDFs are gitignored; this index is the committed artifact.

## Read first

| File | Paper | Read |
|---|---|---|
| `expertplex.pdf` | [ExpertPlex](https://arxiv.org/abs/2607.18002) — PKU, Jul 2026 | **§4 entirely.** §4.1 design space, §4.3 bounded tile-level preemption. This is the paper the project responds to. |
| `lithos.pdf` | [LithOS](https://arxiv.org/abs/2504.15465) — SOSP '25, CMU (Skarlatos) | Kernel atomization and TPC stealing. The opposing thesis: it *manufactures* the preemption points megakernels remove. |

## Context

| File | Paper | Read |
|---|---|---|
| `mpk.pdf` | [MPK](https://arxiv.org/abs/2512.22219) — OSDI '26, CMU Catalyst | ttGraph, event-driven in-kernel scheduling. Apache 2.0, the runnable megakernel compiler. Note the ~10 ms bandwidth floor. |
| `hummingbird.pdf` | [Hummingbird](https://arxiv.org/abs/2601.04071) — Jan 2026 | Microsecond preemption on closed-source GPUs. Engage with it: it raises the bar on "megakernels can't be preempted." |
| `duetserve.pdf` | [DuetServe](https://arxiv.org/abs/2511.04791) | SM partitioning via libsmctrl for prefill/decode. Source of the bandwidth-vs-SM-count curve — **verify that claim in context before citing.** |
| `eventtensor.pdf` | [Event Tensor / ETC](https://arxiv.org/abs/2604.13327) — MLSys '26 | Dynamic megakernels, data-dependent triggers. Its finding that static scheduling wins for dense batch-1 is relevant. |
| `adamk.pdf` | [Ada-MK](https://arxiv.org/abs/2605.11581) | Adaptive megakernel via offline DAG search, Ada/L20, 1–5 ms latency budgets. |
| `fleet.pdf` | [Fleet](https://arxiv.org/abs/2604.15379) — AMD | Megakernels on multi-die GPUs. Its related-work section is the best one-paragraph survey of the field. |
| `mirage.pdf` | [Mirage](https://arxiv.org/abs/2405.05751) — OSDI '25 | The superoptimizer MPK borrows its transpiler from. Background only for this project. |

## Not yet obtained

- **Cooperative Kernels** (Sorensen, Evrard, Donaldson; FSE 2017) — `offer_kill` / `request_fork`. The conceptual ancestor of in-kernel yield, a decade before Hopper. Worth getting; it is the citation that argues the primitive is not architecture-specific in principle.
- **MegaQwen** is code, not a paper: <https://github.com/Infatoshi/MegaQwen> (MIT, CC 8.6+).

## The one-paragraph version

Roughly seven megakernel systems in eighteen months from six institutions, all evaluated on dedicated GPUs against inconsistent baselines. A separate literature on GPU scheduling and preemption — LithOS, REEF, Orion, Hummingbird — assumes conventional kernel-by-kernel execution. The two are nearly disjoint. ExpertPlex bridges them, for single-model prefill/decode phase sharing, using Hopper cluster primitives. This project asks what survives without those primitives.
