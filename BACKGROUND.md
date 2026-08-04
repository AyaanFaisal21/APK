# Background

## Persistent kernels

A conventional launch sizes the grid to the **problem**. Blocks spawn, execute, exit; the hardware scheduler feeds new blocks onto SMs as old ones retire; the kernel ends when the last block finishes. The next operator is a new launch.

A persistent kernel sizes the grid to the **hardware** — enough blocks to fill every SM at target occupancy (72 on an A10, 132 on H100) — and those blocks never exit. They loop:

```cuda
__global__ void persistent(TaskQueue* q) {
    while (true) {
        Task t = q->next();          // atomic fetch-add on a global counter
        if (t.is_terminal()) break;
        execute(t);                  // registers / SMEM persist across iterations
    }
}
```

It is a thread pool: workers start once and pull from a queue instead of spawning per task.

**What it buys**

- **One launch.** Per-operator dispatch overhead disappears.
- **No implicit global barrier.** Under kernel-per-op, kernel *N+1* cannot start until every block of kernel *N* retires. Here a block finishing task A takes task B immediately — even from a later operator — if dependencies allow. This is what CUDA graphs do **not** give you: graphs remove launch overhead while leaving every kernel boundary intact.
- **Resident state.** Registers and shared memory survive between tasks, so intermediates can stay on chip instead of round-tripping HBM.

**What it costs** — you have replaced the hardware scheduler with yourself.

- **Deadlock becomes possible.** A resident block waiting on a task no resident block will run hangs the GPU.
- **Occupancy goes global.** The most register-hungry task caps occupancy for every task; they share one launch configuration.
- **Dynamic shapes get hard.** The grid is fixed at launch, so data-dependent work must be expressed inside the queue.

## Megakernels

A **megakernel** is a persistent kernel whose task set is an entire model forward pass. Persistent kernel is the mechanism; megakernel is one application. Persistent kernels long predate LLM inference — keep the terms separate.

## Adaptive persistent kernels

A plain persistent kernel fixes its work assignment when the queue is built. **Adaptive** means the assignment is renegotiated *while the kernel runs*, from on-GPU state, with no exit and no relaunch.

ExpertPlex's APK carries two concurrent tenants — the prefill and decode phases of one MoE model — and mid-flight it can:

- **Preempt**: a cluster running prefill switches to decode once its current tile commits
- **Reallocate**: idle clusters move to whichever phase has ready work

Both decided on the GPU, no CPU round-trip, CUDA-Graph compatible throughout.

**Why tile granularity.** A tile is the smallest independently completable unit of an operation. Finer requires preserving pipeline state across the switch — expensive. Coarser (kernel- or layer-level) leaves long blocking intervals. Measured tile boundaries land at **2.2–25.3 μs** and are *independent of total operation length*, so a decode request never waits behind a prefill merely because the prefill is long.

**The preemption bound** is one tile execution plus one check epoch.

## The primitive dependency — the crux of this project

ExpertPlex states that independent per-block flag checks are **unsafe**, because high-performance operations pipeline across both warps and CTAs; a block bailing mid-pipeline corrupts state its neighbours depend on. Its answer is to propagate *one cooperative decision* through the memory hierarchy: CTA 0 checks the flag at a tile boundary and writes one device-scope word per cluster.

That design rests on three `sm_90` features:

| Hopper primitive | What it provides | Ampere (`sm_86`) substitute |
|---|---|---|
| Thread-block clusters | CTAs guaranteed co-resident on one GPC, with hardware `cluster.sync()` | none |
| DSMEM | direct addressing of a neighbour CTA's shared memory, on-chip | none — must go through global memory / L2 |
| TMA multicast | async bulk copy broadcast to CTAs in a cluster | `cp.async` (async, no multicast) |

So the propagation path that costs roughly shared-memory latency on Hopper costs an L2 round-trip on Ampere, and scales with resident block count rather than cluster size. **That is the mechanism N1 is about.**

Two consequences worth holding onto:

1. The check epoch — one of the two terms in the preemption bound — should be materially worse on Ampere. Whether it is *dominant* is the empirical question.
2. Ampere pipelines differently (`cp.async`, no multicast, no cross-CTA shared memory). ExpertPlex's *unsafety* claim is a statement about TMA-multicast pipelines. Whether independent checks are unsafe here is a separate question, and the answer could invert: simpler hardware may permit the simpler mechanism.

## Conceptual ancestor

**Cooperative Kernels** (Sorensen, Evrard, Donaldson; FSE 2017) introduced `offer_kill` / `request_fork` — a persistent kernel voluntarily surrendering workgroups without saving state, on OpenCL 2.0 hardware. The primitive is not new and is not architecture-specific in principle; Hopper made it cheap. That lineage is the argument that N1 is worth testing rather than assumed false.
