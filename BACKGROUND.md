# Background

## Persistent kernels

A normal launch sizes the grid to the problem. Blocks start, run, and
exit. A persistent kernel sizes the grid to the hardware. Blocks never
exit. They loop on a task queue:

```cuda
__global__ void persistent(TaskQueue* q) {
    while (true) {
        Task t = q->next();          // atomic fetch-add on a counter
        if (t.is_terminal()) break;
        execute(t);
    }
}
```

It is a thread pool on the GPU.

**Gains:** one launch. No implicit barrier between operators. Registers
and shared memory survive between tasks.

**Costs:** the developer replaces the hardware scheduler. Deadlock
becomes possible. The most register-hungry task caps occupancy for all
tasks. Dynamic shapes must live inside the queue.

## Megakernels

A megakernel is a persistent kernel whose task set is a full model
forward pass. Persistent kernel is the mechanism. Megakernel is one
application of it.

## Adaptive persistent kernels

A plain persistent kernel fixes its work assignment when the queue is
built. Adaptive means the assignment changes while the kernel runs, from
on-GPU state, with no relaunch. ExpertPlex preempts prefill for decode
and reallocates idle clusters this way.

**Tile granularity.** A tile is the smallest independently completable
unit. ExpertPlex tile boundaries land at 2.2 to 25.3 us, independent of
total operation length. The preemption bound is one tile time plus one
check epoch.

## The primitive dependency

ExpertPlex states that independent per-block flag checks are unsafe,
because operations pipeline across warps and CTAs. Its mechanism
propagates one cooperative decision per cluster. That design rests on
three `sm_90` features the A10 does not have:

| Hopper primitive | Provides | Ampere substitute |
|---|---|---|
| Thread-block clusters | CTAs co-resident with hardware sync | none |
| DSMEM | direct access to a neighbor CTA's shared memory | none. Global memory / L2 only |
| TMA multicast | bulk copy broadcast to a cluster | `cp.async`, no multicast |

Two consequences drive this study:

1. Flag propagation costs an L2 round trip on Ampere, not a
   shared-memory hop. Whether that term dominates is the empirical
   question (Phases 3 and 4).
2. The unsafety claim is a statement about Hopper pipelines. Ampere
   pipelines differently. Whether independent checks are unsafe here is
   a separate question (Phase 5).

## Ancestry

Cooperative Kernels (Sorensen, Evrard, Donaldson; FSE 2017) had a
persistent kernel yield workgroups on OpenCL 2.0 hardware. The primitive
is old. Hopper made it cheap. That lineage is why this study tests the
question instead of assuming the answer.
