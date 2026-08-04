# Literature findings

Facts below are split by how they were established. **Verified** means read from the paper PDF in [`papers/`](papers/). **Unverified** means it came from a search summary or a secondary review and has not been checked against the source — do not cite these without reading them first.

---

## Verified from primary sources

### ExpertPlex — the closest prior work
arXiv [2607.18002](https://arxiv.org/abs/2607.18002) · Wu, Jin, Zhang, Wei, Zhong, Zhu, Yang, Jin, Liu · Peking University + independent · July 2026 · **no code released**

**Motive.** MoE serving has a granularity problem. Instance-level prefill/decode disaggregation puts each phase on a separate full-model replica, but MoE weights are large enough that an instance spans tens to hundreds of GPUs, so *"each scaling step changes capacity by hundreds of GPUs and cannot match moderate traffic shifts efficiently."* Colocation via Green Context avoids the duplication but *"partition[s] each GPU by phase and fix[es] phase resources during a kernel"* — it cannot follow layerwise variation in routed expert load, producing either head-of-line blocking or idle reserved SMs.

**Four mechanisms.**
1. **Expert sharing + attention disaggregation** — share the massive MoE experts across both phases, eliminating *"over 95% of duplicate model weights"*; disaggregate only the lightweight attention modules.
2. **Adaptive Persistent Kernel (APK)** — one long-lived kernel per MoE GPU scheduling at tile granularity. Tile boundaries occur every **2.2–25.3 μs**, independent of total operation length. Preempts prefill for decode and reallocates idle CTA clusters with no CPU intervention and no kernel relaunch, staying CUDA-Graph compatible.
3. **Attention-initiated one-sided MoE communication** — attention servers push activations into MoE-side buffers over mostly disjoint network paths without APK coordination, removing MoE-side polling and avoiding cross-phase deadlock and network interference.
4. **Cross-stack placement optimizer** — jointly selects APK policies, placement, resource ratios, parallelism, and overlap to maximize goodput under both phases' SLOs.

**The preemption mechanism.** *"Independent tile-boundary checks are unsafe because high-performance operations pipeline both warps and CTAs across tiles."* Instead APK propagates one cooperative decision through the memory hierarchy: CTA 0 checks the preemption flag at a tile boundary and writes one device-scope word per cluster. Scheduling happens at CTA-cluster granularity *"because CTAs connected by TMA multicast must execute the same tile."* Different clusters run different phases (spatial multiplexing); each cluster switches at tile boundaries (temporal multiplexing). **Preemption is bounded by one tile execution time plus one local cluster check epoch.**

**Results.** Up to **2.01×** goodput over instance-level PD disaggregation, **1.66×** over Green-Context colocation. Table 1 claims sole possession of all five of {CUDA-Graph compatibility, temporal multiplexing, spatial multiplexing, bounded fast preemption, bounded fast reallocation}.

**Testbed.** *"Single-node experiments run on one NVIDIA H800 node with eight GPUs connected by NVLink. Multi-node experiments use up to three machines,"* 8×200 Gbps InfiniBand per node. Models: MiniMax-M2.7, GLM-5.1-FP8. Baselines are all SGLang-based (Colocated, ChunkedPrefill, PDD, PDMux) to isolate system-level differences. Note: *"GLM-5.1-FP8 runs out of memory under this PDD layout on 24 GPUs."*

**Why it cannot be reproduced here.** Four independent blockers: no code released; 8-GPU minimum testbed; frontier MoE models that do not fit in 24 GB; and — the interesting one — the mechanism is built on `sm_90` primitives. TMA appears 23 times in the text, cluster 32, DSMEM 4. The A10 has none of them.

**What it leaves open.** Its two tenants are the prefill and decode phases of *one* MoE model. Multi-model co-location is acknowledged as distinct and unclaimed.

### MPK / Mirage Persistent Kernel
arXiv [2512.22219](https://arxiv.org/abs/2512.22219) · OSDI '26 · CMU Catalyst (Jia) + collaborators · **Apache 2.0**

Compiles multi-GPU model inference into a single megakernel. Introduces **ttGraph**, an SM-level task/event graph: nodes are per-SM tasks or synchronization events; a task fires when its dependent events activate. Compiler passes: operator decomposition, fine-grained dependency analysis, event fusion, normalization, linearization. In-kernel decentralized scheduling across SMs.

Numbers: **1.0–1.7×** single-batch over vLLM and SGLang across models and hardware, largest on smaller models and newer GPUs; **1.1–1.4×** across 8×H100 under tensor parallelism. Qwen3-8B on A100 goes 14.5 ms → 12.5 ms against a **~10 ms floor set by memory bandwidth**. Cross-task software pipelining gives 1.2–1.3× on linear layers, *surpassing cuBLAS*; compute/communication overlap contributes 1.1×. Task bodies come from the Mirage superoptimizer's transpiler.

The bandwidth floor is the important number: megakernels are a lever that largely *finishes* once applied. Remaining headroom after that is bytes-moved (quantization, sparsity) or algorithmic (speculative decoding).

### LithOS — the opposing thesis
arXiv [2504.15465](https://arxiv.org/abs/2504.15465) · SOSP '25 · CMU (Skarlatos) + Meta · Rust

A GPU OS built on: a TPC Scheduler doing spatial scheduling at individual-TPC granularity with **TPC stealing**; **transparent kernel atomization** to reduce head-of-line blocking and allow dynamic reallocation mid-execution; lightweight hardware right-sizing per atom; transparent power management. Reports *"tail latencies by 13× compared to MPS; compared to the best-performing SotA, it reduces tail latencies by 3× while improving aggregate throughput by 1.6×."*

**This is the direct antithesis of a megakernel.** LithOS manufactures preemption points by splitting kernels into atoms; megakernels remove every boundary that could serve as one. Both lines run out of CMU — Skarlatos is listed on the Catalyst faculty page alongside Jia. An unresolved tension inside one group is a good place to stand.

### Hummingbird — microsecond preemption
arXiv [2601.04071](https://arxiv.org/abs/2601.04071) · Jan 2026

*"Enabling microsecond-scale preemption on closed-source GPUs while effectively harvesting idle GPU time slices."* Improves high-priority SLO attainment **9.7×** and **3.5×** over state-of-the-art spatial and temporal sharing respectively; co-located high-priority SLO attainment drops **<1%** versus exclusive execution; low-priority throughput beats temporal-sharing SotA by **2.4×**.

Relevant because it raises the bar on the "megakernels cannot be preempted" framing. Probably *strengthens* the argument — a megakernel defeats even microsecond-scale external preemption, since there is no boundary to preempt at — but it must be engaged with, not ignored.

### Hardware facts
- **A10** — GA102, `sm_86`, CC **8.6**, 24 GB GDDR6, ~600 GB/s, 72 SMs. Same compute capability as an RTX 3090; the 3090 has ~936 GB/s, so expect lower absolute throughput on identical work.
- **MegaQwen** ([github](https://github.com/Infatoshi/MegaQwen), MIT, 26 commits) requires **CC 8.6+**, CUDA 11.8+. Reports 530 tok/s decode on a 3090 for Qwen3-0.6B (3.9× over HuggingFace) — **but degrades to 158 tok/s at longer contexts**, where TensorRT-LLM holds a consistent 355 tok/s. That crossover is itself an unmeasured phenomenon.
- Thread-block clusters, DSMEM, and TMA are all `sm_90`. `cp.async` is available from `sm_80`.

---

## Unverified — check before citing

- **DuetServe** (arXiv [2511.04791](https://arxiv.org/abs/2511.04791), PDF in `papers/`) reportedly measures HBM bandwidth rising super-linearly with SM count — *"20% of SMs already achieve approximately 60% of peak bandwidth."* If accurate this is strong support for the reclaimable-slack premise. **Read §-in-context before relying on it.**
- **Cooperative Kernels** (FSE 2017) — `offer_kill` / `request_fork` as the conceptual ancestor. Cited from secondary sources; PDF not yet obtained.
- Artifact availability for ETC, Ada-MK, Fleet, Kog — reported as unreleased. Verify individually before assuming.
- Aegaeon's reported 82% GPU savings, MuxServe's throughput figures, "Memory-Bound but Not Bandwidth-Limited" — all from a secondary review, none checked.

---

## The landscape, compressed

Roughly seven megakernel systems in eighteen months from six institutions — Hazy (Stanford), MPK (CMU), ETC, Ada-MK, Fleet (AMD), Perseus, Kog, mKernel — evaluated on H100, B200, MI300X, and L20 against inconsistent baselines. Separately, a GPU scheduling/preemption line — LithOS, REEF, Orion, Hummingbird, Cooperative Kernels — assumes conventional kernel-by-kernel execution.

**The two literatures are nearly disjoint.** ExpertPlex is the bridge, and only for single-model phase sharing on Hopper.
