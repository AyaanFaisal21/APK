# Literature findings

**Verified** means read from the paper PDF in [`papers/`](papers/).
**Unverified** means from a summary. Do not cite unverified facts.

## Verified

### ExpertPlex (the paper this project responds to)
arXiv 2607.18002. PKU. July 2026. No code released.

- Problem: MoE serving cannot scale prefill and decode independently at
  fine grain. Instance-level disaggregation moves hundreds of GPUs per
  step. Static GPU partitions cannot follow layerwise expert load.
- Mechanism: one adaptive persistent kernel per MoE GPU. Tile-level
  scheduling. Tile boundaries every 2.2 to 25.3 us. Preemption bound:
  one tile time plus one cluster check epoch.
- Safety design: "independent tile-boundary checks are unsafe because
  high-performance operations pipeline both warps and CTAs across
  tiles." One CTA per cluster checks the flag and writes one word.
  Cluster granularity exists because TMA multicast couples the CTAs.
- Results: up to 2.01x goodput over disaggregation, 1.66x over
  Green-Context colocation. Testbed: 8x H800 per node.
- Cannot be reproduced here: no code, 8-GPU testbed, frontier models,
  and `sm_90` primitives. This project measures the mechanism on
  hardware without those primitives.

### MPK / Mirage Persistent Kernel
arXiv 2512.22219. OSDI '26. CMU. Apache 2.0.

- Compiles model inference into one megakernel. Task graph at SM level.
- 1.0 to 1.7x single-batch over vLLM and SGLang. Qwen3-8B on A100 hits
  a ~10 ms floor set by memory bandwidth.
- The floor is the point: megakernels are a lever that finishes.

### LithOS (the opposing thesis)
arXiv 2504.15465. SOSP '25. CMU.

- A GPU OS. Splits kernels into atoms to create preemption points.
  Megakernels remove those points. The two lines conflict, from the
  same institution.
- Reports 13x lower tail latency than MPS.

### Hummingbird
arXiv 2601.04071. Jan 2026.

- Microsecond-scale preemption on closed-source GPUs, kernel-per-op
  model. Raises the bar the megakernel line must answer.

### Hardware facts

- A10: GA102, `sm_86`, 72 SMs, 24 GB, ~600 GB/s. Same compute
  capability as an RTX 3090.
- Thread-block clusters, DSMEM, and TMA are `sm_90`. `cp.async` exists
  from `sm_80`.
- MegaQwen (github.com/Infatoshi/MegaQwen, MIT) runs on CC 8.6. Decode
  530 tok/s on a 3090, degrading to 158 tok/s at long context. Candidate
  for Phase 6.

## Unverified

- DuetServe (arXiv 2511.04791): 20% of SMs reach ~60% of peak bandwidth.
  Read in context before citing.
- Cooperative Kernels (FSE 2017): cited from secondary sources. PDF not
  yet obtained.
- Aegaeon, MuxServe figures: from a secondary review. Not checked.

## The landscape in one paragraph

Roughly seven megakernel systems in eighteen months, all evaluated on
dedicated GPUs. A separate literature on GPU preemption assumes
kernel-per-op execution. The two are nearly disjoint. ExpertPlex is the
bridge, on Hopper only. This project asks what survives without Hopper.
