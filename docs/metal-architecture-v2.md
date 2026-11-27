# MetalVI native Metal backend v2: execution architecture

> **Status: design document, not implemented yet.** Written 2026-11-27. Section 1 describes the code on the current development branch, part of which is not published yet. That code still uses the old execution model: synchronous submission on a single queue. v2 is built in stages (see the roadmap in the README). All performance numbers here are predictions, not measurements.

**English** · [简体中文](metal-architecture-v2.zh-CN.md)

## 1. Why a v2: the ceilings in the current backend

| # | Problem | Evidence | Effect |
|---|---|---|---|
| P1 | **CPU and GPU in lockstep** | [`metal.rs` `run()`](../crates/victory-gpu/src/backend/metal.rs) commits under a global lock and waits for completion ("the title's fences may be signalled after this returns"). The descriptor ring is reused from the start because the previous submission was waited for. | Frame time ≈ CPU + GPU + a wake-up per submission, instead of max(CPU, GPU). |
| P2 | **Cross-queue sync emulated on the CPU** | `queue.rs` cuts lists at waits for memory values and holds them on the CPU. Bottom-of-pipe writes are done by the CPU after the GPU finishes. A present forces everything held. GPU timestamps are CPU time. There is a single Metal queue. | The console's graphics and async-compute queues collapse into "CPU scheduler + one GPU queue", with no overlap. |
| P3 | **Over-serialised GPU** | Compute encoders are created serial, with a barrier between every pair of dispatches. | Thousands of one-threadgroup dispatches run one after another while the GPU sits mostly idle. |
| P4 | **No TBDR awareness** | Clears are passes of their own and load/store is always Load/Store. There is no memoryless storage and no pass planning. | On a tile-based GPU every pass break stores and reloads whole render targets. This is usually the largest bandwidth cost of a ported immediate-mode renderer. |
| P5 | **Single-threaded, lock-heavy encoding under Rosetta** | All Metal encoding happens at submit time, on one thread, inside a global lock, in x86-64 code. | The title records its lists on many job threads, but our encoding is serial and pays for translation. |
| P6 | **Textures are still copies** | Textures are detiled private copies, re-checked (hashed) before execution. | Buffers are zero-copy; textures are not. |
| P7 | **Safety instrumentation everywhere** | MSC bounds checks and an injected loop budget in every pipeline. | A GPU cost on every shader, and the loop budget changes semantics. |
| P8 | **No binary archives** | The bindings exist but are unused. | Cold pipeline compiles stalled 3D entry for 126–154 s. |
| P9 | **No performance budget** | Acceptance criteria had no numbers and no Series S default. | Route decisions had nothing to stand on. |

Still valid and kept: zero-copy buffers with persistent section-view aliasing, MSC conversion, descriptor-table encoding, the Wine → Metal bridge, the detile kernels, and the safety gates and evidence rules.

## 2. Goal and frame budget (30 fps = 33.3 ms)

| Part | Budget | Notes |
|---|---|---|
| Title CPU critical path | ≤ 33 ms | Outside the backend's control: Rosetta + Wine. This is the final ceiling. |
| Backend encoding | ≤ 6 ms CPU per frame | On encoder threads, **off** the title's critical path. |
| Submit latency | Hidden | Up to 2 frames in flight. |
| GPU | ≤ 30 ms | Series S profile, timed per pass. |
| Long frames | None over 100 ms with a warm cache | Compile and warm-up stalls are reported separately. |

Performance runs default to the **Series S** profile. Comparisons against older Series X captures are kept separate.

## 3. Target architecture

```
 title threads             backend                                    GPU
 list method calls ─► list IR (MetalOp streams, per list part)
 ExecuteCommandLists ─► queue timelines (A) ─► encoder pool (B)
                         fences/labels → MTLSharedEvent           ┌ graphics MTLCommandQueue
                         in-flight budget (H)                     ├ compute  MTLCommandQueue
                    pass planner (C) + barrier translation (D) ───┤ (ordered by events)
                    resources (E) + bindings (F) + pipelines (G)  │
 completion thread per queue ◄──── completion / GPU time ─────────┘
   bottom-of-pipe writes, fence values, resource and ring retirement, duration checks
```

**A. Queue timelines and async submission**
- One `MTLCommandQueue` per title queue.
- Each title fence maps to an `MTLSharedEvent` (event value = fence value).
- In-list waits on memory labels become GPU-side event waits when the producer is known, its values are monotonic, and the compare is `>=`. Everything else falls back to the existing CPU hold, and those fallbacks are counted.
- **The GPU only waits on values whose signal is already enqueued**, so a queue can never wait forever.
- A completion thread per queue does the bottom-of-pipe writes, advances fences, retires transient memory and records GPU time. The submitting thread never waits for the GPU.
- A synchronous mode is kept for debugging only.

**B. Encoding**
- The list IR is recorded at call time.
- Encoding still happens after `ExecuteCommandLists`, because descriptor contents are execution-time state. It runs on a thread pool: one command buffer per list part, ordered with `enqueue`.
- Caches are lock-light. No global lock is held around encoding.
- Indirect command buffers come later, for long uniform draw runs.

**C. TBDR pass planner**
- Fold clears into `loadAction = Clear`.
- Use `DontCare` loads and stores where provably safe (for example `DiscardResource`, or depth that is never read again).
- Do MSAA resolves through the store action.
- Use memoryless storage for transient attachments.
- Never break a pass for barriers that don't touch its attachments.
- Later: reorder independent dispatches out of passes, and use tile or imageblock features.

**D. Synchronisation**
- Compute encoders are concurrent.
- Only the title's own barriers become memory barriers.
- Within a queue, untracked memory is ordered with `MTLFence` (across encoders) or `MTLEvent` (across command buffers).
- Shared events are used only across queues and for CPU-visible fences.

**E. Memory and texture ownership**
- **Buffers:** zero-copy, as today. No re-wrapping of remapped pages inside a live residency set.
- **Linear textures:** buffer-backed textures over the zero-copy memory.
- **CPU-written tiled textures:** private copies refreshed from dirty information: first from known writers (file reads, DirectStorage, xmem), then from page-level write watch, never from full hashing. Only dirty subresources are detiled, on the GPU.
- **GPU-written textures:** stay on the GPU and are written back only when the CPU or an alias needs them.
- **Aliasing:** an index from address range to the live resource invalidates overlapping resources.

**F. Bindings**
- Descriptor-heap entries (`IRDescriptorTableEntry`) are written when views are created, not synced in batches at submit.
- A reverse index from spans to entries rewrites the affected entries after remaps.
- Root parameters live in per-submission ring slices.
- No D3D12 `prepare`, mirroring or shadow buffers.

**G. Shaders and pipelines**
- Binary archives are harvested offline and looked up first; misses are counted.
- Two instrumentation tiers:
  - **U (unverified):** bounds checks and loop budget on.
  - **V (verified):** both off.
- NGG shaders that are really vertex shaders take a vertex-shader path. This avoids the slower mesh path on M1/M2 and the 32 KiB mesh-output limit.
- GPU time is measured per pipeline. Emitting AIR/MSL directly is the escalation path if conversion overhead turns out to dominate.

**H. Safety v2.** This replaces "wait for every command buffer, fuse afterwards".
1. Work bounds at encode time. Indirect arguments are clamped by a small kernel before use.
2. An in-flight budget per queue: count and estimated GPU time.
3. Asynchronous duration and error checks on the completion thread stop all new submissions.
4. Process-wide stop, plus the single-instance run lock.

A command buffer that is already running cannot be interrupted. Observed maxima are not proofs.

**I. Observability**
- Per-pass GPU time from `MTLCounterSampleBuffer` (stage-boundary sampling).
- Per frame: frame-time percentiles, encode time, lock waits, queue busy time, pass and load/store counts, detile/write-back/hash bytes, archive misses, and the U/V tier mix.

**J. Portability.** The backend's interface is a command stream plus resource tables in shared memory, so it can later move to a native arm64 process or come along with arm64 Wine + FEX, away from Rosetta.

## 4. Risks

| Risk | Mitigation |
|---|---|
| The three long-running compute shaders spin for another reason than "waiting on another queue's data" | Static analysis first. Multi-queue support is needed regardless. |
| Shaders that poll memory written by a truly concurrent queue: Apple GPUs give no forward-progress guarantee | Never rely on that. Convert recognised patterns into event waits between dispatches; keep the loop budget and quarantine otherwise. |
| Labels with non-monotonic values, masks or `==` compares | CPU hold fallback, counted. |
| Multi-queue access to zero-copy memory and residency sets is untested | Standalone Metal tests first, never in the full game. |
| GPU timestamp semantics change | Audit how the title uses timestamps before changing them. |
| Title CPU path is the final ceiling | The route decision uses the measured budget split. |
