<h1 align="center">MetalVI</h1>

<p align="center">
<b>An open-source research runtime for running the Xbox Series build of Grand Theft Auto VI on Apple Silicon Macs, with a native Metal backend and zero-copy unified memory.</b><br>
<center> <b>Model Used: Antigravity - Gemini 4 Neon, Gemini 4 Hydrogen (Initial Setup)<p><center>Codex - GPT-6.2 Sol (Development)</p><p><center>Doubao Work - Deepseek V4.1 Flash (Research & Basic Work since I have no enough tokens)</p> <p><center>Claude Code - Claude Opus 5.5 & Claude Sonnet 5.5 (Planning + most adaptions)</p> </b><br>
M1 · M2 · M3 · M4 · M5 &nbsp;|&nbsp; macOS 15+ &nbsp;|&nbsp; Rust · Wine · Metal · Metal Shader Converter
</p>

<p align="center">
<img alt="Status: in active development" src="https://img.shields.io/badge/status-in%20active%20development-orange">
<img alt="Not playable yet" src="https://img.shields.io/badge/playable-not%20yet-red">
<img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-M1%2B-555555">
<img alt="Metal backend" src="https://img.shields.io/badge/GPU-native%20Metal-blue">
<img alt="License GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-lightgrey">
</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

> **Bring your own game.** MetalVI contains **no game files, assets, keys or proprietary DLLs**, and never will. It runs only from a copy of the game that **you legally own and dumped yourself**. It doesn't download, decrypt or unpack anything, and it contains nothing for getting around DRM. See [Legal](#legal).

---

## The short version

Most attempts to run console games on a Mac stack translation layers: the game's D3D12 calls are translated to PC D3D12, which Apple's D3DMetal translates again into Metal. On a Mac that stack has to pretend the GPU is a discrete card, so the console's **unified memory** gets emulated by copying and hashing gigabytes every frame. We measured about 1.4 GB uploaded and 7.5 GB compared per frame, which caps that design at **about 1–2 fps in 3D, however much you optimise it**.

Apple Silicon *has* unified memory, just like the console. **MetalVI drops the second translation layer.** Its renderer talks to **Metal directly** and hands the game's own memory to the GPU **without copying** (`newBufferWithBytesNoCopy` + `MTLResidencySet`). Shaders go through **Apple's Metal Shader Converter**, linked in-process. The ceiling is set by the GPU, not by memcpy.

**Where it stands (2026-11-27):** the game boots, runs its menus, loads into the 3D world and starts the opening mission. Compute work runs almost entirely on native Metal. Drawing and presenting run on the development branch, but most of the 3D picture is still black. Counting stalls, the real 3D average is **1.2–2.5 fps**, about the same as the D3DMetal-based ports. The reason is that today's Metal backend is a **hybrid**: it sits on the old D3D12 preparation path, runs CPU and GPU in lockstep on a single GPU queue, and so has not yet collected the zero-copy win. We have therefore redesigned the **execution architecture (v2, below)**. It is designed but not implemented yet. **It is not playable.** We'd rather say that plainly than overpromise.

> ⚠️ **Safety warning:** during development, experimental Metal paths produced single GPU command buffers that ran for 6–10 seconds. Apple GPUs do not preempt such work for WindowServer, so the whole machine froze, and once the kernel panicked. Until the v2 GPU safety gate is done, **do not run the full game on a Mac that holds work you care about.**

## Architecture

### Two ways to get a console game onto a Mac

```mermaid
flowchart LR
    subgraph OLD["Typical stack: two translation layers"]
        direction TB
        A1["Game x86-64 code<br/>(Rosetta 2)"] --> A2["D3D12.X → PC D3D12<br/>translator"]
        A2 --> A3["D3DMetal<br/>(D3D12 → Metal)"]
        A3 --> A4["Metal"]
        A5[("Console unified memory")] -. "copy + hash<br/>GBs every frame" .-> A2
    end
    subgraph NEW["MetalVI: one direct path"]
        direction TB
        B1["Game x86-64 code<br/>(Rosetta 2)"] --> B2["victory-gpu<br/>recorded commands"]
        B2 --> B3["victorymetal unixlib<br/>(Wine → Metal)"]
        B3 --> B4["Metal"]
        B5[("Game memory")] == "zero-copy<br/>newBufferWithBytesNoCopy" ==> B4
    end
```

### Inside MetalVI

```mermaid
flowchart TB
    subgraph WIN["Windows side: under Wine + Rosetta 2"]
        L["victory-loader<br/>PE loader · console-insn patch · masked-store fix"]
        H["victory-hle<br/>kernel · xmem · DirectStorage · XAsync · input · audio"]
        S["victory-shader<br/>RDNA2 ISA → DXIL recompiler"]
        G["victory-gpu<br/>D3D12.X runtime · heaps · textures · Metal backend"]
        M["victory-metal<br/>safe Rust Metal API · zero-copy chunks · residency"]
    end
    subgraph MAC["macOS side"]
        U["victorymetal.so<br/>Wine unixlib (from dxmt)"]
        C["Metal Shader Converter<br/>DXIL → metallib, in process"]
        MT["Metal / Apple GPU"]
    end
    T["Game executable<br/>(your own copy)"] --> L --> H
    H --> G
    S --> G
    G --> M -- "__wine_unix_call" --> U
    U --> C
    U --> MT
```

### Execution architecture v2 (designed, not implemented yet)

Today's Metal backend replaces the graphics API, but it still executes the way a D3D12 port does:
- every submission waits for the GPU;
- the console's queues are collapsed into one;
- GPU dispatches run one at a time;
- nothing is tuned for Apple's tile-based GPU (TBDR).

The goal of v2 is **an architecture that puts no cap of its own on how much of the hardware we can use**.

```mermaid
flowchart LR
    subgraph T["Title threads"]
        R["List recording<br/>→ list IR"]
    end
    subgraph B["Backend"]
        Q["Queue timelines<br/>fences / memory labels → MTLSharedEvent"]
        P["Pass planner (TBDR)<br/>+ barrier translation"]
        E["Encoder pool<br/>one command buffer per list part"]
        D["Completion threads<br/>bottom-of-pipe writes · retirement · duration checks"]
    end
    subgraph G["Apple GPU"]
        GQ["Graphics queue"]
        CQ["Compute queue"]
    end
    R --> Q --> P --> E
    E --> GQ & CQ
    GQ -. "event ordering" .- CQ
    GQ & CQ --> D --> Q
```

| Piece | What it does | What it fixes |
| --- | --- | --- |
| Async queue timelines | One Metal queue per title queue; fences and memory labels become `MTLSharedEvent`s; completion threads do bottom-of-pipe writes and retire resources | CPU/GPU lockstep; cross-queue sync emulated on the CPU |
| Parallel encoding | A thread pool encodes, ordered with `enqueue`; no global lock | Single-threaded encoding that also pays for Rosetta |
| TBDR pass planner | Clears folded into load actions; DontCare and memoryless where provably safe; no pass breaks for barriers that don't touch the attachments | Render targets stored and reloaded again and again, wasting bandwidth |
| Barrier translation | Concurrent compute encoders; only the title's own barriers are translated | Every GPU dispatch serialised |
| Texture ownership | Linear textures zero-copy; GPU-written textures stay on the GPU; CPU-written ones refreshed from known writers and dirty pages, never by hashing | Textures still on the copy-and-hash model |
| Shader tiers | Verified pipelines drop bounds checks and loop budgets; binary archives; NGG shaders take a vertex-shader path where possible | Instrumentation cost, cold-compile stalls, the 32 KiB mesh-output limit |
| Safety v2 | Work bounds at encode time, clamped indirect arguments, in-flight GPU-time budget, asynchronous checks | Safety bought by waiting on every buffer was itself a performance ceiling |

Full design: [docs/metal-architecture-v2.md](docs/metal-architecture-v2.md).

## Status at a glance

Measured on an M1 Max (64 GB, macOS 15.7.9), as of 2026-11-27. Items marked *in progress* are published but incomplete.

| Area | State |
| --- | --- |
| Boot, menus, loading screens | ✅ 20–30 fps (D3DMetal path) |
| Into the 3D world, opening mission | ✅ reached |
| Title stability (CPU side) | ✅ 10-minute runs with no crash, after the fixes below. It was roughly 1 in 2 runs crashing before |
| Compute shaders on native Metal | ✅ 99.3% of dispatches fully bound |
| GPU texture detiling, zero copy from game memory | ✅ byte-exact against the CPU reference, on by default |
| CPU-side 3D frame time (rendering off) | 35–75 ms (about 46–52 ms with msync) |
| Drawing and presenting on Metal | 🚧 *in progress*: runs, but depends on the old D3D12 preparation path. The G-buffer is correct; after lighting and post-processing the picture is mostly black |
| Real 3D average frame rate, stalls included | ⚠️ 1.2–2.5 fps (it hits the game's own 30 fps cap between stalls, but stalls last seconds) |
| GPU safety (a bound on GPU time per submission) | 🚧 *in progress*: split submissions and an after-the-fact fuse, **no hard bound yet** |
| Shader translation | 🚧 7 failures out of 25,953 programs; 27 mesh pipelines exceed Metal's 32 KiB output limit |
| Execution architecture v2 (async multi-queue, TBDR, parallel encoding) | 📐 designed, not implemented |
| Playable | ❌ not yet |

## What's new in this architecture

### Native Metal backend (`crates/victory-metal`, `crates/victory-gpu/src/backend`, `native/victorymetal`)
- **`victorymetal`**: a Wine unixlib forked from [3Shain/dxmt](https://github.com/3Shain/dxmt)'s `winemetal` (LGPL-2.1), so the Windows-side runtime calls Metal through `__wine_unix_call`. It is loaded on demand, so builds without it still run on D3DMetal.
- **Zero-copy title memory.** Game memory is wrapped in 2 MiB chunks of no-copy `MTLBuffer`s. Remaps are re-wrapped. Everything shaders reach by address is kept in an `MTLResidencySet`.
- **Metal Shader Converter inside the process.** The existing RDNA2 → DXIL recompiler feeds MSC, which converted 99.2% of 32,503 cached shaders offline. Its output is cached on disk. Root signatures are built to match victory's layout, with bounds checks on, because a stray write into zero-copy memory would be real corruption.
- **Descriptor heaps mirrored as `IRDescriptorTableEntry` tables**, filled from victory's view table: buffers, typed buffers, textures and texture views (format and type guarded, since Metal asserts on illegal views), D3D12 and console samplers (deduplicated, because Apple7 aliases past 1024 distinct samplers), and zeroed stand-ins for null descriptors.
- **GPU detiling** of console texture layouts (GFX10 swizzle modes) straight from game memory, with a per-subresource CPU fallback.

### Making Rosetta 2 and Wine behave
- **We found the cause of the random heap corruption.** Under Rosetta 2, every AVX masked store (`vpmaskmovd/q`, `vmaskmovps/pd`) is done as a **read-modify-write of the whole 16/32 bytes**, so the lanes the mask leaves off are written back with stale data. In our test, a `vpmaskmovd` loop silently undid **93% of another thread's writes** to a masked-off lane. The game has 1,130 such stores, and they were corrupting its allocator's free lists. They are now rewritten at load time into per-lane plain stores (`crates/victory-loader/src/masked.rs`, from [VIctory-macOS](https://github.com/pragyaangaur/VIctory-macOS)). Result: zero allocator crashes since.
- **Console-only CPU instructions patched once.** `clzero`, `clwb`, `monitorx`/`mwaitx`, `rdpru`, `rdpid` and SSE4a used to trap about 21,000 times per second in 3D. Each site is now emulated once and then rewritten in place (`console_patch.rs`).
- **No Wine deadlocks from file metadata.** Hot paths read cache files without asking for metadata, because under Rosetta that call could freeze every thread.
- **Fewer memory remaps.** Mapping a page where it is already shown is skipped, because under Wine each remap holds a process-wide lock.
- The game's own hang watchdog can be turned off for slow first runs (`VICTORY_NO_HANG_WATCHDOG=1`).

### The engineering behind it

```mermaid
timeline
    title MetalVI: November 2026
    Audit : Measured the ceiling of the copy-based D3DMetal path (≈1–2 fps) : Chose a direct Metal path
    Zero copy : No-copy MTLBuffers over game memory proven under Wine + Rosetta : Residency sets, remap handling
    Shaders : Metal Shader Converter in process : 99.2 % of 32,503 shaders convert
    Compute on Metal : Descriptor heaps, textures, samplers, typed buffers : 99.3 % of dispatches fully bound
    Stability : Rosetta masked-store race found and fixed : Wine metadata deadlock fixed : 10-minute runs, no crash
    Drawing : Metal draws, present and indirect draws in progress : Root cause of the whole-Mac GPU freezes found
    Redesign : Execution architecture v2 : async multi-queue, TBDR pass planner, parallel encoding, safety v2
```

| | |
| --- | --- |
| 🔬 **Measure before building** | The architecture was chosen from measurements (per-frame upload and hash volume, CPU critical path, thread waits), not guesses. |
| 🧩 **A new backend from the ground up** | Wine unixlib, a safe Rust Metal API, a zero-copy memory manager, MSC integration, descriptor-heap mirroring, texture and sampler translation, GPU detiling, and a pipeline disk cache. |
| 🐞 **Root-causing the hard bugs** | A heap corruption that hit about half of all runs was traced to Rosetta 2 emulating AVX masked stores non-atomically. A dedicated race test proved it, and a load-time patch removed it. A full-process freeze was traced to Wine's mount-manager calls under its global memory lock. |
| ✅ **Verified, not assumed** | Every component ships with tests: byte-exact GPU-vs-CPU detile comparisons, native Metal probes of Apple GPU limits (1024-sampler aliasing, no 64-bit atomics, wave32), unit tests under Wine, and A/B game runs with crash statistics. |
| 🤝 **Built in the open** | Research tasks were split so other developers and AI agents can pick them up and verify them without the game. Fixes from neighbouring projects are credited and kept under compatible licences. |

### By the numbers

| Metric | Value |
| --- | --- |
| Shaders converting through Metal Shader Converter | 32,248 / 32,503 (99.2%) |
| Compute dispatches fully bound on Metal | 99.3% |
| Allocator crashes per run, before → after the masked-store fix | ~45% → 0 |
| Console-instruction traps per second in 3D, before → after | ~21,000 → ~1,100 |
| Masked stores rewritten at load | 1,130 |
| Longest verified run | 10 minutes, no crash |

### Why aim this high
- **Unified memory is the Mac's advantage**, and only a direct Metal path can use it. The copy-based design is capped by bandwidth spent on copying, not by the GPU.
- **Rosetta 2 won't last forever.** Apple supports it fully through macOS 27. Talking to Metal ourselves means the renderer does not depend on D3DMetal, which has no ARM64 build, and leaves a clean path to native ARM64 Wine plus FEX later.
- **Series S is the target profile.** Its GPU budget is within reach of M-series GPUs once translation overhead is gone.

## Performance predictions and device compatibility

> **Everything in this section is a prediction, not a measurement.** The only measured machine is an M1 Max, and the real 3D average today is 1.2–2.5 fps. The numbers below assume v2 is finished, the picture is correct, and the Series S profile is used. We will update them, downwards too, once they are measured.

### Where the overhead comes from (compared with the console running natively)

| Source | Today | Predicted after v2 | Basis |
| --- | --- | --- | --- |
| Rosetta 2 running the title's x86-64 code | ~70–80% of native on scalar code, ~50% on AVX2-heavy code | Unchanged (only leaving Rosetta helps) | Public data and our CPU measurements |
| Wine sync and syscalls | ~15% kernel time with msync; workers wait 59–88% of the time | Further wait-path work, limited headroom | Thread-wait sampling |
| Console instructions and masked stores | Traps 21k/s → 1.1k/s | Near zero | Measured |
| Memory copies and hashing | Buffers zero-copy; 2.4 GB of old shadow copies and per-frame hashing remain | 0 for buffers; textures detiled on the GPU for dirty regions only, a few ms per frame | Old route measured 3–6 s per frame |
| CPU/GPU lockstep, single queue | Frame ≈ CPU + GPU | Overlapped: frame ≈ max(CPU, GPU) | Architecture |
| Backend encoding | Single thread, on the submit path | 3–6 ms CPU per frame, off the critical path | Estimate from ~2,300 draws plus mesh dispatches per frame |
| Shader translation (RDNA2 → DXIL → Metal) | Estimated 1.5–3× ALU, plus safety instrumentation | Estimated 1.3–2× for verified pipelines | **Not measured**; one of the biggest unknowns |
| Ray tracing (RDNA2 BVH instructions in software) | Just starting to run; correctness unknown | Unknown, possibly heavy | Not measured |
| Render-target bandwidth (TBDR) | Untuned, frequent pass breaks | On par with or better than immediate-mode rendering | Architecture |

### Ceiling prediction (M1 Max, Series S profile)

- Frame rate ≈ min(CPU rate, GPU rate), capped by the game's own 30 fps.
- **CPU:** a cutscene ran at 33 ms (30 fps) between stalls; a separate CPU-only measurement gave 46–52 ms (~20 fps). So the CPU ceiling is about **20–30 fps**, depending on the scene.
- **GPU:** 10.4 TFLOPS divided by 1.5–3× translation overhead leaves 3.5–7 effective TFLOPS, roughly a Series S (4 TFLOPS). So the GPU side is about **26–30 fps**.
- **Prediction:** **20–30 fps** in most scenes once finished, and **15–20 fps** in heavy scenes such as driving through the city. A locked 30 fps across the whole game is unlikely. Confidence is low to medium; it hinges on the measured costs of shader translation and ray tracing.

### Device compatibility prediction (after v2, Series S profile)

| Mac (base configuration) | CPU | GPU (approx. FP32) | Memory | Prediction | Verdict |
| --- | --- | --- | --- | --- | --- |
| M1 / M2 (8–10-core GPU) | 4P+4E | 2.6–3.6 TF | 8–24 GB | < 15 fps | ❌ not recommended |
| **Base M1 Pro MacBook Pro** (8-core CPU / 14-core GPU / 16 GB) | 6P+2E | ~4.6 TF, 200 GB/s | 16 GB, tight | 15–20 fps with lower resolution or MetalFX; possible stutter from memory pressure | ⚠️ runs, not smooth |
| M3 (10-core GPU) | 4P+4E | ~3.6 TF, 100 GB/s | 8 GB unsupported; 16/24 GB | 12–20 fps with MetalFX | ⚠️ marginal |
| M4 (10-core GPU) | 4P+6E | ~4.3 TF, 120 GB/s | 16 GB and up | 15–25 fps with MetalFX | ⚠️ lower edge of playable |
| M1 Max (24/32-core GPU) | 8P+2E | 7.8–10.4 TF, 400 GB/s | 32 GB and up | 20–30 fps | ✅ current dev machine |
| M3 Pro (14/18-core GPU) | 5–6P+6E | ~5–6.4 TF, 150 GB/s | 18 GB, tight | 20–28 fps | ✅ mostly playable |
| M4 Pro (16/20-core GPU) | 8–10P+4E | ~6.6–8.3 TF, 273 GB/s | 24 GB and up | 25–30 fps | ✅ recommended |
| M3 Max / M4 Max | 10–12P | ~14–18 TF, 300–546 GB/s | 36 GB and up | Hits the 30 fps cap in most scenes | ✅ best |

Notes:
- **Memory:** on the Series S profile the title itself uses about 7.6 GB. Add Metal texture copies, Wine, Rosetta and caches, and we expect 13–17 GB per process (about 30 GB today, because the old shadow path is still in place). So 8 GB Macs are unsupported, 16 GB Macs will swap, and **24 GB or more is the comfortable range**.
- **CPU:** the title runs about 7 heavy threads. Machines with fewer than 6 performance cores tend to be CPU-bound, even with faster cores (M3, M4). M3 and M4 single-core performance is about 30–50% above M1, which is why the M4 does much better at the same core count.
- **GPU:** dynamic caching on M3/M4 may ease register pressure in translated shaders, which would help, but this is not measured. Their hardware ray tracing cannot be used directly, because the game's BVHs are in AMD's format; rebuilding Metal acceleration structures is a long-term option.
- **MetalFX upscaling** is the main lever for lower-end Macs: render at a lower resolution, then upscale the output.
- **System and disk:** macOS 15 or later (`MTLResidencySet` requires it) and Rosetta 2. The game dump is about 209 GB, plus caches, so 256 GB machines can't hold it.

## Roadmap

Done:
- [x] P0: `victorymetal` unixlib, zero-copy chunks, residency, safe Rust wrappers
- [x] P1: compute on Metal in-game: MSC in-process, heaps, textures, samplers, typed buffers, GPU detile, disk cache
- [x] Stability: Rosetta masked-store race, Wine metadata deadlock, hang watchdog
- [x] P2/P3 prototype: Metal draws, present, indirect draws, depth and MSAA resolve. It depends on the old D3D12 preparation path, which v2 replaces

Execution architecture v2 ([design](docs/metal-architecture-v2.md)):
- [ ] Stage 1: GPU safety gate v2 (encode-time work bounds, clamped indirect arguments, in-flight budget, asynchronous checks)
- [ ] Stage 2: observability (per-pass GPU timing), binary archives, replayable frame packages
- [ ] Stage 3: execution model: async multi-queue, fences and labels as events, completion threads, barrier translation
- [ ] Stages 4–5: native resources, bindings and draw preparation, TBDR pass planner, retiring the old copy, watch and hash machinery
- [ ] Stage 6: route decision from a measured frame-budget split
- [ ] Stage 7: rendering completeness: remaining shaders, 32 KiB mesh output, ExecuteIndirect, ray tracing, the black picture
- [ ] Stage 8: performance: shader tiers, parallel encoding, MetalFX
- [ ] Later: native ARM64 Wine + FEX, so it no longer depends on Rosetta

## Building and running on macOS

### Prerequisites
- Apple Silicon Mac, macOS 15 or newer, Rosetta 2 installed.
- A macOS Wine with `macdrv_functions` (a GameHub/gamesir-labs Wine-Proton build works), plus Apple's Game Porting Toolkit for the D3DMetal path.
- **Apple's Metal Shader Converter**, installed from Apple's developer site (it provides `/usr/local/lib/libmetalirconverter.dylib`). MetalVI does not redistribute it: `native/victorymetal/build.sh` links against your own installed copy.
- Your own dump of the game.

```bash
brew install zig llvm@21 lld@21
rustup target add x86_64-pc-windows-gnullvm
cargo install cargo-zigbuild
```

### Build
```bash
export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$PATH"
cargo zigbuild --release -p victory --target x86_64-pc-windows-gnullvm --locked
# The Metal unixlib, installed into your Wine (WINE_ROOT=/path/to/wine):
native/victorymetal/build.sh --install
```

### Configure (outside the repository)
Create a runtime folder outside this clone, for example `~/metalvi-runtime`, with a `victory.toml`:
```toml
[paths]
game_dir = 'Z:\Users\<you>\path\to\your\game_dump'
cache_dir = 'Z:\Users\<you>\metalvi-runtime\cache'
data_dir = 'Z:\Users\<you>\metalvi-runtime\data'
test_titles = []
```

### Pre-warm shaders, then launch
```bash
cd ~/metalvi-runtime
/path/to/MetalVI/scripts/run-mac.sh /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  warm --jobs 6 2>&1 | tee warm.log
VICTORY_METAL=compute VICTORY_NO_HANG_WATCHDOG=1 \
/path/to/MetalVI/scripts/run-mac.sh /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  run --execute --console series-s 'Z:\Users\<you>\path\to\your\game_dump\title.exe' 2>&1 | tee run.log
```
Always pipe through `2>&1 | tee`: under Wine, redirecting with `>` drops the console output.

### Useful switches
| Variable | Effect |
| --- | --- |
| `VICTORY_METAL=compute` | Run compute work on the native Metal backend |
| `VICTORY_METAL_GPU_DETILE=0` | Detile textures on the CPU instead of the GPU |
| `VICTORY_METAL_TEXTURES=0` | Leave texture slots empty (for bisecting) |
| `VICTORY_METAL_CACHE=<dir>` | Where converted Metal pipelines are cached |
| `VICTORY_PATCH_MASKED=none` | Keep the masked stores as they are (not recommended under Rosetta) |
| `VICTORY_NO_HANG_WATCHDOG=1` | Turn off the game's hang watchdog |
| `VICTORY_INSN_PATCH=0` | Trap-and-emulate console instructions every time |

### Keyboard controls
| Input | Action |
| :--- | :--- |
| `Enter` / `Space` | Confirm (A) |
| `Backspace` | Back (B) |
| `W` `A` `S` `D` | Left stick |
| `I` `J` `K` `L` | Right stick (camera) |
| Arrow keys | D-pad |
| `Q` / `E` | LB / RB |
| `Left Shift` / `Right Shift` | LT / RT |
| `Esc` | Menu |

## FAQ

**Can I play GTA VI on my Mac with this today?**
Not yet. It boots, loads the world and starts the story, but most of the 3D picture is still black, and the real average counting stalls is 1.2–2.5 fps. D3DMetal-based ports reach about 0.1–4 fps in 3D.

**Which Macs will it target?**
Technically any Apple Silicon Mac on macOS 15 or later. By our predictions, getting close to playable takes roughly an M1 Pro or better (or an M4) with 24 GB or more; see [device compatibility](#device-compatibility-prediction-after-v2-series-s-profile). Development and measurements are done on an M1 Max.

**Is this an emulator?**
No CPU emulator of our own. The game's x86-64 code runs through Rosetta 2, and Wine provides Windows. MetalVI implements the console's system libraries (high-level emulation), recompiles the console GPU's RDNA2 shaders, and renders through Metal.

**Does it work with CrossOver / Whisky / GPTK?**
The D3DMetal path runs on a GPTK-capable Wine. The native Metal backend needs our `victorymetal` unixlib installed into the Wine you use.

**How is this different from other ports?**
Most efforts run the game through D3D12 → D3DMetal and fight the cost of emulating unified memory. MetalVI's renderer goes straight to Metal with zero-copy memory. That is more work up front, but it has a much higher ceiling on Apple Silicon.

**Where do I get the game files?**
From a copy you own, dumped by you. We can't help with obtaining game files, and issues or PRs that share or link them will be closed.

## Contributing

Contributions from people **and AI coding agents** are welcome. The work is split into small, objectively checkable tasks (pure functions with tests, byte-exact GPU-vs-CPU comparisons, native Metal probes), so a contribution can be verified without the game.
- Good first areas: Metal state mapping, Metal probes for Apple GPU limits, shader lowering for Metal Shader Converter, test tooling.
- Never commit anything from the game. `cargo xtask policy` and the pre-commit hook enforce this (see `docs/asset-policy.md`).
- Please include how you verified a change (commands and output) in your PR.

## Legal

- MetalVI is an independent research and interoperability project. It is **not affiliated with, endorsed or sponsored by** Rockstar Games, Take-Two Interactive, Microsoft or Apple. *Grand Theft Auto*, *Xbox* and *Metal* are trademarks of their respective owners, used here only to describe compatibility.
- The repository holds only original source code under open-source licences. There are no game binaries, assets, keys, SDK headers or proprietary DLLs. Apple's Metal Shader Converter and Game Porting Toolkit are obtained by each user from Apple under Apple's terms, and are not included.
- Use it only with software you legally own. We do not support piracy or DRM circumvention.

## Credits and licences

MetalVI is licensed **GPL-3.0-or-later** (see [LICENSE](LICENSE)). `native/victorymetal` is **LGPL-2.1-or-later** (from dxmt and Wine, see `native/victorymetal/NOTICE.md`).

Built on the work of:
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**: the original loader, HLE runtime, D3D12 translator and RDNA2 shader recompiler.
- **[3Shain/dxmt](https://github.com/3Shain/dxmt)**: the `winemetal` layer that `victorymetal` is forked from.
- **[pragyaangaur/VIctory-macOS](https://github.com/pragyaangaur/VIctory-macOS)**: the masked-store patch, the hang-watchdog patch, and the Wine metadata and remap fixes (GPL-3.0).
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**, **[JoshuuDev/victory-amd](https://github.com/JoshuuDev/victory-amd)**, **[Zettrox/ztrx](https://github.com/Zettrox/ztrx)**: shader, descriptor, texture, DirectStorage and pipeline work.
- Design references (ideas only): xcl, TomBox, five-star-recomp, KosmicKrisp, MoltenVK, Mullion.

---

### 中文

完整中文说明见 **[README.zh-CN.md](README.zh-CN.md)**。

<sub>Keywords: GTA 6 Mac, GTA VI macOS, Grand Theft Auto VI Apple Silicon, M1 M2 M3 M4 M5, Metal backend, Metal Shader Converter, Xbox Series X|S compatibility layer, RDNA2 shader recompiler, Rosetta 2, Wine, Game Porting Toolkit, D3DMetal, zero-copy unified memory, open source.</sub>
