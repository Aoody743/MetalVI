# MetalVI

**MetalVI** is an independent macOS (Apple Silicon) compatibility runtime built for **Rosetta 2 (AVX2)**, **Wine-Proton**, and **Apple Game Porting Toolkit (`D3DMetal`)**, derived under `GPL-3.0-or-later` from upstream Xbox Series compatibility research.

> **Legal Notice:** This repository contains **zero** game binaries, assets, encryption keys, or proprietary Microsoft/Apple/developer DLLs. Bring your own legally obtained game dump and local macOS Wine + GPTK runtime, and store all runtime data, caches, and `victory.toml` **outside** this repository.

---

## Why MetalVI Exists

Upstream Windows/Linux branches assume native `MEM_WRITE_WATCH` hardware fault tracking, bare-metal AMD/NVIDIA Vulkan/D3D12 features (`f64` shader ops, 1M+ descriptor heaps, `OpenExistingHeapFromAddress`), and Windows CRT behavior that break or stall on Apple Silicon under Rosetta 2 + Wine + D3DMetal. MetalVI bridges those gaps so Apple Silicon Macs can compile, test, pre-warm, and execute into 3D world rendering.

### Key macOS & Apple Silicon Engineering Work

1. **Rosetta 2 / Wine Memory Tracking (`crates/victory-gpu/src/watch.rs`, `shadow.rs`, `crates/victory-hle/src/modules/{dstorage,files,xmem}.rs`)**
   - Automatically detects Darwin Wine via `ntdll!wine_get_host_version` and disables `MEM_WRITE_WATCH` / VEH alias page guards (`fault_tracking()` and `view_guard()` default `false` on macOS).
   - Replaces fault-driven dirty tracking with a zero-fault incremental **`diff_copy`** engine backed by a 4-thread persistent **`HashPool`** (128-bit chunk fingerprints) and lock-safe cached VM region tables.
   - Uses **4 MiB address-bucketed `RANGE_EPOCHS`** so texture view remapping (`XMemMapPhysicalPages`) never invalidates unrelated buffer VM caches across Wine's `wineserver`, while guaranteeing `VirtualQuery` never runs while `MAPPING.read()` is held (preventing Wine `_virtual_mutex` deadlocks).

2. **D3DMetal (GPTK) Translation & Runtime Stability (`crates/victory-gpu`, `crates/victory-shader`)**
   - **Dynamic descriptor heap clamping (`descriptor/host.rs`)**: `probe_view_limit()` clamps view indices cleanly within host limits (`PORTABLE_VIEWS` / `UNCAPPED_VIEWS`).
   - **6 GiB placed-texture budget cap (`resource.rs`)**: D3DMetal's `QueryVideoMemoryInfo` reports an inflated local budget on unified memory; capping placed index textures at `6144 MiB` by default on macOS (overridable via `VICTORY_TEXTURE_BUDGET_MB`) eliminates `MTLCommandBuffer` `OutOfMemory` failures.
   - **Shader & pipeline compatibility (`crates/victory-shader`, `pipeline/make.rs`, `indirect.rs`)**:
     - Pure-integer `f64 <-> f32` `active_f64_sum` fallback when host `DoublePrecisionFloatShaderOps` is absent.
     - Sign-bit preservation for `sin(-0.0)` inside `trig_turns`.
     - Explicit `D3D12SerializeRootSignature` serialization (no inline `[RootSignature(...)]` DXIL assumptions), host `depth_bounds()` gating, `(NULL, 0)` empty pipeline-library blob handling, and default skip of `CS_ScreenProbes_ActivateRays` (`VICTORY_ACTIVATE_RAYS=1` to force) to avoid multi-minute Metal compiler stalls.
     - Offline shader & compute pipeline pre-warming via `victory warm` (`apps/victory`).

3. **Wine `ws2_32.dll` & Generic Mac Launcher (`crates/victory-hle/src/modules/net.rs`, `scripts/run-mac.sh`)**
   - Runtime-probed fallbacks for `WSAConnectByList` and `GetAddrInfoExA`.
   - `scripts/run-mac.sh` auto-discovers GameHub / custom `Wine-Proton` and `GPTK` `libd3dshared.dylib`, seeds `d3dcompiler_47.dll`, sets `ROSETTA_ADVERTISE_AVX=1`, and disables `VICTORY_INPLACE`.

---

## Verified Status (Apple M1 Max, 64 GB Unified Memory, macOS 15.7.9)

- **Host Stack**: GameHub `Wine-Proton 11.0`, Apple `GPTK 3.0-3` (`D3DMetal`), `ROSETTA_ADVERTISE_AVX=1`.
- **Shader Pre-Warming (`victory warm --jobs 6`)**: **25,660** shaders translated (98 refused), **1,722** compute pipelines compiled (0 refused) in 854 s (`~1.4 GiB` cache).
- **Unit & Policy Tests (`--test-threads=1` under Wine-Proton)**:
  - `victory_gpu`: **130 passed, 0 failed, 1 ignored**
  - `victory_hle`: **99 passed, 0 failed, 0 ignored**
  - `cargo xtask policy`: **170 files ok** (0 violations)
- **Runtime Progression**:
  - Startup, legal screens, and loading carousel run smoothly at **20–30 fps** with `0` `dstorage` failures.
  - Enters 3D world rendering (**2,326 draws/frame**, `0` `OutOfMemory` errors with the 6 GiB texture cap).
  - Optimized 3D frame breakdown improves from **13,576 ms/frame** baseline to **9,093.8 ms/frame** (`copies` reduced **4.4×** from `10,512 ms` to `2,374.9 ms`; `textures` reduced **3.4×** from `936 ms` to `278.4 ms`).
  - *Known macOS caveat*: 3D gameplay remains CPU/synchronization-bound (~0.1 fps in full 3D), and an intermittent multi-threaded Rosetta streaming race (`+0x4dac5bf`) can occur during heavy asset streaming.

---

## Building & Running on macOS

### Prerequisites

```bash
brew install zig llvm@21 lld@21
rustup target add x86_64-pc-windows-gnullvm
cargo install cargo-zigbuild
```

### Build

```bash
export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$PATH"
cargo zigbuild --release -p victory --target x86_64-pc-windows-gnullvm --locked
cargo zigbuild --tests --workspace --target x86_64-pc-windows-gnullvm --locked
```

### Configure & Pre-Warm (Outside Repo)

Create a runtime directory outside this clone (for example `~/metalvi-runtime`) containing `victory.toml`:

```toml
[paths]
game_dir = 'Z:\Users\<you>\path\to\game_dump'
cache_dir = 'Z:\Users\<you>\metalvi-runtime\cache'
data_dir = 'Z:\Users\<you>\metalvi-runtime\data'
test_titles = []
```

Pre-warm shaders and compute pipelines once from the runtime directory:

```bash
chmod +x scripts/run-mac.sh
cd ~/metalvi-runtime
/path/to/MetalVI/scripts/run-mac.sh \
  /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  warm --jobs 6 2>&1 | tee warm.log
```

### Launch

> **macOS Wine logging tip:** Redirecting Wine directly with `> run.log` drops Windows console output; always pipe through `2>&1 | tee run.log`.

```bash
cd ~/metalvi-runtime
/path/to/MetalVI/scripts/run-mac.sh \
  /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  run --execute 'Z:\Users\<you>\path\to\game_dump\title.exe' 2>&1 | tee run.log
```

### Keyboard Controls (Built-in GameInput Mapping)

| Input | Action |
| :--- | :--- |
| `Enter` / `Space` | Confirm (`A`) |
| `Backspace` | Back (`B`) |
| `W` / `A` / `S` / `D` | Left Stick |
| `I` / `J` / `K` / `L` | Right Stick (Camera) |
| `Arrow Keys` | D-Pad |
| `Q` / `E` | `LB` / `RB` |
| `Left Shift` / `Right Shift` | `LT` / `RT` |
| `Esc` | Menu |

---

## Credits & License

Licensed under **`GPL-3.0-or-later`** (see [LICENSE](LICENSE)).

MetalVI integrates and builds upon the work of:
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — original loader, HLE, D3D12 translator, and DXIL-to-DXBC pipeline.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** — shader optimization, descriptor management, and relay copy-back architecture.
- **[JoshuuDev/victory-amd](https://github.com/JoshuuDev/victory-amd)** (`myokex/victory-amd`) — partial-mip texture handling, `VICTORY_TEXTURE_SWEEP`, `victory warm`, `dstorage`/`xmem` fixes, and split indirect dispatches.
- **[Zettrox/ztrx](https://github.com/Zettrox/ztrx)** — parallel GPU/CPU pipeline tuning references.
- **[pragyaangaur/VIctory-macOS](https://github.com/pragyaangaur/VIctory-macOS)** & **[ivall/xcl](https://github.com/ivall/xcl)** — community cross-platform references.
