# 环境开关参考（VICTORY_*）

> 由 `metal-backend/tools/switches.py` 自动生成；扫描基线：`13a642e (13a642e531a6ab84b30e463ee9cc5fcfe00454ef)`。
> 本表只做静态源码扫描：未设置时的默认值依据代码模式推断，推断不出的条目标记为“未判定”，需人工核对。注释/语义取自开关所在位置上方的源码注释。

开关总数：**189**；按分组统计见下。脚本自身变量见文末单独一节。

## 分组计数

| 分组 | 数量 |
|---|---|
| Metal | 84 |
| shaders | 32 |
| audio | 4 |
| diagnostics | 31 |
| memory | 18 |
| loader | 5 |
| formats | 2 |
| config | 6 |
| host-gpu | 7 |

## Metal

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_VIEW_LIMIT` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | `VICTORY_VIEW_LIMIT=<null slot>`: translate as on a host whose view heap is capped | apps/victory/src/main.rs:168 | direct |
| `VICTORY_METAL` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | What `VICTORY_METAL` asks for. | crates/victory-gpu/src/backend/metal.rs:31 | direct |
| `VICTORY_METAL_TEXTURES` | 未设置：开启（true） [map_or(true, ...)] | Whether title textures are made on Metal (`VICTORY_METAL_TEXTURES=0`: texture slots stay empty, as before textures; `fill`: made but not filled from title memory; `nofill` is the same). | crates/victory-gpu/src/backend/metal.rs:49 | direct |
| `VICTORY_METAL_GPU_DETILE` | 未设置：开启（true） [map_or(true, ...)] | Whether title-filled textures are detiled on the GPU (`VICTORY_METAL_GPU_DETILE=0`: on the CPU, as before). | crates/victory-gpu/src/backend/metal.rs:56 | direct |
| `VICTORY_METAL_REWRAP` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Counts mapping changes. Recorded Metal lists can still contain old GPU addresses, so the post-map notification cannot safely re-wrap memory. | crates/victory-gpu/src/backend/metal.rs:69 | direct |
| `VICTORY_METAL_SPLIT` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | `VICTORY_METAL_SPLIT=1`: every pass in a command buffer of its own, waited for, so a GPU fault names the pass that made it. | crates/victory-gpu/src/backend/metal.rs:429 | direct |
| `VICTORY_METAL_CENSUS` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | P2 census of vertex/pixel pipelines already accepted by the legacy host. Each stage gets standard MSC conversion on a background thread; only pipelines without a vertex input layout are tested for PSO creation. This does not exercise the… | crates/victory-gpu/src/backend/metal.rs:1154 | direct |
| `VICTORY_METAL_DRAW` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Whether draws are recorded on Metal too. | crates/victory-gpu/src/backend/metal_draw.rs:40 | direct |
| `VICTORY_METAL_CAPTURE_EVERY` | 未设置：0 [unwrap_or] | Every Metal render target this many presents is read back to `VICTORY_METAL_CAPTURE_DIR` (`VICTORY_METAL_CAPTURE_EVERY`, 0: never). | crates/victory-gpu/src/backend/metal_draw.rs:620 | direct |
| `VICTORY_METAL_CAPTURE_DIR` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | Reads Metal texture `resource` back (level 0, slice 0) and writes it as a binary PPM (8-bit RGBA/BGRA formats only; others are skipped) to `<dir>/<name>.ppm`. | crates/victory-gpu/src/backend/metal_draw.rs:629 | direct |
| `VICTORY_METAL_PRESENT` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Whether presents go through Metal. | crates/victory-gpu/src/backend/metal_present.rs:21 | direct |
| `VICTORY_METAL_GPU_DETILE_MIN_KB` | 未设置：4096 [unwrap_or] | The detile on the GPU, reading the title's memory where it is (zero copy), for every subresource the kernels take (F10, `metal_detile`); the CPU does the rest into the same staging. On the GPU only where it pays (F16: under ~6 MB a comma… | crates/victory-gpu/src/backend/metal_texture.rs:228 | direct |
| `VICTORY_TEXTURE_BACK` | 未设置：关闭（存在即开启） [is_some()] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/backend/metal_writeback.rs:61 | direct |
| `VICTORY_CAPTURE` | 未设置：None（可选路径） [map(PathBuf::from)] | The folder frames go to, if capturing. | crates/victory-gpu/src/capture.rs:15 | direct |
| `VICTORY_CAPTURE_FRAMES` | 未设置：空字符串 / 零值 [unwrap_or_default] | Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved too, and their draws traced at debug level. | crates/victory-gpu/src/capture.rs:25 | direct |
| `VICTORY_CAPTURE_WORLD_ALSO` | 未设置：空字符串 / 零值 [unwrap_or_default] | `VICTORY_CAPTURE_WORLD`: also the 2nd and the 6th frame after the first one that ran 500 draws or more (a title's first 3D frames: their numbers depend on how long the menus were shown), and the 30th and the 80th (the title opens with a … | crates/victory-gpu/src/capture.rs:43 | direct |
| `VICTORY_CAPTURE_WORLD` | 未设置：关闭（存在即开启） [is_some()] | Whether `VICTORY_CAPTURE_WORLD` is set. | crates/victory-gpu/src/capture.rs:120 | direct |
| `VICTORY_CAPTURE_MENUS` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Two frames in a row each time: the title draws to two buffers in turn, and every 8th or 16th frame alone only ever showed one of them. Before the world, with `VICTORY_CAPTURE_WORLD` and `VICTORY_CAPTURE_MENUS=1`: every 16th frame too (sm… | crates/victory-gpu/src/capture.rs:154 | direct |
| `VICTORY_COPY_MODE` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | How copies run: 0 guarded (as always), 1 plain, 2 not at all. | crates/victory-gpu/src/copystat.rs:141 | direct |
| `VICTORY_VIEW_INLINE` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether constant buffer views and views of placed memory are named whole in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0` makes every view a table entry, as before. | crates/victory-gpu/src/descriptor.rs:418 | direct |
| `VICTORY_STALE_TEXTURES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a view written over memory that older views describe otherwise makes those stale; `VICTORY_STALE_TEXTURES=0` leaves them as they are. | crates/victory-gpu/src/descriptor/host.rs:271 | direct |
| `VICTORY_TABLED_TEXTURES` | 未设置：开启（=1 关闭） [is_none_or(v != "1")] | Whether slots draws read through tables leave the queue of textures picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as before. | crates/victory-gpu/src/descriptor/host.rs:336 | direct |
| `VICTORY_TEXTURE_SWEEP` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Slots looked at at a time, and the least time between. | crates/victory-gpu/src/descriptor/host.rs:1151 | direct |
| `VICTORY_VIEW_EXTENT` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a buffer view is brought up to date in the host copy its slot shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the copy up by the view's range alone, as before. | crates/victory-gpu/src/descriptor/host.rs:1893 | direct |
| `VICTORY_KEEP_VIEWS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a slot written with another view of the placed texture it shows keeps showing the old one until the new is made; `VICTORY_KEEP_VIEWS=0` makes it null at once, as before. | crates/victory-gpu/src/descriptor/host.rs:2502 | direct |
| `VICTORY_BAKE_TEXTURES_MS` | 未设置：1500（闭包形参默认值） [closure-default] | bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special). The pass ran only from the first world frame on, where it met some 24,000 slots at once and needed 60-80 frames for them. The title bakes its characters' skin and decorations i… | crates/victory-gpu/src/draw.rs:94 | closure |
| `VICTORY_EARLY_TEXTURES_MS` | 未设置：12（闭包形参默认值） [closure-default] | bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special). The pass ran only from the first world frame on, where it met some 24,000 slots at once and needed 60-80 frames for them. The title bakes its characters' skin and decorations i… | crates/victory-gpu/src/draw.rs:94 | closure |
| `VICTORY_WRITE_SKIP` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a buffer range a shader names through an unordered access view is given to shaders as written (snapshot before the submission, compare and write back after) only if that shader's code can store to buffers. The title binds buffers… | crates/victory-gpu/src/draw.rs:119 | direct |
| `VICTORY_INDIRECT_WINDOW` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | never opened: run 87, with the switch, still clamped 5,068 records a frame one at a time. What it costs the guard against garbage records: nothing against a hang (every record is bounded as before and charged to the list's budget as befo… | crates/victory-gpu/src/draw.rs:184 | direct |
| `VICTORY_HEAP_RECHECK` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether textures picked by index are looked at again after their upload (`VICTORY_HEAP_RECHECK=0`: no). | crates/victory-gpu/src/draw.rs:416 | direct |
| `VICTORY_HEAP_TEXTURES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | draws read through tables are not in the queue. shortcut: uploaded once, and again only when the title rewrites the slot. A placed texture not all mapped yet is uploaded with what is there and looked at again (1, 2, 4, ... frames later, … | crates/victory-gpu/src/draw.rs:1606 | direct |
| `VICTORY_TEXTURE_STAGE_MB` | 未设置：192 [unwrap_or] | Bytes of texture memory staged per frame. `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of textures (a scene's first minutes) then drains in fewer frames. | crates/victory-gpu/src/draw.rs:1614 | direct |
| `VICTORY_REUPLOAD_WHEN_FULL` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | No more of them are made while the card has no room (`resource::room`): by what is alive now, not by what was ever staged (4 GB in all, after which later textures stayed empty for the run). The ones nothing names any more are let go, whi… | crates/victory-gpu/src/draw.rs:1637 | direct |
| `VICTORY_TEXTURE_STAGE_MS` | 未设置：250 [unwrap_or] | Also within a time per pass: the pass runs on a thread that records the frame, and making and staging 192 MB of textures took 1.6-2.2 s of each of the world's first twenty frames (run 20). The rest waits for the next frames, as it does p… | crates/victory-gpu/src/draw.rs:1681 | direct |
| `VICTORY_TEXTURE_CATCHUP` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black surfaces at 192 MB a frame) gets three times the bytes and the time until it drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget. | crates/victory-gpu/src/draw.rs:1687 | direct |
| `VICTORY_LATE_WAITS` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | A blocking wait for an event that has happened since the last wait for it ended (a frame took longer than a tick) ends at once, on the tick that passed: sleeping on to the next one cost 17 ms of every 730 ms frame (run 58: "2.0 frame wai… | crates/victory-gpu/src/frame.rs:138 | direct |
| `VICTORY_LAZY_PRELUDE` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a host list's prelude is made when the list first needs one. A prelude runs the texture transitions and twin copies a list needs before it; lists that name no texture never have one, and every host list was made with one all the … | crates/victory-gpu/src/host.rs:900 | direct |
| `VICTORY_CARD_REHASH_RUNS` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether the pieces hashed again after ranges were sent up to a copy in the card's memory are hashed in runs of pieces that touch, each piece once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass per range sent up. These … | crates/victory-gpu/src/host.rs:1803 | direct |
| `VICTORY_WIDE_MERGE` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether what shaders wrote to copies in the card's memory is merged into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by default). The pieces the card's compare found differing come down in one fetch and were then merged … | crates/victory-gpu/src/host.rs:1825 | direct |
| `VICTORY_LATE_WAIT` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]). | crates/victory-gpu/src/host.rs:2001 | direct |
| `VICTORY_RELAY` | 未设置：开启（=2 走特殊分支） [is_none_or(v == "2")] | Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default since the walls of the first room were found drawn in every run with this set and in fewer than half without it). | crates/victory-gpu/src/host.rs:2101 | direct |
| `VICTORY_LATE_WAIT_LISTS` | 未设置：256 [unwrap_or] | The most host lists in the air before the flights are settled (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the relay it stopped the submitting thread twice a frame for 92-103 ms in all while the card stood idle 64-11… | crates/victory-gpu/src/host.rs:2194 | direct |
| `VICTORY_LATE_WAIT_WRITES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a submission with bottom-of-pipe writes may fly (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so the title is told at once, as before; for telling a title that polls such a value from one that does not). | crates/victory-gpu/src/host.rs:2204 | direct |
| `VICTORY_JOIN_COPIES` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | page that is not there instead of faulting (`copy_mapped`), tells the write watch and the guards, and ends the refreshes gathered so far: a call into the kernel for some 480 bytes, 14,042 times a frame in run 86 (6.8 MB; "moves 19 ms" wi… | crates/victory-gpu/src/host.rs:2909 | direct |
| `VICTORY_CARD_PARALLEL_HASH` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/host.rs:3251 | direct |
| `VICTORY_WIDE_REFRESH` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether a submission's refreshes of host copies run on several threads, a copy to a thread (`VICTORY_WIDE_REFRESH=1`; off by default). A submission brings some sixty copies up to date, one after the other on the submitting thread: for ea… | crates/victory-gpu/src/host.rs:5511 | direct |
| `VICTORY_SPARSE_REFRESH` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/host.rs:5564 | direct |
| `VICTORY_DIRECT_WRITES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether title memory is written with the kernel's own call (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks what the pages are (as `VirtualQuery` does: about 2 ms a call in the title's big pools, see [`copy_mapped`… | crates/victory-gpu/src/host.rs:5903 | direct |
| `VICTORY_INDIRECT_SCOPED_BARRIERS` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT. | crates/victory-gpu/src/indirect.rs:479 | direct |
| `VICTORY_LIST_POOL` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether host lists that have run are used again (on by default; `VICTORY_LIST_POOL=0`: each is made new and thrown away, as before). | crates/victory-gpu/src/list.rs:243 | direct |
| `VICTORY_READ_BACK` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | the method's own arguments | crates/victory-gpu/src/list.rs:1130 | direct |
| `VICTORY_CUT_WRITES` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/list.rs:2117 | direct |
| `VICTORY_BVH_QUARANTINE` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store stays in the host's copies and is not copied back into title memory. A check for a first run of the whole builder: a stage that stores where it should not can then not… | crates/victory-gpu/src/pipeline.rs:287 | direct |
| `VICTORY_SKIP_PIPELINES` | 未设置：空字符串 / 零值 [unwrap_or_default] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/pipeline.rs:457 | direct |
| `VICTORY_ROOT_SRV_DIRECT` | 未设置：为真（变量存在则为假） [is_none()] | Whether the title's root SRVs are bound as one-view tables of the buffer's host copy, whose reads past its end return zeros, rather than as root descriptors, which have no size: a mesh shader's vertex fetch has no record limit, and a rea… | crates/victory-gpu/src/pipeline/host.rs:358 | direct |
| `VICTORY_LIBRARY_MAP` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether the library's file is mapped ([`Blob`]) and saved through a mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before. | crates/victory-gpu/src/pipeline/library.rs:81 | direct |
| `VICTORY_LIBRARY_HANDLE` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether the mapped library file's handle is kept open while it is mapped (see the module's notes: without it a save in the same run turns the mapping's pages into zeros as they leave memory). On by default; `VICTORY_LIBRARY_HANDLE=0` clo… | crates/victory-gpu/src/pipeline/library.rs:90 | direct |
| `VICTORY_LIBRARY_TRIM` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Lets the mapped library file's pages go from the process's memory once no pipeline has been loaded for ten seconds. The host reads nearly the whole file while the early pipelines load (run 65: all 1,490 MB of it were in the process's mem… | crates/victory-gpu/src/pipeline/library.rs:374 | direct |
| `VICTORY_LIBRARY_LOCK` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before. | crates/victory-gpu/src/pipeline/library.rs:433 | direct |
| `VICTORY_GRAPHICS_LIBRARY` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether graphics pipelines skip the library and are created directly. D3DMetal's `LoadPipeline` holds one library-wide lock while it compiles the stored pipeline (its blob keeps descriptions, not Metal binaries), so every worker queues b… | crates/victory-gpu/src/pipeline/library.rs:446 | direct |
| `VICTORY_BVH_BUILDER` | 未设置：关闭（=1 开启） [closure: is_some/ok_and(v == "1")] | Whether a shader only shades what the console's ray-tracing structures were hit by: about 1,000 compute programs of the title, all bound at the world's first frame, each 5 to 9 s of AMD's compiler. They stood in the queue before the pass… | crates/victory-gpu/src/pipeline/make.rs:87 | closure |
| `VICTORY_RT_RESOLVE` | 未设置：关闭（=1 开启） [closure: is_some/ok_and(v == "1")] | Whether a shader only shades what the console's ray-tracing structures were hit by: about 1,000 compute programs of the title, all bound at the world's first frame, each 5 to 9 s of AMD's compiler. They stood in the queue before the pass… | crates/victory-gpu/src/pipeline/make.rs:87 | closure |
| `VICTORY_PIPELINE_WAIT_MS` | 未设置：default [unwrap_or] | Waiting allowed per title frame, over all threads. | crates/victory-gpu/src/pipeline/make.rs:167 | direct |
| `VICTORY_PIPELINE_GRAPHICS_GATE_MS` | 未设置：default [unwrap_or] | What a frame may wait while a world loads: pipelines are asked for by the hundred then, and a pass the title runs once is lost if its pipeline is not there. All such waits of a run together are bounded; the title's GPU watchdog bounds on… | crates/victory-gpu/src/pipeline/make.rs:193 | direct |
| `VICTORY_PIPELINE_GATE_S` | 未设置：120 [unwrap_or] | `VICTORY_PIPELINE_GATE_S`: the bound on all of them (0: none of them). | crates/victory-gpu/src/pipeline/make.rs:210 | direct |
| `VICTORY_PIPELINE_WAIT_SHARED` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether the frame's budget is one for all threads together: a thread that waits takes what is left of it and gives back what it did not use. Counted only when a wait had ended, the budget was one per thread: each of the title's five reco… | crates/victory-gpu/src/pipeline/make.rs:284 | direct |
| `VICTORY_SPLIT_COMPUTE` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Long mesh compiles can occupy all of D3DMetal's graphics workers. | crates/victory-gpu/src/pipeline/make.rs:365 | direct |
| `VICTORY_QUEUE_WAIT` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not, and every submission runs when it is made (as before). | crates/victory-gpu/src/queue.rs:100 | direct |
| `VICTORY_SPLIT_WAITS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | one whose wait for a memory value is not met run at once, and each list that waits is held with the lists after it, behind its own waits. Holding a submission whole made the title's queues wait for each other: a graphics submission was h… | crates/victory-gpu/src/queue.rs:119 | direct |
| `VICTORY_CUT_WAITS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | (`crate::list::cut`), and a queue runs a submission part by part: it stands at each wait until the value is in memory, with everything recorded before the wait run and its bottom-of-pipe writes done, as on the console. Held list by list … | crates/victory-gpu/src/queue.rs:138 | direct |
| `VICTORY_WAIT_FORCE` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether a present runs what queues still hold (on by default: a wait is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for finding out whether the title's waits reach across its presents; a wait nothing ever meets then st… | crates/victory-gpu/src/queue.rs:148 | direct |
| `VICTORY_DEPTH_TWINS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | R32 and R16 typeless: the families of the depth formats of one plane. | crates/victory-gpu/src/resource.rs:447 | direct |
| `VICTORY_ALIAS_COPY` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | of them only: the other was filled from title memory, which nothing writes, so everything read through such a view was black. Textures of several levels or layers count too (the screen probes' light atlas has four levels, the camera's sk… | crates/victory-gpu/src/resource.rs:1229 | direct |
| `VICTORY_LAZY_TEXTURES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether host copies are made on first use. On by default; `VICTORY_LAZY_TEXTURES=0` makes each when the title creates its texture, as before. | crates/victory-gpu/src/resource.rs:1271 | direct |
| `VICTORY_TEXTURE_BUDGET_MB` | 未设置：None（可选） [.ok()] | Whether more host copies of textures the title fills may be made for shaders that pick textures by index (`draw.rs` `heap_textures`): always up to `FLOOR` bytes of them, beyond that while the card's memory in use by this process is under… | crates/victory-gpu/src/resource.rs:1388 | direct |
| `VICTORY_TEXTURE_ROOM_PERCENT` | 未设置：90 [map_or] | The share of the card's budget, in percent, up to which host copies of textures picked by index are made ([`room`]): 90 unless `VICTORY_TEXTURE_ROOM_PERCENT` says another (50 to 97). At 90 about a tenth of the budget, 1,020 MB of 10,200,… | crates/victory-gpu/src/resource.rs:1425 | direct |
| `VICTORY_PLACED_FILLER` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Words 1 and 13 of the description are the filler of its layout (after the dimension, and at the end): the title leaves there what its stack held, and two views of one texture that differed only in them each got a resource and a host copy… | crates/victory-gpu/src/resource.rs:2145 | direct |
| `VICTORY_WIDE_BACK` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | On by default; `VICTORY_WIDE_BACK=0` turns it off. | crates/victory-gpu/src/shadow.rs:992 | direct |
| `VICTORY_TEXTURE_NO_WAIT` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | have 4, 15 and 16 frame lines with over 300 ms of "textures", up to 3,708 ms, and on every one of them the wait for this lock is the whole of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12 ms, the texture's own lock … | crates/victory-gpu/src/texture.rs:326 | direct |
| `VICTORY_TEXTURE_COPY` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Whether a texture all of whose memory is mapped is read through the kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by default). Round 3 of the memory work read every such texture that way to be safe from pages taken awa… | crates/victory-gpu/src/texture.rs:388 | direct |
| `VICTORY_TEXTURE_GUARD` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | A texture that changed without the guard seeing it is hashed at every check for a while (the intro's movie: three pictures in turn that only changed every eighth frame, one old picture after another on the screen). | crates/victory-gpu/src/texture.rs:756 | direct |
| `VICTORY_METAL_CPU_CUTS` | 未设置：None（可选） [closure: .ok()] | Whether D3D12-only host copies, descriptor mirroring, and write-watch tracking can be skipped because Metal draws and presents the window (`VICTORY_METAL=compute`, `VICTORY_METAL_DRAW=1`, `VICTORY_METAL_PRESENT=1`), or `VICTORY_METAL_CPU… | crates/victory-gpu/src/watch.rs:149 | closure |
| `VICTORY_GUARD_HOLES` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | range can begin in a block that is not mapped right now: the first pages of the range that are there decide. A range with nothing there counts as a view (its blocks are noted absent, and looked at again when the runtime maps them). Looki… | crates/victory-gpu/src/watch.rs:635 | direct |
| `VICTORY_VIEW_GUARD` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether all the views of a guarded page are guarded together (`VICTORY_VIEW_GUARD`; off by default when page-fault tracking is disabled). | crates/victory-gpu/src/watch.rs:704 | direct |

## shaders

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_HOST_WAVE` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | `VICTORY_HOST_WAVE=32`: translate as for a GPU whose waves are at most 32 lanes | apps/victory/src/main.rs:172 | direct |
| `VICTORY_HOST_MAX_WAVE` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | `VICTORY_HOST_MAX_WAVE=32`: translate as for a host whose waves have at most 32 lanes (as the runtime does on such a card). | apps/victory/src/translate.rs:165 | direct |
| `VICTORY_PAIRS` | 未设置：空字符串 / 零值 [unwrap_or_default] | （注释缺失；语义需人工核对） | apps/victory/src/translate.rs:180 | direct |
| `VICTORY_TRANSLATE_ONLY` | 未设置：None（可选路径，留空等同未设置） [filter/map optional path] | Diagnostics: `VICTORY_TRANSLATE_ONLY=<hash prefix>,...` translates those containers alone. | apps/victory/src/translate.rs:399 | direct |
| `VICTORY_SHADER_CENSUS` | 未设置：错误向上返回（测试/调试路径） [? propagation] | `VICTORY_SHADER_CENSUS=<file>`: one line per emitted program, appended: its label, the console program's instructions by encoding class (a static count of the whole program, not of one thread's path), and the host module's instructions b… | apps/victory/src/translate.rs:566 | direct |
| `VICTORY_EMIT_DUMP` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file. | apps/victory/src/translate.rs:614 | direct |
| `VICTORY_MESH_TEST` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none is issued), 2 for `scissor` (each runs its shaders but rasterises one pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung by what the shader's tri… | crates/victory-gpu/src/draw.rs:303 | direct |
| `VICTORY_D3D12_WARP` | 未设置：关闭（存在即开启） [is_some()] | Diagnostics: the software adapter, on which a shader's wild access is an access violation in this process (a crash report with the address and instruction) instead of a GPU fault and driver reset. | crates/victory-gpu/src/host.rs:655 | direct |
| `VICTORY_SHADER_SAVE` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as translated (`.dxil`), by the hash crash reports name it with | crates/victory-gpu/src/pipeline.rs:161 | direct |
| `VICTORY_MESH_OUTPUT_LIMIT` | 未设置：关闭标志为假（默认常规路径；置 0 才生效） [is_some_and(v == "0")] | The most bytes of outputs a mesh pipeline may have on the host: Metal's 32,768 when Wine runs on macOS (D3DMetal), else no known limit. `VICTORY_MESH_OUTPUT_LIMIT=0` turns the check off. | crates/victory-gpu/src/pipeline/host.rs:951 | direct |
| `VICTORY_PC_DXC` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's). | crates/victory-gpu/src/pipeline/host.rs:1171 | direct |
| `VICTORY_WAVE_WIDE` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:81 | cache-list |
| `VICTORY_WAVE_HALF_SYNC` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:82 | cache-list |
| `VICTORY_WAVE_DEEP_SKIPS` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:83 | cache-list |
| `VICTORY_WAVE_OWN_BITS` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:84 | cache-list |
| `VICTORY_WAVE_LOCAL` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:85 | cache-list |
| `VICTORY_F16_NEAREST` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:86 | cache-list |
| `VICTORY_STRICT_JOINS` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:87 | cache-list |
| `VICTORY_MESH_OUTPUT_STRICT` | 未判定：读取点在其他文件（见 direct/closure 条目） [cache-list] | So are the switches that pick the form of a wave64 program: without them here a run with `VICTORY_WAVE_WIDE=0` was served the translations an earlier run had cached with the switch unset. | crates/victory-gpu/src/shader_cache.rs:88 | cache-list |
| `VICTORY_CONSOLE_DXC` | 未设置：错误向上返回（测试/调试路径） [? propagation] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/diff_tests.rs:655 | direct |
| `VICTORY_DIFF_ONLY` | 未设置：None（可选） [.ok()] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/diff_tests.rs:775 | direct |
| `VICTORY_WARP` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/diff_tests.rs:1177 | direct |
| `VICTORY_WAVE_WHY` | 未设置：关闭（存在即开启） [is_some()] | `VICTORY_WAVE_WHY=1`: why a program keeps 64-lane waves (one line per reason) | crates/victory-shader/src/emit/lower.rs:1458 | direct |
| `VICTORY_FIRST_HALF` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | A first half whose result is also read as data (an f16 second half keeps the other half of its dst, PS_DeferredTextured): evaluated as the pair it starts. Approximate for those other bits (P0 + i P10 on the console); a first half only th… | crates/victory-shader/src/emit/lower.rs:1620 | direct |
| `VICTORY_DESC_WHY` | 未设置：关闭（存在即开启） [is_some()] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/lower.rs:6191 | direct |
| `VICTORY_FIRSTLANE_OWN` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | shortcut: `VICTORY_FIRSTLANE_OWN=1` takes the host wave's own first lane, exact when the value is the same in every lane of the console wave (the ray-tracing builder's scans read a sum of scalars this way, inside a loop whose exit is per… | crates/victory-shader/src/emit/lower.rs:8186 | direct |
| `VICTORY_SHADER_LOOP_LIMIT` | 未设置：None（可选） [.ok()] | The iteration count after which a translated loop is left, per lane and per entry into the loop: [`DEFAULT_LOOP_LIMIT`]. With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or `0x` hex, at least 1) sets it instead. A loo… | crates/victory-shader/src/emit/mod.rs:146 | direct |
| `VICTORY_GROUP_SYNC` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | barrier is the wave's, whatever lanes are live, and a wave that ended is not waited for.) `VICTORY_GROUP_SYNC`: - unset or `2`: every barrier waits, as before. - `1`: the title's `s_barrier` waits only in a block every lane runs, in a pr… | crates/victory-shader/src/emit/mod.rs:173 | direct |
| `VICTORY_SHADER_OPT` | 未设置：None（可选） [.ok()] | is exact, so the switch is for telling a translator fault from a clean-up fault, and for measuring what the clean-up is worth. `VICTORY_SHADER_OPT`: - unset: every pass. - `0`: none (the module as the lowering made it). - a number (decim… | crates/victory-shader/src/emit/mod.rs:197 | direct |
| `VICTORY_REF_PS` | 未设置：crate::HwStage::Ps（闭包形参默认值） [closure-default] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/mod.rs:1129 | closure |
| `VICTORY_REF_VS` | 未设置：crate::HwStage::Vs（闭包形参默认值） [closure-default] | （注释缺失；语义需人工核对） | crates/victory-shader/src/emit/mod.rs:1130 | closure |
| `VICTORY_DXIL` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | Where `dxil.dll` is looked for. | crates/victory-shader/src/emit/validator.rs:122 | direct |

## audio

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_XMA_STREAMS` | 未设置：报错退出（仅测试/调试路径） [unwrap/expect] | Walks and decodes saved input buffers (`VICTORY_XMA_STREAMS=<dir>` of `*.xma` files with their `*.ctx` context words, made outside the repository): the frames walked from the first packet must be the frames the packet headers count, and … | crates/victory-formats/src/xma.rs:1163 | direct |
| `VICTORY_XMA_REFERENCE` | 未设置：报错退出（仅测试/调试路径） [unwrap/expect] | Decodes a stream's packets and compares with a reference decode of the same file (16-bit PCM): `VICTORY_XMA_REFERENCE=<dir>` holding `mono.xmadata`/`mono.ref` and `stereo.xmadata`/`stereo.ref` (made outside the repository). | crates/victory-formats/src/xma.rs:1208 | direct |
| `VICTORY_XMA_DUMP` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | `VICTORY_XMA_DUMP=<dir>`: saves the context words (`<n>-<index>.ctx`) and input buffer 0 (`.xma`) of the first 32 voices enabled, to test the decoder on a title's streams outside the title. | crates/victory-hle/src/modules/audio.rs:669 | direct |
| `VICTORY_AUDIO_CAPTURE` | 未设置：关闭（存在即开启） [is_some()] | SAFETY: plain wait. | crates/victory-hle/src/modules/audio/capture.rs:69 | direct |

## diagnostics

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_PATCH_MASKED` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | Rosetta does AVX masked stores as read-modify-writes of all their lanes, undoing other threads' writes to the lanes masked off (the pool-header crashes): they are redone as plain per-lane stores (`victory_loader::masked`). `VICTORY_PATCH… | apps/victory/src/run.rs:45 | direct |
| `VICTORY_NO_HANG_WATCHDOG` | 未设置：None（可选路径，留空等同未设置） [filter/map optional path] | The title's hang watchdog off, for slow loads and first world frames. | apps/victory/src/run.rs:52 | direct |
| `VICTORY_CRASH_SAVE` | 未设置：空字符串 / 零值 [unwrap_or_default] | Debugging aid: `VICTORY_CRASH_SAVE=<rva>:<len>,...` (hex) writes those title ranges to `crash-<rva>.bin` in the cache dir on a crash. | apps/victory/src/run.rs:177 | direct |
| `VICTORY_WATCH` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | Debugging aid: `VICTORY_WATCH=<rva>,...` (hex) logs the u32 at each title+rva whenever it changes, polled every 5 ms (an engine's state, say). | apps/victory/src/run.rs:197 | direct |
| `VICTORY_DUMP_MATCH` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | `VICTORY_DUMP_MATCH=<dir>;<text>\|<text>...`: the containers whose line holds one of the texts (a name, `#hash`), as the title has them. | apps/victory/src/translate.rs:424 | direct |
| `VICTORY_FAIL_DUMP` | 未设置：None / 空（可选） [bare read] | `VICTORY_FAIL_DUMP=<dir>`: the first 3 containers of each failure reason, as the title has them (`.cdxil`), to work on a translator gap offline | apps/victory/src/translate.rs:453 | direct |
| `VICTORY_TEST_TITLES` | 未判定：经本地 helper 读取，默认值见 helper 实现 [helper-arg] | （注释缺失；语义需人工核对） | crates/victory-core/src/config.rs:82 | helper |
| `VICTORY_LOG` | 未设置：None（可选） [.ok()] | The most verbose level this build can log. | crates/victory-core/src/logging.rs:21 | direct |
| `VICTORY_DUMP_TARGETS` | 未设置：关闭（存在即开启） [is_some()] | With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png` (what each holds at t… | crates/victory-gpu/src/capture.rs:350 | direct |
| `VICTORY_BARRIER_TEST` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether an experiment on barriers is on (`VICTORY_BARRIER_TEST`): bit 0 for `groups` (the title's barrier groups record no wait for earlier shader writes), bit 1 for `indirect` (the pass that bounds an indirect command runs without its t… | crates/victory-gpu/src/copystat.rs:237 | direct |
| `VICTORY_NULL_MAGENTA` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | With `VICTORY_NULL_MAGENTA=1`, a 2D colour texture view that shows nothing yet shows magenta: on the screen, a texture that is not there is told from one whose contents are black. False if it is to be null. | crates/victory-gpu/src/descriptor/host.rs:2538 | direct |
| `VICTORY_SCISSOR_TEST` | 未设置：不等于该值（默认非该分支） [== Ok("...")] | An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`): they are all widened to the whole target. The title's room at the start of the prologue lost its walls in five runs of seven, and in those runs the player's b… | crates/victory-gpu/src/draw.rs:294 | direct |
| `VICTORY_UNWATCHED_FRAMES` | 未设置：1 [unwrap_or] | `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame). | crates/victory-gpu/src/draw.rs:1484 | direct |
| `VICTORY_TRAIL` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | The entries kept. | crates/victory-gpu/src/host.rs:313 | direct |
| `VICTORY_D3D12_DEBUG` | 未设置：关闭（存在即开启） [is_some()] | The debug layer only in builds with diagnostics. | crates/victory-gpu/src/host.rs:615 | direct |
| `VICTORY_D3D12_GBV` | 未设置：分支跳过（可选路径，不启用） [if let Ok/Some] | GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range descriptor heap indices and the like become messages naming the command instead of page faults. Without state tracking (directly indexed textures are not transitioned). | crates/victory-gpu/src/host.rs:626 | direct |
| `VICTORY_D3D12_BACKGROUND` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the driver compiling shaders again on the "D3D Background Thread"s, for minutes after pipelines were made or loaded) is switched off. An experiment: while those threads ran, th… | crates/victory-gpu/src/host.rs:671 | direct |
| `VICTORY_D3D12_DRED` | 未设置：关闭（存在即开启） [is_some()] | Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics builds). | crates/victory-gpu/src/host.rs:1443 | direct |
| `VICTORY_GPU_TIMING` | 未设置：关闭（存在即开启） [is_some()] | Whether each list of a submission runs and is timed on its own, slow ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics builds; the commands are noted with `VICTORY_D3D12_DRED` on). | crates/victory-gpu/src/host.rs:1458 | direct |
| `VICTORY_GPU_NULL` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | set behind them. A wait for a memory value that a flight's bottom-of-pipe write satisfies counts as met: the waiting part goes to the card behind that flight, which is the order the wait asks for. What it assumes beyond today: memory the… | crates/victory-gpu/src/host.rs:1982 | direct |
| `VICTORY_COPY_ASK` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | A title destination that is not all mapped is left alone. Asking Windows first (`VirtualQuery`) walks the page tables to the end of the destination's region: in the title's big blocks that is milliseconds a copy, and it was a seventh of … | crates/victory-gpu/src/host.rs:5744 | direct |
| `VICTORY_WAIT_ASK` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/list.rs:1915 | direct |
| `VICTORY_FAULT_TRACKING` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether dirty-page tracking (`MEM_WRITE_WATCH` and `PAGE_READONLY` guards) is active. Disabled by default under Wine on Darwin/Rosetta 2 where AVX masked stores across protected page boundaries abort translation (`VICTORY_FAULT_TRACKING=… | crates/victory-gpu/src/watch.rs:112 | direct |
| `VICTORY_ASK_ONCE` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | On by default since run 81 (with card buffers: 375-450 ms a frame against 573-647, the picture unchanged over four minutes); `VICTORY_ASK_ONCE=0` asks per range, as before. | crates/victory-gpu/src/watch.rs:305 | direct |
| `VICTORY_GUARD_ASK` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | A range whose first block the guards hold already (read-only by our hand, or noted absent) is a view: no question to Windows. The question walks the page tables to the end of the pages that share the first one's state, and a guarded view… | crates/victory-gpu/src/watch.rs:559 | direct |
| `VICTORY_KEYS` | 未设置：空字符串 / 零值 [unwrap_or_default] | Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for 0.2 s that long after input starts, with or without focus. A key is a letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`, `Left`, `Right` (the stand-in pad… | crates/victory-hle/src/modules/gameinput.rs:1208 | direct |
| `VICTORY_CLZERO_ASK` | 未设置：开启（=1 关闭） [is_none_or(v != "1")] | Whether what Windows said about memory is remembered per thread (the regions it found writable, the code bytes it found readable) and not asked again at every emulated instruction. A query walks the pages of the region it answers for: as… | crates/victory-loader/src/console_insn.rs:30 | direct |
| `VICTORY_STALL_DUMP_S` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Starts the diagnostics-only present watchdog. `0` disables it. On Wine, suspension is opt-in because the host can be stopped holding its VM lock. | crates/victory-loader/src/stacks.rs:31 | direct |
| `VICTORY_STACKS` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | Reads `VICTORY_STACKS` and, if set, starts a thread that prints every thread's stack at each listed time (seconds from now). | crates/victory-loader/src/stacks.rs:109 | direct |
| `VICTORY_TRAP_LOG` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | In a child process of the next test: calls an unimplemented import that traps, with the crash log where `VICTORY_TRAP_LOG` says. | crates/victory-loader/src/thunks.rs:160 | direct |
| `VICTORY_WRITE_LOG` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Thread id (low 32 bits), milliseconds since the first note (high 32). | crates/victory-loader/src/writelog.rs:66 | direct |

## memory

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_MEMORY_MAP` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether the address space is walked for the memory map line ([`log_memory_map`]). On by default (this module is in diagnostics builds only); `VICTORY_MEMORY_MAP=0` leaves the walk out. | crates/victory-gpu/src/copystat.rs:487 | direct |
| `VICTORY_ROOT_VIEW_MB` | 未设置：无上限（u64::MAX） [map_or(u64::MAX, ...)] | The host address of a root view: its buffer's host copy, or for a constant buffer not 256-byte aligned there (the console needs 16) a copy in upload memory. shortcut: a root view has no size; constant buffers are taken as 64 KiB (cut at … | crates/victory-gpu/src/draw.rs:464 | direct |
| `VICTORY_ROOT_UAV_MB` | 未设置：无上限（u64::MAX） [map_or(u64::MAX, ...)] | must be dropped. shortcut: a root view has no size: the view, the refresh and the copy back cover all that is left of the title's buffer (`VICTORY_ROOT_UAV_MB` sets a limit for all three). | crates/victory-gpu/src/draw.rs:1180 | direct |
| `VICTORY_MAC_RAM_BUDGET_MB` | 未设置：32768 [unwrap_or] | Room for discretionary caches, not a limit on the title's own allocations. Don't rely on DXGI's local-memory budget on a unified-memory Mac. | crates/victory-gpu/src/host.rs:135 | direct |
| `VICTORY_SCRATCH_POOL` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Whether lists share their scratch chunks: a list takes a chunk when it first needs one, a small one first, and gives all of them back when it is reset or dropped, for the next list. Before, a list kept every chunk it ever took (4 MiB eac… | crates/victory-gpu/src/host.rs:1551 | direct |
| `VICTORY_INPLACE_PROBE` | 未设置：0 [unwrap_or] | `VICTORY_INPLACE_PROBE`: 0 (unset) nothing; 1 the probe on memory of its own at the first sweep, and the census 30 and 150 seconds later; 2 also opens one part of the title's own memory for a moment at the first census. | crates/victory-gpu/src/inplace.rs:63 | direct |
| `VICTORY_INPLACE` | 未设置：0 [unwrap_or] | the title maps pages. What it assumes, for 2: nothing reads title memory that a submission not yet run writes through such a buffer. With every submission waited for (the default) that holds; with `VICTORY_LATE_WAIT` it does not, since t… | crates/victory-gpu/src/inplace.rs:90 | direct |
| `VICTORY_INPLACE_MB` | 未设置：4096 [unwrap_or] | The most title memory in place at once, in bytes (`VICTORY_INPLACE_MB`, 4096): opening locks the pages into the PC's memory, also those the title never touched. | crates/victory-gpu/src/inplace.rs:98 | direct |
| `VICTORY_INPLACE_PROBE_MB` | 未设置：64usize [unwrap_or] | （注释缺失；语义需人工核对） | crates/victory-gpu/src/inplace.rs:1080 | direct |
| `VICTORY_CARD_BUFFERS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | before, and the ranges that changed in it are copied to the card's side before each submission (`Host::card_before`). What shaders wrote is found on the card: after the submission's lists a compute pass of ours compares each declared ran… | crates/victory-gpu/src/shadow.rs:156 | direct |
| `VICTORY_CARD_BUFFERS_MB` | 未设置：1024（闭包形参默认值） [closure-default] | From 1 MiB up and 1,024 MB in all by default (it was 16 and 512): the shader writes of the smaller buffers are then found by the card's hashes too, and with that the first room's walls were drawn in every run. | crates/victory-gpu/src/shadow.rs:175 | closure |
| `VICTORY_CARD_BUFFERS_MIN_MB` | 未设置：1（闭包形参默认值） [closure-default] | From 1 MiB up and 1,024 MB in all by default (it was 16 and 512): the shader writes of the smaller buffers are then found by the card's hashes too, and with that the first room's walls were drawn in every run. | crates/victory-gpu/src/shadow.rs:175 | closure |
| `VICTORY_CARD_WINDOWS_MB` | 未设置：0 [unwrap_or] | Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of windows"), so a window on the card needs only what a card copy has already: the CPU's side takes title memory as before, and what changed in it goes up before each submi… | crates/victory-gpu/src/shadow.rs:269 | direct |
| `VICTORY_KEEP_SNAPSHOTS` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which 100 MB changed. Kept, a piece of it is taken again only after something changed the copy there: a refresh from title memory ([`Shadow::copy`]), shader writes found after… | crates/victory-gpu/src/shadow.rs:1898 | direct |
| `VICTORY_KEEP_SNAPSHOTS_MB` | 未设置：256 [unwrap_or] | `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with kept snapshots beside copies in the card's memory, ran the PC out of memory in 27 s; kept snapshots are the PC's memory, all of it resident). | crates/victory-gpu/src/shadow.rs:1909 | direct |
| `VICTORY_COMMITTED_QUERY` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | Whether the two questions above and below are put to `VirtualQuery`, as they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the length of the run of pages in the same state, which it finds by walking the page tables to t… | crates/victory-gpu/src/texture.rs:1089 | direct |
| `VICTORY_MAPPED_BY_PAGE` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | one not there. With neither the first nor the last page committed the answer is none without a look at the pages between (two queries for the slots whose texture is away, which are asked again and again). | crates/victory-gpu/src/texture.rs:1174 | direct |
| `VICTORY_XMEM_PERSISTENT` | 未设置：开启（=0 关闭） [!is_some_and(v == "0")] | SAFETY: our handle; title views keep the section alive on their own. | crates/victory-hle/src/modules/xmem.rs:380 | direct |

## loader

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_SAVE_STOP` | 未设置：走 else 分支（提前返回，功能不启用） [let Ok/Some else] | In a child process of the crash test, stops for good at the step `VICTORY_SAVE_STOP` names and says so. | crates/victory-hle/src/modules/xgamesave.rs:960 | direct |
| `VICTORY_SAVE_CHILD` | 未设置：走 else 分支（提前返回，功能不启用） [tuple else] | Run by the crash test in a child process; does nothing otherwise. | crates/victory-hle/src/modules/xgamesave.rs:1012 | direct |
| `VICTORY_SAVE_OP` | 未设置：走 else 分支（提前返回，功能不启用） [tuple else] | Run by the crash test in a child process; does nothing otherwise. | crates/victory-hle/src/modules/xgamesave.rs:1012 | direct |
| `VICTORY_INSN_PATCH` | 未设置：开启（true） [map_or(true, ...)] | Padding runs taken (start, end). | crates/victory-loader/src/console_patch.rs:259 | direct |
| `VICTORY_EXIT_CHILD` | 未设置：为真（变量存在则为假） [is_none()] | In a child process of the next test: ends the other threads but one kept thread, and checks that a busy thread stopped and the kept one did not. | crates/victory-loader/src/threads.rs:407 | direct |

## formats

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_BCPACK_TABLES` | 未设置：报错退出（仅测试/调试路径） [unwrap/expect] | Compares with reference outputs made outside the repository: `VICTORY_BCPACK_TABLES` (the table file) and `VICTORY_BCPACK_CASES`, a directory of `<name>.bc<n>` packed chunks with `<name>.out` beside them. | crates/victory-formats/src/bcpack.rs:1576 | direct |
| `VICTORY_BCPACK_CASES` | 未设置：报错退出（仅测试/调试路径） [unwrap/expect] | Compares with reference outputs made outside the repository: `VICTORY_BCPACK_TABLES` (the table file) and `VICTORY_BCPACK_CASES`, a directory of `<name>.bc<n>` packed chunks with `<name>.out` beside them. | crates/victory-formats/src/bcpack.rs:1577 | direct |

## config

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_CONFIG` | 未设置：None（可选路径） [map(PathBuf::from)] | Loads configuration from the environment and the config file. | crates/victory-core/src/config.rs:49 | direct |
| `VICTORY_CACHE_DIR` | 未判定：经本地 helper 读取，默认值见 helper 实现 [helper-arg] | （注释缺失；语义需人工核对） | crates/victory-core/src/config.rs:88 | helper |
| `VICTORY_DATA_DIR` | 未判定：经本地 helper 读取，默认值见 helper 实现 [helper-arg] | （注释缺失；语义需人工核对） | crates/victory-core/src/config.rs:91 | helper |
| `VICTORY_GAME_DIR` | 未判定：经本地 helper 读取，默认值见 helper 实现 [helper-arg] | （注释缺失；语义需人工核对） | crates/victory-core/src/config.rs:95 | helper |
| `VICTORY_METAL_CACHE` | 未设置：None（可选路径，留空等同未设置） [filter/map optional path] | Where Metal Shader Converter's output for a shader and root signature is kept between runs: `VICTORY_METAL_CACHE`, else `<VICTORY_CACHE_DIR>/metal`. The name is a hash of the DXIL, the signature and the converter version. | crates/victory-gpu/src/backend/metal.rs:1063 | direct |
| `VICTORY_DISPLAY_MODE` | 未设置：None（可选） [.ok()] | The size of the display the console reports: `VICTORY_DISPLAY_MODE=WxH` if set (as a console reports the television it is plugged into: a Series X on a 1080p set lists a 1080p mode), else 3840x2160. shortcut: not yet the host display's o… | crates/victory-gpu/src/dxgi.rs:230 | direct |

## host-gpu

| name | 未设时默认值 | 作用 | file:line | 来源 |
|---|---|---|---|---|
| `VICTORY_SWEEP` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`). | crates/victory-gpu/src/host.rs:243 | direct |
| `VICTORY_ACTIVATE_RAYS` | 未判定：走 match 默认分支，需人工核对 [match/as_deref] | Whether a shader is left out for what it costs the host's driver: `CS_ScreenProbes_ActivateRays` of the title (39f0e9aeb156a61c, 66,501 instructions, 9,948 branches). As translated before the clean-up passes (`emit::shader_opt`), AMD's t… | crates/victory-gpu/src/pipeline/make.rs:70 | direct |
| `VICTORY_WARM_WINDOWS` | 未设置：开启（=0 关闭） [is_none_or(v != "0")] | Held from the look to the removal: `loose` hands a copy out under the same lock, so none is handed out in between. | crates/victory-gpu/src/shadow.rs:2041 | direct |
| `VICTORY_WINDOW_KB` | 未设置：None（可选） [.ok()] | The grid the windows of asks with a length are cut on, in bytes: 4 MiB unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least). A window is the PC's memory, all of it, however little of it is read: 803 windows held 3.3 GB i… | crates/victory-gpu/src/shadow.rs:3726 | direct |
| `VICTORY_WINDOW_TAILS` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | address ever asked for in the cell is memory the host zeroed and nothing reads (513 of run 65's 782 windows were asked for that way, 2,052 MB; of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the window is the cell's t… | crates/victory-gpu/src/shadow.rs:3765 | direct |
| `VICTORY_RANGE_REMAP` | 未设置：关闭（=1 开启） [is_some/ok_and(v == "1")] | never gave up are "reserved, never committed", and a thread reading them in place takes an access violation: run 71 in `Layout::detile`, run 74 in the hash before it, both under `Staging::refresh_now`, which had asked whether the texture… | crates/victory-gpu/src/watch.rs:886 | direct |
| `VICTORY_DSTORAGE_RETRIES` | 未设置：12 [unwrap_or] | A page the title has just made, or a view the host refuses for a moment (commit or address space short while a load maps many at once), can come right a little later: tried again before the request fails (`VICTORY_DSTORAGE_RETRIES`, defa… | crates/victory-hle/src/modules/dstorage.rs:1031 | direct |

## T8 测试脚本自身变量（不属于 victory-fork 运行时开关）

以下变量来自 `metal-backend/T8/run-t8.sh`：它们是测试脚本的内部变量、导出给 Wine/游戏进程的环境变量，或脚本引用的外部输入。**归属测试脚本**，与上表源码开关分开登记。

| name | 归属 | 取值 / 默认 | 备注 |
|---|---|---|---|
| `METALVI_INSERT_DYLIB` | exported | "$project/victory-fork/scripts/macos/metal_concurrent_compile.dylib" | — |
| `RUST_LOG` | exported | info | — |
| `VICTORY_CACHE_DIR` | exported | "$(to_z " | — |
| `VICTORY_DATA_DIR` | exported | "$(to_z " | — |
| `VICTORY_GPU_NULL` | exported | "${GPU_NULL:-1}" | — |
| `VICTORY_KEYS` | exported | "${KEYS:-10:Enter,18:Enter,26:Enter,34:Enter,42:Enter,53:Enter,64:Enter,75:Enter,87:Enter,100:Enter,115:Enter,130:Enter}" | — |
| `VICTORY_MAC_RAM_BUDGET_MB` | exported | 32768 | — |
| `VICTORY_METAL_CACHE` | exported | "$(to_z " | MSC output, shared by runs |
| `VICTORY_NO_HANG_WATCHDOG` | exported | "${NO_WATCHDOG:-1}" | the title kills itself when frames are slow |
| `VICTORY_RANGE_REMAP` | exported | 0 | — |
| `VICTORY_XMEM_PERSISTENT` | exported | 0 | — |
| `WINEDLLOVERRIDES` | exported | "mscoree,mshtml=;d3dcompiler_47=n,b${MUTE_AUDIO:+;winecoreaudio.drv=}${HIDDEN:+;winemac.drv=d}" | — |
| `WINEPREFIX` | exported | "$state/prefix" | — |
| `WINESERVER` | exported | "$WINE_ROOT/bin/wineserver" | — |
| `WINE_ROOT` | exported | "${WINE_ROOT_OVERRIDE:-$project/runtime-env/wine-gamesir}" | — |
| `MUTE_AUDIO` | internal | "${MUTE_AUDIO-}" | MUTE_AUDIO=1 disables the audio driver: the game then stalls in its menus (it waits for audio to play); mute the Mac instead |
| `WINEMSYNC` | internal | 1 | — |
| `here` | internal | "$project/metal-backend/T8" | — |
| `label` | internal | "${1:?label}" | — |
| `project` | internal | "$HOME/Documents/GTA VI" | — |
| `secs` | internal | "${2:-330}" | — |
| `seed` | internal | "$project/runtime-env/matched-view-index.eitsSX" | — |
| `state` | internal | "$here/run-$label" | — |
| `EXE` | referenced | $here/target/x86_64-pc-windows-gnullvm/diag/victory.exe | 脚本引用的外部/调用方环境变量 |
| `GPU_NULL` | referenced | 1 | 脚本引用的外部/调用方环境变量 |
| `KEYS` | referenced | 10:Enter,18:Enter,26:Enter,34:Enter,42:Enter,53:Enter,64:Enter,75:Enter,87:Enter,100:Enter,115:Enter,130:Enter | 脚本引用的外部/调用方环境变量 |
| `MODE` | referenced | cpu | 脚本引用的外部/调用方环境变量 |
| `MSYNC` | referenced | — | 脚本引用的外部/调用方环境变量 |
| `NO_WATCHDOG` | referenced | 1 | 脚本引用的外部/调用方环境变量 |
| `SAMPLE_AT` | referenced | 100 140 180 220 260 | 脚本引用的外部/调用方环境变量 |
| `WINE_ROOT_OVERRIDE` | referenced | $project/runtime-env/wine-gamesir | 脚本引用的外部/调用方环境变量 |

## 附录：扫描时跳过的非 VICTORY_ 项

以下调用未列入上表（非 `VICTORY_` 前缀或非常量字面量）：

| 字面量 | file:line | 原因 |
|---|---|---|
| `<non-literal>` | crates/victory-core/src/config.rs:73 | 非 VICTORY_ 前缀或非常量字面量 |
| `LOCALAPPDATA` | crates/victory-core/src/config.rs:131 | 非 VICTORY_ 前缀或非常量字面量 |
| `XDG_CACHE_HOME` | crates/victory-core/src/config.rs:132 | 非 VICTORY_ 前缀或非常量字面量 |
| `HOME` | crates/victory-core/src/config.rs:134 | 非 VICTORY_ 前缀或非常量字面量 |
| `RUST_LOG` | crates/victory-core/src/logging.rs:22 | 非 VICTORY_ 前缀或非常量字面量 |
| `F59_DXIL` | crates/victory-gpu/src/backend/metal_texture/descriptor_coverage.rs:197 | 非 VICTORY_ 前缀或非常量字面量 |
| `F59_CASE` | crates/victory-gpu/src/backend/metal_texture/descriptor_coverage.rs:198 | 非 VICTORY_ 前缀或非常量字面量 |
| `F59_STRICT` | crates/victory-gpu/src/backend/metal_texture/descriptor_coverage.rs:203 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-gpu/src/draw.rs:92 | 非 VICTORY_ 前缀或非常量字面量 |
| `1` | crates/victory-gpu/src/pipeline/make.rs:86 | 非 VICTORY_ 前缀或非常量字面量 |
| `set {name} to the compiled DXIL` | crates/victory-gpu/src/primitive/d3d12_test.rs:14 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-gpu/src/shader_cache.rs:92 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-gpu/src/shadow.rs:169 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-gpu/src/watch.rs:144 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-hle/src/modules/audio/capture.rs:84 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-hle/src/modules/lifecycle.rs:272 | 非 VICTORY_ 前缀或非常量字面量 |
| `F42_SHADER_DIR` | crates/victory-metal/src/indirect/tests.rs:491 | 非 VICTORY_ 前缀或非常量字面量 |
| `<non-literal>` | crates/victory-shader/src/emit/mod.rs:1117 | 非 VICTORY_ 前缀或非常量字面量 |
