# MetalVI 原生 Metal 后端 v2：执行架构设计

> **状态：设计文档，尚未实现。** 2026-11-27 制定。第 1 节描述的是当前代码，它仍是旧的执行模型：同步提交、单队列。v2 的实现按阶段推进，阶段划分见 README 的路线图。性能数字都是预测，不是实测。

[English](metal-architecture-v2.md) · **简体中文**

## 1. 为什么要修订：当前架构的天花板（代码已核实）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| P1 | **CPU 与 GPU 锁步** | [`metal.rs` `run()`](../crates/victory-gpu/src/backend/metal.rs) 持全局 `SUBMISSION` 锁，提交后等待完成，注释原文 "the title's fences may be signalled after this returns"；`submit_checked` 另持一把锁，逐个命令缓冲 `commit → wait`；描述符 ring 能每次从头复用，前提是"上一次已等完"；`queue.rs` `run_metal` 写明 "waited for before returning"。 | 帧时间 ≈ CPU + GPU + 每次提交的唤醒延迟，而不是两者取大。S1 的安全措施就是靠等待实现的，把锁步固化了。 |
| P2 | **跨队列同步全靠 CPU 模拟** | `queue.rs`：列表在"等待内存值"处被切开（`cuts`），由 CPU 按条件挂起或放行（`Held`、`release_held`）；bottom-of-pipe 写由 CPU 在列表跑完后执行（`Written::write`）；present 强制放行所有挂起项（`forces`）；连 GPU 时间戳也是 CPU 时钟（`Written::GpuTimestamp`）。Metal 只有一个 `queue`。 | 主机的图形队列和异步计算队列被压成"CPU 调度 + 单条 GPU 队列"，没有任何重叠。任何跨队列依赖都要先回到 CPU。 |
| P3 | **GPU 端过度串行** | `victory-metal/src/lib.rs` 以 `concurrent=false` 创建计算编码器；分发之间一律插 barrier；拆开的命令缓冲之间用 `SharedEvent` 串联。 | 游戏有大量 1 个线程组（64 线程）的小分发，串行执行时 GPU 大部分时间闲置。 |
| P4 | **没有考虑 TBDR** | 清除单独成一个 render pass；load/store 基本固定为 Load/Store；没有 memoryless，也没有 pass 规划。 | Apple GPU 是分块延迟渲染。每断一次 pass，整张渲染目标都要存一遍、再读一遍。对移植过来的立即模式引擎，这往往是最大的带宽开销。 |
| P5 | **编码单线程、锁多、跑在 Rosetta 下** | 所有 `MetalOp` 都在执行时、在提交线程上、在全局锁里编码；`MetalBackend` 的状态分散在十几个 `Mutex<HashMap>` 里；整个后端是 x86-64 代码。 | 游戏在多个 job 线程上录制列表（内部线程等待测量），我们的编码却串成一条线，而且每条指令都要付 Rosetta 的翻译开销。 |
| P6 | **纹理仍是拷贝模型** | `texture.rs` 的 `refresh()` 在每次执行前做检查（含哈希）；纹理是 detile 之后的私有副本；现有 detile 器不支持渲染目标和深度用的 XOR 模式。 | 缓冲区已经零拷贝，纹理没有，零拷贝的收益只拿到一半。 |
| P7 | **着色器的安全插桩一直开着** | `native/victorymetal/src/unix/msc_unix.c` 对所有管线开启 `IRCompatibilityFlagBoundsCheck`；`victory-shader` 的 `lower.rs` 在每个循环里插 `victory.loop_budget` 计数。 | 所有着色器都要付这份 GPU 开销，循环预算还会改变语义。 |
| P8 | **没用二进制归档** | `sys.rs` 里已有 `binary_archive` 字段，后端从未使用。 | 冷编译时进 3D 卡了 126–154 秒。 |
| P9 | **计划没有性能目标** | v1 的验收写"降为目标值"，但从没定过数值；没有帧预算，没提 Series S；性能测量工具排到阶段 7。 | 路线判断没有依据。在锁步模型上和 D3DMetal 比，结论会偏向不利。 |

**仍然成立的部分**：缓冲区零拷贝和别名（`chunks.rs`、`physical_alias`）、MSC 管线转换、描述符表编码（`ir.rs`）、Wine 到 Metal 的桥、纹理 detile 内核、安全关卡和取证纪律。这些是新架构的地基，不推倒重来。

---

## 2. 目标与性能预算

### 2.1 目标
在目标机器和场景上，得到画面、游戏状态和渲染工作都正确且稳定的结果，**并且架构本身不给硬件利用率设上限**：CPU 和 GPU 重叠执行，多个队列并行，GPU 内部不做多余的串行化，渲染目标的带宽按 TBDR 优化，编码可以并行。25–30 fps、连续运行 30 分钟是远期目标，不作为承诺；能否达到最终取决于游戏 CPU 路径（Rosetta + Wine）。

### 2.2 帧预算（30 fps = 33.3 ms，预算按架构目标设定，需实测校准）

| 部分 | 预算 | 说明 |
|---|---|---|
| 游戏 CPU 关键路径 | ≤ 33 ms | 不受后端控制。关闭 GPU 的纯 CPU 测量为 46–52 ms（msync），过场动画在停顿之间达到 33 ms。这是最终上限。 |
| 后端编码（CPU） | 每帧 ≤ 6 ms 总 CPU，**不在游戏关键路径上** | 放到编码线程或线程池，与游戏并行。 |
| 提交到 GPU 的延迟 | 被流水线隐藏 | 最多允许 2 帧在途。 |
| GPU | ≤ 30 ms（Series S 配置） | 按 pass 计时。 |
| 稳态长帧 | 缓存热时，单帧不超过 100 ms | 编译和预热的卡顿单独统计。 |

### 2.3 主机配置
此前的上限分析结论是 M1 Max 能覆盖 Series S，但覆盖不了 Series X。**性能测试默认用 `--console series-s`**（`Console::SeriesS`，像素量约为 Series X 的 1/2.25）。和历史截图对比时用 series-x，两种配置的数据不混用。

---

## 3. 目标架构

```
 游戏线程（多个）            后端                                   GPU
 ─────────────            ──────────────────────────────         ───────────────
 列表方法调用 ──► 列表 IR（MetalOp 流，按列表、按切分段）
 ExecuteCommandLists ──► 队列时间线（A）──► 编码线程池（B）
                           │  fence/标签 → MTLSharedEvent       ┌─ 图形 MTLCommandQueue
                           │  在途预算（H）                     ├─ 计算 MTLCommandQueue
                           ▼                                    │
                     Pass 规划（C）+ 屏障翻译（D）──► 命令缓冲 ─┤  （event 排序）
                     资源（E）+ 绑定（F）+ 管线（G）            │
 完成线程（每个队列一个）◄──────────── 完成 / GPU 时间 ────────────┘
   ├ bottom-of-pipe 写、fence 推进、资源回收、ring 释放
   └ 时长检测 → 熔断（H）
 可观测性（I）贯穿各层；整个后端以命令流为接口（J），可以整体迁走
```

下面每个模块按"现状 → 设计 → 风险/验收"写。

### A. 队列时间线与异步提交（解决 P1、P2）

**现状**：同步等待；跨队列依赖由 CPU 挂起和放行；bottom-of-pipe 写在等待之后由 CPU 执行。

**设计**
1. **队列映射**：游戏的每个队列（图形、计算）各对应一个 `MTLCommandQueue`，共享同一个驻留集，并加入每个队列。主机队列的种类和数量以 `queue.rs` 登记的为准。
2. **fence**：每个游戏 fence 对应一个 `MTLSharedEvent`，fence 值就是 event 值。`Signal` 在命令缓冲末尾 `encodeSignalEvent`；`Wait` 在下一个命令缓冲开头 `encodeWaitForEvent`。
3. **内存标签等待**（列表中途"等待某地址达到某值"）：
   - 编码时能确定生产者的，转成 GPU 端等待。生产者是另一队列某段末尾的 bottom-of-pipe 写，写入值单调递增，比较方式是 `>=`。该写入映射为"标签 event"的 signal，等待方 `encodeWaitForEvent`。
   - 非单调值、带掩码或 `==` 比较、找不到生产者的情况，保留现有的 CPU 挂起机制（`Held`），并计数上报。
   - **安全规则：GPU 只等待"产生它的 signal 已经入队"的值。** 否则在 CPU 端挂起。这样不会出现 GPU 队列永远等不到 signal 的情况。
4. **完成线程**：每个队列一个专用线程，按顺序等待已提交的命令缓冲（`waitUntilCompleted`，或 `MTLSharedEventListener`）。命令缓冲完成后，它负责：
   - 执行该段的 bottom-of-pipe 写（保留现有 `Written::write` 的语义和 `note_write` 记录）；
   - 推进 fence 的 CPU 可见值，唤醒等在 `SetEventOnCompletion` 上的游戏线程；
   - 回收临时资源和 ring 区段；
   - 读取 `GPUStartTime`/`GPUEndTime` 交给安全模块。
   - 提交线程从此不再等待 GPU。
   - 不在 Unix 侧回调里调用 PE 代码，规避 Wine 跨边界回调的问题。
5. **GPU 时间戳**：`Written::GpuTimestamp` 改为基于命令缓冲的 GPU 时间，或保持 CPU 时钟但在完成时写入。要先确认游戏是否用时间戳差值做动态分辨率或帧节奏。
6. **ring 与临时内存**：一次性描述符表、参数区、上传暂存区都改成按提交分配的环形区段，由完成线程释放。区段用完时阻塞编码（这本身就是背压），不能覆盖仍在途的区段。
7. **present**：交换链的呈现放进图形队列时间线，在命令缓冲里 `presentDrawable`（或保留现有呈现路径，但改为由 event 驱动，不再同步等待）。帧节奏由游戏的 fence 和在途上限控制。

**风险/验收**
- 在合成测试中证明三件事：两个队列加 fence 的执行顺序正确；标签 event 的生产者和消费者正确；CPU 挂起的回退路径正确。故障注入至少覆盖：event 创建失败、完成线程退出、ring 用尽。
- 对照现有 CPU 调度：同一组合成列表，两条路径的执行顺序和 bottom-of-pipe 写序列一致。
- `VICTORY_METAL_SYNC=1` 保留旧的逐个等待模式，只作调试和对照用。

### B. 录制与编码（解决 P5）

**设计**
1. **列表 IR**：沿用 `MetalOp` 流，每个列表、每个切分段一份。录制阶段只记录游戏状态和参数，不调用 Metal。
2. **编码时机**：D3D12 语义下，描述符堆的内容到执行时才确定，所以编码仍然在 `ExecuteCommandLists` 之后，但改为**交给编码线程池**：一个列表段对应一个 `MTLCommandBuffer`，各段并行编码；先按提交顺序 `enqueue` 预定顺序，编码完再 `commit`。游戏线程只负责入队，立即返回。
3. **去锁**：
   - 管线、采样器、视图缓存改为"预计算键 + 读多写少"的结构（分片锁或 RCU）。
   - 编码线程各自持有本地的 ring 区段和参数写指针。
   - 跨线程只共享不可变对象（`Arc`）。
   - 不允许有全局"提交锁"包住编码。
4. **大批量绘制**：同一管线、同一渲染目标、只有参数不同的长串绘制，后续可以评估用 ICB（Indirect Command Buffer），由 GPU 侧生成。这是性能阶段的优化项，不是起点。
5. **Rosetta**：后端热路径的 CPU 时间要单独测量（见 I）。超出预算时的升级方向见 J。

**验收**：编码耗时的分布、编码线程数、锁等待时间可以统计；多线程编码与单线程编码对同一组合成帧产生相同的 GPU 结果。

### C. Pass 规划层（TBDR，解决 P4）

**设计**：在编码之前，对一个列表段的 op 流做一次规划。
1. **合并清除**：清除后紧跟同一目标的绘制时，把清除并进该 pass 的 `loadAction=Clear`。只有确实需要单独清除时（例如部分矩形清除），才保留清除 pass。
2. **load 动作**：首次写入的目标，在能证明整张会被覆盖时用 `DontCare`。主机的快速清除元数据（CMASK/DCC/HTILE 一类）和清除消除 pass 要先识别，不能误判为"整张覆盖"。
3. **store 动作**：后续不会再读的附件用 `DontCare`，例如游戏 `DiscardResource` 的对象，或段内可证明不再被读的深度。MSAA 用 resolve store action，不另起解析 pass。
4. **memoryless**：只在同一 pass 内产生并消耗的临时深度或 MSAA 附件，才用 `MTLStorageModeMemoryless`。必须能在当前段内证明之后不再读；不能证明就保守存储。
5. **不断 pass**：不涉及当前附件的 barrier 和资源状态转换，不打断 render pass。
6. **重排**（性能阶段再做）：夹在同一目标两组绘制之间的计算分发，只有在资源区间分析证明它与这些绘制互不读写时，才允许挪出去以合并 pass。
7. **tile 能力**（升级项）：POPS 已映射到光栅顺序组（ROV），保留。延迟光照、OIT 等后续可以评估 imageblock 或 tile shader。

**验收**：每帧报告 render pass 数、各附件的 load/store 分布、memoryless 字节数、因清除或 barrier 断开的 pass 数。合成测试覆盖：清除合并、DontCare 判定、resolve、深度丢弃。规划前后的输出逐像素一致。

### D. 同步与屏障（解决 P3）

**设计**
1. **计算编码器**一律用 concurrent 分派类型。
2. **只翻译游戏自己的 barrier**：列表 API 的 `ResourceBarrier`（UAV、状态转换、别名）、X 扩展的等待和缓存刷新，转成 `memoryBarrierWithScope`（buffers / textures / render targets），或针对具体资源的 barrier。没有 barrier 的相邻分发允许并发执行。
3. **同一队列内跨编码器、跨命令缓冲**：通过 GPU 地址访问的无追踪内存，用 `MTLFence`（编码器之间）或非共享的 `MTLEvent`（命令缓冲之间）。`MTLSharedEvent` 只用于跨队列和 CPU 可见的 fence。
4. **保守开关**：`VICTORY_METAL_SERIAL=1` 恢复"每个分发之间都插 barrier"，用于二分定位同步错误。

**验收**：合成测试覆盖：无 barrier 时并发执行，有 UAV barrier 时顺序正确；跨编码器的 fence 正确；读后写、写后读、写后写三类危险各一项。在 API Validation 下运行通过。

### E. 内存与资源（解决 P6）

**缓冲区**：保持 noCopy span 加持久 section 视图别名的方案（已修复 span 爆炸）。约束不变：**不在活跃的驻留集里给重映射后的页重新建包装**（这曾导致内核 panic）。

**纹理所有权模型**（替代每次执行前的 `refresh()` 哈希检查）：
1. **线性纹理**：用建在 noCopy 缓冲上的 buffer-backed texture，真正零拷贝。需要满足 Metal 的线性对齐和行距要求，不满足的退回到副本。
2. **CPU 写入的分块纹理**：保留 Metal 私有副本。脏信息来源按优先级：
   - 已知写入者直接通知：ReadFile、DStorage、xmem 的写，已有 writelog 和 `note_write` 钩子；
   - 页级写监视（只看页，不做哈希）；
   - 只在前两者都无法覆盖的地方保留哈希，并计数。
   - 只对脏子资源或脏 tile 做 GPU detile。
3. **GPU 写入的纹理**（渲染目标、UAV）：所有权归 GPU，内容只存在于 Metal 纹理里。只有证实 CPU 会读，或者被别名为其他格式、被拷贝到 CPU 可见内存时，才写回。写回走 GPU 编码（tile 内核加 blit），由完成线程收尾。CPU 读的检测方案（例如读保护）必须先在独立的小测试里验证，不能直接在游戏里试。
4. **XOR 模式**：渲染目标和深度只要不需要回到 CPU，就不必 detile。需要写回或给 CPU 读时，再补 XOR 的 detile/tile 内核，并用 detile 测试的方法逐字节对照。
5. **别名**：多个放置资源共用同一段游戏内存时，按"地址区间 → 当前有效的 Metal 资源"建索引。新资源写入时，让重叠的旧资源失效。同格式、同布局的别名可以考虑放进 `MTLHeap` 共享存储；不同格式之间的别名语义本身就是未定义的，不做保证。
6. **压缩纹理**（BC）：按需暂存，再 blit。

**验收**：每帧报告 detile 字节数、写回字节数、哈希字节数（目标是接近 0）、副本内存总量。合成测试覆盖：CPU 写后 GPU 读、GPU 写后 CPU 读、别名失效、线性零拷贝。

### F. 绑定（独立准备路径的核心）

**设计**
1. **描述符堆**：一个常驻的 `MTLBuffer`，存放 `IRDescriptorTableEntry`。在 `CreateShaderResourceView` 等调用时直接写入表项，取代"执行时批量同步脏槽"（`sync_heap` 单次曾达 1.35 秒）。表项指向 noCopy span 的 GPU 地址。为此维护"span → 引用它的表项"的反向索引：span 因重映射或别名变化时，重写受影响的表项，并在下一次提交前生效。
2. **根参数**：常量、根 CBV/SRV/UAV、一次性描述符表，从游戏状态直接生成，写进按提交分配的 ring 区段（见 A6）。复用 `ir.rs` 的编码和现有的根签名规则。
3. **空描述符**用共享的 1×1 零纹理和零缓冲（已有 `null_texture`）。
4. **采样器**：沿用按 Metal 描述去重的做法（Apple7 超过 1024 个不同采样器会采样出错）。
5. **不依赖 D3D12**：独立模式下不调用 `draw.rs::prepare_for`、`bind`、`refresh_slot`，也不调用 `shadow::buffer_extent` 一类影子机制。缓冲区的边界改从资源表获取。

**验收**：同一合成样本在新旧路径下得到相同的绑定和输出。计数证明独立模式下 D3D12 准备函数调用次数为 0；不支持的绑定形式逐项报告，不静默跳过。

### G. 着色器与管线（解决 P7、P8）

1. **二进制归档（提前到阶段 2）**：
   - 离线用 `victory warm --metal` 生成 MSC 产物，再收集成 `MTLBinaryArchive`；
   - 运行时以 archive 优先，性能测试中开启 `fail_on_binary_archive_miss` 并计数未命中；
   - archive 的键包含 MSC 版本、翻译器版本和设备系列。
2. **插桩分级**：
   - **U 级（未验证）**：开启 `BoundsCheck` 和 `loop_budget`；
   - **V 级（已验证）**：两者都关闭。验证依据：离线重放没有越界，循环有可证明的上界，或者在受控运行中没有触发预算。
   - 两种版本都进 archive。性能报告给出 U/V 两级的占比和 GPU 时间。
3. **NGG / 网格**：对能证明每线程对应一个顶点、没有跨 lane 图元组装的着色器，生成顶点着色器路径（手动读取顶点）。这样避开 M1 上 mesh shader 偏慢的问题和 32 KiB 输出上限。只在必须时才用 mesh；超出上限时，再按 v1 的思路评估属性缓冲方案。
4. **wave**：保持主机最大 wave 为 32。WaveSize64 的跨半区操作逐个计数，评估改写。
5. **效率测量**：按管线统计 GPU 时间（见 I），与 DXIL 指令数对照，找出转换开销最大的着色器。此前估计比原生慢 1.5–3 倍，但从未测过。
6. **升级路径**：转换开销确认是瓶颈时，再评估直接生成 AIR/MSL（绕过 DXIL → MSC），或针对 Apple GPU 的翻译优化（减少 EXEC 守卫、原生 simdgroup 32）。

### H. 安全模型 v2（替代"逐个等待 + 事后熔断"）

全程安全规则见第 4 节。机制分四层：
1. **编码时限定工作量**：
   - 每个命令缓冲有工作量上限：分发的线程组总数、绘制的图元数，用上一次测得的每管线 GPU 时间估算出的时长；
   - 直接参数在编码时检查；
   - 间接参数在使用前由一个小计算内核钳位到上限，并记录被钳位的次数（钳位是安全措施，不算正确性通过）；
   - U 级着色器保留循环预算。
2. **在途预算**：每个队列的在途命令缓冲数和估算 GPU 时间都有上限（例如 ≤ 2 个，或估算 ≤ 50 ms）。超过上限时阻塞编码线程，不阻塞游戏线程。
3. **异步检测**：完成线程检查实际 GPU 时长、错误状态和超时（包括系统报告的 `ImpactingInteractivity` 一类错误）。超过阈值时停止所有新提交。被隔离的管线必须记录和报告，**不能作为正确性或性能的"通过"**。
4. **兜底**：现有的进程级熔断、`check-intro.sh` 的原子单实例锁和挂起停止保留。逐个等待模式（`VICTORY_METAL_SYNC=1`）只作调试。

限制必须写明：命令缓冲一旦开始执行，就不能被打断；观察到的最大值不等于所有输入的上界；跨进程、其他 GPU 用户不在控制范围内。

### I. 可观测性（提前到阶段 2）

- 按 pass 计 GPU 时间：`MTLCounterSampleBuffer`，在 stage boundary 采样（M1 支持）。另有命令缓冲的 `GPUStartTime`/`GPUEndTime`。
- 每帧指标：
  - 墙钟帧时间（p50/p95/p99/最大）；
  - CPU：游戏关键路径时间、编码时间、锁等待；
  - GPU：每队列忙碌时间、在途深度；
  - pass 数、load/store 分布；
  - detile、写回、哈希字节数；
  - archive 未命中数、U/V 级占比；
  - 被钳位、隔离、跳过的工作数。
- 配合 Metal System Trace 使用；指标以低开销模式默认开启，诊断截图另行开关。

### J. 可迁移性（为脱离 Rosetta 留路）

后端热路径目前是 x86-64 代码，跑在 Rosetta 下。现在**不迁移**，但接口按可迁移设计：列表 IR、资源表、fence 和标签都表示为共享内存里的命令流和描述结构，不依赖 PE 侧的指针语义。将来可以整体迁到原生 arm64 进程（GPU 服务进程），或者随 arm64 Wine + FEX一起迁走。阶段 8 测出编码 CPU 超出预算时，再评估是否启动。

---

## 6. 风险与未决问题

| 风险 | 应对 |
|---|---|
| 跨队列自旋等待假设不成立，三个长着色器另有原因 | 阶段 1 静态分析先行；架构上多队列本来就需要，不依赖这个假设。 |
| 着色器在一个分发内部轮询另一队列写的内存：两个队列真正并发后，Apple GPU 不保证前向进展 | 不依赖并发轮询能成功。能识别的这类模式，在两次分发之间转成 event 等待；识别不了的，保留循环预算和隔离。 |
| 标签值非单调，或用掩码、`==` 比较 | 回退到 CPU 挂起，并计数。计数高时再研究专门的转换方式。 |
| 多队列共享 noCopy 内存和驻留集的行为未经测试 | 先做独立的小测试（规则 7）。 |
| Wine 下完成线程、event listener 的行为和延迟 | 先在合成测试中测量唤醒延迟；不在 Unix 侧回调里调用 PE 代码。 |
| GPU 时间戳语义改变，影响游戏的动态分辨率或帧节奏 | 先审计游戏怎么使用时间戳，再决定改不改。 |
| MSC 和 Metal 本身的限制（32 KiB、wave64、描述符模型） | G3、G4、G6。 |
| 游戏 CPU 路径就是最终上限 | 由阶段 6 的拆分结果决定是否投入 CPU 方向，以及 J。 |
