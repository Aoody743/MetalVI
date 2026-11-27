<h1 align="center">MetalVI</h1>

<p align="center">
<b>让 GTA VI（Xbox Series 版）在 Apple Silicon Mac 上运行的开源研究项目：原生 Metal 后端，统一内存零拷贝。</b><br>
M1 · M2 · M3 · M4 · M5 &nbsp;|&nbsp; macOS 15+ &nbsp;|&nbsp; Rust · Wine · Metal · Metal Shader Converter
</p>

<p align="center">
<img alt="积极开发中" src="https://img.shields.io/badge/状态-积极开发中-orange">
<img alt="暂不可玩" src="https://img.shields.io/badge/可玩-暂不可玩-red">
<img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-M1%2B-555555">
<img alt="原生 Metal" src="https://img.shields.io/badge/GPU-原生%20Metal-blue">
<img alt="GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-lightgrey">
</p>

<p align="center"><a href="README.md">English</a> · <b>简体中文</b></p>

> **请自备正版游戏。** 本仓库**不包含任何游戏文件、资源、密钥或专有 DLL**，以后也不会包含。它只能使用**你自己合法拥有、自行导出**的游戏副本运行，不提供下载、解密或解包，也不包含任何绕过 DRM 的内容。详见[法律声明](#法律声明)。

---

## 一句话说明

常见的做法是层层转译：游戏的 D3D12 调用先被翻译成 PC 的 D3D12，再由苹果的 D3DMetal 翻译成 Metal。在 Mac 上，这条路必须把 GPU 当成独立显卡，于是主机的**统一内存**只能靠每帧复制、哈希几个 GB 来模拟。我们实测：每帧上传约 1.4 GB、比对约 7.5 GB。所以无论怎么优化，**3D 场景大约只能到 1–2 fps**。

而 Apple Silicon 本身就是统一内存，和主机一样。**MetalVI 去掉了第二层转译**：渲染器**直接调用 Metal**，把游戏自己的内存**零拷贝**交给 GPU（`newBufferWithBytesNoCopy` + `MTLResidencySet`）；着色器通过**苹果官方的 Metal Shader Converter** 在进程内转换。性能上限由 GPU 决定，而不是被内存复制卡住。

**当前进展（2026-11-27）：** 游戏能启动、进入菜单、加载进 3D 世界并开始开场任务。计算工作已经基本跑在原生 Metal 上；绘制和上屏已能执行，但 3D 画面大部分还是黑的。算上停顿，3D 的真实平均帧率只有 **1.2–2.5 fps**，和 D3DMetal 路线的项目差不多。原因是现在的 Metal 后端还是一个**混合体**：它叠在旧的 D3D12 准备路径上，CPU 和 GPU 锁步执行，只用一条 GPU 队列，零拷贝的收益还没拿到。为此我们重新设计了**执行架构 v2**（见下文），目前是设计阶段，还没有实现。**目前不能玩**，这一点我们如实说明。

> ⚠️ **安全警告：** 开发过程中，实验性的 Metal 路径曾让单个 GPU 命令缓冲连续占用 6–10 秒。Apple GPU 不会为 WindowServer 抢占这类工作，结果整机卡死，有一次触发了内核 panic。在 v2 的 GPU 安全关卡完成之前，**请不要在存有重要工作的 Mac 上运行完整游戏。**

## 架构

### 两种把主机游戏搬到 Mac 的方式

```mermaid
flowchart LR
    subgraph OLD["常见方案：两层转译"]
        direction TB
        A1["游戏 x86-64 代码<br/>（Rosetta 2）"] --> A2["D3D12.X → PC D3D12<br/>翻译器"]
        A2 --> A3["D3DMetal<br/>（D3D12 → Metal）"]
        A3 --> A4["Metal"]
        A5[("主机统一内存")] -. "每帧复制 + 哈希<br/>数 GB" .-> A2
    end
    subgraph NEW["MetalVI：一条直连路径"]
        direction TB
        B1["游戏 x86-64 代码<br/>（Rosetta 2）"] --> B2["victory-gpu<br/>录制命令"]
        B2 --> B3["victorymetal unixlib<br/>（Wine → Metal）"]
        B3 --> B4["Metal"]
        B5[("游戏内存")] == "零拷贝<br/>newBufferWithBytesNoCopy" ==> B4
    end
```

### MetalVI 内部结构

```mermaid
flowchart TB
    subgraph WIN["Windows 侧：运行在 Wine + Rosetta 2 下"]
        L["victory-loader<br/>PE 加载 · 主机专有指令补丁 · 掩码存储修复"]
        H["victory-hle<br/>内核 · xmem · DirectStorage · XAsync · 输入 · 音频"]
        S["victory-shader<br/>RDNA2 指令 → DXIL 重编译器"]
        G["victory-gpu<br/>D3D12.X 运行时 · 描述符堆 · 纹理 · Metal 后端"]
        M["victory-metal<br/>安全的 Rust Metal 接口 · 零拷贝内存块 · 常驻集"]
    end
    subgraph MAC["macOS 侧"]
        U["victorymetal.so<br/>Wine unixlib（源自 dxmt）"]
        C["Metal Shader Converter<br/>DXIL → metallib，进程内转换"]
        MT["Metal / Apple GPU"]
    end
    T["游戏可执行文件<br/>（你自己的副本）"] --> L --> H
    H --> G
    S --> G
    G --> M -- "__wine_unix_call" --> U
    U --> C
    U --> MT
```

### 执行架构 v2（设计完成，尚未实现）

现在的 Metal 后端只替换了"图形 API"，执行方式还是 D3D12 移植时那一套：每次提交都等 GPU 做完；主机的多个队列压成一条；GPU 上的分发全部串行；没有针对 Apple GPU 的分块渲染（TBDR）做优化。v2 的目标是**让架构本身不给硬件利用率设上限**。

```mermaid
flowchart LR
    subgraph T["游戏线程（多个）"]
        R["列表录制<br/>→ 列表 IR"]
    end
    subgraph B["后端"]
        Q["队列时间线<br/>fence / 内存标签 → MTLSharedEvent"]
        P["Pass 规划（TBDR）<br/>+ 屏障翻译"]
        E["编码线程池<br/>每段一个命令缓冲"]
        D["完成线程<br/>bottom-of-pipe 写 · 回收 · 时长检测"]
    end
    subgraph G["Apple GPU"]
        GQ["图形队列"]
        CQ["计算队列"]
    end
    R --> Q --> P --> E
    E --> GQ & CQ
    GQ -. "event 排序" .- CQ
    GQ & CQ --> D --> Q
```

| 模块 | 做什么 | 解决什么 |
| --- | --- | --- |
| 异步队列时间线 | 每个游戏队列对应一个 Metal 队列；fence 和内存标签映射为 `MTLSharedEvent`；完成线程负责 bottom-of-pipe 写和资源回收 | CPU 与 GPU 锁步；跨队列同步只能靠 CPU |
| 并行编码 | 编码交给线程池，按顺序 `enqueue`；去掉全局锁 | 单线程编码，还要付 Rosetta 的开销 |
| TBDR pass 规划 | 清除并进 loadAction；按需用 DontCare 和 memoryless；不涉及附件的 barrier 不断开 pass | 渲染目标被反复存回、读出的带宽浪费 |
| 屏障翻译 | 计算编码器改为 concurrent；只翻译游戏自己的 barrier | GPU 上的分发全部串行 |
| 纹理所有权 | 线性纹理零拷贝；GPU 写的纹理只留在 GPU；CPU 写的纹理按已知写入者和页面脏位刷新，不再哈希 | 纹理仍是"复制 + 哈希"模型 |
| 着色器分级 | 已验证的管线去掉边界检查和循环预算；二进制归档；NGG 优先走顶点着色器路径 | 安全插桩的开销；冷编译卡顿；32 KiB 网格输出上限 |
| 安全模型 v2 | 编码时限定工作量、钳位间接参数、限制在途 GPU 时间、异步检测 | 用"逐个等待"换来的安全本身就是性能上限 |

完整设计见 [docs/metal-architecture-v2.zh-CN.md](docs/metal-architecture-v2.zh-CN.md)。

## 状态一览

测试机：M1 Max（64 GB，macOS 15.7.9）。截至 2026-11-27。标 *开发中* 的项目只发布了一部分，最新的安全和内存目录改动还在本地、未提交。

| 方面 | 状态 |
| --- | --- |
| 启动、菜单、加载画面 | ✅ 20–30 fps（D3DMetal 路径） |
| 进入 3D 世界、开场任务 | ✅ 已到达 |
| 游戏本身的稳定性（CPU 侧） | ✅ 修复后 10 分钟长测不崩溃（之前大约每 2 轮崩 1 轮） |
| 计算着色器跑在原生 Metal 上 | ✅ 99.3% 的 dispatch 完整绑定 |
| GPU 解 tile（直接从游戏内存零拷贝读取） | ✅ 与 CPU 结果逐字节一致，默认开启 |
| CPU 侧 3D 帧时间（关闭渲染时） | 35–75 ms（开 msync 后约 46–52 ms） |
| 通过 Metal 绘制、上屏 | 🚧 *开发中*：能执行，但依赖旧 D3D12 准备路径；G-Buffer 正确，光照和后处理之后大部分是黑的 |
| 3D 含停顿的真实平均帧率 | ⚠️ 1.2–2.5 fps（停顿之间能到游戏自带的 30 fps 上限，但有多秒长停顿） |
| GPU 安全（限制单次提交的 GPU 时长） | 🚧 *开发中*：已有拆分提交和事后熔断，**还没有硬上限** |
| 着色器翻译 | 🚧 25,953 个程序中剩 7 个失败；另有 27 条网格管线超出 Metal 的 32 KiB 输出上限 |
| 执行架构 v2（异步多队列、TBDR、并行编码） | 📐 设计完成，未实现 |
| 可以玩 | ❌ 暂不可玩 |

## 新架构的亮点

### 原生 Metal 后端
- **victorymetal**：基于 [3Shain/dxmt](https://github.com/3Shain/dxmt) 的 `winemetal`（LGPL-2.1）派生的 Wine unixlib，让 Windows 侧通过 `__wine_unix_call` 直接调用 Metal。它是按需加载的，没装的话程序照样走 D3DMetal 运行。
- **游戏内存零拷贝**：游戏内存按 2 MiB 一块包装成免复制的 `MTLBuffer`，页重映射时重新包装；所有被着色器按地址访问的资源都放进 `MTLResidencySet` 保持常驻。
- **Metal Shader Converter 进程内转换**：现有的 RDNA2 → DXIL 重编译器直接接 MSC，离线测试中 32,503 个着色器有 99.2% 转换成功，转换结果缓存在磁盘上。根签名按 victory 的布局生成，并开启越界检查，因为在零拷贝内存上一次越界写就是真实的内存损坏。
- **描述符堆镜像**成 `IRDescriptorTableEntry` 表：涵盖缓冲、类型化缓冲、纹理和纹理视图（会校验格式和类型，因为非法视图会让 Metal 直接断言）、D3D12 和主机采样器（会去重，因为 Apple7 超过 1024 个不同采样器后会串用）、空描述符（用清零的 1×1 纹理替代）。
- **GPU 解 tile**：直接从游戏内存把主机的纹理布局（GFX10 swizzle 模式）解开，GPU 处理不了的子资源回退到 CPU。

### 驯服 Rosetta 2 和 Wine
- **找到了随机堆损坏的真正原因。** Rosetta 2 把所有 AVX 掩码存储（`vpmaskmovd/q`、`vmaskmovps/pd`）实现成**对整块 16/32 字节的读-改-写**，掩码本该跳过的通道会被写回旧值。我们的竞争测试里，一个 `vpmaskmovd` 循环悄悄抹掉了另一个线程 **93% 的写入**。游戏里有 1,130 条这类指令，它们一直在破坏游戏分配器的空闲链表。现在这些指令在加载时被改写成逐通道的普通存储（`crates/victory-loader/src/masked.rs`，来自 [VIctory-macOS](https://github.com/pragyaangaur/VIctory-macOS)），之后分配器崩溃一次都没再出现。
- **主机专有 CPU 指令只修补一次**：`clzero`、`clwb`、`monitorx`/`mwaitx`、`rdpru`、`rdpid`、SSE4a 以前在 3D 中每秒触发约 21,000 次异常，现在每处只模拟一次，然后原地改写（`console_patch.rs`）。
- **消除 Wine 文件元数据死锁**：热点路径读缓存文件时不再查询文件元数据，因为在 Rosetta 下这个调用可能让所有线程卡死。
- **减少内存重映射**：目标位置已经显示同一个物理页时直接跳过，因为在 Wine 下每次重映射都要占用一把进程级的全局锁。
- 首次运行较慢时，可以关闭游戏自带的卡死看门狗（`VICTORY_NO_HANG_WATCHDOG=1`）。

### 背后的工程投入

```mermaid
timeline
    title MetalVI：2026 年 11 月
    架构审计 : 实测基于复制的 D3DMetal 路线上限约 1–2 fps : 决定改走 Metal 直连
    零拷贝 : 在 Wine + Rosetta 下验证免复制 MTLBuffer : 常驻集与重映射处理
    着色器 : Metal Shader Converter 进程内集成 : 32,503 个着色器 99.2% 转换成功
    计算上 Metal : 描述符堆、纹理、采样器、类型化缓冲 : 99.3% dispatch 完整绑定
    稳定性 : 发现并修复 Rosetta 掩码存储竞争 : 修复 Wine 元数据死锁 : 10 分钟长测不崩
    绘制 : 开发中：Metal 绘制、上屏、间接绘制 : 找到 GPU 长批次导致整机卡死的根因
    重新设计 : 执行架构 v2：异步多队列、TBDR pass 规划、并行编码、安全模型 v2
```

| | |
| --- | --- |
| 🔬 **先测量再动手** | 架构选择基于实测数据（每帧上传和哈希量、CPU 关键路径、线程等待），而不是凭猜测。 |
| 🧩 **从零搭建新后端** | Wine unixlib、安全的 Rust Metal 接口、零拷贝内存管理、MSC 集成、描述符堆镜像、纹理与采样器翻译、GPU 解 tile、管线磁盘缓存。 |
| 🐞 **攻克疑难问题** | 大约一半的运行都会出现的堆损坏，最终定位到 Rosetta 2 对 AVX 掩码存储的非原子模拟：用专门的竞争测试证实，再用加载期补丁根除。整个进程卡死的问题，定位到 Wine 在持有全局内存锁时调用挂载管理器。 |
| ✅ **验证而非假设** | 每个组件都配有测试：GPU 与 CPU 解 tile 逐字节对比、Apple GPU 硬件限制的原生 Metal 探针（1024 采样器串用、无 64 位原子、wave32）、Wine 下的单元测试，以及带崩溃统计的 A/B 实机对照。 |
| 🤝 **开放协作** | 研究任务拆成了无需游戏文件也能验收的小任务，开发者和 AI 智能体都可以参与。借鉴的周边项目修复都注明出处，并保持许可证兼容。 |

### 关键数字

| 指标 | 数值 |
| --- | --- |
| 通过 Metal Shader Converter 转换成功的着色器 | 32,248 / 32,503（99.2%） |
| Metal 上完整绑定的计算 dispatch | 99.3% |
| 每轮运行出现分配器崩溃的比例：修复前 → 修复后 | 约 45% → 0 |
| 3D 中主机专有指令每秒触发的异常：修复前 → 修复后 | 约 21,000 → 约 1,100 |
| 加载时改写的掩码存储指令 | 1,130 条 |
| 已验证的最长连续运行 | 10 分钟，零崩溃 |

### 为什么追求更高的上限
- **统一内存是 Mac 的天然优势**，只有直连 Metal 才能真正用上。基于复制的方案，瓶颈在于花在复制上的带宽，而不是 GPU。
- **Rosetta 2 不会永远存在**：苹果只承诺完整支持到 macOS 27。自己直接调用 Metal，渲染器就不依赖没有 ARM64 版本的 D3DMetal，将来也能顺利迁移到原生 ARM64 Wine + FEX。
- **以 Series S 为目标画质**：去掉转译开销之后，它的 GPU 负载在 M 系列芯片的能力范围内。

## 性能预测与设备兼容性

> **这一节全部是预测，不是实测。** 唯一的实测平台是 M1 Max，目前 3D 的真实平均帧率是 1.2–2.5 fps。下面的数字假设 v2 架构已经完成、画面已经正确，并使用 Series S 配置。实测之后我们会更新这些数字，包括把它们改低。

### 性能损耗来自哪里（与主机原生运行相比）

| 来源 | 现状 | v2 完成后的预测 | 依据 |
| --- | --- | --- | --- |
| Rosetta 2 运行游戏的 x86-64 代码 | 标量约为原生的 70–80%，AVX2 重的代码约 50% | 不变（只有脱离 Rosetta 才能改善） | 公开资料，加上我们的 CPU 测量 |
| Wine 的同步与系统调用 | 开 msync 后内核时间约 15%，工作线程 59–88% 的时间在等待 | 继续优化等待路径，空间有限 | 线程等待采样 |
| 主机专有指令和掩码存储补丁 | 异常次数 21k/s → 1.1k/s | 接近 0 | 实测 |
| 内存复制和哈希 | 缓冲已零拷贝；旧路径残留 2.4 GB 影子副本，每帧还在做哈希 | 缓冲 0；纹理只对脏区做 GPU 解 tile，每帧约几毫秒 | 旧路线实测每帧 3–6 秒 |
| CPU 与 GPU 锁步、单队列 | 帧时间 ≈ CPU + GPU | 两者重叠，帧时间 ≈ max(CPU, GPU) | 架构推导 |
| 后端编码 | 单线程，在提交路径上 | 每帧 3–6 ms CPU，不在关键路径上 | 估算：每帧约 2,300 次绘制加上网格分发 |
| 着色器翻译（RDNA2 → DXIL → Metal） | 估计 ALU 开销 1.5–3 倍，另有安全插桩 | 已验证管线去掉插桩后，估计 1.3–2 倍 | **未实测**，最大的不确定项之一 |
| 光线追踪（RDNA2 的 BVH 指令用软件模拟） | 刚能执行，正确性未知 | 未知，可能很重 | 未测 |
| 渲染目标带宽（TBDR） | 没有优化，pass 频繁断开 | 和立即模式渲染持平或更好 | 架构推导 |

### 上限预测（M1 Max，Series S 配置）

- 帧率约等于 min(CPU 能跑多快, GPU 能跑多快)，并且受游戏自带的 30 fps 上限约束。
- **CPU**：过场动画在停顿之间达到 33 ms（30 fps）；另一组纯 CPU 测量是 46–52 ms（约 20 fps）。所以 CPU 的上限大约在 **20–30 fps**，取决于场景。
- **GPU**：M1 Max 是 10.4 TFLOPS，按翻译开销 1.5–3 倍折算，有效算力 3.5–7 TFLOPS，大致覆盖 Series S（4 TFLOPS）。GPU 侧大约 **26–30 fps**。
- **预测**：完成后，大多数场景 **20–30 fps**，重场景（比如在城市里开车）**15–20 fps**。整个游戏稳定锁 30 fps 的可能性不高。这个预测的置信度是低到中等，主要取决于着色器翻译开销和光追开销的实测结果。

### 设备兼容性预测（v2 完成后，Series S 配置）

| 机型（最低配置） | CPU | GPU（约 FP32） | 内存 | 预测 | 结论 |
| --- | --- | --- | --- | --- | --- |
| M1 / M2（8–10 核 GPU） | 4P+4E | 2.6–3.6 TF | 8–24 GB | < 15 fps | ❌ 不推荐 |
| **M1 Pro 最低配 MacBook Pro**（8 核 CPU / 14 核 GPU / 16 GB） | 6P+2E | 约 4.6 TF，200 GB/s | 16 GB 偏紧 | 15–20 fps，需要降分辨率或用 MetalFX；可能因内存不足出现卡顿 | ⚠️ 能运行，但不流畅 |
| M3（10 核 GPU） | 4P+4E | 约 3.6 TF，100 GB/s | 8 GB 不支持；16/24 GB | 12–20 fps，需要 MetalFX | ⚠️ 勉强 |
| M4（10 核 GPU） | 4P+6E | 约 4.3 TF，120 GB/s | 16 GB 起 | 15–25 fps，需要 MetalFX | ⚠️ 可玩下限 |
| M1 Max（24/32 核 GPU） | 8P+2E | 7.8–10.4 TF，400 GB/s | 32 GB 起 | 20–30 fps | ✅ 当前开发机 |
| M3 Pro（14/18 核 GPU） | 5–6P+6E | 约 5–6.4 TF，150 GB/s | 18 GB 偏紧 | 20–28 fps | ✅ 基本可玩 |
| M4 Pro（16/20 核 GPU） | 8–10P+4E | 约 6.6–8.3 TF，273 GB/s | 24 GB 起 | 25–30 fps | ✅ 推荐 |
| M3 Max / M4 Max | 10–12P | 约 14–18 TF，300–546 GB/s | 36 GB 起 | 大多数场景达到 30 fps 上限 | ✅ 最佳 |

几点说明：
- **内存**：Series S 配置下，游戏自身就要约 7.6 GB，再加上 Metal 纹理副本、Wine、Rosetta 和各种缓存，预计进程占用 13–17 GB（目前带着旧影子路径，约 30 GB）。所以 8 GB 机型不支持，16 GB 机型会有换页卡顿，**24 GB 以上比较合适**。
- **CPU**：游戏约有 7 条重负载线程。性能核少于 6 个的机型，即使单核更快（M3、M4），也容易受 CPU 限制。M3 到 M4 单核比 M1 快约 30–50%，所以同样核数时 M4 好得多。
- **GPU**：M3 和 M4 有动态缓存，可能降低翻译后着色器的寄存器压力，这是有利因素，但未实测。它们的硬件光追不能直接用：游戏的光追是 AMD 的 BVH 格式，要用上就得重建 Metal 加速结构，这是远期选项。
- **MetalFX 超分**是低端机型最主要的手段：以较低分辨率渲染，再放大输出。
- **系统和磁盘**：需要 macOS 15 及以上（`MTLResidencySet` 要求），以及 Rosetta 2。游戏导出约 209 GB，再加缓存，256 GB 硬盘的机型放不下。

## 路线图

已完成：
- [x] P0：victorymetal unixlib、零拷贝内存块、常驻集、安全的 Rust 封装
- [x] P1：游戏内计算跑在 Metal 上：MSC 进程内转换、描述符堆、纹理、采样器、类型化缓冲、GPU 解 tile、磁盘缓存
- [x] 稳定性：Rosetta 掩码存储竞争、Wine 元数据死锁、卡死看门狗
- [x] P2/P3 原型：Metal 绘制、上屏、间接绘制、深度与 MSAA 解析。它依赖旧 D3D12 准备路径，v2 会替换掉

执行架构 v2（[设计文档](docs/metal-architecture-v2.zh-CN.md)）：
- [ ] 阶段 1：GPU 安全关卡 v2（编码时限定工作量、钳位间接参数、在途预算、异步检测）
- [ ] 阶段 2：可观测性（按 pass 计 GPU 时间）、二进制归档、可离线重放的帧包
- [ ] 阶段 3：执行模型：异步多队列、fence 和标签映射为 event、完成线程、屏障翻译
- [ ] 阶段 4–5：独立的资源、绑定和绘制准备，TBDR pass 规划，关闭旧的复制、监视和哈希机制
- [ ] 阶段 6：路线判断：按帧预算拆分实测结果
- [ ] 阶段 7：补齐渲染语义：剩余着色器、32 KiB 网格输出、ExecuteIndirect、光追、画面发黑
- [ ] 阶段 8：性能：着色器分级、并行编码、MetalFX
- [ ] 远期：原生 ARM64 Wine + FEX，摆脱对 Rosetta 的依赖

## 在 macOS 上编译和运行

### 准备
- Apple Silicon Mac，macOS 15 或更新版本，已安装 Rosetta 2。
- 带 `macdrv_functions` 的 macOS Wine（GameHub/gamesir-labs 的 Wine-Proton 可用）；走 D3DMetal 路径还需要苹果的 Game Porting Toolkit。
- **苹果的 Metal Shader Converter**：从苹果开发者网站自行安装（会提供 `/usr/local/lib/libmetalirconverter.dylib`）。本项目不分发它，`native/victorymetal/build.sh` 链接的是你本机安装的副本。
- 你自己导出的游戏副本。

```bash
brew install zig llvm@21 lld@21
rustup target add x86_64-pc-windows-gnullvm
cargo install cargo-zigbuild
```

### 编译
```bash
export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$PATH"
cargo zigbuild --release -p victory --target x86_64-pc-windows-gnullvm --locked
# 编译 Metal unixlib 并安装到你的 Wine（WINE_ROOT=/path/to/wine）：
native/victorymetal/build.sh --install
```

### 配置（放在仓库外面）
在仓库外新建一个运行目录，比如 `~/metalvi-runtime`，里面放一个 `victory.toml`：
```toml
[paths]
game_dir = 'Z:\Users\<you>\path\to\your\game_dump'
cache_dir = 'Z:\Users\<you>\metalvi-runtime\cache'
data_dir = 'Z:\Users\<you>\metalvi-runtime\data'
test_titles = []
```

### 预热着色器，然后启动
```bash
cd ~/metalvi-runtime
/path/to/MetalVI/scripts/run-mac.sh /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  warm --jobs 6 2>&1 | tee warm.log
VICTORY_METAL=compute VICTORY_NO_HANG_WATCHDOG=1 \
/path/to/MetalVI/scripts/run-mac.sh /path/to/MetalVI/target/x86_64-pc-windows-gnullvm/release/victory.exe \
  run --execute --console series-s 'Z:\Users\<you>\path\to\your\game_dump\title.exe' 2>&1 | tee run.log
```
请始终用 `2>&1 | tee` 记录日志：在 Wine 下直接用 `>` 重定向会丢失控制台输出。

### 常用开关
| 变量 | 作用 |
| --- | --- |
| `VICTORY_METAL=compute` | 计算工作走原生 Metal 后端 |
| `VICTORY_METAL_GPU_DETILE=0` | 纹理改用 CPU 解 tile |
| `VICTORY_METAL_TEXTURES=0` | 纹理槽位留空（用于二分排查） |
| `VICTORY_METAL_CACHE=<dir>` | 转换好的 Metal 管线缓存位置 |
| `VICTORY_PATCH_MASKED=none` | 不改写掩码存储（Rosetta 下不建议） |
| `VICTORY_NO_HANG_WATCHDOG=1` | 关闭游戏自带的卡死看门狗 |
| `VICTORY_INSN_PATCH=0` | 主机专有指令改回每次都陷入模拟 |

### 键盘操作
| 按键 | 作用 |
| :--- | :--- |
| `Enter` / `Space` | 确认（A） |
| `Backspace` | 返回（B） |
| `W` `A` `S` `D` | 左摇杆 |
| `I` `J` `K` `L` | 右摇杆（视角） |
| 方向键 | 十字键 |
| `Q` / `E` | LB / RB |
| `左 Shift` / `右 Shift` | LT / RT |
| `Esc` | 菜单 |

## 常见问题

**现在能在 Mac 上玩 GTA VI 吗？**
还不能。游戏能启动、加载世界、开始剧情，但 3D 画面大部分还是黑的，算上停顿的真实平均帧率只有 1.2–2.5 fps。D3DMetal 路线的项目，3D 大约是 0.1–4 fps。

**会支持哪些 Mac？**
技术上支持 macOS 15 及以上的所有 Apple Silicon Mac。但按我们的预测，想要接近可玩，大约需要 M1 Pro 以上（或 M4），内存 24 GB 以上。详见[设备兼容性预测](#设备兼容性预测v2-完成后series-s-配置)。开发和测试在 M1 Max 上进行。

**这是模拟器吗？**
我们没有自己写 CPU 模拟器。游戏的 x86-64 代码由 Rosetta 2 运行，Wine 提供 Windows 环境。MetalVI 实现了主机的系统库（高层模拟），重编译主机 GPU 的 RDNA2 着色器，并通过 Metal 渲染。

**能和 CrossOver / Whisky / GPTK 一起用吗？**
D3DMetal 路径可以在支持 GPTK 的 Wine 上运行；原生 Metal 后端需要把我们的 victorymetal unixlib 安装到你使用的 Wine 里。

**和其他移植项目有什么不同？**
多数项目走 D3D12 → D3DMetal，和模拟统一内存的开销苦苦缠斗。MetalVI 的渲染器直连 Metal、内存零拷贝：前期工作量更大，但在 Apple Silicon 上的上限高得多。

**游戏文件从哪里来？**
来自你自己拥有、自行导出的副本。我们不提供任何获取游戏文件的帮助，分享或链接游戏文件的 issue/PR 会被直接关闭。

## 参与贡献

欢迎开发者和 **AI 编程智能体**参与。工作被拆成了可以客观验收的小任务：带测试的纯函数、GPU 与 CPU 逐字节对比、原生 Metal 探针等，不需要游戏文件就能完成和验证。
- 适合上手的方向：Metal 状态映射、Apple GPU 限制的 Metal 探针、面向 Metal Shader Converter 的着色器降级写法、测试工具。
- 绝对不要提交任何游戏内容。`cargo xtask policy` 和 pre-commit 钩子会强制检查（见 `docs/asset-policy.md`）。
- 提交 PR 时请写明你是如何验证的（命令和输出）。

## 法律声明

- MetalVI 是独立的研究与互操作项目，**与** Rockstar Games、Take-Two Interactive、微软、苹果**均无关联，也未获得其认可或赞助**。*Grand Theft Auto*、*Xbox*、*Metal* 均为各自所有者的商标，这里仅用于说明兼容性。
- 仓库只包含开源许可下的原创源代码，不含游戏二进制、资源、密钥、SDK 头文件或专有 DLL。苹果的 Metal Shader Converter 和 Game Porting Toolkit 由每位用户按苹果的条款自行获取，本仓库不包含。
- 请只配合你合法拥有的软件使用。我们不支持盗版，也不支持绕过 DRM。

## 鸣谢与许可

MetalVI 采用 **GPL-3.0-or-later** 许可（见 [LICENSE](LICENSE)）。`native/victorymetal` 采用 **LGPL-2.1-or-later**（来自 dxmt 和 Wine，见 `native/victorymetal/NOTICE.md`）。

本项目建立在以下工作之上：
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**：原始的加载器、HLE 运行时、D3D12 翻译器和 RDNA2 着色器重编译器。
- **[3Shain/dxmt](https://github.com/3Shain/dxmt)**：victorymetal 派生自它的 `winemetal` 层。
- **[pragyaangaur/VIctory-macOS](https://github.com/pragyaangaur/VIctory-macOS)**：掩码存储补丁、卡死看门狗补丁，以及 Wine 元数据和重映射修复（GPL-3.0）。
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**、**[JoshuuDev/victory-amd](https://github.com/JoshuuDev/victory-amd)**、**[Zettrox/ztrx](https://github.com/Zettrox/ztrx)**：着色器、描述符、纹理、DirectStorage 和管线方面的工作。
- 设计参考（仅借鉴思路）：xcl、TomBox、five-star-recomp、KosmicKrisp、MoltenVK、Mullion。

<sub>关键词：GTA 6 Mac、GTA VI macOS、GTA6 苹果电脑、Apple Silicon、M1 M2 M3 M4 M5、Metal 后端、Metal Shader Converter、Xbox Series 兼容层、RDNA2 着色器重编译、Rosetta 2、Wine、Game Porting Toolkit、D3DMetal、统一内存零拷贝、开源</sub>
