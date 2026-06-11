---
title: NCU Introduction
date: 2026-06-11
tags: [CUDA, GPU, NCU]
summary: 整理 Nsight Compute 中 GPU Speed Of Light Throughput 与 PM Sampling 相关指标的中文翻译与含义，帮助快速判断 Kernel 的计算、缓存和显存资源利用情况。
---

# NCU GPU Speed Of Light 指标笔记

Nsight Compute 中的 **GPU Speed Of Light Throughput** 页面用于观察 Kernel 对 GPU 理论峰值资源的利用情况。它的核心思想是把实际达到的吞吐量和硬件理论峰值做比较：

实际吞吐量 / 理论最大吞吐量 * 100%

因此，这类指标通常以百分比展示。数值越高，表示对应硬件资源越接近满负载运行；数值越低，说明该资源不是当前 Kernel 的主要压力点，或者 Kernel 没有充分触发对应硬件路径。

## 总览指标

| 英文指标 | 中文名称 | 说明 | 观察重点 |
| --- | --- | --- | --- |
| Compute Throughput | 计算吞吐率 | 衡量 CUDA Core、Tensor Core、FMA、ALU 等计算执行资源相对于理论峰值的利用率。 | 如果该值很高，通常说明 Kernel 更偏计算受限；如果很低，可能瓶颈不在计算单元。 |
| Memory Throughput | 内存吞吐率 | 衡量整个内存层次结构的综合吞吐利用率，包括 L1/TEX、L2、DRAM 以及相关访存通路。 | 这是内存系统的总体压力指标，需要结合 L1、L2、DRAM 的细分项判断瓶颈位置。 |
| L1/TEX Cache Throughput | L1/TEX Cache 吞吐率 | 衡量 L1 Cache 与 Texture Cache 相关路径的吞吐利用率。 | 如果该值很高，说明一级缓存、Load/Store 或纹理路径压力较大。 |
| L2 Cache Throughput | L2 Cache 吞吐率 | 衡量 L2 Cache 相关读写、sector、crossbar 等路径的吞吐利用率。 | 如果 L2 高而 DRAM 低，可能说明访存主要被 L2 命中或 L2 内部路径限制。 |
| DRAM Throughput | DRAM 吞吐率 | 衡量显存带宽相对于理论显存带宽的利用率。 | 如果 DRAM 接近峰值，通常说明 Kernel 可能受显存带宽限制；如果很低，显存不是主要瓶颈。 |

## Kernel 运行信息

这些指标不是吞吐率本身，但它们提供了理解 Kernel 性能的运行背景。

| 英文指标 | 中文名称 | 说明 | 观察重点 |
| --- | --- | --- | --- |
| Duration | 执行时间 | Kernel 从开始到结束的耗时。 | 用于和优化前后的版本直接比较，也是计算带宽、FLOPS 等派生指标的基础。 |
| Elapsed Cycles | 总周期数 | Kernel 执行期间经过的 GPU 时钟周期数。 | 可以和 Duration、频率一起检查测量是否合理。 |
| SM Active Cycles | SM 活跃周期数 | SM 处于活跃状态、正在执行或等待相关工作的周期数。 | 如果明显低于总周期数，可能存在调度空洞、并发不足或 Kernel 太小的问题。 |
| SM Frequency | SM 频率 | Kernel 执行期间 SM 核心的平均运行频率。 | GPU 频率会受功耗、温度、Boost 策略影响，分析性能时需要记录。 |
| DRAM Frequency | DRAM 频率 | Kernel 执行期间显存的平均运行频率。 | 显存频率会影响理论带宽上限，比较不同实验时需要保持环境一致。 |

## Compute Throughput Breakdown

**Compute Throughput Breakdown** 用来查看计算资源内部不同执行流水线的利用情况。它不是只统计浮点计算，也会包含指令发射、Load/Store、地址计算、分支控制等与计算执行相关的路径。

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| SM: Inst Executed Pipe Lsu | SM Load/Store 流水线执行指令吞吐率 | 表示由 LSU 执行的访存相关指令占理论峰值的比例。该项高时，Kernel 可能有较多 load/store 指令压力。 |
| SM: Issue Active | SM 指令发射活跃率 | 表示 SM 指令调度器处于发射指令状态的比例。该值反映指令发射层面的忙碌程度。 |
| SM: Inst Executed | SM 指令执行吞吐率 | 表示 SM 实际执行指令的总体吞吐情况。它可以反映 Kernel 的总体指令执行强度。 |
| SM: Mio Inst Issued | MIO 指令发射吞吐率 | 表示 Memory Input Output 相关指令的发射活跃程度，通常与访存、共享内存、特殊数据移动等路径相关。 |
| SM: Mio2rf Writeback Active | MIO 到寄存器文件回写活跃率 | 表示 MIO 路径向寄存器文件写回结果的活跃程度。该项高时，可能说明数据回写路径较忙。 |
| SM: Pipe Fma Cycles Active | FMA 流水线活跃周期占比 | 表示浮点乘加流水线处于活跃状态的比例，常用于观察 FP32 FMA 计算压力。 |
| SM: Pipe Fmaheavy Cycles Active | FMA Heavy 流水线活跃周期占比 | 表示较重 FMA 相关执行路径的活跃程度，通常与特定浮点计算指令路径有关。 |
| SM: Pipe Alu Cycles Active | ALU 流水线活跃周期占比 | 表示整数、逻辑、基础算术等 ALU 路径的活跃比例。 |
| SM: Mio Pq Read Cycles Active | MIO 队列读活跃周期占比 | 表示 MIO pipeline queue 读侧处于活跃状态的比例。 |
| SM: Mio Pq Write Cycles Active | MIO 队列写活跃周期占比 | 表示 MIO pipeline queue 写侧处于活跃状态的比例。 |
| SM: Inst Executed Pipe Adu | 地址计算单元执行指令吞吐率 | ADU 负责地址计算等操作。该项高时，说明地址计算相关指令占比较高。 |
| SM: Inst Executed Pipe Cbu Pred On Any | 控制/分支单元执行指令吞吐率 | CBU 与控制流、分支、谓词等指令相关。该项可用于观察控制流压力。 |
| SM: Inst Executed Pipe Uniform | Uniform 流水线执行指令吞吐率 | Uniform pipeline 处理 warp 内一致的 uniform 指令。该项通常用于观察 uniform 路径是否有压力。 |
| IDC: Request Cycles Active | 指令数据缓存请求活跃周期占比 | 表示指令相关缓存请求处于活跃状态的比例。 |
| SM: Inst Executed Pipe Ipa | IPA 流水线执行指令吞吐率 | IPA 通常与插值、属性处理等特定图形/计算路径相关，普通 CUDA Kernel 中常见为低值。 |
| SM: Inst Executed Pipe Tex | TEX 流水线执行指令吞吐率 | 表示 Texture pipeline 执行指令的吞吐情况。普通全局内存访问未必主要走该路径。 |
| SM: Inst Executed Pipe Xu | XU 流水线执行指令吞吐率 | XU 是特殊执行单元路径，可能与特定特殊函数或硬件指令有关。 |
| SM: Instruction Throughput Internal Activity | 指令吞吐内部活动 | 表示指令吞吐相关的内部活动占比，用于补充观察 SM 内部指令路径。 |
| SM: Memory Throughput Internal Activity | 内存吞吐内部活动 | 表示内存吞吐相关的内部活动占比，用于补充观察 SM 内部访存路径。 |
| SM: Pipe Fp64 Cycles Active | FP64 流水线活跃周期占比 | 表示双精度计算流水线活跃程度。没有 FP64 计算时通常为 0。 |
| SM: Pipe Tensor Cycles Active | Tensor Core 流水线活跃周期占比 | 表示 Tensor Core 执行路径的活跃程度。没有使用 Tensor Core 时通常为 0。 |
| SM: Pipe Tensor Cycles Active V2 | Tensor Core V2 流水线活跃周期占比 | 表示新版 Tensor Core 指标口径下的流水线活跃程度。 |

## Memory Throughput Breakdown

**Memory Throughput Breakdown** 用于拆解内存系统中各级缓存、crossbar、DRAM 以及请求路径的利用率。分析访存瓶颈时，通常需要同时看 L1、L2 和 DRAM，而不是只看 Memory Throughput 总值。

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| L1: Lsuin Requests | L1 来自 LSU 的请求吞吐率 | 表示 L1/TEX 接收到来自 Load/Store Unit 的请求强度。该项高时，说明普通 load/store 访问压力较大。 |
| L1: Data Pipe Lsu Wavefronts | L1 数据通路 LSU wavefront 吞吐率 | 表示 L1 数据管线处理 LSU wavefront 的活跃程度。wavefront 可理解为缓存访问被拆分后的内部处理单元。 |
| L1: Lsu Writeback Active | L1 LSU 写回活跃率 | 表示 LSU 路径写回相关活动的占比，常与 store 或 load 结果回写有关。 |
| L1: M Xbar2l1tex Read Sectors | Crossbar 到 L1/TEX 读 sector 吞吐率 | 表示从 crossbar 到 L1/TEX 的读 sector 活动。sector 是 cache line 的更小传输粒度。 |
| L2: T Sectors | L2 总 sector 吞吐率 | 表示 L2 Cache 处理 sector 的总体强度，是观察 L2 压力的重要指标。 |
| L2: Lts2xbar Cycles Active | L2 到 Crossbar 活跃周期占比 | 表示 L2 向 crossbar 发送数据或响应时的活跃程度。 |
| L1: Data Bank Reads | L1 数据 bank 读吞吐率 | 表示 L1 数据 bank 读访问强度。该项高时，可能说明 L1 读路径较忙。 |
| L2: Xbar2lts Cycles Active | Crossbar 到 L2 活跃周期占比 | 表示 crossbar 向 L2 发送请求时的活跃程度。 |
| L2: D Sectors | L2 数据 sector 吞吐率 | 表示 L2 数据路径处理 sector 的吞吐情况。 |
| GPU: Compute Memory Access Throughput Internal Activity | GPU 计算访存吞吐内部活动 | 表示计算访存相关内部路径的活动强度，用于补充判断访存系统压力。 |
| L2: T Tag Requests | L2 标签查询请求吞吐率 | 表示 L2 tag lookup 请求强度。该项高时，说明 L2 正在频繁进行缓存标签查询。 |
| L1: M L1tex2xbar Req Cycles Active | L1/TEX 到 Crossbar 请求活跃周期占比 | 表示 L1/TEX 向 crossbar 发出请求的活跃程度。 |
| L1: Data Bank Writes | L1 数据 bank 写吞吐率 | 表示 L1 数据 bank 写访问强度。 |
| DRAM: Cycles Active | DRAM 活跃周期占比 | 表示显存处于活跃服务状态的周期比例，是判断显存带宽压力的重要指标。 |
| DRAM: Dram Sectors | DRAM sector 吞吐率 | 表示 DRAM 实际传输 sector 的强度。它比单纯的活跃周期更接近数据传输量。 |
| L2: D Sectors Fill Device | L2 从设备显存填充的 sector 吞吐率 | 表示 L2 从 GPU device memory 拉取数据进行填充的强度。 |
| L1: Texin Sm2tex Req Cycles Active | SM 到 TEX 请求活跃周期占比 | 表示 SM 向 Texture/L1TEX 路径发出请求的活跃程度。 |
| GPU: Compute Memory Request Throughput Internal Activity | GPU 计算内存请求内部活动 | 表示计算内存请求相关内部路径的活动强度。 |
| L1: Data Pipe Tex Wavefronts | L1 TEX 数据管线 wavefront 吞吐率 | 表示 TEX 路径在 L1 数据管线中处理 wavefront 的强度。 |
| L1: F Wavefronts | L1 F 路径 wavefront 吞吐率 | 表示 L1 中特定 F 路径处理 wavefront 的强度，普通 CUDA Kernel 中可能较低。 |
| L1: Tex Writeback Active | L1 TEX 写回活跃率 | 表示 TEX 路径写回活动的占比。 |
| L2: D Atomic Input Cycles Active | L2 原子操作输入活跃周期占比 | 表示 L2 处理 atomic 输入请求的活跃程度。没有原子操作时通常为 0。 |
| L2: D Sectors Fill Sysmem | L2 从系统内存填充的 sector 吞吐率 | 表示 L2 从主机系统内存方向填充数据的强度。普通 GPU device memory 访问中通常较低。 |

## PM Sampling

**PM Sampling** 是 Nsight Compute 对性能监控指标进行周期性采样后生成的时间线视图。它不是只给出整个 Kernel 的平均值，而是展示 workload 在运行过程中性能指标如何随时间变化。

这部分数据通常会跨多个 profiling pass 收集。它适合用来观察：

- **Kernel 行为是否稳定**：例如吞吐率是否从头到尾基本一致，还是某些阶段明显下降。
- **是否存在阶段性瓶颈**：例如前半段计算密集，后半段访存密集。
- **是否存在尾部效应**：例如 Kernel 末尾只剩少量 CTA，导致 SM 利用率下降。
- **是否存在采样数据丢失**：如果 dropped samples 不为 0，说明采样结果可能不完整。

### PM Sampling 采样配置

| 英文指标 | 中文名称 | 说明 | 观察重点 |
| --- | --- | --- | --- |
| Maximum Sampling Interval | 最大采样间隔 | 两次性能指标采样之间允许的最大时间间隔，通常以微秒为单位。 | 间隔越小，时间线越细；但采样开销和数据量也可能更大。 |
| # Pass Groups | Pass 分组数量 | NCU 为收集全部采样指标而划分的 profiling pass group 数量。 | pass 越多，说明一次运行无法同时采集所有指标，需要多次采样合并。 |
| Maximum Buffer Size | 最大缓冲区大小 | 存储采样数据的最大缓冲区容量，通常以 MB 为单位。 | 如果指标多、采样密度高、Kernel 时间长，可能需要更大的缓冲区。 |
| Dropped Samples | 丢弃采样数量 | 因缓冲区、采样压力或其他限制而未能记录的采样点数量。 | 理想情况下应为 0；如果不为 0，时间线分析需要谨慎。 |

## PM Sampling Overview

Overview 部分给出 workload 在采样时间线上的整体执行状态，重点关注活跃 warp、CTA 调度、SM 活跃周期和指令执行效率。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Average Active Warps Per Cycle | 平均每周期活跃 Warp 数 | 统计整个 Kernel 生命周期内，平均每个时钟周期有多少 warp 处于活跃状态。 | 用于观察实际 warp 利用率，以及理论 occupancy 是否真正转化为执行中的活跃 warp。 |
| Total Active Warps Per Cycle | 总活跃 Warp 数 | 统计所有 SM 上每个周期实际活跃 warp 的总数。 | 可以反映整体并行度。如果该值偏低，可能存在并发不足、大量 stall 或尾部效应。 |
| Blocks Launched | 已启动 Block 数量 | 记录 Kernel 执行过程中 CTA，也就是 block 的启动情况。 | 用于观察 CTA 调度是否均匀，以及 Kernel 末尾是否因为剩余 block 较少而出现 tail effect。 |
| SM Active Cycles | SM 活跃周期数 | 表示 SM 真正处于活跃状态、正在执行或等待相关工作的周期数。 | 可以近似理解为 SM busy time，用来判断 SM 是否长时间没有工作可做。 |
| Executed IPC Active | 活跃周期内执行 IPC | IPC 是 Instructions Per Cycle，表示 SM 活跃周期内平均每周期执行的指令数。 | 用于观察指令发射效率和流水线利用率。IPC 偏低时，需要结合 stall 原因和指令结构分析。 |

## PM Sampling SM 指标

SM 部分主要观察 Streaming Multiprocessor 内部执行资源的利用情况，包括 ALU、FMA 和 Tensor Core 等计算路径。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| SM Throughput | SM 吞吐率 | 综合反映 SM 整体执行资源的利用率。 | 如果该值较高，说明 SM 内部执行资源压力较大；如果较低，可能瓶颈在访存、同步或调度。 |
| SM ALU Pipe Throughput | ALU 流水线吞吐率 | 反映整数运算、逻辑运算、地址运算等 ALU 路径的吞吐利用情况。 | 地址计算、索引计算、整数逻辑较多的 Kernel 可能会提高该指标。 |
| SM FMA Light Pipe Throughput | 轻量 FMA 流水线吞吐率 | 反映普通浮点乘加路径的利用情况，例如 `c += a * b` 这类 FMA 操作。 | 用于观察普通浮点计算是否占据主要执行压力。 |
| SM FMA Heavy Pipe Throughput | 重型 FMA 流水线吞吐率 | 反映更高吞吐或更重的浮点 FMA 执行路径利用情况，常与 FP32 FMA 等计算密集指令相关。 | GEMM、stencil、向量计算等高强度浮点 Kernel 可能会重点关注该项。 |
| SM Tensor Pipe Throughput | Tensor Core 流水线吞吐率 | 反映 Tensor Core 执行路径的利用情况。 | 使用 `mma.sync`、WMMA、CuTe GEMM、CUTLASS 或 Triton matmul 时，应重点观察该指标。 |

## PM Sampling DRAM 指标

DRAM 部分用于观察显存读写带宽随时间的变化。它可以帮助判断 Kernel 是否真正打到了 HBM/GDDR 的带宽瓶颈。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| DRAM Throughput | 显存吞吐率 | 表示实际显存带宽相对于理论峰值带宽的利用率。 | 如果持续接近峰值，Kernel 很可能受显存带宽限制。 |
| DRAM Read Bandwidth | 显存读取带宽 | 表示从 HBM/GDDR 读取数据的速度。 | 读密集 Kernel 应重点观察该指标，例如 copy、gather、GEMM 加载矩阵数据等。 |
| DRAM Write Bandwidth | 显存写入带宽 | 表示向 HBM/GDDR 写回数据的速度。 | 写密集或读写混合 Kernel 应结合 read bandwidth 一起判断带宽压力。 |

## PM Sampling L2 Cache 指标

L2 Cache 是 SM 与 DRAM 之间的重要缓存层。L2 指标可以帮助判断访问是否主要停留在缓存层，还是频繁落到显存。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| L2 Throughput | L2 缓存吞吐率 | 表示 L2 Cache 整体工作强度，包括读写请求、sector 处理和内部数据传输。 | 如果 L2 高而 DRAM 低，说明压力可能主要集中在 L2 或缓存命中路径。 |
| L2 Hit Rate | L2 缓存命中率 | 表示访问 L2 时成功命中的比例。公式可理解为 `Hit / (Hit + Miss)`。 | 命中率高通常意味着 DRAM 压力降低；命中率低则可能导致更多显存访问。 |

## PM Sampling L1 Cache 指标

L1 Cache 部分主要观察 SM 附近的一级缓存访问强度、命中情况和回写行为。对局部性、访存合并、共享内存/L1 路径分析很有帮助。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| L1 Throughput | L1 缓存吞吐率 | 表示 L1 Cache 整体访问强度。 | 如果 L1 吞吐率很高，说明大量请求停留在 SM 附近缓存层或 L1/TEX 路径。 |
| Writeback Throughput | 回写吞吐率 | 表示 L1 向下一级缓存回写数据的速度，通常是 L1 到 L2。 | store 较多、缓存写回较频繁的 Kernel 需要关注该项。 |
| Hit Rate | L1 缓存命中率 | 表示 L1 Cache 命中的比例。 | 命中率高说明局部性较好；命中率低可能意味着访问模式不规则或工作集超过 L1 容量。 |
| Wavefronts (Data) | 数据访问波前数 | wavefront 是 L1 Cache 内部一次处理的一批访问请求，该指标表示 L1 数据访问请求数量。 | 可用于观察 L1 数据路径的请求压力，尤其适合分析访存合并和访问模式。 |

## PM Sampling Workload Execution

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Workload Execution | 工作负载执行时间轴 | 表示 Kernel 真正执行的时间范围，通常在 PM Sampling 页面中显示为一条连续色带。 | 用于观察 Kernel 开始和结束时间、是否存在空洞、是否存在等待阶段，以及尾部阶段是否变短变稀疏。 |

## Compute Workload Analysis

**Compute Workload Analysis** 用于对 Streaming Multiprocessor，也就是 SM 的计算资源进行更细粒度分析。它关注的是 Kernel 在 SM 上如何执行指令、如何使用计算流水线，以及调度器是否能够持续发射指令。

这部分通常用于分析：

- **IPC 表现**：每个周期实际执行或发射了多少指令。
- **计算流水线利用率**：FMA、ALU、FP64、Tensor Core 等路径是否繁忙。
- **指令发射效率**：调度器的 issue slot 是否被充分利用。
- **SM 活跃程度**：SM 是否持续处于 busy 状态。
- **潜在计算瓶颈**：某条流水线过忙时，可能成为整个 Kernel 的主要限制。

## Compute Workload 顶部关键指标

顶部关键指标用于快速判断 Kernel 的整体指令执行效率和 SM 忙碌程度。

| 英文指标 | 中文名称 | 单位 | 详细说明 |
| --- | --- | --- | --- |
| Executed IPC Elapsed | 平均周期执行 IPC | inst/cycle | 以整个 Kernel elapsed cycles 为分母，统计平均每周期执行的指令数。它包含不活跃周期，因此更接近全局平均效率。 |
| Executed IPC Active | 活跃周期执行 IPC | inst/cycle | 只在 SM 活跃周期内统计平均每周期执行的指令数，用于观察 SM 真正工作期间的执行效率。 |
| Issued IPC Active | 活跃周期发射 IPC | inst/cycle | 只在 SM 活跃周期内统计平均每周期发射的指令数，用于观察调度器 issue 能力是否被充分利用。 |
| SM Busy | SM 忙碌率 | % | 表示 SM 处于忙碌状态的比例。该指标越高，说明 SM 越少处于空闲状态。 |
| Issue Slots Busy | 指令发射槽忙碌率 | % | 表示调度器 issue slots 被使用的比例。该值偏低时，可能说明 warp stall、依赖、访存等待或并行度不足限制了指令发射。 |

## Compute Workload 诊断结果

Nsight Compute 会在该页面给出一些自动诊断信息。这些建议不能直接替代人工分析，但可以帮助快速定位优先观察的区域。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Low Utilization | 低利用率 | 表示 NCU 判断某些计算资源利用率偏低，可能存在未充分利用 SM 或流水线的情况。 | 需要结合 occupancy、stall 原因、访存指标一起判断低利用率的原因。 |
| Est. Local Speedup | 预计局部加速空间 | Estimated Local Speedup，用于估算如果消除当前局部瓶颈，理论上还可能获得多少性能提升。 | 这是局部估计，不等于整体一定能提升这么多；优化后可能暴露新的瓶颈。 |
| Guidance | 优化建议 | NCU 自动给出的优化方向，例如提高最繁忙流水线的利用率。 | 建议只作为线索，最终仍要结合 Kernel 代码结构和实际瓶颈判断。 |

## Compute Workload 关键性能指标

Key Performance Indicators 会列出当前页面认为最关键的性能指标。它通常用于把注意力集中到最值得分析的 metric 上。

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active | FMA 流水线活跃周期占理论峰值比例 | 表示 FMA pipeline 在活跃周期内相对于 sustained peak 的利用程度。该指标高时，说明浮点乘加路径较忙，可能接近计算瓶颈。 |

## Pipe Utilization (% of active cycles)

**Pipe Utilization (% of active cycles)** 表示在 SM 活跃期间，各计算流水线真正工作的时间占比。它回答的问题是：在 SM 已经 active 的周期里，某条 pipeline 有多少时间处于工作状态。

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| FMA | 浮点乘加流水线 | 执行浮点乘加类指令，是 FP32 计算密集 Kernel 中最常见的计算路径之一。 |
| ALU | 算术逻辑流水线 | 执行整数运算、逻辑运算、部分地址或控制相关计算。 |
| FP64 | 双精度流水线 | 执行普通双精度浮点计算。没有 FP64 指令时通常很低或为 0。 |
| Tensor (All) | 全部 Tensor Core 流水线 | 统计所有 Tensor Core 相关执行路径的总体活跃情况。 |
| Tensor (FP) | 浮点 Tensor Core 流水线 | 统计浮点 Tensor Core 指令路径，例如 FP16/BF16/TF32 等矩阵计算。 |
| Tensor (INT) | 整数 Tensor Core 流水线 | 统计整数 Tensor Core 指令路径，例如 INT8 矩阵计算。 |

## Pipe Utilization (% of peak instructions executed)

**Pipe Utilization (% of peak instructions executed)** 表示当前流水线相对于理论最大指令执行能力，实际使用了多少。它回答的问题是：这条 pipeline 的指令吞吐是否接近硬件峰值。

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| LSU | Load Store Unit，加载存储单元 | 执行 load/store 相关指令。该项高时，说明访存指令执行路径较忙。 |
| FMA | 浮点乘加流水线 | 执行普通浮点乘加指令，常见于 FP32 数值计算。 |
| ALU | 算术逻辑流水线 | 执行整数、逻辑、基础算术等指令。 |
| ADU | Address Unit，地址计算单元 | 执行地址计算相关指令，复杂索引、非连续访问可能提高该项压力。 |
| CBU | Control Branch Unit，控制/分支单元 | 执行控制流、分支和谓词相关指令。 |
| Uniform | Uniform 流水线 | 执行 warp 内一致的 uniform 指令。 |
| FMA (FP16) | 半精度浮点乘加流水线 | 执行 FP16 相关的浮点乘加指令。 |
| FP64 (DMMA) | 双精度矩阵乘加流水线 | 执行 double precision matrix multiply accumulate 相关路径。 |
| FP64 (FP64) | 普通双精度流水线 | 执行常规 FP64 双精度浮点指令。 |
| TEX | Texture 流水线 | 执行 texture 或 L1TEX 相关指令路径。 |
| Tensor (FP) | 浮点 Tensor Core 流水线 | 执行浮点 Tensor Core 矩阵指令。 |
| Tensor (INT) | 整数 Tensor Core 流水线 | 执行整数 Tensor Core 矩阵指令。 |
| XU | 特殊功能执行单元 | 执行特殊功能或特定硬件指令路径。 |

## 流水线类别对照

| 类别 | 包含流水线 | 说明 |
| --- | --- | --- |
| 访存相关 | LSU、TEX | 主要对应 load/store、texture、L1TEX 相关执行路径。 |
| 浮点计算 | FMA、FMA (FP16)、FP64 | 主要对应普通浮点、半精度浮点和双精度计算。 |
| 整数计算 | ALU | 主要对应整数算术、逻辑运算和部分基础计算。 |
| Tensor Core | Tensor (FP)、Tensor (INT) | 主要对应矩阵乘加等 Tensor Core 指令。 |
| 地址计算 | ADU | 主要对应地址生成、索引计算和访存地址相关操作。 |
| 分支控制 | CBU | 主要对应分支、控制流和谓词相关指令。 |
| 特殊执行单元 | XU | 主要对应特殊函数或特定硬件执行路径。 |
| Uniform 执行 | Uniform | 主要对应 warp 内一致的 uniform 指令路径。 |

## Compute Workload 页面结构总结

| 页面区域 | 作用 |
| --- | --- |
| IPC Statistics | 查看指令执行效率，重点关注 executed IPC 和 issued IPC。 |
| SM Busy | 查看 SM 活跃程度，判断 SM 是否有足够工作可执行。 |
| Issue Slots Busy | 查看调度器 issue slot 利用率，判断指令发射是否充分。 |
| Low Utilization | 查看 NCU 自动性能诊断，快速发现低利用率问题。 |
| Key Performance Indicators | 查看当前最关键性能指标，确定优先分析的 metric。 |
| Pipe Utilization (% active cycles) | 查看各流水线在 SM 活跃周期内的工作时间占比。 |
| Pipe Utilization (% peak instructions) | 查看各流水线相对于理论峰值指令执行能力的利用率。 |

## Compute Workload 英文标题对照

| 英文标题 | 中文名称 |
| --- | --- |
| Compute Workload Analysis | 计算工作负载分析 |
| Executed IPC Elapsed | 平均周期执行 IPC |
| Executed IPC Active | 活跃周期执行 IPC |
| Issued IPC Active | 活跃周期发射 IPC |
| SM Busy | SM 忙碌率 |
| Issue Slots Busy | 指令发射槽忙碌率 |
| Low Utilization | 低利用率 |
| Est. Local Speedup | 预计局部加速空间 |
| Pipe Utilization | 流水线利用率 |
| Guidance | 优化建议 |
| Peak Instructions Executed | 理论峰值指令执行能力 |

## Memory Workload Analysis

**Memory Workload Analysis** 用于对 GPU 内存系统进行详细分析。相比 Speed Of Light 中的总体吞吐率，这个页面会进一步拆开 shared memory、L1/TEX、L2、DRAM、global memory、local memory 等不同层次和访问类型。

这部分通常用于分析：

- **内存吞吐量**：Kernel 是否对内存系统形成较高压力。
- **内存子系统忙碌程度**：Mem Busy、Mem Pipes Busy 是否说明内存路径已经接近饱和。
- **L1/L2 Cache 命中率**：访问是否主要命中缓存，还是频繁下探到显存。
- **带宽利用率**：是否已经接近硬件可提供的最大带宽。
- **Shared Memory 使用情况**：是否存在 bank conflict、请求过多或 shared load/store 压力。
- **Global Memory 访问情况**：全局内存 load/store 是否合并良好，sector/request 是否合理。
- **L1/TEX、L2、DRAM 工作情况**：定位瓶颈到底发生在一级缓存、二级缓存还是显存。

内存可能成为 Kernel 性能瓶颈的常见原因包括：

- **Mem Busy 较高**：内存子系统长期处于忙碌状态。
- **Max Bandwidth 较高**：访问已经接近理论带宽上限。
- **Mem Pipes Busy 较高**：内存流水线接近饱和，即使带宽未满，也可能被内部 pipeline 限制。

## Memory Workload 顶部关键指标

顶部关键指标用于快速判断当前 Kernel 的内存压力来自哪里。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Memory Throughput | 内存吞吐率 | 表示整体内存系统相对于理论峰值的吞吐利用情况。 | 用于判断 Kernel 是否存在较强的内存压力，但还需要结合 L1、L2、DRAM 细分项定位。 |
| L1/TEX Hit Rate | L1/TEX 缓存命中率 | 表示访问 L1/TEX Cache 时命中的比例。 | 命中率高通常说明一级缓存局部性较好；命中率低可能导致更多 L2 或 DRAM 请求。 |
| L2 Hit Rate | L2 缓存命中率 | 表示访问 L2 Cache 时命中的比例。 | L2 命中率高可以减少 DRAM 压力；命中率低通常意味着更多显存访问。 |
| L2 Compression Success Rate | L2 压缩成功率 | 表示 L2 Cache Compression 成功压缩数据的比例。 | 压缩成功率高时，可能降低 L2/DRAM 带宽消耗。 |
| Mem Busy | 内存系统忙碌率 | 表示内存子系统处于忙碌状态的比例。 | 如果该值高，说明内存系统可能正在限制 Kernel。 |
| Max Bandwidth | 峰值带宽利用率 | 表示当前内存访问对最大可用带宽的利用程度。 | 如果接近峰值，说明 Kernel 可能受带宽上限限制。 |
| Mem Pipes Busy | 内存流水线忙碌率 | 表示内存 pipeline 的忙碌程度。 | 如果该值高但带宽不高，可能说明瓶颈在内部内存流水线，而不是外部带宽。 |
| L2 Compression Ratio | L2 压缩比 | 表示 L2 压缩机制带来的数据压缩比例。 | 用于观察硬件压缩是否有效降低实际传输流量。 |

## Memory Chart

**Memory Chart** 展示一次内存访问在 GPU 内存层次结构中的流动过程。它可以帮助把 NCU 表格中的指标和实际硬件路径对应起来。

| 图中节点 | 中文名称 | 对应内容 | 说明 |
| --- | --- | --- | --- |
| Kernel | Kernel 执行单元 | CUDA Kernel 中的线程、warp、CTA | 所有访存请求的来源，包括 global、shared、local、texture 等访问。 |
| Global | 全局内存访问 | global memory | 对应常见的全局内存 load/store，例如 `A[row * lda + col]`。 |
| Local | 局部内存访问 | local memory | 通常对应寄存器溢出后的本地内存访问，物理上通常仍会落到显存路径。 |
| Texture | 纹理内存访问 | texture object、texture fetch | 对应纹理对象或纹理读取路径。 |
| Surface | 表面内存访问 | surface object | 对应 CUDA surface object 相关访问。 |
| Load Global Store Shared | 全局内存加载到共享内存 | global memory 到 shared memory 的搬运 | 常见于 `cp.async` 或 gmem 到 smem 的数据搬运路径。 |
| Shared | 共享内存访问 | `__shared__` | 对应 Kernel 中对 shared memory 的显式读写。 |
| Shared Memory | 共享内存模块 | SM 内部 shared memory 硬件 | 表示 SM 内部的共享内存硬件资源。 |
| L1/TEX Cache | 一级缓存 / 纹理缓存 | L1 Cache 与 Texture Cache | 负责 Global 到 L1、Texture 等缓存访问路径。 |
| L2 Cache | 二级缓存 | GPU 所有 SM 共享的 L2 Cache | 连接 SM/L1 与 DRAM，是全局共享缓存层。 |
| Device Memory | 设备显存 | GDDR、HBM | 对应 GPU device memory，也是全局内存访问最终可能到达的位置。 |
| System Memory | 系统内存 | CPU 主存 | 对应 host/system memory，统一内存或特定访问可能涉及该路径。 |
| Peer Memory | Peer GPU 内存 | NVLink、PCIe P2P | 对应多 GPU 场景下访问其他 GPU 的显存。 |
| L2 Compression | L2 缓存压缩 | Cache Compression | Ampere 及以后架构支持的压缩机制，用于降低 L2/DRAM 带宽消耗。 |

## Shared Memory 统计

Shared Memory 统计用于观察 shared memory 的读写请求、指令数量、wavefront、峰值利用率和 bank conflict。分析 CUDA tiling、GEMM、transpose、reduction 等 Kernel 时，这部分非常关键。

| 访问类型 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Shared Load | Shared Memory 读取 | 从 shared memory 读取数据的操作。 |
| Shared Load Matrix | Matrix Load 读取 | 面向矩阵操作的 shared memory load，常见于 Tensor Core 相关数据加载路径。 |
| Shared Store | Shared Memory 写入 | 向 shared memory 写入数据的操作。 |
| Shared Store From Global Load | 从 Global Load 直接写入 Shared | 表示 global 到 shared 的搬运路径，例如 `cp.async` 或类似 gmem 到 smem 的加载。 |
| Shared Atomic | Shared Atomic 操作 | 对 shared memory 执行 atomic 操作。 |
| Other | 其它 Shared 操作 | 无法归入上述类别的 shared memory 相关操作。 |
| Total | 总计 | 所有 shared memory 访问类型的汇总。 |

### Shared Memory 统计列

| 列名 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Instructions | 指令数 | 执行的 shared memory 相关指令数量。 |
| Requests | 请求数 | 由指令产生的 shared memory 访问请求数量。 |
| Wavefronts | 波前数 | shared memory 内部处理访问请求的 wavefront 数量。 |
| % Peak | 峰值利用率百分比 | 相对于 shared memory 理论峰值能力的利用率。 |
| Bank Conflicts | Bank 冲突次数 | shared memory bank conflict 的次数或强度。该值高时，可能需要调整布局或访问模式。 |

## L1/TEX Cache 统计

L1/TEX Cache 统计用于观察一级缓存与纹理缓存中的访问类型、命中率、sector 数量和访问字节数。它能帮助判断 global/local/surface/atomic 等请求是否在 L1/TEX 路径上形成压力。

| 访问类型 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Local Load | Local Memory 读取 | 读取 local memory，常见于寄存器溢出或线程私有数组无法放入寄存器的情况。 |
| Global Load | Global Memory 读取 | 从 global memory 读取数据，是最常见的全局访存类型。 |
| Global Load To Shared Store (access) | Global 到 Shared 访问 | 表示 global memory 数据被加载并写入 shared memory 的访问路径。 |
| Global Load To Shared Store (bypass) | Global 到 Shared 旁路访问 | 表示 global 到 shared 的搬运绕过部分缓存路径，常见于特定异步拷贝或旁路策略。 |
| Surface Load | Surface 读取 | 从 surface object 读取数据。 |
| Global Store | Global 写入 | 向 global memory 写入数据。 |
| Local Store | Local 写入 | 向 local memory 写入数据。 |
| Surface Store | Surface 写入 | 向 surface object 写入数据。 |
| Global Reduction | Global 归约 | 对 global memory 执行 reduction 类操作。 |
| DSMEM Reduction | DSMEM 归约 | 对 distributed shared memory 或相关共享内存路径执行 reduction 操作。 |
| Surface Reduction | Surface 归约 | 对 surface memory 执行 reduction 类操作。 |
| Global Atomic ALU | Global Atomic 运算 | 对 global memory 执行 atomic add、atomic min/max 等 ALU 类原子操作。 |
| Global Atomic CAS | Global Atomic CAS | 对 global memory 执行 compare-and-swap 类型原子操作。 |
| Surface Atomic ALU | Surface Atomic 运算 | 对 surface memory 执行 ALU 类原子操作。 |
| Surface Atomic CAS | Surface Atomic CAS | 对 surface memory 执行 compare-and-swap 类型原子操作。 |

### L1/TEX Cache 统计列

| 列名 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Instructions | 指令数 | 产生 L1/TEX 访问的指令数量。 |
| Requests | 请求数 | L1/TEX 收到的访问请求数量。 |
| Wavefronts | 波前数 | L1/TEX 内部处理访问请求时形成的 wavefront 数量。 |
| % Peak | 峰值利用率 | 相对于 L1/TEX 理论峰值处理能力的利用率。 |
| Sectors | Cache Sector 数量 | L1/TEX 实际访问的 cache sector 数量。 |
| Sectors/Req | 每请求 Sector 数 | 平均每个请求产生多少 sector，常用于观察访存合并程度。 |
| Hit Rate | 命中率 | L1/TEX 访问命中的比例。 |
| Bytes | 访问字节数 | L1/TEX 路径实际处理的数据量。 |

## L2 Cache 统计

L2 Cache 统计用于观察二级缓存的访问行为，包括来自 L1/TEX 的读写、atomic、ECC 流量、GPU 总流量、命中率、sector miss 和吞吐量。

| 访问类型 | 中文名称 | 详细说明 |
| --- | --- | --- |
| L1/TEX Load | 来自 L1/TEX 的读取 | L2 接收到来自 L1/TEX 的读请求。 |
| L1/TEX Store | 来自 L1/TEX 的写入 | L2 接收到来自 L1/TEX 的写请求。 |
| L1/TEX Atomic ALU | 来自 L1/TEX 的 Atomic 运算 | L2 处理来自 L1/TEX 的 ALU 类 atomic 请求。 |
| L1/TEX Atomic CAS | 来自 L1/TEX 的 Atomic CAS | L2 处理来自 L1/TEX 的 compare-and-swap atomic 请求。 |
| L1/TEX Total | 来自 L1/TEX 的总请求 | L1/TEX 到 L2 的所有请求汇总。 |
| ECC Total | ECC 相关流量 | 与 ECC 校验、纠错相关的 L2 流量。 |
| GPU Total | GPU 总流量 | GPU 侧经过 L2 的总请求或总数据流量。 |

### L2 Cache 统计列

| 列名 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Requests | 请求数 | L2 接收到的访问请求数量。 |
| Sectors | Cache Sector 数量 | L2 实际处理的 cache sector 数量。 |
| Sectors/Req | 每请求 Sector 数 | 平均每个 L2 请求对应的 sector 数量，用于观察访问粒度和合并情况。 |
| % Peak | 峰值利用率 | 相对于 L2 理论峰值处理能力的利用率。 |
| Hit Rate | 命中率 | L2 访问命中的比例。 |
| Bytes | 数据量（字节） | L2 路径处理的数据总量。 |
| Throughput | 吞吐量 | L2 单位时间内处理的数据量或请求量。 |
| Sector Misses to Device | 未命中并访问显存次数 | L2 sector miss 后继续访问 device memory 的次数或数量。 |
| Sector Misses | Sector 未命中次数 | L2 sector 未命中的总次数。 |

## Scheduler Statistics

**Scheduler Statistics** 用于统计 GPU Warp Scheduler 的工作情况。每个 SM 内部包含多个 warp scheduler，每个 scheduler 负责管理一组 warp，从中选择已经 ready 的 warp，并发射下一条指令。

这部分通常用于分析：

- **Warp 是否充足**：active warp 是否足够多，能不能覆盖访存和指令依赖延迟。
- **Warp 是否经常 stall**：active warp 多但 eligible warp 少，通常说明大量 warp 卡在依赖、访存或同步上。
- **Scheduler 是否空闲**：如果 no eligible 占比高，scheduler 有发射能力但没有 ready warp。
- **延迟隐藏是否充分**：足够多的 eligible warp 可以帮助 scheduler 在某些 warp 等待时切换到其他 warp 执行。

## Scheduler 顶部关键指标

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Active Warps Per Scheduler | 每个调度器的活跃 Warp 数 | 表示已经分配给 scheduler 管理、尚未结束执行的 warp 数量。它包含正在执行、stall 中、等待访存或等待同步的 warp。 | 如果 active warp 很低，说明 occupancy 或并行度可能不足。 |
| Eligible Warps Per Scheduler | 每个调度器的可调度 Warp 数 | 表示当前已经 ready、可以立即发射下一条指令的 warp 数量。 | 如果 active warp 高但 eligible warp 低，通常说明大部分 warp 正在等待依赖或访存。 |
| Issued Warp Per Scheduler | 每个调度器实际发射 Warp 数 | 表示每周期真正被 scheduler 选中并发射指令的 warp 数量。 | 该值越接近 scheduler 发射上限，说明 issue 能力利用越充分。 |
| No Eligible | 无可调度 Warp 周期占比 | 表示当前周期内没有任何 warp 可供调度的比例。 | 该值高时，issue slot 会被浪费，常见原因是访存延迟、同步等待或依赖链过长。 |
| One or More Eligible | 至少存在一个可调度 Warp 周期占比 | 表示当前周期至少有一个 warp 可以发射下一条指令的比例。 | 该值高说明 scheduler 经常有可选 warp，有利于隐藏延迟。 |

## 调度器工作流程

```mermaid
flowchart LR
    A[Warp Pool] --> B[Active Warp]
    B --> C[Eligible Warp]
    C --> D[Scheduler 选择]
    D --> E[Issued Warp]
    E --> F[执行指令]
```

这个流程可以理解为：warp 先进入 scheduler 管理范围成为 active warp；当它的依赖、访存、同步条件都满足后，才会变成 eligible warp；scheduler 再从 eligible warp 中选择一个发射指令。


## Scheduler 关键性能指标

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| smsp__issue_active.avg.per_cycle_active | 每活跃周期平均发射指令数 | 表示 SMSP 在活跃周期内平均每周期发射指令的数量。该指标越高，通常说明 scheduler 发射效率越好。 |

## Warps Per Scheduler

**Warps Per Scheduler** 图表展示 scheduler 所管理 warp 的不同状态。它能帮助判断问题是 warp 数量不足，还是 warp 数量足够但多数不可调度。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| GPU Maximum Warps Per Scheduler | 调度器支持的最大 Warp 数 | 硬件层面每个 scheduler 最多支持的 warp 数量。 | 这是硬件上限，实际可驻留 warp 往往受资源限制。 |
| Theoretical Warps Per Scheduler | 理论 Warp 数 | 理论上能够驻留在 scheduler 中的 warp 数量。 | 受 block size、occupancy、registers、shared memory 等因素限制。 |
| Active Warps Per Scheduler | 活跃 Warp 数 | 当前真正驻留在 scheduler 中、尚未结束的 warp 数量。 | 如果低于 theoretical warps，可能存在资源配置或 tail effect 影响。 |
| Eligible Warps Per Scheduler | 可调度 Warp 数 | 当前 ready、能够立即发射指令的 warp 数量。 | 这是判断延迟隐藏能力的核心指标之一。 |
| Issued Warp Per Scheduler | 实际发射 Warp 数 | 最终被 scheduler 选中并执行指令的 warp 数量。 | 反映 scheduler 实际产出，通常和 issue slot utilization 一起观察。 |

## Warp 生命周期

```mermaid
flowchart LR
    A[创建 Warp] --> B[Active Warp]
    B --> C[等待依赖 / 访存 / 同步]
    C --> D[Eligible Warp]
    D --> E[Scheduler 选择]
    E --> F[Issued Warp]
    F --> G[执行指令]
```

这里最关键的区分是 **Active Warp 不等于 Eligible Warp**。active 只表示 warp 已经驻留并由 scheduler 管理；eligible 才表示 warp 当前已经准备好，可以被立即发射。

## Scheduler Statistics 指标对照

| 英文名称 | 中文名称 | 说明 |
| --- | --- | --- |
| Scheduler | 调度器 | SM 内部负责选择 ready warp 并发射指令的硬件单元。 |
| Warp Pool | Warp 池 | scheduler 可管理或可选择的 warp 集合。 |
| Active Warp | 活跃 Warp | 已经驻留在 SM 中、尚未结束的 warp。 |
| Eligible Warp | 可调度 Warp | 已经 ready、可以立即发射下一条指令的 warp。 |
| Issued Warp | 已发射 Warp | 被 scheduler 选中并发射指令的 warp。 |
| Issue Slot | 发射槽 | scheduler 每周期可用于发射指令的硬件槽位。 |
| Warp Stall | Warp 阻塞 | warp 因依赖、访存、同步等原因暂时不能发射指令。 |
| Latency Hiding | 延迟隐藏 | 通过切换到其他 ready warp 执行来覆盖访存或指令延迟。 |
| No Eligible | 无可调度 Warp | 当前周期没有 ready warp 可供 scheduler 发射。 |
| One or More Eligible | 至少存在一个可调度 Warp | 当前周期至少有一个 ready warp 可以发射。 |
| Issue Slot Utilization | 发射槽利用率 | issue slot 被实际使用的比例。 |
| Warps Per Scheduler | 每调度器 Warp 统计 | 展示每个 scheduler 中不同状态 warp 的数量。 |
| GPU Maximum Warps Per Scheduler | 调度器最大 Warp 容量 | 硬件支持的每 scheduler 最大 warp 数。 |
| Theoretical Warps Per Scheduler | 理论 Warp 容量 | 受资源限制后理论可驻留的每 scheduler warp 数。 |
| Active Warps Per Scheduler | 活跃 Warp 数 | 每个 scheduler 中已驻留且未结束的 warp 数。 |
| Eligible Warps Per Scheduler | 可调度 Warp 数 | 每个 scheduler 中 ready 的 warp 数。 |
| Issued Warp Per Scheduler | 实际发射 Warp 数 | 每个 scheduler 实际发射的 warp 数。 |

## Scheduler Statistics 页面结构总结

| 页面区域 | 作用 |
| --- | --- |
| Scheduler Metrics | 查看调度器总体统计，包括 active、eligible、issued warp 等核心指标。 |
| Issue Slot Utilization | 分析发射槽利用率，判断 scheduler 发射能力是否被充分使用。 |
| Est. Local Speedup | 查看当前调度瓶颈对应的理论局部优化空间。 |
| Key Performance Indicator | 查看核心调度指标，例如每活跃周期平均发射指令数。 |
| Guidance | 查看 Nsight Compute 自动生成的优化建议。 |
| Warps Per Scheduler | 查看不同状态 warp 的分布，判断是 warp 不足还是 warp 不 ready。 |

## Warp State Statistics

**Warp State Statistics** 用于分析 Kernel 执行期间 warp 所处的各种状态。它回答的问题是：一个 warp 为什么没有继续发射下一条指令。

Warp 状态通常用于描述：

- **Warp 是否已经 ready**：是否准备好发射下一条指令。
- **Warp 是否被阻塞**：是否因为依赖、访存、同步或资源限制而 stall。
- **Warp 正在等待什么资源**：例如 global memory、shared memory、barrier、math pipe、dispatch unit 等。

这个页面是分析 GPU 性能瓶颈最重要的页面之一。Scheduler Statistics 告诉我们 scheduler 有没有 ready warp，而 Warp State Statistics 进一步解释 ready warp 少的原因。

## Warp State 顶部关键指标

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Warp Cycles Per Issued Instruction | 每条已发射指令对应的 Warp 周期数 | 表示平均发射一条指令需要消耗多少 warp cycles。 | 数值越高，说明每次成功 issue 之间等待越多，可能存在较严重 stall。 |
| Warp Cycles Per Executed Instruction | 每条已执行指令对应的 Warp 周期数 | 表示平均执行一条指令需要消耗多少 warp cycles。 | 可用于观察实际执行效率，通常需要结合 issued instruction 一起看。 |
| Avg. Active Threads Per Warp | 每个 Warp 平均活跃线程数 | 表示 warp 中平均有多少线程处于 active 状态。 | 如果明显低于 32，可能存在分支发散、边界判断或线程掩码导致的利用率下降。 |
| Avg. Not Predicated Off Threads Per Warp | 每个 Warp 平均未被 Predicate 屏蔽线程数 | 表示 warp 中没有被 predicate off 的线程数量。 | 如果该值低，说明虽然线程 active，但很多线程被谓词屏蔽，没有真正参与指令执行。 |

## Warp Stall 分析结果

Nsight Compute 会在 Warp State 页面给出主要 stall 原因和预计优化空间。它通常会指出当前最值得关注的阻塞类型。

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Mio Throttle Stalls | MIO 流水线限流阻塞 | 表示 warp 正在等待 MIO，也就是 Memory Input/Output 相关队列或流水线空闲。MIO 相关指令可能包括 shared memory、特殊数学指令、动态分支和部分访存指令。 | 如果该项突出，通常需要检查 shared memory 访问、特殊指令密度、MIO 指令比例和相关 pipeline 压力。 |
| Est. Speedup | 预计加速空间 | Nsight Compute 估算如果减少当前主要 stall，理论上还能获得多少性能提升。 | 这是局部估计，不能直接等价为最终加速比；消除一个 stall 后可能暴露新的瓶颈。 |
| Guidance | 优化建议 | Nsight Compute 自动生成的性能优化建议。 | 例如 `Increase the average number of instructions issued per cycle`，可以理解为提高每周期发射的指令数量。 |

## Warp State 关键性能指标

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| smsp__issue_active.avg.per_cycle_active | 每活跃周期平均发射指令数 | 表示 SMSP 在活跃周期内平均每周期发射指令的数量。该指标越高，说明 scheduler 发射效率通常越好。 |
| smsp__average_mio_throttle | 平均 MIO 限流阻塞周期 | 表示 warp 因 MIO throttle 平均阻塞的周期数。该值高时，说明 MIO 相关资源可能成为瓶颈。 |

## Warp Stall

**Warp Stall** 用于分析 warp 为什么没有继续执行下一条指令。它把 warp 无法 issue 的原因拆成不同类别，例如等待访存、等待同步、等待 pipeline、没有被 scheduler 选中等。

| 英文区域 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Warp State (All Cycles) | Warp 状态统计（全部周期） | 统计 warp 在各种状态下平均消耗了多少周期，单位通常是 cycles per instruction。 |
| Warp Stall Reasons | Warp 阻塞原因分类 | 按阻塞类型拆解 warp stall，例如 scoreboard、barrier、MIO throttle、LG throttle 等。 |

## Warp 状态说明

| 英文状态 | 中文名称 | 详细说明 | 常见原因或观察重点 |
| --- | --- | --- | --- |
| Stall MIO Throttle | MIO 流水线限流阻塞 | MIO 单元或相关队列接近饱和，warp 无法继续发射 MIO 相关指令。 | 常见于 shared memory 压力较高、特殊函数较多、部分访存或动态分支路径繁忙。 |
| Stall Not Selected | 未被调度器选中 | warp 已经 ready，也就是 eligible，但 scheduler 选择了其他 warp。 | 该项不一定是坏事，通常说明 scheduler 有多个 ready warp 可选。 |
| Stall Barrier | 同步屏障阻塞 | warp 正在等待 barrier 完成。 | 常见于 `__syncthreads()`、`cuda::barrier` 或 block 内同步密集的 Kernel。 |
| Stall Long Scoreboard | 长延迟依赖阻塞 | warp 正在等待高延迟操作返回。 | 通常对应 global memory、L2、DRAM 等高延迟访存依赖。 |
| Stall Wait | 等待状态 | warp 正在等待其他资源或事件。 | 需要结合上下文和其他 stall 指标判断具体来源。 |
| Selected | 已被调度执行 | warp 被 scheduler 选中，正在发射指令。 | 这是有效执行状态，通常希望 selected 占比更高。 |
| Stall Short Scoreboard | 短延迟依赖阻塞 | warp 正在等待较短延迟的依赖返回。 | 常见于寄存器依赖、shared memory 依赖、L1 Cache 依赖等。 |
| Stall Dispatch Stall | 发射器阻塞 | dispatch 单元无法继续发射当前指令。 | 可能与后端 pipeline、发射路径或资源冲突有关。 |
| Stall Branch Resolving | 分支解析阻塞 | warp 正在等待 branch target 或 predicate 计算完成。 | 分支复杂、谓词计算或控制流较多时可能升高。 |
| Stall No Instruction | 无可执行指令 | 当前 warp 没有新的指令可执行。 | 可能与指令获取、控制流或执行流状态有关。 |
| Stall Math Pipe Throttle | 数学流水线限流 | FMA、ALU 等数学流水线繁忙，无法接收更多指令。 | 计算密集型 Kernel 中可能出现，说明某些 math pipe 接近饱和。 |
| Stall IMC Miss | 指令缓存未命中 | Instruction Cache Miss 导致 warp 等待指令获取。 | 指令 footprint 大、代码路径复杂时可能出现。 |
| Stall Drain | 流水线排空等待 | Kernel 结束前等待流水线中已有工作排空。 | 常见于 Kernel 尾部，通常要结合占比判断是否值得优化。 |
| Stall LG Throttle | Local/Global 访存限流 | local/global memory 访问过多，load/store 队列或相关路径已满。 | 常见于全局访存压力大、访存不合并、寄存器溢出到 local memory 等情况。 |
| Stall Tex Throttle | 纹理流水线限流 | Texture pipeline 或 L1TEX 相关路径繁忙。 | 使用 texture fetch 或 L1TEX 路径压力较高时需要关注。 |
| Stall Misc | 其它原因阻塞 | 无法归类到其他具体类型的阻塞。 | 如果占比很高，需要结合源码、指令和其他 profiler 页面进一步定位。 |
| Stall Membar | 内存屏障阻塞 | warp 正在等待 memory barrier 完成。 | 常见于 `__threadfence()`、`membar` 或需要严格内存可见性的场景。 |
| Stall Sleeping | 休眠状态 | warp 暂时不参与调度。 | 通常表示 warp 被挂起或处于不活跃等待状态。 |

## Warp 生命周期示意

```mermaid
flowchart LR
    A[Warp 创建] --> B[Ready]
    B --> C[Selected]
    C --> D[执行指令]
    D --> E[依赖产生]
    E --> F[Stall]
    F --> G[依赖满足]
    G --> B
```

这个循环说明 warp 的执行不是连续不断的。每次执行指令后，都可能因为数据依赖、访存、同步或资源限制进入 stall；当等待条件满足后，warp 才重新回到 ready 状态，等待 scheduler 再次选择。

## Warp State 指标对照

| 英文名称 | 中文名称 | 说明 |
| --- | --- | --- |
| Warp State Statistics | Warp 状态统计 | 分析 Kernel 执行期间 warp 处于哪些状态。 |
| Warp Stall | Warp 阻塞 | warp 无法继续发射下一条指令的状态。 |
| Stall MIO Throttle | MIO 限流阻塞 | 等待 MIO 相关队列或流水线空闲。 |
| Stall Not Selected | 未被选中 | warp 已 ready，但 scheduler 选择了其他 warp。 |
| Stall Barrier | 同步屏障阻塞 | 等待 `__syncthreads()`、`cuda::barrier` 等同步完成。 |
| Stall Long Scoreboard | 长延迟依赖阻塞 | 等待 global memory、L2、DRAM 等长延迟操作返回。 |
| Stall Short Scoreboard | 短延迟依赖阻塞 | 等待寄存器、shared memory、L1 等较短延迟依赖返回。 |
| Stall Dispatch Stall | 发射器阻塞 | dispatch 单元无法继续发射。 |
| Stall Branch Resolving | 分支解析阻塞 | 等待 branch target 或 predicate 解析完成。 |
| Stall No Instruction | 无指令可执行 | 当前 warp 没有新的指令可发射。 |
| Stall Math Pipe Throttle | 数学流水线限流 | FMA、ALU 等数学流水线繁忙。 |
| Stall IMC Miss | 指令缓存未命中 | instruction cache miss 导致等待取指。 |
| Stall Drain | 流水线排空 | Kernel 尾部等待流水线排空。 |
| Stall LG Throttle | Global 访存限流 | local/global memory 访问路径或队列繁忙。 |
| Stall Tex Throttle | Texture 流水线限流 | texture 或 L1TEX 路径繁忙。 |
| Stall Membar | 内存屏障阻塞 | 等待 `__threadfence()` 或 `membar` 完成。 |
| Stall Sleeping | Warp 休眠 | warp 暂时不参与调度。 |

## Warp State 页面结构总结

| 页面区域 | 作用 |
| --- | --- |
| Warp Cycles Statistics | 查看 warp 执行效率，例如每条 issued/executed instruction 对应多少 warp cycles。 |
| Active Threads Statistics | 查看 warp 内线程活跃度，判断分支发散和 predicate off 的影响。 |
| Stall Analysis | 查看 Nsight Compute 自动识别出的主要阻塞原因。 |
| Key Performance Indicators | 查看关键阻塞指标，例如 issue active 和 MIO throttle。 |
| Guidance | 查看自动优化建议，作为进一步分析的线索。 |
| Warp State Chart | 查看各种 warp 状态占比或 cycles per instruction。 |
| Warp Stall Reasons | 查看 warp 阻塞原因分类，定位等待访存、同步、pipeline 还是调度问题。 |

## Instruction Statistics

**Instruction Statistics** 用于统计 Kernel 实际执行的底层 SASS 汇编指令。它展示了指令总数、发射指令数、每个 scheduler 平均指令数，以及不同 opcode 在整个 Kernel 中的占比。

这部分通常用于分析：

- **指令混合类型**：Kernel 主要由浮点计算、访存、地址计算、分支还是同步指令组成。
- **执行流水线依赖**：如果 opcode 类型很集中，说明 Kernel 可能高度依赖少数 pipeline。
- **并行执行机会**：更丰富的指令组合有时有助于不同 pipeline 并行工作，从而隐藏部分延迟。
- **Issued 与 Executed 差异**：issued instructions 和 executed instructions 的统计口径不同，在系统指令或特定等待周期存在时可能出现差异。

## Instruction Statistics 顶部指标

| 英文指标 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Executed Instructions | 已执行指令数 | Kernel 实际执行完成的 SASS 指令数量。 |
| Issued Instructions | 已发射指令数 | Warp scheduler 发射出去的 SASS 指令数量。 |
| Avg. Executed Instructions Per Scheduler | 每个调度器平均执行指令数 | 平均每个 scheduler 实际执行的指令数量。 |
| Avg. Issued Instructions Per Scheduler | 每个调度器平均发射指令数 | 平均每个 scheduler 发射的指令数量。 |

## Executed Instruction Mix

**Executed Instruction Mix** 展示不同 opcode 的执行数量。它用于识别 Kernel 的指令组成，而不是直接判断最终瓶颈。

| Opcode | 中文名称 | 说明 |
| --- | --- | --- |
| FFMA | 浮点融合乘加指令 | 常见于 FP32 GEMM、stencil、向量计算等计算密集 Kernel。 |
| LDS | Shared Memory Load | 表示从 shared memory 读取数据。 |
| IADD3 | 三操作数整数加法 | 常用于地址计算、循环变量更新、索引计算等整数路径。 |
| ISETP | 整数比较并设置谓词 | 常用于边界判断、循环判断、条件分支和 predicate 生成。 |
| BRA | 分支跳转指令 | 表示控制流跳转，例如循环或条件分支。 |
| MOV | 数据移动指令 | 用于寄存器移动、常量加载或中间值传递。 |
| IMAD | 整数乘加指令 | 常用于复杂地址计算，例如二维索引展开、stride 计算。 |
| LDG | Global Memory Load | 表示从 global memory 读取数据。 |
| LOP3 | 三输入逻辑操作 | 用于 bit-level 逻辑计算、条件处理或编译器生成的布尔逻辑。 |
| BSYNC | 分支同步相关指令 | 与分支同步或控制流收敛相关。 |
| BSSY | 分支同步设置指令 | 用于设置同步点，帮助处理分支控制流。 |
| STS | Shared Memory Store | 表示向 shared memory 写入数据，常见于 global 到 shared 的 staging 阶段。 |
| SEL | 选择指令 | 根据 predicate 在两个值之间选择，常用于条件表达式。 |
| BAR | Barrier 同步指令 | 对应 block 内同步，例如 `__syncthreads()` 或相关 barrier 操作。 |
| IMNMX | 整数最小/最大指令 | 用于整数 min/max 计算，常见于边界裁剪或索引限制。 |
| CALL | 调用指令 | 表示函数调用或编译器未完全内联的调用路径。 |
| LEA | 地址生成指令 | 用于有效地址计算。 |
| S2R | 特殊寄存器读取 | 读取特殊寄存器，例如 thread/block/lane 相关寄存器。 |
| STG | Global Memory Store | 表示向 global memory 写入数据。 |
| SHF | 位移/拼接类指令 | 用于位操作、移位或数据拼接。 |
| FMUL | 浮点乘法指令 | 表示单独的浮点乘法操作。 |
| ULDC | Uniform Load Constant | 从 uniform/constant 路径加载数据。 |
| EXIT | 退出指令 | 表示线程或 warp 执行路径结束。 |
| USHF | Uniform Shift | uniform 路径上的位移类操作。 |
| UIADD3 | Uniform 三操作数整数加法 | uniform 路径上的整数加法，通常用于 warp 内一致的地址或控制计算。 |

## Instruction Statistics 阅读方法

- **先看主导 opcode**：如果 FFMA、HMMA、DMMA 等计算指令占主导，优先结合 Compute Workload 分析计算 pipeline；如果 LDG、STG、LDS、STS 占主导，优先结合 Memory Workload 和 Warp Stall 分析访存路径。
- **再看计算与访存比例**：FFMA 高且 LDS 也高，通常表示 shared memory tiling 后的计算阶段；LDG/STG 高则更像 global memory bandwidth 或 memory pipeline 压力。
- **注意同步和分支指令**：BAR、BRA、BSSY、BSYNC 虽然数量可能不高，但如果配合 Stall Barrier 或 Stall Branch Resolving 升高，仍可能影响性能。
- **结合 pipeline 利用率解释 opcode**：opcode mix 只告诉你执行了什么指令，不直接告诉你瓶颈在哪里。瓶颈还要结合 Pipe Utilization、Scheduler Statistics、Warp State 和 Memory Workload 判断。

## Instruction Statistics 指标对照

| 英文名称 | 中文名称 | 说明 |
| --- | --- | --- |
| Instruction Statistics | 指令统计 | 统计 Kernel 执行的底层 SASS 指令和 opcode mix。 |
| Executed Instructions | 已执行指令数 | 实际完成执行的 SASS 指令数量。 |
| Issued Instructions | 已发射指令数 | scheduler 发射出去的 SASS 指令数量。 |
| Avg. Executed Instructions Per Scheduler | 每调度器平均执行指令数 | 平均每个 scheduler 执行的指令数量。 |
| Avg. Issued Instructions Per Scheduler | 每调度器平均发射指令数 | 平均每个 scheduler 发射的指令数量。 |
| Executed Instruction Mix | 已执行指令组成 | 按 opcode 统计已执行指令的数量分布。 |
| Opcode | 操作码 | SASS 指令类型，例如 FFMA、LDS、LDG、IADD3。 |
| Executed Warp-Level Instructions/Opcode | 每 opcode 的 warp 级已执行指令数 | 图中横轴，表示不同 opcode 对应的 warp-level executed instruction 数量。 |
## Launch Statistics

**Launch Statistics** 用于汇总 Kernel 启动配置。它描述 grid 如何划分为 block、每个线程和每个 block 使用了多少 GPU 资源，以及这些资源配置会如何影响 occupancy、并行度和设备利用率。

这部分通常用于分析：

- **Grid 和 block 配置是否合理**：grid size、block size、thread 数量决定 Kernel 的并行规模。
- **资源占用是否限制 occupancy**：register、shared memory、block size 都可能限制每个 SM 上能同时驻留的 block/warp 数量。
- **Shared memory 使用情况**：静态 shared memory、动态 shared memory、driver shared memory 会共同影响每个 block 的 shared memory 消耗。
- **设备规模和调度背景**：SM 数量、waves per SM 可以帮助判断是否存在并行度不足或 tail effect。

## Launch Statistics 指标说明

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Grid Size | Grid 大小 | Kernel 启动时 grid 中的 block 总数。 | Grid size 决定有多少 CTA/block 可以被调度到 SM 上执行。 |
| Block Size | Block 大小 | 每个 block 中包含的 thread 数量。 | Block size 会影响 warp 数量、occupancy、调度粒度和 shared memory/register 分配。 |
| Threads | 总线程数 | Kernel 启动的总线程数量，通常等于 `Grid Size * Block Size`。 | 用于判断整体并行规模是否足够覆盖 GPU。 |
| Waves Per SM | 每个 SM 的 wave 数 | 表示平均每个 SM 需要执行多少轮 block。 | 如果 waves per SM 很低，可能存在并行度不足；如果不是整数，尾部 wave 可能导致 tail effect。 |
| Registers Per Thread | 每线程寄存器数 | 每个 thread 使用的 register 数量。 | register 使用过高会降低 occupancy，使每个 SM 可驻留的 warp/block 数减少。 |
| Static Shared Memory Per Block | 每 block 静态共享内存 | 编译期确定的 `__shared__` 内存使用量。 | 静态 shared memory 越高，每个 SM 可同时驻留的 block 可能越少。 |
| Dynamic Shared Memory Per Block | 每 block 动态共享内存 | Kernel 启动时通过第三个 launch 参数配置的动态 shared memory。 | 常见于需要运行时决定 tile 大小或缓冲区大小的 Kernel。 |
| Driver Shared Memory Per Block | 每 block 驱动共享内存 | CUDA driver 或运行时为 Kernel 额外分配的 shared memory 开销。 | 通常不是用户代码直接声明，但会计入 shared memory 资源占用。 |
| Shared Memory Configuration Size | 共享内存配置大小 | 当前 SM 上 shared memory/L1 相关配置下可用的 shared memory 容量。 | 用于理解每个 block 的 shared memory 使用是否会限制 occupancy。 |
| Function Cache Configuration | 函数缓存配置 | Kernel 的 cache preference 配置，例如偏向 L1、偏向 shared memory，或不指定偏好。 | 会影响 L1/shared memory 资源划分和某些访存行为。 |
| # SMs | SM 数量 | 当前 GPU 上参与执行的 Streaming Multiprocessor 数量。 | 用于计算并行覆盖程度、waves per SM 和整体设备利用率。 |
| Uses Green Context | 是否使用 Green Context | 表示 Kernel 是否运行在 CUDA Green Context 相关执行环境中。 | 一般性能分析中主要用于确认执行上下文类型。 |

## Launch Statistics 阅读方法

- **先看 block/grid 并行度**：如果 grid size 太小，SM 可能吃不满；如果 waves per SM 很低，Kernel 可能存在尾部利用率下降。
- **再看资源限制**：registers per thread 和 shared memory per block 是限制 occupancy 的常见原因。
- **结合 Scheduler/Warp 指标判断效果**：Launch Statistics 只说明理论启动配置，实际是否能隐藏延迟还要看 active warps、eligible warps 和 warp stall。
- **不要单独追求高 occupancy**：更高 occupancy 有助于隐藏延迟，但如果 register 降低导致 spill，或者 block size 破坏访存模式，性能反而可能下降。

## Occupancy

**Occupancy** 表示每个 SM 上实际活跃 warp 数量与硬件最大可活跃 warp 数量之间的比例。换句话说，它衡量 GPU 的 warp 执行容量有多少正在被使用。

Occupancy 主要用于分析：

- **延迟隐藏能力**：更高的 active warp 数通常更容易隐藏访存延迟和指令依赖延迟。
- **资源限制来源**：register、shared memory、block size、warp 数量都可能限制每个 SM 上能驻留的 block/warp。
- **理论值与实际值差异**：theoretical occupancy 和 achieved occupancy 差异较大时，通常说明 workload 执行不均衡、尾部效应或运行期调度情况影响了实际占用率。

需要注意的是，**更高 occupancy 不一定总是带来更高性能**。如果 Kernel 已经被某条计算流水线、访存带宽或同步开销限制，继续提高 occupancy 可能收益很小；但很低的 occupancy 通常会削弱延迟隐藏能力。

## Occupancy 顶部指标

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Theoretical Occupancy | 理论占用率 | 根据 block size、register 使用量、shared memory 使用量和硬件上限计算出的理论 occupancy。 | 用于判断当前启动配置在资源约束下最多能达到多少 occupancy。 |
| Theoretical Active Warps per SM | 每 SM 理论活跃 Warp 数 | 理论上每个 SM 可以同时驻留的 active warp 数量。 | 它是 theoretical occupancy 的 warp 数形式。 |
| Achieved Occupancy | 实际占用率 | Kernel 实际执行期间测得的 occupancy。 | 如果明显低于 theoretical occupancy，需要检查 workload 是否不均衡、是否存在 tail effect 或调度不足。 |
| Achieved Active Warps Per SM | 每 SM 实际活跃 Warp 数 | Kernel 实际执行期间每个 SM 平均活跃的 warp 数量。 | 用于观察运行期真实活跃 warp 数，而不是只看理论配置。 |
| Block Limit Registers | 寄存器限制的每 SM Block 上限 | 由每线程 register 使用量决定的每个 SM 最多可驻留 block 数。 | 如果该项较低，说明 register pressure 正在限制 occupancy。 |
| Block Limit Shared Mem | 共享内存限制的每 SM Block 上限 | 由每 block shared memory 使用量决定的每个 SM 最多可驻留 block 数。 | 如果该项较低，说明 shared memory 使用量正在限制 occupancy。 |
| Block Limit Warps | Warp 数限制的每 SM Block 上限 | 由每个 block 的 warp 数量和 SM 最大 warp 容量决定的 block 上限。 | block size 越大，每个 block 占用 warp 越多，可能降低可驻留 block 数。 |
| Block Limit SM | SM 硬件限制的每 SM Block 上限 | 由 SM 架构本身支持的最大 resident block 数决定。 | 这是硬件级 block 驻留上限，不由单个 Kernel 资源使用直接决定。 |

## Occupancy 图表说明

Occupancy 页面通常会给出多个假设分析图，用来展示如果改变某个资源参数，理论 warp occupancy 会如何变化。这些图不是实际运行过程，而是基于当前 Kernel 配置做的资源限制推演。

| 图表标题 | 中文名称 | 横轴 | 纵轴 | 说明 |
| --- | --- | --- | --- | --- |
| Impact of Varying Register Count Per Thread | 改变每线程寄存器数量对 Occupancy 的影响 | Registers Per Thread | Warp Occupancy | 展示每个线程使用不同 register 数量时，理论 warp occupancy 如何变化。register 越多，通常可驻留 warp 越少。 |
| Impact of Varying Block Size | 改变 Block Size 对 Occupancy 的影响 | Block Size | Warp Occupancy | 展示 block size 改变时，理论 warp occupancy 如何变化。block size 会影响每个 block 的 warp 数、调度粒度和资源分配。 |
| Impact of Varying Shared Memory Usage Per Block | 改变每 Block 共享内存使用量对 Occupancy 的影响 | Shared Memory Per Block | Warp Occupancy | 展示每个 block 使用不同 shared memory 容量时，理论 warp occupancy 如何变化。shared memory 使用越多，通常可驻留 block 越少。 |

## Occupancy 阅读方法

- **先比较 theoretical 和 achieved**：理论 occupancy 高但实际 occupancy 低，通常说明实际执行阶段存在不均衡、尾部效应或调度不足。
- **再看 block limit 来源**：如果 `Block Limit Registers` 最低，优先关注 register pressure；如果 `Block Limit Shared Mem` 最低，优先关注 shared memory 使用量。
- **结合 Warp State 判断是否真的需要更高 occupancy**：如果主要 stall 是 long scoreboard，更高 occupancy 可能有助于隐藏访存延迟；如果瓶颈是 math pipe throttle，提高 occupancy 未必有效。
- **结合代码调整参数**：block size、每线程计算量、寄存器使用、shared memory tile 大小都会改变 occupancy，但优化目标应该是整体性能，而不是单独把 occupancy 拉满。

## GPU and Memory Workload Distribution

**GPU and Memory Workload Distribution** 用于分析 workload 在 GPU 计算单元和内存层次结构中的分布情况。它统计 SM、SMP、SMSP、L1、L2、DRAM 等模块的 active cycles 和 elapsed cycles，帮助判断不同硬件单元的工作量是否均衡。

这部分通常用于分析：

- **SM 之间是否负载均衡**：不同 SM 的 active cycles 差异过大时，可能存在 workload imbalance。
- **内存层次是否参与充分**：L1、L2、DRAM 的 active cycles 可以反映不同缓存/显存层的工作强度。
- **计算与内存活动是否匹配**：SM active cycles 很高但 DRAM active cycles 很低，说明显存可能不是主要压力点；反过来则可能表示访存压力更突出。
- **总周期与平均周期关系**：average、min、max、sum 可以帮助观察各硬件实例之间的分布差异。

## Workload Distribution 顶部指标

| 英文指标 | 中文名称 | 详细说明 | 观察重点 |
| --- | --- | --- | --- |
| Average SM Active Cycles | 平均 SM 活跃周期数 | 每个 SM 平均处于 active 状态的周期数。 | 用于观察 SM 平均工作时间。 |
| Total SM Elapsed Cycles | SM 总经过周期数 | 所有 SM 的 elapsed cycles 总和。 | 用于衡量 SM 层面的总体运行周期规模。 |
| Average SMSP Active Cycles | 平均 SMSP 活跃周期数 | 每个 SMSP 平均处于 active 状态的周期数。SMSP 是 SM 内部的子分区。 | 用于观察 SM 子分区层面的活跃程度。 |
| Total SMSP Elapsed Cycles | SMSP 总经过周期数 | 所有 SMSP 的 elapsed cycles 总和。 | 用于衡量 SM 子分区整体运行周期规模。 |
| Average L1 Active Cycles | 平均 L1 活跃周期数 | L1/TEX Cache 平均处于 active 状态的周期数。 | 用于观察一级缓存路径的平均工作强度。 |
| Total L1 Elapsed Cycles | L1 总经过周期数 | 所有 L1/TEX 实例的 elapsed cycles 总和。 | 用于衡量一级缓存层面的总体运行周期规模。 |
| Average L2 Active Cycles | 平均 L2 活跃周期数 | L2 Cache 平均处于 active 状态的周期数。 | 用于观察二级缓存路径的平均工作强度。 |
| Total L2 Elapsed Cycles | L2 总经过周期数 | 所有 L2 分区或实例的 elapsed cycles 总和。 | 用于衡量 L2 层面的总体运行周期规模。 |
| Average DRAM Active Cycles | 平均 DRAM 活跃周期数 | DRAM 平均处于 active 状态的周期数。 | 用于观察显存层面的平均工作强度。 |
| Total DRAM Elapsed Cycles | DRAM 总经过周期数 | 所有 DRAM 分区或通道的 elapsed cycles 总和。 | 用于衡量显存层面的总体运行周期规模。 |

## Workload Distribution 表格指标

Workload Distribution 表格通常按硬件模块列出 active cycles 的 `Average`、`Min`、`Max` 和 `Sum`。这些统计量用于观察不同实例之间的负载分布。

| 表格行 | 中文名称 | 详细说明 |
| --- | --- | --- |
| SM Active Cycles | SM 活跃周期数 | 每个 SM 处于 active 状态的周期统计。 |
| SMSP Active Cycles | SMSP 活跃周期数 | 每个 SM 子分区处于 active 状态的周期统计。 |
| L1 Active Cycles | L1 活跃周期数 | 每个 L1/TEX Cache 实例处于 active 状态的周期统计。 |
| L2 Active Cycles | L2 活跃周期数 | 每个 L2 Cache 分区或实例处于 active 状态的周期统计。 |
| DRAM Active Cycles | DRAM 活跃周期数 | 每个 DRAM 分区或通道处于 active 状态的周期统计。 |

| 统计列 | 中文名称 | 详细说明 |
| --- | --- | --- |
| Average | 平均值 | 所有对应硬件实例 active cycles 的平均值。 |
| Min | 最小值 | 对应硬件实例中 active cycles 的最小值。 |
| Max | 最大值 | 对应硬件实例中 active cycles 的最大值。 |
| Sum | 总和 | 所有对应硬件实例 active cycles 的总和。 |

## Workload Distribution 阅读方法

- **先看 average 和 sum**：average 反映单个硬件实例的平均活跃程度，sum 反映整个硬件层级的总工作量。
- **再看 min/max 差异**：如果 min 和 max 差距很大，说明不同 SM、cache 分区或 DRAM 分区之间可能存在负载不均衡。
- **结合 Memory Workload 定位层级**：L1/L2/DRAM active cycles 的差异可以和吞吐率、hit rate、sector miss 等指标一起判断内存压力发生在哪一层。
- **结合 Scheduler 和 Warp State 解释 SM 活跃度**：SM active cycles 高不代表执行效率高，还需要结合 issue active、eligible warps 和 stall reasons 判断。

## 常见术语对照

| 英文术语 | 中文名称 | 说明 |
| --- | --- | --- |
| LSU | Load Store Unit，加载存储单元 | 负责处理 load/store 等访存指令，是 CUDA Kernel 中最常见的访存执行路径。 |
| MIO | Memory Input Output 单元 | 与内存输入输出、数据移动、共享内存或特殊访存路径相关的执行单元集合。 |
| FMA | Fused Multiply-Add，浮点乘加流水线 | 执行乘加融合计算，常用于 FP32/FP64 数值计算。 |
| ALU | Arithmetic Logic Unit，算术逻辑单元 | 执行整数、逻辑、基础算术等操作。 |
| ADU | Address Unit，地址计算单元 | 负责地址生成和地址计算，复杂索引表达式可能增加该路径压力。 |
| CBU | Control/Branch Unit，控制流/分支控制单元 | 处理分支、控制流和谓词相关指令。 |
| Uniform | Uniform 流水线 | 处理 warp 内一致的 uniform 指令或 uniform 数据路径。 |
| Tensor Pipe | Tensor Core 流水线 | 执行矩阵乘加等 Tensor Core 指令。 |
| FP64 Pipe | 双精度流水线 | 执行 FP64 双精度浮点计算。 |
| L1TEX | L1/Texture Cache | SM 附近的一级缓存和纹理缓存路径，承担低延迟缓存访问。 |
| Xbar | Crossbar，交叉开关 | GPU 内部连接 SM、L1/TEX、L2 等模块的数据交换网络。 |
| Sector | Cache Line Sector | cache line 被拆分后的更小传输或统计粒度。 |
| Wavefront | Cache 访问波前 | 缓存系统内部处理一次访问请求时使用的工作单元。 |
| Tag Request | Cache 标签查询请求 | 缓存判断命中或缺失时进行的标签查询。 |
| Writeback | 数据回写 | 数据从执行单元、缓存或内部路径写回目标位置的过程。 |
| Device Fill | 从 GPU 设备内存填充 | L2 或缓存层从 GPU device memory 拉取数据进行填充。 |
| Sysmem Fill | 从主机系统内存填充 | L2 或缓存层从 host/system memory 拉取数据进行填充。 |
| DRAM | 显存 | GPU 的外部显存，是全局内存访问最终可能到达的存储层。 |

## 阅读这些指标的基本思路

- **先看总览指标**：通过 Compute Throughput、Memory Throughput、DRAM Throughput 判断 Kernel 更接近计算受限、缓存路径受限，还是显存带宽受限。
- **再看细分项定位路径**：如果 Memory Throughput 很高但 DRAM 很低，说明压力可能集中在 L1/L2 或内部请求路径，而不是显存带宽本身。
- **不要只看单个百分比**：某个指标高只说明对应硬件路径忙，不一定等价于性能好。优化时要结合执行时间、访存模式、occupancy、指令结构一起判断。
- **结合 Kernel 语义解释指标**：例如 copy kernel、transpose kernel、GEMM kernel、reduction kernel 的热点路径不同，同一个指标在不同 Kernel 中含义也会不同。
