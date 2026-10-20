---
title: Infra 面试真题与解析
summary: 九道 AI Infra 面试真题，覆盖 DP 负载、双机部署、推理框架、DSpark、DeepGEMM、GDN 与自动调优。
order: 2
---

# Infra 面试真题与解析

本节分析分布式部署、推理引擎、模型架构与性能调优相关的面试问题，重点说明实现机制、资源约束与验证方法。

**参考资料查阅日期：2026-10-20。** 框架功能与参数以对应版本文档为准。部署方案的分析假设和性能验证条件在相关章节中分别说明。

## DP 部署中的 rank 0 负载不均衡

> 问题：分布式推理采用 DP 部署时，只有 rank 0 负载过高，应如何定位和处理？

### 核心回答

负载分析需要明确 **CPU、GPU 计算、显存、通信和请求队列**各项指标，并确认 rank 0 对应 DP 副本编号、TP 组内编号还是全局进程编号。进程职责和瓶颈位置是选择优化措施的依据。

普通整模型 DP 的副本处理不同请求；MoE 场景的 Attention DP 与共享专家组则可能逐层协同。vLLM 文档也明确区分了这两种执行关系。[vLLM DP 部署说明](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)

### 排查方法

不同负载现象对应的排查方向如下：

| 现象 | 需要验证的原因 | 对应处理方向 |
| --- | --- | --- |
| rank 0 的 CPU 高，GPU 请求相对均衡 | API 接入、分词、采样后处理或日志集中在同一进程/节点 | 剖析前端与引擎耗时，扩展前端进程、分离入口职责，处理 CPU/NUMA 亲和性 |
| rank 0 收到更多请求或输入 token | 路由固定、长连接绑定、请求长度不均或缓存亲和造成热点 | 按活跃生成请求、待处理输入 token 等估计成本，结合前缀复用选择实例 |
| 请求数量相近，但 rank 0 执行更久 | 长输入/长输出占比、抢占重算、功耗或链路差异 | 对比工作量分布、缓存占用、GPU 频率与通信时间线 |
| 单卡执行时间显著较长，其他卡等待集合通信 | 热门专家集中、阶段失衡或某个进程成为慢节点 | 检查专家 token 分布、分片策略和跨卡/跨机拓扑 |

记录每个 rank 的请求数、输入 token 数、活跃生成数、排队长度与阶段耗时，并与 GPU/CPU 时间线对齐。请求长度分布会影响实际工作量，仅比较请求数量可能掩盖长输入带来的不均衡。

### 适用条件与限制

- **缓存亲和与负载均衡需要折中**：同类请求长期集中于同一副本，可能提高缓存命中率，同时形成负载热点。设置副本负载上限可限制热点扩张。
- **优先调整新请求的路由**：迁移正在生成的请求还涉及 KV/循环状态传输与一致性维护，需要单独评估迁移成本。
- **共享 EP 组需要一致的成员关系**：参与集合通信的进程必须协调变更，局部停止进程可能导致其他进程阻塞。

排查顺序应覆盖进程职责、资源指标和实际 token 工作量。共享专家组还需检查专家热点与同步等待。

## 双机 16 卡 H100 的 700B FP8 部署方案

> 问题：使用两台各配备 8 张 H100 的服务器部署 700B 模型，采用 FP8 精度，要求 TTFT 小于 2 秒、TPOT 不超过 50 ms。在不调整 `max-num-seqs` 等调度参数的条件下，有哪些优化方案？

### 部署条件与性能目标

需要明确 700B 表示总参数还是每 token 激活参数、模型采用 Dense 还是 MoE，以及 H100 型号、显存容量、节点间网络、输入输出长度分布、请求到达率和延迟统计分位数。SLO（服务等级目标）的可达性取决于这些条件。

### 显存估算与并行方案

**以下按每卡 80 GB、700B 为总参数、全部权重以每参数 1 字节近似估算**，使用十进制 GB。NVIDIA DGX H100 的 8 卡系统规格列出了 640 GB GPU 显存。[NVIDIA DGX H100 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-h100-datasheet.pdf)

$$
M_{\mathrm{weights}}\approx700\times10^9\times1\ \mathrm{byte}
=700\ \mathrm{GB}
$$

16 卡标称总容量约 1280 GB，平均分片的纯权重约为 43.75 GB/卡。实际占用还包括未量化参数、缩放因子、KV/循环状态、激活、通信缓冲和 CUDA Graph 工作区，因此需要逐卡核算显存分配。

| 候选方式 | 该假设下的分析 |
| --- | --- |
| 单机一个完整 FP8 副本，双机普通 DP | 每台只有约 640 GB，小于约 700 GB 的纯权重；不能直接部署两个完整副本 |
| TP=8、PP=2、DP=1 | 可以作为跨两节点分片的候选，层内 TP 留在机内；但 PP 的阶段串行与气泡可能影响单请求延迟 |
| TP=16、PP=1、DP=1 | 可以作为另一候选；每层跨机集合通信可能成为代价，需要与前者实测比较 |
| 一台 Prefill、一台 Decode，各自存完整模型 | 同样受到每台 640 GB 的限制；即使允许细粒度分离，也要重新核算权重与状态放置 |

各候选方案均要求模型实现与框架支持相应拓扑及功能组合。

**对于 MoE 模型**，计算量主要取决于激活参数，显存仍需容纳总参数。Attention DP、EP 与 TP 的组合需要分别核算复制参数、分片参数和通信成本。

### 固定调度预算下的优化方向

- **定位关键路径**：将首 token 延迟分解为前端、排队和 Prefill 开销，将生成延迟分解为计算、KV 访存、集合通信和 CPU 调度开销。50 ms 的 TPOT 对应单请求约 20 token/s，需要与集群总吞吐分别统计。
- **优化并行与拓扑**：比较 TP/PP 方案；MoE 模型还需评估 EP、专家负载与计算通信重叠，检查 GPU—网卡亲和性、RDMA 路径和实际链路带宽。
- **优化执行路径**：确认 FP8 GEMM、注意力算子、融合和图执行路径适配 H100，并检查是否存在低效的后端回退。
- **复用重复前缀**：对确有共同输入的业务评估前缀缓存；必须分别报告冷缓存与热缓存结果，不能用预先命中的数据代替冷启动目标。
- **评估投机解码**：如果匹配的 draft/MTP 路径可用，比较每轮实际接受长度与额外验证成本，同时检查显存和所选并行方式的兼容性。[vLLM 投机解码适用范围与兼容性](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

实验应固定题目限制的调度参数，包括受限范围内的 `max_num_batched_tokens` 等配置，并保持精度、输入长度和压测负载一致。

### 性能验收

在相同模型、精度、输入输出分布和到达率下，记录目标分位数的 TTFT、TPOT、ITL、有效吞吐及失败率。若资源下界和实测结果均无法满足目标，需要评估资源扩容或业务条件调整。

P/D 分离能够减少阶段干扰，其可行性取决于状态传输成本、权重放置与资源容量。[DistServe 论文](https://arxiv.org/abs/2401.09670)

## vLLM 与 SGLang 的架构比较

> 问题：vLLM 与 SGLang 在架构设计和服务能力上有哪些差异？

### 核心回答

vLLM 与 SGLang 都是大模型推理与服务框架，覆盖调度、缓存、算子、并行和接口适配。比较应围绕**设计侧重点、具体版本实现和目标工作负载**展开。

| 维度 | vLLM | SGLang |
| --- | --- | --- |
| 早期代表性设计 | PagedAttention 与分页 KV 管理，面向高吞吐服务 | RadixAttention，以 radix tree 组织可复用前缀，并提供结构化生成相关能力 |
| 执行与调度 | V1 重构降低 CPU 开销，结合 token 调度、分块 Prefill、编译与图执行 | 强调 CPU 调度与 GPU 执行重叠，结合前缀缓存与批次调度 |
| 现在的功能交集 | 前缀缓存、结构化输出、投机解码、多模态和分布式服务等能力均需按版本核对 | 同样覆盖这些服务需求，具体模型与后端成熟度因版本而异 |
| 比较重点 | 模型实现、硬件后端、部署方式与现有系统的适配成本 | 相同维度，尤其要检查目标工作负载的前缀复用及路由行为 |

历史设计可参阅 [vLLM V1 介绍](https://vllm.ai/blog/2025-01-27-v1-alpha-release)、[SGLang RadixAttention 介绍](https://www.lmsys.org/blog/2024-01-17-sglang/)和 [SGLang 重叠调度说明](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)。这些早期文章适合解释设计来源，不能直接作为 2026 年版本的性能排名。

### 选型与评测

固定模型 revision、权重与 KV 精度、采样设置、上下文、到达率、硬件和网络，再分别比较冷/热缓存负载。结果至少包含 TTFT/ITL 的尾延迟、有效吞吐、显存、错误率与质量回归。

两者均具备前缀缓存等服务能力，具体性能受模型专用算子、并行组合、缓存命中分布和 CPU 开销影响。选型还需考虑部署维护与故障定位成本。

## DP Attention 的执行机制

> 问题：DP Attention 的执行机制与适用场景是什么？

### 核心回答

DP Attention，也称 Attention DP/DPA，是让**注意力子层对不同请求做数据并行，而 FFN/MoE 子层可以继续做 TP 或 EP**。它不要求每个注意力 rank 都保存完整的专家集合。

在覆盖多张卡的 MoE 组中，不同注意力组持有各自请求的 KV，进入 MoE 阶段后将 token 分发至对应专家。采用 EP 时的数据流如下：

```mermaid
flowchart LR
    A["各注意力组处理不同请求"] --> B["按专家路由分发 token"]
    B --> C["跨 rank 的专家组执行 MoE"]
    C --> D["合并结果并回到所属请求"]
    D --> E["进入下一层"]
```

请求在注意力阶段独立，并不表示整条前向都独立。专家阶段仍可能需要全组同步。

### MLA 模型中的收益条件

当 TP 实现复制压缩 KV，或注意力在较高 TP 度数下产生较小计算块时，按请求划分注意力工作可能改善缓存利用和执行效率。实际收益取决于 KV 分片方式、批量大小，以及跨阶段通信的额外成本。

SGLang 的官方指南区分整模型 DP 和 DPA，并给出了 DPA 与 EP 的组合；查阅时推荐用 `--attn-dp-size` 指定注意力 DP 组，旧 `--enable-dp-attention` 写法处于弃用迁移中。[SGLang DP/DPA 指南](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)

### 大规模 MoE 部署中的应用

对于专家权重占主导的 700B MoE 模型，Attention DP + EP 可复制必要的注意力/稠密参数，并将专家权重分散到全组。显存估算需要同时包含分片参数和复制参数。

Dense 模型的参数结构和通信模式不同，需要单独评估。不同框架中 DP、TP、EP 参数的分组含义也可能存在差异，迁移配置时应核对实际进程组与权重分布。

## DSpark 的架构与验证调度

> 问题：DSpark 的架构设计如何改善投机解码的效率？

### 核心回答

DeepSeek 公开的 **DSpark 投机解码方案**结合半自回归草稿生成与置信度驱动的验证调度，在减少生成轮次的同时控制草稿和验证的额外开销。[DSpark 论文](https://arxiv.org/abs/2607.05147)

### 两个关键设计

- **并行骨干与轻量顺序模块**：计算密集的草稿网络并行处理候选块，通过轻量 Markov 等顺序头引入块内 token 依赖，改善纯并行草稿后半段的失配问题。
- **按置信度分配验证长度**：估计候选前缀能被接受的概率，再结合硬件代价与当前负载安排验证预算，避免为价值较低的后缀支付完整验证成本。

单位时间内提交的 token 数可近似表示为：

$$
R_{\mathrm{accepted}}\approx
\frac{\text{一轮中实际提交的 token 数}}
{t_{\mathrm{draft}}+t_{\mathrm{verify}}+t_{\mathrm{schedule}}}
$$

接受长度增加时，额外验证成本也可能降低实际生成速率，因此需要联合评估接受长度与每轮耗时。

### 执行效率与无损性

验证长度缩短后，执行必须真正减少计算；如果依然填充到原来的大形状，理论收益可能被抵消。SGLang 的集成讨论了可变验证长度、CUDA Graph 和重叠执行等问题，也说明了不同负载下验证预算会变化。[DSpark 在 SGLang 中的实现说明](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)

无损性要求目标模型验证、标准拒绝采样和验证前缀选择共同满足正确性约束。前缀选择还需满足论文中的因果条件，避免依赖未来候选产生选择偏差。[DSpark 调度与无损性分析](https://arxiv.org/html/2607.05147v1#S3.SS2)

评测应覆盖常规自回归解码、固定长度投机与 DSpark，比较每轮耗时、提交长度、尾延迟和显存，并确认草稿模型与目标模型的适配关系。官方 [DeepSpec 仓库](https://github.com/deepseek-ai/DeepSpec)提供了草稿训练与接受率评测流程，具体加速收益需按模型和负载测量。

## DeepGEMM 的计算原语与 MoE 优化

> 问题：DeepGEMM 的主要功能、优化机制和适用场景是什么？

### 核心回答

DeepGEMM 是 DeepSeek 开源的高性能计算原语库，提供低精度矩阵乘法（GEMM）和 MoE 分组矩阵乘法（grouped GEMM）等能力。查阅版本还包含 FP8/FP4/BF16、融合 MoE 和索引器相关原语。[DeepGEMM 官方说明](https://github.com/deepseek-ai/DeepGEMM)

### MoE 分组矩阵乘法

专家 $e$ 的一层线性变换可以写成：

$$
Y_e=X_eW_e,\qquad
X_e\in\mathbb{R}^{m_e\times d},\quad
W_e\in\mathbb{R}^{d\times f}
$$

专家的 $d,f$ 往往相同，但收到的 token 数 $m_e$ 不同。逐个启动小矩阵乘法可能难以充分使用 GPU；分组执行可以让多个专家任务共享调度与设备资源。

DeepGEMM 提供面向不同 token 布局的接口，例如 Prefill 场景的连续分组，以及 Decode/CUDA Graph 场景下用有效长度掩码表达的分组。实际调用还需要满足缩放因子、布局、对齐与硬件要求。[DeepGEMM 接口说明](https://github.com/deepseek-ai/DeepGEMM#interfaces)

### 算子优化机制

官方 SM90 FP8 实现采用 TMA 数据搬运、WGMMA Tensor Core 计算、多级缓冲和线程分工，实现访存与计算重叠，并按矩阵块组织资源。[SM90 FP8 实现](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh)

更小的数据类型能降低传输量，但量化、转置、token 重排和缩放处理都可能增加成本。评估应覆盖这些环节及实际 $m_e$ 分布，不能只比较孤立大矩阵的峰值 FLOPS。

### 与 DeepEP 的关系

DeepEP 侧重专家并行中的 token 分发与回收通信；DeepGEMM 侧重计算原语。Mega MoE 进一步融合并重叠通信与计算，使两类能力在融合执行路径中形成交集。[DeepEP 官方说明](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)

MoE 层的性能分析需要同时覆盖专家负载、通信、矩阵形状与算子执行时间，单个 GEMM 的性能收益应通过整层测量验证。

## Gated DeltaNet 与 Qwen3.5 的混合架构

> 问题：Gated DeltaNet 的架构与设计背景是什么？Qwen3.5 采用该结构的原因是什么？

### 核心回答

**Gated DeltaNet（GDN，门控 Delta 网络）**将历史信息写入固定维度的矩阵状态，结合状态衰减与基于预测误差的增量更新，在循环推理时减少保存历史位置完整 K/V 的需求。[Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)

### 核心状态更新

以下采用 $S_t\in\mathbb{R}^{d_v\times d_k}$ 的列向量记法，展示单个头的核心递推，省略整层的投影、局部卷积、归一化和输出门控：

$$
\bar S_t=\alpha_t S_{t-1}
$$

$$
\delta_t=v_t-\bar S_tk_t,\qquad
S_t=\bar S_t+\beta_t\delta_tk_t^\top,\qquad
o_t=S_tq_t
$$

$q_t,k_t$ 是查询与键，$v_t$ 是要写入的值；$\alpha_t$ 控制旧状态的衰减，$\beta_t$ 控制沿当前键方向的更新幅度。$\bar S_tk_t$ 是已有状态对当前值的预测，$\delta_t$ 是修正误差。

它与直接累加 $v_tk_t^\top$ 的区别，是能沿当前键方向修正已有关联；门控则提供遗忘与状态管理能力。论文还给出了分块并行形式，所以训练和 Prefill 不必都按逐 token 串行循环执行。[门控 Delta 递推与分块算法](https://arxiv.org/html/2412.06464v1#S3)

### 混合架构的设计取舍

以 **Qwen3.5-397B-A17B** 为例，官方模型卡给出每组“三层 Gated DeltaNet + 一层 Gated Attention”的混合布局，并在这些层后使用 MoE。3:1 为该模型的具体配置，其他变体需核对各自的层布局。[Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

从系统角度分析，多数层通过循环状态降低长序列处理与缓存压力，部分全注意力层保留直接访问历史位置的能力。这一分析解释了混合结构的资源取舍；各组件对模型质量的贡献需要结合消融实验评估。

### 适用条件与限制

- 固定大小状态不能无限无损保存任意历史，状态容量与精确检索能力需要权衡。
- GDN 不是对 softmax attention 的简单等价改写，也不等同于普通滑动窗口注意力。
- 混合模型保留全注意力层的 KV Cache，这部分缓存仍随上下文长度增长。
- 服务系统要同时管理 KV、循环状态与投机验证后的状态提交/回退；不同状态的缓存和复用方式不能默认相同。

## vLLM 对 DeepSeek-V4-Pro 的适配

> 问题：vLLM 对 DeepSeek-V4-Pro 的支持涉及哪些模型适配和服务能力？

### 模型与版本

截至查阅日期，vLLM 已提供 **DeepSeek-V4-Pro** 的官方部署配方。预览权重、`DeepSeek-V4-Pro-0813` 正式权重和其他量化变体的适配条件需要分别核对。

官方模型卡说明，0813 正式版基于预览模型结构并附带 DSpark 模块，还使用专门的消息编码方式。[DeepSeek-V4-Pro-0813 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)

### 执行链路与适配范围

| 层面 | 需要核对的内容 |
| --- | --- |
| 模型结构 | 能否正确执行 CSA/HCA 混合注意力、mHC 与 MoE 等模块，缓存表示是否正确 |
| 数值格式 | 具体权重变体采用什么格式，硬件与后端是否支持对应量化和计算路径 |
| 并行部署 | 注意力、专家组与状态如何分布，每卡显存及通信是否满足要求 |
| 投机解码 | checkpoint 是否携带所需 draft，DSpark/MTP 方法、采样与图执行是否兼容 |
| 服务协议 | 消息编码、推理输出解析、工具调用和停止条件是否正确 |

vLLM 配方将该模型描述为总参数约 1.6T、每 token 激活约 49B 的 MoE，并区分了混合精度预览权重等变体。配方的基础支持标记为 vLLM 0.20.0+，对附带 DSpark 的相关变体还注明了 0.25.0 的要求；这些是**该配方查阅时的版本条件**，部署时应固定镜像和模型 revision。[vLLM DeepSeek-V4-Pro 配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)

### 部署验证

部署验证应覆盖模型实现、算子与状态、并行通信、服务正确性及性能验收。NVIDIA 路径的官方实现可用于核对具体模块与执行逻辑。[vLLM DeepSeek-V4 NVIDIA 实现](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/nvidia/model.py)

官方配方说明适配条件，具体硬件上的质量与性能仍需实测验证。1.6T 模型的权重、状态与工作区占用，也需要按实际格式重新估算。

## vLLM 自动调优的设计与验证

> 问题：如何设计 vLLM 自动调优流程，并验证配置收益？

### 优化目标

自动调优是带约束的配置搜索：固定模型、精度与业务分布，在满足尾延迟、失败率和质量目标的前提下，提高有效吞吐或降低资源成本。

一个可用的请求级目标是：

$$
\mathrm{Goodput}(\theta)=
\frac{\#\{r:\mathrm{TTFT}_r\le\tau_f,
\mathrm{TPOT}_r\le\tau_d,\ r\text{ 成功}\}}
{T_{\mathrm{measurement}}}
$$

$\theta$ 是服务配置，$\tau_f,\tau_d$ 是业务阈值。目标分位数和可接受的超时、拒绝率还要单独作为约束，避免通过大量拒绝请求“优化”成功请求的延迟。

### 分层搜索与测量

| 层次 | 候选项 | 验证重点 |
| --- | --- | --- |
| 资源与拓扑 | TP/PP/DP，MoE 的 EP，节点放置 | 先排除显存不足、不支持的组合和通信明显不合算的方案 |
| 缓存与执行 | KV 格式、算子后端、编译/图执行、缓存策略 | 精度、显存、预热成本与长上下文行为 |
| 调度 | 允许调整时再搜索批量与 token 预算 | TTFT、ITL、吞吐和抢占的折中 |
| 投机 | draft、候选长度与相关策略 | 实际提交长度、验证时间与额外显存 |

可以先用小规模网格或随机搜索做基线，再对高价值区域使用贝叶斯优化等方法。每次试验保留配置、版本、工作负载、指标和失败原因；OOM、启动失败、超时及质量回归都应记录。

**搜索空间应严格遵守部署约束，受限的调度参数保持固定。**

### vLLM 参数扫描工具

官方 `vllm bench sweep serve` 可以启动不同服务配置并逐组压测；`serve_workload` 可以探索到达率或并发变化下的延迟与吞吐曲线。[vLLM 参数扫描文档](https://docs.vllm.ai/en/latest/benchmarking/sweeps/)

以下配置演示小模型的参数扫描流程。`serve-params.json` 定义候选服务配置：

```json
[
    {"max_num_seqs": 32, "max_num_batched_tokens": 2048},
    {"max_num_seqs": 64, "max_num_batched_tokens": 4096}
]
```

扫描命令如下：

```shell
TUNING_MODEL='Qwen/Qwen3-0.6B'

vllm bench sweep serve \
    --serve-cmd "vllm serve ${TUNING_MODEL}" \
    --bench-cmd "vllm bench serve --model ${TUNING_MODEL} --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 200 --request-rate 2 --goodput ttft:2000 tpot:50" \
    --serve-params serve-params.json \
    --num-runs 3 \
    --output-dir tuning-results \
    --experiment-name baseline-sweep
```

`ttft:2000` 和 `tpot:50` 的阈值单位是毫秒，用于统计满足条件的请求。该命令为依据 CLI 文档编写的配置示例，尚无 GPU 压测结果。随机输入适用于初步测量，真实业务与前缀缓存收益需要使用对应请求分布验证。[vLLM 在线压测参数](https://docs.vllm.ai/en/latest/cli/bench/serve/)

### 测量可靠性与上线验证

- **区分冷与热**：记录编译、图捕获和缓存预热；压测工具可能重置缓存，前缀复用实验需设计一致的预热与测量过程。
- **覆盖负载分布**：包含长短输入、输出长度、前缀重复、突发流量和取消行为，避免只对单一随机长度最优。
- **处理测量噪声**：重复实验，检查 GPU 频率、其他进程干扰及性能差异是否稳定。
- **保留独立验证集**：在未参与选参的请求分布上复测，并检查质量与错误率。
- **逐步上线**：离线筛选后再用小流量验证，保留上一版配置与回退触发条件；重启型参数不应频繁在线切换。

有效的自动调优流程应包含目标与约束定义、分层配置搜索、可复现实验、独立负载验证和小流量上线验证。
