---
title: FlashAttention-2 源码学习笔记
date: 2026-08-01
tags: [FlashAttention, CUDA, PyTorch, GPU 编程, Attention]
summary: 从 flash_attn/flash_attn_interface.py 的前向入口开始，逐步追踪 FlashAttention-2 从 Python API、C++ dispatch 到 CUDA kernel 的源码路径。
---

# FlashAttention-2 源码学习笔记

本文是 FlashAttention-2 源码学习笔记。最终目标是一路追到 CUDA kernel：理解 Python API 如何进入 C++ extension，C++ dispatch 如何选择具体模板实例，以及 `flash_fwd_kernel.h` 里的 tile、softmax 和 PV 计算如何组织。

这一节先从 `flash_attn/flash_attn_interface.py` 的公开前向入口开始。先把入口接口、参数语义和数据布局理顺，后面再沿着这些入口继续下钻到 `csrc/flash_attn/flash_api.cpp` 和 `csrc/flash_attn/src/flash_fwd_kernel.h`。这样读 CUDA kernel 时，能明确每个 kernel 参数来自哪个 Python API，以及不同接口为什么会走到同一类底层 forward。

FA2 的 Python 前向接口名字很多，但核心不是“有很多套注意力算法”，而是为了适配真实模型里的不同 **Q/K/V 存储方式**、**定长 / 变长 batch** 和 **推理 KV cache**。这些接口最终会收敛到底层 C++/CUDA extension，例如 `flash_attn_gpu.fwd`、`flash_attn_gpu.varlen_fwd` 和 `flash_attn_gpu.fwd_kvcache`。

## 源码阅读路线

这一节对应路线图里的第一层：

```mermaid
flowchart TD
    A["Python 前向入口<br/>flash_attn_interface.py"] --> B["Autograd wrapper<br/>FlashAttnFunc 等"]
    B --> C["C++/PyBind 入口<br/>flash_api.cpp"]
    C --> D["Launch dispatch<br/>run_mha_fwd"]
    D --> E["CUDA kernel 模板<br/>flash_fwd_launch_template.h"]
    E --> F["Forward 主体<br/>flash_fwd_kernel.h"]
    F --> G["Online softmax<br/>softmax.h"]
```

本节先回答：调用方应该选哪个 Python 前向接口，每个接口的参数和返回值是什么意思，以及这些接口如何收敛到底层 CUDA forward。后续章节会继续沿着这张图往下读。

## 接口总览

| 接口 | 输入布局 | 主要场景 | 是否支持 backward | 备注 |
| --- | --- | --- | --- | --- |
| `flash_attn_func` | `q`、`k`、`v` 三个独立 tensor | 最通用的定长 batch attention | 支持 | 支持 MQA/GQA，读源码时建议先从它开始。 |
| `flash_attn_qkvpacked_func` | `qkv` 合在一个 tensor，形状含维度 `3` | self-attention 训练，QKV 由同一个线性层投影得到 | 支持 | backward 直接返回 `dqkv`，避免把 `dq/dk/dv` 再拼回去。 |
| `flash_attn_kvpacked_func` | `q` 独立，`kv` 合在一个 tensor，形状含维度 `2` | cross-attention 或 K/V 已经打包的 MQA/GQA | 支持 | backward 直接返回 `dq` 和 `dkv`。 |
| `flash_attn_varlen_func` | 压平后的 `q`、`k`、`v`，配合 `cu_seqlens_*` | 不同样本长度差异较大的训练 / 推理 prefill | 支持 | 避免 padding 到同一个长度。 |
| `flash_attn_varlen_qkvpacked_func` | 压平后的 `qkv`，配合一个 `cu_seqlens` | 变长 self-attention，Q/K/V 等长 | 支持 | varlen + qkv packed 的组合。 |
| `flash_attn_varlen_kvpacked_func` | 压平后的 `q` 和 `kv`，配合两组 `cu_seqlens_*` | 变长 cross-attention 或 GQA/MQA | 支持 | varlen + kv packed 的组合。 |
| `flash_attn_with_kvcache` | 当前 `q` + 已分配的 `k_cache/v_cache` | LLM decode / incremental inference | 不支持 | 可以原地更新 KV cache，并在同一个 kernel 中完成 attention。 |

## 命名规则

这些名字可以按三个维度拆开理解：

| 名字片段 | 含义 | 典型形状 |
| --- | --- | --- |
| `qkvpacked` | Q、K、V 已经堆在同一个 tensor 中 | `(batch, seqlen, 3, nheads, headdim)` 或 `(total, 3, nheads, headdim)` |
| `kvpacked` | K、V 已经堆在同一个 tensor 中，Q 单独传入 | `(batch, seqlen_k, 2, nheads_k, headdim)` 或 `(total_k, 2, nheads_k, headdim)` |
| `varlen` | batch 中每条序列长度不同，token 被压平成一维 `total` | `q: (total_q, nheads, headdim)` |
| `with_kvcache` | 使用已有 KV cache 做 decode 或增量推理 | `k_cache/v_cache` 保存历史 token 的 K/V |

换句话说，普通接口解决“怎么传 Q/K/V”，`varlen` 解决“怎么表示不等长序列”，`with_kvcache` 解决“推理时怎么复用历史 K/V”。

## 常见参数语义

下面这些参数会在多个接口中重复出现。

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `dropout_p` | `float` | attention dropout 概率。推理时应设为 `0.0`。如果大于 `0`，forward 会保存随机状态供 backward 复现 dropout。 |
| `softmax_scale` | `Optional[float]` | softmax 前对 `QK^T` 的缩放系数。若为 `None`，源码中使用 `headdim ** (-0.5)`。 |
| `causal` | `bool` | 是否启用因果 mask。对于 `seqlen_q != seqlen_k` 的情况，FA2 使用右下角对齐的 causal mask。 |
| `window_size` | `tuple[int, int]` | sliding window local attention 的左右窗口。默认 `(-1, -1)` 表示不限制上下文窗口。 |
| `softcap` | `float` | attention score softcap。`0.0` 或小于等于 `0` 表示关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi bias 的斜率，常见形状为 `(nheads,)` 或 `(batch_size, nheads)`，dtype 通常为 fp32。 |
| `deterministic` | `bool` | 是否使用确定性 backward。forward 本身是确定性的；确定性 backward 会稍慢并使用更多内存。 |
| `return_attn_probs` | `bool` | 是否返回 attention 概率相关中间结果。源码说明该选项主要用于测试，返回的概率不保证有完全正确的缩放。 |

## Attention score 修饰参数

基础 scaled dot-product attention 可以写成：

$$
S_{b,h,i,j} = \alpha \cdot Q_{b,i,h,:} K_{b,j,h,:}^{T}
$$

$$
P_{b,h,i,j} = \operatorname{softmax}_{j}(S_{b,h,i,j})
$$

$$
O_{b,i,h,:} = \sum_j P_{b,h,i,j} V_{b,j,h,:}
$$

其中 $\alpha$ 就是 `softmax_scale`，默认值是 $1 / \sqrt{d_h}$。`causal`、`window_size`、`softcap` 和 `alibi_slopes` 都是在这个基础 score $S$ 上加约束或偏置，它们回答的是四类不同问题：

| 参数 | 解决的问题 | 对 score 的作用 |
| --- | --- | --- |
| `causal` | 自回归模型不能看未来 token。 | 把未来位置的 score 置为 $-\infty$。 |
| `window_size` | 长序列不一定需要看全局上下文，可以限制局部窗口。 | 把窗口外位置的 score 置为 $-\infty$。 |
| `softcap` | score 过大时 softmax 可能过尖，训练或推理中需要限制 score 幅度。 | 用 $\tanh$ 对 score 做平滑限幅。 |
| `alibi_slopes` | 不使用显式位置 embedding 时，也希望注意力带有相对位置偏置。 | 给不同距离的 key 加线性位置 bias。 |

### `causal`

**原理**

`causal=True` 表示第 `i` 个 query token 不能看它“未来”的 key token。最常见的 self-attention 场景里，`seqlen_q == seqlen_k == L`，保留条件是：

$$
j \le i
$$

也就是：

$$
S'_{b,h,i,j} =
\begin{cases}
S_{b,h,i,j}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

softmax 后，$-\infty$ 对应的概率会变成 `0`，因此输出不会使用未来 token 的 `V`。

**为什么需要**

- GPT 这类 decoder-only 模型做 next-token prediction 时，当前位置只能依赖历史 token。
- 推理 decode 时，新 token 只能看已经生成的 KV cache，不能看还没生成的未来位置。
- causal mask 是 attention 里的结构约束，不是数值优化技巧；没有它，自回归训练会发生信息泄漏。

**`seqlen_q != seqlen_k` 的情况**

这两个长度当然可以不一样。典型场景有：

| 场景 | `seqlen_q` | `seqlen_k` | 为什么不同 |
| --- | --- | --- | --- |
| decode 单步推理 | `1` | 历史 cache 长度 + 当前 token | 当前只算一个 query，但 key/value 来自完整上下文。 |
| chunked decode / 多 token decode | 当前 chunk 长度 | 历史 cache 长度 + 当前 chunk 长度 | 一次处理多个新 query，同时可以看更长的 KV cache。 |
| cross-attention | decoder 侧长度 | encoder 侧长度 | Q 来自 decoder，K/V 来自 encoder，本来就是两条序列。 |
| varlen / unpadding | 每个样本的 query 长度 | 每个样本的 key 长度 | 不同样本或不同来源序列长度不一致。 |

对于 `seqlen_q != seqlen_k` 且 `causal=True`，FA2 使用**右下角对齐**。保留条件不是简单的 $j \le i$，而是：

$$
j \le i + seqlen_k - seqlen_q
$$

等价地说，query 序列的最后一个位置和 key 序列的最后一个位置对齐。mask 公式是：

$$
S'_{b,h,i,j} =
\begin{cases}
S_{b,h,i,j}, & j \le i + seqlen_k - seqlen_q \\
-\infty, & j > i + seqlen_k - seqlen_q
\end{cases}
$$

这个设计对 decode 很自然。假设 `seqlen_q = 1`，`seqlen_k = 5`，唯一的 query 是当前新 token，它应该能看见 5 个 key：

```text
1 1 1 1 1
```

如果仍然用左上角对齐的 $j \le i$，它只能看第一个 key，明显不符合 KV cache 推理语义。

再看源码 docstring 里的例子：`seqlen_q = 2`，`seqlen_k = 5`，右下角 causal mask 是：

```text
1 1 1 1 0
1 1 1 1 1
```

第一个 query 对齐到倒数第二个 key，所以不能看最后一个 key；第二个 query 对齐到最后一个 key，所以能看全部 key。

如果 `seqlen_q = 5`，`seqlen_k = 2`，右下角 causal mask 是：

```text
0 0
0 0
0 0
1 0
1 1
```

前几行没有任何可见 key，输出会是 0。这个情况在普通 self-attention 中不常见，但接口需要覆盖更一般的 query/key 长度组合。

### `window_size`

**原理**

`window_size=(left, right)` 表示 local attention。第 `i` 个 query 只允许看一个局部 key 区间。

当 `seqlen_q == seqlen_k` 时，保留条件是：

$$
i - left \le j \le i + right
$$

也就是：

$$
S'_{b,h,i,j} =
\begin{cases}
S_{b,h,i,j}, & i - left \le j \le i + right \\
-\infty, & \text{otherwise}
\end{cases}
$$

FA2 为了统一 `seqlen_q != seqlen_k` 的场景，也采用右下角对齐。源码里的窗口边界等价于：

$$
i + seqlen_k - seqlen_q - left \le j \le i + seqlen_k - seqlen_q + right
$$

所以一般形式是：

$$
S'_{b,h,i,j} =
\begin{cases}
S_{b,h,i,j}, & i + seqlen_k - seqlen_q - left \le j \le i + seqlen_k - seqlen_q + right \\
-\infty, & \text{otherwise}
\end{cases}
$$

默认 `window_size=(-1, -1)` 表示不启用 local attention。源码里还把 causal 看成一个特殊 local mask：`window_size_left = infinity`，`window_size_right = 0`。

**为什么需要**

- 长序列 attention 的完整 score 是 $L_q \times L_k$，局部窗口可以减少有效注意力范围。
- 许多任务中当前位置主要依赖邻近 token，例如局部上下文建模、滑动窗口 LLM、长上下文分块处理。
- 对 CUDA kernel 来说，local attention 还可以跳过部分无效 tile，减少无意义的 QK/PV 工作。

**和 `causal` 的关系**

- `causal=True` 时，本质是只能看左侧历史：$j \le i$。
- `window_size=(left, right)` 更一般，可以只看左边、右边或双侧窗口。
- 在 FA2 forward dispatch 中，local 分支只在非 causal 场景单独打开；causal 被当作专门优化过的特殊情况。

### `softcap`

**原理**

`softcap > 0` 时，FA2 会对 attention score 做平滑限幅。用户视角可以理解成：

$$
\widetilde{S}_{b,h,i,j}
= C \cdot \tanh\left(\frac{S_{b,h,i,j}}{C}\right)
$$

其中 $C$ 就是用户传入的 `softcap`，$S$ 是常规缩放后的 attention score。这样当 $|S|$ 很小时：

$$
C \cdot \tanh(S / C) \approx S
$$

当 $|S|$ 很大时：

$$
C \cdot \tanh(S / C) \rightarrow \pm C
$$

所以它不会改变小 score 的局部线性行为，但会把极大或极小 score 平滑压到 `[-C, C]` 附近。

从源码实现看，C++ 参数打包时会把用户传入的 `softcap=C` 拆成：

```text
params.softcap = softmax_scale / C
params.scale_softmax = C
```

CUDA kernel 里先对 QK 累加结果调用 `tanh(score * params.softcap)`，随后 softmax 路径再乘回 `params.scale_softmax`，整体效果就是上面的 $C \cdot \tanh(\alpha QK^T / C)$。

**为什么需要**

- 普通 attention score 可能因为 Q/K 范数较大而非常尖锐，softmax 后接近 one-hot。
- score 过尖会让模型对少数位置过度自信，也可能带来数值和训练稳定性问题。
- softcap 是一种连续可导的限幅方式，比直接 clamp 更平滑。

**约束**

从 `flash_api.cpp` 看，当前 FA2 中 `softcap > 0` 时要求 `dropout_p == 0`：

```text
Softcapping does not support dropout for now
```

所以它更常出现在推理或不使用 attention dropout 的路径中。

### `alibi_slopes`

**原理**

ALiBi 的全称是 Attention with Linear Biases。它不直接修改 Q/K/V，而是在 attention score 上加入相对位置 bias。

非 causal 场景下，FA2 docstring 和源码中的语义可以写成：

$$
S'_{b,h,i,j}
= S_{b,h,i,j}
- m_{b,h} \cdot \left|i + seqlen_k - seqlen_q - j\right|
$$

其中 $m_{b,h}$ 来自 `alibi_slopes`。如果 `alibi_slopes` 形状是 `(nheads,)`，表示每个 head 一条 slope；如果是 `(batch_size, nheads)`，表示每个 batch、每个 head 都可以有自己的 slope。

右下角对齐项 `seqlen_k - seqlen_q` 的原因和 causal/window 一样：当 Q/K 长度不同，最后一个 query 与最后一个 key 对齐，距离从这个对齐关系出发计算。

causal 场景下，由于合法区域里 key 都在 query 的左侧，距离公式可以化简。源码里 causal ALiBi 分支对同一个 key column 加同一组 bias，本质上仍然是在表达“越远的历史 token，位置 bias 越不利”。

**为什么需要**

- 位置关系不一定要通过绝对位置 embedding 注入，也可以直接作为 attention score bias。
- ALiBi 让不同 head 带有不同的距离偏好：有些 head 更关注近处，有些 head 可以看得更远。
- 对长上下文外推有帮助，因为它使用相对距离的线性 bias，不依赖固定长度的位置表。

**和 mask 的区别**

- `causal` / `window_size` 是硬约束：不允许看的位置直接变成 $-\infty$，softmax 后概率为 0。
- `alibi_slopes` 是软偏置：位置越远 score 越低，但仍然可能被注意到。
- `softcap` 是数值变换：限制 score 幅度，不直接表达可见性或位置距离。

## `flash_attn_func`

**用途**

`flash_attn_func` 是最通用的定长 batch 前向接口。调用方分别传入 `q`、`k`、`v` 三个 tensor，适合先学习接口链路，也适合大多数直接使用场景。

**原型**

```python
def flash_attn_func(
    q,  # Query tensor，形状是 (batch_size, seqlen_q, nheads, headdim)。
    k,  # Key tensor，形状是 (batch_size, seqlen_k, nheads_k, headdim)。
    v,  # Value tensor，形状是 (batch_size, seqlen_k, nheads_k, headdim)。
    dropout_p=0.0,  # attention dropout 概率，推理时保持 0.0。
    softmax_scale=None,  # QK^T 的缩放系数，None 时使用 headdim ** -0.5。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口，(-1, -1) 表示全局 attention。
    softcap=0.0,  # score softcap，0.0 表示关闭。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `q` | `torch.Tensor` | Query，形状为 `(batch_size, seqlen_q, nheads, headdim)`。最后一维需要是连续或可被源码转成连续。 |
| `k` | `torch.Tensor` | Key，形状为 `(batch_size, seqlen_k, nheads_k, headdim)`。`nheads_k` 可小于 `nheads`，用于 MQA/GQA。 |
| `v` | `torch.Tensor` | Value，形状为 `(batch_size, seqlen_k, nheads_k, headdim)`，通常与 `k` 的 batch、序列长度和 KV head 数一致。 |
| `dropout_p` | `float` | attention dropout 概率，默认 `0.0`。推理时保持 `0.0`。 |
| `softmax_scale` | `Optional[float]` | softmax 前的缩放系数，默认 `None` 时使用 `q.shape[-1] ** (-0.5)`。 |
| `causal` | `bool` | 是否启用 causal mask，默认 `False`。当 `seqlen_q != seqlen_k` 时使用右下角对齐。 |
| `window_size` | `tuple[int, int]` | local attention 的左右窗口，默认 `(-1, -1)` 表示无限窗口。 |
| `softcap` | `float` | score softcap 参数，默认 `0.0` 表示关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi bias 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward，默认 `False`。只影响 backward 路径。 |
| `return_attn_probs` | `bool` | 是否额外返回 `softmax_lse` 和 `S_dmask`，默认 `False`。主要用于测试。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(batch_size, seqlen_q, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 形状为 `(batch_size, nheads, seqlen_q)`；`S_dmask` 是测试用 attention 概率 / dropout mask 信息。 |

**使用场景**

- Q、K、V 已经是三个独立 tensor。
- 需要 MQA/GQA，其中 `nheads % nheads_k == 0`。
- 初学源码时从这个接口进入最清晰：`flash_attn_func -> FlashAttnFunc.apply -> _flash_attn_forward -> flash_attn_gpu.fwd`。

## `flash_attn_qkvpacked_func`

**用途**

`flash_attn_qkvpacked_func` 用于 Q、K、V 已经存放在同一个 `qkv` tensor 的 self-attention。它能减少 Python 层拆分和 backward 时梯度重新拼接的开销。

**原型**

```python
def flash_attn_qkvpacked_func(
    qkv,  # 打包后的 Q/K/V，形状是 (batch_size, seqlen, 3, nheads, headdim)。
    dropout_p=0.0,  # attention dropout 概率。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `qkv` | `torch.Tensor` | 打包后的 Q/K/V，形状为 `(batch_size, seqlen, 3, nheads, headdim)`，其中维度 `3` 分别对应 Q、K、V。 |
| `dropout_p` | `float` | attention dropout 概率，默认 `0.0`。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数，默认 `None` 时使用 `headdim ** (-0.5)`。 |
| `causal` | `bool` | 是否启用 causal mask，默认 `False`。 |
| `window_size` | `tuple[int, int]` | local attention 的窗口范围，默认 `(-1, -1)`。 |
| `softcap` | `float` | score softcap，默认 `0.0` 表示关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward。 |
| `return_attn_probs` | `bool` | 是否额外返回测试用 attention 概率信息。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(batch_size, seqlen, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 形状为 `(batch_size, nheads, seqlen)`；`S_dmask` 主要用于测试。 |

**使用场景**

- 标准 self-attention 中，Q/K/V 由同一个线性层投影后自然组成一个 `qkv` tensor。
- backward 需要 `dqkv`，希望避免把 `dq`、`dk`、`dv` 再手动 concat。
- 不适合 MQA/GQA；源码注释建议这类场景使用 `flash_attn_kvpacked_func` 或 `flash_attn_func`。

## `flash_attn_kvpacked_func`

**用途**

`flash_attn_kvpacked_func` 用于 Q 单独存放、K/V 打包存放的定长接口。它常见于 cross-attention、MQA/GQA，或者模型内部已经把 K/V 放在同一个 tensor 的情况。

**原型**

```python
def flash_attn_kvpacked_func(
    q,  # Query tensor，形状是 (batch_size, seqlen_q, nheads, headdim)。
    kv,  # 打包后的 K/V，形状是 (batch_size, seqlen_k, 2, nheads_k, headdim)。
    dropout_p=0.0,  # attention dropout 概率。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `q` | `torch.Tensor` | Query，形状为 `(batch_size, seqlen_q, nheads, headdim)`。 |
| `kv` | `torch.Tensor` | 打包后的 K/V，形状为 `(batch_size, seqlen_k, 2, nheads_k, headdim)`，其中维度 `2` 分别对应 K、V。 |
| `dropout_p` | `float` | attention dropout 概率，默认 `0.0`。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数，默认 `None` 时使用 `headdim ** (-0.5)`。 |
| `causal` | `bool` | 是否启用 causal mask，默认 `False`。当 `seqlen_q != seqlen_k` 时使用右下角对齐。 |
| `window_size` | `tuple[int, int]` | local attention 窗口。对于 `seqlen_q != seqlen_k`，窗口位置同样按右下角对齐语义解释。 |
| `softcap` | `float` | score softcap，默认关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward。 |
| `return_attn_probs` | `bool` | 是否额外返回测试用 attention 概率信息。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(batch_size, seqlen_q, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 形状为 `(batch_size, nheads, seqlen_q)`；`S_dmask` 主要用于测试。 |

**使用场景**

- Q 与 K/V 的来源不同，例如 cross-attention。
- 使用 MQA/GQA，`nheads_k` 小于 `nheads`，并要求 `nheads` 能被 `nheads_k` 整除。
- K/V 本来就是 packed layout，backward 希望直接得到 `dkv`。

## `flash_attn_varlen_func`

**用途**

`flash_attn_varlen_func` 是最通用的变长接口。它把 batch 内所有 token 沿序列维压平，通过 `cu_seqlens_q` 和 `cu_seqlens_k` 描述每条序列的边界，从而避免 padding 带来的无效计算。

**原型**

```python
def flash_attn_varlen_func(
    q,  # 压平后的 Query，形状是 (total_q, nheads, headdim)。
    k,  # 压平后的 Key，形状是 (total_k, nheads_k, headdim)。
    v,  # 压平后的 Value，形状是 (total_k, nheads_k, headdim)。
    cu_seqlens_q,  # Query 侧累积序列长度，形状是 (batch_size + 1,)。
    cu_seqlens_k,  # Key/Value 侧累积序列长度，形状是 (batch_size + 1,)。
    max_seqlen_q,  # batch 内最大 query 长度。
    max_seqlen_k,  # batch 内最大 key/value 长度。
    dropout_p=0.0,  # attention dropout 概率。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
    block_table=None,  # 可选 paged KV / block table。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `q` | `torch.Tensor` | 压平后的 Query，形状为 `(total_q, nheads, headdim)`。 |
| `k` | `torch.Tensor` | 压平后的 Key，形状为 `(total_k, nheads_k, headdim)`。 |
| `v` | `torch.Tensor` | 压平后的 Value，形状为 `(total_k, nheads_k, headdim)`。 |
| `cu_seqlens_q` | `torch.Tensor` | Query 侧累积序列长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32`。第 `i` 条序列范围是 `[cu_seqlens_q[i], cu_seqlens_q[i + 1])`。 |
| `cu_seqlens_k` | `torch.Tensor` | Key/Value 侧累积序列长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32`。 |
| `max_seqlen_q` | `int` | batch 内最大 query 序列长度，用于 kernel launch 和 block 规模判断。 |
| `max_seqlen_k` | `int` | batch 内最大 key/value 序列长度。 |
| `dropout_p` | `float` | attention dropout 概率。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数。 |
| `causal` | `bool` | 是否启用 causal mask。 |
| `window_size` | `tuple[int, int]` | local attention 窗口。 |
| `softcap` | `float` | score softcap，默认关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward。 |
| `return_attn_probs` | `bool` | 是否额外返回测试用 attention 概率信息。 |
| `block_table` | `Optional[torch.Tensor]` | 可选 block table。从接口和调用点看，它用于底层 varlen forward 的 block / paged KV 访问路径；普通 varlen attention 可传 `None`。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(total_q, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 形状为 `(nheads, total_q)` 或与底层实现对应的变长布局；`S_dmask` 主要用于测试。 |

**使用场景**

- batch 内序列长度差异很大，不希望 padding 到最大长度。
- 数据已经经过 unpad / pack，形成 `(total, ...)` 形状。
- 需要支持 Q 和 K/V 长度不同的变长 attention。

## `flash_attn_varlen_qkvpacked_func`

**用途**

`flash_attn_varlen_qkvpacked_func` 是变长 self-attention 的 packed 版本。它假设每个 token 的 Q/K/V 都在同一个 `qkv` tensor 中，并且 Q、K、V 的序列边界相同。

**原型**

```python
def flash_attn_varlen_qkvpacked_func(
    qkv,  # 压平后的 Q/K/V，形状是 (total, 3, nheads, headdim)。
    cu_seqlens,  # 累积序列长度，形状是 (batch_size + 1,)。
    max_seqlen,  # batch 内最大序列长度。
    dropout_p=0.0,  # attention dropout 概率。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `qkv` | `torch.Tensor` | 压平后的 Q/K/V，形状为 `(total, 3, nheads, headdim)`，`total` 是 batch 中 token 总数。 |
| `cu_seqlens` | `torch.Tensor` | 累积序列长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32`。同一组边界同时用于 Q、K、V。 |
| `max_seqlen` | `int` | batch 内最大序列长度。 |
| `dropout_p` | `float` | attention dropout 概率。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数。 |
| `causal` | `bool` | 是否启用 causal mask。 |
| `window_size` | `tuple[int, int]` | local attention 窗口，默认不限制。 |
| `softcap` | `float` | score softcap，默认关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward。 |
| `return_attn_probs` | `bool` | 是否额外返回测试用 attention 概率信息。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(total, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 对应每个 query token 的 logsumexp；`S_dmask` 主要用于测试。 |

**使用场景**

- 变长 self-attention，Q/K/V 序列边界完全相同。
- 上游已经把 padded batch unpad 成 `(total, 3, nheads, headdim)`。
- 训练时希望 backward 直接返回 `dqkv`。

## `flash_attn_varlen_kvpacked_func`

**用途**

`flash_attn_varlen_kvpacked_func` 是变长 attention 的 KV packed 版本。它适合 Q 与 K/V 序列边界不同，且 K/V 已经打包在一个 tensor 的场景。

**原型**

```python
def flash_attn_varlen_kvpacked_func(
    q,  # 压平后的 Query，形状是 (total_q, nheads, headdim)。
    kv,  # 压平后的 K/V，形状是 (total_k, 2, nheads_k, headdim)。
    cu_seqlens_q,  # Query 侧累积序列长度。
    cu_seqlens_k,  # Key/Value 侧累积序列长度。
    max_seqlen_q,  # batch 内最大 query 长度。
    max_seqlen_k,  # batch 内最大 key/value 长度。
    dropout_p=0.0,  # attention dropout 概率。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    deterministic=False,  # 是否使用确定性 backward。
    return_attn_probs=False,  # 是否返回测试用 attention 概率信息。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `q` | `torch.Tensor` | 压平后的 Query，形状为 `(total_q, nheads, headdim)`。 |
| `kv` | `torch.Tensor` | 压平后的 K/V，形状为 `(total_k, 2, nheads_k, headdim)`。 |
| `cu_seqlens_q` | `torch.Tensor` | Query 侧累积序列长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32`。 |
| `cu_seqlens_k` | `torch.Tensor` | Key/Value 侧累积序列长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32`。 |
| `max_seqlen_q` | `int` | batch 内最大 query 序列长度。 |
| `max_seqlen_k` | `int` | batch 内最大 key/value 序列长度。 |
| `dropout_p` | `float` | attention dropout 概率。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数。 |
| `causal` | `bool` | 是否启用 causal mask。 |
| `window_size` | `tuple[int, int]` | local attention 窗口。 |
| `softcap` | `float` | score softcap，默认关闭。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `deterministic` | `bool` | 是否使用确定性 backward。 |
| `return_attn_probs` | `bool` | 是否额外返回测试用 attention 概率信息。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(total_q, nheads, headdim)`。 |
| `(out, softmax_lse, S_dmask)` | `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | 当 `return_attn_probs=True` 时返回。`softmax_lse` 对应每个 query token 的 logsumexp；`S_dmask` 主要用于测试。 |

**使用场景**

- 变长 cross-attention。
- 变长 MQA/GQA，其中 `nheads_k` 小于 `nheads`。
- K/V 已经打包，backward 希望直接得到 `dkv`。

## `flash_attn_with_kvcache`

**用途**

`flash_attn_with_kvcache` 是推理场景的前向接口。它读取已经分配好的 `k_cache` 和 `v_cache`，可选地把新来的 `k`、`v` 原地写入 cache，然后基于更新后的 cache 完成 attention。源码明确说明该接口 **不支持 backward**。

**原型**

```python
def flash_attn_with_kvcache(
    q,  # 当前 query，形状是 (batch_size, seqlen, nheads, headdim)。
    k_cache,  # 历史 key cache，普通 cache 或 paged KV cache 布局。
    v_cache,  # 历史 value cache，布局与 k_cache 对应。
    k=None,  # 可选的新 key，会被写入 k_cache。
    v=None,  # 可选的新 value，会被写入 v_cache。
    rotary_cos=None,  # 可选 RoPE cos 表。
    rotary_sin=None,  # 可选 RoPE sin 表。
    cache_seqlens=None,  # 每条序列当前 cache 长度。
    cache_batch_idx=None,  # 当前 batch 到 cache batch 的索引映射。
    cache_leftpad=None,  # KV cache 左侧 padding 起点。
    block_table=None,  # paged KV cache 的 block table。
    softmax_scale=None,  # QK^T 的缩放系数。
    causal=False,  # 是否启用 causal mask。
    window_size=(-1, -1),  # local attention 窗口。
    softcap=0.0,  # score softcap。
    rotary_interleaved=True,  # RoPE 是否使用 interleaved 维度布局。
    alibi_slopes=None,  # 可选 ALiBi 斜率。
    num_splits=0,  # split-KV 的切分数量，0 表示启发式选择。
    return_softmax_lse=False,  # 是否返回 softmax_lse。
):
    ...
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `q` | `torch.Tensor` | 当前 query，形状为 `(batch_size, seqlen, nheads, headdim)`。 |
| `k_cache` | `torch.Tensor` | Key cache。普通布局为 `(batch_size_cache, seqlen_cache, nheads_k, headdim)`；若使用 `block_table`，布局为 `(num_blocks, page_block_size, nheads_k, headdim)`。源码要求最后一维连续。 |
| `v_cache` | `torch.Tensor` | Value cache，布局与 `k_cache` 对应。源码要求最后一维连续。 |
| `k` | `Optional[torch.Tensor]` | 新写入 cache 的 key，形状为 `(batch_size, seqlen_new, nheads_k, headdim)`。若不为 `None`，会从 `cache_seqlens` 指定的位置写入 `k_cache`。 |
| `v` | `Optional[torch.Tensor]` | 新写入 cache 的 value，形状与 `k` 对应。 |
| `rotary_cos` | `Optional[torch.Tensor]` | RoPE cos 表，形状为 `(seqlen_ro, rotary_dim / 2)`。只有传入 `k` 和 `v` 时才适用。 |
| `rotary_sin` | `Optional[torch.Tensor]` | RoPE sin 表，形状与 `rotary_cos` 对应。 |
| `cache_seqlens` | `Optional[Union[int, torch.Tensor]]` | KV cache 中每条序列已有长度。若是 `int`，源码会扩展成形状 `(batch_size,)`、dtype 为 `torch.int32` 的 tensor。 |
| `cache_batch_idx` | `Optional[torch.Tensor]` | 当前 batch 到 cache batch 的索引，形状为 `(batch_size,)`，dtype 为 `torch.int32`。为 `None` 时默认使用 `[0, 1, ..., batch_size - 1]`。 |
| `cache_leftpad` | `Optional[torch.Tensor]` | 每条序列在 KV cache 中的左 padding 起点，形状为 `(batch_size,)`，dtype 为 `torch.int32`。 |
| `block_table` | `Optional[torch.Tensor]` | paged KV cache 的 block table，形状为 `(batch_size, max_num_blocks_per_seq)`，dtype 为 `torch.int32`。 |
| `softmax_scale` | `Optional[float]` | softmax 前缩放系数，默认使用 `headdim ** (-0.5)`。 |
| `causal` | `bool` | 是否启用 causal mask。 |
| `window_size` | `tuple[int, int]` | local attention 窗口。 |
| `softcap` | `float` | score softcap，默认关闭。 |
| `rotary_interleaved` | `bool` | RoPE 维度布局。`True` 表示交错组合维度 `0&1, 2&3`；`False` 表示 GPT-NeoX 风格的前后半区组合。 |
| `alibi_slopes` | `Optional[torch.Tensor]` | ALiBi 斜率，形状为 `(nheads,)` 或 `(batch_size, nheads)`。 |
| `num_splits` | `int` | split-KV 数量。`0` 表示使用启发式自动决定；`1` 表示不切分；大于 `1` 表示沿 KV 序列切成多个 chunk。源码注释建议不了解时不要改。 |
| `return_softmax_lse` | `bool` | 是否返回 attention score 的 logsumexp。 |

**返回值**

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| `out` | `torch.Tensor` | 默认返回值，形状为 `(batch_size, seqlen, nheads, headdim)`。 |
| `(out, softmax_lse)` | `tuple[torch.Tensor, torch.Tensor]` | 当 `return_softmax_lse=True` 时返回。`softmax_lse` 形状为 `(batch_size, nheads, seqlen)`。 |

**副作用 / 约束**

- 如果 `k` 和 `v` 不为 `None`，该接口会 **原地更新** `k_cache` 和 `v_cache`。
- `k_cache.stride(-1) == 1` 且 `v_cache.stride(-1) == 1`，否则源码会触发 `assert`。
- 使用 paged KV cache 时，`page_block_size` 必须是 `256` 的倍数。
- 该接口用于推理，不支持 backward。

**使用场景**

- LLM decode 阶段，每一步只处理少量新 token。
- 已经预分配 KV cache，希望在一个 kernel 中完成“写入新 KV + 读取历史 KV + attention”。
- 使用 paged KV cache 或 split-KV 优化长上下文 decode。

## 定长、变长和 KV cache 的选择

```mermaid
flowchart TD
    A["准备调用 FA2 前向"] --> B{"是否是 decode / KV cache 推理"}
    B -- "是" --> C["flash_attn_with_kvcache"]
    B -- "否" --> D{"batch 内序列是否已压平为 total tokens"}
    D -- "是" --> E{"QKV 是否打包"}
    D -- "否" --> F{"QKV 是否打包"}
    E -- "QKV 打包" --> G["flash_attn_varlen_qkvpacked_func"]
    E -- "KV 打包" --> H["flash_attn_varlen_kvpacked_func"]
    E -- "都不打包" --> I["flash_attn_varlen_func"]
    F -- "QKV 打包" --> J["flash_attn_qkvpacked_func"]
    F -- "KV 打包" --> K["flash_attn_kvpacked_func"]
    F -- "都不打包" --> L["flash_attn_func"]
```

这张图只按 Python 接口选择来理解。真正的 CUDA kernel dispatch 还会继续根据 dtype、head_dim、causal、dropout、local attention、是否 split-KV 等条件选择模板实例。

## 从源码推断的内部包装关系

公开接口本身不直接调用 C++ extension，而是先进入对应的 `torch.autograd.Function`。这样 Python 层可以保存 backward 需要的 tensor，例如 `q`、`k`、`v`、`out_padded`、`softmax_lse`、`rng_state`。

| 公开接口 | Autograd wrapper | 底层 forward |
| --- | --- | --- |
| `flash_attn_func` | `FlashAttnFunc` | `_wrapped_flash_attn_forward` -> `flash_attn_gpu.fwd` |
| `flash_attn_qkvpacked_func` | `FlashAttnQKVPackedFunc` | `_wrapped_flash_attn_forward` -> `flash_attn_gpu.fwd` |
| `flash_attn_kvpacked_func` | `FlashAttnKVPackedFunc` | `_wrapped_flash_attn_forward` -> `flash_attn_gpu.fwd` |
| `flash_attn_varlen_func` | `FlashAttnVarlenFunc` | `_wrapped_flash_attn_varlen_forward` -> `flash_attn_gpu.varlen_fwd` |
| `flash_attn_varlen_qkvpacked_func` | `FlashAttnVarlenQKVPackedFunc` | `_wrapped_flash_attn_varlen_forward` -> `flash_attn_gpu.varlen_fwd` |
| `flash_attn_varlen_kvpacked_func` | `FlashAttnVarlenKVPackedFunc` | `_wrapped_flash_attn_varlen_forward` -> `flash_attn_gpu.varlen_fwd` |
| `flash_attn_with_kvcache` | 无 autograd wrapper | `flash_attn_gpu.fwd_kvcache` |

注意 `qkvpacked` 和 `kvpacked` 的“打包”主要影响 Python 层如何从输入 tensor 中切出 `q/k/v`，以及 backward 如何组织梯度返回；进入底层 CUDA forward 时，仍然会把 Q、K、V 作为独立 tensor 传入。

## 注意点

- **head_dim padding**：从实现看，如果 `headdim` 不是 `8` 的倍数，Python wrapper 会在最后一维 pad 到 `8` 的倍数，返回前再裁掉原始 `headdim`。
- **last dimension contiguity（最后一维连续）**：源码通过 `maybe_contiguous` 保证很多输入的最后一维连续；`flash_attn_with_kvcache` 对 cache tensor 直接用 `assert` 检查。
- **MQA/GQA 约束**：当 `nheads_k < nheads` 时，`nheads` 必须能被 `nheads_k` 整除。每个 KV head 会服务一组 query heads。
- **`return_attn_probs` 不是常规训练输出**：源码注释说明它主要用于测试，返回的概率不保证有完全正确的缩放，并且会带来额外内存开销。
- **causal mask 右下角对齐**：当 `seqlen_q != seqlen_k` 时，FA2 的 causal mask 不是简单左上角对齐，而是按右下角对齐，用于支持 decode / cross-length 场景。
- **Triton ROCm 分支不是 NVIDIA FA2 主路径**：文件顶部的 `USE_TRITON_ROCM` 是 AMD ROCm fallback / backend 选择。NVIDIA 上正常使用的是 `flash_attn_2_cuda`。

## 本节小结

这一节的目标不是停在 Python API，而是为后续读 C++/CUDA kernel 建立参数和数据布局的入口地图。先抓住这几类调用形态：

```python
# 最通用的定长入口。
flash_attn_func(q, k, v)

# self-attention 中 Q/K/V 已经打包。
flash_attn_qkvpacked_func(qkv)

# K/V 已经打包，Q 单独传入。
flash_attn_kvpacked_func(q, kv)

# 变长 batch：把 token 压平，并用 cu_seqlens 描述边界。
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

# 推理 decode：复用并可原地更新 KV cache。
flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens)
```

这些接口的差别，本质上是为了让调用方按已有 tensor layout 直接进入高性能路径，减少额外的 reshape、concat、padding 和无效 token 计算。下一步读源码时，可以先固定最简单的 `flash_attn_func(q, k, v)` 路径：

```text
flash_attn_func
-> FlashAttnFunc.apply
-> _wrapped_flash_attn_forward
-> flash_attn_gpu.fwd
-> csrc/flash_attn/flash_api.cpp::mha_fwd
-> csrc/flash_attn/flash_api.cpp::run_mha_fwd
-> csrc/flash_attn/src/flash_fwd_launch_template.h
-> csrc/flash_attn/src/flash_fwd_kernel.h
```

也就是说，本节解决“Python 入口和参数语义”，后续章节会继续解决“C++ 如何 dispatch”和“CUDA kernel 如何计算 attention”。
