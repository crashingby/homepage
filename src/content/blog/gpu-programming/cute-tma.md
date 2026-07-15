---
title: CuTe TMA 设计与 API 笔记
date: 2026-07-15
tags: [CUDA, CuTe, CUTLASS, TMA, Hopper, SM90]
summary: 对照 CUTLASS CuTe 源码整理 SM90 TMA 的 tensor map 描述符、swizzle、make_tma_atom / tma_partition、SM90_TMA_LOAD 系列 API、cluster launch 和 kernel 内部 barrier / pipeline 协议。
---

# CuTe TMA 设计与 API 笔记

TMA 是 Hopper / SM90 引入的 Tensor Memory Accelerator。它不是传统的“每个线程手写地址、从 global load 到寄存器、再 store 到 shared”的搬运方式，而是把一块多维 tensor tile 描述成一个 **tensor map descriptor**，然后用 `cp.async.bulk.tensor` 这类指令让硬件负责地址生成和异步搬运。

在 CUDA 原始 API 里，TMA 的重点是“如何描述一个多维 tensor”：全局内存基地址、rank、每一维大小、stride、一次搬运的 box shape、shared memory swizzle、L2 promotion、越界填充值等。

在 CuTe 里，这些信息被拆进了更高层的对象：

| CUDA / PTX 视角 | CuTe 里的来源 | 作用 |
| --- | --- | --- |
| `CUtensorMap` / TMA descriptor | `make_tma_copy`、`make_tma_atom` 内部生成的 `TmaDescriptor` | 描述 global tensor、TMA box、swizzle、L2 policy。 |
| global rank / shape / stride | `Tensor<GEngine, GLayout>` | 从 GMEM tensor 的 layout 推导。 |
| box shape / element stride | `SLayout` 和 `CTA_Tiler` | 从 SMEM tile 和 CTA tile 推导一次 TMA 指令搬多少。 |
| shared-memory swizzle | `ComposedLayout<Swizzle<...>, smem_ptr_flag, Layout<...>>` | 同时服务 TMA 写入和 GMMA 读取。 |
| async completion | `ClusterTransactionBarrier` / `mbarrier` | TMA load 完成时按 transaction bytes 通知 barrier。 |
| multicast | `cluster_size`、`cta_layout`、`create_tma_multicast_mask` | 一个 TMA load 可以把数据送到 cluster 中多个 CTA 的 shared memory。 |

所以 CuTe TMA 的核心问题是：

> 用 CuTe 的 `Tensor` / `Layout` / `Copy_Atom` 把 CUDA tensor map 描述符、PTX TMA 指令和 kernel 里的 barrier 协议组织起来。

这篇笔记主要对照这些源码：

| 源码文件 | 主要内容 |
| --- | --- |
| `cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp` | GMMA shared memory swizzle layout：`Layout_MN_SW128_Atom` 等。 |
| `cutlass/include/cute/atom/copy_traits_sm90_tma.hpp` | `make_tma_copy`、`make_tma_atom`、`tma_partition`、multicast mask。 |
| `cutlass/include/cute/arch/copy_sm90_tma.hpp` | `SM90_TMA_LOAD`、`SM90_TMA_STORE`、`SM90_TMA_REDUCE_ADD` 等 arch-level TMA 指令封装。 |
| `cutlass/include/cute/arch/copy_sm90_desc.hpp` | `TmaDescriptor`、swizzle / L2 / OOB fill 到 `CUtensorMap` 枚举的映射。 |
| `cutlass/include/cute/arch/cluster_sm90.hpp` | cluster-level barrier 和 CTA rank 查询。 |
| `cutlass/include/cutlass/cluster_launch.hpp` | host 侧 cluster launch 封装。 |
| `cutlass/include/cutlass/arch/barrier.h` | `ClusterBarrier`、`ClusterTransactionBarrier`。 |
| `cutlass/include/cutlass/pipeline/sm90_pipeline.hpp` | `PipelineState` 环形 stage 和 phase。 |
| `cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` | CuTe TMA + WGMMA 教程示例，本文只看 TMA 相关部分。 |

## CUDA TMA 描述符到底描述什么

TMA descriptor 可以理解成“给硬件看的多维 tensor 地址公式”。对 rank 为 $r$ 的 tensor，硬件拿到一个 TMA 坐标：

$$
c = (c_0, c_1, \dots, c_{r-1})
$$

然后根据 descriptor 计算 global memory 地址。CuTe 源码里会检查：

```cpp
assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");
```

也就是说 TMA 的第 0 维是 fastest-changing 维度，它在元素单位上的 stride 隐含为 1。换成字节地址，大致是：

$$
\text{addr}(c) =
\text{base} +
c_0 \cdot \text{sizeof}(T) +
\sum_{d=1}^{r-1} c_d \cdot \text{globalStrideBytes}_d
$$

这里的 `globalStrideBytes[d]` 是 descriptor 里存的 byte stride。CuTe 会先从 GMEM tensor 的 layout 得到元素 stride，再乘上 `sizeof(TmaInternalType)` 转成 byte stride。

descriptor 还描述一次 TMA 指令搬运的 box：

| 字段 | 含义 | CuTe 里的来源 |
| --- | --- | --- |
| `rank` / `dim` | TMA tensor rank，硬件最多支持 5D。 | `tma_gbasis` 的 rank，CuTe 会把多余尾部模式 group 到最多 5D。 |
| `globalDim` | global tensor 每个 TMA 维度的大小。 | `gtensor` 的 shape 加上 TMA basis 映射。 |
| `globalStrides` | global tensor 第 1 维及之后的 byte stride。第 0 维隐含 contiguous。 | `gtensor` 的 stride。 |
| `boxDim` | 一次 TMA 指令搬的 tile shape。 | `SLayout` / `CTA_Tiler` 推导出的 SMEM box shape。 |
| `elementStrides` | box 内每个 TMA 维度的 element stride。 | 通常是 `{1,1,1,1,1}`，im2col / gather-scatter 会更复杂。 |
| `interleave` | global tensor map interleave。 | CuTe 常规路径里用 `CU_TENSOR_MAP_INTERLEAVE_NONE`。 |
| `swizzle` | shared memory swizzle。 | 从 SMEM layout 的 `Swizzle<B,M,S>` 推导。 |
| `l2Promotion` | L2 promotion hint。 | CuTe 常规路径默认 `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`。 |
| `oobFill` | 越界填充值策略。 | CuTe 常规路径默认 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`。 |

先看 CUDA 原始写法。下面这段是2D 示例：Host 侧创建 `CUtensorMap`，Device 侧把它作为 kernel 参数使用。

```cpp
#include <cuda.h>
#include <cudaTypedefs.h>

constexpr uint64_t GMEM_WIDTH  = 4096;
constexpr uint64_t GMEM_HEIGHT = 4096;
constexpr uint32_t SMEM_WIDTH  = 64;
constexpr uint32_t SMEM_HEIGHT = 64;

/**
 * @brief 为一个 2D row-major int tensor 创建 TMA tensor map。
 *
 * @param tensor_ptr global memory 指针，要求满足 TMA descriptor 的对齐约束。
 * @return 可以传给 kernel 的 `CUtensorMap`。
 */
CUtensorMap make_tensor_map(int* tensor_ptr) {
    CUtensorMap tensor_map{};

    // TMA 文档习惯把最快变化维度放在前面。
    // 对 row-major int[GMEM_HEIGHT][GMEM_WIDTH] 来说，x 是第 0 维。
    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank] = {
        GMEM_WIDTH,
        GMEM_HEIGHT,
    };

    // `global_strides` 的单位是字节，不是元素个数。
    // rank=2 时只需要 rank-1 个 stride，因为第 0 维 stride 隐含为 1 个元素。
    uint64_t global_strides[rank - 1] = {
        GMEM_WIDTH * sizeof(int),
    };

    // `box_dim` 描述一次 TMA 指令搬到 shared memory 的 tile 大小。
    uint32_t box_dim[rank] = {
        SMEM_WIDTH,
        SMEM_HEIGHT,
    };

    // `element_strides` 表示 box 内每一维的采样步长。
    // `{1, 1}` 表示每个元素都搬，不做稀疏采样。
    uint32_t element_strides[rank] = {1, 1};

    auto encode_tiled = get_cuTensorMapEncodeTiled();
    CUresult result = encode_tiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,
        tensor_ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // 真实代码里要检查 result，这里只突出 descriptor 字段。
    return tensor_map;
}
```

这样前面的名词就能和 API 参数直接对上：

| 名词 | CUDA 代码里的变量 / 参数 | 注意点 |
| --- | --- | --- |
| base address | `tensor_ptr` | global memory 基地址，TMA 要求对齐。 |
| rank | `rank` | tensor 维度，bulk-tensor TMA 支持 1D 到 5D。 |
| global shape | `global_dim` | 单位是元素个数。 |
| global stride | `global_strides` | 单位是字节；不包含第 0 维 stride。 |
| box shape | `box_dim` | 一条 TMA 指令搬到 shared memory 的 tile shape。 |
| element stride | `element_strides` | box 内每一维采样步长。 |
| interleave | `CU_TENSOR_MAP_INTERLEAVE_NONE` | 常规 GEMM 路径通常不用 interleave。 |
| swizzle | `CU_TENSOR_MAP_SWIZZLE_NONE` / `CU_TENSOR_MAP_SWIZZLE_128B` | 决定 TMA 写入 shared memory 时是否做 swizzle。 |
| L2 promotion | `CU_TENSOR_MAP_L2_PROMOTION_NONE` | L2 hint。CuTe 常规路径默认 128B promotion。 |
| OOB fill | `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` | 越界填充策略。 |

如果要启用 128B swizzle，CUDA 代码里改的是 `cuTensorMapEncodeTiled` 的 swizzle 参数：

```cpp
CUresult result = encode_tiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,
    tensor_ptr,
    global_dim,
    global_strides,
    box_dim,
    element_strides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

CuTe 做的事情，本质上就是把这些数组和枚举从 `Tensor` / `Layout` 里推导出来。源码里的真实函数名是 `make_tma_copy_desc`，在 `cute/atom/copy_traits_sm90_tma.hpp`。下面是裁剪后的关键片段，不是一个新造 API。

```cpp
/**
 * @brief 从 CuTe GMEM tensor、TMA basis 和 SMEM swizzle 构造 `TmaDescriptor`。
 *
 * @tparam TmaInternalType descriptor 中使用的元素类型。
 * @tparam GEngine GMEM tensor 的 engine 类型。
 * @tparam GLayout GMEM tensor 的 layout 类型。
 * @tparam TShape `tma_gbasis` 的 shape 类型。
 * @tparam TStride `tma_gbasis` 的 stride 类型。
 * @tparam B SMEM swizzle 的位宽编码。
 * @tparam M SMEM swizzle 的 base 编码。
 * @tparam S SMEM swizzle 的 shift 编码。
 * @param gtensor 原始 GMEM tensor，对应 CUDA 示例里的 `tensor_ptr + global_dim + global_strides`。
 * @param tma_gbasis TMA 维度到 GMEM 逻辑维度的映射，用来推导 `global_dim` 和 `box_dim`。
 * @param swizzle 从 SMEM layout 中抽出的 swizzle，对应 `CUtensorMapSwizzle`。
 * @param num_multicast multicast CTA 数，用来调整每个 CTA 实际接收的 SMEM box。
 * @return CuTe TMA copy traits 需要的 descriptor 和辅助 stride 信息。
 */
template <class TmaInternalType,
          class GEngine, class GLayout,
          class TShape, class TStride,
          int B, int M, int S>
CUTE_HOST_RTC
auto make_tma_copy_desc(Tensor<GEngine,GLayout> const& gtensor,
                        Layout<TShape,TStride>  const& tma_gbasis,
                        Swizzle<B,M,S>          const& swizzle,
                        uint32_t                       num_multicast) {
    constexpr int tma_dim = decltype(rank(tma_gbasis))::value;

    Tensor gtensor_T = recast<TmaInternalType>(gtensor);
    void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());

    cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
    cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
    fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis),
                               gmem_prob_shape, gmem_prob_stride);

    // CUDA tensor map 不存第 0 维 stride，因此 CuTe 要求第 0 维是元素连续维。
    assert(gmem_prob_stride[0] == 1 &&
           "Majorness of smem doesn't match majorness of gmem");

    // CUDA tensor map 的 global stride 单位是字节。
    for (uint64_t& stride : gmem_prob_stride) {
        stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
    }

    cute::array<uint32_t, 5> smem_box_shape  = {1,1,1,1,1};
    cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};
    for_each(make_seq<tma_dim>{}, [&](auto i) {
        smem_box_shape[i] *= size<i>(tma_gbasis);
    });

    TmaDescriptor tma_desc{};

    CUtensorMapDataType tma_format =
        TMA::to_CUtensorMapDataType<TmaInternalType>();
    CUtensorMapInterleave tma_interleave =
        CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion tma_l2Promotion =
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    CUtensorMapFloatOOBfill tma_oobFill =
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);
    TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);
    CUtensorMapSwizzle smem_swizzle =
        TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);

    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,
        tma_format,
        tma_dim,
        gmem_address,
        gmem_prob_shape.data(),
        gmem_prob_stride.data() + 1,  // 第 0 维 stride 隐含为 1。
        smem_box_shape.data(),
        smem_box_stride.data(),
        tma_interleave,
        smem_swizzle,
        tma_l2Promotion,
        tma_oobFill);
}
```

这段代码里有两个很重要的约束：

- `gmem_address` 必须 16B 对齐。
- 第 0 维必须是 contiguous 维度，`gmem_prob_stride[0] == 1`。

这就是为什么 CuTe 的 TMA 不是“任意 layout 都能直接 TMA”。它会尽量根据 SMEM layout 找最大 contiguous vector，但最终必须符合硬件 tensor map 的限制。

## GMMA swizzle layout 和 TMA swizzle 的关系

TMA 往 shared memory 写，GMMA 从 shared memory 读。为了让两者都高效，CuTe 通常用 GMMA 认可的 swizzled shared-memory layout 来定义 `sA` / `sB`。

这里有三个容易混在一起的概念：

- **地址 bit swizzle**：shared memory 的最终地址不是直接用线性 offset，而是把地址里的某些 bit 做 XOR / 重排，让访问分散到不同 bank。
- **bit 单位 layout**：CuTe 源码里带 `_Bits` 后缀的 atom，layout 的 offset 单位是 bit，不是元素。
- **元素单位 layout**：用户写 `half_t` / `float` kernel 时，坐标通常按元素数理解。CuTe 用 `upcast<sizeof_bits<Type>::value>` 把 bit 单位 layout 转成元素单位 layout。

### 什么是地址 bit swizzle

shared memory bank conflict 本质上和地址低位有关。假设逻辑上有一个二维 tile：

$$
\text{logical\_offset} = i \cdot s_i + j \cdot s_j
$$

如果直接拿这个 offset 当 shared memory 地址，不同线程可能频繁落到同一个 bank。swizzle 的思路是：**逻辑坐标不变，但最终 shared memory 地址的若干 bit 会被重排**。

可以粗略理解成：

$$
\text{physical\_offset} = \text{swizzle}(\text{logical\_offset})
$$

在 CuTe 源码里，这个动作体现为：

```cpp
ComposedLayout<Swizzle<B,M,S>, smem_ptr_flag, Layout<Shape<...>,Stride<...>>>
```

也就是：

```text
逻辑坐标 -> 普通 Layout 算出 offset -> Swizzle 改写 offset 的地址 bit -> shared memory 地址
```

这里的 `Swizzle<B,M,S>` 是 CuTe 对 swizzle pattern 的类型级编码。对 SM90 GMMA 这组 layout，`M=4`、`S=3` 是固定形态，`B=0/1/2/3` 决定它最后对应 interleave / 32B / 64B / 128B swizzle。

注意这里说的 bit swizzle 不是“元素值的 bit 被打乱”。数据值本身不变，变的是 **shared memory 地址的映射**。

### 为什么源码先用 bit 单位 layout

`mma_traits_sm90_gmma.hpp` 里有一组常用 layout atom：

```cpp
/**
 * @brief SM90 GMMA 使用的 M/N-major shared-memory layout atom，单位是 bit。
 *
 * @details
 * 这些 `_Bits` layout 描述的是地址 bit offset。这样同一套 swizzle pattern
 * 可以服务 `half_t`、`float`、FP8 等不同元素类型，后面再按元素 bit 宽转换。
 */
using Layout_MN_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                   Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;
```

以 `Layout_MN_SW128_Atom_Bits` 为例：

```text
Shape  = (1024, 8)
Stride = (1, 1024)
单位   = bit
```

这表示普通 layout 部分先把 `(i,j)` 映射成 bit offset：

$$
\text{bit\_offset}(i,j) = i + 1024j
$$

然后 `Swizzle<3,4,3>` 再对这个 bit offset 做地址 swizzle。这里 `1024 bit = 128B`，所以它对应 CUDA tensor map 里的 `CU_TENSOR_MAP_SWIZZLE_128B`。

为什么不直接写成元素单位？因为元素类型不同：

| 类型 | 元素 bit 数 | 1024 bit 等于多少元素 |
| --- | --- | --- |
| `half_t` / FP16 | 16 | 64 个元素 |
| `float` / FP32 | 32 | 32 个元素 |
| FP8 | 8 | 128 个元素 |

用 bit 单位先描述硬件地址 pattern，再转成元素单位，可以避免为每个元素类型手写一套 atom。

### 元素单位的 swizzle 是怎么来的

源码里的类型别名是：

```cpp
/**
 * @brief 把 bit 单位的 GMMA layout atom 转成 Type 元素单位的 layout atom。
 *
 * @details
 * `sizeof_bits<Type>::value` 是一个元素占多少 bit。`upcast<N>` 会把 bit offset
 * 缩放成以 N-bit 元素为单位的 offset。
 */
template <class Type>
using Layout_MN_SW128_Atom =
    decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));
```

`upcast` 在 `cute/layout.hpp` 里的规则可以概括成：

```cpp
/**
 * @brief 把 layout 的 offset 单位从更小粒度提升到 N 倍粒度。
 *
 * @details
 * 对 stride 为 1 的连续 mode，shape 会除以 N。
 * 对其他静态 stride，stride 会除以 N。
 */
template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto upcast(Shape const& shape, Stride const& stride);
```

所以对于 FP16：

```cpp
using AtomBits = GMMA::Layout_MN_SW128_Atom_Bits;
using AtomF16  = GMMA::Layout_MN_SW128_Atom<half_t>;
```

可以按下面这样理解：

```text
bit 单位:
  Shape  = (1024, 8)
  Stride = (1, 1024)

FP16 元素单位，N = sizeof_bits<half_t> = 16:
  Shape  = (64, 8)
  Stride = (1, 64)
```

也就是说，`Layout_MN_SW128_Atom<half_t>` 的坐标已经按 half 元素来数，但底层 swizzle pattern 仍然是同一个 128B shared-memory swizzle。

所以“元素单位的 swizzle”不是另一种硬件 swizzle，而是同一个 **地址 bit swizzle** 在元素坐标系统里的表示。

### `Major::MN` 和 `Major::K` 的区别

GMMA 的 operand 通常可以看成二维 shared-memory tensor：

- A operand：逻辑坐标是 `(M, K)`。
- B operand：逻辑坐标是 `(N, K)`。

`Major::MN` 和 `Major::K` 描述的是：**GMMA descriptor 认为哪个逻辑方向是主要连续方向**。

| major | 对 A operand 的含义 | 对 B operand 的含义 | 直观理解 |
| --- | --- | --- | --- |
| `GMMA::Major::MN` | M 方向优先 / M-major | N 方向优先 / N-major | 非 K 维连续或更内层。 |
| `GMMA::Major::K` | K 方向优先 / K-major | K 方向优先 / K-major | K 维连续或更内层。 |

这会影响两个地方：

- 选择哪一种 SMEM layout atom：`Layout_MN_SW128_Atom<T>` 或 `Layout_K_SW128_Atom<T>`。
- 构造哪一种 GMMA descriptor：`make_gmma_desc<Major::MN>` 或 `make_gmma_desc<Major::K>`，以及 `SM90_...<MajorA, MajorB>` 的模板参数。

源码里的 canonical form 也能看出差别：

```cpp
/**
 * @brief Major-MN 的 descriptor canonical layout。
 *
 * @details
 * M/N-major 版本把 `(T, ..., m)` 放在第一组 mode，
 * 第二组 mode 是 `(8, k)`。它服务 A 的 M-major 或 B 的 N-major。
 */
// make_gmma_desc<Major::MN>
// LayoutType::B128:
//   Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))

/**
 * @brief Major-K 的 descriptor canonical layout。
 *
 * @details
 * K-major 版本把 `(T, 2)` 放在第二组 mode，
 * 让 K 方向成为 GMMA 认可的主要连续方向。
 */
// make_gmma_desc<Major::K>
// LayoutType::B128:
//   Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1,T))
```

`mma_traits_sm90_gmma.hpp` 里有一组常用 layout atom：

```cpp
/**
 * @brief SM90 GMMA 使用的 M/N-major shared-memory layout atom。
 *
 * @details
 * 这些 layout 先以 bit 为单位描述基本 tile，再通过 upcast 转成元素单位。
 * `Swizzle<0,4,3>` 表示无 32/64/128B swizzle，`Swizzle<1,4,3>` 到
 * `Swizzle<3,4,3>` 分别对应 32B、64B、128B swizzle。
 */
using Layout_MN_INTER_Atom_Bits =
    ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag,
                   Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  =
    ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag,
                   Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  =
    ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag,
                   Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                   Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

/**
 * @brief SM90 GMMA 使用的 K-major shared-memory layout atom。
 *
 * @details
 * K-major 版本把 contiguous 方向放在 K 维，适合 B 矩阵或 K-major operand。
 */
using Layout_K_INTER_Atom_Bits =
    ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag,
                   Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits  =
    ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag,
                   Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits  =
    ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag,
                   Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits =
    ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                   Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

再通过 `upcast<sizeof_bits<Type>::value>` 转成元素单位：

```cpp
/**
 * @brief 把 bit 单位的 GMMA layout atom 转成 Type 元素单位的 layout atom。
 */
template <class Type>
using Layout_MN_SW128_Atom =
    decltype(upcast<sizeof_bits<Type>::value>(Layout_MN_SW128_Atom_Bits{}));

/**
 * @brief 根据 GMMA operand major 选择 M/N-major 或 K-major layout atom。
 */
template <class Type, Major tnsp>
using Layout_SW128_Atom =
    typename conditional<tnsp == Major::MN,
                         Layout_MN_SW128_Atom<Type>,
                         Layout_K_SW128_Atom<Type>>::type;
```

`Major tnsp` 这个模板参数就是一个选择器：

```cpp
// 等价于 GMMA::Layout_MN_SW128_Atom<half_t>
using A_MN_Atom = GMMA::Layout_SW128_Atom<half_t, GMMA::Major::MN>;

// 等价于 GMMA::Layout_K_SW128_Atom<half_t>
using A_K_Atom = GMMA::Layout_SW128_Atom<half_t, GMMA::Major::K>;
```

它的好处是可以把 mainloop 写成模板：

```cpp
/**
 * @brief 根据 GMMA major 参数选择 A/B 的 shared-memory swizzle layout。
 *
 * @tparam MajorA A operand 在 GMMA descriptor 中的 major 方向。
 * @tparam MajorB B operand 在 GMMA descriptor 中的 major 方向。
 */
template <GMMA::Major MajorA, GMMA::Major MajorB,
          class TA, class TB, class MMA_Atom>
void build_layouts_and_mma() {
    auto sA_atom = GMMA::Layout_SW128_Atom<TA, MajorA>{};
    auto sB_atom = GMMA::Layout_SW128_Atom<TB, MajorB>{};

    auto sA = tile_to_shape(sA_atom, make_shape(Int<128>{}, Int<64>{}, Int<3>{}));
    auto sB = tile_to_shape(sB_atom, make_shape(Int<128>{}, Int<64>{}, Int<3>{}));

    TiledMMA tiled_mma = make_tiled_mma(MMA_Atom{});
}
```

真实代码里更常见的是直接把 SMEM layout atom 和 MMA atom 写成一对，保证两边的 major 一致。

这些 `SW32` / `SW64` / `SW128` 和 CUDA tensor map 里的 `CUtensorMapSwizzle` 可以这样对应：

| CuTe swizzle | CUDA tensor map swizzle | 常见含义 |
| --- | --- | --- |
| `Swizzle<0,4,3>` | `CU_TENSOR_MAP_SWIZZLE_NONE` | 不启用 32B/64B/128B swizzle。 |
| `Swizzle<1,4,3>` | `CU_TENSOR_MAP_SWIZZLE_32B` | 32B swizzle。 |
| `Swizzle<2,4,3>` | `CU_TENSOR_MAP_SWIZZLE_64B` | 64B swizzle。 |
| `Swizzle<3,4,3>` | `CU_TENSOR_MAP_SWIZZLE_128B` | 128B swizzle。 |

CuTe 的转换在 `copy_sm90_desc.hpp`：

```cpp
/**
 * @brief 把 CuTe SMEM swizzle 编码转换成 CUDA driver tensor map swizzle 枚举。
 *
 * @param t swizzle bits，例如 DISABLE、B32、B64、B128。
 * @param b swizzle base，例如 16B / 32B / 64B atom base。
 * @return `cuTensorMapEncodeTiled` 需要的 `CUtensorMapSwizzle`。
 */
inline CUtensorMapSwizzle
to_CUtensorMapSwizzle(SmemSwizzleBits const& t, SmemSwizzleBase const& b) {
    switch (t) {
        case SmemSwizzleBits::DISABLE:
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        case SmemSwizzleBits::B32:
            return CU_TENSOR_MAP_SWIZZLE_32B;
        case SmemSwizzleBits::B64:
            return CU_TENSOR_MAP_SWIZZLE_64B;
        case SmemSwizzleBits::B128:
            return CU_TENSOR_MAP_SWIZZLE_128B;
        default:
            throw std::runtime_error("Unsupported swizzle.");
    }
}
```

教程里的 `gemm_nt` 使用 `MN/MN`：

```cpp
using TA = half_t;
using TB = half_t;

auto sA_layout_atom = GMMA::Layout_MN_SW128_Atom<TA>{};
auto sB_layout_atom = GMMA::Layout_MN_SW128_Atom<TB>{};

auto sA = tile_to_shape(sA_layout_atom, make_shape(bM, bK, bP));
auto sB = tile_to_shape(sB_layout_atom, make_shape(bN, bK, bP));

TiledMMA tiled_mma =
    make_tiled_mma(SM90_64x64x16_F16F16F16_SS<
        GMMA::Major::MN,
        GMMA::Major::MN>{});
```

这和它的 GMEM stride 是一致的：

```cpp
// A: (M,K)，M 方向 stride 为 1。
auto dA = make_stride(Int<1>{}, ldA);

// B: (N,K)，N 方向 stride 为 1。
auto dB = make_stride(Int<1>{}, ldB);
```

另一个路径使用 `K/K`：

```cpp
auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

TiledMMA tiled_mma =
    make_tiled_mma(SM90_64x64x16_F16F16F16_SS<
        GMMA::Major::K,
        GMMA::Major::K>{});
```

对应 GMEM stride 是：

```cpp
// A: (M,K)，K 方向 stride 为 1。
auto dA = make_stride(ldA, Int<1>{});

// B: (N,K)，K 方向 stride 为 1。
auto dB = make_stride(ldB, Int<1>{});
```

因此选择 `Major::MN` 还是 `Major::K` 时，先看 **SMEM 里准备给 GMMA 的 operand layout**，再让 `TiledMMA` 的 major 模板参数和这个 layout 对齐。TMA descriptor 也会从同一个 `sA(_,_,0)` / `sB(_,_,0)` 里提取 swizzle，所以 TMA 写入和 GMMA 读取必须使用同一种 layout 约定。

这里的 `sA` 同时影响两件事：

- `make_tma_atom` 会从它提取 shared-memory swizzle，填到 tensor map descriptor。
- `thr_mma.make_fragment_A(...)` 会根据它生成 GMMA descriptor，让 GMMA 能按同一种 swizzle 读取 shared memory。

## 从教程示例看 CuTe TMA 的使用线路

`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` 里有一个 `gemm_tn` 示例。WGMMA 不是本文重点，这里只看 TMA 如何设置和发起。

### 完整源码注释版：`SharedStorage`、`gemm_device`、`gemm_tn`

先把关键源码完整放在这里。下面是学习注释版：保留原始变量名、控制流和 CuTe API 调用，只补中文 Doxygen 注释和行内解释。后面再拆 `make_tma_atom`、`get_tma_tensor`、`tma_partition`、barrier 和 pipeline 时，都可以回到这段代码里对照。

```cpp
/**
 * @brief `gemm_device` 使用的动态 shared memory 布局。
 *
 * @tparam ElementA A operand 的元素类型，例如 `half_t`。
 * @tparam ElementB B operand 的元素类型，例如 `half_t`。
 * @tparam SmemLayoutA A 在 shared memory 中的 CuTe layout，形状是 `(BLK_M, BLK_K, PIPE)`。
 * @tparam SmemLayoutB B 在 shared memory 中的 CuTe layout，形状是 `(BLK_N, BLK_K, PIPE)`。
 *
 * @details
 * 这个结构体会被放进 kernel 的 dynamic shared memory。
 *
 * - `A` / `B` 是 TMA 写入、GMMA 读取的 shared memory staging buffer。
 * - `tma_barrier` 是 TMA producer barrier，等 TMA load 写完 shared memory。
 * - `mma_barrier` 是 MMA consumer barrier，通知 TMA producer 某个 pipe 已经消费完。
 */
template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  // A 的 shared memory buffer。
  // `cosize_v<SmemLayoutA>` 是这个 layout 覆盖的元素个数。
  // `alignas(128)` 是 TMA / GMMA 常见 shared-memory 对齐要求。
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;

  // B 的 shared memory buffer，逻辑形状由 `SmemLayoutB` 决定。
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  // 每个 pipeline stage 一个 TMA transaction barrier。
  // `size<2>(SmemLayoutA{})` 对应第三维 PIPE 数。
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];

  // 每个 pipeline stage 一个 MMA consumer barrier。
  // A/B 的 PIPE 数必须一致，所以这里用 `SmemLayoutA` 的 PIPE 数即可。
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

/**
 * @brief SM90 TMA + WGMMA 的 device kernel。
 *
 * @tparam ProblemShape GEMM 问题形状类型，运行时值是 `(M, N, K)`。
 * @tparam CtaTiler CTA tile 形状类型，运行时值是 `(BLK_M, BLK_N, BLK_K)`。
 * @tparam TA A operand 元素类型。
 * @tparam SmemLayoutA A 的 shared memory layout 类型，形状 `(BLK_M, BLK_K, PIPE)`。
 * @tparam TmaA A 的 TMA atom / descriptor 类型。
 * @tparam TB B operand 元素类型。
 * @tparam SmemLayoutB B 的 shared memory layout 类型，形状 `(BLK_N, BLK_K, PIPE)`。
 * @tparam TmaB B 的 TMA atom / descriptor 类型。
 * @tparam TC C operand 元素类型。
 * @tparam CStride C 的 GMEM stride 类型。
 * @tparam TiledMma CuTe tiled MMA 对象类型，描述 WGMMA tile 和线程布局。
 * @tparam Alpha alpha 标量类型。
 * @tparam Beta beta 标量类型。
 *
 * @param shape_MNK GEMM 问题形状 `(M,N,K)`。
 * @param cta_tiler CTA tile 形状 `(BLK_M,BLK_N,BLK_K)`。
 * @param A A 的 GMEM 指针。这里真正的 TMA 描述已经在 `tma_a` 里。
 * @param tma_a A 的 TMA atom，作为 grid constant 传入，内部持有 tensor map descriptor。
 * @param B B 的 GMEM 指针。这里真正的 TMA 描述已经在 `tma_b` 里。
 * @param tma_b B 的 TMA atom，作为 grid constant 传入。
 * @param C C 的 GMEM 指针，用 epilogue 写回。
 * @param dC C 的 stride layout，和 `(M,N)` congruent。
 * @param mma WGMMA tiled MMA 对象。
 * @param alpha epilogue 中 `C = alpha * accum + beta * C` 的 alpha。
 * @param beta epilogue 中 `C = alpha * accum + beta * C` 的 beta。
 */
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // `shape_MNK` 必须是三维问题形状：(M,N,K)。
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});

  // `cta_tiler` 必须是三维 CTA tile 形状：(BLK_M,BLK_N,BLK_K)。
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  // TMA / WGMMA 这类 SMEM layout 通常要静态可知，方便编译期生成 descriptor 和检查。
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  // A 的 shared memory layout 第 0 维必须对应 BLK_M。
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));

  // B 的 shared memory layout 第 0 维必须对应 BLK_N。
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));

  // A 的 shared memory layout 第 1 维必须对应 BLK_K。
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));

  // B 的 shared memory layout 第 1 维必须对应 BLK_K。
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  // C 的 stride layout 要能描述 `(M,N)` 这个二维矩阵。
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

  // 把动态问题形状拆成 M/N/K 三个运行时整数。
  auto [M, N, K] = shape_MNK;

  // `get_tma_tensor` 生成的是 TMA coordinate tensor，不是普通 GMEM data tensor。
  // 它使用 host 侧 `make_tma_atom` 保存下来的 TMA stride 信息。
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor

  // B 同理，逻辑上是 `(N,K)` 的 TMA coordinate tensor。
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor

  // C 不走 TMA，这里是普通 GMEM tensor，epilogue 会直接写回 C。
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // 当前 CTA 的逻辑坐标。x 对应 M tile，y 对应 N tile，K 方向用 `_` 保留成 tile 序列。
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)

  // 从完整 A coordinate tensor 切出当前 CTA 的 A tile。
  // Step<_1, X, _1> 表示使用 cta_tiler 的 M 和 K 维，跳过 N 维。
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)

  // 从完整 B coordinate tensor 切出当前 CTA 的 B tile。
  // Step<X, _1, _1> 表示使用 cta_tiler 的 N 和 K 维，跳过 M 维。
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

  // 从完整 C data tensor 切出当前 CTA 要写回的 C tile。
  // Step<_1,_1,X> 表示使用 M 和 N 维，跳过 K 维。
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // 声明 dynamic shared memory 的原始字节入口。
  extern __shared__ char shared_memory[];

  // 复用上面的 `SharedStorage` 模板，把这段 dynamic shared memory 解释成 A/B buffer + barriers。
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  // A 的 shared memory tensor，形状是 `(BLK_M,BLK_K,PIPE)`。
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});

  // B 的 shared memory tensor，形状是 `(BLK_N,BLK_K,PIPE)`。
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  // TMA 专用 partitioner：
  // - `Int<0>{}` 和 `Layout<_1>{}` 表示当前示例不做 multicast。
  // - `group_modes<0,2>` 把 `(BLK_M,BLK_K,PIPE)` 变成 `((BLK_M,BLK_K),PIPE)`。
  // - TMA 负责第 0 个 grouped mode，也就是一整个 `(BLK_M,BLK_K)` tile。
  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA),
                                    group_modes<0,2>(gA));            // (TMA,k), (TMA,PIPE)

  // B 的 TMA partition 同理，只是空间 tile 是 `(BLK_N,BLK_K)`。
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB),
                                    group_modes<0,2>(gB));            // (TMA,k), (TMA,PIPE)

  // 每个 pipe 中会发起两次 TMA load：一次搬 A stage，一次搬 B stage。
  // transaction barrier 要知道这两次 TMA 总共会写多少字节。
  constexpr int tma_transaction_bytes =
      sizeof(make_tensor_like(tensor<0>(tAsA))) +
      sizeof(make_tensor_like(tensor<0>(tBsB)));

  // PIPE 数来自 `tAsA` 的第 1 维，也就是 shared memory pipeline stage 数。
  auto K_PIPE_MAX = size<1>(tAsA);

  // K 方向总共有多少个 tile 要搬。`tAgA` 的第 1 维就是 k tile 序列。
  int k_tile_count = size<1>(tAgA);

  // 当前要从 GMEM/TMA coordinate tensor 读取的 k tile 编号。
  int k_tile = 0;

  // 每个 warp 内一致的 warp id，用来只让 warp 0 发起 TMA / 初始化 barrier。
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // 每个 warp 选一个 lane。最终只有 `(warp_idx == 0 && lane_predicate)` 的线程做 producer 工作。
  int lane_predicate = cute::elect_one_sync();

  // producer barrier：TMA 写 shared memory，消费者要等它完成。
  uint64_t* producer_mbar = smem.tma_barrier;

  // consumer barrier：WGMMA 消费 shared memory，生产者要等它释放 pipe。
  uint64_t* consumer_mbar = smem.mma_barrier;

  // transaction barrier 会同时跟踪 arrive count 和 TMA transaction bytes。
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;

  // 普通 cluster barrier 只跟踪 arrive count / phase，用来表示 MMA 消费完成。
  using ConsumerBarType = cutlass::arch::ClusterBarrier;

  // 每个 pipe 各初始化一对 producer / consumer barrier。
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      // 只有一个 elected TMA producer 会 arrive，所以 producer arrive count 是 1。
      ProducerBarType::init(&producer_mbar[pipe], 1);

      // 一个 WGMMA warpgroup 有 128 个线程，全部消费完才释放 pipe。
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }

  // barrier 初始化发生在 shared memory 中，需要 cluster 范围同步后才能安全使用。
  cluster_sync();

  // 预取阶段：先把所有 pipe 填满。
  // 这一步让后面的 mainloop 一开始就有数据可算。
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // 先告诉 transaction barrier：这个 phase 要等多少 TMA 字节完成。
      // 这个调用同时执行 arrive，所以 producer arrival count 也会减 1。
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            tma_transaction_bytes);

      // 发起 A 的 TMA load：从第 `k_tile` 个 K tile 搬到 shared memory 第 `pipe` 个 stage。
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));

      // 发起 B 的 TMA load。A/B 共用同一个 producer barrier，所以 bytes 要加总。
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }

    // 这个 k tile 已经发起 TMA 了，从剩余 tile 计数中扣掉。
    --k_tile_count;

    // 下一次预取 / 主循环发起下一个 K tile。
    ++k_tile;
  }

  // 当前线程在 WGMMA tiled MMA 中的逻辑切片。
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);

  // WGMMA 从 shared memory A 读取的 descriptor tensor。
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)

  // WGMMA 从 shared memory B 读取的 descriptor tensor。
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)

  // 当前线程负责的 C accumulator / output tile。
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // 创建 accumulator fragment，并清零。
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // 在 SM90 SS WGMMA 路线中，A/B fragment 实际上是 GMMA descriptor view，不是传统寄存器 fragment。
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  // TMA producer 的环形 pipe 状态：记录下一次写哪个 pipe，以及 barrier phase。
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();

  // MMA consumer 的环形 pipe 状态：记录下一次读哪个 pipe，以及 barrier phase。
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  // 主循环会继续跑到已经预取的 pipe 都被消费完。
  // `k_tile_count > -K_PIPE_MAX` 是为了在没有新 tile 后，继续 drain pipeline。
  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // consumer 选择当前要读取的 shared-memory pipe。
    int read_pipe = read_state.index();

    // 等这个 pipe 对应的 TMA load 完成。
    // `read_state.phase()` 是 mbarrier parity phase，每绕环一圈翻转。
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // 开始一个 warpgroup MMA batch。
    warpgroup_arrive();

    // 对当前 pipe 的 A/B descriptor 执行 WGMMA，累加到 tCrC。
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);

    // 提交 warpgroup MMA batch。
    warpgroup_commit_batch();

    // 等当前 warpgroup 所有 MMA 完成，保证这个 pipe 的 shared memory 已经被消费。
    warpgroup_wait<0>();

    // 通知 producer：当前 `read_pipe` 已经消费完，可以复用。
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);

    // consumer 环形状态前进到下一个 pipe；必要时 phase 翻转。
    ++read_state;

    // 如果还有新的 K tile，就让 elected TMA producer 继续填充 write pipe。
    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0))
    {
      // producer 选择当前要写入的 shared-memory pipe。
      int pipe = write_state.index();

      // 等 consumer 释放这个 pipe，避免 TMA 覆盖还没被 WGMMA 消费完的数据。
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());

      // 给这个 producer barrier phase 设置新的 expected transaction bytes。
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            tma_transaction_bytes);

      // 发起下一块 A tile 的 TMA load。
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));

      // 发起下一块 B tile 的 TMA load。
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));

      // producer 环形状态前进到下一个 pipe；必要时 phase 翻转。
      ++write_state;
    }

    // 这一轮主循环消耗 / 尝试发起了一个 K tile。
    --k_tile_count;
    ++k_tile;
  }

  // epilogue：把 accumulator 写回 C。
  // 示例里是 unpredicated，实际通用 kernel 需要处理边界 tile。
  axpby(alpha, tCrC, beta, tCgC);
}

/**
 * @brief Host 侧配置并启动 TN GEMM。
 *
 * @tparam TA A operand 元素类型。
 * @tparam TB B operand 元素类型。
 * @tparam TC C operand 元素类型。
 * @tparam Alpha alpha 标量类型。
 * @tparam Beta beta 标量类型。
 *
 * @param m GEMM 的 M 维大小。
 * @param n GEMM 的 N 维大小。
 * @param k GEMM 的 K 维大小。
 * @param alpha epilogue 的 alpha。
 * @param A A 的 device 指针。
 * @param ldA A 的 leading dimension。TN 路线里 A 的 K 方向 stride 为 1。
 * @param B B 的 device 指针。
 * @param ldB B 的 leading dimension。TN 路线里 B 的 K 方向 stride 为 1。
 * @param beta epilogue 的 beta。
 * @param C C 的 device 指针。
 * @param ldC C 的 leading dimension。
 * @param stream CUDA stream。
 */
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // 把传入的运行时尺寸转成 CuTe 后续会使用的局部变量。
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  // GEMM 问题形状，后续 kernel 中会拆成 `(M,N,K)`。
  auto prob_shape = make_shape(M, N, K);

  // TN 路线：A 按 `(M,K)` 看，K 方向 stride 为 1，M 方向 stride 是 ldA。
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)

  // TN 路线：B 按 `(N,K)` 看，K 方向 stride 为 1，N 方向 stride 是 ldB。
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)

  // C 按 `(M,N)` 看，M 方向 stride 为 1，N 方向 stride 是 ldC。
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // CTA tile 的 M 维。
  auto bM = Int<128>{};

  // CTA tile 的 N 维。
  auto bN = Int<128>{};

  // CTA tile 的 K 维，每个 mainloop step 消费一个 BLK_K。
  auto bK = Int<64>{};

  // CTA tiler 是 `(BLK_M, BLK_N, BLK_K)`，同时用于 local_tile 和 launch grid 计算。
  auto cta_tiler = make_shape(bM, bN, bK);

  // shared memory pipeline stage 数。
  auto bP = Int<3>{};

  // TN 路线里 A/B 都是 K-major shared-memory layout。
  // 这里使用 128B swizzle，匹配 GMMA::Major::K。
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{},
                          make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{},
                          make_shape(bN,bK,bP));

  // WGMMA atom 也声明 A/B 都是 Major::K，必须和上面的 SMEM layout 约定一致。
  TiledMMA tiled_mma =
      make_tiled_mma(SM90_64x64x16_F16F16F16_SS<
          GMMA::Major::K,
          GMMA::Major::K>{});

  // Host 侧创建普通 GMEM data tensor，用来让 `make_tma_atom` 检查 shape/stride 并编码 descriptor。
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // 为 A 创建 TMA atom：
  // - `SM90_TMA_LOAD{}` 选择 global -> shared TMA load。
  // - `mA` 提供 GMEM base/shape/stride。
  // - `sA(_,_,0)` 只取一个 pipe 的 SMEM layout，用来描述一次 TMA 的 box。
  // - `make_shape(bM,bK)` 描述 CTA-local A tile。
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0),
                                 make_shape(bM,bK));

  // B 的 TMA atom 同理，只是 CTA-local tile 是 `(bN,bK)`。
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0),
                                 make_shape(bN,bK));

  // 一个 CTA 使用的线程数由 tiled_mma 决定。
  dim3 dimBlock(size(tiled_mma));

  // 每个 cluster 里 x 方向放 2 个 CTA；这个示例不在 y/z 方向组 cluster。
  dim3 dimCluster(2, 1, 1);

  // gridDim 仍然是 CTA 个数，但 cluster launch 要求能按 dimCluster 整 cluster 分组。
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));

  // dynamic shared memory 字节数，包含 A/B staging buffer 和每个 pipe 的 barrier。
  int smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  // 实例化模板 kernel，类型全部来自上面构造出的 CuTe 对象。
  auto* kernel_ptr =
      &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                   TA, decltype(sA), decltype(tmaA),
                   TB, decltype(sB), decltype(tmaB),
                   TC, decltype(dC), decltype(tiled_mma),
                   decltype(alpha), decltype(beta)>;

  // 告诉 CUDA runtime 这个 kernel 需要的动态 shared memory 上限。
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smemBytes));

  // CUTLASS cluster launch 参数。
  cutlass::ClusterLaunchParams params = {
      dimGrid,
      dimBlock,
      dimCluster,
      smemBytes
  };

  // 用 cluster launch 启动 kernel，并把 TMA atom 作为 kernel 参数传入。
  cutlass::Status status =
      cutlass::launch_kernel_on_cluster(
          params,
          reinterpret_cast<void const*>(kernel_ptr),
          prob_shape, cta_tiler,
          A, tmaA,
          B, tmaB,
          C, dC, tiled_mma,
          alpha, beta);

  // 检查 CUDA launch 错误。
  CUTE_CHECK_LAST();

  // 检查 CUTLASS cluster launcher 返回的状态。
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}
```

这里有一个容易卡住的小点：源码里的
`using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;`
是在函数作用域里创建一个**同名类型别名**。等号右边先查到外层的类模板 `SharedStorage<...>`，等号左边的新名字再在函数作用域里遮住外层模板名。所以后面的
`SharedStorage& smem = ...` 指的是已经实例化好的具体 shared-memory storage 类型。

### Host 侧：创建 GMEM tensor 和 TMA atom

Host 侧先把裸指针包装成 CuTe tensor，再用 `make_tma_atom` 创建 TMA copy atom。

```cpp
/**
 * @brief 为 A/B 构造 GMEM tensor，并根据 GMEM tensor、SMEM layout 和 CTA tile 创建 TMA atom。
 *
 * @param A A 矩阵的 global memory 指针。
 * @param B B 矩阵的 global memory 指针。
 * @param dA A 的 stride layout。
 * @param dB B 的 stride layout。
 * @param sA A 在 shared memory 中的 layout，包含 pipeline mode。
 * @param sB B 在 shared memory 中的 layout，包含 pipeline mode。
 * @return `tmaA` / `tmaB` 是只含 descriptor 的 non-executable TMA atom。
 */
Tensor mA = make_tensor(A, make_shape(M,K), dA);
Tensor mB = make_tensor(B, make_shape(N,K), dB);

Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0),
                               make_shape(bM,bK));
Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0),
                               make_shape(bN,bK));
```

几个细节：

- `mA` / `mB` 是完整 GMEM tensor，不是当前 CTA 的 tile。
- `sA(_,_,0)` 只取一个 pipeline stage 的 SMEM layout，因为 descriptor 描述的是“一次 TMA 写入一个 stage 的 box”。
- `make_shape(bM,bK)` 是 CTA tile shape，告诉 CuTe 一个 CTA 在 A 上取 `(bM,bK)`。
- `SM90_TMA_LOAD{}` 说明这是 GMEM 到 SMEM 的 TMA load。

### Host 侧：cluster launch

TMA multicast 和 cluster-level barrier 都依赖 SM90 cluster launch。示例里用 CUTLASS 的封装：

```cpp
/**
 * @brief 使用 CUDA cluster launch 启动 kernel。
 *
 * @details
 * `dimCluster(2, 1, 1)` 表示一个 cluster 中 x 方向有 2 个 CTA。
 * `dimGrid` 通常要按 cluster 维度 round up，保证 grid 可以按 cluster 分组。
 */
int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
dim3 dimBlock(size(tiled_mma));
dim3 dimCluster(2, 1, 1);
dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
             round_up(size(ceil_div(n, bN)), dimCluster.y));

cutlass::ClusterLaunchParams params = {
    dimGrid,
    dimBlock,
    dimCluster,
    smem_size
};

cutlass::Status status = cutlass::launch_kernel_on_cluster(
    params, kernel_ptr,
    prob_shape, cta_tiler,
    A, tmaA,
    B, tmaB,
    C, dC, tiled_mma,
    alpha, beta);
```

这里的 `ceil_div(m, bM)` 先算出 M 方向需要多少个 CTA tile：

$$
\text{tile\_m} = \left\lceil \frac{m}{bM} \right\rceil
$$

外层的 `round_up(..., dimCluster.x)` 是为了 **cluster launch 的整 cluster 分组**。`dimGrid` 传给 CUDA 的仍然是 CTA grid 维度，不是 cluster grid 维度；但当 `dimCluster(2,1,1)` 时，CUDA 会把 x 方向每 2 个 CTA 组成一个 cluster。

因此 `dimGrid.x` 最好是 `dimCluster.x` 的整数倍：

```text
tile_m = 5
dimCluster.x = 2

如果 dimGrid.x = 5:
  最后一个 cluster 只有 1 个 CTA，不是完整 cluster。

round_up(5, 2) = 6:
  一共 6 个 CTA，可以组成 3 个完整 cluster。
```

这行代码可以拆成：

```cpp
int cta_tiles_m = size(ceil_div(m, bM));
int cta_tiles_n = size(ceil_div(n, bN));

// gridDim 是 CTA 数，但 cluster launch 希望它能被 clusterDim 整除。
dim3 dimGrid(round_up(cta_tiles_m, dimCluster.x),
             round_up(cta_tiles_n, dimCluster.y));
```

这样做可能会多 launch 一些 **padding CTA**。例如真实只需要 5 个 M tile，却 launch 了 6 个 CTA。通用 kernel 里应该对这些 padding CTA 做边界保护；这个 CuTe tutorial 的 epilogue 写着 `unpredicated`，更偏向演示 TMA / WGMMA 主流程，通常假设问题规模和 tile / cluster 形状配合得比较好。

注意这里把 `tmaA` / `tmaB` 作为 kernel 参数传入。它们里面带着已经在 host 侧编码好的 tensor map descriptor。

### Kernel 侧：TMA tensor、CTA tile 和 partition

进入 kernel 后，第一步不是直接拿裸指针算地址，而是用 TMA atom 生成 TMA 坐标 tensor：

```cpp
/**
 * @brief 在 kernel 内为 TMA atom 创建坐标 tensor，并切出当前 CTA 的 tile。
 *
 * @details
 * `get_tma_tensor` 返回的是 TMA 坐标空间里的 tensor。
 * `local_tile` 再根据 CTA tiler 和 CTA 坐标切出当前 CTA 负责的 tile。
 */
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));

Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});

Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
```

这段和 SM80 `sgemm_sm80.cu` 看起来很像，但 `mA` 的语义已经变了。

SM80 里是：

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});    // (BLK_M,BLK_K,k)
```

这个 `mA` 是 **global memory data tensor**，里面的元素访问会走 global pointer。

TMA 里是：

```cpp
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                     // (M,K)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});    // (BLK_M,BLK_K,k)
```

这个 `mA` 是 **TMA coordinate tensor**，不是直接读数据的 tensor。源码里的 `get_tma_tensor` 是：

```cpp
/**
 * @brief 根据完整 GMEM shape 生成 TMA 坐标 tensor。
 *
 * @tparam GShape 完整 GMEM tensor 的 shape 类型，例如 `(M,K)`。
 * @param g_shape 完整 GMEM tensor 的 shape。
 * @return 一个 coordinate tensor，layout 使用 descriptor 构造时保存的 TMA stride。
 */
template <class GShape>
CUTE_HOST_DEVICE constexpr
auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape),
                               decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
}
```

所以 `get_tma_tensor(make_shape(M,K))` 做的是：

```text
完整逻辑形状:      (M, K)
descriptor stride: aux_params_.g_stride_
返回 tensor:       (M, K) 里的每个点能生成 TMA 指令需要的 tensor coordinate
```

它和 SM80 的 `make_tensor(make_gmem_ptr(A), ...)` 的区别是：

| 路线 | `mA` 表示什么 | `gA = local_tile(...)` 表示什么 |
| --- | --- | --- |
| SM80 cp.async | global memory 数据 tensor | 当前 CTA 要搬的数据地址 tile：`(BLK_M,BLK_K,k)`。 |
| SM90 TMA | TMA 坐标 tensor | 当前 CTA 要交给 TMA 指令的坐标 tile：`(BLK_M,BLK_K,k)`。 |

接下来用 `tma_partition` 把 GMEM tile 和 SMEM tile 变成 TMA copy 能接受的形状：

```cpp
/**
 * @brief 把 A/B 的 GMEM tile 和 SMEM tile 重排成 TMA copy 的 src/dst tensor。
 *
 * @details
 * `group_modes<0,2>` 会把 M/K 或 N/K 的 tile mode 合成 TMA mode，
 * 保留 pipeline / k tile 这类剩余 mode。返回的 tensor 形状大致是：
 *
 * - `tAgA`: `(TMA, k)`，第 0 维由一条或多条 TMA 指令覆盖。
 * - `tAsA`: `(TMA, PIPE)`，第 1 维是 shared-memory pipeline stage。
 */
auto [tAgA, tAsA] = tma_partition(
    tma_a, Int<0>{}, Layout<_1>{},
    group_modes<0,2>(sA),
    group_modes<0,2>(gA));

auto [tBgB, tBsB] = tma_partition(
    tma_b, Int<0>{}, Layout<_1>{},
    group_modes<0,2>(sB),
    group_modes<0,2>(gB));
```

这里的 Tensor 形状可以按下面理解：

| 变量 | group 前形状 | group 后形状 | 含义 |
| --- | --- | --- | --- |
| `gA` | `(BLK_M, BLK_K, k)` | `((BLK_M, BLK_K), k)` | 把一个 CTA 的 A tile 合成 TMA 要覆盖的主 mode，剩下的 `k` 是第几个 K tile。 |
| `sA` | `(BLK_M, BLK_K, PIPE)` | `((BLK_M, BLK_K), PIPE)` | 把一个 shared-memory stage 的 A tile 合成 TMA 写入主 mode，剩下的 `PIPE` 是 pipeline stage。 |
| `gB` | `(BLK_N, BLK_K, k)` | `((BLK_N, BLK_K), k)` | B tile 同理。 |
| `sB` | `(BLK_N, BLK_K, PIPE)` | `((BLK_N, BLK_K), PIPE)` | B 的 shared-memory stage。 |

为什么要 `group_modes<0,2>`？

因为对 TMA 来说，`(BLK_M, BLK_K)` 或 `(BLK_N, BLK_K)` 是 **一块 tensor tile**，不是线程级 copy 里的两个独立 per-thread mode。TMA 指令负责把这一整块 tile 搬进 shared memory，所以 CuTe 先把 tile 的两个空间维度合成一个 “TMA tile mode”。

再看 SM80 的 partition：

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)
```

SM80 是每个线程都有自己的 `ThrCopy`，所以 `tAgA` / `tAsA` 里会出现：

```text
CPY, CPY_M, CPY_K
```

这些 mode 描述“当前线程负责搬哪几个元素”。

TMA 的 partition 是：

```cpp
auto [tAgA, tAsA] = tma_partition(...);
```

TMA 通常由一个 elected lane 发起，硬件搬整块 tile，所以 **没有 per-thread 的 `CPY_M/CPY_K` 切分**。`tAgA` / `tAsA` 的第 0 维变成 TMA 指令关心的主 mode：

```text
tAgA: (TMA, k)
tAsA: (TMA, PIPE)
```

更细一点说，`tma_partition` 会根据 SMEM layout 反推出最大的 contiguous vector，再把 mode-0 切成 TMA 指令可以发出的 `(TMA, TMA_Iter)` 形态；示例注释里把它简写成 `(TMA,k)` 和 `(TMA,PIPE)`。

所以两条路线可以对照成：

| 步骤 | SM80 `sgemm_sm80.cu` | SM90 TMA tutorial |
| --- | --- | --- |
| 完整 A tensor | `make_tensor(make_gmem_ptr(A), (M,K), dA)` | `tma_a.get_tma_tensor((M,K))` |
| CTA A tile | `local_tile(...) -> (BLK_M,BLK_K,k)` | `local_tile(...) -> (BLK_M,BLK_K,k)` |
| copy 切分 | `thr_copy.partition_S/D` | `group_modes<0,2>` 后 `tma_partition` |
| source tensor | `(CPY,CPY_M,CPY_K,k)` | `(TMA,k)` |
| dest tensor | `(CPY,CPY_M,CPY_K,PIPE)` | `(TMA,PIPE)` |
| 发起者 | 每个线程发自己的 `cp.async` | 一个 elected lane 发整块 TMA |

### Kernel 侧：transaction bytes

TMA load 完成后会通知 mbarrier。这个通知不是“完成了一个 bool”，而是按 bytes 减少 transaction count。

示例里计算一次 pipe 需要等待的总字节数：

```cpp
/**
 * @brief 计算一个 pipeline stage 中 A/B 两次 TMA load 总共会写入多少字节。
 *
 * @details
 * `tensor<0>(tAsA)` 取出 TMA mode 对应的一个 stage，`make_tensor_like`
 * 生成同形状的值 tensor，`sizeof` 得到该 stage 的写入字节数。
 */
constexpr int tma_transaction_bytes =
    sizeof(make_tensor_like(tensor<0>(tAsA))) +
    sizeof(make_tensor_like(tensor<0>(tBsB)));
```

后面调用：

```cpp
ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                      tma_transaction_bytes);
```

含义是：

- arrival count 减 1，表示“生产者已经把这一阶段的 TMA 操作发布出去了”。
- expected transaction count 增加 `tma_transaction_bytes`，表示“这个 barrier 还要等这么多 TMA 写入字节完成”。

只有 arrival count 到 0 且 transaction count 也到 0，这个 barrier phase 才完成。

### Kernel 侧：发起 TMA copy

`make_tma_atom` 返回的是 non-executable atom。发起 TMA 前要用 `.with(...)` 绑定 mbarrier：

```cpp
/**
 * @brief 绑定 mbarrier 后发起 A/B 的 TMA load。
 *
 * @details
 * `tma_a.with(producer_mbar[pipe])` 会生成 executable TMA load traits。
 * `copy(...)` 最终展开到 `cp.async.bulk.tensor.*.shared::cluster.global...`。
 */
ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                      tma_transaction_bytes);
copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
```

这里不是所有线程都发 TMA。示例用：

```cpp
if ((warp_idx == 0) && lane_predicate) {
    // only one elected thread issues TMA
}
```

`warp_idx == 0` 选第 0 个 warp，`cute::elect_one_sync()` 再从这个 warp 里选一个 lane。最终只有一个线程发起 TMA 指令。

## `make_tma_copy` 和 `make_tma_atom`

CuTe 有两条相近线路：

- `make_tma_copy`：构造 CTA-collective `TiledCopy`，更像完整的 tiled copy 对象。
- `make_tma_atom`：构造实验性的 `Copy_Atom`，再配合 `tma_partition` 使用。教程示例用的是这一条。

先用 SM80 的写法做参照。`sgemm_sm80.cu` 里 host 侧创建的是线程级 copy 对象：

```cpp
/**
 * @brief SM80 cp.async 的 tiled copy：描述每个线程如何搬一小片数据。
 *
 * @details
 * `Thr layout` 描述线程如何铺在 tile 上。
 * `Val layout` 描述每个线程一次搬几个值。
 */
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // 16x8 个线程，K-major 排布。
    Layout<Shape< _1,_8>>{});               // 每个线程搬 1x8 个值。
```

这个对象的核心是 **thread layout + value layout**。所以 kernel 里要：

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);  // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA);  // (CPY,CPY_M,CPY_K,PIPE)
```

TMA 的 host 侧对象不是这样。TMA 不关心“每个线程搬几个元素”，它关心的是：

- global tensor 的 base / shape / stride 怎么编码进 tensor map。
- shared-memory tile 的 box shape 和 swizzle 是什么。
- 一个 CTA tile 对应 global tensor 的哪一块。
- 如果 multicast，cluster 中几个 CTA 共享同一个 TMA load。

所以 `make_tma_atom` 里没有 thread layout / value layout，它的输入是：

```cpp
Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0),
                               make_shape(bM,bK));
```

把这行按语义拆开：

| 实参 | 语义 | 对应 CUDA TMA descriptor 字段 |
| --- | --- | --- |
| `SM90_TMA_LOAD{}` | 选择 GMEM -> SMEM 的 TMA load 指令族。 | 后续会走 `cp.async.bulk.tensor.*.shared::cluster.global...`。 |
| `mA` | 完整 GMEM tensor，形状 `(M,K)`，带 stride `dA`。 | base address、globalDim、globalStrides、元素类型。 |
| `sA(_,_,0)` | 一个 pipeline stage 的 SMEM layout，形状 `(BLK_M,BLK_K)`。 | boxDim、elementStrides、shared-memory swizzle。 |
| `make_shape(bM,bK)` | 一个 CTA 在 A 上负责的 tile shape。 | TMA box 对应的 CTA-local tile。 |
| 默认 `Int<1>{}` | 不做 multicast。 | multicast size 为 1。 |

模板参数不用手写，大多由这几个实参推导出来：

| 模板参数 | 在示例里由谁推导 | 含义 |
| --- | --- | --- |
| `CopyOp` | `SM90_TMA_LOAD{}` | TMA 操作类型。 |
| `GEngine` / `GLayout` | `mA` | GMEM tensor 的指针 engine 和 layout。 |
| `SLayout` | `sA(_,_,0)` | 单个 SMEM stage 的 layout。 |
| `CTA_Tiler` | `make_shape(bM,bK)` | CTA tile shape。 |
| `Cluster_Size` | 默认参数 | multicast cluster size。 |
| `TmaInternalType` | 默认 `void` | 用 `GEngine::value_type` 作为 tensor map 元素类型。 |

### `make_tma_copy`

完整重载：

```cpp
/**
 * @brief 构造 CuTe CTA-collective TMA tiled copy 对象。
 *
 * @tparam TmaInternalType descriptor 中使用的 TMA 元素类型，默认是 GMEM value type。
 * @tparam CopyOp TMA copy 操作，例如 `SM90_TMA_LOAD{}`。
 * @tparam GEngine GMEM tensor engine 类型。
 * @tparam GLayout GMEM tensor layout 类型。
 * @tparam SLayout SMEM layout 类型。
 * @tparam CTA_Tiler CTA tile shape / layout 类型。
 * @tparam Cluster_Size multicast cluster size 类型。
 *
 * @param copy_op TMA 操作：load、multicast load、store 或 im2col 变体。
 * @param gtensor 参与 TMA 的 global-memory tensor。
 * @param slayout 参与 TMA 的 shared-memory layout。
 * @param cta_tiler 每个 CTA 在 GMEM tensor 上切出的 tile。
 * @param cluster_size multicast 参与 CTA 数。非 multicast 通常是 `Int<1>{}`。
 * @return 可以进一步 partition 并由 `copy(...)` 发起的 TMA tiled copy 对象。
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout,
                   CTA_Tiler const& cta_tiler,
                   Cluster_Size const& cluster_size);
```

常用重载：

```cpp
/**
 * @brief 使用 SMEM layout 的 product shape 作为 CTA tile，cluster size 默认为 1。
 */
template <class CopyOp, class GEngine, class GLayout, class SLayout>
CUTE_HOST_RTC
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout);

/**
 * @brief 使用 SMEM layout 的 product shape 作为 CTA tile，但显式指定 multicast cluster size。
 */
template <class CopyOp, class GEngine, class GLayout,
          class SLayout, class Cluster_Size>
CUTE_HOST_RTC
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout,
                   Cluster_Size const& cluster_size);
```

`make_tma_copy` 内部做的事可以概括成四步：

1. 从 `slayout` 拆出 swizzle 部分和非 swizzle layout。
2. 对非 swizzle SMEM layout 求右逆，找到 shared memory 中最大的 contiguous vector。
3. 把这个 vector 映射回 GMEM 维度，构造 TMA basis。
4. 调 `cuTensorMapEncodeTiled` 编码 descriptor，再封装成 CuTe copy traits。

如果用 `make_tma_copy` 这条路线，使用方式更接近 SM80 的 `TiledCopy`：

```cpp
/**
 * @brief 用 `make_tma_copy` 构造一个完整 TMA tiled copy。
 *
 * @details
 * 这条路线会返回带 descriptor 的 tiled copy 对象。
 * 后续可以像普通 `TiledCopy` 一样 `get_slice`，再 partition source / destination。
 */
auto tmaA = make_tma_copy(SM90_TMA_LOAD{}, mA, sA(_,_,0),
                          make_shape(bM,bK), Int<1>{});
```

kernel 内大致是：

```cpp
/**
 * @brief `make_tma_copy` 路线的分区方式。
 *
 * @param cta_idx_in_cluster 当前 CTA 在 cluster 内的 logical id。
 */
auto cta_tma_a = tmaA.get_slice(cta_idx_in_cluster);

Tensor tAgA = cta_tma_a.partition_S(gA); // TMA source tensor
Tensor tAsA = cta_tma_a.partition_D(sA); // TMA destination tensor

copy(tmaA.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
```

这条路线和 `make_tma_atom + tma_partition` 的区别可以这样记：

| 路线 | 分区入口 | 读起来像什么 |
| --- | --- | --- |
| `make_tma_copy` | `cta_tma.partition_S/D(...)` | 像 SM80 `TiledCopy` 的 TMA 版本。 |
| `make_tma_atom` | `tma_partition(tma_atom, ..., stensor, gtensor)` | 显式调用 TMA 专用 partitioner。 |

教程示例选择 `make_tma_atom`，所以后面重点看 atom 路线。

### `make_tma_atom`

教程示例用的是 `make_tma_atom`：

```cpp
/**
 * @brief 构造一个 TMA Copy_Atom。
 *
 * @details
 * 返回值里带有 TMA descriptor 和 `Copy_Traits`。它还不能直接执行 load，
 * kernel 内需要先 `tma_partition`，再通过 `.with(mbarrier)` 绑定 barrier。
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size = Int<1>>
CUTE_HOST_RTC
auto make_tma_atom(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout,
                   CTA_Tiler const& cta_tiler,
                   Cluster_Size const& cluster_size = {});
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `copy_op` | TMA 操作类型，例如 `SM90_TMA_LOAD{}`、`SM90_TMA_LOAD_MULTICAST{}`、`SM90_TMA_STORE{}`。 |
| `gtensor` | 全局内存 tensor，用于提取 base、shape、stride。 |
| `slayout` | 一个 TMA box 对应的 shared-memory layout。 |
| `cta_tiler` | CTA tile shape，用于描述每个 CTA 在 global tensor 上负责哪块。 |
| `cluster_size` | multicast 参与 CTA 数，默认 1。 |
| `TmaInternalType` | 可选内部元素类型。默认 `void` 表示使用 `GEngine::value_type`。 |

`make_tma_atom` 完成后，host 侧得到的 `tmaA` / `tmaB` 已经带了 tensor map descriptor，但还没有进入具体 CTA 的 tile，也没有绑定 mbarrier。因此它的使用分成三步：

1. **生成 TMA 坐标 tensor**：`Tensor mA = tma_a.get_tma_tensor(make_shape(M,K))`。
2. **切出当前 CTA tile**：`Tensor gA = local_tile(..., cta_coord, Step<...>{})`。
3. **TMA 专用 partition**：`auto [tAgA, tAsA] = tma_partition(tma_a, ..., group_modes(...), group_modes(...))`。

完整地串起来就是：

```cpp
/**
 * @brief `make_tma_atom` 路线从 descriptor 到可 copy tensor 的完整路径。
 */
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                  // (M,K)
Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                       Step<_1, X,_1>{});                          // (BLK_M,BLK_K,k)
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()),
                        SmemLayoutA{});                            // (BLK_M,BLK_K,PIPE)

auto gA_tma = group_modes<0,2>(gA);                                 // ((BLK_M,BLK_K),k)
auto sA_tma = group_modes<0,2>(sA);                                 // ((BLK_M,BLK_K),PIPE)

auto [tAgA, tAsA] =
    tma_partition(tma_a, Int<0>{}, Layout<_1>{}, sA_tma, gA_tma);    // (TMA,k), (TMA,PIPE)
```

这里的 `Int<0>{}, Layout<_1>{}` 不是数据 layout，而是最小的 **multicast layout**：只有 1 个 CTA 参与，当前 CTA 的 multicast 坐标是 0，并且这个坐标映射到 logical TMA id 0。也就是“不做 multicast，也不做 TMA tile 分片”。

这段里每个模板参数都可以从实参推导：

| `tma_partition` 模板参数 | 示例实参 | 含义 |
| --- | --- | --- |
| `Args...` | `tma_a` 的 `Copy_Atom<Args...>` | 里面包含 descriptor、TMA 每条指令的元素数、aux stride 等。 |
| `CtaCoord` | `Int<0>{}` | 当前 CTA 在 multicast layout 里的坐标。这里不 multicast，所以是 0。 |
| `TShape` / `TStride` | `Layout<_1>{}` | CTA 到 multicast logical id 的 layout。这里只有 1 个接收 CTA。 |
| `SEngine` / `SLayout` | `sA_tma` | grouped 后的 SMEM tensor。 |
| `GTensors...` | `gA_tma` | grouped 后的 GMEM coordinate tensor，可以有多个。 |

这样看，`make_tma_atom` 不是“创建一个神秘对象”，而是把 **CUDA tensor map descriptor** 放进 CuTe copy atom，后续再通过 `get_tma_tensor` 和 `tma_partition` 接回 CuTe Tensor 代数。

### GEMM 便捷封装

`copy_traits_sm90_tma.hpp` 里还提供了 GEMM 语义更强的封装：

| API | 保留的 CTA tile mode | multicast 方向 | 典型用途 |
| --- | --- | --- | --- |
| `make_tma_copy_A_sm90` | 从 `MNK` 中移除 `N`，保留 `MK`。 | 沿 `N` 方向 multicast。 | A operand load。 |
| `make_tma_copy_B_sm90` | 从 `MNK` 中移除 `M`，保留 `NK`。 | 沿 `M` 方向 multicast。 | B operand load。 |
| `make_tma_copy_C_sm90` | 从 `MNK` 中移除 `K`，保留 `MN`。 | 不做 multicast。 | C / epilogue load-store。 |

例如 A operand：

```cpp
/**
 * @brief 为 GEMM A operand 构造 SM90 TMA copy。
 *
 * @details
 * A 的逻辑 tile 是 `(M,K)`，所以从主循环的 `(M,N,K)` CTA tiler 中移除 N。
 * 如果 cluster 在 N 方向有多个 CTA，这些 CTA 可以共享同一个 A tile，
 * 因此 A load 可以沿 N 方向 multicast。
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto make_tma_copy_A_sm90(CopyOp const& copy_op,
                          Tensor<GEngine,GLayout> const& gtensor,
                          SLayout const& slayout,
                          CTA_Tiler const& cta_tiler,
                          Cluster_Size const& cluster_size) {
    auto cta_tiler_mk = remove<1>(cta_tiler);
    auto cluster_size_n = size<1>(cluster_size);
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mk);
    auto cta_t_tile = make_layout(cluster_size_n);
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value,
                                  typename GEngine::value_type,
                                  TmaInternalType>;
    return detail::make_tma_copy_tiled<TmaType>(
        copy_op, gtensor, slayout, cta_t_tile, cta_v_tile);
}
```

## multicast layout

在讲 `tma_partition` 之前，必须先把 **multicast layout** 拆清楚。这个名字很容易误导人，因为它不是 A/B/C 矩阵的元素 layout，也不是 shared memory 的 swizzle layout。它描述的是 **cluster 内 CTA 到 TMA multicast 逻辑编号的映射**。

可以把它写成一个很小的函数：

$$
\text{cta\_layout}: \text{cta\_coord} \rightarrow \text{logical\_tma\_tid}
$$

这里：

- `cta_coord` 是当前 CTA 在 multicast group 里的坐标。例如沿 N 方向有两个 CTA 共享同一块 A tile，那么 `cta_coord` 可以是 `0` 或 `1`。
- `logical_tma_tid` 是 CuTe 给 TMA partition 用的逻辑编号。它决定当前 CTA 在一个 TMA tile 里负责哪一段。
- `cosize(cta_layout)` 是 multicast group 的大小，也就是这次 TMA tile 要被拆给几个 CTA 一起发起。

源码里的注释说得很直接：

```cpp
// The "logical TMA tid" is a map from the CTA rank to its logical id
// within the instruction. It works like a mask or ordering on the CTAs.
// For non-multicast TMA, all CTAs should map to 0. For multicast TMA
// of size 4, CTAs will be mapped to {0,1,2,3}.
```

翻译成 CuTe 视角就是：`multicast layout` 不是为了算元素地址，而是为了告诉 `tma_partition`：**当前 CTA 是 multicast group 里的第几号 TMA producer**。

### 它和 multicast mask 的区别

TMA multicast 里有两个问题，经常被混在一起：

| 问题 | CuTe 里对应什么 | 回答的语义 |
| --- | --- | --- |
| 当前 CTA 负责发起哪一片 TMA copy？ | `cta_layout(cta_coord)` | TMA tile 在 TMA mode 上怎么分片。 |
| 这条 TMA 指令写到哪些 CTA 的 shared memory？ | `multicast_mask` | cluster 中哪些 CTA 接收这次 copy。 |

也就是说：

- **multicast layout 管分工**：CTA 0 搬第 0 片，CTA 1 搬第 1 片，依此类推。
- **multicast mask 管广播目标**：当前这条 `cp.async.bulk.tensor...multicast::cluster` 要写到哪些 CTA 的 SMEM。

如果有 2 个 CTA 沿 N 方向共享同一块 A tile，一个常见协议是：

1. CTA 0 根据 `cta_layout(0) = 0`，发起 A tile 的前半片 TMA load。
2. CTA 1 根据 `cta_layout(1) = 1`，发起 A tile 的后半片 TMA load。
3. 两条 TMA 指令都带同一个 `multicast_mask = 0b0011`，表示 CTA 0 和 CTA 1 都接收。
4. 两条 TMA 都完成后，CTA 0 和 CTA 1 的 shared memory 里都有完整的 A tile。

所以，**multicast 并不只是“一个 CTA 把整块数据广播给别人”**。在 CuTe 的 TMA partition 模型里，更常见的理解是：多个 CTA 把一个逻辑 TMA tile 分片发起，每个分片再 multicast 给同一组接收 CTA。

### 非 multicast 的最小例子

教程源码里写的是：

```cpp
auto [tAgA, tAsA] = tma_partition(
    tma_a,
    Int<0>{},      // 当前 CTA 的 multicast 坐标。
    Layout<_1>{},  // 只有 1 个 logical TMA tid：0。
    sA_tma,
    gA_tma);
```

这个 `Layout<_1>{}` 可以理解成：

$$
\text{cta\_layout}(0) = 0
$$

因此：

- `cosize(cta_layout) = 1`，TMA tile 不需要拆给多个 CTA。
- `cta_layout(Int<0>{}) = 0`，当前 CTA 的 logical TMA id 是 0。
- `multicast_offset = 0`，`tma_partition` 不会给 TMA mode 加额外偏移。

这就是不做 multicast 的含义。源码里的默认重载也走同样逻辑，只是默认 layout 写成 `Layout<_1,_0>{}`：

```cpp
return tma_partition(copy_atom, Int<0>{}, Layout<_1,_0>{}, stensor, gtensor);
```

对单元素 layout 来说，`Layout<_1>{}` 和 `Layout<_1,_0>{}` 都只会把唯一坐标 0 映射到 0，所以教程里直接写 `Layout<_1>{}`。

### 2 个 CTA multicast 的例子

假设一个 cluster 在 N 方向有 2 个 CTA，它们计算不同的 C tile：

```cpp
// CTA 0: 负责 C 的 (m, n0) tile
// CTA 1: 负责 C 的 (m, n1) tile
```

这两个 CTA 的 A operand 都是同一块 `(BLK_M, BLK_K)`，因为 A 不依赖 N。因此 A load 可以沿 N 方向 multicast。

在这个场景里，可以用一个一维 multicast layout：

```cpp
using CtaLayoutN = Layout<Shape<_2>, Stride<_1>>;

CtaLayoutN cta_layout_n{};

// cta_layout_n(0) = 0
// cta_layout_n(1) = 1
```

它的含义是：

| `cta_coord` | `cta_layout_n(cta_coord)` | 当前 CTA 负责的 TMA 片段 |
| --- | --- | --- |
| `0` | `0` | 第 0 片。 |
| `1` | `1` | 第 1 片。 |

如果 TMA mode 上一次 logical tile 有 $T$ 个元素，multicast group 大小是 $G = \text{cosize}(\text{cta\_layout}) = 2$，那么每个 CTA 负责：

$$
\text{local\_span} = \frac{T}{G}
$$

当前 CTA 的偏移是：

$$
\text{multicast\_offset}
= \text{cta\_layout}(\text{cta\_coord}) \times \frac{T}{G}
$$

这正对应 `tma_partition` 源码里的计算：

```cpp
auto multicast_offset =
    cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
```

所以：

- CTA 0 的 `multicast_offset = 0 * T/2`，从 TMA mode 的前半段开始。
- CTA 1 的 `multicast_offset = 1 * T/2`，从 TMA mode 的后半段开始。

调用形态大概是：

```cpp
/**
 * @brief 2 个 CTA 沿 N 方向共同发起 A 的 multicast TMA load。
 *
 * @param cta_coord_n 当前 CTA 在 N 方向 multicast group 里的坐标，取值 0 或 1。
 * @param multicast_mask 接收这次 TMA copy 的 CTA bitmask，例如 0b0011。
 */
auto [tAgA, tAsA] = tma_partition(
    tma_a,
    cta_coord_n,
    cta_layout_n,
    sA_tma,
    gA_tma);

copy(tma_a.with(producer_mbar[pipe], multicast_mask),
     tAgA(_, k_tile),
     tAsA(_, pipe));
```

注意这里 `tma_partition` 只解决“当前 CTA 发哪一片”。`multicast_mask` 仍然要在 `.with(...)` 里传给 executable TMA traits，底层指令才知道这片数据要写到哪些 CTA 的 shared memory。

### 和 GEMM A/B multicast 方向的关系

GEMM 里判断 multicast 方向时，可以先看 operand 是否依赖某个 CTA tile 维度：

| operand | 逻辑 tile | 不依赖的 CTA 维度 | 可以 multicast 的方向 |
| --- | --- | --- | --- |
| A | `(M,K)` | `N` | cluster 内多个 N tile 可以共享 A。 |
| B | `(N,K)` | `M` | cluster 内多个 M tile 可以共享 B。 |
| C | `(M,N)` | 通常不共享 | 一般不做 multicast。 |

这也是 `make_tma_copy_A_sm90` 和 `make_tma_copy_B_sm90` 的源码逻辑：

```cpp
// A: Keep only MK modes from MNK, mcast along N mode for this M load.
auto cta_tiler_mk = remove<1>(cta_tiler);
auto cluster_size_n = size<1>(cluster_size);
auto cta_t_tile = make_layout(cluster_size_n);

// B: Keep only NK modes from MNK, mcast along M mode for this N load.
auto cta_tiler_nk = remove<0>(cta_tiler);
auto cluster_size_m = size<0>(cluster_size);
auto cta_t_tile = make_layout(cluster_size_m);
```

所以 `multicast layout` 的核心不是“数据怎么排”，而是“cluster 中几个 CTA 如何合作发起同一份 operand 的 TMA load”。

## `tma_partition`

`make_tma_atom` 只负责生成 atom 和 descriptor。真正把 tensor 重排成 TMA copy 参数的是 `tma_partition`。

完整重载：

```cpp
/**
 * @brief 按 TMA atom 的单指令搬运形状，对 SMEM / GMEM tensor 做分区。
 *
 * @tparam Args `Copy_Atom` 内部 traits 参数。
 * @tparam CtaCoord 当前 CTA 在 cluster / multicast layout 中的坐标。
 * @tparam TShape CTA multicast layout 的 shape 类型。
 * @tparam TStride CTA multicast layout 的 stride 类型。
 * @tparam SEngine SMEM tensor engine 类型。
 * @tparam SLayout SMEM tensor layout 类型。
 * @tparam GTensors 一个或多个 GMEM tensor 类型。
 *
 * @param copy_atom TMA copy atom。
 * @param cta_coord 当前 CTA 的 multicast 坐标。
 * @param cta_layout CTA 坐标到 logical multicast id 的 layout。
 * @param stensor SMEM tensor，通常形状是 `(TMATile, Rest...)`。
 * @param gtensors 一个或多个 GMEM tensor，形状需要和 `stensor` 第 0 维匹配。
 * @return tuple，顺序是 `(gtensors..., stensor)` 对应的 TMA 分区 tensor。
 */
template <class... Args,
          class CtaCoord,
          class TShape, class TStride,
          class SEngine, class SLayout,
          class... GTensors>
CUTE_DEVICE
auto tma_partition(Copy_Atom<Args...> const& copy_atom,
                   CtaCoord const& cta_coord,
                   Layout<TShape,TStride> const& cta_layout,
                   Tensor<SEngine,SLayout> const& stensor,
                   GTensors const&... gtensors);
```

默认重载：

```cpp
/**
 * @brief 非 multicast 场景的简化重载。
 *
 * @details
 * 等价于 `cta_coord = Int<0>{}`，`cta_layout = Layout<_1,_0>{}`。
 */
template <class... Args, class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto tma_partition(Copy_Atom<Args...> const& copy_atom,
                   Tensor<SEngine,SLayout> const& stensor,
                   Tensor<GEngine,GLayout> const& gtensor);
```

源码里最关键的几步：

```cpp
/**
 * @brief TMA partition 的核心逻辑。
 *
 * @details
 * 1. 去掉 SMEM swizzle 后求右逆，得到 smem index 到 smem coord 的映射。
 * 2. 按 TMA 单指令元素数切出 `(TMA, TMA_Iter)`。
 * 3. 根据 multicast layout 给 TMA mode 加偏移。
 * 4. 对每个 tensor 做同样的 compose / coalesce / domain_offset。
 */
Layout inv_smem_layout =
    right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

auto multicast_offset =
    cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
auto multicast_coord = make_coord(make_coord(multicast_offset, Int<0>{}));

return cute::transform(make_tuple(gtensors..., stensor), [&](auto&& tensor) {
    auto R = rank(tensor);
    auto tlayout_V = append<R>(layout_V, _);
    Tensor tensor_v =
        coalesce(tensor.compose(tlayout_V), Shape<Shape<_1,_1>>{});
    auto coord = append<R>(multicast_coord, Int<0>{});
    return domain_offset(coord, tensor_v);
});
```

为什么返回 `(gtensors..., stensor)`？

因为 CuTe copy 的调用习惯是：

```cpp
copy(copy_atom_or_traits, src, dst);
```

所以对于一个 GMEM tensor 和一个 SMEM tensor，写成：

```cpp
auto [tAgA, tAsA] = tma_partition(..., sA, gA);
copy(tma_a.with(mbar), tAgA(_,k_tile), tAsA(_,pipe));
```

`tAgA` 是 source，`tAsA` 是 destination。虽然函数参数里 `stensor` 先传入，但返回时 `stensor` 被放到最后，正好符合 `copy(src, dst)`。

## multicast mask

前面的 `multicast layout` 解决“当前 CTA 负责发起哪一片 TMA copy”。这里的 `multicast_mask` 解决另一个问题：**这条 TMA multicast 指令写到哪些 CTA 的 shared memory**。

TMA multicast 的底层指令需要一个 `uint16_t multicast_mask`。每一位表示 cluster 里的一个 CTA 是否接收这次 TMA load。

CuTe 提供：

```cpp
/**
 * @brief 根据 CTA layout 和当前 CTA 坐标生成 TMA multicast mask。
 *
 * @param cta_layout_vmnk CTA cluster layout，常按 `(V,M,N,K)` 或类似逻辑组织。
 * @param cta_coord_vmnk 当前 CTA 的逻辑坐标。
 * @return 16-bit mask，每一位对应 cluster 中一个 CTA rank。
 */
template <class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                                   CtaCoord  const& cta_coord_vmnk);
```

还有投影版本：

```cpp
/**
 * @brief 在指定 mode 上做 projection 后生成 multicast mask。
 *
 * @details
 * 例如 A operand 常沿 N 方向 multicast，可以把 M/K 固定，只展开 N 方向 CTA。
 */
template <int Mode, int... Modes, class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                                   CtaCoord  const& cta_coord_vmnk);
```

直观理解：

- 如果不 multicast，mask 就是 `0b0001`。
- 如果 cluster 中 rank 0 和 rank 1 都要接收，mask 可能是 `0b0011`。
- 如果当前 elected CTA 不是 rank 0，CuTe 会根据 `elected_cta` 对 mask 做 shift。

## `Copy_Traits` 和 `.with(...)`

TMA load 的 `Copy_Traits<SM90_TMA_LOAD, ...>` 是 non-executable 的：它有 descriptor，但没有 mbarrier。因此不能直接执行 `copy(...)`。

源码里的设计是用 `.with(...)` 生成 executable traits：

```cpp
/**
 * @brief 只持有 descriptor 的 non-executable TMA load traits。
 *
 * @details
 * 这个对象可以查询 TMA descriptor，也可以生成 TMA coord tensor。
 * 但它还没有绑定 mbarrier，所以不能直接发起 TMA load。
 */
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_> {
    TmaDescriptor tma_desc_;
    AuxParams_ aux_params_;

    /**
     * @brief 返回当前 traits 持有的 TMA descriptor。
     */
    CUTE_HOST_DEVICE constexpr
    TmaDescriptor const* get_tma_descriptor() const {
        return &tma_desc_;
    }

    /**
     * @brief 绑定 mbarrier，构造 executable TMA load traits。
     *
     * @param tma_mbar shared memory 中的 mbarrier。
     * @param multicast_mask 为了和 multicast load 保持统一而保留，普通 load 忽略。
     * @param cache_hint L2 cache hint，默认 `EVICT_NORMAL`。
     * @return 可以传给 `copy(...)` 执行的 `SM90_TMA_LOAD_OP` traits。
     */
    CUTE_HOST_DEVICE constexpr
    Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
    with(uint64_t& tma_mbar,
         uint16_t const& multicast_mask = 0,
         TMA::CacheHintSm90 const& cache_hint =
             TMA::CacheHintSm90::EVICT_NORMAL) const {
        return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
    }
};
```

multicast load 多一个 `multicast_mask`：

```cpp
/**
 * @brief 绑定 mbarrier 和 multicast mask，构造 executable multicast TMA load traits。
 *
 * @param tma_load_mbar shared memory 中的 mbarrier。
 * @param multicast_mask cluster 中接收本次 TMA load 的 CTA bitmask。
 * @param cache_hint L2 cache hint。
 * @return 可以传给 `copy(...)` 执行的 multicast TMA load traits。
 */
CUTE_HOST_DEVICE constexpr
Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
with(uint64_t& tma_load_mbar,
     uint16_t const& multicast_mask,
     TMA::CacheHintSm90 const& cache_hint =
         TMA::CacheHintSm90::EVICT_NORMAL) const;
```

store 不需要 mbarrier 作为 completion object，因为 TMA store 走的是 bulk async group：

```cpp
copy(tma_store, tCsC, tCgC);
tma_store_arrive();
tma_store_wait<0>();
```

也就是说：

- **TMA load**：常见完成信号是 `mbarrier::complete_tx::bytes`，等待 `ClusterTransactionBarrier`。
- **TMA store / reduce**：常见完成信号是 `cp.async.bulk.commit_group` / `wait_group`。

## `SM90_TMA_LOAD` 系列 arch API

`copy_sm90_tma.hpp` 是更底层的 arch-level 封装。平时写 CuTe 代码不直接调用这些 `copy`，而是用：

```cpp
copy(tma_atom.with(mbar), src, dst);
```

但理解这些结构体可以看清最终落到哪种 PTX 指令。

### load

`SM90_TMA_LOAD` 支持 1D 到 5D 坐标，并带有 `PREFETCH`：

```cpp
/**
 * @brief 发起 2D TMA load，从 global tensor map 拷贝到 shared memory。
 *
 * @param desc_ptr TMA descriptor / CUtensorMap 指针。
 * @param mbar_ptr shared memory 中的 mbarrier 指针。
 * @param cache_hint L2 cache hint。
 * @param smem_ptr shared memory 目的地址。
 * @param crd0 TMA 第 0 维坐标。
 * @param crd1 TMA 第 1 维坐标。
 */
struct SM90_TMA_LOAD_2D {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         uint64_t* mbar_ptr,
         uint64_t cache_hint,
         void* smem_ptr,
         int32_t const& crd0,
         int32_t const& crd1);

    /**
     * @brief 只预取 tensor map 相关的 L2 数据，不写 shared memory。
     */
    struct PREFETCH {
        CUTE_HOST_DEVICE static void
        copy(void const* desc_ptr,
             int32_t const& crd0,
             int32_t const& crd1);
    };
};
```

对应 PTX 形态大致是：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global
  .mbarrier::complete_tx::bytes.L2::cache_hint
  [smem_ptr], [desc, {crd0, crd1}], [mbar], cache_hint;
```

系列整理：

| API | 维度 | 方向 | 是否需要 mbarrier | 说明 |
| --- | --- | --- | --- | --- |
| `SM90_TMA_LOAD_1D` 到 `SM90_TMA_LOAD_5D` | 1D 到 5D | GMEM -> SMEM | 需要 | 发起 TMA load。 |
| `SM90_TMA_LOAD` | 1D 到 5D overload wrapper | GMEM -> SMEM | 需要 | 根据参数个数转发到对应维度。 |
| `SM90_TMA_LOAD::PREFETCH` | 1D 到 5D | GMEM descriptor / L2 prefetch | 不需要 | 预取 TMA 相关 global 数据到 L2。 |

### multicast load

multicast 版本多一个 `uint16_t multicast_mask`：

```cpp
/**
 * @brief 发起 multicast TMA load，把同一份 GMEM tile 送到多个 CTA 的 SMEM。
 *
 * @param desc_ptr TMA descriptor。
 * @param mbar_ptr 当前 CTA 或目标 CTA 的 mbarrier。
 * @param multicast_mask cluster 内接收数据的 CTA bitmask。
 * @param cache_hint L2 cache hint。
 * @param smem_ptr shared memory 目的地址。
 * @param crd0 TMA 第 0 维坐标。
 * @param crd1 TMA 第 1 维坐标。
 */
struct SM90_TMA_LOAD_MULTICAST_2D {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         uint64_t* mbar_ptr,
         uint16_t multicast_mask,
         uint64_t cache_hint,
         void* smem_ptr,
         int32_t const& crd0,
         int32_t const& crd1);
};
```

系列整理：

| API | 维度 | 方向 | 额外参数 | 说明 |
| --- | --- | --- | --- | --- |
| `SM90_TMA_LOAD_MULTICAST_1D` 到 `SM90_TMA_LOAD_MULTICAST_5D` | 1D 到 5D | GMEM -> 多个 CTA 的 SMEM | `multicast_mask` | cluster multicast load。 |
| `SM90_TMA_LOAD_MULTICAST` | 1D 到 5D overload wrapper | GMEM -> 多 CTA SMEM | `multicast_mask` | 根据坐标参数个数转发。 |
| `SM90_TMA_LOAD_MULTICAST::PREFETCH` | 1D 到 5D | L2 prefetch | 无 mask | 复用 `SM90_TMA_LOAD::PREFETCH`。 |

### im2col load

卷积场景下，TMA 可以用 im2col mode 做带 offset 的 tensor load。CuTe 封装了：

| API | 维度 | 方向 | 说明 |
| --- | --- | --- | --- |
| `SM90_TMA_LOAD_IM2COL_3D` | 3D | GMEM -> SMEM | 坐标通常类似 `(c, w, n)`，带 `w_offset`。 |
| `SM90_TMA_LOAD_IM2COL_4D` | 4D | GMEM -> SMEM | 坐标类似 `(c, w, h, n)`，带 `w/h` offset。 |
| `SM90_TMA_LOAD_IM2COL_5D` | 5D | GMEM -> SMEM | 坐标类似 `(c, w, h, d, n)`，带 `w/h/d` offset。 |
| `SM90_TMA_LOAD_IM2COL` | 3D 到 5D wrapper | GMEM -> SMEM | 根据参数个数转发。 |
| `SM90_TMA_LOAD_IM2COL_MULTICAST_3D` 到 `5D` | 3D 到 5D | GMEM -> 多 CTA SMEM | im2col + multicast。 |
| `SM90_TMA_LOAD_IM2COL_MULTICAST` | 3D 到 5D wrapper | GMEM -> 多 CTA SMEM | 根据参数个数转发。 |

一般 GEMM 不需要 im2col。卷积 lowering 或隐式 GEMM kernel 才会碰到它。

### store / reduce / bulk copy

TMA store 从 SMEM 写回 GMEM：

```cpp
/**
 * @brief 发起 2D TMA store，从 shared memory 写回 global tensor map。
 *
 * @param desc_ptr TMA descriptor。
 * @param smem_ptr shared memory 源地址。
 * @param crd0 TMA 第 0 维坐标。
 * @param crd1 TMA 第 1 维坐标。
 */
struct SM90_TMA_STORE_2D {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         void const* smem_ptr,
         int32_t const& crd0,
         int32_t const& crd1);
};
```

系列整理：

| API | 维度 | 方向 | 说明 |
| --- | --- | --- | --- |
| `SM90_TMA_STORE_1D` 到 `SM90_TMA_STORE_5D` | 1D 到 5D | SMEM -> GMEM | 发起 TMA store。 |
| `SM90_TMA_STORE` | 1D 到 5D wrapper | SMEM -> GMEM | 根据坐标参数个数转发。 |
| `SM90_TMA_STORE_IM2COL_3D` 到 `5D` | 3D 到 5D | SMEM -> GMEM | im2col store。 |
| `SM90_TMA_STORE_IM2COL` | 3D 到 5D wrapper | SMEM -> GMEM | 根据参数个数转发。 |
| `SM90_TMA_REDUCE_ADD_1D` 到 `5D` | 1D 到 5D | SMEM reduce-add 到 GMEM | 发起 `cp.reduce.async.bulk.tensor.*.add`。 |
| `SM90_TMA_REDUCE_ADD` | 1D 到 5D wrapper | SMEM reduce-add 到 GMEM | 根据坐标参数个数转发。 |

TMA store 还有几个同步辅助函数：

```cpp
/**
 * @brief 在后续 TMA store 之前，为 shared memory store 建立 async proxy 可见性。
 */
CUTE_HOST_DEVICE static void tma_store_fence();

/**
 * @brief 提交当前 warp 发出的 TMA store bulk async group。
 */
CUTE_HOST_DEVICE static void tma_store_arrive();

/**
 * @brief 等待直到最多还有 Count 个已提交 TMA store group 未完成。
 */
template <int Count>
CUTE_HOST_DEVICE static void tma_store_wait();
```

还有非 tensor map 的 bulk copy：

| API | 方向 | PTX 形态 | 说明 |
| --- | --- | --- | --- |
| `SM90_BULK_COPY_G2S` | GMEM -> SMEM | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | 按字节数拷贝，不使用多维 tensor map。 |
| `SM90_BULK_COPY_G2S::PREFETCH` | GMEM -> L2 | `cp.async.bulk.prefetch.L2.global` | bulk prefetch。 |
| `SM90_BULK_COPY_S2G` | SMEM -> GMEM | `cp.async.bulk.global.shared::cta.bulk_group` | 按字节数写回。 |
| `SM90_BULK_COPY_AUTO` | 由 traits 选择 | 无直接 copy 函数 | 用于 higher-level traits 自动选择路径。 |

## Cluster launch 和 cluster API

SM90 cluster 是一组 CTA 的集合。TMA multicast、cluster barrier、remote shared memory address mapping 都建立在 cluster 上。

### `cutlass::ClusterLaunchParams`

```cpp
/**
 * @brief CUTLASS host 侧 cluster kernel launch 参数。
 *
 * @param grid_dims grid 维度，通常已经按 cluster 维度 round up。
 * @param block_dims block 维度。
 * @param cluster_dims 每个 cluster 中 CTA 的三维形状。
 * @param smem_size_in_bytes kernel 动态 shared memory 字节数。
 * @param cuda_stream CUDA stream。
 */
struct ClusterLaunchParams {
    dim3 grid_dims{1, 1, 1};
    dim3 block_dims{1, 1, 1};
    dim3 cluster_dims{1, 1, 1};
    int smem_size_in_bytes = 0;
    cudaStream_t cuda_stream = nullptr;
};
```

### `cutlass::launch_kernel_on_cluster`

```cpp
/**
 * @brief 用 CUDA cluster launch 启动 kernel。
 *
 * @tparam Args kernel 参数类型。
 * @param params cluster launch 参数。
 * @param kernel_ptr kernel 函数指针。
 * @param args kernel 参数。
 * @return `cutlass::Status::kSuccess` 表示启动成功。
 */
template<class... Args>
CUTLASS_HOST cutlass::Status
launch_kernel_on_cluster(const ClusterLaunchParams& params,
                         void const* kernel_ptr,
                         Args&&... args);
```

它内部会把参数地址组装成 `void* kernel_params[]`，再走 CUTLASS 的 cluster launcher，底层对应 CUDA 的 `cudaLaunchKernelExC`。

### kernel 内的 cluster API

`cute/arch/cluster_sm90.hpp` 提供了一组很薄的 PTX 封装：

| API | PTX / 作用 | 说明 |
| --- | --- | --- |
| `cute::cluster_arrive_relaxed()` | `barrier.cluster.arrive.relaxed.aligned` | 到达 cluster barrier，但 relaxed。 |
| `cute::cluster_arrive()` | `barrier.cluster.arrive.aligned` | 到达 cluster barrier。 |
| `cute::cluster_wait()` | `barrier.cluster.wait.aligned` | 等待 cluster barrier。 |
| `cute::cluster_sync()` | `cluster_arrive(); cluster_wait();` | cluster 范围同步。 |
| `cute::cluster_grid_dims()` | 读 `%nclusterid.{x,y,z}` | 返回 grid 中 cluster 数量。 |
| `cute::cluster_id_in_grid()` | 读 `%clusterid.{x,y,z}` | 当前 cluster 在 grid 中的坐标。 |
| `cute::block_id_in_cluster()` | 读 `%cluster_ctaid.{x,y,z}` | 当前 CTA 在 cluster 内的三维坐标。 |
| `cute::cluster_shape()` | 读 `%cluster_nctaid.{x,y,z}` | cluster 形状。 |
| `cute::block_rank_in_cluster()` | 读 `%cluster_ctarank` | 当前 CTA 在 cluster 内的一维 rank。 |
| `cute::set_block_rank(smemAddr, rank)` | `mapa.shared::cluster.u32` | 把 shared memory 地址映射到 cluster 内某个 CTA。 |
| `cute::elect_one_sync()` | `elect.sync` | 每个 warp 选出一个 lane，常用于只让一个 lane 发 TMA。 |

示例里初始化 barrier 后立刻：

```cpp
cluster_sync();
```

这是因为 barrier 初始化只由一个 elected lane 做，其他 CTA / warp 必须等初始化对 cluster 可见后才能使用这些 mbarrier。

## kernel 内部的 barrier 和 pipeline 协议

教程示例里用了两类 barrier：

```cpp
using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
```

这两个名字非常准确：

- producer 是 TMA，它往 shared memory 生产数据，需要 transaction bytes。
- consumer 是 GMMA/WGMMA，它消耗 shared memory 数据，只需要普通 arrive / wait。

### `canonical_warp_idx_sync` 和 `elect_one_sync`

```cpp
/**
 * @brief 返回 warp 内一致的 warp index。
 *
 * @details
 * 使用 `__shfl_sync` 从 lane 0 广播 `threadIdx.x / 32`。
 * 调用时要求 warp 内线程收敛。
 */
CUTLASS_DEVICE
int canonical_warp_idx_sync() {
    return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
}
```

`cute::elect_one_sync()` 在 SM90 上用 `elect.sync`，否则退化成 lane 0：

```cpp
/**
 * @brief 从当前 warp 选出一个 lane。
 *
 * @return 被选中的 lane 返回 true，其他 lane 返回 false。
 */
CUTE_HOST_DEVICE uint32_t elect_one_sync();
```

TMA 指令通常只需要一个线程发起，所以常见判断是：

```cpp
int warp_idx = cutlass::canonical_warp_idx_sync();
int lane_predicate = cute::elect_one_sync();

if ((warp_idx == 0) && lane_predicate) {
    // 只有第 0 个 warp 里的一个 elected lane 负责发 TMA。
}
```

### `ClusterBarrier`

`ClusterBarrier` 是普通 mbarrier，只看 arrival count 和 phase：

```cpp
/**
 * @brief cluster-aware mbarrier 封装。
 *
 * @details
 * 可以初始化 arrival count，本 CTA arrive，也可以对 cluster 内远端 CTA 的
 * barrier 执行 arrive。`wait(phase)` 会等待对应 parity phase 完成。
 */
struct ClusterBarrier {
    using ValueType = uint64_t;

    static void init(ValueType const* smem_ptr, uint32_t arrive_count);
    static void wait(ValueType const* smem_ptr, uint32_t phase);
    static bool try_wait(ValueType const* smem_ptr, uint32_t phase);
    static bool test_wait(ValueType const* smem_ptr,
                          uint32_t phase,
                          uint32_t pred);
    static void arrive(ValueType const* smem_ptr);
    static void arrive(ValueType const* smem_ptr,
                       uint32_t cta_id,
                       uint32_t pred);
    static void invalidate(ValueType const* smem_ptr);
};
```

教程里 consumer barrier 的初始化是：

```cpp
ConsumerBarType::init(&consumer_mbar[pipe], 128);
```

因为一个 warpgroup 有 128 个线程。GMMA 消费完一个 pipe 后，每个相关线程 arrive，表示这一 stage 的 shared memory 可以被 producer 重用。

### `ClusterTransactionBarrier`

TMA load 需要 `ClusterTransactionBarrier`，因为它不仅要等“生产者 arrive”，还要等异步内存事务完成。

```cpp
/**
 * @brief 支持 transaction bytes 的 SM90 cluster mbarrier。
 *
 * @details
 * 和普通 barrier 不同，它还维护 expected transaction count。
 * `arrive_and_expect_tx` 会同时执行 arrive，并增加 expected transaction bytes。
 * TMA load 完成时，硬件按完成字节数扣减 transaction count。
 */
struct ClusterTransactionBarrier : public ClusterBarrier {
    static void arrive_and_expect_tx(ValueType const* smem_ptr,
                                     uint32_t transaction_bytes);
    static void arrive_and_expect_tx(ValueType const* smem_ptr,
                                     uint32_t transaction_bytes,
                                     uint32_t cta_id,
                                     uint32_t pred);
    static void expect_transaction(ValueType const* smem_ptr,
                                   uint32_t transaction_bytes);
    static void complete_transaction(ValueType const* smem_ptr,
                                     uint32_t dst_cta_id,
                                     uint32_t transaction_bytes,
                                     uint32_t pred);
};
```

普通 `arrive` 的逻辑是：

```text
arrival_count -= 1
```

`arrive_and_expect_tx(bytes)` 的逻辑是：

```text
arrival_count -= 1
expected_transaction_count += bytes
```

TMA load 指令完成时，硬件再做：

```text
expected_transaction_count -= completed_bytes
```

所以 `ProducerBarType::wait(&producer_mbar[pipe], phase)` 通过的条件是：

```text
arrival_count == 0 && expected_transaction_count == 0
```

这和普通 barrier 的差别非常关键。普通 barrier 只能说明“某个线程已经发话了”，transaction barrier 还能说明“它发出去的异步内存操作真的完成了”。

### barrier 初始化

示例中每个 pipeline stage 都有一对 barrier：

```cpp
/**
 * @brief 初始化每个 pipeline stage 的 producer / consumer barrier。
 *
 * @details
 * producer barrier 只等一个 elected TMA issuing lane arrive。
 * consumer barrier 等一个 128-thread warpgroup 消费完成。
 */
CUTE_UNROLL
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
        ProducerBarType::init(&producer_mbar[pipe], 1);
        ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
}

cluster_sync();
```

这里 `cluster_sync()` 不是可有可无。mbarrier 在 shared memory 里，初始化由一个 lane 执行，其他 CTA / warp 后续会读这些 barrier 状态，必须先保证初始化可见。

### `PipelineState`

多 stage pipeline 需要同时追踪当前 pipe index 和 phase。CUTLASS 用 `PipelineState<Stages>`：

```cpp
/**
 * @brief 环形 pipeline stage 状态。
 *
 * @tparam Stages_ pipeline stage 数。
 *
 * @details
 * `index_` 表示当前 stage 下标。
 * `phase_` 是 mbarrier parity phase，每绕环一圈翻转一次。
 * `count_` 是累计前进次数。
 */
template <uint32_t Stages_>
struct PipelineState {
    int index_ = 0;
    uint32_t phase_ = 0;
    uint32_t count_ = 0;

    int index() const;
    uint32_t phase() const;
    uint32_t count() const;

    /**
     * @brief 前进到下一个 stage；如果绕回 0，则翻转 phase。
     */
    void operator++();

    /**
     * @brief 一次前进多个 stage，并按跨越次数更新 phase。
     */
    PipelineState& advance(uint32_t num_iterations);
};
```

示例中有两个 state：

```cpp
auto write_state = cutlass::PipelineState<K_PIPE_MAX>();  // TMA writes
auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();  // MMA reads
```

- `write_state` 是 producer 视角：下一次 TMA 写哪个 pipe。
- `read_state` 是 consumer 视角：下一次 GMMA 读哪个 pipe。

### 主循环的时序

主循环可以拆成这几步：

```cpp
while (k_tile_count > -K_PIPE_MAX) {
    int read_pipe = read_state.index();

    // 1. consumer 等 TMA producer 完成该 pipe 的数据写入。
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // 2. GMMA / WGMMA 消费该 pipe。
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    // 3. consumer 通知 producer：这个 pipe 已经消费完，可以复用了。
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    // 4. producer 如果还有新 tile，就等 consumer 释放目标 pipe，然后发下一次 TMA。
    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0)) {
        int pipe = write_state.index();
        ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());

        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                              tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));

        ++write_state;
    }

    --k_tile_count;
    ++k_tile;
}
```

这个协议里有两个方向的依赖：

```text
TMA producer -> ProducerBarType -> GMMA consumer
GMMA consumer -> ConsumerBarType -> TMA producer
```

第一条保证 consumer 不会读还没搬完的 shared memory。

第二条保证 producer 不会覆盖还没被消费完的 shared memory stage。

## 一张总流程图

把 host 和 device 合起来看，CuTe TMA 的完整路径是：

```text
host:
  GMEM pointer + GMEM layout
      |
      v
  make_tensor(...)
      |
      v
  make_tma_atom(copy_op, gtensor, smem_layout_for_one_stage, cta_tile)
      |
      v
  TMA descriptor / Copy_Atom
      |
      v
  launch_kernel_on_cluster(..., tma_atom, ...)

device:
  tma_atom.get_tma_tensor(full_shape)
      |
      v
  local_tile(..., cta_coord)
      |
      v
  tma_partition(tma_atom, ..., stensor, gtensor)
      |
      v
  ProducerBarType::arrive_and_expect_tx(bytes)
      |
      v
  copy(tma_atom.with(mbarrier), tma_src, tma_dst)
      |
      v
  ProducerBarType::wait(phase)
      |
      v
  GMMA / WGMMA consume SMEM
      |
      v
  ConsumerBarType::arrive(...)
```

## 常见坑

### 把 `make_tma_atom` 当成发起拷贝

`make_tma_atom` 是 host 侧 descriptor / atom 构造，不会搬数据。真正发起 TMA 的是 device 侧：

```cpp
copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
```

### 忘记 `.with(mbarrier)`

`SM90_TMA_LOAD` traits 没有 mbarrier 时是 non-executable。TMA load 必须有 completion mbarrier，否则 consumer 没法知道数据什么时候真的写完。

### transaction bytes 算错

`arrive_and_expect_tx(bytes)` 里的 `bytes` 必须覆盖同一个 barrier phase 中所有 TMA load 写入的字节数。示例里 A/B 两次 load 共用同一个 `producer_mbar[pipe]`，所以 bytes 是 A stage 加 B stage。

### phase 没跟着环形 pipe 翻转

mbarrier wait 用的是 parity phase。`PipelineState` 每绕过 `K_PIPE_MAX` 一圈会 `phase_ ^= 1`。如果手写 pipeline，很容易只更新 pipe index，忘了 phase。

### swizzle 只想着 GMMA，忘了 TMA descriptor

SMEM swizzle 不只是 GMMA descriptor 的事。TMA descriptor 里也要填 `CUtensorMapSwizzle`，否则 TMA 写入和 GMMA 读取会对不上。

### cluster launch 和普通 launch 混用

如果 kernel 内用了 cluster API、multicast、cluster barrier，就需要用 cluster launch 路径。CUTLASS 示例通过 `ClusterLaunchParams` 和 `launch_kernel_on_cluster` 处理。

## 小结

CuTe TMA 的抽象层次可以记成三层：

1. **descriptor 层**：`make_tma_copy` / `make_tma_atom` 根据 GMEM tensor、SMEM layout、CTA tile 生成 TMA descriptor。
2. **partition 层**：`tma_partition` 把 GMEM / SMEM tensor 重排成 `copy(...)` 能接受的 TMA source / destination。
3. **execution 层**：`.with(mbarrier)` 绑定 completion barrier，`copy(...)` 发出 `cp.async.bulk.tensor`，再用 `ClusterTransactionBarrier` 和 `PipelineState` 管理异步完成与 stage 复用。

如果只看一行代码：

```cpp
copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
```

它背后其实已经包含了：

- CUDA tensor map descriptor。
- GMEM 多维坐标到地址的映射。
- SMEM swizzle 和 TMA box。
- cluster / multicast 信息。
- mbarrier transaction bytes 完成信号。
- CuTe layout partition 后的 source / destination tensor。

这也是 CuTe TMA 最有价值的地方：把 Hopper TMA 很硬件化的一组约束，折叠进了 `Tensor`、`Layout`、`Copy_Atom` 和 `PipelineState` 这几类对象里。
