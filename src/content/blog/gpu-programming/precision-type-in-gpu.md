---
title: GPU 常见精度类型笔记
date: 2026-07-12
tags: [CUDA, GPU 编程, Precision, Tensor Core, FP8]
summary: 整理 GPU 中常见浮点和整数精度的 bit 表示、数值范围、精度取舍，以及 CUDA C++ 中对应的类型、头文件和使用场景。
---

# GPU 常见精度类型笔记

这篇笔记整理 GPU 编程里常见的数据精度：`FP64`、`FP32`、`TF32`、`FP16`、`BF16`、`FP8`、`FP4`、`FP6`、`INT8`、`INT4` 等。重点回答几个问题：

- 每种类型的 bit 是怎么分配的？
- 能表示多大的范围？
- 精度大概有多少？
- 在 GPU / Tensor Core 里通常怎么用？
- CUDA C++ 里有没有对应的类型、头文件或 fragment 类型？

本文主要参考：

- CUDA Math API：<https://docs.nvidia.com/cuda/cuda-math-api/index.html>
- CUDA Programming Guide：<https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>
- 本机 CUDA 头文件：`cuda_fp16.h`、`cuda_bf16.h`、`cuda_fp8.h`、`crt/mma.h`

## 先看总表

下面这张表先给直觉。范围是典型值，不同低精度格式在是否保留 Inf / NaN、是否饱和、是否配合 scale 使用上会有差异。

| 类型 | 总位数 | 位布局 | 典型范围 | 精度特点 | CUDA 侧常见类型 |
| --- | --- | --- | --- | --- | --- |
| `FP64` / `double` | 64 | `s1 e11(bias=1023) m52` | 约 $\pm 1.8 \times 10^{308}$ | 精度最高，代价最大 | `double` |
| `FP32` / `float` | 32 | `s1 e8(bias=127) m23` | 约 $\pm 3.4 \times 10^{38}$ | 通用训练 / 累加常用 | `float` |
| `TF32` | 19 个有效计算位，通常存于 FP32 | `s1 e8(bias=127) m10` | 近似 FP32 范围 | FP32 范围 + FP16 级尾数 | `nvcuda::wmma::precision::tf32`，输入常是 `float` |
| `FP16` / half | 16 | `s1 e5(bias=15) m10` | 最大 `65504` | 范围小，尾数 10 bit | `__half`、`half`、`__half2`、`half2` |
| `BF16` / bfloat16 | 16 | `s1 e8(bias=127) m7` | 近似 FP32 范围 | 范围大，尾数短 | `__nv_bfloat16`、`nv_bfloat16`、`__nv_bfloat162` |
| `FP8 E4M3` | 8 | `s1 e4(bias=7) m3` | 约 $\pm 448$ | 精度比 E5M2 高，范围更小 | `__nv_fp8_e4m3`、`__nv_fp8x2_e4m3`、`__nv_fp8x4_e4m3` |
| `FP8 E5M2` | 8 | `s1 e5(bias=15) m2` | 约 $\pm 57344$ | 范围更大，精度更低 | `__nv_fp8_e5m2`、`__nv_fp8x2_e5m2`、`__nv_fp8x4_e5m2` |
| `FP6 E2M3` | 6 | 常见约定 `s1 e2(bias=1) m3` | 小范围，需 scale | 新低精度，常和 scale 配合 | CUDA 13 Math API：`cuda_fp6.h` 相关类型 |
| `FP6 E3M2` | 6 | 常见约定 `s1 e3(bias=3) m2` | 比 E2M3 范围大 | 比 E2M3 精度低 | CUDA 13 Math API：`cuda_fp6.h` 相关类型 |
| `FP4 E2M1` | 4 | 常见约定 `s1 e2(bias=1) m1` | 很小，强依赖 scale | 极低精度，推理量化用 | CUDA 13 Math API：`cuda_fp4.h` 相关类型 |
| `INT8` | 8 | 整数补码或无符号 | `[-128, 127]` 或 `[0, 255]` | 固定步长，通常要 scale / zero-point | `int8_t`、`uint8_t`、`char`，Tensor Core 支持 INT8 MMA |
| `INT4` | 4 | signed / unsigned nibble | `[-8, 7]` 或 `[0, 15]` | 常 packed 存储，强依赖 scale | WMMA experimental `s4` / `u4`，库里常 packed 到 `int8_t` / `uint32_t` |
| `INT1` / binary | 1 | bit | `0/1` 或 `-1/+1` 语义 | 二值网络 / 位运算 | WMMA experimental `b1` |

读这张表时要分清两件事：

- **存储格式**：数据在 global memory / shared memory / 文件里用多少 bit 存。
- **计算格式**：Tensor Core 或 CUDA core 真正用什么格式乘加、累加。

很多低精度 GEMM 都是“低精度输入，高精度累加”：例如 FP16 输入 + FP32 accumulator、INT8 输入 + INT32 accumulator、FP8 输入 + FP16/FP32 accumulator。

## 浮点数的位布局

大多数浮点数可以用这套模型理解：

```text
sign | exponent | mantissa
符号 | 指数     | 尾数 / fraction
```

正常数通常表示为：

$$
(-1)^s \times (1.f) \times 2^{e - bias}
$$

其中：

- `s` 是符号位。
- `e` 是指数位保存的无符号整数。
- `bias` 是指数偏置。
- `f` 是尾数字段表示的小数部分。
- 正常数有隐藏的 leading `1`，所以有效精度通常是 `mantissa_bits + 1`。

对常见 IEEE 风格浮点，指数偏置通常由指数位数决定：

$$
bias = 2^{k-1} - 1
$$

其中 $k$ 是 exponent（指数）字段的 bit 数。例如：

| 格式 | 指数位数 $k$ | bias 计算 | bias |
| --- | --- | --- | --- |
| FP64 | 11 | $2^{10} - 1$ | 1023 |
| FP32 / BF16 / TF32 | 8 | $2^7 - 1$ | 127 |
| FP16 | 5 | $2^4 - 1$ | 15 |
| FP8 E5M2 | 5 | $2^4 - 1$ | 15 |
| FP8 E4M3 | 4 | $2^3 - 1$ | 7 |
| FP6 E3M2 | 3 | $2^2 - 1$ | 3 |
| FP6 E2M3 / FP4 E2M1 | 2 | $2^1 - 1$ | 1 |

特殊指数通常用于：

- `e = 0`：zero 或 subnormal（次正规数）。
- `e = all ones`：Inf / NaN，具体是否支持取决于格式。

低精度格式经常会为了多给普通数范围或数值点，牺牲一部分特殊值设计。所以 FP8 / FP4 / FP6 的细节不能简单套 IEEE FP32 的规则。

## 最大有限值怎么算

如果一个浮点格式保留 `e = all ones` 给 Inf / NaN，那么最大有限正常数通常这样取：

- **sign = 0**：取正数。
- **exponent 取最大可用值**：不能用全 1，所以是 $2^k - 2$。
- **mantissa 全 1**：让有效数字尽量接近 2。

设指数位数是 $k$，尾数字段位数是 $m$，则：

$$
E_{\max} = (2^k - 2) - bias
$$

尾数全 1 时，正常数的 significand 是：

$$
1 + (1 - 2^{-m}) = 2 - 2^{-m}
$$

所以最大有限值可以写成：

$$
x_{\max} = (2 - 2^{-m}) \times 2^{E_{\max}}
$$

### 最大值计算：以 FP16 为例

FP16 的字段是 `s1 e5 m10`：

- 指数位 $k = 5$，所以 $bias = 2^{4} - 1 = 15$。
- exponent 全 1 是 `11111_2 = 31`，通常预留给 NaN 和 $\infty$。
- 最大可用 exponent 是 `11110_2 = 30`。
- 实际指数是 $30 - 15 = 15$。
- mantissa 有 10 bit，全 1 时是 $1 + (1 - 2^{-10}) = 2 - 2^{-10}$。

因此：

$$
x_{\max}
= (2 - 2^{-10}) \times 2^{15}
= 1.9990234375 \times 32768
= 65504
$$

这就是 FP16 最大有限值 `65504` 的来源。

### 常见格式最大值速查

| 格式 | $k$ | $m$ | bias | $E_{\max}$ | 最大有限值 |
| --- | --- | --- | --- | --- | --- |
| FP64 | 11 | 52 | 1023 | 1023 | 约 $1.797693 \times 10^{308}$ |
| FP32 | 8 | 23 | 127 | 127 | 约 $3.4028235 \times 10^{38}$ |
| BF16 | 8 | 7 | 127 | 127 | 约 $3.389531 \times 10^{38}$ |
| TF32 | 8 | 10 | 127 | 127 | 约 $3.401162 \times 10^{38}$ |
| FP16 | 5 | 10 | 15 | 15 | `65504` |
| FP8 E5M2 | 5 | 2 | 15 | 15 | 约 `57344` |
| FP8 E4M3 | 4 | 3 | 7 | 8 或依具体 FP8 特殊值规则 | 常见约 `448` |

注意：FP8 E4M3、FP4、FP6 这类 micro format 的特殊值设计和 IEEE FP32/FP16 不完全一样，所以最大值最好以具体规范或 CUDA 类型文档为准。上面的公式用于建立直觉，具体到 FP8 E4M3 时要记住它的有限最大值常见写作 `448`。

## FP32

`FP32` 是最常见的单精度浮点格式，也是 CUDA C++ 里的 `float`。

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 8 | 指数，bias = 127。 |
| fraction | 23 | 尾数字段，正常数实际有效精度为 24 bit。 |

| 指标 | 值 |
| --- | --- |
| 最大有限值 | 约 $3.4028235 \times 10^{38}$ |
| 最小正规正数 | $2^{-126} \approx 1.175494 \times 10^{-38}$ |
| 最小 subnormal 正数 | $2^{-149} \approx 1.401298 \times 10^{-45}$ |
| 机器 epsilon 量级 | $2^{-23} \approx 1.19 \times 10^{-7}$ |

**使用场景**

- 通用 CUDA kernel 的默认浮点类型。
- 深度学习训练中常作为 accumulator、master weight 或 optimizer state。
- Tensor Core 混合精度里经常作为累加类型。

**CUDA 类型**

```cpp
float x = 1.0f;
float4 v;  // 向量类型，不是新的数值格式，只是 4 个 float 打包。
```

## FP64

`FP64` 是双精度浮点格式，对应 CUDA C++ 的 `double`。

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 11 | 指数，bias = 1023。 |
| fraction | 52 | 尾数字段，正常数实际有效精度为 53 bit。 |

| 指标 | 值 |
| --- | --- |
| 最大有限值 | 约 $1.797693 \times 10^{308}$ |
| 最小正规正数 | $2^{-1022} \approx 2.225074 \times 10^{-308}$ |
| 最小 subnormal 正数 | $2^{-1074} \approx 4.940656 \times 10^{-324}$ |
| 机器 epsilon 量级 | $2^{-52} \approx 2.22 \times 10^{-16}$ |

**使用场景**

- HPC、科学计算、数值模拟。
- 对误差非常敏感的归约、求解器、矩阵分解。

**注意点**

- 消费级 GPU 的 FP64 吞吐通常远低于 FP32。
- 数据带宽和寄存器压力也更大。

**CUDA 类型**

```cpp
double x = 1.0;
double2 pair;
```

## FP16

`FP16` 也叫 half precision。它是深度学习训练和推理里最常见的低精度浮点之一。

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 5 | 指数，bias = 15。 |
| fraction | 10 | 尾数字段，正常数实际有效精度为 11 bit。 |

| 指标 | 值 |
| --- | --- |
| 最大有限值 | `65504` |
| 最小正规正数 | $2^{-14} \approx 6.10352 \times 10^{-5}$ |
| 最小 subnormal 正数 | $2^{-24} \approx 5.96046 \times 10^{-8}$ |
| 机器 epsilon 量级 | $2^{-10} \approx 9.77 \times 10^{-4}$ |

**特点**

- 优点是存储小、带宽省、Tensor Core 吞吐高。
- 缺点是指数范围小，容易 overflow / underflow。
- 训练时通常需要 loss scaling，或者让 accumulator 保持 FP32。

**CUDA 类型**

使用头文件：

```cpp
#include <cuda_fp16.h>
```

常见类型：

| 类型 | 含义 |
| --- | --- |
| `__half` | CUDA half 基础类型。 |
| `half` | `__half` 的别名，CUDA 文档说它意图作为 half 的一等类型。 |
| `__half2` | 两个 half 打包，常用于向量化 half 运算。 |
| `half2` | `__half2` 的别名。 |
| `__half_raw` / `__half2_raw` | 只暴露原始 bit 的 raw 类型，适合静态初始化或位级搬运。 |

示例：

```cpp
#include <cuda_fp16.h>

__global__ void half_kernel(const float* input, half* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // float -> half，默认四舍五入到 nearest-even 语义的转换函数之一。
    output[idx] = __float2half(input[idx]);
}
```

**`half2` 为什么常见**

`half2` 把两个 FP16 放在一个 32-bit 容器里，很多 intrinsic 可以一次处理两个 half：

```cpp
__half2 a;
__half2 b;
__half2 c = __hadd2(a, b);
```

这类 vectorized half 运算可以提升吞吐和内存访问效率。

## BF16

`BF16` 是 bfloat16。它也是 16 bit，但它的设计目标和 FP16 不同：**保留 FP32 的指数范围，牺牲尾数精度**。

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 8 | 指数，bias = 127，和 FP32 一样。 |
| fraction | 7 | 尾数字段，正常数实际有效精度为 8 bit。 |

| 指标 | 值 |
| --- | --- |
| 最大有限值 | 约 $3.39 \times 10^{38}$，接近 FP32 |
| 最小正规正数 | $2^{-126}$，接近 FP32 |
| 最小 subnormal 正数 | 约 $2^{-133}$ |
| 机器 epsilon 量级 | $2^{-7} \approx 7.8125 \times 10^{-3}$ |

**FP16 和 BF16 的核心区别**

| 类型 | 指数范围 | 尾数精度 | 直觉 |
| --- | --- | --- | --- |
| FP16 | 小 | 相对更高 | 数值范围窄，但小范围内更细。 |
| BF16 | 大，接近 FP32 | 更低 | 不容易 overflow / underflow，但量化更粗。 |

**使用场景**

- 训练中比 FP16 更不容易数值溢出。
- 大模型训练和推理中很常见。
- 常用 BF16 输入 + FP32 累加。

**CUDA 类型**

使用头文件：

```cpp
#include <cuda_bf16.h>
```

常见类型：

| 类型 | 含义 |
| --- | --- |
| `__nv_bfloat16` | CUDA bfloat16 基础类型。 |
| `nv_bfloat16` | `__nv_bfloat16` 的别名。 |
| `__nv_bfloat162` | 两个 BF16 打包。 |
| `nv_bfloat162` | `__nv_bfloat162` 的别名。 |
| `__nv_bfloat16_raw` / `__nv_bfloat162_raw` | raw bit 表示。 |

示例：

```cpp
#include <cuda_bf16.h>

__global__ void bf16_kernel(const float* input, nv_bfloat16* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = __float2bfloat16(input[idx]);
}
```

## TF32

`TF32` 是 TensorFloat-32。它不是普通 C++ 里的一个独立存储类型，更像是 NVIDIA Tensor Core 面向 FP32 输入的一种计算格式。

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 8 | 指数，bias = 127，和 FP32 一样。 |
| fraction | 10 | 尾数字段，接近 FP16 的尾数位数。 |

| 指标 | 值 |
| --- | --- |
| 范围 | 接近 FP32 |
| 精度 | 接近 FP16 的尾数精度 |
| 存储 | 通常仍以 `float` 存储和传参 |
| 计算 | Tensor Core 内部按 TF32 乘法，常用 FP32 累加 |

**直觉**

TF32 是“FP32 范围 + 低尾数精度”的折中。它让很多原本写成 FP32 GEMM 的代码可以在 Ampere 以后自动或半自动用 Tensor Core 加速。

**CUDA 类型和接口**

普通 CUDA C++ 里通常还是写：

```cpp
float a;
float b;
float c;
```

但在 WMMA 里有一个精度 tag：

```cpp
#include <mma.h>

using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 8,
               wmma::precision::tf32, wmma::row_major> a_frag;
```

本机 CUDA 头文件 `crt/mma.h` 里可以看到：

```cpp
namespace nvcuda {
namespace wmma {
namespace precision {
    struct tf32;
}
}
}
```

并且有 `__float_to_tf32(float)` 这类帮助函数，用于把 FP32 值转换成 TF32 表示再参与 WMMA 路径。

**注意点**

- TF32 不是 `__tf32` 这种可以像 `__half` 一样到处声明的普通 CUDA 数据类型。
- cuBLAS / cuDNN 里是否使用 TF32 通常由 math mode 或 compute type 控制。
- 如果你需要严格 FP32 数值行为，要注意库是否默认启用了 TF32。

## FP8

FP8 是 8 bit 浮点。NVIDIA CUDA Math API 里主要提供两种 FP8 数据格式：

- `E4M3`：4 bit exponent，3 bit mantissa。
- `E5M2`：5 bit exponent，2 bit mantissa。

二者都只有 8 bit，所以范围和精度必须取舍。

### FP8 E4M3

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 4 | 指数，bias = 7。 |
| mantissa | 3 | 尾数字段。 |

| 指标 | 典型值 |
| --- | --- |
| 最大有限值 | 约 `448` |
| 最小正规正数 | $2^{-6} = 0.015625$ |
| 最小 subnormal 正数 | $2^{-9} = 0.001953125$ |
| 精度 | 比 E5M2 更高 |

**使用场景**

- 更适合权重或激活中范围较受控、希望多一点尾数精度的场景。
- 常配合 per-tensor / per-channel / per-block scaling。

### FP8 E5M2

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 5 | 指数，bias = 15。 |
| mantissa | 2 | 尾数字段。 |

| 指标 | 典型值 |
| --- | --- |
| 最大有限值 | 约 `57344` |
| 最小正规正数 | $2^{-14} \approx 6.10352 \times 10^{-5}$ |
| 最小 subnormal 正数 | $2^{-16} \approx 1.52588 \times 10^{-5}$ |
| 精度 | 比 E4M3 更低，但范围更大 |

**使用场景**

- 更适合梯度这类动态范围更大的数据。
- 仍然强依赖 scale，否则 8 bit 很容易溢出或量化过粗。

### CUDA FP8 类型

CUDA Math API 说明 FP8 intrinsics 需要：

```cpp
#include <cuda_fp8.h>
```

常见类型：

| 类型 | 含义 |
| --- | --- |
| `__nv_fp8_e4m3` | 单个 E4M3 FP8 值。 |
| `__nv_fp8_e5m2` | 单个 E5M2 FP8 值。 |
| `__nv_fp8x2_e4m3` | 两个 E4M3 FP8 值打包。 |
| `__nv_fp8x2_e5m2` | 两个 E5M2 FP8 值打包。 |
| `__nv_fp8x4_e4m3` | 四个 E4M3 FP8 值打包。 |
| `__nv_fp8x4_e5m2` | 四个 E5M2 FP8 值打包。 |

示例：

```cpp
#include <cuda_fp8.h>

__global__ void fp8_store_kernel(const float* input, __nv_fp8_e4m3* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 构造函数会把 float 转成 E4M3 FP8，具体转换和饱和语义看 CUDA 版本文档。
    output[idx] = __nv_fp8_e4m3(input[idx]);
}
```

**注意点**

- FP8 类型主要是存储 / 搬运 / 转换类型。真正高性能计算通常通过 Tensor Core 或库路径完成。
- 大多数 FP8 模型还需要保存 scale，例如 `x_fp8 = round(x / scale)`，计算时再配合 scale 恢复数值意义。
- 不同框架的 FP8 recipe 会决定 E4M3 / E5M2 用在哪些张量上。

## FP4 和 FP6

CUDA 13.3 Math API 已经列出 FP4 和 FP6 intrinsics：

- FP4：`cuda_fp4.h`，例如 E2M1 相关类型。
- FP6：`cuda_fp6.h`，例如 E2M3、E3M2 相关类型。

它们比 FP8 更激进，通常不是“直接拿来当普通 float 用”，而是和 scale、packed layout、Tensor Core / library kernel 紧密绑定。

### FP4 E2M1

FP4 只有 4 bit。常见 E2M1 可以理解为：

| 字段 | 位数 | 含义 |
| --- | --- | --- |
| sign | 1 | 符号位。 |
| exponent | 2 | 指数，常见 E2M1 约定 bias = 1。 |
| mantissa | 1 | 尾数字段。 |

**特点**

- 可表示点非常少。
- 单独看原始值意义有限，几乎总要配合 scale。
- 适合极低比特推理量化，例如权重量化。

### FP6 E2M3 / E3M2

FP6 有 6 bit，常见两种取舍：

| 格式 | 位布局 | 常见 bias | 直觉 |
| --- | --- | --- | --- |
| E2M3 | `s1 e2 m3` | 1 | 指数少、尾数多，范围更小但精度相对更高。 |
| E3M2 | `s1 e3 m2` | 3 | 指数多、尾数少，范围更大但精度更低。 |

**注意点**

- 官方文档提示这类新低精度操作在特定 GPU target 上才更能受益于 native hardware support，其他目标可能走 emulation path。
- 如果本机 CUDA Toolkit 没有 `cuda_fp4.h` / `cuda_fp6.h`，说明本机版本还没提供这些头文件；需要以安装的 CUDA 版本为准。

## 整数精度

整数类型没有 exponent / mantissa。它们表达的是固定步长的离散整数。

整数的 bit 表示要先区分 signed 和 unsigned：

- **unsigned integer（无符号整数）**：所有 bit 都表示数值，$n$ bit 范围是 $[0, 2^n - 1]$。
- **signed integer（有符号整数）**：现代 GPU / CPU 基本按 two's complement（补码）理解，$n$ bit 范围是 $[-2^{n-1}, 2^{n-1}-1]$。

补码的读法可以记成：

$$
\text{signed\_value} =
\begin{cases}
u, & \text{最高位为 0} \\
u - 2^n, & \text{最高位为 1}
\end{cases}
$$

其中 $u$ 是把这 $n$ 个 bit 当成无符号整数时的值。

### INT32

| 类型 | 范围 |
| --- | --- |
| `int32_t` | `[-2147483648, 2147483647]` |
| `uint32_t` | `[0, 4294967295]` |

`uint32_t` 和 `int32_t` 的 bit 宽度相同，区别是解释方式：

| bit pattern | `uint32_t` 解释 | `int32_t` 补码解释 |
| --- | --- | --- |
| `00000000 00000000 00000000 00000000` | 0 | 0 |
| `00000000 00000000 00000000 00000001` | 1 | 1 |
| `01111111 11111111 11111111 11111111` | 2147483647 | 2147483647 |
| `10000000 00000000 00000000 00000000` | 2147483648 | -2147483648 |
| `11111111 11111111 11111111 11111111` | 4294967295 | -1 |

**使用场景**

- INT8 GEMM 的 accumulator 常用 INT32。
- 索引、计数、offset。

### INT8

| 类型 | 范围 |
| --- | --- |
| `int8_t` | `[-128, 127]` |
| `uint8_t` | `[0, 255]` |

8 bit 的例子更容易看清补码：

| bit pattern | `uint8_t` 解释 | `int8_t` 补码解释 |
| --- | --- | --- |
| `00000000` | 0 | 0 |
| `00000001` | 1 | 1 |
| `01111111` | 127 | 127 |
| `10000000` | 128 | -128 |
| `11111110` | 254 | -2 |
| `11111111` | 255 | -1 |

所以同样的 8 个 bit，放进 `uint8_t` 和 `int8_t` 里看到的数值可能完全不同。量化 kernel 里要特别小心 signed / unsigned 语义，不要只看存储字节。


**CUDA 类型**

```cpp
#include <cstdint>

int8_t s8;
uint8_t u8;
char4 packed4;  // 常见向量化搬运容器，不等于 4-bit。
```

Tensor Core 支持 INT8 MMA，但在 CUDA C++ 层通常通过 WMMA、PTX、CUTLASS、cuBLASLt 或框架 kernel 使用。

### INT4

INT4 只有 4 bit：

| 语义 | 范围 |
| --- | --- |
| signed INT4 | `[-8, 7]` |
| unsigned INT4 | `[0, 15]` |

INT4 也可以用同样的补码规则理解：

| bit pattern | unsigned INT4 | signed INT4 补码 |
| --- | --- | --- |
| `0000` | 0 | 0 |
| `0001` | 1 | 1 |
| `0111` | 7 | 7 |
| `1000` | 8 | -8 |
| `1110` | 14 | -2 |
| `1111` | 15 | -1 |

因为 C++ 没有标准 `int4_t` 标量类型，所以 INT4 通常 packed 存储：

- 两个 INT4 放进一个 `uint8_t`。
- 八个 INT4 放进一个 `uint32_t`。
- kernel 内用 bit 操作解包。

示意：

```cpp
uint8_t pack_int4(uint8_t lo, uint8_t hi)
{
    return static_cast<uint8_t>((hi << 4) | (lo & 0x0f));
}

uint8_t unpack_lo(uint8_t x)
{
    return x & 0x0f;
}

uint8_t unpack_hi(uint8_t x)
{
    return x >> 4;
}
```

WMMA 里可以看到 experimental sub-byte 类型 tag：

```cpp
#include <mma.h>

nvcuda::wmma::experimental::precision::u4; // unsigned 4-bit
nvcuda::wmma::experimental::precision::s4; // signed 4-bit
nvcuda::wmma::experimental::precision::b1; // 1-bit
```

这些是 WMMA fragment 的 element tag，不是普通 C++ 标量变量类型。

## 浮点类型转换

实际 CUDA 代码里，经常需要在 `float`、`half`、`bfloat16`、`fp8` 之间转换。转换时要关心三件事：

- **舍入方式**：截断、round to nearest even、round toward zero 等会影响误差分布。
- **饱和 / 溢出行为**：低精度格式范围小，超出范围时是饱和、变 Inf、还是产生其他特殊值，要看接口语义。
- **是否只是存储转换**：例如 FP32 -> BF16 通常是把 FP32 尾数截短或舍入，真正计算时可能又转回 FP32 / Tensor Core 内部格式。

### CUDA 内置转换函数

CUDA 对 FP16 和 BF16 提供了直接转换函数。常见头文件是：

```cpp
#include <cuda_fp16.h>
#include <cuda_bf16.h>
```

示例：

```cpp
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void convert_kernel(
    const float* f32_input,
    __half* f16_output,
    __nv_bfloat16* bf16_output,
    float* f32_from_f16,
    float* f32_from_bf16)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = f32_input[idx];

    // FP32 -> FP16。会发生范围和尾数精度收缩。
    __half h = __float2half(value);
    f16_output[idx] = h;

    // FP32 -> BF16。保留 FP32 的 8 bit exponent，但尾数变短。
    __nv_bfloat16 b = __float2bfloat16(value);
    bf16_output[idx] = b;

    // 反向转换回 FP32。注意这只能恢复低精度值对应的 FP32 表示，
    // 不能恢复转换时已经丢掉的尾数信息。
    f32_from_f16[idx] = __half2float(h);
    f32_from_bf16[idx] = __bfloat162float(b);
}
```

FP8 也有对应构造和转换接口，常见头文件是：

```cpp
#include <cuda_fp8.h>
```

示例：

```cpp
#include <cuda_fp8.h>

__global__ void fp8_convert_kernel(
    const float* input,
    __nv_fp8_e4m3* e4m3_output,
    __nv_fp8_e5m2* e5m2_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = input[idx];

    // E4M3 更偏精度，E5M2 更偏范围。
    e4m3_output[idx] = __nv_fp8_e4m3(value);
    e5m2_output[idx] = __nv_fp8_e5m2(value);
}
```

### 位运算模拟：FP32 转 BF16

BF16 和 FP32 的位布局很适合拿来理解转换：

```text
FP32: s1 e8 m23  ->  32 bit
BF16: s1 e8 m7   ->  16 bit
```

BF16 保留 FP32 的高 16 bit：

- sign 1 bit 保留。
- exponent 8 bit 保留。
- fraction 只保留最高 7 bit。
- FP32 低 16 bit 被丢弃或参与舍入。

如果只是截断，可以这样理解：

```cpp
#include <cstdint>

__device__ uint16_t float_to_bf16_truncate(float value)
{
    // CUDA device 代码里可以用 __float_as_uint 做按位解释，
    // 它不会做数值转换，只是把 float 的 bit 当成 uint32_t 看。
    uint32_t bits = __float_as_uint(value);
    return static_cast<uint16_t>(bits >> 16);
}
```

如果想模拟 round to nearest 的直觉版本，可以先加半个被舍弃区间，再截断：

```cpp
#include <cstdint>

__device__ uint16_t float_to_bf16_round_simple(float value)
{
    uint32_t bits = __float_as_uint(value);

    // 要丢掉低 16 bit。低 16 bit 的“一半”是 bit 15，
    // 也就是 0x00008000。先加半个单位，再右移截断。
    bits += 0x00008000u;
    return static_cast<uint16_t>(bits >> 16);
}
```

如果在 host C++ 里写同样的“按位解释”，常见写法是 `std::memcpy` 或 C++20 `std::bit_cast`，避免违反 strict aliasing：

```cpp
#include <cstdint>
#include <cstring>

uint16_t float_to_bf16_truncate_host(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}
```

### 为什么加 `0x00008000`

可以先用十进制四舍五入类比。

如果要把 `3.1415` 保留两位小数，可以先加 `0.005`，再截断：

- `3.141 + 0.005 = 3.146`，截断到两位小数是 `3.14`。
- `3.146 + 0.005 = 3.151`，截断到两位小数是 `3.15`。

本质是：**如果被舍弃部分大于等于半个单位，加上半个单位会向保留部分进位**。

FP32 转 BF16 时：

- FP32 有 32 bit。
- BF16 保留高 16 bit。
- 要舍弃低 16 bit。
- 低 16 bit 的范围是 `0x0000` 到 `0xffff`。
- 这个范围的一半就是 bit 15 置 1，也就是 `0x00008000`。

所以：

```text
bits += 0x00008000
bf16 = bits >> 16
```

直觉上就是“先加半个低 16 bit 单位，再截断”。

**为什么比纯截断好**

- 纯截断总是把低位丢掉，误差有方向性。
- 加半单位再截断更接近 round to nearest，误差更对称。
- 在深度学习长链路计算里，误差偏置可能不断积累，所以舍入策略会影响稳定性。

真实硬件和 CUDA 内置函数还会处理 NaN、Inf、round-to-nearest-even、饱和等边界情况。上面的代码主要用于理解 BF16 为什么可以看成“FP32 高 16 bit 加舍入”。

## CUDA 类型速查

| 精度 | 头文件 | CUDA C++ 类型 / tag | 说明 |
| --- | --- | --- | --- |
| FP64 | 无特殊头文件 | `double` | C++ 内建类型。 |
| FP32 | 无特殊头文件 | `float` | C++ 内建类型。 |
| TF32 | `<mma.h>` | `nvcuda::wmma::precision::tf32` | WMMA precision tag，不是普通存储类型。 |
| FP16 | `<cuda_fp16.h>` | `__half`、`half`、`__half2`、`half2` | half 和 half2 类型及 intrinsic。 |
| BF16 | `<cuda_bf16.h>` | `__nv_bfloat16`、`nv_bfloat16`、`__nv_bfloat162` | bfloat16 类型及 intrinsic。 |
| FP8 E4M3 | `<cuda_fp8.h>` | `__nv_fp8_e4m3`、`__nv_fp8x2_e4m3`、`__nv_fp8x4_e4m3` | FP8 E4M3 存储 / 转换类型。 |
| FP8 E5M2 | `<cuda_fp8.h>` | `__nv_fp8_e5m2`、`__nv_fp8x2_e5m2`、`__nv_fp8x4_e5m2` | FP8 E5M2 存储 / 转换类型。 |
| FP6 | `<cuda_fp6.h>` | CUDA 13 Math API 中的 FP6 类型 | 新低精度，依赖 CUDA 版本和目标架构。 |
| FP4 | `<cuda_fp4.h>` | CUDA 13 Math API 中的 FP4 类型 | 新低精度，依赖 CUDA 版本和目标架构。 |
| INT32 | `<cstdint>` | `int32_t`、`uint32_t` | 常用 accumulator / index。 |
| INT8 | `<cstdint>` | `int8_t`、`uint8_t` | 量化推理常用输入。 |
| INT4 | `<mma.h>` 或自定义 packed | `wmma::experimental::precision::s4/u4`，或 packed `uint8_t` | 没有标准 C++ 标量 `int4_t`。 |
| INT1 | `<mma.h>` 或 bit packed | `wmma::experimental::precision::b1`，或 packed bit | 二值 / 位运算。 |



## 训练和推理里的常见组合

| 场景 | 常见输入 | 常见累加 | 说明 |
| --- | --- | --- | --- |
| 传统训练 | FP32 | FP32 | 稳定但慢、显存大。 |
| 混合精度训练 | FP16 | FP32 | 需要关注 overflow / loss scaling。 |
| 大模型训练 | BF16 | FP32 | 范围接近 FP32，更省心。 |
| Ampere FP32 GEMM 加速 | TF32 | FP32 | 输入仍像 FP32，Tensor Core 内部用 TF32。 |
| FP8 训练 / 推理 | FP8 E4M3 / E5M2 | FP16 / FP32 | 需要 scale 和 recipe。 |
| INT8 推理 | INT8 | INT32 或 FP32 后处理 | 量化推理常见。 |
| INT4 / FP4 权重量化 | INT4 / FP4 | FP16 / BF16 / FP32 | 权重极低比特，activation 常更高精度。 |

## 选择精度时看什么

- **范围够不够**：梯度和 activation 容易有大动态范围，BF16 / E5M2 比 FP16 / E4M3 更抗 overflow。
- **精度够不够**：权重小扰动是否重要，尾数越短量化误差越大。
- **累加类型是什么**：低精度输入如果用高精度累加，通常会稳很多。
- **硬件是否原生支持**：同样是 FP8 / FP4，是否走 Tensor Core、是否 emulation，性能差别很大。
- **库路径是否支持**：cuBLASLt、CUTLASS、Transformer Engine、PyTorch kernel 是否有对应 fast path。
- **scale 粒度是否合理**：低比特格式的效果往往由 scale 策略决定，而不是只由 bit 数决定。

## 小结

可以用一句话记住这些类型的取舍：

- `FP64`：最稳，最贵。
- `FP32`：通用基准。
- `TF32`：FP32 范围，Tensor Core 更快，尾数变短。
- `FP16`：范围小，吞吐高，训练要小心溢出。
- `BF16`：范围接近 FP32，尾数更粗，大模型训练常用。
- `FP8 E4M3`：更偏精度。
- `FP8 E5M2`：更偏范围。
- `FP4 / FP6 / INT4`：极低比特，通常离不开 scale 和专用 kernel。
- `INT8`：推理量化经典选择。

CUDA 里确实有不少对应类型：`__half`、`__nv_bfloat16`、`__nv_fp8_e4m3`、`__nv_fp8_e5m2` 等是真正的 CUDA C++ 类型；`tf32`、`s4/u4/b1` 更像 WMMA fragment 的 precision tag；`INT4` 这类通常要 packed 存储，不能当成普通 C++ 标量类型来用。
