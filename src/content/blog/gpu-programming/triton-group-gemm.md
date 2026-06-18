---
title: Triton 分组 GEMM
date: 2026-06-18
tags: [Triton, GPU, GEMM]
summary: 拆解 Triton 分组 GEMM 的源码、调度方式、元数据组织和 tile 级计算流程。
---

# Triton 分组 GEMM

分组 GEMM（Grouped GEMM）的目标是：**一次 kernel launch 处理一组相互独立的矩阵乘法**。

普通 GEMM 只计算一个矩阵乘法：

$$
C = A \times B,\quad A \in \mathbb{R}^{M \times K},\ B \in \mathbb{R}^{K \times N},\ C \in \mathbb{R}^{M \times N}
$$

分组 GEMM 计算的是一组 GEMM：

$$
C_g = A_g \times B_g,\quad g \in [0, group\_size)
$$

每个 GEMM 可以有不同的 $M$、$N$、$K$，也可以有不同的 leading dimension。代码中的关键设计是：**把所有 GEMM 的 tile 摊平成一个全局 tile 序列，然后让固定数量的 Triton program 以跨步方式消费这些 tile**。

## 为什么需要分组 GEMM

在很多推理和训练场景里，一个 batch 内可能有多个小矩阵乘法：

- **MoE 模型**：不同 expert 收到不同数量的 token，每个 expert 对应一个小 GEMM。
- **变长序列推理**：不同请求的 shape 不完全一致，直接合并成大矩阵不方便。
- **小 GEMM 批量执行**：如果每个小 GEMM 都单独 launch 一个 kernel，launch overhead 和 SM 利用率都会比较差。

分组 GEMM 的思路是：

- 把多个 GEMM 的输入指针、输出指针、shape 和 stride 收集成元数据数组。
- 只 launch 一个 Triton kernel。
- kernel 内部根据全局 tile id 判断当前 program 应该计算哪一个 GEMM 的哪一个 tile。
- 这样可以降低 kernel launch 次数，并让多个小 GEMM 共同填满 GPU。

## 计算任务如何被摊平

对于第 $g$ 个 GEMM：

$$
A_g[M_g, K_g] \times B_g[K_g, N_g] = C_g[M_g, N_g]
$$

如果使用 tile shape：

$$
BLOCK\_SIZE\_M \times BLOCK\_SIZE\_N
$$

那么这个 GEMM 的 tile 数量是：

$$
num\_m\_tiles = \lceil M_g / BLOCK\_SIZE\_M \rceil
$$

$$
num\_n\_tiles = \lceil N_g / BLOCK\_SIZE\_N \rceil
$$

$$
num\_tiles_g = num\_m\_tiles \times num\_n\_tiles
$$

所有 GEMM 的 tile 会按顺序串成一个全局 tile 序列：先放 `GEMM 0` 的所有 tile，再放 `GEMM 1` 的所有 tile，继续放 `GEMM 2` 的所有 tile。这个序列的每个元素都对应某个输出矩阵 `C_g` 上的一个二维 tile。

kernel 里每个 Triton program 初始拿到一个 `tile_idx = tl.program_id(0)`，然后每完成一个 tile，就执行：

```python
tile_idx += NUM_SM
```

这意味着 program 0 会处理全局 tile `0, NUM_SM, 2 * NUM_SM, ...`，program 1 会处理全局 tile `1, 1 + NUM_SM, 1 + 2 * NUM_SM, ...`。每个 program 都以 `NUM_SM` 为步长，在全局 tile 队列中拿属于自己的任务。

这种调度方式把所有 GEMM 的 tile 当作一个全局任务队列，用固定数量的 virtual SM 来切分任务。

## 数据结构总览

host 端会准备这些设备数组：

| 名称 | 形状 | 含义 |
|---|---:|---|
| `d_a_ptrs` | `[group_size]` | 每个 GEMM 的 A 矩阵起始地址。 |
| `d_b_ptrs` | `[group_size]` | 每个 GEMM 的 B 矩阵起始地址。 |
| `d_c_ptrs` | `[group_size]` | 每个 GEMM 的 C 矩阵起始地址。 |
| `d_g_sizes` | `[group_size * 3]` | 每个 GEMM 的 `M, N, K`。 |
| `d_g_lds` | `[group_size * 3]` | 每个 GEMM 的 `lda, ldb, ldc`。 |

其中 `d_g_sizes` 的布局是 `[M0, N0, K0, M1, N1, K1, M2, N2, K2, ...]`。

`d_g_lds` 的布局是 `[lda0, ldb0, ldc0, lda1, ldb1, ldc1, ...]`。

## 每个 program 负责什么区域

在 Triton 里，`tl.program_id(0)` 返回当前 program 在一维 grid 中的编号。这个编号可以理解成一个 **CTA / program 的逻辑编号**。源码里把 grid 设置为：

```python
grid = lambda META: (META['NUM_SM'], )
```

所以一次 kernel launch 会启动 `NUM_SM` 个 Triton program。这里的 `NUM_SM` 更准确地说是**虚拟并发度**，不一定严格等于物理 SM 数量。它表示有多少个 program 同时以跨步方式消费全局 tile 队列。

每个 program 负责的不是固定的某一个 GEMM，而是全局 tile 序列中的一串 tile：

| program id | 负责的全局 tile 编号 |
|---:|---|
| `0` | `0`, `0 + NUM_SM`, `0 + 2 * NUM_SM`, ... |
| `1` | `1`, `1 + NUM_SM`, `1 + 2 * NUM_SM`, ... |
| `p` | `p`, `p + NUM_SM`, `p + 2 * NUM_SM`, ... |

每个全局 tile 最终会映射到某个 GEMM 的输出矩阵 `C_g` 上的一个区域：

- **行范围**：`tile_m_idx * BLOCK_SIZE_M` 到 `(tile_m_idx + 1) * BLOCK_SIZE_M - 1`。
- **列范围**：`tile_n_idx * BLOCK_SIZE_N` 到 `(tile_n_idx + 1) * BLOCK_SIZE_N - 1`。

也就是说，一个 program 每次拿到一个 `tile_idx` 后，先判断它属于哪个 GEMM，再把它还原成 `(tile_m_idx, tile_n_idx)`，最后负责计算 `C_g` 上一个 `BLOCK_SIZE_M x BLOCK_SIZE_N` 的输出块。

以 `BLOCK_SIZE_M = 128`、`BLOCK_SIZE_N = 128` 为例，如果某个 program 当前拿到的 tile 被解析成 `tile_m_idx = 2`、`tile_n_idx = 3`，那么它负责的是：

- `C_g` 的第 `256 ~ 383` 行。
- `C_g` 的第 `384 ~ 511` 列。

这个输出区域需要沿 K 维度循环加载多块 A/B tile，做完全部累加后写回 C。

## 中文逐行注释版源码

下面的代码保留原始实现结构，但把英文注释全部改成中文，并补充每个关键变量的语义。

```python
import torch  # 导入 PyTorch，用于创建输入矩阵、保存输出矩阵，并调用 torch.matmul 做正确性和性能对照。

import triton  # 导入 Triton Python 前端，用于声明 JIT kernel、autotune 配置和 benchmark。
import triton.language as tl  # 导入 Triton kernel 内部语言，tl 中包含 load、store、dot、arange 等 GPU 编程原语。


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,  # 每个 program 在 M 维度上计算 128 行 C。
            'BLOCK_SIZE_N': 128,  # 每个 program 在 N 维度上计算 128 列 C。
            'BLOCK_SIZE_K': 32,  # 每次 K 循环加载 32 个 K 元素做一次分块 dot。
            'NUM_SM': 84,  # 使用 84 个虚拟 SM，也就是 launch 84 个 Triton program。
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,  # 使用 128 行 tile。
            'BLOCK_SIZE_N': 128,  # 使用 128 列 tile。
            'BLOCK_SIZE_K': 32,  # K 维度每次推进 32。
            'NUM_SM': 128,  # 使用 128 个虚拟 SM，让更多 program 并行消费全局 tile 队列。
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,  # 使用更小的 M tile，适合较小矩阵或寄存器压力更敏感的场景。
            'BLOCK_SIZE_N': 64,  # 使用更小的 N tile。
            'BLOCK_SIZE_K': 32,  # K tile 仍保持 32。
            'NUM_SM': 84,  # 使用 84 个虚拟 SM。
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,  # 使用 64 行 tile。
            'BLOCK_SIZE_N': 64,  # 使用 64 列 tile。
            'BLOCK_SIZE_K': 32,  # K 维度每次推进 32。
            'NUM_SM': 128,  # 使用 128 个虚拟 SM。
        }),
    ],
    key=['group_size'],  # autotune 的缓存键，group_size 变化时重新选择配置。
)
@triton.jit  # 把 Python 函数 JIT 编译成 Triton GPU kernel。
def grouped_matmul_kernel(
    group_a_ptrs,  # 设备端指针数组，group_a_ptrs[g] 保存第 g 个 A 矩阵的起始地址。
    group_b_ptrs,  # 设备端指针数组，group_b_ptrs[g] 保存第 g 个 B 矩阵的起始地址。
    group_c_ptrs,  # 设备端指针数组，group_c_ptrs[g] 保存第 g 个 C 矩阵的起始地址。
    group_gemm_sizes,  # 设备端 shape 数组，按 [M, N, K] 连续保存每个 GEMM 的规模。
    g_lds,  # 设备端 leading dimension 数组，按 [lda, ldb, ldc] 连续保存每个 GEMM 的行跨度。
    group_size,  # GEMM 的数量，也就是 group 中有多少个独立矩阵乘。
    NUM_SM: tl.constexpr,  # 编译期常量，表示虚拟 SM 数量，同时也是 grid 中 program 的数量。
    BLOCK_SIZE_M: tl.constexpr,  # 编译期常量，表示每个 tile 在 M 维度上的元素个数。
    BLOCK_SIZE_N: tl.constexpr,  # 编译期常量，表示每个 tile 在 N 维度上的元素个数。
    BLOCK_SIZE_K: tl.constexpr,  # 编译期常量，表示每次 dot 在 K 维度上消费的元素个数。
):
    tile_idx = tl.program_id(0)  # 当前 program 的初始全局 tile 编号，范围是 [0, NUM_SM)。
    last_problem_end = 0  # 当前 GEMM 之前所有 GEMM 的 tile 数量前缀和，用于判断 tile_idx 属于哪个 GEMM。

    for g in range(group_size):  # 按顺序扫描每一个 GEMM，寻找当前 tile_idx 落在哪个 GEMM 的 tile 区间内。
        gm = tl.load(group_gemm_sizes + g * 3)  # 第 g 个 GEMM 的 M，也就是 A/C 的行数。
        gn = tl.load(group_gemm_sizes + g * 3 + 1)  # 第 g 个 GEMM 的 N，也就是 B/C 的列数。
        gk = tl.load(group_gemm_sizes + g * 3 + 2)  # 第 g 个 GEMM 的 K，也就是 A 的列数和 B 的行数。

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)  # M 维度需要多少个 tile，cdiv 表示向上取整除法。
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)  # N 维度需要多少个 tile。
        num_tiles = num_m_tiles * num_n_tiles  # 第 g 个 GEMM 的总 tile 数。

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            k = gk  # 当前 GEMM 的 K 大小，后面的 K 循环会沿这个维度做累加。

            lda = tl.load(g_lds + g * 3)  # A 的 leading dimension，表示 A 相邻两行之间的元素跨度。
            ldb = tl.load(g_lds + g * 3 + 1)  # B 的 leading dimension，表示 B 相邻两行之间的元素跨度。
            ldc = tl.load(g_lds + g * 3 + 2)  # C 的 leading dimension，表示 C 相邻两行之间的元素跨度。

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))  # 第 g 个 A 矩阵的起始指针。
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))  # 第 g 个 B 矩阵的起始指针。
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))  # 第 g 个 C 矩阵的起始指针。

            tile_idx_in_gemm = tile_idx - last_problem_end  # 当前全局 tile 在第 g 个 GEMM 内部的局部 tile 编号。
            tile_m_idx = tile_idx_in_gemm // num_n_tiles  # 当前 tile 在 C 的 M 方向上的 tile 坐标。
            tile_n_idx = tile_idx_in_gemm % num_n_tiles  # 当前 tile 在 C 的 N 方向上的 tile 坐标。

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 当前 tile 覆盖的 A/C 行索引。
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 当前 tile 覆盖的 B/C 列索引。
            offs_k = tl.arange(0, BLOCK_SIZE_K)  # 当前 K 分块内部的 K 偏移。

            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]  # A tile 指针矩阵，形状为 [BM, BK]。
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]  # B tile 指针矩阵，形状为 [BK, BN]。
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # C tile 的 FP32 累加器。

            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):  # 沿 K 维度分块循环，每次处理 BLOCK_SIZE_K 个元素。
                tl.multiple_of(a_ptrs, [16, 16])  # 告诉编译器 A 指针矩阵在两个维度上满足 16 对齐，有助于优化访存。
                tl.multiple_of(b_ptrs, [16, 16])  # 告诉编译器 B 指针矩阵在两个维度上满足 16 对齐。

                a = tl.load(a_ptrs)  # 从 global memory 加载 A 的 [BM, BK] tile。
                b = tl.load(b_ptrs)  # 从 global memory 加载 B 的 [BK, BN] tile。
                accumulator += tl.dot(a, b)  # 执行矩阵乘累加，把 [BM, BK] x [BK, BN] 累加到 [BM, BN]。

                a_ptrs += BLOCK_SIZE_K  # A 指针沿 K 方向向右移动一个 K tile。
                b_ptrs += BLOCK_SIZE_K * ldb  # B 指针沿 K 方向向下移动一个 K tile。

            c = accumulator.to(tl.float16)  # 把 FP32 累加结果转换成 FP16，匹配输出 C 的 dtype。

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 当前 C tile 的行索引。
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 当前 C tile 的列索引。
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]  # C tile 指针矩阵，形状为 [BM, BN]。

            tl.store(c_ptrs, c)  # 把当前 C tile 写回 global memory。

            tile_idx += NUM_SM  # 当前 program 跳到下一个由自己负责的全局 tile。

        last_problem_end = last_problem_end + num_tiles  # 更新前缀和，进入下一个 GEMM 的 tile 区间。


def group_gemm_fn(group_A, group_B):
    device = torch.device('cuda')  # 指定所有元数据和输出矩阵都放在 CUDA 设备上。
    assert len(group_A) == len(group_B)  # 每个 A 都必须有一个对应的 B。
    group_size = len(group_A)  # group 中 GEMM 的数量。

    A_addrs = []  # 保存每个 A 矩阵的 device pointer。
    B_addrs = []  # 保存每个 B 矩阵的 device pointer。
    C_addrs = []  # 保存每个 C 矩阵的 device pointer。
    g_sizes = []  # 保存所有 GEMM 的 M、N、K。
    g_lds = []  # 保存所有 GEMM 的 lda、ldb、ldc。
    group_C = []  # 保存每个 GEMM 的输出矩阵 C。

    for i in range(group_size):
        A = group_A[i]  # 第 i 个 GEMM 的左矩阵 A。
        B = group_B[i]  # 第 i 个 GEMM 的右矩阵 B。
        assert A.shape[1] == B.shape[0]  # 检查 A 的 K 和 B 的 K 是否相等。

        M, K = A.shape  # 从 A 读取 M 和 K。
        K, N = B.shape  # 从 B 读取 K 和 N，这里的 K 会覆盖前一行的 K，但数值应相同。
        C = torch.empty((M, N), device=device, dtype=A.dtype)  # 为当前 GEMM 分配输出矩阵 C。

        group_C.append(C)  # 保存 C，函数最终会返回所有输出矩阵。
        A_addrs.append(A.data_ptr())  # 保存 A 的设备地址。
        B_addrs.append(B.data_ptr())  # 保存 B 的设备地址。
        C_addrs.append(C.data_ptr())  # 保存 C 的设备地址。
        g_sizes += [M, N, K]  # 按 [M, N, K] 追加当前 GEMM 的 shape。
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]  # 按 [lda, ldb, ldc] 追加当前 GEMM 的行跨度。

    d_a_ptrs = torch.tensor(A_addrs, device=device)  # 把 A 指针列表拷贝成 GPU 上的一维 tensor。
    d_b_ptrs = torch.tensor(B_addrs, device=device)  # 把 B 指针列表拷贝成 GPU 上的一维 tensor。
    d_c_ptrs = torch.tensor(C_addrs, device=device)  # 把 C 指针列表拷贝成 GPU 上的一维 tensor。
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)  # 把 shape 元数据放到 GPU 上。
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)  # 把 leading dimension 元数据放到 GPU 上。

    grid = lambda META: (META['NUM_SM'], )  # kernel grid 是一维的，program 数由 autotune 选择的 NUM_SM 决定。
    grouped_matmul_kernel[grid](
        d_a_ptrs,  # 传入 A 指针数组。
        d_b_ptrs,  # 传入 B 指针数组。
        d_c_ptrs,  # 传入 C 指针数组。
        d_g_sizes,  # 传入 shape 元数据。
        d_g_lds,  # 传入 leading dimension 元数据。
        group_size,  # 传入 GEMM 数量。
    )

    return group_C  # 返回所有 GEMM 的输出矩阵。


group_m = [1024, 512, 256, 128]  # 每个 GEMM 的 M 列表。
group_n = [1024, 512, 256, 128]  # 每个 GEMM 的 N 列表。
group_k = [1024, 512, 256, 128]  # 每个 GEMM 的 K 列表。
group_A = []  # 保存所有 A 矩阵。
group_B = []  # 保存所有 B 矩阵。

assert len(group_m) == len(group_n)  # 检查 M 列表和 N 列表长度一致。
assert len(group_n) == len(group_k)  # 检查 N 列表和 K 列表长度一致。
group_size = len(group_m)  # GEMM 数量。

for i in range(group_size):
    M = group_m[i]  # 第 i 个 GEMM 的 M。
    N = group_n[i]  # 第 i 个 GEMM 的 N。
    K = group_k[i]  # 第 i 个 GEMM 的 K。
    A = torch.rand((M, K), device="cuda", dtype=torch.float16)  # 创建第 i 个 A 矩阵。
    B = torch.rand((K, N), device="cuda", dtype=torch.float16)  # 创建第 i 个 B 矩阵。
    group_A.append(A)  # 保存 A。
    group_B.append(B)  # 保存 B。

tri_out = group_gemm_fn(group_A, group_B)  # 调用 Triton 分组 GEMM，得到输出列表。
ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]  # 使用 PyTorch matmul 逐个计算参考结果。

for i in range(group_size):
    assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)  # 检查 Triton 输出和 PyTorch 输出是否足够接近。


def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META['NUM_SM'], )  # 性能测试时只 launch kernel，不包含元数据构造开销。
    grouped_matmul_kernel[grid](
        a_ptrs,  # A 指针数组。
        b_ptrs,  # B 指针数组。
        c_ptrs,  # C 指针数组。
        sizes,  # shape 元数据。
        lds,  # leading dimension 元数据。
        group_size,  # GEMM 数量。
    )


def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):  # 逐个取出 A 和 B。
        torch.matmul(a, b)  # 使用 PyTorch/cuBLAS 逐个执行 GEMM。


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # benchmark 的横轴变量名是 N。
        x_vals=[2**i for i in range(7, 11)],  # N 取 128、256、512、1024。
        line_arg='provider',  # 不同曲线由 provider 参数区分。
        line_vals=['cublas', 'triton'],  # provider 的两个取值分别是 cublas 和 triton。
        line_names=["cuBLAS", "Triton"],  # 图中曲线显示名称。
        styles=[('green', '-'), ('blue', '-')],  # cuBLAS 用绿色实线，Triton 用蓝色实线。
        ylabel="runtime(ms)",  # y 轴表示运行时间，单位是毫秒。
        plot_name="group-gemm-performance",  # 图名，同时也是保存文件名。
        args={},  # benchmark 额外固定参数，这里为空。
    ))
def benchmark(N, provider):
    group_size = 4  # benchmark 固定一组 4 个 GEMM。
    group_A = []  # 保存 benchmark 输入 A。
    group_B = []  # 保存 benchmark 输入 B。
    A_addrs = []  # 保存 A 指针。
    B_addrs = []  # 保存 B 指针。
    C_addrs = []  # 保存 C 指针。
    g_sizes = []  # 保存 shape 元数据。
    g_lds = []  # 保存 leading dimension 元数据。
    group_C = []  # 保存输出 C。

    for i in range(group_size):
        A = torch.rand((N, N), device="cuda", dtype=torch.float16)  # 创建第 i 个方阵 A。
        B = torch.rand((N, N), device="cuda", dtype=torch.float16)  # 创建第 i 个方阵 B。
        C = torch.empty((N, N), device="cuda", dtype=torch.float16)  # 创建第 i 个输出 C。
        group_A.append(A)  # 保存 A。
        group_B.append(B)  # 保存 B。
        group_C.append(C)  # 保存 C。
        A_addrs.append(A.data_ptr())  # 保存 A 地址。
        B_addrs.append(B.data_ptr())  # 保存 B 地址。
        C_addrs.append(C.data_ptr())  # 保存 C 地址。
        g_sizes += [N, N, N]  # 当前 GEMM 是 N x N x N。
        g_lds += [N, N, N]  # 连续 row-major 方阵的 lda、ldb、ldc 都是 N。

    d_a_ptrs = torch.tensor(A_addrs, device="cuda")  # A 指针元数据放到 GPU。
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")  # B 指针元数据放到 GPU。
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")  # C 指针元数据放到 GPU。
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")  # shape 元数据放到 GPU。
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")  # stride 元数据放到 GPU。

    quantiles = [0.5, 0.2, 0.8]  # 返回中位数、较低分位和较高分位，用于观察波动。

    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_perf_fn(group_A, group_B),  # benchmark PyTorch/cuBLAS 逐个 GEMM 的开销。
            quantiles=quantiles,  # 使用指定分位数统计。
        )

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size),  # benchmark Triton grouped GEMM kernel。
            quantiles=quantiles,  # 使用指定分位数统计。
        )

    return ms, max_ms, min_ms  # 返回 benchmark 需要展示的时间统计值。


benchmark.run(show_plots=True, print_data=True)  # 运行 benchmark，显示图像并打印数据。
```

## kernel 参数如何理解

`grouped_matmul_kernel` 的参数可以分成四类。

**指针数组**：

- `group_a_ptrs`：保存每个 A 矩阵的地址。
- `group_b_ptrs`：保存每个 B 矩阵的地址。
- `group_c_ptrs`：保存每个 C 矩阵的地址。

这些不是二维矩阵本身，而是一个指针表。kernel 先根据 `g` 找到第几个 GEMM，再从表里取出对应矩阵地址。

**shape 元数据**：

- `group_gemm_sizes[g * 3 + 0]` 是第 `g` 个 GEMM 的 `M`。
- `group_gemm_sizes[g * 3 + 1]` 是第 `g` 个 GEMM 的 `N`。
- `group_gemm_sizes[g * 3 + 2]` 是第 `g` 个 GEMM 的 `K`。

**stride 元数据**：

- `g_lds[g * 3 + 0]` 是 `lda`。
- `g_lds[g * 3 + 1]` 是 `ldb`。
- `g_lds[g * 3 + 2]` 是 `ldc`。

这里的 leading dimension 本质是行跨度。对于 PyTorch 默认 contiguous 的二维矩阵，`stride(0)` 通常就是一行有多少个元素。

**编译期常量**：

- `BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K` 决定 tile shape。
- `NUM_SM` 决定 launch 多少个 Triton program。

Triton 里标成 `tl.constexpr` 的参数会参与编译期特化。不同 tile shape 会生成不同 kernel 版本，所以适合交给 autotune 选择。

## 全局 tile 调度逻辑

代码里最关键的逻辑是：

```python
tile_idx = tl.program_id(0)
last_problem_end = 0
```

`tile_idx` 表示当前 program 要处理的全局 tile 编号。`last_problem_end` 表示当前 GEMM 之前一共已经有多少个 tile。

对于每个 GEMM，代码会计算：

```python
num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
num_tiles = num_m_tiles * num_n_tiles
```

然后判断 `tile_idx` 是否落在当前 GEMM 的 tile 区间内：

```python
tile_idx >= last_problem_end
tile_idx < last_problem_end + num_tiles
```

如果满足条件，就说明当前 program 要计算这个 GEMM 的某个 tile。

## 从全局 tile 到局部 tile 坐标

当确定 `tile_idx` 属于第 `g` 个 GEMM 后，先得到它在这个 GEMM 内部的 tile 编号：

```python
tile_idx_in_gemm = tile_idx - last_problem_end
```

然后把一维 tile 编号还原成二维 tile 坐标：

```python
tile_m_idx = tile_idx_in_gemm // num_n_tiles
tile_n_idx = tile_idx_in_gemm % num_n_tiles
```

这相当于按 row-major 顺序排列 C 矩阵的 tile：先遍历第 0 个 M tile 行里的所有 N tile，再进入第 1 个 M tile 行，依次得到 `(0,0)`、`(0,1)`、`(0,2)`、`(1,0)`、`(1,1)`、`(1,2)` 这样的二维坐标。

`tile_m_idx` 决定当前 tile 负责 C 的哪几行，`tile_n_idx` 决定当前 tile 负责 C 的哪几列。

## 单个 tile 的 GEMM 过程

当前 program 负责一个 $BLOCK\_SIZE\_M \times BLOCK\_SIZE\_N$ 的 C tile。

它会构造 A tile 的指针矩阵：

```python
a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
```

其中：

- `offs_am[:, None]` 是 A 的行偏移，形状为 `[BM, 1]`。
- `offs_k[None, :]` 是 A 的列偏移，形状为 `[1, BK]`。
- 二者广播后得到 `[BM, BK]` 的 A tile 地址矩阵。

B tile 的指针矩阵是：

```python
b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
```

其中：

- `offs_k[:, None]` 是 B 的行偏移，形状为 `[BK, 1]`。
- `offs_bn[None, :]` 是 B 的列偏移，形状为 `[1, BN]`。
- 二者广播后得到 `[BK, BN]` 的 B tile 地址矩阵。

之后沿 K 维度循环：

```python
for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_SIZE_K
    b_ptrs += BLOCK_SIZE_K * ldb
```

每次循环做一次：

$$
[BM, BK] \times [BK, BN] \rightarrow [BM, BN]
$$

最终 `accumulator` 中就是当前 C tile 的结果。

## 为什么 tile_idx 要加 NUM_SM

每个 program 不只计算一个 tile。它计算完当前 tile 后执行：

```python
tile_idx += NUM_SM
```

假设 `NUM_SM = 4`，全局 tile 编号按 `0, 1, 2, 3, 4, 5, 6, 7, ...` 排列，那么：

- program 0 计算 tile `0, 4, 8, ...`
- program 1 计算 tile `1, 5, 9, ...`
- program 2 计算 tile `2, 6, 10, ...`
- program 3 计算 tile `3, 7, 11, ...`

这是一种静态跨步调度。它的好处是：

- launch 的 program 数固定，不需要为每个 GEMM 单独 launch。
- 小 GEMM 的 tile 可以和其他 GEMM 的 tile 混在一起执行。
- 当 group 里有多个小矩阵时，更容易填满 GPU。

限制是：

- 每个 program 要线性扫描 group，才能判断 tile 属于哪个 GEMM。
- 如果 group 很大，扫描开销会增加。
- 当前源码假设 full tile，没有处理边界 mask。

## 当前实现的边界假设

源码里有一个重要简化：**假设每个 tile 都是完整 tile**。

也就是默认 `M` 能被 `BLOCK_SIZE_M` 整除，`N` 能被 `BLOCK_SIZE_N` 整除，`K` 能被 `BLOCK_SIZE_K` 整除。

但代码里虽然用 `tl.cdiv` 计算 tile 数，真正 `tl.load` 和 `tl.store` 时没有 mask：

```python
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
tl.store(c_ptrs, c)
```

如果矩阵尺寸不能整除 tile size，最后一个 tile 会越界读写。因此真实工程版本需要加 mask：

```python
a_mask = (offs_am[:, None] < gm) & (current_k_offsets[None, :] < gk)
b_mask = (current_k_offsets[:, None] < gk) & (offs_bn[None, :] < gn)
c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)
```

然后使用：

```python
tl.load(a_ptrs, mask=a_mask, other=0.0)
tl.load(b_ptrs, mask=b_mask, other=0.0)
tl.store(c_ptrs, c, mask=c_mask)
```

这个版本为了突出分组调度，没有展开边界处理。

## host 端为什么要传指针数组

每个 GEMM 的 A、B、C 都是独立 tensor，地址不连续，shape 也可能不同。kernel 不能只靠一个 base pointer 推出所有矩阵。

所以 host 端先收集：

```python
A_addrs.append(A.data_ptr())
B_addrs.append(B.data_ptr())
C_addrs.append(C.data_ptr())
```

再把这些地址变成 GPU tensor：

```python
d_a_ptrs = torch.tensor(A_addrs, device=device)
d_b_ptrs = torch.tensor(B_addrs, device=device)
d_c_ptrs = torch.tensor(C_addrs, device=device)
```

kernel 内部再通过：

```python
a_ptr = tl.load(group_a_ptrs + g)
```

拿到第 `g` 个矩阵的真实地址。

这就是 grouped GEMM 和普通 GEMM 的核心差异之一：**普通 GEMM 的数据位置由一个 base pointer 和 stride 描述，grouped GEMM 需要一张矩阵指针表**。

## autotune 的作用

源码里 autotune 的搜索空间主要有两个维度：

- tile shape：`128x128x32` 或 `64x64x32`。
- virtual SM 数量：`84` 或 `128`。

`BLOCK_SIZE_M/N/K` 会影响：

- 单个 program 的计算量。
- accumulator 的寄存器占用。
- load/store 的访存粒度。
- Tensor Core 指令利用情况。

`NUM_SM` 会影响：

- 同时有多少 program 在消费全局 tile 队列。
- 每个 program 需要处理多少个 tile。
- 是否能充分填满 GPU。

对 grouped GEMM 来说，`NUM_SM` 不一定等于物理 SM 数量。它更像一个**虚拟工作队列并发度**，需要结合 group size、矩阵尺寸和 GPU 型号调。

## 性能测试在测什么

benchmark 中比较的是：

- `cublas`：用 PyTorch 对每组矩阵逐个调用 `torch.matmul`。
- `triton`：把所有矩阵作为一个 group，用一次 Triton grouped GEMM kernel 计算。

为了避免把元数据构造时间算进 Triton kernel，代码提供了：

```python
triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size)
```

这个函数只做 kernel launch，不重新准备 tensor 和元数据。

因此 benchmark 更接近比较：

- 多次 cuBLAS GEMM launch 的总时间。
- 一次 Triton grouped GEMM launch 的总时间。

当每个 GEMM 比较小、group size 又比较大时，grouped GEMM 通常更容易占优，因为它减少了 launch overhead，并让多个小任务共享一次 GPU 调度。

## 这份实现的核心思想

这份 Triton grouped GEMM 的核心不是单个 GEMM tile 的数学计算，因为那部分和普通 matmul kernel 很像。真正值得关注的是：

- **用指针数组描述多个独立矩阵**：每个 GEMM 的 A/B/C 地址都通过数组传入。
- **用 shape/stride 元数据描述每个 GEMM**：kernel 内部根据 `g` 读取 `M/N/K/lda/ldb/ldc`。
- **把所有 GEMM 的 tile 摊平成一个全局序列**：不同 GEMM 的 tile 共享一个全局 tile id 空间。
- **用固定数量 program 跨步消费 tile**：`tile_idx += NUM_SM` 让每个 program 处理多个分散 tile。
- **用普通 tiled GEMM 计算单个 tile**：每个 tile 内部仍然是 `tl.load -> tl.dot -> tl.store`。

理解这几个点，就能把 grouped GEMM 看成一个两层问题：

- 外层是 **跨 GEMM 的任务调度问题**。
- 内层是 **单个 tile 的矩阵乘实现问题**。

## 后续可以继续优化的方向

- **边界 mask**：支持任意 $M,N,K$，避免 full tile 假设。
- **分组扫描优化**：当 `group_size` 很大时，可以用 prefix sum 或更直接的 tile-to-group 映射减少扫描成本。
- **更细的 autotune 参数**：加入 `num_warps`、`num_stages`、不同 `BLOCK_SIZE_K`。
- **支持更多 dtype**：例如 BF16、FP8 或带 scale 的 mixed precision。
- **支持 batch 内重复 shape 的聚合优化**：如果多个 GEMM shape 相同，可以减少元数据读取和控制流分支。
