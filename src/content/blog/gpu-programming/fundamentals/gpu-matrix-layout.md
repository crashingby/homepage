---
title: GPU 矩阵布局：如何理解逻辑矩阵和内存优先的关系
date: 2026-09-21
tags: [CUDA, cuBLAS, GEMM, Matrix Layout, GPU 编程]
summary: 从传统 cuBLAS 的 column-major 契约出发，逐项解释 GEMM 的 M/N/K、leading dimension、转置选项与 row-major 参数映射。
---

# GPU 矩阵布局

本文的目标不是优化 GEMM，而是把常见的困惑拆开：逻辑矩阵和实际存储计算需求的内存布局到底要如何对应？

## cuBLAS GEMM 的固定数学语义

以单精度接口为例：

```cpp
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc);
```

无论指针来自 C++ 的 `std::vector`、`cudaMalloc` 还是别的内存所有者，接口都按下式计算：

$$
C \leftarrow \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(B) + \beta \cdot C
$$

其中结果矩阵 $C$ 的逻辑形状永远是 $M \times N$。`m`、`n`、`k` 描述的是**运算后的逻辑矩阵**，不是某个原始 buffer 的字节数：

| 参数 | 含义 | 在 GEMM 中的位置 |
| --- | --- | --- |
| `m` | $C$ 的行数 | $\operatorname{op}(A)$ 的行数 |
| `n` | $C$ 的列数 | $\operatorname{op}(B)$ 的列数 |
| `k` | 归约维度 | $\operatorname{op}(A)$ 的列数，也是 $\operatorname{op}(B)$ 的行数 |
| `alpha` | 矩阵乘积的缩放系数 | 乘在 $\operatorname{op}(A)\operatorname{op}(B)$ 前 |
| `beta` | 原有 $C$ 的缩放系数 | 乘在写回前的旧 $C$ 前 |

因此，若 `m = 2, n = 3, k = 4`，则最终一定在计算一个 $2 \times 3$ 的 $C$，并对每个元素做长度为 4 的点积。`A` 和 `B` 在**未转置时**分别是 $2 \times 4$ 与 $4 \times 3$。

### `N`、`T`、`C`：接口实际拿哪一个矩阵参与乘法

`transa` 和 `transb` 分别控制 `A`、`B` 在公式中对应的 `op`：

| 选项 | `op(X)` | 适用场景 |
| --- | --- | --- |
| `CUBLAS_OP_N`（`N`） | $X$ | 不转置 |
| `CUBLAS_OP_T`（`T`） | $X^T$ | 实数矩阵的转置，或复数矩阵只交换行列、不共轭 |
| `CUBLAS_OP_C`（`C`） | $X^H$ | 复数矩阵的共轭转置；对实数等价于 `T` |

所以人们常写的 `NN`、`TN`、`NT`、`TT`，只是 `(transa, transb)` 的缩写。例如 `TN` 是：

$$
C_{M\times N} = \alpha A^T_{M\times K} B_{K\times N} + \beta C_{M\times N}
$$

`m/n/k` 的确描述的是 **`op(A)`、`op(B)` 和 `C` 在本次运算中应当呈现的形状** 。但 `T` 不会令 cuBLAS 先把整个 `A` 读到某个临时矩阵、再写出一个转置副本；它是**每次逻辑取元素时的坐标变换**。实际 GEMM kernel 会以 tile、warp 和寄存器 fragment 实现这一点，但从 API 语义看，等价于“先根据 `op` 选取元素，再做点积”。

### 将 `op`、`M/N/K` 与 `ld` 放进同一个寻址式

为了避免把“原始矩阵”和“参与乘法的矩阵”混为一谈，先给原始 device buffer 对应的矩阵取名 $A_0$、$B_0$。对任意输出坐标 $(i,j)$，其中 $0\le i<M$、$0\le j<N$，以及归约坐标 $0\le r<K$，cuBLAS 的可观察语义等价于：

```cpp
float acc = 0.0F;
for (int r = 0; r < k; ++r) {
    const float a = (transa == CUBLAS_OP_N)
        ? A[i + r * lda]  // op(A0)[i, r] = A0[i, r]
        : A[r + i * lda]; // op(A0)[i, r] = A0[r, i]

    const float b = (transb == CUBLAS_OP_N)
        ? B[r + j * ldb]  // op(B0)[r, j] = B0[r, j]
        : B[j + r * ldb]; // op(B0)[r, j] = B0[j, r]

    acc += a * b;
}
C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
```

这不是 cuBLAS 的真实 kernel 源码，而是其 **column-major 接口必须满足的逐元素语义**。它把三件事分开了：

- **`m/n/k` 先决定逻辑循环范围**：`i` 走 $[0,M)$，`j` 走 $[0,N)$，`r` 走 $[0,K)$。因此它们规定了 `op(A0)` 是 $M\times K$、`op(B0)` 是 $K\times N$，以及 `C` 是 $M\times N$。
- **`T` 在寻址前交换逻辑坐标**：`N` 时读取 $A_0[i,r]$；`T` 时要得到同一个逻辑位置的 $\operatorname{op}(A_0)[i,r]$，于是改读 $A_0[r,i]$。它没有生成转置后的连续 buffer。
- **`ld` 始终属于原始物理 buffer**：cuBLAS 在得到原始矩阵坐标后，才套 column-major 地址公式 `row + col * ld`。所以 `lda` 是 $A_0$ 的列间距，不是转置后 `op(A0)` 的列间距。

也可以把形状与寻址关系压缩成下表：

| 操作数 | 选项 | 原始矩阵的物理形状 | 逻辑乘法时取的元素 | 原始 buffer 中的偏移 |
| --- | --- | --- | --- | --- |
| `A` | `N` | $A_0: M\times K$ | $\operatorname{op}(A_0)_{i,r}=A_{0,i,r}$ | `i + r * lda` |
| `A` | `T` / `C` | $A_0: K\times M$ | $\operatorname{op}(A_0)_{i,r}=A_{0,r,i}$ | `r + i * lda` |
| `B` | `N` | $B_0: K\times N$ | $\operatorname{op}(B_0)_{r,j}=B_{0,r,j}$ | `r + j * ldb` |
| `B` | `T` / `C` | $B_0: N\times K$ | $\operatorname{op}(B_0)_{r,j}=B_{0,j,r}$ | `j + r * ldb` |
| `C` | 无选项 | $C: M\times N$ | $C_{i,j}$ | `i + j * ldc` |

对于复数 `CUBLAS_OP_C`，地址和 `T` 完全相同，只是读取元素后还会取复共轭。实数 `float` / `double` 下，`C` 与 `T` 的数值结果相同。

### column-major 下，元素究竟落在哪个地址

传统 BLAS 的基本地址公式是：

$$
\operatorname{offset}_{\text{col}}(i, j) = i + j \cdot ld
$$

这里使用从 0 开始的坐标；`i` 是行号、`j` 是列号，`ld` 是该矩阵的 leading dimension。也就是说，**同一列的相邻元素连续**，从一列跳到下一列要跨过 `ld` 个元素。

`ld` 并不等于“矩阵的某个神秘维度”；它就是这段 column-major 内存中相邻两列起始地址的距离。紧凑存储时，`ld` 等于矩阵的物理行数；若有列间 padding，或矩阵是更大矩阵的子矩阵，则 `ld` 可以更大。

以一个 $2 \times 4$ 的矩阵为例：

$$
A =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8
\end{bmatrix}
$$

当 `lda = 2` 时，其 column-major buffer 是 `[1, 5, 2, 6, 3, 7, 4, 8]`。例如 $A_{1,2}=7$ 的偏移为 $1 + 2\cdot2=5$。

若 `lda = 3`，每一列都额外留一个 padding 元素，buffer 则是 `[1, 5, -, 2, 6, -, 3, 7, -, 4, 8, -]`。此时同一个 $A_{1,2}$ 的偏移为 $1 + 2\cdot3=7$。`-` 不属于逻辑矩阵，GEMM 不会读取它。

对于 `cublasSgemm` 的 column-major 契约，leading dimension 的合法下界可直接由各个**原始矩阵**的物理行数得到：

| 参数 | `trans` 为 `N` 时，原始矩阵形状与下界 | `trans` 为 `T` / `C` 时，原始矩阵形状与下界 |
| --- | --- | --- |
| `A`, `lda` | $A$ 为 $M \times K$；`lda >= max(1, M)` | $A$ 为 $K \times M$；`lda >= max(1, K)` |
| `B`, `ldb` | $B$ 为 $K \times N$；`ldb >= max(1, K)` | $B$ 为 $N \times K$；`ldb >= max(1, N)` |
| `C`, `ldc` | $C$ 为 $M \times N$；`ldc >= max(1, M)` | 不受 `transa` / `transb` 影响 |

最后一行很关键：`C` 没有转置选项，因此 `ldc` 总是其 column-major 物理行距，最小就是 `m`。`ld*` 传小通常会得到 `CUBLAS_STATUS_INVALID_VALUE`；传大不是“改变运算尺寸”，而是在相邻列之间保留间隔。

### 一个从读入到写回的完整 `NN` 例子

下面故意选择非紧凑的 `lda = 3`、`ldb = 5`、`ldc = 4`，以便区分**逻辑形状**和**物理列距**。调用为：

```cpp
// A: 2 x 4，B: 4 x 3，C: 2 x 3，全部按 column-major 解释。
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            2, 3, 4,
            &alpha,
            device_a, 3,
            device_b, 5,
            &beta,
            device_c, 4);
```

设 `alpha = 2`、`beta = 0.5`，逻辑矩阵为：

$$
A =
\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix},\quad
B =
\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\\10&11&12\end{bmatrix},\quad
C_{\text{old}} =
\begin{bmatrix}100&200&300\\400&500&600\end{bmatrix}
$$

三段 device buffer 的物理内容应分别按列排列。`-1` 只是人为放入 padding 的哨兵值：

| buffer | column-major 的线性内容 | 逻辑元素如何定位 |
| --- | --- | --- |
| `A`, `lda = 3` | `[1, 5, -1, 2, 6, -1, 3, 7, -1, 4, 8, -1]` | `A[i,j] = A[i + j * 3]` |
| `B`, `ldb = 5` | `[1, 4, 7, 10, -1, 2, 5, 8, 11, -1, 3, 6, 9, 12, -1]` | `B[i,j] = B[i + j * 5]` |
| `C`, `ldc = 4` | `[100, 400, -1, -1, 200, 500, -1, -1, 300, 600, -1, -1]` | `C[i,j] = C[i + j * 4]` |

cuBLAS 读取这三段内存后，做的不是“按 buffer 顺序相乘”，而是严格按逻辑下标计算：

$$
C_{i,j} \leftarrow 2\cdot\sum_{r=0}^{3} A_{i,r}B_{r,j}+0.5\cdot C_{i,j}
$$

例如要写 $C_{1,2}$，库会读取：

$$
\begin{aligned}
A_{1,0}, A_{1,1}, A_{1,2}, A_{1,3} &= [5, 6, 7, 8] \\
B_{0,2}, B_{1,2}, B_{2,2}, B_{3,2} &= [3, 6, 9, 12] \\
C_{\text{old},1,2} &= 600
\end{aligned}
$$

于是：

$$
C_{1,2}=2\cdot(5\cdot3+6\cdot6+7\cdot9+8\cdot12)+0.5\cdot600=720
$$

它会写到 `device_c[1 + 2 * 4]`，即物理偏移 9；不会碰同一列的 padding 位置 10、11。整个逻辑结果为：

$$
C_{\text{new}}=
\begin{bmatrix}190&260&330\\516&618&720\end{bmatrix}
$$

因此 `C` 的完整物理 buffer 变成 `[190, 516, -1, -1, 260, 618, -1, -1, 330, 720, -1, -1]`。这也解释了 `beta` 的实际意义：当 `beta != 0` 时，`C` 既是**输入**也是**输出**；只有当 `beta = 0` 时，旧的逻辑 `C` 元素在数学上不参与结果。

### `TN` 例子：转置改变的是原始 A 的形状

仍希望计算刚才的 $A_{2\times4}B_{4\times3}$，但现在把 `A` 以它的转置存入内存：

$$
A_{\text{stored}}=
\begin{bmatrix}
1&5\\2&6\\3&7\\4&8
\end{bmatrix}_{4\times2}
$$

此时调用：

```cpp
cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            2, 3, 4,
            &alpha,
            device_a_stored, 4,
            device_b, 4,
            &beta,
            device_c, 2);
```

这里的 `TN` 表示 $\operatorname{op}(A_{\text{stored}})=A_{\text{stored}}^T$、$\operatorname{op}(B)=B$。这组参数的逻辑目标仍是：

$$
\operatorname{op}(A_{\text{stored}})\in\mathbb{R}^{2\times4},\quad
\operatorname{op}(B)\in\mathbb{R}^{4\times3},\quad
C\in\mathbb{R}^{2\times3}
$$

但原始 $A_{\text{stored}}$ 在转置**之前**必须是 $K\times M=4\times2$，其紧凑 column-major buffer 是 `[1, 2, 3, 4, 5, 6, 7, 8]`。故 `lda = 4`，而不是看见转置结果有 2 行就写 `lda = 2`。

具体看同一个逻辑元素 $\operatorname{op}(A_{\text{stored}})_{1,r}$：由于 `transa = T`，它会读原始 $A_{\text{stored},r,1}$，偏移为 `r + 1 * lda`。当 `r = 0, 1, 2, 3` 时，读取偏移 `[4, 5, 6, 7]`，正好得到 `[5, 6, 7, 8]`。随后与未转置的 $B_{r,2}$ 相乘，`B` 的偏移为 `r + 2 * ldb`。因此 `T` 是在**每一次读取的坐标选择阶段**就生效；`ld` 则在坐标已经换成原始矩阵坐标后，负责把坐标落到实际地址。

对同一个 `m/n/k`，`NN` 的原始 A 是 $M\times K$，而 `TN` 的原始 A 是 $K\times M$；这正是许多 `lda` 错误的来源。

### 保持 A、B、C 都是 row-major：交换操作数和 M/N

传统 cuBLAS 没有 `layout = row_major` 参数。要正确地把 row-major GEMM 映射给它，不能先背“交换 A/B 和 M/N”，而要先看清：**一段连续内存既可以被解释为 row-major 矩阵，也可以被解释为另一个形状的 column-major 矩阵。**

下面先只讨论最常见、也最值得先理解的目标：A、B、C 都是未转置的 row-major 矩阵。

$$
\underbrace{C_{\mathrm r}}_{M\times N}
= \alpha\underbrace{A_{\mathrm r}}_{M\times K}
\underbrace{B_{\mathrm r}}_{K\times N}
+ \beta\underbrace{C_{\mathrm r}}_{M\times N}
$$

下标 $\mathrm r$ 只表示“我们希望按 row-major 理解它”，不是另一份数据。

#### 同一段 buffer 的两种视图

取 $M=2$、$N=3$、$K=4$。应用按 row-major 生成：

$$
A_{\mathrm r}=
\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}_{2\times4},\qquad
B_{\mathrm r}=
\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\\10&11&12\end{bmatrix}_{4\times3}
$$

它们在内存中的线性顺序分别是：

```cpp
// row-major buffer，不存在任何预先转置。
device_a = [1, 2, 3, 4, 5, 6, 7, 8];
device_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
```

如果 cuBLAS 把 `device_a` 解释为一个 **$K\times M=4\times2$ 的 column-major** 矩阵，它看到的是：

$$
A_{\mathrm c}=
\begin{bmatrix}1&5\\2&6\\3&7\\4&8\end{bmatrix}_{4\times2}
=A_{\mathrm r}^T
$$

这是因为 column-major 的 $4\times2$ 矩阵逐列展开正好也是 `[1, 2, 3, 4, 5, 6, 7, 8]`。数据本身没有动；$A_{\mathrm c}=A_{\mathrm r}^T$ 只是**对同一段 bytes 的数学视图**。

同理，若 cuBLAS 把 `device_b` 解释为一个 **$N\times K=3\times4$ 的 column-major** 矩阵，它看到的是：

$$
B_{\mathrm c}=
\begin{bmatrix}1&4&7&10\\2&5&8&11\\3&6&9&12\end{bmatrix}_{3\times4}
=B_{\mathrm r}^T
$$

输出 buffer 也一样。应用想把它当作 row-major 的 $C_{\mathrm r}\in\mathbb{R}^{M\times N}$，而 cuBLAS 会把同一段地址当作 column-major 的 $C_{\mathrm c}\in\mathbb{R}^{N\times M}$；两者满足 $C_{\mathrm c}=C_{\mathrm r}^T$。

| 原始 row-major 视图 | 形状 | cuBLAS 应采用的 column-major 视图 | 形状 | 紧凑内存跨度 |
| --- | --- | --- | --- | --- |
| $A_{\mathrm r}$ | $M\times K$ | $A_{\mathrm c}=A_{\mathrm r}^T$ | $K\times M$ | `stride_a = K` |
| $B_{\mathrm r}$ | $K\times N$ | $B_{\mathrm c}=B_{\mathrm r}^T$ | $N\times K$ | `stride_b = N` |
| $C_{\mathrm r}$ | $M\times N$ | $C_{\mathrm c}=C_{\mathrm r}^T$ | $N\times M$ | `stride_c = N` |

这里 `stride_a/b/c` 是应用 row-major 相邻两行的元素距离。由于上述 column-major 视图恰好把它们当作相邻两列的距离，它们也就是传给 cuBLAS 的 `lda/ldb/ldc`。

#### 真正要计算的是 `BᵀAᵀ`，不是 `AᵀBᵀ`

应用目标是 $A_{\mathrm r}B_{\mathrm r}$，形状为：

$$
(M\times K)(K\times N)=M\times N
$$

将等式两边转置时，乘法顺序必须反过来：

$$
(A_{\mathrm r}B_{\mathrm r})^T
=B_{\mathrm r}^TA_{\mathrm r}^T
$$

其维度完全吻合：

$$
\underbrace{B_{\mathrm r}^T}_{N\times K}
\underbrace{A_{\mathrm r}^T}_{K\times M}
=\underbrace{C_{\mathrm r}^T}_{N\times M}
$$

因此传给 cuBLAS 的是 $B_{\mathrm c}A_{\mathrm c}=C_{\mathrm c}$。**不是 $A^TB^T$。** 后者的维度是 $(K\times M)(N\times K)$，除非碰巧 $M=N$，否则中间维度 $M$ 与 $N$ 根本不匹配。

这时参数就不再是记忆题，而是从上式直接读出来：cuBLAS 的 `m/n/k` 必须是 $N/M/K$，第一个操作数必须是 `B`，第二个必须是 `A`。

```cpp
// 应用语义（所有矩阵均为 row-major）：
// C_r[M, N] = alpha * A_r[M, K] * B_r[K, N] + beta * C_r[M, N]
//
// cuBLAS 语义（同一批 buffer 被 reinterpret 为 column-major）：
// C_c[N, M] = alpha * B_c[N, K] * A_c[K, M] + beta * C_c[N, M]
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            device_b, N,  // B_c: N x K，ld = 应用中 B 的 row stride
            device_a, K,  // A_c: K x M，ld = 应用中 A 的 row stride
            &beta,
            device_c, N); // C_c: N x M，ld = 应用中 C 的 row stride
```

#### 用一个输出元素核对地址

仍看应用要得到的 row-major 元素 $C_{\mathrm r}[i,j]$。它的物理偏移为 `i * N + j`。在 cuBLAS 的 column-major 视图中，它正是 $C_{\mathrm c}[j,i]$，偏移为：

$$
j+i\cdot\underbrace{N}_{ldc}=i\cdot N+j
$$

两者是同一个地址。对这个输出元素，cuBLAS 的点积读法为：

$$
C_{\mathrm c}[j,i] = \sum_{r=0}^{K-1} B_{\mathrm c}[j,r]A_{\mathrm c}[r,i]
$$

将两个 column-major 地址还原到原应用 buffer：

$$
\begin{aligned}
B_{\mathrm c}[j,r] &: j+r\cdot\underbrace{N}_{ldb} = r\cdot N+j = B_{\mathrm r}[r,j] \\
A_{\mathrm c}[r,i] &: r+i\cdot\underbrace{K}_{lda} = i\cdot K+r = A_{\mathrm r}[i,r]
\end{aligned}
$$

所以 cuBLAS 实际从同一段内存读出的值，逐项就是应用想要的 $A_{\mathrm r}[i,r]B_{\mathrm r}[r,j]$；只是它在自己的坐标系统中把这次乘法命名为 $B_{\mathrm c}A_{\mathrm c}$。

若 A、B、C 有 row-major padding，只需把示例中的 `K/N/N` 换成真实的 `stride_a/stride_b/stride_c`。它们仍是 cuBLAS 调用中的 `lda/ldb/ldc`，而 `m/n/k` 仍然是 `N/M/K`。

#### 原始 row-major 计算也带转置时

若目标写成 $C_{\mathrm r}=\operatorname{op}(A_{\mathrm r})\operatorname{op}(B_{\mathrm r})$，先对**整个乘法结果**转置：

$$
C_{\mathrm r}^T=\operatorname{op}(B_{\mathrm r})^T\operatorname{op}(A_{\mathrm r})^T
$$

原则依旧：交换 A/B、交换 `m/n`，并将原来的 `transb` 作为 cuBLAS 的第一个转置选项、原来的 `transa` 作为第二个。最容易出错的是原始 buffer 的 row stride；它仍由 A、B、C 转置**前**各自的 row-major 物理形状决定。初学时建议先把上面的 `NN` 映射和地址等式写清楚，再处理带转置的组合。

这套恒等式适合理解和兼容传统 cuBLAS。后续讨论 `cuBLASLt` 时，可以看到它通过矩阵 layout descriptor 直接表达 row-major，从接口层面消除这层“转置视图”的推导。

## cuBLASLt：用描述符直接表达矩阵布局

上一节的 row-major 映射在数学上没有问题，但调用者必须把自己的 $A\times B$ 改写成 cuBLAS 眼中的 $B^T\times A^T$。`cuBLASLt`（轻量、可配置的 GEMM API）换了一种分工：**指针只表示一段 device memory；矩阵的类型、形状、步幅和存储顺序全部由描述符显式声明。**

因此，row-major 不再是“把数据伪装成 column-major 转置视图”的技巧，而是矩阵 layout 的一个属性。官方 API 中 `CUBLASLT_MATRIX_LAYOUT_ORDER` 的默认值仍为 `CUBLASLT_ORDER_COL`；只有主动设置成 `CUBLASLT_ORDER_ROW` 后，库才按行优先解释该矩阵。[cuBLASLt 的矩阵 layout 属性](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutattribute-t) 与 [order 定义](https://docs.nvidia.com/cuda/cublas/index.html#cublasltorder-t) 都对此作了规定。

### 一个 GEMM 被拆成四类描述

传统 `cublasSgemm` 把形状、转置和 leading dimension 压在一长串位置参数中。`cublasLtMatmul` 把它们拆开，让每一类信息有明确归属：

| 描述对象 | 负责回答的问题 | 典型内容 |
| --- | --- | --- |
| `cublasLtMatmulDesc_t` | **做什么运算**？ | compute type、scale type、`transa` / `transb`、epilogue、`alpha` / `beta` 的 pointer mode |
| `cublasLtMatrixLayout_t` | **每个矩阵在内存中是什么样子**？ | 元素类型、`rows`、`cols`、`ld`、`order`、batch count / batch stride |
| `cublasLtMatmulPreference_t` | **允许算法使用多少资源**？ | 例如最大 workspace 字节数 |
| `cublasLtMatmulAlgo_t` | **最终选中了什么实现**？ | 由 heuristic 返回的候选算法及其 workspace 要求 |

其中本节的重点是 `cublasLtMatrixLayout_t`。它并不持有矩阵数据，也不分配 device memory；它只是一个可复用的“这根指针应如何被解释”的说明书。描述符创建接口是：

```cpp
cublasStatus_t cublasLtMatrixLayoutCreate(
    cublasLtMatrixLayout_t* layout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);
```

`type`、`rows`、`cols`、`ld` 给出初始属性；其余属性通过 `cublasLtMatrixLayoutSetAttribute()` 设置。虽然 `rows/cols/ld` 已经在 create 中出现，但默认 `order` 是 column-major。实践中应在创建后立刻设置目标 order，并始终按**最终的 order**理解 `ld`。

### `rows`、`cols`、`ld` 与 `order` 一起定义地址函数

对普通的实数 row/column-major 矩阵，layout descriptor 实际上定义了逻辑坐标 $(i,j)$ 到元素偏移的函数：

| `order` | 元素地址 | `ld` 的含义 | 紧凑存储时 |
| --- | --- | --- | --- |
| `CUBLASLT_ORDER_COL` | `offset(i, j) = i + j * ld` | 相邻**列**起点的元素距离 | `ld = rows` |
| `CUBLASLT_ORDER_ROW` | `offset(i, j) = i * ld + j` | 相邻**行**起点的元素距离 | `ld = cols` |

所以 `ld` 不是独立于 layout 的固定概念。相同的 `ld = 8`：在 column-major 中表示“下一列从 8 个元素之后开始”，在 row-major 中表示“下一行从 8 个元素之后开始”。两种 order 都要求 `ld` 足够大，避免不同行或列的元素重叠；普通紧凑矩阵下分别要求 `ld >= rows` 与 `ld >= cols`。

这正是 cuBLASLt 比传统 cuBLAS 直观的地方：`rows = M, cols = K, order = ROW` 就直接表示一个 row-major 的 $M\times K$ 矩阵；不需要把它在脑中改名成 $K\times M$ 的 column-major 矩阵。

### 用描述符直接表达 row-major 的 `A[M,K] * B[K,N]`

继续使用 $M=2$、$N=3$、$K=4$。应用希望计算：

$$
D_{2\times3}=\alpha A_{2\times4}B_{4\times3}+\beta C_{2\times3}
$$

这里特意把输出叫做 $D$：cuBLASLt 的通用数学模型是

$$
D=\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C
$$

`C` 是累加项输入，`D` 是输出；两者可以是同一块内存（in-place），也可以是两块不同内存（out-of-place）。若没有额外的残差输入，最常见的做法就是令 `device_c == device_d` 且 `c_layout == d_layout`。

对于三个紧凑 row-major buffer，布局描述应当是：

| 矩阵 | `rows × cols` | `ld` | `order` | 线性 buffer 的解释 |
| --- | --- | --- | --- | --- |
| `A` | `M × K = 2 × 4` | `K = 4` | `CUBLASLT_ORDER_ROW` | `A[i,j] = device_a[i * 4 + j]` |
| `B` | `K × N = 4 × 3` | `N = 3` | `CUBLASLT_ORDER_ROW` | `B[i,j] = device_b[i * 3 + j]` |
| `C` | `M × N = 2 × 3` | `N = 3` | `CUBLASLT_ORDER_ROW` | `C[i,j] = device_c[i * 3 + j]` |
| `D` | `M × N = 2 × 3` | `N = 3` | `CUBLASLT_ORDER_ROW` | `D[i,j] = device_d[i * 3 + j]` |

注意这张表中的形状就是应用中的真实形状：A 仍是 $2\times4$，B 仍是 $4\times3$，输出仍是 $2\times3$。没有交换 A/B，也没有交换 M/N。

下面是只保留“描述矩阵”这一核心动作的代码摘要：

```cpp
cublasLtMatrixLayout_t a_layout = nullptr;
cublasLtMatrixLayout_t b_layout = nullptr;
cublasLtMatrixLayout_t c_layout = nullptr;
cublasLtMatrixLayout_t d_layout = nullptr;

cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F, M, K, K);
cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F, K, N, N);
cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, M, N, N);
cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F, M, N, N);

const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
for (cublasLtMatrixLayout_t layout : {a_layout, b_layout, c_layout, d_layout}) {
    cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                     &order, sizeof(order));
}
```

`CUDA_R_32F` 表示每个 matrix element 在内存中是一个 FP32。它与 `CUBLAS_COMPUTE_32F` 是不同层次的概念：前者属于 matrix layout，说明怎样读写 A/B/C/D；后者属于 matmul descriptor，说明点积和缩放在什么计算精度下进行。混合精度 GEMM 中，这两个选择通常不同，例如 A/B 可为 FP16 或 BF16、累加为 FP32、D 再写回 FP16。

### 由 operation descriptor 连接布局和数学运算

layout 只说明每一块内存中的“矩阵长什么样”，并没有说明 A、B 如何相乘。该信息属于 `cublasLtMatmulDesc_t`：

```cpp
cublasLtMatmulDesc_t operation_desc = nullptr;
cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

const cublasOperation_t no_transpose = CUBLAS_OP_N;
cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                &no_transpose, sizeof(no_transpose));
cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                &no_transpose, sizeof(no_transpose));
```

`TRANSA` / `TRANSB` 的语义仍和上一节的 `op(A)` / `op(B)` 一样，但 shape 检查基于各自 layout 的 `rows/cols`。在本例中两个选项都是 `N`，所以直接得到：

$$
\operatorname{op}(A):2\times4,\qquad
\operatorname{op}(B):4\times3,\qquad
D:2\times3
$$

若将 `TRANSA` 设为 `T`，则 A 的 layout 仍描述**原始** A；要让转置后的 `op(A)` 成为 $M\times K$，原始 A layout 必须是 $K\times M$。这条形状规则和传统 cuBLAS 相同；不同的是它不再与“所有矩阵必为 column-major”绑定。

最终调用会同时接收四个指针和四个 layout：

```cpp
cublasLtMatmul(handle, operation_desc,
               &alpha,
               device_a, a_layout,
               device_b, b_layout,
               &beta,
               device_c, c_layout,
               device_d, d_layout,
               &algorithm, workspace, workspace_bytes, stream);
```

这里不再单独传 `m/n/k/lda/ldb/ldc`：cuBLASLt 从 `a_layout`、`b_layout`、`c_layout`、`d_layout` 与 `operation_desc` 联合推导和校验它们。对于 out-of-place 情况，C 与 D 必须有相同的数据类型、行列数、batch size 和 memory order，但 `ld` 可以不同；这是允许输入带 padding、输出采用不同 row stride 的原因。[`cublasLtMatmul` 的 C/D 约束](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul) 有明确说明。

### batch 也是 layout 的一部分

传统 strided-batched GEMM 将 batch 信息塞进另一套参数。cuBLASLt 将它归入每个 matrix layout：

| 属性 | 含义 | 对连续的相同形状 batch |
| --- | --- | --- |
| `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` | 此矩阵参与的 batch 数 | 设为 `batch_count` |
| `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET` | 下一个 batch 矩阵的起始偏移，单位为**元素** | 紧凑矩阵通常为 `rows * ld` |
| `CUBLASLT_MATRIX_LAYOUT_BATCH_MODE` | batch 的寻址方式 | 默认是 `CUBLASLT_BATCH_MODE_STRIDED` |

例如有 16 个、每个都是 row-major $2\times4$ 且 `ld=4` 的 A 矩阵时，`batch_count = 16`，`strided_batch_offset = 2 * 4 = 8`。第 `b` 个矩阵的逻辑元素 $A_b[i,j]$ 位于 `device_a[b * 8 + i * 4 + j]`。A、B、C、D 可以各自有不同的 batch offset；它们的 layout 分别描述自己的地址规则。

### 不止 row-major / column-major，但不要把特殊 order 当作通用开关

`CUBLASLT_ORDER_COL` 与 `CUBLASLT_ORDER_ROW` 是普通二维矩阵最常用的 order。cuBLASLt 还定义了如 `CUBLASLT_ORDER_COL32`、`CUBLASLT_ORDER_COL4_4R2_8C`、`CUBLASLT_ORDER_COL32_2R_4R4` 的交错 / 分块 layout，用于某些整数 Tensor Core 路径。它们的 `ld` 不再是简单的行或列步幅，而会涉及固定宽度 tile 的地址规则。

这些特殊 order 不是“设置后自动更快”的选项：数据本身必须已经按对应 tile 格式打包，且可用性受数据类型、GPU 架构、转置选项和算法限制共同约束。对于常规 FP16、BF16、TF32、FP32 GEMM，应先从 `ROW` 或 `COL` 开始，让 heuristic 在符合 layout 的算法中选择实现；只有需要对接已打包的 low-precision 数据时，再研究专用 order。

### 这一节的记忆点

- `cublasLtMatrixLayout_t` 是矩阵的**内存合同**：`type + rows + cols + ld + order`，外加可选的 batch 描述。
- `order = ROW` 时，`ld` 是 row stride；`order = COL` 时，`ld` 是 column stride。
- `rows/cols` 永远描述原始矩阵；是否转置由独立的 matmul descriptor 决定。
- 对 row-major $A[M,K]B[K,N]$，直接把 A/B/C/D 分别描述为 $M\times K$、$K\times N$、$M\times N$、$M\times N$ 即可；无需传统 cuBLAS 的 A/B 和 M/N 交换。


## Ampere `mma.sync`：从数学 GEMM 到 `.row.col`

理解这条链时，顺序必须固定为：**数学 GEMM → shared-memory 地址 → CuTe view → LDSM → rmem fragment → MMA**。`_TN`、LDSM 的 N/T、`.row/.col` 分别处在不同层，不能互相替代。

> **`.row.col` 的一句话总结：它固定的是 `(lane, register 内标量) →` 逻辑 A/B 坐标的合同，不是一个线性寄存器 buffer 的存储顺序。该合同中，A 的局部 FP16 pair 固定 $m$、沿 K 相邻（row direction）；B 的局部 FP16 pair 固定 $n$、沿 K 相邻（column direction）。LDSM 的职责是把 shared memory 中的元素放入这些固定 slot。**

以一个 warp 的指令为例：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

它计算 $M=16,N=8,K=16$：

$$
D(m,n)=\sum_{k=0}^{15} A(m,k)B(k,n)+C(m,n).
$$

这里必须先按**数学 shape**定义行/列优先；“哪一维连续”会随矩阵 shape 改变：

| 数学矩阵 | row-major（行优先） | column-major（列优先） |
| --- | --- | --- |
| $A[M,K]$ | K 连续，$\operatorname{addr}=mK+k$ | M 连续，$\operatorname{addr}=kM+m$ |
| $B[K,N]$ | N 连续，$\operatorname{addr}=kN+n$ | K 连续，$\operatorname{addr}=nK+k$ |
| $C[M,N]$ | N 连续，$\operatorname{addr}=mN+n$ | M 连续，$\operatorname{addr}=nM+m$ |

例如，“A 的 K 连续”和“B 的 N 连续”都叫 row-major；不能因为两者连续的维度不同，就把其中一个误叫成列优先。

### 先确定 shared memory 中真正的 A/B

本章先固定数学 $A[M,K]$、$B[K,N]$ 都 row-major：

$$
\operatorname{addr}_A(m,k)=mK+k,\qquad
\operatorname{addr}_B(k,n)=kN+n.
$$

因此 A 的 K 连续，B 的 N 连续。B 的 row-major 是针对数学坐标 $(k,n)$，不是稍后 CuTe 使用的 `(n,k)`。

### 用 CuTe view 访问同一段 B buffer

CuTe 的 SM80 MMA 路径以 $s_A(m,k)=A(m,k)$、$s_B(n,k)=B(k,n)$ 表达输入。B 的 `(N,K)` 只是数学 $B[K,N]$ 的 view，不是转置或搬运。

```cpp
// A[M,K]，K 连续。
auto sA = cute::make_tensor(
    cute::make_smem_ptr(shared_a),
    cute::make_layout(cute::make_shape(M, K), cute::LayoutRight{}));
// stride: (K, 1)

// B[K,N]，N 连续；CuTe 按 sB(n,k) == B(k,n) 访问。
auto sB = cute::make_tensor(
    cute::make_smem_ptr(shared_b),
    cute::make_layout(cute::make_shape(N, K), cute::LayoutLeft{}));
// stride: (1, N)，addr(sB(n,k)) = n + k*N。
```

所以

$$
\operatorname{addr}_{s_B}(n,k)=n+kN=kN+n=\operatorname{addr}_B(k,n).
$$

**数学 row-major 的 $B[K,N]$，在 CuTe `(N,K)` view 中是 `LayoutLeft`。** 不先确定坐标轴，谈 B 的行/列优先没有意义。

### `.row.col` 先定义 rmem slot，不定义线性寄存器 buffer

对 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`，PTX 规定了每一个 `(lane, register 内标量)` 对应 A/B 哪个逻辑元素。官方的 fragment 图正是在画这张映射，而不是把 32 个线程的寄存器拼成一段普通的 row-major 或 column-major 数组。

令：

```text
g = lane_id >> 2       // 0 ... 7
t = lane_id & 3        // 0 ... 3
```

FP16 A operand 有 4 个 `.f16x2`，即标量 `a0 ... a7`；B operand 有 2 个 `.f16x2`，即标量 `b0 ... b3`。其坐标关系可归纳为：

| operand | 标量的逻辑坐标 | 一个 FP16 pair 的方向 |
| --- | --- | --- |
| A | $a_i$ 的 $m$ 是 $g$ 或 $g+8$；$k=2t+(i\mathbin{\&}1)$，后半组再加 8 | `a0,a1`、`a2,a3` 等固定 $m$，沿 K 相邻 |
| B | $b_i$ 的 $k=2t+(i\mathbin{\&}1)$，后半组再加 8；$n=g$ | `b0,b1`、`b2,b3` 固定 $n$，沿 K 相邻 |

例如 lane 0 的前几个标量为：

| lane 0 的 slot | 对应逻辑元素 |
| --- | --- |
| `a0, a1` | $A(0,0), A(0,1)$ |
| `a2, a3` | $A(8,0), A(8,1)$ |
| `b0, b1` | $B(0,0), B(1,0)$ |
| `b2, b3` | $B(8,0), B(9,0)$ |

所以 `.row.col` 的 `row/col` 可以从**局部 FP16 pair 的逻辑方向**理解：

- A 的 pair 固定 $m$、K 变化；对 $A[M,K]$ 来说这是 row direction。
- B 的 pair 固定 $n$、K 变化；对 $B[K,N]$ 来说这是 column direction。

但这不等于“先按 `T0,T1,...,T31`，再按 register 编号把标量串起来，就是一个普通 row-major / column-major buffer”。跨 lane 的顺序由上述非平凡映射决定；正确的判断标准永远是图或坐标公式。PTX 的完整 fragment 图与公式见 [`mma.m16n8k16` floating-point fragment](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-16816-float)。

### LDSM 的 N/T：改变 shared-to-register 的协作映射

`ldmatrix`（CuTe 中常简称 LDSM）由整个 warp 从 shared memory 协作装入一个或多个 $8\times8$ half tile。官方称 `.trans` 为“以 column-major format 加载”；从数据流角度，它改变的是：**一个 shared 元素最终进入哪个 lane 的哪个 register slot。** 它不是在 shared memory 中生成一份转置后的 buffer。

先忽略 swizzle、padding 和 `x1/x2/x4`，把一个物理 $8\times8$ source tile 记为 $S(r,c)$，把 load 后应送给消费者的逻辑坐标网格记为 $F$：

$$
\text{LDSM N}:\quad F(r,c)=S(r,c),
$$

$$
\text{LDSM T}:\quad F(c,r)=S(r,c).
$$

第二式就是“转置读取”的精确含义：source 中的 $S(r,c)$ 没有移动，但它被送往 destination 的 $(c,r)$ slot。实际硬件还要求 warp 中特定线程提供各行的起始地址；因此不能只把 PTX 指令的 N 改成 T、却保持原来的 source layout 和地址映射不变。CuTe 的 source view 加上 `Copy_Atom` 正是同时构造这两部分映射。

对 B，这两条等价路径特别直观。令最终 B fragment 的坐标 view 为：

$$
F(n,k)=B(k,n).
$$

| 数学 B 的 SMEM 布局 | source tile 的坐标解释 | LDSM 后的 F | 为什么正确 |
| --- | --- | --- | --- |
| column-major，K 连续 | $S(n,k)=B(k,n)$ | N：$F(n,k)=S(n,k)$ | source 已按 B fragment 所需的 K 方向排列 |
| row-major，N 连续 | $S(k,n)=B(k,n)$ | T：$F(n,k)=S(k,n)$ | T 把 source 的 $(k,n)$ 送到 B slot 的 $(n,k)$ |

两行最终都满足 $F(n,k)=B(k,n)$，所以进入 MMA 的是同一套 B rmem slot，指令仍计算 $A(m,k)B(k,n)$，而不是 $A\times B^T$。

把这件事与 A/B 的 local pair 合在一起，方向上的选择为：

| 数学 operand | SMEM 连续方向 | MMA 固定要求的局部 pair | 方向上的处理 |
| --- | --- | --- | --- |
| $A[M,K]$ | K 连续 | 固定 $m$，K 相邻 | 直接 LDSM N |
| $A[M,K]$ | M 连续 | 固定 $m$，K 相邻 | 写入 SMEM 时改为 K 连续，或 LDSM T |
| $B[K,N]$ | K 连续 | 固定 $n$，K 相邻 | 直接 LDSM N |
| $B[K,N]$ | N 连续 | 固定 $n$，K 相邻 | 写入 SMEM 时改为 K 连续，或 LDSM T |

这里的“直接”只指**局部 pair 的方向不需要交换**；实际应选 `x1/x2/x4`、具体 Copy Atom 以及是否能直接发射 LDSM，仍由 microtile 形状、swizzle、对齐和 CuTe layout 共同决定。

本节固定的 row-major A/B 恰好落在表的第一、四行：

| shared-memory 数学对象 | CuTe view | Copy Atom | 实际 LDSM |
| --- | --- | --- | --- |
| $A[M,K]$，K 连续 | `(M,K)` `LayoutRight` | `SM75_U32x4_LDSM_N` | `ldmatrix.x4` |
| $B[K,N]$，N 连续 | `(N,K)` `LayoutLeft` | `SM75_U16x4_LDSM_T` | `ldmatrix.x2.trans` |

A 每 lane 有 4 个 32-bit register，所以使用 `x4`。B 每 lane 有 2 个 32-bit register；`SM75_U16x4_LDSM_T` 实际发射 `ldmatrix.x2.trans`，产生这两个 B register。B 的 $K=16,N=8$ tile 沿 K 切成两个 $8\times8$ 子块，因此是 `x2`。

> **`.row.col` 不规定 shared memory 必须采用哪种线性布局；它固定 A/B 的 rmem slot。数学 row-major B 的 N 连续路径可用 LDSM T，数学 column-major B 的 K 连续路径可用 LDSM N；两条路径最终填入相同的 $F(n,k)=B(k,n)$ slot。**

### `thread_mma` 只划分 fragment；LDSM 才读取 SMEM

```cpp
auto tiled_mma = cute::make_tiled_mma(
    cute::SM80_16x8x16_F32F16F16F32_TN{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{});
auto thread_mma = tiled_mma.get_slice(threadIdx.x);
auto rA = thread_mma.partition_fragment_A(sA);
auto rB = thread_mma.partition_fragment_B(sB);
```

这两行只描述当前 lane 的 rmem slot，尚未读取 shared memory。真正将 source view 填入 rmem 的是：

```cpp
auto copy_a = cute::make_tiled_copy_A(
    cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::half_t>{}, tiled_mma);
auto copy_b = cute::make_tiled_copy_B(
    cute::Copy_Atom<cute::SM75_U16x4_LDSM_T, cute::half_t>{}, tiled_mma);

auto ca = copy_a.get_slice(threadIdx.x);
auto cb = copy_b.get_slice(threadIdx.x);
cute::copy(cute::SM75_U32x4_LDSM_N{}, ca.partition_S(sA), ca.retile_D(rA));
cute::copy(cute::SM75_U16x4_LDSM_T{}, cb.partition_S(sB), cb.retile_D(rB));
```

`make_tiled_copy_A/B` 组合 source 坐标、shared stride、LDSM N/T 和 Atom 的目标 register slot。若把上例 B 的 `LayoutLeft` 错配为非转置 `SM75_U32x2_LDSM_N`，CuTe 会在编译期拒绝：这是 LDSM 与 source layout 不匹配，不是 `.row.col` 不支持 row-major B。

### `.row.col` 只消费最终的 rmem contract

```cpp
auto rC = thread_mma.make_fragment_C(/* C(M,N) */);
cute::clear(rC);
cute::gemm(tiled_mma, rA, rB, rC);
```

`.row` 表示 `rA` 的跨 lane/register 排列符合 A operand 合同；`.col` 表示 `rB` 的排列符合 B operand 合同。它们不描述 shared buffer 的线性地址。因此 shared B 即使是 N 连续，指令仍计算：

$$
D(m,n)=\sum_k A(m,k)B(k,n)+C(m,n).
$$

实际单 warp 验证取 $A(m,k)=m+1$、$B(k,n)=n+1$，累加四个 $K=16$ 子块，得到 $D(m,n)=64(m+1)(n+1)$。该数学 row-major A/B 路径已数值通过；SASS 中可见 A 的 `LDSM.16.M88.4`、B 的 `LDSM.16.MT88.2` 与 `HMMA.16816.F32`。

### `_TN` 为什么仍对应 `.row.col`

`SM80_16x8x16_F32F16F16F32_TN` 不表示本 kernel 的物理 A 必须转置、物理 B 必须不转置。`_TN` 是 CUTLASS/CuTe 沿用 BLAS 默认 column-major 基准给最终 rmem operand format 起的名字：`N` 不交换坐标轴，仍是 column-oriented view；`T` 交换坐标轴，成为 row-oriented view。因此 $T\rightarrow\texttt{row}$、$N\rightarrow\texttt{col}$，`_TN` 对应 `.row.col`。

SM70 的 `m8n8k4` 有四种形式：

| Atom 后缀 | A 的 PTX layout | B 的 PTX layout |
| --- | --- | --- |
| `_NN` | `.col` | `.col` |
| `_NT` | `.col` | `.row` |
| `_TN` | `.row` | `.col` |
| `_TT` | `.row` | `.row` |

SM80 的 `m16n8k16` 语法固定为 `.row.col`，没有另外三种 PTX 变体，所以 CuTe 只有 `_TN` Atom。它描述 rmem 的终点；数学 row-major A/B 是送达这个终点的一条 load path。

### 排查 layout 的固定顺序

1. 目标是否是 $D(m,n)=\sum_k A(m,k)B(k,n)+C(m,n)$？
2. 对 $A[M,K]$、$B[K,N]$，哪一维连续？由此各自是 row-major 还是 column-major？
3. `shared_a` 中 $A(m,k)$、`shared_b` 中 $B(k,n)$ 的地址分别是什么？
4. `sB(n,k)` 是否仍访问同一个 $B(k,n)$？
5. `.row.col` 需要的 A row / B col rmem 方向，是否已与各自 SMEM 微块方向一致？
6. 不一致时，是在写入 SMEM 时转换，还是由 LDSM T 转换？
7. `make_tiled_copy_A/B` 是否与 `thread_mma` fragment 对齐？
8. MMA 输出是否仍满足第 1 条公式？
