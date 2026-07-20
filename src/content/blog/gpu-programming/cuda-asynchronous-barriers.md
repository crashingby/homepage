---
title: CUDA Asynchronous Barriers 笔记
date: 2026-07-10
tags: [CUDA, Asynchronous Barrier, Cooperative Groups, Hopper, GPU 编程]
summary: 整理 CUDA 异步屏障的 arrive/wait 分离语义、phase 机制、cuda::barrier 接口、显式阶段跟踪、提前退出、完成函数和生产者-消费者模式。
---

# CUDA Asynchronous Barriers 笔记

这篇笔记整理 CUDA Programming Guide 里的
[Asynchronous Barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
章节。

先记住一句话：

> `cuda::barrier` 可以看成 `__syncthreads()` 的更细粒度版本：它把“我已经到达同步点”和“我现在需要等待同步完成”拆成了两个动作。

传统的 `__syncthreads()` 是一个阻塞式同步点：线程执行到这里以后必须停住，直到整个 thread block 都到达。异步屏障的核心变化是：

- **split arrive/wait（到达和等待分离）**：线程可以先调用 `arrive()` 表示自己已经完成前置工作，然后继续做不依赖同步结果的计算，最后再调用 `wait()`。
- **支持更灵活的参与集合**：屏障的 expected arrival count（期望到达数）由初始化时指定，不一定等于整个 block 的线程数。
- **可以配合异步内存操作**：Hopper 之后的 transaction barrier（事务屏障）可以把“线程到齐”和“异步搬运完成”绑定在一起。
- **适合 warp specialization（warp 特化）**：生产者 warp 和消费者 warp 可以用单向同步构造流水线，而不必每一步都全员阻塞。

如果只是让整个 block 或整个 warp 简单同步，仍然优先使用 `__syncthreads()` 或 `__syncwarp()`。这些基础同步原语更直接，也通常有更好的简单场景性能。

## 总体模型

异步屏障围绕一个循环使用的 phase（阶段）工作。每个 phase 都有一个 countdown（倒计数），线程或事务通过 arrive 操作把 countdown 向下减；当 countdown 归零时，这个 phase 完成，屏障自动进入下一个 phase。

```mermaid
flowchart LR
    A["初始化屏障<br>expected arrival count = N"] --> B["当前 phase"]
    B --> C["线程调用 arrive()"]
    C --> D["countdown 递减"]
    D --> E{"countdown == 0 ?"}
    E -- "否" --> F["wait(token) 继续等待"]
    E -- "是" --> G["phase 完成"]
    G --> H["countdown 自动重置为 N"]
    H --> I["进入下一个 phase"]
```

`arrive()` 返回的 `arrival_token` 和当时的 phase 绑定。后续 `wait(std::move(token))` 会判断这个 token 对应的 phase 是否已经完成：

- 如果 phase 已经推进，`wait()` 不会阻塞。
- 如果 phase 还没推进，`wait()` 会阻塞到 countdown 归零。
- 如果线程已经阻塞在 `wait()` 中，phase 推进后线程会被唤醒。

这也是异步屏障比 `__syncthreads()` 更灵活的地方：**同步完成条件仍然严格，但等待动作可以延后**。

## 头文件和基本写法

使用高级 C++ API 通常需要：

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>
```

常见写法如下：

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void init_barrier_kernel()
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ barrier_t bar;
    cg::thread_block block = cg::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }

    // 初始化本身也需要同步。所有线程必须在使用 bar 之前看到构造完成的状态。
    block.sync();
}
```

这里有一个容易忽略的点：**屏障初始化之前不能使用屏障同步**。因此初始化后还需要用 `block.sync()` 或 `__syncthreads()` 做一次 bootstrapping（引导同步）。

## `cuda::barrier`

**用途**

`cuda::barrier` 表示一组 device 线程之间可重复使用的同步点。它既能表达普通线程同步，也能作为异步内存操作的完成条件。

**原型**

```cpp
template <cuda::thread_scope Scope,
          class CompletionFunction = /* empty completion */>
class cuda::barrier;
```

实际头文件中的内部类型名会随 CUDA 版本变化，学习时重点看两个模板参数：

| 模板参数 | 含义 |
| --- | --- |
| `Scope` | 屏障的同步和内存可见性范围，例如 `cuda::thread_scope_block`。 |
| `CompletionFunction` | 可选完成函数。每个 phase 完成时执行一次，默认是空函数。 |

**常用类型**

```cpp
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
```

最常见的屏障对象放在 shared memory 中：

```cpp
__shared__ barrier_t bar;
```

**生命周期 / 不变量**

- 屏障必须先初始化，再参与 arrive/wait。
- 如果屏障对象放在 shared memory 中，通常由一个线程初始化，然后用 `__syncthreads()` 或 `block.sync()` 引导同步。
- 每个 phase 的 arrive 数量必须和屏障协议匹配。少 arrive 会导致等待线程无法通过；多 arrive 或 token 使用错误会进入未定义行为。
- 屏障可重复使用，但必须遵守 phase token 的使用规则。

## `cuda::thread_scope`

**用途**

`cuda::thread_scope` 描述同步操作的作用范围和内存可见性范围。它不只用于 `cuda::barrier`，也会出现在 CUDA 原子操作、pipeline 等接口里。

**枚举值**

```cpp
enum thread_scope {
    thread_scope_system = __ATOMIC_SYSTEM,
    thread_scope_device = __ATOMIC_DEVICE,
    thread_scope_block  = __ATOMIC_BLOCK,
    thread_scope_thread = __ATOMIC_THREAD
};
```

| 枚举值 | 同步范围 | 典型使用场景 |
| --- | --- | --- |
| `thread_scope_block` | 当前 thread block 内部 | 最常见。shared memory 协作、block 内 producer-consumer、block 内异步拷贝同步。 |
| `thread_scope_device` | 当前 GPU device | 更大范围的 device 侧同步语义，通常需要配合更严格的执行模型。 |
| `thread_scope_system` | 系统范围，包括 CPU 和其他 GPU | 统一内存、系统级可见性或跨设备协作等细粒度场景。 |
| `thread_scope_thread` | 当前线程 | 对屏障本身意义很小，更常见于原子操作的最弱同步范围。 |

写 `cuda::barrier<cuda::thread_scope_block>` 时，意思是：**参与者在同一个 block 中，同步完成后只需要保证 block 范围内的内存可见性**。这也是 block 内 shared memory 协作最常用的配置。

## `init`

**用途**

初始化屏障对象，并指定每个 phase 需要多少次 arrive 才算完成。

**原型**

```cpp
void init(cuda::barrier<Scope, CompletionFunction>* bar,
          cuda::std::ptrdiff_t expected,
          CompletionFunction completion = CompletionFunction{});
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `bar` | `cuda::barrier<Scope, CompletionFunction>*` | 指向要初始化的屏障对象，通常位于 shared memory。 |
| `expected` | `cuda::std::ptrdiff_t` | 当前 phase 的 expected arrival count。倒计数从这个值开始。 |
| `completion` | `CompletionFunction` | 可选完成函数。最后一次 arrive 使 phase 完成后、等待线程放行前执行一次。 |

**副作用 / 约束**

- `init()` 必须发生在任何线程调用 `arrive()`、`wait()` 或 `arrive_and_wait()` 之前。
- 如果只有一个线程调用 `init()`，需要在初始化后执行一次 block 级同步，确保其他线程看到初始化结果。
- `expected` 不一定等于 `block.size()`，但后续每个 phase 的 arrive 协议必须和它一致。

## `barrier::arrive`

**用途**

表示当前线程已经到达屏障，并返回一个和当前 phase 绑定的 token。

**原型**

```cpp
using arrival_token = /* implementation-defined */;

arrival_token arrive(cuda::std::ptrdiff_t update = 1);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `update` | `cuda::std::ptrdiff_t` | 本次 arrive 对 countdown 的减少量，默认是 1。 |

**返回值**

| 类型 | 含义 |
| --- | --- |
| `arrival_token` | 记录当前 phase 的令牌，后续传给 `wait()`。 |

**副作用 / 约束**

- `arrive()` 本身不等待，它只更新屏障状态并返回 token。
- `update > 1` 表示一次调用贡献多个到达名额。只有当你能严格保证协议正确时才应该这么做，例如一个线程代表一个已经收敛的 warp 到达。
- 调用 `arrive()` 时，当前 phase 的 countdown 必须非零。
- 如果某次 `arrive()` 让 countdown 归零，屏障会自动完成当前 phase，并重置到下一 phase。

**使用场景**

```cpp
auto token = bar.arrive();

// 这里可以做不依赖本轮同步结果的工作。
do_independent_work();

bar.wait(cuda::std::move(token));
```

这个模式适合隐藏同步等待时间：线程先声明“我前置工作做完了”，然后趁等待其他线程的时间继续做独立计算。

## `barrier::arrive_and_wait`

**用途**

`arrive_and_wait()` 是 `arrive()` + `wait()` 的便捷组合：当前线程先对屏障贡献一次到达信号，然后立刻等待当前 phase 完成。

它的语义更接近传统的阻塞式屏障，但仍然使用 `cuda::barrier` 的 phase / expected count 机制。

**原型**

```cpp
void arrive_and_wait();
```

**等价理解**

可以把它近似理解成：

```cpp
auto token = bar.arrive();
bar.wait(cuda::std::move(token));
```

但写成 `arrive_and_wait()` 更直接，也能避免手动保存和移动 `arrival_token`。

**返回值**

| 类型 | 含义 |
| --- | --- |
| `void` | 不返回 token，因为等待动作已经在函数内部完成。 |

**副作用 / 约束**

- 当前线程会先 arrive，再阻塞等待本 phase 完成。
- 它没有 split arrive/wait 的 overlap 空间：调用后线程不会继续执行独立工作，而是直接等待。
- 和 `arrive()` 一样，调用时当前 phase 的 countdown 必须非零。
- 所有参与线程仍然必须满足屏障初始化时的 expected arrival count，否则等待线程会卡住。

**使用场景**

当你只需要“到达并等待”，不需要在 arrive 和 wait 之间插入独立计算时，可以用它让代码更清楚：

```cpp
// producer 必须等到当前 buffer 被 consumer 释放，才能覆盖 shared memory。
ready[buffer_id].arrive_and_wait();

fill_shared_buffer(buffer_id);
```

在 producer-consumer 示例中，`ready[buffer_id].arrive_and_wait()` 表示 producer 既贡献了自己的 ready 到达信号，也等待 consumer 释放该 buffer；`filled[buffer_id].arrive_and_wait()` 表示 consumer 既贡献了自己的 filled 到达信号，也等待 producer 填满该 buffer。

## `barrier::wait`

**用途**

等待某个 `arrival_token` 对应的 phase 完成。

**原型**

```cpp
void wait(arrival_token&& token) const;
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `token` | `arrival_token&&` | `arrive()` 返回的 phase token。调用时通常写 `cuda::std::move(token)` 或 `std::move(token)`。 |

**副作用 / 约束**

- `wait()` 只能使用当前 phase 或直接前一个 phase 的 token。使用更旧或无效 token 是未定义行为。
- 如果 token 对应的 phase 已经完成，`wait()` 会直接返回。
- 如果 phase 尚未完成，`wait()` 会阻塞到 phase 推进。

## 为什么 `cuda::barrier` 的接口里看不见 `phase` 参数？

`phase` 一直都在；高层 `cuda::barrier` 只是把它封装进 `arrival_token`，不让调用者手写 `0` 或 `1`。

```cpp
auto token = bar.arrive();
// token 绑定这一次 arrive 所在的 phase。

do_independent_work();
bar.wait(cuda::std::move(token));
// wait 根据 token 知道自己要等哪一代。
```

也就是说，`arrival_token` 不是普通的“我到过这里”的布尔标记；它是“我在第几代 barrier 到达”的不透明凭证。因而高层接口的正确用法是把 token 交回 `wait()`，而不是自行推导 phase。

只有降到 `cuda::ptx::mbarrier_try_wait_parity()` 这类底层接口时，phase 才会重新作为显式的 `0/1` parity 参数出现：

```cpp
// 低层 mbarrier：调用者自己维护每轮的 parity。
while (!cuda::ptx::mbarrier_try_wait_parity(native_handle, parity)) {
}
parity ^= 1;
```

因此后面“显式阶段跟踪”一节并不是另一套机制，而是把 `arrival_token` 在高层 API 中替你保存的那部分状态拿出来手动管理。CUTLASS 的 `ClusterBarrier::wait(ptr, phase)` 也属于这一层：它直接包装 `mbarrier` 的 parity wait；同时还能让 cluster 内其他 CTA 对目标 CTA 的 barrier 执行远端 `arrive`。

## Phase 使用规则

异步屏障最容易出错的地方不是 API 调用本身，而是 phase 规则。可以把每个 phase 看成一个严格的批次：

```mermaid
sequenceDiagram
    participant T0 as 线程 0
    participant T1 as 线程 1
    participant B as barrier

    T0->>B: arrive() -> token0
    T0->>T0: independent work
    T1->>B: arrive() -> token1
    B-->>B: countdown 归零，phase 推进
    T0->>B: wait(token0)
    B-->>T0: 已完成，直接返回
    T1->>B: wait(token1)
    B-->>T1: 已完成，直接返回
```

关键约束：

- `arrive()` 必须发生在屏障当前 phase 中。
- `wait(token)` 必须发生在 token 对应 phase 或紧接着的下一 phase 中。
- 如果某个线程的 `arrive()` 让 countdown 归零，在屏障被下一轮 `arrive()` 复用之前，相关线程必须按协议完成 `wait()`。
- 不要把 token 长期保存到后面很多轮再用。

直观理解：`arrival_token` 不是“永久门票”，它只是在当前同步轮次附近有效。

## Warp Entanglement

异步屏障在硬件层面会受到 warp divergence（warp 分歧）的影响。

如果一个 warp 中的 32 个线程收敛地执行 arrive-on 操作，硬件可以把它合并成更少的屏障更新；如果这些线程因为分支分歧而各自执行，屏障可能需要处理更多次独立更新。

| 情况 | 屏障更新特点 | 建议 |
| --- | --- | --- |
| warp 收敛执行 arrive | 更新次数少，开销更低 | 最理想。 |
| warp 严重分歧后 arrive | 可能产生更多屏障更新 | 在 arrive 前用 `__syncwarp()` 重新收敛。 |
| 只有一个 lane 代表 warp `arrive(32)` | 更新次数少，但协议更脆弱 | 只在能证明 32 个 lane 都已经完成对应工作时使用。 |

实践里可以记住一个简单规则：**如果 arrive 前有 warp 内分支，而且后续确实希望整个 warp 一起到达，就先 `__syncwarp()` 再 arrive。**

## 显式阶段跟踪

除了保存 `arrival_token`，CUDA 还提供了更底层的 `cuda::ptx::mbarrier_try_wait_parity()` 系列接口，用 phase parity（阶段极性）跟踪屏障翻转。

**用途**

显式阶段跟踪适合更底层的异步内存操作协议：有些线程只负责等待数据完成，不一定参与普通 token 式 arrive；这时用 parity 观察 phase 翻转会更自然。

**基本接口**

```cpp
bool cuda::ptx::mbarrier_try_wait_parity(
    uint64_t* bar,
    const uint32_t& phase_parity);
```

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `bar` | `uint64_t*` | 底层 mbarrier 对象地址，或由 `cuda::device::barrier_native_handle()` 取得的 native handle。 |
| `phase_parity` | `const uint32_t&` | 等待的 phase 极性。偶数 phase 为 0，奇数 phase 为 1。 |

初始 phase 的 parity 是 0。每完成一个 phase，parity 在 0 和 1 之间翻转。

**示例**

```cpp
#include <cuda/barrier>
#include <cuda/ptx>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void compute(float* data, int iteration);

__global__ void split_arrive_wait_kernel(int iteration_count, float* data)
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ barrier_t bar;
    int parity = 0;

    cg::thread_block block = cg::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();

    for (int i = 0; i < iteration_count; ++i) {
        // 当前线程到达，但不在这里阻塞。
        (void)cuda::ptx::mbarrier_arrive(
            cuda::device::barrier_native_handle(bar));

        compute(data, i);

        // 等待当前 parity 对应的 phase 完成。
        while (!cuda::ptx::mbarrier_try_wait_parity(
            cuda::device::barrier_native_handle(bar), parity)) {
        }

        parity ^= 1;
    }
}
```

**注意点**

- `mbarrier_try_wait_parity()` 是 try-wait 风格接口，示例里用 `while` 轮询把它变成阻塞等待。
- 每轮完成后必须更新本地 `parity`，否则下一轮可能等待错误的 phase。
- 这类接口更接近 PTX mbarrier 指令，适合需要控制底层同步协议的代码；普通 block 内同步优先用 `cuda::barrier` 的 `arrive()` / `wait()`。
- 显式 phase tracking 只适用于 thread-block 或 cluster scope 的 shared-memory barrier。

## `barrier::arrive_and_drop`

**用途**

当某个线程之后不再参与屏障同步时，先完成当前 phase 的 arrive 义务，再把后续 phase 的 expected arrival count 减少。

**原型**

```cpp
void arrive_and_drop();
```

**副作用 / 约束**

- 对当前 phase：贡献一次 arrive，避免当前轮次少一个到达信号。
- 对后续 phase：把 expected arrival count 减少 1，使屏障之后不再等待这个线程。
- 如果线程直接 `return` 而不 drop，剩余线程很容易在当前或下一 phase 死锁。

**示例**

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ bool should_exit(int iteration);
__device__ void do_work(int iteration);

__global__ void early_exit_kernel(int iteration_count)
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ barrier_t bar;
    cg::thread_block block = cg::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();

    for (int i = 0; i < iteration_count; ++i) {
        if (should_exit(i)) {
            bar.arrive_and_drop();
            return;
        }

        do_work(i);

        auto token = bar.arrive();
        bar.wait(cuda::std::move(token));
    }
}
```

一个典型场景是：线程块大小大于有效数据规模，部分线程在若干轮后不再有任务。如果它们还被 expected count 算进去，就必须用 `arrive_and_drop()` 正确退出同步协议。

## Completion Function

**用途**

`cuda::barrier<Scope, CompletionFunction>` 支持每个 phase 完成时执行一个 completion function（完成函数）。它在最后一次 arrive 之后、等待线程被放行之前执行一次。

完成函数适合表达“所有线程写完 shared memory 后，由一个执行上下文做一次汇总”的逻辑，例如 block 内归约、更新阶段状态、切换双缓冲索引等。

**内存可见性**

- 当前 phase 中，已经 arrive 的线程在 arrive 前完成的内存操作，对执行完成函数的线程可见。
- 完成函数中的内存操作，在等待线程从 `wait()` 返回后对它们可见。

**示例**

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <type_traits>
#include <utility>

namespace cg = cooperative_groups;

__device__ int independent_work(int value);

__global__ void block_sum_kernel(const int* input, int* output)
{
    constexpr int BlockSize = 128;

    cg::thread_block block = cg::this_thread_block();
    __shared__ int smem[BlockSize];

    auto completion_fn = [&] {
        int sum = 0;
        for (int i = 0; i < BlockSize; ++i) {
            sum += smem[i];
        }
        *output = sum;
    };

    using completion_fn_t = decltype(completion_fn);
    using barrier_t =
        cuda::barrier<cuda::thread_scope_block, completion_fn_t>;

    __shared__ std::aligned_storage_t<
        sizeof(barrier_t),
        alignof(barrier_t)> bar_storage;

    barrier_t* bar = reinterpret_cast<barrier_t*>(&bar_storage);

    if (block.thread_rank() == 0) {
        init(bar, block.size(), completion_fn);
    }
    block.sync();

    smem[block.thread_rank()] = input[block.thread_rank()];

    auto token = bar->arrive();

    // 可以执行与 smem 归约无关的工作。
    (void)independent_work(block.thread_rank());

    bar->wait(cuda::std::move(token));
}
```

**为什么这里用原始存储**

带捕获的 lambda 通常不是默认可构造对象。直接写：

```cpp
__shared__ barrier_t bar;
```

会要求编译器能默认构造 `bar`，但它不知道如何给 completion function 传入捕获上下文。因此示例中先用对齐的 shared memory 存储保留空间，再由一个线程调用 `init()` 或 placement new 完成构造。

## 跟踪异步内存操作

从 compute capability 9.0 开始，shared memory 中 thread-block 或 cluster scope 的异步屏障可以显式跟踪 asynchronous transaction（异步事务）。这类屏障通常和 TMA（Tensor Memory Accelerator）等异步搬运机制配合使用。

先把模型讲清楚：普通 `cuda::barrier` 的 phase 完成条件只有一个计数器。

- **arrival count**：参与线程是否都已经 arrive。

初始化时：

```cpp
init(&bar, expected);
```

可以理解成当前 phase 的 arrival count 从 `expected` 开始。每次：

```cpp
bar.arrive(update);
```

都会让 arrival count 减少 `update`。当 arrival count 变成 0，本 phase 完成，barrier 自动进入下一 phase。

transaction barrier 在这个基础上又多跟踪一个东西：

- **transaction count**：绑定到该 phase 的异步事务是否完成，单位由具体异步操作决定，常见是字节数。

所以 SM90+ 上可以把 phase 完成条件理解成：

```text
arrival count == 0
并且
当前 phase 预期的 transaction 都完成
```

这就是它解决的问题：普通 barrier 只能回答“线程都到了吗”，transaction barrier 还要回答“线程发起的异步内存事务也完成了吗”。

CUDA C++ 里相关接口包括：

```cpp
cuda::device::barrier_arrive_tx(bar, arrive_count_update, transaction_count_update);
cuda::device::barrier_expect_tx(bar, transaction_count_update);
```

二者的区别可以先记成：

| 接口 | 对 arrival count 的影响 | 对 transaction count 的影响 | 典型用途 |
| --- | --- | --- | --- |
| `bar.arrive(update)` | 减少 `update` | 不改变 | 只同步线程到达。 |
| `cuda::device::barrier_arrive_tx(bar, arrive_update, tx_update)` | 减少 `arrive_update` | 增加 `tx_update` | 当前线程 arrive，同时声明本 phase 还要等一批异步事务。 |
| `cuda::device::barrier_expect_tx(bar, tx_update)` | 不改变 | 增加 `tx_update` | 只声明本 phase 还要等一批异步事务，线程之后再用普通 `arrive()` 到达。 |

### `cuda::device::barrier_arrive_tx`

**用途**

在 arrive 的同时增加当前 phase 需要跟踪的 transaction count。它等价于把下面两件事做成一次原子更新：

```text
arrival count -= arrive_count_update
expected transaction count += transaction_count_update
```

注意这里的方向刚好相反：

- `arrive_count_update` 是 **减 arrival count**，表示“我到了”。
- `transaction_count_update` 是 **加 expected transaction count**，表示“这个 phase 还要额外等这么多异步事务”。

所以：

```cpp
auto token = cuda::device::barrier_arrive_tx(bar, 1, 1024);
```

可以读成：

```text
当前线程到达屏障：arrival count 减 1
同时声明本 phase 还要等待 1024 个 transaction 单位完成
返回当前 phase 的 arrival_token
```

如果这个 transaction 单位来自异步内存搬运，`1024` 常见含义就是“等待 1024 字节对应的异步事务完成”。具体单位由发起事务的异步原语定义；在常见异步搬运协议里，它就是字节数。

**原型**

```cpp
cuda::barrier<cuda::thread_scope_block>::arrival_token
cuda::device::barrier_arrive_tx(
    cuda::barrier<cuda::thread_scope_block>& bar,
    cuda::std::ptrdiff_t arrive_count_update,
    cuda::std::ptrdiff_t transaction_count_update);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `bar` | `cuda::barrier<cuda::thread_scope_block>&` | 要更新的屏障对象。 |
| `arrive_count_update` | `cuda::std::ptrdiff_t` | 本次 arrive 对 arrival count 的减少量。 |
| `transaction_count_update` | `cuda::std::ptrdiff_t` | 本 phase 新增要等待的 transaction 数量。常见异步搬运场景中可理解为要等待的字节数。 |

**返回值**

| 类型 | 含义 |
| --- | --- |
| `arrival_token` | 和当前 phase 绑定的 token，后续传给 `bar.wait(cuda::std::move(token))`。 |

**副作用 / 约束**

- `bar` 必须位于 shared memory 中。
- 只支持 compute capability 9.0 或更高。
- `arrive_count_update` 至少为 1。
- `transaction_count_update` 可以为 0；为 0 时，这次调用就退化成“带 token 的 arrive”。
- `arrive_count_update` 和 `transaction_count_update` 都有实现上限，文档给出的上限是 `(1 << 20) - 1`。
- 这个函数原子执行，并且这次调用 strongly happens-before 当前 phase 的 completion step 开始。

最后一点很关键：如果一个线程既要让 arrival count 减少，又要为本 phase 增加 expected transaction count，把两件事合在 `barrier_arrive_tx` 里可以避免竞态。

### 计数器怎么变化

假设一个 block 有 4 个参与线程：

```cpp
init(&bar, 4);
```

当前 phase 初始状态可以理解成：

| 状态 | 值 |
| --- | --- |
| `arrival_count` | 4 |
| `expected_transaction_count` | 0 |

现在 thread 0 发起一批异步事务，并调用：

```cpp
auto token0 = cuda::device::barrier_arrive_tx(bar, 1, 1024);
```

状态变成：

| 状态 | 值 | 解释 |
| --- | --- | --- |
| `arrival_count` | 3 | thread 0 已经 arrive，所以从 4 减到 3。 |
| `expected_transaction_count` | 1024 | 当前 phase 还要等待 1024 个 transaction 单位完成。 |

随后其他三个线程调用：

```cpp
auto token = bar.arrive();
```

当三个普通 arrive 都完成后：

| 状态 | 值 | phase 是否完成 |
| --- | --- | --- |
| `arrival_count` | 0 | 还不一定完成 |
| `expected_transaction_count` | 1024 | 还要等异步事务完成 |

这时线程已经“到齐”，但 phase 仍不能完成。只有当绑定到这个 phase 的异步事务也完成，transaction 部分被清掉后，等待这个 phase 的 `bar.wait(token)` 才会返回。

可以画成：

```mermaid
flowchart LR
    A["init(&bar, 4)<br>arrival=4, tx=0"] --> B["thread 0<br>barrier_arrive_tx(bar, 1, 1024)"]
    B --> C["arrival=3<br>tx=1024"]
    C --> D["thread 1/2/3<br>arrive()"]
    D --> E["arrival=0<br>tx=1024"]
    E --> F{"异步事务完成了吗？"}
    F -- "否" --> G["wait(token) 继续阻塞"]
    F -- "是" --> H["phase 完成<br>barrier 进入下一 phase"]
```

这就是 `transaction_count_update` 的直观含义：**它不是另一个 arrival decrement，而是给当前 phase 增加一笔“未来必须完成的异步事务账单”**。

**示例**

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void track_transaction_kernel()
{
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    cg::thread_block block = cg::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();

    // transaction_count_update = 0：
    // 只展示接口形状，语义接近普通 arrive。
    auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);
    bar.wait(cuda::std::move(token));
}
```

如果要展示 transaction 语义，可以写成下面这种抽象骨架：

```cpp
__global__ void track_async_transaction_kernel()
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ barrier_t bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    barrier_t::arrival_token token;

    if (threadIdx.x == 0) {
        constexpr cuda::std::ptrdiff_t transaction_bytes = 1024;

        // 这里代表“发起一批会在完成时通知 bar 的异步内存事务”。
        // 具体事务可以来自某个异步搬运原语；本节只关心 barrier 计数模型。
        launch_async_memory_transaction_associated_with(bar, transaction_bytes);

        // 当前线程 arrive，同时声明本 phase 要额外等待 transaction_bytes。
        token = cuda::device::barrier_arrive_tx(
            bar,
            1,
            transaction_bytes);
    } else {
        // 其他线程只表达普通到达。
        token = bar.arrive();
    }

    // wait 会同时等待：
    // 1. arrival count 归零；
    // 2. 当前 phase 的 transaction 完成。
    bar.wait(cuda::std::move(token));
}
```

这个例子里的 `launch_async_memory_transaction_associated_with()` 是概念占位，不是 CUDA 标准 API。它只是强调：transaction barrier 的关键不是普通线程多 arrive 一次，而是把某个异步内存事务的完成条件挂到当前 phase 上。

### `cuda::device::barrier_expect_tx`

**用途**

只增加当前 phase 的 expected transaction count，不执行 arrive。

可以把它理解成：

```text
expected transaction count += transaction_count_update
arrival count 不变
```

它适合把“声明要等多少事务”和“线程 arrive”拆开写的场景。

**原型**

```cpp
void cuda::device::barrier_expect_tx(
    cuda::barrier<cuda::thread_scope_block>& bar,
    cuda::std::ptrdiff_t transaction_count_update);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `bar` | `cuda::barrier<cuda::thread_scope_block>&` | 要更新 expected transaction count 的屏障对象。 |
| `transaction_count_update` | `cuda::std::ptrdiff_t` | 当前 phase 新增要等待的 transaction 数量。可以为 0，常见上限是 `(1 << 20) - 1`。 |

**示例**

```cpp
#include <cuda/barrier>

__global__ void expect_then_arrive_kernel()
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    __shared__ barrier_t bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr cuda::std::ptrdiff_t transaction_bytes = 1024;

        // 先声明当前 phase 还要等一批异步事务。
        cuda::device::barrier_expect_tx(bar, transaction_bytes);

        // 再发起会通知该 barrier 的异步事务。
        launch_async_memory_transaction_associated_with(bar, transaction_bytes);
    }

    // 所有线程按普通方式 arrive。
    auto token = bar.arrive();
    bar.wait(cuda::std::move(token));
}
```

这里的重点是顺序：`barrier_expect_tx()` 必须在当前 phase 可能完成之前执行。否则 arrival count 可能已经归零，phase 已经进入下一轮，你再增加 transaction count 就不是你想要的同步协议了。

### 两个接口怎么选

| 写法 | 推荐场景 |
| --- | --- |
| `barrier_arrive_tx(bar, arrive_update, tx_update)` | 同一个线程既要 arrive，又要声明本 phase 要等待的异步事务。它把两个更新原子合并，最不容易写错。 |
| `barrier_expect_tx(bar, tx_update)` + `bar.arrive()` | 声明 transaction 和线程 arrive 分属不同位置，或者需要先登记事务量、后面再按普通协议 arrive。 |
| `bar.arrive()` | 只需要同步线程，不需要跟踪异步事务。 |

一个实用判断是：

- 如果“发起异步事务的线程”也正好要代表自己到达屏障，用 `barrier_arrive_tx()`。
- 如果你要先登记一批事务，但线程到达逻辑在后面统一处理，用 `barrier_expect_tx()`。

### 常见误解

- **误解 1：`transaction_count_update` 会让 arrival count 再减一次。**  
  不会。arrival count 只由 `arrive_count_update` 减少；`transaction_count_update` 增加的是 expected transaction count。

- **误解 2：arrival count 归零就一定放行。**  
  对普通 barrier 是这样；对 transaction barrier，还要等当前 phase 的 transaction 完成。

- **误解 3：`transaction_count_update` 一定表示事务个数。**  
  不一定。它的单位由异步操作定义。常见异步搬运场景里，它通常按字节数理解，所以会看到 `sizeof(smem_tile)` 这类写法。

- **误解 4：`barrier_expect_tx()` 可以随便晚点调用。**  
  不行。它必须在当前 phase 完成前发生。否则它可能更新到错误 phase，或者破坏同步协议。

**注意点**

- transaction barrier 是 Hopper / SM90 之后更重要的能力，老架构上不能按这个模型做测试。
- 它解决的是“人到了，数据也到了吗”的问题，特别适合异步 global-to-shared 或 TMA 搬运。
- 普通 `arrive()` 只能表达线程到达，不能表达异步事务完成。
- `barrier_arrive_tx()` 的可用性可以用 `__cccl_lib_local_barrier_arrive_tx` 这类 feature flag 检查。

## 生产者-消费者模式

异步屏障很适合表达 block 内 producer-consumer（生产者-消费者）流水线。典型做法是把一个 block 里的 warp 分成两类：

- **producer warp**：负责把数据搬到 shared memory。
- **consumer warps**：负责消费 shared memory 中已经准备好的数据。

双缓冲时，每个 buffer 至少需要两个状态：

| 状态屏障 | 含义 |
| --- | --- |
| `ready[i]` | buffer `i` 已空，可以被 producer 填充。 |
| `filled[i]` | buffer `i` 已满，可以被 consumer 消费。 |

两个 buffer 就需要四个屏障：

| 屏障 | 含义 |
| --- | --- |
| `bar[0]` | buffer 0 ready。 |
| `bar[1]` | buffer 1 ready。 |
| `bar[2]` | buffer 0 filled。 |
| `bar[3]` | buffer 1 filled。 |

```mermaid
sequenceDiagram
    participant P as Producer warp
    participant R as ready barrier
    participant F as filled barrier
    participant C as Consumer warps

    C->>R: 初始 arrive，声明 buffer 可填
    P->>R: arrive_and_wait，等待 buffer 可填
    P->>P: 填充 shared buffer
    P->>F: arrive，声明 buffer 已满
    C->>F: arrive_and_wait，等待 buffer 已满
    C->>C: 消费 shared buffer
    C->>R: arrive，声明 buffer 再次可填
```

这个模式的重点是“单向同步”：

- producer 等待 ready，但不等待 filled。
- consumer 等待 filled，但不等待 ready。
- 双缓冲让 producer 填充下一个 buffer 的同时，consumer 处理上一个 buffer。

**核心代码骨架**

```cpp
#include <cuda/barrier>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__device__ void produce(
    barrier_t* ready,
    barrier_t* filled,
    float* buffer,
    int buffer_len,
    const float* input,
    int n)
{
    int lane = threadIdx.x % warpSize;

    for (int i = 0; i < n / buffer_len; ++i) {
        int buffer_id = i % 2;

        // ready[buffer_id] 表示这个 shared buffer 已经空出来，可以重新填充。
        // arrive_and_wait() 等价于先 arrive 再 wait：
        // - producer warp 在这里贡献自己的到达信号；
        // - 如果 consumer warps 还没有通过 ready[buffer_id].arrive() 释放这个 buffer，
        //   producer 会阻塞，避免覆盖仍在被消费的数据。
        ready[buffer_id].arrive_and_wait();

        float* current_buffer = buffer + buffer_id * buffer_len;
        const float* current_input = input + i * buffer_len;

        for (int j = lane; j < buffer_len; j += warpSize) {
            current_buffer[j] = current_input[j];
        }

        // filled[buffer_id] 表示这个 shared buffer 已经填满，可以被消费。
        // producer 这里只 arrive，不 wait：
        // - 这会唤醒正在 filled[buffer_id].arrive_and_wait() 上等待的 consumer；
        // - producer 自己不阻塞，可以马上进入下一轮，去等待另一个 buffer 的 ready 信号。
        (void)filled[buffer_id].arrive();
    }
}

__device__ void consume(
    barrier_t* ready,
    barrier_t* filled,
    const float* buffer,
    int buffer_len,
    float* output,
    int n)
{
    int consumer_tid = threadIdx.x - warpSize;
    int consumer_count = blockDim.x - warpSize;

    // 开局时两个 buffer 都是空的。consumer 先对 ready[0] 和 ready[1] arrive，
    // 相当于提前告诉 producer：两个 buffer 都可以被首次填充。
    // 这里不 wait，因为 consumer 的下一步是等待 filled，真正的数据依赖在 filled 上。
    (void)ready[0].arrive();
    (void)ready[1].arrive();

    for (int i = 0; i < n / buffer_len; ++i) {
        int buffer_id = i % 2;

        // filled[buffer_id] 表示 producer 已经把当前 buffer 填满。
        // consumer 在这里 arrive_and_wait()：
        // - consumer warps 贡献自己的到达信号；
        // - 如果 producer 还没执行 filled[buffer_id].arrive()，consumer 会阻塞；
        // - 一旦 expected arrival count 满足，consumer 就可以安全读取 shared buffer。
        filled[buffer_id].arrive_and_wait();

        const float* current_buffer = buffer + buffer_id * buffer_len;
        float* current_output = output + i * buffer_len;

        for (int j = consumer_tid; j < buffer_len; j += consumer_count) {
            current_output[j] = current_buffer[j] * 2.0f;
        }

        // consumer 消费完以后只 arrive，不 wait，通知 producer：
        // 这个 buffer 已经重新变成 ready 状态，可以被下一轮填充。
        (void)ready[buffer_id].arrive();
    }
}

__global__ void producer_consumer_kernel(
    int n,
    const float* input,
    float* output,
    int buffer_len)
{
    extern __shared__ float buffer[];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t bar[4];

    // bar[0], bar[1] 分别表示 buffer 0/1 ready；
    // bar[2], bar[3] 分别表示 buffer 0/1 filled。
    // 这里 expected arrival count 写 blockDim.x，意味着 producer 和 consumer
    // 两边所有线程都必须参与每个 phase 的 arrive 协议。
    if (threadIdx.x < 4) {
        init(&bar[threadIdx.x], blockDim.x);
    }

    // barrier 对象由前 4 个线程初始化。初始化完成前，其他线程不能调用 arrive/wait，
    // 所以这里必须用传统 block 级同步做一次 bootstrapping。
    __syncthreads();

    if (threadIdx.x < warpSize) {
        // warp 0 是 producer，负责把 global memory 数据搬进双缓冲 shared memory。
        produce(bar, bar + 2, buffer, buffer_len, input, n);
    } else {
        // 其余 warps 是 consumer，等待 filled 信号后读取 shared memory 并写回 output。
        consume(bar, bar + 2, buffer, buffer_len, output, n);
    }
}
```

这里虽然只有 producer warp 负责搬运，consumer warps 负责计算，但四个屏障都用 `blockDim.x` 初始化。因此每个 phase 必须由整个 block 的线程共同贡献 arrive。也就是说，**不做某类工作，不等于不参与对应屏障协议**。

### 逐步拆解

虽然所有线程都在“签到”，但只有一部分线程会在某个屏障上真正等待。理解这个例子时，不要只看谁调用了 `arrive()`，还要看谁调用了 `wait()` 或 `arrive_and_wait()`。

为了方便理解，假设只有两个 buffer：

- buffer 0 和 buffer 1 轮流作为双缓冲。
- `ready[i]` 表示 buffer `i` 已经空出来，可以被 producer 填充。
- `filled[i]` 表示 buffer `i` 已经填满，可以被 consumer 消费。

#### 第一步：开局，consumer 先释放两个空 buffer

consumer warps 一进入 `consume()`，先执行：

```cpp
(void)ready[0].arrive();
(void)ready[1].arrive();
```

这两个 arrive 不会阻塞 consumer。它们的含义是：**buffer 0 和 buffer 1 一开始都是空的，producer 可以填**。

随后 consumer 进入循环，第一次会执行：

```cpp
filled[0].arrive_and_wait();
```

这才是 consumer 的第一个真正阻塞点。因为此时 producer 还没有填好 buffer 0，也还没有对 `filled[0]` arrive，所以 consumer 会在这里等待。

producer warp 一进入 `produce()`，第一次会执行：

```cpp
ready[0].arrive_and_wait();
```

这里 producer 自己贡献 producer warp 的 arrive；前面 consumer warps 已经对 `ready[0]` 贡献过 arrive。因为 `ready[0]` 的 expected arrival count 是 `blockDim.x`，两边线程的 arrive 凑齐以后，`ready[0]` 这个 phase 完成，producer 不再阻塞，开始填充 buffer 0。

#### 第二步：buffer 0 被填满，consumer 被唤醒

producer 填完 buffer 0 后执行：

```cpp
(void)filled[0].arrive();
```

这个 arrive 的含义是：**buffer 0 已经填满**。producer 不在 `filled[0]` 上等待，打完这个信号后会继续进入下一轮，准备处理 buffer 1。

consumer 原本阻塞在：

```cpp
filled[0].arrive_and_wait();
```

consumer 自己已经对 `filled[0]` arrive 过，现在 producer 也 arrive 了，`filled[0]` 的 expected arrival count 凑齐，consumer 被唤醒，开始读取并处理 buffer 0。

#### 第三步：双缓冲开始重叠

时间线推进到这里后，producer 和 consumer 开始并行处理不同 buffer：

- **producer warp 处理 buffer 1**：producer 进入 `i = 1`，执行 `ready[1].arrive_and_wait()`。因为 consumer 在开局时已经预先对 `ready[1]` arrive，所以 producer 对 `ready[1]` arrive 后，这个 phase 很快完成，producer 可以直接填充 buffer 1。
- **consumer warps 处理 buffer 0**：consumer 刚从 `filled[0].arrive_and_wait()` 返回，正在消费 buffer 0 中的数据。

当 consumer 消费完 buffer 0 后，会执行：

```cpp
(void)ready[0].arrive();
```

这表示 buffer 0 又空出来了。等 producer 填完 buffer 1、下一轮回到 buffer 0 时，就可以通过 `ready[0].arrive_and_wait()` 继续填充 buffer 0。

所以这个例子的关键不是“四个屏障都让所有线程一起停下”，而是：

- `ready` 这条同步边主要让 producer 等 consumer 释放 buffer。
- `filled` 这条同步边主要让 consumer 等 producer 填满 buffer。
- producer 和 consumer 在不同 buffer 上交错前进，从而形成一个简单的 block 内流水线。

## 使用建议

- **简单全 block 同步优先 `__syncthreads()`**：异步屏障不是替代所有同步的银弹。
- **把 expected count 当成协议的一部分**：初始化时写多少，后续每个 phase 就必须严格满足多少。
- **arrive 和 wait 之间只放独立工作**：不要在 `wait()` 之前读取依赖本轮同步结果的数据。
- **分支退出必须 drop**：参与过屏障序列的线程提前退出时，用 `arrive_and_drop()` 维护后续 phase 的 expected count。
- **warp 分歧后先收敛**：如果希望 warp 共同 arrive，分支后用 `__syncwarp()` 减少 warp entanglement 带来的屏障更新开销。
- **底层 PTX 接口谨慎使用**：`mbarrier_*` 更适合 TMA、显式 phase tracking 或性能敏感的同步协议；普通同步先用 `cuda::barrier`。

## 小结

`cuda::barrier` 的价值不在于“比 `__syncthreads()` 更高级”，而在于它能表达 `__syncthreads()` 不方便表达的同步协议：

- 先 arrive，再延后 wait。
- 让 producer 和 consumer 做单向同步。
- 让线程提前退出同步序列。
- 在 phase 完成时执行一次 completion function。
- 在 Hopper 之后跟踪异步内存事务完成。

理解它时不要只盯着 API 名字，而要始终追踪三个状态：**expected arrival count、当前 phase、token 或 parity 属于哪一轮**。这三件事对了，异步屏障的代码才不会在复杂流水线里悄悄卡死。
