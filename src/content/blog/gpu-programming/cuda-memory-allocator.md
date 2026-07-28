---
title: CUDA Stream-Ordered Memory Allocator 笔记
date: 2026-07-28
tags: [CUDA, GPU 编程, Memory Pool, Stream Ordered Allocator, Runtime API]
---

# CUDA Stream-Ordered Memory Allocator 笔记

这篇笔记整理 CUDA 的 **Stream-Ordered Memory Allocator（流有序内存分配器）**。它对应 `cudaMallocAsync`、`cudaFreeAsync` 和 `cudaMemPool*` 系列 API，是比普通 `cudaMalloc/cudaFree` 更适合高频分配、库间共享和流水线调度的内存管理方式。

参考：

- [CUDA Programming Guide：Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- CUDA 13.3 本机头文件：`/usr/local/cuda-13.3/include/cuda_runtime_api.h`、`/usr/local/cuda-13.3/include/cuda_runtime.h`、`/usr/local/cuda-13.3/include/driver_types.h`

## 为什么需要流有序分配器

传统 `cudaMalloc` / `cudaFree` 的问题不是“不能用”，而是它们很容易把内存生命周期和全局同步绑在一起。对于频繁临时分配的 workload，例如推理服务的中间 buffer、图像 pipeline、多库组合计算，分配释放本身可能变成调度瓶颈。

流有序分配器的核心变化是：

- `cudaMallocAsync` 把**分配操作**插入到某个 stream 中。
- `cudaFreeAsync` 把**释放操作**插入到某个 stream 中。
- 指针从 API 返回后不代表立刻可以被所有 GPU work 使用；它必须等 stream 执行到 allocation 操作之后才可用。
- free 也不是 host 立即回收，而是 stream 执行到 free 操作后，后续 GPU work 才不能再访问该指针。
- memory pool（内存池）负责缓存和复用物理内存，减少向 OS / driver 反复申请释放的成本。

```mermaid
flowchart LR
  A["cudaMallocAsync<br/>插入 stream"] --> B["kernel / memcpy<br/>使用指针"]
  B --> C["cudaFreeAsync<br/>插入 stream"]
  C --> D["pool 可复用或归还物理内存"]
```

这个模型的关键词是 **stream-ordering contract（流顺序契约）**：只要所有访问都发生在 allocation 之后、free 之前，allocator 就能在不全局同步的前提下安全复用内存。

## 和普通分配的区别

| 机制 | 同步语义 | 内存来源 | 适合场景 |
| --- | --- | --- | --- |
| `cudaMalloc` / `cudaFree` | 容易引入全局同步，尤其 `cudaFree` 需要确认没有未完成访问。 | runtime 直接管理 device allocation。 | 低频分配、初始化阶段分配、简单程序。 |
| `cudaMallocAsync` / `cudaFreeAsync` | 分配和释放按 stream 排序，不阻塞 host，也不要求所有 stream 停下来。 | 当前 memory pool 或显式指定 pool。 | 高频临时 buffer、pipeline、多 stream、多库共享 allocator。 |
| 自己写 caching allocator | 语义完全由应用维护。 | 自己管理大块 `cudaMalloc` 后切片。 | 需要极细粒度控制，但维护成本高。 |
| `cudaMemPool*` | runtime/driver 理解 pool、stream、event 和 reuse policy。 | 显式或默认 memory pool。 | 想降低分配成本，同时保留 CUDA 对 stream 依赖的理解。 |

## 支持能力查询

使用前建议查询设备属性。CUDA 13.3 头文件中相关属性包括：

```cpp
enum cudaDeviceAttr {
    /** 设备是否支持 cudaMallocAsync 和 cudaMemPool 系列 API。 */
    cudaDevAttrMemoryPoolsSupported = 115,

    /** mempool IPC 支持哪些 cudaMemAllocationHandleType handle。 */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,

    /** 设备是否支持 HOST_NUMA location 的 cudaMallocAsync / cudaMemPool API。 */
    cudaDevAttrHostNumaMemoryPoolsSupported = 142,

    /** 设备是否支持 HOST location 的 cudaMallocAsync / cudaMemPool API。 */
    cudaDevAttrHostMemoryPoolsSupported = 144,
};
```

**使用场景**

- `cudaDevAttrMemoryPoolsSupported` 是能否使用 stream-ordered allocator 的基础开关。
- `cudaDevAttrMemoryPoolSupportedHandleTypes` 用于判断是否能创建 IPC-capable pool，例如是否支持 `cudaMemHandleTypePosixFileDescriptor`。
- `cudaDevAttrHostNumaMemoryPoolsSupported` / `cudaDevAttrHostMemoryPoolsSupported` 是 CUDA 13.x 中更高级的 host / NUMA pool 场景，普通 device pool 不需要它们。

示例：

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

/**
 * @brief 检查 CUDA Runtime API 返回值，失败时打印文件和行号后退出。
 *
 * @param status CUDA Runtime API 的返回值。
 * @param file 调用点所在源文件。
 * @param line 调用点所在行号。
 */
void check_cuda(cudaError_t status, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                 file, line, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

/**
 * @brief 查询设备是否支持 stream-ordered memory allocator。
 *
 * @param device_id CUDA device ordinal。
 * @return 如果设备支持 cudaMallocAsync / cudaMemPool 系列 API，返回 true。
 */
bool device_supports_memory_pool(int device_id) {
    int supported = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supported,
                                      cudaDevAttrMemoryPoolsSupported,
                                      device_id));
    return supported != 0;
}
```

## 核心类型

### `cudaMemPool_t`

**用途**

`cudaMemPool_t` 是 CUDA memory pool 的 opaque handle（不透明句柄）。应用只能通过 Runtime API 查询、设置和销毁它，不能解引用内部结构。

**来源**

```cpp
typedef struct CUmemPoolHandle_st* cudaMemPool_t;
```

**注意点**

- default pool 由 CUDA 创建，不能用 `cudaMemPoolDestroy` 销毁。
- explicit pool 由 `cudaMemPoolCreate` 创建，用 `cudaMemPoolDestroy` 销毁。
- imported pool 来自 IPC handle，不能用于 `cudaMallocFromPoolAsync` 创建新 allocation。

### `cudaMemPoolAttr`

**用途**

描述 memory pool 可查询或可设置的属性，包括复用策略、释放阈值、统计水位、位置和导出能力。

**枚举来源**

```cpp
enum cudaMemPoolAttr {
    /** 允许根据 event / null stream 等显式依赖，复用其它 stream 中异步释放的内存。 */
    cudaMemPoolReuseFollowEventDependencies = 0x1,

    /** 允许复用已经执行完成的 free，即使没有显式 stream 依赖。 */
    cudaMemPoolReuseAllowOpportunistic = 0x2,

    /** 允许 allocator 插入内部 stream 依赖，以满足复用某块已 free 内存的顺序要求。 */
    cudaMemPoolReuseAllowInternalDependencies = 0x3,

    /** pool 在尝试归还 OS 前允许保留的 reserved memory 字节数。 */
    cudaMemPoolAttrReleaseThreshold = 0x4,

    /** 当前 pool 已从系统保留的 backing memory 字节数。 */
    cudaMemPoolAttrReservedMemCurrent = 0x5,

    /** reserved memory 高水位，只能重置为 0。 */
    cudaMemPoolAttrReservedMemHigh = 0x6,

    /** 当前应用仍在使用、不能复用的 pool memory 字节数。 */
    cudaMemPoolAttrUsedMemCurrent = 0x7,

    /** used memory 高水位，只能重置为 0。 */
    cudaMemPoolAttrUsedMemHigh = 0x8,

    /** pool 的分配类型，值类型为 cudaMemAllocationType。 */
    cudaMemPoolAttrAllocationType = 0x9,

    /** pool 可导出的 handle 类型，值类型为 cudaMemAllocationHandleType。 */
    cudaMemPoolAttrExportHandleTypes = 0xA,

    /** pool 的 location id，例如 device ordinal 或 NUMA node id。 */
    cudaMemPoolAttrLocationId = 0xB,

    /** pool 的 location type，值类型为 cudaMemLocationType。 */
    cudaMemPoolAttrLocationType = 0xC,

    /** pool 最大大小，0 表示系统相关的默认上限。 */
    cudaMemPoolAttrMaxPoolSize = 0xD,

    /** pool 是否启用硬件解压用途。 */
    cudaMemPoolAttrHwDecompressEnabled = 0xE,
};
```

**属性语义**

| 枚举 | 值类型 | 可设置 | 含义 |
| --- | --- | --- | --- |
| `cudaMemPoolReuseFollowEventDependencies` | `int` | 是 | 允许 allocator 根据 event 等跨 stream 依赖复用已释放内存，默认启用。 |
| `cudaMemPoolReuseAllowOpportunistic` | `int` | 是 | 如果某个异步 free 实际已经完成，即使没有显式依赖，也允许机会性复用，默认启用。 |
| `cudaMemPoolReuseAllowInternalDependencies` | `int` | 是 | 允许 allocator 自动插入内部依赖来复用内存，默认启用。 |
| `cudaMemPoolAttrReleaseThreshold` | `cuuint64_t` | 是 | pool 保留 reserved memory 的阈值。超过后，下一次 stream/event/device synchronize 时可能归还 OS。 |
| `cudaMemPoolAttrReservedMemCurrent` | `cuuint64_t` | 否 | pool 当前从系统保留的 backing memory。 |
| `cudaMemPoolAttrReservedMemHigh` | `cuuint64_t` | 只能设 0 | `ReservedMemCurrent` 的高水位。设为 0 表示重置统计。 |
| `cudaMemPoolAttrUsedMemCurrent` | `cuuint64_t` | 否 | 应用当前仍使用的 allocation 总量。 |
| `cudaMemPoolAttrUsedMemHigh` | `cuuint64_t` | 只能设 0 | `UsedMemCurrent` 的高水位。设为 0 表示重置统计。 |
| `cudaMemPoolAttrAllocationType` | `cudaMemAllocationType` | 否 | 查询 pool 管理 pinned 还是 managed allocation。 |
| `cudaMemPoolAttrExportHandleTypes` | `cudaMemAllocationHandleType` | 否 | 查询 pool 支持的 IPC export handle 类型。imported pool 查询时通常为 `cudaMemHandleTypeNone`。 |
| `cudaMemPoolAttrLocationId` | `int` | 否 | 查询 pool 驻留位置 id。若 location type 为 `cudaMemLocationTypeInvisible`，通常为 `cudaInvalidDeviceId`。 |
| `cudaMemPoolAttrLocationType` | `cudaMemLocationType` | 否 | 查询 pool 驻留位置类型。 |
| `cudaMemPoolAttrMaxPoolSize` | `cuuint64_t` | 创建时由 props 决定 | 查询 pool 最大大小。0 表示无显式最大值或系统默认。 |
| `cudaMemPoolAttrHwDecompressEnabled` | `int` | 创建时由 usage 决定 | 查询 pool 是否启用硬件解压用途。 |

### `cudaMemAllocationType`

**用途**

描述 pool 中 allocation 的类型。CUDA 13.3 中 stream-ordered pool 可以是 pinned allocation，也可以是 managed allocation。

**枚举来源**

```cpp
enum cudaMemAllocationType {
    /** 无效分配类型。 */
    cudaMemAllocationTypeInvalid = 0x0,

    /** pinned allocation，活跃使用期间不能从当前位置迁移。 */
    cudaMemAllocationTypePinned = 0x1,

    /** managed memory allocation，由 Unified Memory 管理。 */
    cudaMemAllocationTypeManaged = 0x2,

    /** 枚举哨兵值，不作为实际分配类型使用。 */
    cudaMemAllocationTypeMax = 0x7FFFFFFF
};
```

**注意点**

- 创建普通 device pool 时，`allocType` 通常填 `cudaMemAllocationTypePinned`，这里的 pinned 指的是底层 allocation 不迁移，不是上一章讲的 pinned host staging buffer。
- 创建 managed memory pool 时，`allocType` 填 `cudaMemAllocationTypeManaged`，并且 `handleTypes` 必须为 `cudaMemHandleTypeNone`，因为 managed memory pool 不支持 IPC。

### `cudaMemAllocationHandleType`

**用途**

描述 memory pool 是否支持导出为 OS / fabric handle，用于 IPC 或跨节点共享。

**枚举来源**

```cpp
enum cudaMemAllocationHandleType {
    /** 不允许任何 export 机制。 */
    cudaMemHandleTypeNone = 0x0,

    /** POSIX 文件描述符，只允许在 POSIX 系统上使用。 */
    cudaMemHandleTypePosixFileDescriptor = 0x1,

    /** Win32 NT handle。 */
    cudaMemHandleTypeWin32 = 0x2,

    /** Win32 KMT handle。 */
    cudaMemHandleTypeWin32Kmt = 0x4,

    /** fabric handle，用于支持 fabric 的跨节点共享场景。 */
    cudaMemHandleTypeFabric = 0x8
};
```

**注意点**

- `cudaMemHandleTypeNone` 表示 pool 不支持 IPC。
- 想让 `cudaMemPoolExportToShareableHandle` 成功，创建 pool 时 `cudaMemPoolProps::handleTypes` 必须包含目标 handle type。
- `cudaMemHandleTypeFabric` 还依赖 IMEX channel 等系统配置，普通单机应用很少直接使用。

### `cudaMemLocationType` 和 `cudaMemLocation`

`cudaMemLocation` 在上一篇内存笔记里已经介绍过。这里它用于描述 pool 的驻留位置。

```cpp
enum cudaMemLocationType {
    /** 无效位置，也可作为未指定位置的编码值。 */
    cudaMemLocationTypeInvalid = 0,

    /** 未指定 preferred location，常用于 managed memory pool。 */
    cudaMemLocationTypeNone = 0,

    /** GPU device 位置，此时 id 表示 GPU device ordinal。 */
    cudaMemLocationTypeDevice = 1,

    /** 普通 host 位置，此时 id 忽略。 */
    cudaMemLocationTypeHost = 2,

    /** host NUMA node 位置，此时 id 表示 CPU NUMA node id。 */
    cudaMemLocationTypeHostNuma = 3,

    /** 当前 CPU 线程最近的 host NUMA node，此时 id 忽略。 */
    cudaMemLocationTypeHostNumaCurrent = 4,

    /** 位置不可见但 device 可访问，id 固定为 cudaInvalidDeviceId。 */
    cudaMemLocationTypeInvisible = 5
};

struct cudaMemLocation {
    enum cudaMemLocationType type;
    union {
        int id;
    };
};
```

**在 mempool 中的合法组合**

| location type | 常见用途 | 注意点 |
| --- | --- | --- |
| `cudaMemLocationTypeDevice` | 创建某个 GPU 上的 device pool。 | `id` 是 GPU device ordinal。默认 pool 最常见就是这种位置。 |
| `cudaMemLocationTypeHost` | 创建普通 host memory pool。 | `id` 忽略；不支持 IPC，`handleTypes` 必须为 `cudaMemHandleTypeNone`。 |
| `cudaMemLocationTypeHostNuma` | 创建驻留在某个 CPU NUMA node 的 host pool。 | `id` 是 NUMA node id；是否支持可查 `cudaDevAttrHostNumaMemoryPoolsSupported`。 |
| `cudaMemLocationTypeHostNumaCurrent` | 不适合 `cudaMemPoolCreate`。 | CUDA 13.3 头文件说明，创建 pool 时指定它会返回 `cudaErrorInvalidValue`。 |
| `cudaMemLocationTypeNone` | managed memory pool 无 preferred location。 | 仅在 `allocType == cudaMemAllocationTypeManaged` 时用于表达没有偏好位置。 |
| `cudaMemLocationTypeInvisible` | 查询 imported / fabric pool 时可能出现。 | 普通创建参数里不应该手写。 |

### `cudaMemAccessFlags` 和 `cudaMemAccessDesc`

**用途**

描述某个 location 对 pool allocation 的访问权限。主要用于 `cudaMemPoolSetAccess` 和 `cudaMemPoolGetAccess`。

**来源**

```cpp
enum cudaMemAccessFlags {
    /** 默认不可访问。 */
    cudaMemAccessFlagsProtNone = 0,

    /** 只读访问。 */
    cudaMemAccessFlagsProtRead = 1,

    /** 读写访问。 */
    cudaMemAccessFlagsProtReadWrite = 3
};

struct cudaMemAccessDesc {
    /** 要修改访问权限的位置，例如某个 GPU。 */
    struct cudaMemLocation location;

    /** 该位置上的访问权限。 */
    enum cudaMemAccessFlags flags;
};
```

**注意点**

- memory pool 的 access 设置不跟随 `cudaDeviceEnablePeerAccess`。
- 对 pool 开启另一个 GPU 的访问前，最好先用 `cudaDeviceCanAccessPeer` 查询 peer capability。
- 对 pool 的 access 修改作用于 pool 中所有 allocation，不只是未来新分配的 allocation。

### `cudaMemPoolProps`

**用途**

创建 explicit memory pool 时使用的配置结构体。它决定 pool 的分配类型、导出能力、驻留位置、最大大小和特殊用途。

**结构体来源**

```cpp
struct cudaMemPoolProps {
    /** pool allocation 类型，例如 cudaMemAllocationTypePinned 或 cudaMemAllocationTypeManaged。 */
    enum cudaMemAllocationType allocType;

    /** 支持导出的 handle 类型；为 cudaMemHandleTypeNone 时不支持 IPC。 */
    enum cudaMemAllocationHandleType handleTypes;

    /** allocation 驻留位置，例如 device、host、host NUMA node 或 managed preferred location。 */
    struct cudaMemLocation location;

    /** Windows 专用安全属性；只有 handleTypes 包含 cudaMemHandleTypeWin32 时使用。 */
    void* win32SecurityAttributes;

    /** pool 最大大小；0 表示使用系统相关默认值或无显式上限。 */
    size_t maxSize;

    /** pool 用途 bitmask，例如 cudaMemPoolCreateUsageHwDecompress。 */
    unsigned short usage;

    /** 保留字段，必须清零。 */
    unsigned char reserved[54];
};
```

**成员变量**

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `allocType` | `cudaMemAllocationType` | pool 中 allocation 的类型。普通 device pool 常用 `cudaMemAllocationTypePinned`；managed pool 用 `cudaMemAllocationTypeManaged`。 |
| `handleTypes` | `cudaMemAllocationHandleType` | IPC / export handle 类型。多个 bit 可以组合；`cudaMemHandleTypeNone` 表示不可导出。 |
| `location` | `cudaMemLocation` | pool allocation 驻留位置或 managed preferred location。 |
| `win32SecurityAttributes` | `void*` | Windows handle 安全属性。非 Win32 handle 场景必须为 0。 |
| `maxSize` | `size_t` | pool 最大大小。managed memory pool 要求为 0。 |
| `usage` | `unsigned short` | 特殊用途 bitmask。`cudaMemPoolCreateUsageHwDecompress` 表示硬件解压 buffer。 |
| `reserved` | `unsigned char[54]` | 保留字段，必须清零。 |

**约束**

- 创建 host pool 且 `location.type == cudaMemLocationTypeHost` 时，`handleTypes` 必须为 `cudaMemHandleTypeNone`。
- 创建 host NUMA pool 时，`location.type == cudaMemLocationTypeHostNuma` 且 `location.id` 是 NUMA node id。
- 创建 managed memory pool 时，`allocType == cudaMemAllocationTypeManaged`、`handleTypes == cudaMemHandleTypeNone`、`maxSize == 0`、`usage == 0`。
- managed memory pool 要求系统中所有 device 都有非零 `concurrentManagedAccess`，否则 `cudaMemPoolCreate` 返回 `cudaErrorNotSupported`。

### `cudaMemPoolPtrExportData`

**用途**

IPC 中导出某个 pool allocation 的 opaque data（不透明数据）。它不是 OS handle，可以用共享内存、socket、pipe 等任意 IPC 机制传递给导入进程。

**来源**

```cpp
struct cudaMemPoolPtrExportData {
    unsigned char reserved[64];
};
```

**注意点**

- 先导出 / 导入 pool handle，再导出 / 导入具体 allocation。
- 导入 allocation 后，导入进程必须通过 event 或其它同步手段保证 allocation 已经在导出进程的 stream 中完成。
- 导入进程必须先释放 imported pointer，导出进程才能释放原始 pointer。

## 核心 API

### `cudaMallocAsync`

**用途**

按 stream 顺序分配 device memory。分配操作被插入到 `hStream` 中，后续同一 stream 的 kernel / memcpy 可以按顺序使用该 allocation。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMallocAsync(
    void** devPtr,
    size_t size,
    cudaStream_t hStream);
```

CUDA C++ 模板重载：

```cpp
template <class T>
cudaError_t cudaMallocAsync(
    T** ptr,
    size_t size,
    cudaStream_t stream);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `devPtr` / `ptr` | `void**` / `T**` | 输出参数，返回 stream-ordered allocation 指针。 |
| `size` | `size_t` | 分配字节数。 |
| `hStream` / `stream` | `cudaStream_t` | 建立分配顺序和选择 current memory pool 的 stream。 |

**返回值**

| 类型 | 含义 |
| --- | --- |
| `cudaError_t` | 成功返回 `cudaSuccess`；失败可能是 `cudaErrorInvalidValue`、`cudaErrorNotSupported`、`cudaErrorOutOfMemory`。 |

**副作用 / 约束**

- 指针会立即返回，但 GPU work 必须等 allocation 操作在 stream 中执行完成后才能访问。
- 默认从 `hStream` 所属 device 的 current memory pool 分配。
- `cudaMallocAsync` 选择 allocation 位置时主要看 stream / memory pool，不要只看调用线程当前 device。
- stream capture 期间会创建 graph allocation node，此时 allocation 由 graph 拥有，不是 pool 拥有。

### `cudaFreeAsync`

**用途**

按 stream 顺序释放 allocation。释放操作执行到之前，该指针仍可被满足顺序约束的 GPU work 使用；执行到之后不能再访问。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaFreeAsync(
    void* devPtr,
    cudaStream_t hStream);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `devPtr` | `void*` | 要释放的 device pointer，可来自 `cudaMallocAsync`，也可来自普通 `cudaMalloc`。 |
| `hStream` | `cudaStream_t` | 建立释放顺序的 stream。 |

**副作用 / 约束**

- 所有访问必须发生在 free 操作执行前；free 后继续访问是 undefined behavior。
- `cudaFreeAsync` 返回后再调用 `cudaPointerGetAttributes(devPtr)` 也是 undefined behavior，即使某些 stream 尚未执行到 free。
- `cudaMalloc` 分配的内存可以用 `cudaFreeAsync` 释放，但仍要保证所有访问在 free 前完成。
- `cudaMallocAsync` 分配的内存也可以用 `cudaFree` 释放，但这时 CUDA 不会帮你等待 async work 完成，应用必须先同步。

### `cudaMallocFromPoolAsync`

**用途**

从显式指定的 memory pool 中按 stream 顺序分配内存。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(
    void** ptr,
    size_t size,
    cudaMemPool_t memPool,
    cudaStream_t stream);
```

CUDA C++ 还提供一个 `cudaMallocAsync(ptr, size, memPool, stream)` 重载，它只是 `cudaMallocFromPoolAsync` 的 alternate spelling：

```cpp
cudaError_t cudaMallocAsync(
    void** ptr,
    size_t size,
    cudaMemPool_t memPool,
    cudaStream_t stream);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `ptr` | `void**` / `T**` | 输出参数，返回从指定 pool 分配的指针。 |
| `size` | `size_t` | 分配字节数。 |
| `memPool` | `cudaMemPool_t` | 指定 memory pool。它可以来自不同于 `stream` 所在 device 的位置，但访问权限和位置必须合法。 |
| `stream` | `cudaStream_t` | 建立 stream-ordering contract 的 stream。 |

**使用场景**

- 不想把某个 pool 设置成 device current pool，只想某次分配显式指定 pool。
- 多库共享同一个 pool。
- 从 IPC-capable explicit pool 创建可导出的 allocation。

### `cudaDeviceGetDefaultMemPool`

**用途**

查询某个 device 的 default memory pool。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(
    cudaMemPool_t* memPool,
    int device);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t*` | 输出参数，返回 device 的默认 pool。 |
| `device` | `int` | CUDA device ordinal。 |

**注意点**

- default pool 中是该 device 本地的 non-migratable device allocation。
- default pool 也叫 implicit pool，不需要显式创建，不能销毁。
- default pool 不支持 IPC。

### `cudaDeviceSetMemPool` / `cudaDeviceGetMemPool`

**用途**

设置或查询某个 device 的 current memory pool。未显式指定 pool 的 `cudaMallocAsync` 会从 stream device 的 current pool 分配。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetMemPool(
    int device,
    cudaMemPool_t memPool);

extern __host__ cudaError_t CUDARTAPI cudaDeviceGetMemPool(
    cudaMemPool_t* memPool,
    int device);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `device` | `int` | 要设置或查询 current pool 的 CUDA device。 |
| `memPool` | `cudaMemPool_t` / `cudaMemPool_t*` | `Set` 中是输入 pool，`Get` 中是输出参数。 |

**副作用 / 约束**

- 传给 `cudaDeviceSetMemPool` 的 pool 必须 local to 指定 device。
- 如果销毁的是某 device 的 current pool，该 device 会回到 default pool。
- 如果只是某次分配想用其它 pool，优先用 `cudaMallocFromPoolAsync`，不要频繁切换 current pool。

### `cudaMemPoolCreate`

**用途**

创建 explicit memory pool。与 default pool 相比，它可以设置 IPC 能力、最大大小、驻留位置、managed pool 等属性。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolCreate(
    cudaMemPool_t* memPool,
    const struct cudaMemPoolProps* poolProps);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t*` | 输出参数，返回新创建的 pool handle。 |
| `poolProps` | `const cudaMemPoolProps*` | 创建属性，必须按目标 pool 类型正确填充并清零保留字段。 |

**返回值**

| 类型 | 含义 |
| --- | --- |
| `cudaError_t` | 成功返回 `cudaSuccess`；失败可能是 `cudaErrorInvalidValue` 或 `cudaErrorNotSupported`。 |

**副作用 / 约束**

- `cudaMemHandleTypeNone` 创建不可 IPC 的 pool。
- host pool 不支持 IPC，`handleTypes` 必须为 0。
- managed memory pool 不支持 IPC，`handleTypes` 必须为 `cudaMemHandleTypeNone`。
- `cudaMemLocationTypeHostNumaCurrent` 不能用于 `cudaMemPoolCreate`。

### `cudaMemPoolDestroy`

**用途**

销毁 explicit memory pool。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(
    cudaMemPool_t memPool);
```

**副作用 / 约束**

- 不能销毁 device default pool。
- 如果 pool 中仍有未释放 pointer，或 free 操作尚未完成，`cudaMemPoolDestroy` 会立即返回；相关资源会在没有 outstanding allocation 后自动释放。
- 销毁 current pool 会让对应 device 回到 default pool。

### `cudaMemPoolSetAttribute` / `cudaMemPoolGetAttribute`

**用途**

设置或查询 memory pool 属性，包括 release threshold、reuse policy 和统计信息。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(
    cudaMemPool_t memPool,
    enum cudaMemPoolAttr attr,
    void* value);

extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(
    cudaMemPool_t memPool,
    enum cudaMemPoolAttr attr,
    void* value);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t` | 要修改或查询的 pool。 |
| `attr` | `cudaMemPoolAttr` | 属性枚举。 |
| `value` | `void*` | 指向属性值的指针。具体类型由 `attr` 决定。 |

**注意点**

- `cudaMemPoolAttrReleaseThreshold` 的值类型是 `cuuint64_t`。
- reuse policy 的值类型是 `int`，0 表示禁用，非 0 表示启用。
- `ReservedMemHigh` / `UsedMemHigh` 只能设置为 0，用于重置高水位统计。
- 当前统计属性只是 allocator 视角，不等价于整张 GPU 的显存占用。

### `cudaMemPoolTrimTo`

**用途**

尝试把 pool 缓存的物理内存归还给 OS / driver，直到 pool reserved memory 小于某个目标，或者没有更多可安全释放的 backing allocation。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(
    cudaMemPool_t memPool,
    size_t minBytesToKeep);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t` | 要 trim 的 pool。 |
| `minBytesToKeep` | `size_t` | trim 后希望至少保留的 reserved bytes。 |

**注意点**

- 未释放 allocation 一定是 outstanding，不能被 trim。
- 已经 `cudaFreeAsync` 但 host 还没有通过 synchronize 观察到完成的释放，也可能仍被视为 outstanding。
- IPC export / import pool 当前通常不支持向 OS 释放物理块，`cudaMemPoolTrimTo` 对这类 pool 可能无效。

### `cudaMemPoolSetAccess` / `cudaMemPoolGetAccess`

**用途**

设置或查询其它 location 对 pool allocation 的访问权限，主要服务 multi-GPU 和 imported pool 场景。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAccess(
    cudaMemPool_t memPool,
    const struct cudaMemAccessDesc* descList,
    size_t count);

extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAccess(
    enum cudaMemAccessFlags* flags,
    cudaMemPool_t memPool,
    struct cudaMemLocation* location);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t` | 被修改或查询的 pool。 |
| `descList` | `const cudaMemAccessDesc*` | 访问权限描述数组，每个元素描述一个 location。 |
| `count` | `size_t` | `descList` 中元素数量。 |
| `flags` | `cudaMemAccessFlags*` | 输出参数，返回某 location 的访问权限。 |
| `location` | `cudaMemLocation*` | 要查询的访问方位置。 |

**副作用 / 约束**

- 默认 pool allocation 只保证 resident device 可访问，这个访问不能撤销。
- 给其它 GPU 开权限前应先确认 peer access 能力。
- 频繁改变 pool access 不推荐；一旦某 GPU 需要访问某 pool，通常应在 pool 生命周期内保持该权限。

### CUDA 13.3 的 location/type 级 pool API

CUDA 13.3 还提供了更通用的 pool 查询/设置 API：不只按 device 管理 pool，也可以按 `cudaMemLocation + cudaMemAllocationType` 管理。

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(
    cudaMemPool_t* memPool,
    struct cudaMemLocation* location,
    enum cudaMemAllocationType type);

extern __host__ cudaError_t CUDARTAPI cudaMemGetMemPool(
    cudaMemPool_t* memPool,
    struct cudaMemLocation* location,
    enum cudaMemAllocationType type);

extern __host__ cudaError_t CUDARTAPI cudaMemSetMemPool(
    struct cudaMemLocation* location,
    enum cudaMemAllocationType type,
    cudaMemPool_t memPool);
```

**参数**

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `memPool` | `cudaMemPool_t*` / `cudaMemPool_t` | 查询时是输出参数，设置时是输入 pool。 |
| `location` | `cudaMemLocation*` | 目标位置，例如 device、host、host NUMA node。 |
| `type` | `cudaMemAllocationType` | pool allocation 类型，例如 pinned 或 managed。 |

**合法组合**

| allocation type | 合法 location type | 含义 |
| --- | --- | --- |
| `cudaMemAllocationTypePinned` | `cudaMemLocationTypeDevice` | device 上的 stream-ordered device allocation pool。等价覆盖传统 device mempool 场景。 |
| `cudaMemAllocationTypePinned` | `cudaMemLocationTypeHost` | host memory pool。 |
| `cudaMemAllocationTypePinned` | `cudaMemLocationTypeHostNuma` | 指定 NUMA node 的 host memory pool。 |
| `cudaMemAllocationTypeManaged` | `cudaMemLocationTypeDevice` | managed pool，以某 GPU 为 preferred location。 |
| `cudaMemAllocationTypeManaged` | `cudaMemLocationTypeHost` / `cudaMemLocationTypeHostNuma` | managed pool，以 host / host NUMA node 为 preferred location。 |
| `cudaMemAllocationTypeManaged` | `cudaMemLocationTypeNone` | managed pool，不设置 preferred location。 |

**注意点**

- 当 `location.type == cudaMemLocationTypeDevice` 且 `type == cudaMemAllocationTypePinned` 时，`cudaMemSetMemPool` 等价于对 `location.id` 调用 `cudaDeviceSetMemPool`。
- 设置 current pool 时，`location` 和 `type` 必须与 pool 自身属性匹配，否则返回 `cudaErrorInvalidValue`。
- 如果只是一次性从某个 pool 分配，仍建议用 `cudaMallocFromPoolAsync`，不要为了单次分配改变 current pool。

### IPC API 速览

**用途**

让一个进程创建 IPC-capable pool，并把 pool handle 和具体 allocation 分享给其它进程。

**原型**

```cpp
extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* shareableHandle,
    cudaMemPool_t memPool,
    enum cudaMemAllocationHandleType handleType,
    unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* memPool,
    void* shareableHandle,
    enum cudaMemAllocationHandleType handleType,
    unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportPointer(
    struct cudaMemPoolPtrExportData* exportData,
    void* ptr);

extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportPointer(
    void** ptr,
    cudaMemPool_t memPool,
    struct cudaMemPoolPtrExportData* exportData);
```

**接口分工**

| 接口 | 作用 |
| --- | --- |
| `cudaMemPoolExportToShareableHandle` | 把 IPC-capable pool 导出为 OS-native handle，例如 POSIX fd。 |
| `cudaMemPoolImportFromShareableHandle` | 在另一个进程中用 OS handle 创建 imported pool。 |
| `cudaMemPoolExportPointer` | 导出某个 pool allocation 的 opaque export data。 |
| `cudaMemPoolImportPointer` | 在导入进程中用 imported pool 和 export data 得到可访问指针。 |

**约束**

- imported pool 不能用于 `cudaDeviceSetMemPool` 或 `cudaMallocFromPoolAsync`。
- imported pool 不继承导出进程设置的 access，需要导入进程自己调用 `cudaMemPoolSetAccess`。
- imported pointer 必须在导入进程先释放，导出进程才能释放原始 pointer。
- allocation 可以先 export/import，但访问必须等导出进程中 allocation 操作完成，通常用 IPC event 同步。

## 使用示例

### 同一 stream 分配、使用、释放

```cpp
#include <cuda_runtime.h>

#include <cstddef>

/**
 * @brief 对向量执行简单缩放，演示 stream-ordered allocation 的使用区间。
 *
 * @param data device 指针，指向长度至少为 n 的 float 数组。
 * @param scale 缩放系数。
 * @param n 元素数量。
 */
__global__ void scaleKernel(float* __restrict__ data, float scale, int n) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= n) {
        return;
    }
    data[global_idx] *= scale;
}

/**
 * @brief 在同一个 stream 内完成 async allocate、kernel 使用和 async free。
 *
 * @param n 元素数量。
 * @param stream 建立分配、使用和释放顺序的 CUDA stream。
 */
void run_single_stream_allocation(int n, cudaStream_t stream) {
    float* data = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    CHECK_CUDA(cudaMallocAsync(&data, bytes, stream));
    CHECK_CUDA(cudaMemsetAsync(data, 0, bytes, stream));

    constexpr int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    scaleKernel<<<grid_size, block_size, 0, stream>>>(data, 2.0f, n);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFreeAsync(data, stream));
}
```

这个例子里不需要额外 event，因为 allocation、memset、kernel、free 都在同一个 stream 中，天然满足 stream order。

### 跨 stream 使用和释放

```cpp
/**
 * @brief 演示跨 stream 使用 async allocation 时需要显式事件依赖。
 *
 * @param n 元素数量。
 * @param producer_stream 负责分配的 stream。
 * @param consumer_stream 负责使用和释放的 stream。
 */
void run_cross_stream_allocation(int n,
                                 cudaStream_t producer_stream,
                                 cudaStream_t consumer_stream) {
    float* data = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    cudaEvent_t allocation_ready = nullptr;
    CHECK_CUDA(cudaEventCreateWithFlags(&allocation_ready, cudaEventDisableTiming));

    CHECK_CUDA(cudaMallocAsync(&data, bytes, producer_stream));
    CHECK_CUDA(cudaEventRecord(allocation_ready, producer_stream));

    // consumer_stream 必须等待 allocation 在 producer_stream 中完成。
    CHECK_CUDA(cudaStreamWaitEvent(consumer_stream, allocation_ready, 0));

    constexpr int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    scaleKernel<<<grid_size, block_size, 0, consumer_stream>>>(data, 3.0f, n);
    CHECK_CUDA(cudaGetLastError());

    // free 也放在 consumer_stream 中，确保发生在 kernel 使用之后。
    CHECK_CUDA(cudaFreeAsync(data, consumer_stream));

    CHECK_CUDA(cudaEventDestroy(allocation_ready));
}
```

如果跨 stream 使用 allocation，却没有 `cudaStreamWaitEvent` 或其它依赖，行为是 undefined behavior。这里不是“性能可能差”，而是语义上就不成立。

### 显式 pool 和 RAII 封装

```cpp
#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

/**
 * @brief 独占持有一个 explicit CUDA memory pool。
 *
 * 这个类只管理 pool handle 的生命周期，不管理从 pool 中分配出的指针。
 * 销毁 pool 时，CUDA 会在 outstanding allocation 都结束后释放相关资源。
 */
class CudaMemoryPool {
public:
    /**
     * @brief 在指定 device 上创建一个普通 device memory pool。
     *
     * @param device_id CUDA device ordinal。
     */
    explicit CudaMemoryPool(int device_id) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device_id;
        props.maxSize = 0;
        props.usage = 0;

        CHECK_CUDA(cudaMemPoolCreate(&mPool, &props));
    }

    /**
     * @brief 销毁 explicit pool。
     */
    ~CudaMemoryPool() {
        if (mPool != nullptr) {
            cudaMemPoolDestroy(mPool);
        }
    }

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    /**
     * @brief 转移 pool 句柄所有权。
     *
     * @param other 被移动的 pool 对象。
     */
    CudaMemoryPool(CudaMemoryPool&& other) noexcept
        : mPool(std::exchange(other.mPool, nullptr)) {}

    /**
     * @brief 转移 pool 句柄所有权。
     *
     * @param other 被移动的 pool 对象。
     * @return 当前对象引用。
     */
    CudaMemoryPool& operator=(CudaMemoryPool&& other) noexcept {
        if (this != &other) {
            if (mPool != nullptr) {
                cudaMemPoolDestroy(mPool);
            }
            mPool = std::exchange(other.mPool, nullptr);
        }
        return *this;
    }

    /**
     * @brief 返回底层 cudaMemPool_t 句柄。
     *
     * @return 当前对象持有的 CUDA memory pool。
     */
    cudaMemPool_t get() const noexcept {
        return mPool;
    }

private:
    cudaMemPool_t mPool = nullptr;
};

/**
 * @brief 从显式 pool 中分配一段 stream-ordered device buffer。
 *
 * @param pool CUDA memory pool。
 * @param bytes 分配字节数。
 * @param stream 建立分配顺序的 stream。
 * @return 新分配的 device pointer。
 */
void* allocate_from_pool(cudaMemPool_t pool, std::size_t bytes, cudaStream_t stream) {
    void* ptr = nullptr;
    CHECK_CUDA(cudaMallocFromPoolAsync(&ptr, bytes, pool, stream));
    return ptr;
}
```

这个 RAII 类只负责 pool 本身。实际 allocation 的释放仍然应该通过 `cudaFreeAsync(ptr, stream)` 表达清楚的使用结束点。

### 设置 release threshold 和读取统计

```cpp
#include <cuda_runtime.h>

#include <limits>

/**
 * @brief memory pool 使用统计。
 */
struct PoolUsageStats {
    cuuint64_t reserved = 0;
    cuuint64_t reserved_high = 0;
    cuuint64_t used = 0;
    cuuint64_t used_high = 0;
};

/**
 * @brief 设置 memory pool 的 release threshold。
 *
 * @param pool CUDA memory pool。
 * @param threshold_bytes pool 在同步点前允许保留的 reserved memory 字节数。
 */
void set_release_threshold(cudaMemPool_t pool, cuuint64_t threshold_bytes) {
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
                                       cudaMemPoolAttrReleaseThreshold,
                                       &threshold_bytes));
}

/**
 * @brief 查询 memory pool 的 reserved / used 统计。
 *
 * @param pool CUDA memory pool。
 * @return 当前统计值和高水位。
 */
PoolUsageStats get_pool_usage_stats(cudaMemPool_t pool) {
    PoolUsageStats stats{};
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
                                       cudaMemPoolAttrReservedMemCurrent,
                                       &stats.reserved));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
                                       cudaMemPoolAttrReservedMemHigh,
                                       &stats.reserved_high));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
                                       cudaMemPoolAttrUsedMemCurrent,
                                       &stats.used));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
                                       cudaMemPoolAttrUsedMemHigh,
                                       &stats.used_high));
    return stats;
}

/**
 * @brief 重置 memory pool 的统计高水位。
 *
 * @param pool CUDA memory pool。
 */
void reset_pool_high_watermarks(cudaMemPool_t pool) {
    cuuint64_t zero = 0;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
                                       cudaMemPoolAttrReservedMemHigh,
                                       &zero));
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
                                       cudaMemPoolAttrUsedMemHigh,
                                       &zero));
}
```

如果希望服务在 warmup 后长期复用 pool 内存，可以把 `cudaMemPoolAttrReleaseThreshold` 设得较高，减少同步点后频繁 shrink。若某个阶段结束后明确不再需要大量显存，可以 `cudaStreamSynchronize` 后调用 `cudaMemPoolTrimTo(pool, 0)`。

### 多 GPU pool access

```cpp
/**
 * @brief 允许 accessing_device 读写 resident_device 上 pool 分配出的内存。
 *
 * @param pool resident_device 上的 CUDA memory pool。
 * @param resident_device pool allocation 驻留的 GPU。
 * @param accessing_device 需要访问该 pool allocation 的 GPU。
 * @return CUDA Runtime API 返回值。
 */
cudaError_t enable_pool_access_on_peer(cudaMemPool_t pool,
                                       int resident_device,
                                       int accessing_device) {
    int can_access_peer = 0;
    cudaError_t status = cudaDeviceCanAccessPeer(&can_access_peer,
                                                 accessing_device,
                                                 resident_device);
    if (status != cudaSuccess) {
        return status;
    }
    if (can_access_peer == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    cudaMemAccessDesc access_desc{};
    access_desc.location.type = cudaMemLocationTypeDevice;
    access_desc.location.id = accessing_device;
    access_desc.flags = cudaMemAccessFlagsProtReadWrite;

    return cudaMemPoolSetAccess(pool, &access_desc, 1);
}
```

注意：pool access 不是 per-allocation 设置，而是作用于整个 pool。不要在 hot path 上频繁切换访问权限。

### 创建 managed memory pool

```cpp
/**
 * @brief 创建一个没有 preferred location 的 managed memory pool。
 *
 * @return 新创建的 managed memory pool。
 */
cudaMemPool_t create_managed_pool_without_preferred_location() {
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypeManaged;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeNone;
    props.location.id = 0;
    props.maxSize = 0;
    props.usage = 0;

    cudaMemPool_t pool = nullptr;
    CHECK_CUDA(cudaMemPoolCreate(&pool, &props));
    return pool;
}
```

managed memory pool 仍然服务 stream-ordered allocation，但 allocation 语义是 managed memory。它不是普通 device pool 的“更快版本”，而是把 stream-ordered allocator 和 Unified Memory 结合起来；是否适合要看 CPU/GPU 访问模式。

## 复用策略怎么理解

stream-ordered allocator 的复用策略可以理解为三层越来越激进的内存复用：

| 策略 | 默认 | 含义 | 代价 |
| --- | --- | --- | --- |
| `cudaMemPoolReuseFollowEventDependencies` | 启用 | 只要 stream 依赖证明 free 发生在新 allocation 之前，就允许跨 stream 复用。 | 语义清楚，推荐保留。 |
| `cudaMemPoolReuseAllowOpportunistic` | 启用 | 如果 allocator 观察到某个 free 已经完成，即使没有显式依赖，也可以复用。 | 可能引入运行间差异，分配地址/复用行为更不稳定。 |
| `cudaMemPoolReuseAllowInternalDependencies` | 启用 | allocator 可自动插入依赖来建立安全复用顺序。 | 可能引入意料之外的 stream 串行化。 |

如果你在做 benchmark、debug 地址复用、或者希望调度依赖完全显式，可以考虑关闭 opportunistic / internal dependency：

```cpp
/**
 * @brief 禁用可能带来非确定性复用行为的 pool reuse policy。
 *
 * @param pool CUDA memory pool。
 */
void disable_nondeterministic_reuse(cudaMemPool_t pool) {
    int disabled = 0;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
                                       cudaMemPoolReuseAllowOpportunistic,
                                       &disabled));
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
                                       cudaMemPoolReuseAllowInternalDependencies,
                                       &disabled));
}
```

## 和 CUDA Graph 的关系

`cudaMallocAsync` / `cudaFreeAsync` 在 stream capture 中会变成 graph allocation / free node。此时 allocation 的所有权归 graph，而不是 memory pool。pool 的属性会用于设置 node 的创建参数。

需要注意：

- capture 时可以把 async allocation 纳入 graph，适合固定拓扑的 pipeline。
- graph allocation 的生命周期跟 graph 执行和 graph 内存节点有关，不要简单套用普通 pool allocation 的 mental model。
- NVIDIA 文档特别提到，`cudaGraphAddMemsetNode` 不适用于 stream-ordered allocator 创建的内存，但对这类 allocation 的 memset 可以通过 stream capture 捕获。

## 常见坑点

- **跨 stream 使用必须建立依赖**。allocation stream 和使用 stream 不同时，需要 event、stream wait 或其它同步保证。
- **free 后不要再查 pointer attributes**。`cudaFreeAsync` 返回后，`cudaPointerGetAttributes` 该指针是 undefined behavior。
- **`cudaFree` 释放 async allocation 不会自动同步**。如果 `cudaMallocAsync` 后用 `cudaFree`，应用必须先确认所有 GPU work 完成。
- **pool current 设置不要频繁切换**。一次性指定 pool 时用 `cudaMallocFromPoolAsync` 更清晰。
- **release threshold 影响显存占用曲线**。阈值太低可能频繁归还 OS，阈值太高会保留更大 footprint。
- **mempool access 不跟随 peer access API**。`cudaDeviceEnablePeerAccess` 不会自动改变 pool accessibility。
- **IPC import pool 不能分配新内存**。它只能 import allocation，不能作为 current pool 或传给 `cudaMallocFromPoolAsync`。
- **host / host NUMA / managed pool 是高级用法**。先确认设备属性支持，再确认访问模式真的需要它们。

## 选型建议

| 场景 | 建议 |
| --- | --- |
| 程序初始化时分配几块长期 device buffer | `cudaMalloc` 足够，简单清晰。 |
| 每个 iteration 都有临时 device buffer | 优先考虑 `cudaMallocAsync` / `cudaFreeAsync`。 |
| 多个库共享同一 GPU 内存缓存策略 | 创建 explicit pool，并把 pool handle 传给库。 |
| 想减少同步点后显存反复释放再申请 | 设置较高 `cudaMemPoolAttrReleaseThreshold`。 |
| 想阶段结束后释放 footprint | 同步后调用 `cudaMemPoolTrimTo(pool, minBytesToKeep)`。 |
| 多 GPU 访问同一 pool allocation | 用 `cudaMemPoolSetAccess`，不要只依赖 `cudaDeviceEnablePeerAccess`。 |
| 进程间共享 GPU allocation | 创建 IPC-capable pool，先导出 pool handle，再导出 allocation data。 |
| CPU/GPU 混合访问且希望 managed semantics | 考虑 managed memory pool，但仍要关注 page fault 和迁移。 |

一句话总结：**stream-ordered allocator 把“分配/释放”也变成 stream 中可排序的工作，从而让内存生命周期跟 GPU 调度语义对齐。**
