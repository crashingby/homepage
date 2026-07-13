---
title: C++ 对齐与填充笔记
date: 2026-07-13
tags: [C++, 内存布局, 对齐, padding, ABI]
summary: 从字节、地址和对象表示开始，整理 C/C++ 中的对齐要求、结构体填充、C 语言控制方式，以及 C++ 语言和标准库提供的对齐工具。
---

# C++ 对齐与填充笔记

对齐（alignment）和填充（padding）是 C/C++ 内存布局里很基础但很容易被忽略的部分。

它们回答的是这些问题：

- 一个 `int` 为什么通常要放在 4 字节边界上？
- 为什么 `struct { char c; int i; }` 的大小不是 5？
- `alignas(64)` 到底改变了什么？
- `#pragma pack(1)` 为什么可能让程序变慢，甚至在某些平台上出问题？
- C++ 标准库里 `std::align`、`std::max_align_t`、对齐版 `operator new` 分别解决什么？

先记住一句话：

> 对齐是对象地址的约束，填充是编译器为了满足这些约束而插入的无名字节。

## 从内存字节开始

现代机器上，内存通常可以想象成一排连续编号的 byte（字节）：

```text
address:  0x1000  0x1001  0x1002  0x1003  0x1004  0x1005  ...
byte:     [  ?  ][  ?  ][  ?  ][  ?  ][  ?  ][  ?  ]
```

C/C++ 里最小的可寻址单元是 byte。标准里的 byte 不强制必须是 8 bit，而是由 `CHAR_BIT` 决定；现代常见平台通常是 8 bit。

```cpp
#include <climits>

static_assert(CHAR_BIT == 8);  // 常见平台成立，但不是 C++ 标准强制。
```

一个对象会占用若干 byte。比如常见 64-bit 平台上：

```cpp
sizeof(char)   == 1
sizeof(int)    == 4
sizeof(double) == 8
```

对象在内存中的字节序列叫 object representation（对象表示）。C++ 允许用 `char`、`unsigned char` 或 `std::byte` 观察对象的底层字节：

```cpp
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

int x = 0x12345678;
auto bytes = std::bit_cast<std::array<std::byte, sizeof(int)>>(x);
```

这里看到的 byte 顺序还会受 endian（大小端）影响。对齐讨论通常不关心 byte 内部值是什么，而关心对象的起始地址能不能满足硬件和 ABI 要求。

## 什么是对齐

每个对象类型都有一个 alignment requirement（对齐要求）。

```cpp
alignof(char)   // 通常是 1
alignof(int)    // 通常是 4
alignof(double) // 通常是 8
```

如果一个类型 `T` 的对齐要求是 `A`，那么 `T` 对象的起始地址通常必须是 `A` 的整数倍：

$$
\text{address}(obj) \bmod A = 0
$$

例如 `int` 的对齐为 4，那么合法地址通常是：

```text
0x1000, 0x1004, 0x1008, ...
```

不理想地址是：

```text
0x1001, 0x1002, 0x1003, ...
```

为什么要对齐？

- CPU 读写对齐地址通常更快。
- 某些架构不支持未对齐访问，直接触发异常。
- ABI 规定了函数调用、结构体布局、数组元素布局等规则，编译器必须遵守。
- SIMD、cache line、DMA、GPU 等场景会有更强对齐偏好。

## 什么是填充

结构体成员不是简单地一个接一个紧贴着放。编译器会在成员之间或结构体末尾插入 padding byte（填充字节），保证每个成员和数组元素都满足对齐要求。

考虑：

```cpp
struct A {
    char c;
    int  i;
};
```

常见平台：

```cpp
sizeof(char)  == 1
alignof(char) == 1
sizeof(int)   == 4
alignof(int)  == 4
```

布局通常是：

```text
offset 0: char c
offset 1: padding
offset 2: padding
offset 3: padding
offset 4: int i
offset 5: int i
offset 6: int i
offset 7: int i
```

所以：

```cpp
sizeof(A)  == 8
alignof(A) == 4
```

### 结构体布局公式

这一节不要一上来就背公式。先把编译器想成拿着一个 `offset` 游标，从结构体开头往后摆成员。

规则只有两句：

- 每放一个成员之前，先把当前 `offset` 补到这个成员对齐要求的整数倍。
- 成员放完以后，`offset` 往后移动 `sizeof(member)`。

把“补到对齐边界”写成函数，就是：

$$
\operatorname{align\_up}(x, A) = \left\lceil \frac{x}{A} \right\rceil \cdot A
$$

它的意思是：把偏移 `x` 向上补到 `A` 的整数倍。

比如按 4 字节对齐：

| `x` | `align_up(x, 4)` | 解释 |
| --- | --- | --- |
| 0 | 0 | 已经在 4 字节边界上。 |
| 1 | 4 | 需要补 3 个 byte。 |
| 2 | 4 | 需要补 2 个 byte。 |
| 3 | 4 | 需要补 1 个 byte。 |
| 4 | 4 | 已经对齐。 |
| 5 | 8 | 需要补到下一个 4 字节边界。 |

再看这个结构体：

```cpp
struct A {
    char c;
    int  i;
};
```

常见平台上：

```cpp
sizeof(char)  == 1
alignof(char) == 1
sizeof(int)   == 4
alignof(int)  == 4
```

编译器摆放成员的过程可以写成这张表：

| 步骤 | 当前 `offset` | 成员 | 成员对齐 | 对齐后 offset | 插入 padding | 放完后 offset |
| --- | --- | --- | --- | --- | --- | --- |
| 开始 | 0 | - | - | - | - | 0 |
| 放 `c` | 0 | `char c` | 1 | `align_up(0, 1) = 0` | 0 byte | `0 + sizeof(char) = 1` |
| 放 `i` | 1 | `int i` | 4 | `align_up(1, 4) = 4` | 3 byte | `4 + sizeof(int) = 8` |

所以布局是：

```text
offset 0: char c
offset 1: padding
offset 2: padding
offset 3: padding
offset 4: int i
offset 5: int i
offset 6: int i
offset 7: int i
```

此时成员已经摆完，最后还要决定整个结构体本身的对齐。

结构体整体对齐通常是所有成员对齐要求的最大值，除非你用 `alignas` 或 packing 改变它：

$$
\operatorname{alignof}(S) = \max_i \operatorname{alignof}(m_i)
$$

对 `A` 来说：

$$
\operatorname{alignof}(A) = \max(1,4) = 4
$$

结构体大小也必须补到 `alignof(A)` 的整数倍。这个补齐是为了让数组里的每个元素都能正确对齐。

$$
\operatorname{sizeof}(S) = \operatorname{align\_up}(\operatorname{last\_end}, \operatorname{alignof}(S))
$$

对 `A` 来说，最后一个成员结束在 offset 8：

$$
\operatorname{sizeof}(A) = \operatorname{align\_up}(8, 4) = 8
$$

所以：

```cpp
sizeof(A)  == 8
alignof(A) == 4
```

如果最后补齐时插入了 padding，这部分叫 tail padding（尾部填充）。例如：

```cpp
struct B {
    int  i;
    char c;
};
```

常见布局是：

```text
offset 0-3: int i
offset 4:   char c
offset 5-7: tail padding
```

`B` 的成员实际只用到 5 byte，但：

$$
\operatorname{sizeof}(B) = \operatorname{align\_up}(5, 4) = 8
$$

原因是数组必须成立：

```cpp
B arr[2];
```

如果 `sizeof(B)` 是 5，那么 `arr[1]` 会从 `arr[0]` 后面 5 byte 开始，地址可能不是 4 的倍数，里面的 `int i` 就不对齐了。因此编译器把 `sizeof(B)` 补到 8。

## 成员顺序会影响大小

看两个结构体：

```cpp
struct Bad {
    char   a;
    double b;
    int    c;
};

struct Good {
    double b;
    int    c;
    char   a;
};
```

常见平台：

```cpp
alignof(double) == 8
alignof(int)    == 4
alignof(char)   == 1
```

`Bad` 通常布局为：

```text
offset 0:  char a
offset 1-7: padding
offset 8-15: double b
offset 16-19: int c
offset 20-23: tail padding
```

所以：

```cpp
sizeof(Bad) == 24
```

`Good` 通常布局为：

```text
offset 0-7:   double b
offset 8-11:  int c
offset 12:    char a
offset 13-15: tail padding
```

所以：

```cpp
sizeof(Good) == 16
```

经验规则：**把对齐要求大的成员放前面，通常能减少 padding**。

但这不是绝对优化原则。公开 ABI、序列化格式、硬件寄存器布局、协议字段顺序，不能为了省几个 byte 随便改成员顺序。

## 观察布局：`sizeof`、`alignof`、`offsetof`

最常用的三个工具：

```cpp
#include <cstddef>
#include <cstdio>

struct A {
    char c;
    int  i;
};

int main()
{
    std::printf("sizeof(A)  = %zu\n", sizeof(A));
    std::printf("alignof(A) = %zu\n", alignof(A));
    std::printf("offsetof(c) = %zu\n", offsetof(A, c));
    std::printf("offsetof(i) = %zu\n", offsetof(A, i));
}
```

`offsetof` 的用途是查询成员相对结构体起点的 byte offset。它要求类型满足标准布局类型相关约束；不要把它随便用在复杂 C++ 类层次上。

## C 语言怎么控制对齐

C 语言里的对齐控制可以分成两类：

- 标准 C 提供的 `_Alignof`、`_Alignas`。
- 编译器扩展提供的 `#pragma pack`、`__attribute__((packed))`、`__attribute__((aligned))`、`__declspec(align)` 等。

### `_Alignof`

C11 引入 `_Alignof`，用于查询类型的对齐要求。

```c
#include <stdio.h>

int main(void)
{
    printf("_Alignof(int) = %zu\n", _Alignof(int));
    printf("_Alignof(double) = %zu\n", _Alignof(double));
}
```

C11 的 `<stdalign.h>` 还提供了宏：

```c
#include <stdalign.h>

alignof(int)
```

在 C++ 里，`alignof` 是语言关键字，不需要 `<stdalign.h>`。

### `_Alignas`

`_Alignas` 用于提高对象或类型的对齐要求。

```c
#include <stdalign.h>

struct alignas(64) CacheLineCounter {
    long long value;
};
```

在 C11 中也可以直接写：

```c
struct _Alignas(64) CacheLineCounter {
    long long value;
};
```

它表达的是：这个类型或对象至少要按 64 字节对齐。

注意：标准对齐工具通常用于**提高对齐**，不要指望它安全地降低一个类型本来需要的对齐。

### `#pragma pack`

`#pragma pack` 是常见但非标准的编译器扩展。它会限制结构体成员的最大对齐。

```c
#include <stddef.h>
#include <stdio.h>

#pragma pack(push, 1)
struct PackedHeader {
    char c;
    int  i;
};
#pragma pack(pop)

int main(void)
{
    printf("sizeof(PackedHeader) = %zu\n", sizeof(struct PackedHeader));
    printf("offsetof(i) = %zu\n", offsetof(struct PackedHeader, i));
}
```

在常见编译器上，`PackedHeader::i` 可能放在 offset 1，结构体大小可能变成 5。

这适合二进制协议、文件格式、网络包头等场景，但有明显代价：

- 成员可能未对齐，访问变慢。
- 某些架构上未对齐访问可能触发异常。
- 对 packed 成员取地址，再当作普通 `int*` 使用，可能产生未定义行为或编译器警告。

### GCC / Clang attribute

GCC / Clang 常见写法：

```c
struct __attribute__((packed)) Packed {
    char c;
    int  i;
};

struct __attribute__((aligned(64))) Aligned {
    int value;
};
```

也可以组合：

```c
struct __attribute__((packed, aligned(4))) Header {
    char c;
    int  i;
};
```

含义要分清：

| 写法 | 含义 |
| --- | --- |
| `packed` | 尽量减少成员之间 padding，可能降低成员对齐。 |
| `aligned(N)` | 提高类型或对象的对齐要求。 |

### MSVC `__declspec(align)`

MSVC 常见写法：

```c
__declspec(align(64))
struct AlignedCounter {
    long long value;
};
```

packing 则常用：

```c
#pragma pack(push, 1)
struct Header {
    char c;
    int  i;
};
#pragma pack(pop)
```

### C 语言里的内存分配对齐

标准 `malloc` 返回的指针会满足基本类型的常规最大对齐需求。C11 还引入了 `aligned_alloc`：

```c
#include <stdlib.h>

void* p = aligned_alloc(64, 1024);
free(p);
```

注意：C11 `aligned_alloc(alignment, size)` 要求 `size` 是 `alignment` 的整数倍。

POSIX 里常见：

```c
#include <stdlib.h>

void* p = NULL;
int rc = posix_memalign(&p, 64, 1024);
free(p);
```

`posix_memalign` 要求 alignment 是 `sizeof(void*)` 的倍数，并且是 2 的幂。

## C++ 语言层怎么控制对齐

### `alignof`

`alignof(T)` 查询类型 `T` 的对齐要求。

```cpp
#include <cstddef>
#include <cstdio>

struct A {
    char c;
    int  i;
};

int main()
{
    std::printf("alignof(char) = %zu\n", alignof(char));
    std::printf("alignof(int)  = %zu\n", alignof(int));
    std::printf("alignof(A)    = %zu\n", alignof(A));
}
```

### `alignas`

`alignas` 用于指定更强对齐。

```cpp
struct alignas(64) CacheLineCounter {
    long long value;
};

static_assert(alignof(CacheLineCounter) == 64);
```

也可以修饰变量：

```cpp
alignas(32) float buffer[8];
```

常见用途：

- SIMD 数据需要 16 / 32 / 64 字节对齐。
- cache line 隔离，减少 false sharing。
- 与硬件、DMA、GPU、文件格式交互。

注意点：

- `alignas` 通常用于提高对齐，不应试图把类型对齐降到低于自然要求。
- `alignas(0)` 会被忽略。
- 多个 `alignas` 同时出现时，有效对齐通常取最严格的那个。

## C++ 标准库怎么做

C++ 标准库提供的是一组类型化工具，帮助你查询、分配、调整和假设对齐。

### `std::max_align_t`

**用途**

`std::max_align_t` 是一个对齐要求至少和所有标量类型一样严格的类型。它常用于实现通用内存池时，保证一块默认内存能放大多数普通对象。

**头文件**

```cpp
#include <cstddef>
```

**示例**

```cpp
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("alignof(std::max_align_t) = %zu\n", alignof(std::max_align_t));
}
```

**注意点**

`std::max_align_t` 不代表所有可能类型的最大对齐。用户可以定义 over-aligned type（过对齐类型）：

```cpp
struct alignas(64) OverAligned {
    int x;
};

static_assert(alignof(OverAligned) > alignof(std::max_align_t));
```

### `std::align`

**用途**

`std::align` 在一段原始 buffer 中找出一个满足对齐和大小要求的地址，并更新指针和剩余空间。

**头文件**

```cpp
#include <memory>
```

**原型**

```cpp
void* std::align(std::size_t alignment,
                 std::size_t size,
                 void*& ptr,
                 std::size_t& space);
```

**参数**

| 参数 | 含义 |
| --- | --- |
| `alignment` | 目标对齐，通常是 2 的幂。 |
| `size` | 要放置的对象或内存块大小。 |
| `ptr` | 输入为 buffer 当前起点；成功后更新为对齐后的地址。 |
| `space` | 输入为剩余空间；成功后更新为对齐后还剩多少空间。 |

**返回值**

| 返回值 | 含义 |
| --- | --- |
| 非空指针 | 找到了满足对齐和大小的位置。 |
| `nullptr` | 当前空间不够。 |

**示例**

```cpp
#include <cstddef>
#include <memory>
#include <new>

struct alignas(32) Vec4 {
    float x, y, z, w;
};

int main()
{
    alignas(64) std::byte storage[256];

    void* ptr = storage;
    std::size_t space = sizeof(storage);

    void* aligned = std::align(alignof(Vec4), sizeof(Vec4), ptr, space);
    if (aligned == nullptr) {
        return 1;
    }

    // placement new：在已经对齐的原始内存上构造对象。
    Vec4* v = new (aligned) Vec4{1.0f, 2.0f, 3.0f, 4.0f};

    // 手动调用析构。Vec4 没有资源，这里只是展示生命周期规则。
    v->~Vec4();
}
```

`std::align` 不分配内存，它只是在已有内存里找一个对齐位置。

### 对齐版 `operator new`

C++17 引入对齐版 allocation function，用于支持 over-aligned type。

先看 `std::align_val_t` 是什么。

`std::align_val_t` 定义在 `<new>` 中。它可以理解成一个“专门表示对齐值的类型”，标准库里大致长这样：

```cpp
namespace std {
    enum class align_val_t : std::size_t {};
}
```

它不是 allocator，也不负责分配内存。它的作用是参与函数重载，让编译器能区分下面两种申请：

```cpp
#include <new>

void* operator new(std::size_t size);
void* operator new(std::size_t size, std::align_val_t alignment);
```

两者的含义不同：

| 重载 | 含义 |
| --- | --- |
| `operator new(size)` | 申请一块普通对齐要求的原始内存。 |
| `operator new(size, std::align_val_t{A})` | 申请一块至少满足 `A` 字节对齐的原始内存。 |

所以 `std::align_val_t{64}` 的意思不是“分配 64 byte”，而是“这次 allocation 要求返回地址至少 64 字节对齐”。

对应的全局 allocation / deallocation 函数原型是：

```cpp
#include <new>

void* operator new(std::size_t size, std::align_val_t alignment);
void operator delete(void* ptr, std::align_val_t alignment) noexcept;
```

正常使用对象时不需要直接调用它们。对于过对齐类型，编译器会自动选择对齐版 `operator new`：

```cpp
struct alignas(64) Node {
    int value;
};

int main()
{
    Node* p = new Node{42};  // 编译器会选择合适的对齐版 operator new。
    delete p;
}
```

这里 `Node` 的对齐要求是 64。如果这个对齐超过默认 `new` 能保证的对齐，表达式：

```cpp
new Node{42}
```

概念上会变成一次带对齐参数的 allocation，再在返回的内存上构造对象：

```cpp
void* raw = ::operator new(sizeof(Node), std::align_val_t{alignof(Node)});
Node* p = new (raw) Node{42};
```

释放时也要匹配：

```cpp
p->~Node();
::operator delete(p, std::align_val_t{alignof(Node)});
```

上面这段是为了说明机制。普通业务代码应该继续写：

```cpp
Node* p = new Node{42};
delete p;
```

编译器会负责选择匹配的 allocation 和 deallocation 函数。

把它拆开看：

| 名字 | 角色 |
| --- | --- |
| `std::size_t size` | 需要申请多少 byte 的原始存储。 |
| `std::align_val_t alignment` | 这块原始存储的起始地址至少要满足多少字节对齐。 |
| `operator new` | 只申请原始内存，不构造对象。 |
| placement new | 在已有原始内存上构造对象。 |
| `operator delete` | 只释放原始内存，不析构对象。 |

`std::align_val_t` 做成独立类型，而不是直接用 `std::size_t`，是为了让重载语义清楚：

```cpp
// 一个 size_t：申请 64 byte。
void* a = ::operator new(64);

// 一个 size_t 加一个 align_val_t：申请 64 byte，并要求 64 字节对齐。
void* b = ::operator new(64, std::align_val_t{64});

::operator delete(a);
::operator delete(b, std::align_val_t{64});
```

如果第二个参数也只是 `std::size_t`，调用点就很容易变成“两个整数”，可读性和重载选择都会变差。

### `std::assume_aligned`

C++20 提供 `std::assume_aligned<N>(ptr)`，告诉编译器某个指针满足 `N` 字节对齐。

**头文件**

```cpp
#include <memory>
```

**示例**

```cpp
#include <memory>

void saxpy(float* x, float* y, int n)
{
    float* ax = std::assume_aligned<32>(x);
    float* ay = std::assume_aligned<32>(y);

    for (int i = 0; i < n; ++i) {
        ay[i] = 2.0f * ax[i] + ay[i];
    }
}
```

注意：这是给编译器的承诺，不是运行时检查。如果 `x` 或 `y` 实际没有 32 字节对齐，行为可能是未定义的。它适合在你已经通过分配器、API 契约或手动检查保证对齐之后使用。

### `std::hardware_destructive_interference_size`

C++17 在 `<new>` 中提供两个硬件干涉大小常量：

```cpp
#include <new>

std::hardware_destructive_interference_size
std::hardware_constructive_interference_size
```

常见用途是减少 false sharing（伪共享）：

```cpp
#include <atomic>
#include <new>

struct CounterPair {
    alignas(std::hardware_destructive_interference_size)
    std::atomic<long long> a;

    alignas(std::hardware_destructive_interference_size)
    std::atomic<long long> b;
};
```

`a` 和 `b` 被尽量放到不同 cache line，避免两个线程频繁写不同变量但抢同一 cache line。

### `std::aligned_storage` 和 `std::aligned_union`

旧代码里常见：

```cpp
#include <type_traits>

using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;
```

它用于创建一块有指定大小和对齐的未初始化存储。

不过 `std::aligned_storage` 和 `std::aligned_union` 在 C++23 中已经 deprecated。新代码更推荐：

- 使用 `alignas(T) std::byte storage[sizeof(T)]`。
- 使用 `std::allocator` / polymorphic allocator。
- 使用更明确的对象生命周期工具，如 placement new、`std::construct_at`、`std::destroy_at`。

示例：

```cpp
#include <cstddef>
#include <memory>

struct Widget {
    int x;
};

int main()
{
    alignas(Widget) std::byte storage[sizeof(Widget)];

    Widget* p = std::construct_at(reinterpret_cast<Widget*>(storage), 42);
    std::destroy_at(p);
}
```

更严格地写对象生命周期代码时，还要注意 `std::launder` 等规则。普通业务代码不要轻易手写对象池。

## packed 结构体的坑

packed 结构体最常见的陷阱是：成员本身可能未对齐。

```cpp
#pragma pack(push, 1)
struct Header {
    char c;
    int  value;
};
#pragma pack(pop)
```

`value` 可能位于 offset 1。如果你写：

```cpp
int* p = &header.value;
```

这个指针可能不是 4 字节对齐。某些编译器会警告，某些平台上访问可能有问题。

更稳妥的做法是：对外部二进制格式，用 byte buffer 解析，必要时用 `std::memcpy` 读写字段：

```cpp
#include <array>
#include <cstdint>
#include <cstring>

std::uint32_t read_u32_unaligned(const std::byte* p)
{
    std::uint32_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}
```

`std::memcpy` 处理的是 byte 序列，不要求源地址按 `std::uint32_t` 对齐。之后 `value` 是本地对齐良好的对象。

## 常见建议

- 普通结构体优先让编译器自然对齐，不要默认 `pack(1)`。
- 如果关心结构体大小，先调整成员顺序，再考虑 packing。
- 二进制协议和文件格式不要直接依赖 C++ 结构体布局，除非 ABI、端序、packing、版本都被严格控制。
- 对 SIMD / cache line / DMA / GPU 等场景，用 `alignas` 或专门分配器显式表达对齐契约。
- `std::assume_aligned` 只在你已经证明指针对齐时使用。
- 对齐和对象生命周期是两件事：一块内存对齐了，不代表对象已经构造了。

## 小结

对齐可以用一个公式记住：

$$
\text{address} \bmod \text{alignof}(T) = 0
$$

结构体布局可以用另一个公式记住：

$$
\operatorname{offset}(m_i)
=
\operatorname{align\_up}(\operatorname{previous\_end}, \operatorname{alignof}(m_i))
$$

C 语言主要靠 `_Alignof`、`_Alignas` 和编译器扩展控制布局；C++ 把 `alignof`、`alignas` 做成语言设施，又在标准库里提供了 `std::align`、`std::max_align_t`、对齐版 `operator new`、`std::assume_aligned` 等工具。

真正写代码时，最好把对齐当成一种类型和 API 契约：谁分配内存、保证几字节对齐、谁可以假设这个对齐，都要说清楚。
