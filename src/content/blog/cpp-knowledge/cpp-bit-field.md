---
title: C++ 位域笔记
date: 2026-07-09
tags: [C++, 位运算, 内存布局, bit field]
summary: 整理 C++ 位域的语法、本质、对齐关系、使用场景和常见坑点，并和手写 bit mask 做对比。
---

# C++ 位域笔记

位域（bit field）是 C / C++ 里一个比较底层的结构体成员语法。它允许我们在 `struct` 或 `class` 里声明“只占若干个 bit 的整数成员”。

最简单的例子：

```cpp
struct PacketHeader {
    unsigned int version : 4;  // 只使用 4 bit 表示版本号。
    unsigned int type    : 4;  // 只使用 4 bit 表示消息类型。
    unsigned int length  : 8;  // 只使用 8 bit 表示长度。
};
```

这段代码表达的是：`version`、`type`、`length` 不是普通 `unsigned int` 成员，而是几个被压缩到同一存储单元里的字段。

## 为什么需要位域

在没有位域的时候，如果想把多个小范围状态塞进一个整数里，通常会手写 bit mask：

```cpp
#include <cstdint>

/**
 * @brief 手写位掩码方式保存包头字段。
 */
struct PacketHeaderMask {
    std::uint16_t raw = 0;  // 底层 16 bit 原始数据。
};

/**
 * @brief 设置 4 bit version 字段。
 *
 * @param header 需要修改的包头。
 * @param version 版本号，只保留低 4 bit。
 */
void setVersion(PacketHeaderMask& header, std::uint16_t version)
{
    // 先清除低 4 bit，再写入新的 version。
    header.raw = static_cast<std::uint16_t>((header.raw & ~0x000Fu) | (version & 0x000Fu));
}

/**
 * @brief 读取 4 bit version 字段。
 *
 * @param header 只读包头。
 * @return 低 4 bit 中保存的 version。
 */
std::uint16_t getVersion(const PacketHeaderMask& header)
{
    return static_cast<std::uint16_t>(header.raw & 0x000Fu);
}
```

这种写法可控、明确、适合协议解析和硬件寄存器，但缺点也明显：

- 每个字段都要手写 mask 和 shift，代码容易变吵。
- 字段多了以后，可读性会下降。
- 很容易写错掩码范围，例如少清一位或多移一位。

位域的目标就是把这种“小整数压缩存储”的意图写进类型声明里：

```cpp
struct PacketHeaderBits {
    unsigned int version : 4;  // 0..15。
    unsigned int type    : 4;  // 0..15。
    unsigned int length  : 8;  // 0..255。
};
```

于是访问时看起来就像普通成员：

```cpp
PacketHeaderBits header{};
header.version = 2;
header.type = 7;
header.length = 128;
```

它适合表达：

- 很多字段本身只需要几个 bit。
- 希望结构体语义比手写 bit mask 更直观。
- 内存里有大量对象，压缩字段能明显节省空间。
- 代码主要在同一个编译器 / ABI 下使用，不需要跨平台精确定义二进制布局。

## 基本语法

位域只能作为类或结构体的**非静态数据成员**出现，语法是：

```cpp
struct Name {
    type member_name : width;
};
```

其中：

| 部分 | 含义 |
| --- | --- |
| `type` | 位域的基底类型，通常使用 `unsigned int`、`std::uint32_t` 这类整数类型。 |
| `member_name` | 位域成员名。也可以省略，形成 unnamed bit field（无名位域）。 |
| `width` | 位域宽度，单位是 bit，必须是整数常量表达式。 |

示例：

```cpp
#include <cstdint>

/**
 * @brief 用位域描述一个简单状态字。
 */
struct StatusWord {
    std::uint8_t ready : 1;   // 1 bit：是否就绪。
    std::uint8_t error : 1;   // 1 bit：是否出错。
    std::uint8_t mode  : 2;   // 2 bit：最多表示 4 种模式。
    std::uint8_t code  : 4;   // 4 bit：最多表示 16 个状态码。
};
```

注意：位域宽度限制的是**存储位数**，不是自动帮你做语义校验。比如 `mode : 2` 最多只能保存 2 bit，给它赋值 `7` 时，高位会被截断或产生实现相关行为，实际工程里应该自己保证输入范围。

## 位域宽度

位域宽度决定这个成员占多少个 bit：

```cpp
struct Example {
    unsigned int a : 1;  // 只能保存 0 或 1。
    unsigned int b : 3;  // 可以保存 0..7。
    unsigned int c : 10; // 可以保存 0..1023。
};
```

对于无符号位域，宽度为 `N` 时，通常能表示：

$$
0 \sim 2^N - 1
$$

所以：

| 位域 | 可表示范围 |
| --- | --- |
| `unsigned int x : 1` | `0..1` |
| `unsigned int x : 2` | `0..3` |
| `unsigned int x : 4` | `0..15` |
| `unsigned int x : 8` | `0..255` |

如果使用有符号位域，例如：

```cpp
struct SignedBits {
    int value : 3;
};
```

它的负数表示、溢出行为和具体范围更容易受到实现影响。除非你明确需要有符号小整数，否则位域一般优先使用无符号类型。

## 本质模型

位域可以先按“编译器帮你在一个整数存储单元里分配若干 bit”来理解：

```cpp
struct Flags {
    unsigned int a : 1;
    unsigned int b : 2;
    unsigned int c : 5;
};
```

逻辑上它像这样：

```text
一个 unsigned int 存储单元

bit:   [ c c c c c ][ b b ][ a ]
width:      5         2     1
```

但这里有一个非常重要的点：**C++ 标准不保证位域在内存中的具体排列顺序**。

也就是说，下面这些细节通常都是实现定义的：

- 第一个位域放在低位还是高位。
- 跨存储单元时怎么分配。
- 相邻不同基底类型的位域是否合并到同一个存储单元。
- 有符号位域的具体表示方式。
- 结构体整体大小和 padding 如何安排。

所以位域适合表达“进程内的紧凑状态”，但不适合直接当成跨平台网络协议、文件格式或 GPU/硬件寄存器的稳定二进制布局。需要精确二进制格式时，手写 `std::uint32_t raw` + mask 通常更可靠。

## `:0` 的特殊语法

位域里可以写无名成员：

```cpp
struct Example {
    unsigned int a : 3;
    unsigned int   : 5; // 无名位域，占 5 bit padding。
    unsigned int b : 8;
};
```

无名位域常用于跳过一些 bit，不暴露成员名。

更特殊的是宽度为 `0` 的无名位域：

```cpp
struct Example {
    unsigned int a : 3;
    unsigned int   : 0; // 强制下一个位域从新的 unsigned int 存储单元开始。
    unsigned int b : 8;
};
```

`unsigned int : 0` 的含义是：**结束当前存储单元，后面的位域从下一个该类型对齐的存储单元重新开始**。

可以把它理解成位域里的“强制换行”：

```text
没有 :0:
    [ a ][ b ] 可能被编译器塞进同一个 unsigned int 存储单元

有 unsigned int :0:
    [ a ][ padding 到当前 unsigned int 结束 ]
    [ b ][ 从新的 unsigned int 存储单元开始 ]
```

示例：

```cpp
#include <iostream>

struct WithoutZeroWidth {
    unsigned int a : 3;
    unsigned int b : 8;
};

struct WithZeroWidth {
    unsigned int a : 3;
    unsigned int   : 0; // 强制 b 从新的 unsigned int 存储单元开始。
    unsigned int b : 8;
};

int main()
{
    std::cout << sizeof(WithoutZeroWidth) << '\n';
    std::cout << sizeof(WithZeroWidth) << '\n';
}
```

在常见编译器上，`WithZeroWidth` 往往比 `WithoutZeroWidth` 更大，因为它主动放弃了当前存储单元剩余的 bit。

## 位域和对齐

位域和对齐关系很密切，但要分清楚两层：

- **位域宽度**决定字段需要多少 bit。
- **基底类型和 ABI**决定这些 bit 会被放进什么存储单元，以及结构体整体如何对齐。

例如：

```cpp
struct A {
    unsigned int x : 1;
    unsigned int y : 1;
};
```

虽然 `x + y` 只需要 2 bit，但结构体 `A` 的大小不一定是 1 字节。很多实现会用 `unsigned int` 作为分配单元，所以 `sizeof(A)` 可能是 4。

如果换成：

```cpp
#include <cstdint>

struct B {
    std::uint8_t x : 1;
    std::uint8_t y : 1;
};
```

有些编译器会让它更紧凑，但这仍然受到编译器实现和目标 ABI 的影响。你不能只看位域宽度就断言 `sizeof(B) == 1`。

下面这个例子可以用来观察当前编译器的布局：

```cpp
#include <cstdint>
#include <iostream>

struct UseUnsignedInt {
    unsigned int a : 1;
    unsigned int b : 1;
    unsigned int c : 1;
};

struct UseUint8 {
    std::uint8_t a : 1;
    std::uint8_t b : 1;
    std::uint8_t c : 1;
};

int main()
{
    std::cout << "sizeof(UseUnsignedInt) = " << sizeof(UseUnsignedInt) << '\n';
    std::cout << "alignof(UseUnsignedInt) = " << alignof(UseUnsignedInt) << '\n';

    std::cout << "sizeof(UseUint8) = " << sizeof(UseUint8) << '\n';
    std::cout << "alignof(UseUint8) = " << alignof(UseUint8) << '\n';
}
```

这段代码的输出不应该当成跨平台保证，只能当成“当前编译器、当前 ABI 下的观察结果”。

## 不能取地址

位域成员不是普通对象成员，不能对它取地址：

```cpp
struct Flags {
    unsigned int ready : 1;
};

int main()
{
    Flags flags{};

    // 错误：不能获取 bit field 的地址。
    // auto* p = &flags.ready;
}
```

原因很直接：位域成员可能只是某个存储单元里的几个 bit，不一定有独立地址。

同理，位域也不能作为普通引用长期绑定：

```cpp
struct Flags {
    unsigned int ready : 1;
};

int main()
{
    Flags flags{};

    // 错误或不可取：位域不是可以稳定引用的独立对象。
    // unsigned int& ref = flags.ready;
}
```

如果确实需要传给函数，通常传值或通过 setter / getter 包装。

## 位域和 `bool`

位域可以用 `bool` 作为基底类型：

```cpp
struct BooleanFlags {
    bool ready : 1;
    bool error : 1;
    bool dirty : 1;
};
```

这种写法适合表达多个开关位，访问也很直观：

```cpp
BooleanFlags flags{};
flags.ready = true;
flags.error = false;
```

不过它仍然有前面提到的布局问题：不要因为三个 `bool : 1` 就假设整个结构体只占 3 bit。结构体最终大小仍由编译器布局和对齐决定。

## 位域和枚举

位域经常和 `enum class` 配合使用，但不能直接把 `enum class` 当成位域类型到处乱用。更稳妥的写法是用无符号整数保存，再提供转换函数。

```cpp
#include <cstdint>

enum class Mode : std::uint8_t {
    kIdle = 0,
    kRead = 1,
    kWrite = 2,
    kError = 3,
};

/**
 * @brief 用 2 bit 保存模式字段。
 */
struct DeviceState {
    std::uint8_t mode : 2;  // 保存 Mode 的底层整数值。
    std::uint8_t busy : 1;  // 是否忙碌。
};

/**
 * @brief 读取模式字段并转回强类型枚举。
 *
 * @param state 设备状态。
 * @return 当前模式。
 */
Mode getMode(const DeviceState& state)
{
    return static_cast<Mode>(state.mode);
}

/**
 * @brief 设置模式字段。
 *
 * @param state 需要修改的设备状态。
 * @param mode 新模式。
 */
void setMode(DeviceState& state, Mode mode)
{
    state.mode = static_cast<std::uint8_t>(mode);
}
```

这种写法比直接暴露魔法数字更清楚。

## 位域适合什么场景

位域适合下面这些场景：

- **进程内状态压缩**：比如对象数量巨大，每个对象有很多小范围状态。
- **调试 / 教学结构**：用于把一个状态字拆成多个字段，便于阅读。
- **编译器和平台固定的内部结构**：比如同一项目内部只针对固定编译器和架构。
- **软硬件边界上的辅助视图**：可以临时用于解释寄存器字段，但不要把它当成唯一可靠布局。

例如，一个任务调度器里的任务状态：

```cpp
#include <cstdint>

/**
 * @brief 用位域保存任务的紧凑状态。
 */
struct TaskFlags {
    std::uint8_t ready       : 1; // 任务是否可以被调度。
    std::uint8_t running     : 1; // 任务是否正在运行。
    std::uint8_t finished    : 1; // 任务是否已经完成。
    std::uint8_t priority    : 2; // 任务优先级，范围 0..3。
    std::uint8_t retry_count : 3; // 已重试次数，范围 0..7。
};
```

这种结构读起来比一个裸 `std::uint8_t flags` 更直观。

## 位域不适合什么场景

位域不适合下面这些场景：

- **跨平台二进制协议**：位顺序、padding 和对齐都不够稳定。
- **文件格式直接落盘**：不同编译器可能生成不同布局。
- **网络包直接收发**：大小端、位序、padding 都可能踩坑。
- **需要原子更新某个位**：位域成员无法直接做标准库原子操作。
- **需要取地址或传引用**：位域没有独立地址。
- **性能热点且要求精确指令控制**：手写 mask 更可控。

如果要写网络协议或文件格式，通常建议这样：

```cpp
#include <cstdint>

/**
 * @brief 用明确整数和 bit mask 解析稳定二进制格式。
 */
class PacketHeader {
public:
    /**
     * @brief 使用原始 16 bit 包头构造解析对象。
     *
     * @param raw_header 网络或文件中读取到的原始包头。
     */
    explicit PacketHeader(std::uint16_t raw_header)
        : mRawHeader(raw_header)
    {
    }

    /**
     * @brief 获取 version 字段。
     *
     * @return 低 4 bit 表示的 version。
     */
    std::uint16_t getVersion() const
    {
        return static_cast<std::uint16_t>(mRawHeader & 0x000Fu);
    }

    /**
     * @brief 获取 type 字段。
     *
     * @return 第 4 到第 7 bit 表示的 type。
     */
    std::uint16_t getType() const
    {
        return static_cast<std::uint16_t>((mRawHeader >> 4) & 0x000Fu);
    }

private:
    std::uint16_t mRawHeader = 0; // 原始包头，布局由协议文档定义，不依赖 C++ 位域布局。
};
```

虽然这段代码更啰嗦，但它的二进制语义完全由 mask 和 shift 决定，更适合稳定边界。

## 和 `std::bitset` 的区别

`std::bitset` 和位域都和 bit 有关，但关注点不一样。

| 工具 | 适合表达 | 访问方式 | 大小 |
| --- | --- | --- | --- |
| 位域 | 一个结构体里的多个小字段 | `obj.field` | 字段数量和宽度在类型里固定 |
| `std::bitset<N>` | 一组同质 bit 集合 | `bits.test(i)` / `bits.set(i)` | `N` 在编译期固定 |
| 手写 mask | 精确二进制布局 | `raw & mask` / `raw >> shift` | 完全由自己控制 |

简单判断：

- 想表达“这个对象有 `ready/error/mode` 这些小字段”，可以考虑位域。
- 想表达“有 1024 个 visited 状态”，用 `std::bitset<1024>` 更合适。
- 想表达“协议第 3 到第 7 bit 的含义必须跨平台稳定”，用手写 mask 更合适。

## 常见坑点

### 不要假设内存布局

下面这个结构体看起来像一个 16 bit 包头：

```cpp
struct Header {
    unsigned int version : 4;
    unsigned int type    : 4;
    unsigned int length  : 8;
};
```

但你不能保证：

- `sizeof(Header) == 2`。
- `version` 一定位于最低 4 bit。
- 它在 GCC、Clang、MSVC 上布局完全一致。
- 它和网络字节序天然匹配。

如果这些条件对你很重要，就不要直接依赖位域布局。

### 谨慎使用有符号位域

```cpp
struct BadExample {
    int value : 3;
};
```

这类写法容易让读者疑惑：`value` 到底能表示 `-4..3`，还是别的范围？溢出时怎样？如果不是确实需要负数，优先写：

```cpp
struct BetterExample {
    unsigned int value : 3;
};
```

### 不要把位域当作原子 bit

位域成员不是独立对象，也不能直接写成：

```cpp
// 错误：std::atomic 不能直接包住某个位域成员。
// std::atomic_ref<unsigned int> ref(flags.ready);
```

多线程下如果多个线程同时修改同一个底层存储单元里的不同位域，也可能出现数据竞争。需要并发安全时，通常使用一个 `std::atomic<std::uint32_t>` 保存原始位，并用 `fetch_or` / `fetch_and` 这类原子位运算。

## 推荐写法

如果只是进程内状态压缩，可以这样写：

```cpp
#include <cstdint>

/**
 * @brief 保存连接状态的紧凑标志位。
 */
struct ConnectionFlags {
    std::uint8_t connected : 1; // 是否已经建立连接。
    std::uint8_t encrypted : 1; // 是否启用加密。
    std::uint8_t retrying  : 1; // 是否处于重试状态。
    std::uint8_t priority  : 2; // 优先级，范围 0..3。
};
```

如果需要稳定二进制布局，可以这样写：

```cpp
#include <cstdint>

/**
 * @brief 使用原始整数保存稳定布局的连接标志。
 */
class ConnectionFlagsMask {
public:
    /**
     * @brief 判断连接是否已经建立。
     *
     * @return 如果 connected bit 为 1，则返回 true。
     */
    bool isConnected() const
    {
        return (mBits & kConnectedMask) != 0;
    }

    /**
     * @brief 设置连接状态。
     *
     * @param connected 是否已经建立连接。
     */
    void setConnected(bool connected)
    {
        if (connected) {
            mBits |= kConnectedMask;
        } else {
            mBits &= static_cast<std::uint8_t>(~kConnectedMask);
        }
    }

private:
    static constexpr std::uint8_t kConnectedMask = 1u << 0; // connected 字段占第 0 bit。

    std::uint8_t mBits = 0; // 原始 bit 集合，由显式 mask 控制布局。
};
```

两种写法没有绝对高下。位域更像“把 bit 级字段写成成员语义”；手写 mask 更像“我要完全控制二进制格式”。

## 总结

位域可以这样记：

- **它是结构体成员语法**，不是独立容器。
- **它适合表达小范围字段**，例如多个 flag、mode、code。
- **它能节省空间**，但结构体最终大小仍受基底类型、padding、alignment 和 ABI 影响。
- **它的二进制布局不够可移植**，不要直接拿来定义跨平台协议格式。
- **`:0` 是强制结束当前存储单元**，可以影响后续位域的对齐和结构体大小。
- **不能取位域成员地址**，因为它不是有独立地址的普通对象。

实际工程里可以遵循一个简单原则：

> 进程内紧凑状态可以用位域；跨边界稳定格式优先用显式整数和 bit mask。
