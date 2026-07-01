---
title: C++ std::bitset 笔记
date: 2026-07-01
tags: [C++, STL, bitset, 位运算]
summary: 整理 std::bitset 的设计动机、本质模型和常用接口，补充 C 语言位掩码写法与 C++ bitset 的对照。
---

# C++ std::bitset 笔记

`std::bitset` 定义在头文件 `<bitset>` 中：

```cpp
#include <bitset>
```

它的核心用途是：**用固定数量的二进制位表示一组开关、状态、集合或可达性信息**。

```cpp
template <std::size_t N>
class bitset;
```

其中 `N` 是编译期常量，表示这个 `bitset` 里一共有多少个 bit。

## 为什么需要 `std::bitset`

在 C 语言里，如果想保存一组布尔开关，常见写法有两种。

第一种是直接开数组：

```c
#include <stdbool.h>

bool visited[1024];
```

这种写法直观，但一个 `bool` 通常至少占 1 字节。如果只需要 1024 个二值状态，它会占大约 1024 字节，而理论上 1024 个 bit 只需要 128 字节。

第二种是手写位掩码：

```c
#include <stdint.h>
#include <stdbool.h>

uint32_t flags = 0;

// 设置第 3 位。
flags |= (1u << 3);

// 清除第 3 位。
flags &= ~(1u << 3);

// 翻转第 3 位。
flags ^= (1u << 3);

// 判断第 3 位是否为 1。
bool enabled = (flags & (1u << 3)) != 0;
```

这种写法很省空间，也很快，但缺点明显：

- 需要自己处理移位、掩码、边界和整数宽度。
- 超过 `32` 或 `64` 位后，需要自己维护多个整数块。
- 代码可读性差，`flags |= (1u << i)` 不如 `bits.set(i)` 直观。
- 容易写出未定义行为，比如 `1u << 32`。

`std::bitset<N>` 就是把这种“位集合”能力封装成一个类型：

- 空间上仍然按 bit 压缩存储。
- 接口上提供 `set()`、`reset()`、`test()`、`count()` 等更清楚的方法。
- 大小 `N` 在编译期确定，适合固定范围的状态集合。
- 支持整体位运算，比如 `&`、`|`、`^`、`~`。

## 本质模型

可以把 `std::bitset<N>` 理解成一个固定长度的 bit 数组：

```cpp
std::bitset<8> bits;
```

逻辑上它有 8 个位置：

```text
index:  7 6 5 4 3 2 1 0
bit:    0 0 0 0 0 0 0 0
```

注意两个常见约定：

- `bits[0]` 表示**最低位**，也就是数值意义上的 $2^0$。
- `to_string()` 打印时从高位到低位输出，所以字符串最左边是 `bits[N - 1]`。

例如：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits;
    bits.set(0);
    bits.set(3);

    std::cout << bits << '\n'; // 00001001
}
```

这里设置的是第 `0` 位和第 `3` 位，但打印出来时高位在左、低位在右。

从实现角度看，标准不规定 `std::bitset` 必须怎么存。但常见实现会把它存成若干个无符号整数块，例如：

```text
bitset<128>  ->  两个 64-bit 整数块
bitset<1024> ->  十六个 64-bit 整数块
```

所以它的本质不是 `bool[N]`，而是**固定长度的压缩位容器**。

## 类接口概览

**原型**

```cpp
#include <bitset>

template <std::size_t N>
class std::bitset;
```

**模板参数**

| 参数 | 含义 |
| --- | --- |
| `N` | bit 数量，必须是编译期常量。`std::bitset<1024>` 表示 1024 个二值状态。 |

**重要约束**

| 约束 | 说明 |
| --- | --- |
| 固定大小 | `N` 在编译期确定，构造后不能扩容或缩容。 |
| 不存储元素对象 | 每一位不是独立 `bool` 对象，因此不能拿到普通 `bool&`。 |
| 下标方向 | `0` 是最低位，`N - 1` 是最高位。 |
| 适合固定范围 | 如果大小运行期才知道，通常用 `std::vector<bool>` 或第三方动态位图。 |

## 构造方式

### 默认构造

默认构造会把所有 bit 初始化为 `0`。

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits;
    std::cout << bits << '\n'; // 00000000
}
```

### 从整数构造

可以从 `unsigned long long` 等整数构造。低位会填到 `bits[0]`、`bits[1]` 等位置。

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits(13);   // 13 = 0b00001101

    std::cout << bits << '\n'; // 00001101
    std::cout << bits[0] << '\n'; // 1
    std::cout << bits[1] << '\n'; // 0
    std::cout << bits[2] << '\n'; // 1
    std::cout << bits[3] << '\n'; // 1
}
```

### 从字符串构造

字符串构造时，字符串左侧对应高位，右侧对应低位。

```cpp
#include <bitset>
#include <iostream>
#include <string>

int main()
{
    std::bitset<8> bits(std::string("10110000"));

    std::cout << bits << '\n';    // 10110000
    std::cout << bits[0] << '\n'; // 0，最右边字符对应最低位
    std::cout << bits[7] << '\n'; // 1，最左边字符对应最高位
}
```

## 访问单个 bit

### `operator[]`

`operator[]` 用来访问某一位。

```cpp
bool operator[](std::size_t pos) const;
reference operator[](std::size_t pos);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits;

    bits[2] = true;
    bits[5] = true;

    std::cout << bits << '\n';    // 00100100
    std::cout << bits[2] << '\n'; // 1
}
```

注意：非 `const` 版本返回的不是普通 `bool&`，而是 `std::bitset<N>::reference` 代理对象。因为 bit 被压缩存储，单独一位没有真实地址。

### `test`

`test(pos)` 判断某一位是否为 `1`。

```cpp
bool test(std::size_t pos) const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00010000");

    if (bits.test(4)) {
        std::cout << "第 4 位已设置\n";
    }
}
```

`test(pos)` 会检查下标范围，`pos >= N` 时会抛出 `std::out_of_range`。相比之下，`operator[]` 不做这种范围检查。

## 修改单个 bit

### `set`

`set(pos, value)` 设置某一位；不带参数的 `set()` 会把所有 bit 设置为 `1`。

```cpp
bitset& set();
bitset& set(std::size_t pos, bool value = true);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits;

    bits.set(1);        // 第 1 位设为 1
    bits.set(3, true);  // 第 3 位设为 1
    bits.set(1, false); // 第 1 位重新设为 0

    std::cout << bits << '\n'; // 00001000

    bits.set();                // 全部设为 1
    std::cout << bits << '\n'; // 11111111
}
```

### `reset`

`reset(pos)` 清除某一位；不带参数的 `reset()` 会把所有 bit 清零。

```cpp
bitset& reset();
bitset& reset(std::size_t pos);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("11111111");

    bits.reset(2);
    bits.reset(7);

    std::cout << bits << '\n'; // 01111011

    bits.reset();
    std::cout << bits << '\n'; // 00000000
}
```

### `flip`

`flip(pos)` 翻转某一位；不带参数的 `flip()` 会翻转全部 bit。

```cpp
bitset& flip();
bitset& flip(std::size_t pos);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00001111");

    bits.flip(0);
    bits.flip(7);

    std::cout << bits << '\n'; // 10001110

    bits.flip();
    std::cout << bits << '\n'; // 01110001
}
```

## 查询整体状态

### `count`

`count()` 返回值为 `1` 的 bit 数量。

```cpp
std::size_t count() const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("10110100");

    std::cout << bits.count() << '\n'; // 4
}
```

### `size`

`size()` 返回 bitset 的总 bit 数，也就是模板参数 `N`。

```cpp
std::size_t size() const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<128> bits;

    std::cout << bits.size() << '\n'; // 128
}
```

### `any`、`none`、`all`

这三个接口用于判断整体状态。

```cpp
bool any() const;  // 是否至少有一个 bit 为 1
bool none() const; // 是否所有 bit 都为 0
bool all() const;  // 是否所有 bit 都为 1
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<4> bits;

    std::cout << bits.none() << '\n'; // 1
    std::cout << bits.any() << '\n';  // 0

    bits.set();

    std::cout << bits.all() << '\n';  // 1
}
```

## 位运算接口

### `&`、`|`、`^`

`std::bitset` 支持按位与、按位或、按位异或。

```cpp
bitset operator&(const bitset& rhs) const;
bitset operator|(const bitset& rhs) const;
bitset operator^(const bitset& rhs) const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> a("11001100");
    std::bitset<8> b("10101010");

    std::cout << (a & b) << '\n'; // 10001000
    std::cout << (a | b) << '\n'; // 11101110
    std::cout << (a ^ b) << '\n'; // 01100110
}
```

这类操作适合表达集合关系：

- `a & b`：交集。
- `a | b`：并集。
- `a ^ b`：只在其中一个集合出现的元素。

### `~`

`~bits` 返回按位取反后的新 `bitset`。

```cpp
bitset operator~() const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00001111");

    std::cout << (~bits) << '\n'; // 11110000
}
```

### `&=`、`|=`、`^=`

这些接口会原地修改当前对象。

```cpp
bitset& operator&=(const bitset& rhs);
bitset& operator|=(const bitset& rhs);
bitset& operator^=(const bitset& rhs);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> reachable("00001111");
    std::bitset<8> next("00110011");

    reachable |= next;

    std::cout << reachable << '\n'; // 00111111
}
```

## 移位接口

### `<<`、`>>`

`<<` 和 `>>` 返回移位后的新对象。

```cpp
bitset operator<<(std::size_t pos) const;
bitset operator>>(std::size_t pos) const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00001101");

    std::cout << (bits << 2) << '\n'; // 00110100
    std::cout << (bits >> 1) << '\n'; // 00000110
}
```

### `<<=`、`>>=`

`<<=` 和 `>>=` 会原地移位。

```cpp
bitset& operator<<=(std::size_t pos);
bitset& operator>>=(std::size_t pos);
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00001101");

    bits <<= 2;
    std::cout << bits << '\n'; // 00110100

    bits >>= 3;
    std::cout << bits << '\n'; // 00000110
}
```

移位很适合做可达性 DP。例如 `dp << w` 可以表示“所有原本可达的和，都再加上 `w`”。

## 转换接口

### `to_string`

`to_string()` 把 bitset 转成字符串，高位在左，低位在右。

```cpp
std::string to_string() const;
```

示例：

```cpp
#include <bitset>
#include <iostream>
#include <string>

int main()
{
    std::bitset<8> bits;
    bits.set(0);
    bits.set(7);

    std::string s = bits.to_string();
    std::cout << s << '\n'; // 10000001
}
```

### `to_ulong`、`to_ullong`

这两个接口把 bitset 转成整数。

```cpp
unsigned long to_ulong() const;
unsigned long long to_ullong() const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> bits("00001101");

    std::cout << bits.to_ulong() << '\n';  // 13
    std::cout << bits.to_ullong() << '\n'; // 13
}
```

注意：如果 `bitset` 中的值无法放进目标整数类型，会抛出 `std::overflow_error`。

```cpp
#include <bitset>
#include <iostream>
#include <stdexcept>

int main()
{
    std::bitset<128> bits;
    bits.set(100);

    try {
        std::cout << bits.to_ullong() << '\n';
    } catch (const std::overflow_error& e) {
        std::cout << "超出 unsigned long long 可表示范围\n";
    }
}
```

## 比较接口

### `==`、`!=`

两个 `bitset<N>` 可以判断是否完全相等。

```cpp
bool operator==(const bitset& rhs) const;
bool operator!=(const bitset& rhs) const;
```

示例：

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<8> a("10101010");
    std::bitset<8> b("10101010");
    std::bitset<8> c("11110000");

    std::cout << (a == b) << '\n'; // 1
    std::cout << (a != c) << '\n'; // 1
}
```

## 输入输出

`std::bitset` 支持流输出和流输入。

```cpp
std::ostream& operator<<(std::ostream& os, const bitset<N>& bits);
std::istream& operator>>(std::istream& is, bitset<N>& bits);
```

示例：

```cpp
#include <bitset>
#include <iostream>
#include <sstream>

int main()
{
    std::bitset<8> bits("10110000");
    std::cout << bits << '\n'; // 10110000

    std::istringstream input("00001111");
    input >> bits;

    std::cout << bits << '\n'; // 00001111
}
```

## 常用接口速查

| 接口 | 用途 | 示例 |
| --- | --- | --- |
| `bits[i]` | 访问或修改第 `i` 位 | `bits[3] = true;` |
| `bits.test(i)` | 判断第 `i` 位是否为 `1`，带范围检查 | `if (bits.test(3)) {}` |
| `bits.set(i)` | 把第 `i` 位设为 `1` | `bits.set(3);` |
| `bits.set(i, false)` | 把第 `i` 位设为 `0` | `bits.set(3, false);` |
| `bits.set()` | 全部设为 `1` | `bits.set();` |
| `bits.reset(i)` | 把第 `i` 位清零 | `bits.reset(3);` |
| `bits.reset()` | 全部清零 | `bits.reset();` |
| `bits.flip(i)` | 翻转第 `i` 位 | `bits.flip(3);` |
| `bits.flip()` | 翻转全部 bit | `bits.flip();` |
| `bits.count()` | 统计 `1` 的数量 | `bits.count();` |
| `bits.size()` | 返回总 bit 数 | `bits.size();` |
| `bits.any()` | 是否至少有一个 `1` | `bits.any();` |
| `bits.none()` | 是否全是 `0` | `bits.none();` |
| `bits.all()` | 是否全是 `1` | `bits.all();` |
| `bits.to_string()` | 转成字符串 | `bits.to_string();` |
| `bits.to_ulong()` | 转成 `unsigned long` | `bits.to_ulong();` |
| `bits.to_ullong()` | 转成 `unsigned long long` | `bits.to_ullong();` |

## 使用场景

### 状态集合

当状态编号是固定范围时，`bitset` 很适合做集合。

```cpp
#include <bitset>
#include <iostream>

int main()
{
    std::bitset<128> visited;

    visited.set(42);

    if (visited.test(42)) {
        std::cout << "状态 42 已访问\n";
    }
}
```

### 可达性 DP

如果状态范围固定，`bitset` 可以把一批布尔状态打包处理。

```cpp
#include <bitset>
#include <iostream>
#include <vector>

int main()
{
    constexpr int kMaxSum = 32;
    std::vector<int> nums = {2, 3, 7};

    // dp[s] = true 表示当前可以凑出和 s。
    std::bitset<kMaxSum> dp;
    dp.set(0);

    for (int value : nums) {
        dp |= (dp << value);
    }

    std::cout << dp.test(12) << '\n'; // 1，2 + 3 + 7 = 12
}
```

### 小范围异或状态

如果异或值范围固定，例如只可能是 `0..1023`，`bitset<1024>` 可以保存“哪些异或值可达”。

```cpp
#include <bitset>
#include <iostream>

int main()
{
    constexpr int kMaxXor = 1024;
    std::bitset<kMaxXor> reachable;

    reachable.set(0);

    const int value = 13;
    std::bitset<kMaxXor> next;

    for (int xor_value = 0; xor_value < kMaxXor; xor_value++) {
        if (reachable.test(xor_value)) {
            next.set(xor_value ^ value);
        }
    }

    std::cout << next.test(13) << '\n'; // 1
}
```

## 和其他容器的区别

| 类型 | 大小 | 特点 | 适合场景 |
| --- | --- | --- | --- |
| `bool[N]` | 编译期固定 | 简单直观，但通常按字节存储 | 小规模布尔数组 |
| `std::array<bool, N>` | 编译期固定 | 比原生数组更安全，但仍不是按 bit 压缩的重点工具 | 需要标准容器接口的小规模布尔数组 |
| `std::bitset<N>` | 编译期固定 | 按 bit 压缩，支持位运算和计数 | 固定范围状态集合、可达性 DP、位掩码 |
| `std::vector<bool>` | 运行期动态 | 标准库特化，通常按 bit 压缩，但接口有代理对象坑点 | 大小运行期才知道的位集合 |
| `uint64_t` 掩码 | 固定 64 位 | 极快、极轻，但范围小，可读性较差 | 少量 flag、底层性能敏感代码 |

## 注意点

- `std::bitset<N>` 的 `N` 必须是编译期常量，不能用运行期变量决定大小。
- `bits[0]` 是最低位，`to_string()` 最左边是最高位。
- `operator[]` 不检查越界，`test()`、`set(pos)`、`reset(pos)`、`flip(pos)` 会检查范围并可能抛出 `std::out_of_range`。
- `to_ulong()` 和 `to_ullong()` 可能抛出 `std::overflow_error`。
- `std::bitset<N>` 的非 `const operator[]` 返回代理对象，不是普通 `bool&`。
- 如果状态范围很大但稀疏，`std::unordered_set<int>` 可能比巨大 `bitset` 更合适。

## 一句话总结

`std::bitset<N>` 是 C++ 对固定长度位集合的标准封装：它保留了 C 语言位掩码的紧凑和高效，又提供了更清楚、更不容易写错的接口。只要状态范围是编译期固定的，尤其是布尔 DP、访问标记、权限标记和小范围状态集合，`std::bitset` 都是非常值得优先考虑的工具。
