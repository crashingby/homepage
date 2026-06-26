---
title: std::numeric_limits 学习笔记
date: 2026-06-26
tags: [CPP, STL]
summary: 从 C 语言极值宏引入 std::numeric_limits，整理数值类型边界、精度、浮点特性和常见使用场景。
---

# std::numeric_limits 学习笔记

`std::numeric_limits<T>` 是 C++ 标准库里用来查询**数值类型边界和性质**的模板类。

它回答的问题通常是：

- `int` 的最大值是多少？
- `long long` 能表示多大的范围？
- `float` 有没有无穷大？
- `double` 的最小正正规数是多少？
- 一个类型是整数、浮点、有符号还是无符号？

在学习 `std::numeric_limits` 之前，最好先看 C 语言是怎么做这件事的。因为 C 的做法更原始，也更能说明 C++ 为什么要设计出一个类型化的标准库接口。

## C 语言里的极值宏

C 语言通过宏来描述基础数值类型的范围。

整数类型的极值主要在 `<limits.h>` 中：

```c
#include <limits.h>
```

常见宏如下：

| 宏 | 含义 |
| --- | --- |
| `CHAR_BIT` | 一个 `char` 占多少 bit |
| `CHAR_MIN` | `char` 的最小值 |
| `CHAR_MAX` | `char` 的最大值 |
| `SCHAR_MIN` | `signed char` 的最小值 |
| `SCHAR_MAX` | `signed char` 的最大值 |
| `UCHAR_MAX` | `unsigned char` 的最大值 |
| `SHRT_MIN` | `short` 的最小值 |
| `SHRT_MAX` | `short` 的最大值 |
| `USHRT_MAX` | `unsigned short` 的最大值 |
| `INT_MIN` | `int` 的最小值 |
| `INT_MAX` | `int` 的最大值 |
| `UINT_MAX` | `unsigned int` 的最大值 |
| `LONG_MIN` | `long` 的最小值 |
| `LONG_MAX` | `long` 的最大值 |
| `ULONG_MAX` | `unsigned long` 的最大值 |
| `LLONG_MIN` | `long long` 的最小值 |
| `LLONG_MAX` | `long long` 的最大值 |
| `ULLONG_MAX` | `unsigned long long` 的最大值 |

浮点类型的范围和精度主要在 `<float.h>` 中：

```c
#include <float.h>
```

常见宏如下：

| 宏 | 含义 |
| --- | --- |
| `FLT_MIN` | `float` 的最小正正规数 |
| `FLT_MAX` | `float` 的最大有限值 |
| `FLT_EPSILON` | `float` 中 `1` 和下一个可表示值之间的差 |
| `FLT_DIG` | `float` 可无误表示的十进制有效位数 |
| `DBL_MIN` | `double` 的最小正正规数 |
| `DBL_MAX` | `double` 的最大有限值 |
| `DBL_EPSILON` | `double` 中 `1` 和下一个可表示值之间的差 |
| `DBL_DIG` | `double` 可无误表示的十进制有效位数 |
| `LDBL_MIN` | `long double` 的最小正正规数 |
| `LDBL_MAX` | `long double` 的最大有限值 |
| `LDBL_EPSILON` | `long double` 中 `1` 和下一个可表示值之间的差 |

### C 宏的基本用法

```cpp
#include <climits>
#include <cfloat>
#include <iostream>

int main() {
    std::cout << "int 最大值: " << INT_MAX << '\n';
    std::cout << "long long 最大值: " << LLONG_MAX << '\n';
    std::cout << "double 最大有限值: " << DBL_MAX << '\n';
    std::cout << "double 最小正正规数: " << DBL_MIN << '\n';
}
```

C++ 里也可以使用这些宏。对应的 C++ 头文件通常是：

- `<climits>` 对应 C 的 `<limits.h>`。
- `<cfloat>` 对应 C 的 `<float.h>`。

## C 宏有什么不方便

C 宏足够直接，但它的表达能力比较有限。

- **不是类型接口**：`INT_MAX`、`LLONG_MAX`、`DBL_MAX` 都是不同名字，无法写成一个统一的模板逻辑。
- **不适合泛型编程**：如果函数模板里有一个类型 `T`，无法通过 `T_MAX` 这种形式得到它的最大值。
- **整数和浮点接口割裂**：整数在 `<limits.h>`，浮点在 `<float.h>`，名字体系也不统一。
- **宏没有命名空间**：宏不受 `namespace` 管理，名字会直接进入预处理环境。
- **不能自然扩展到自定义类型**：标准库和用户类型无法用同一套宏机制表达“这个类型的最大值、最小值、是否有无穷大”。

所以 C++ 提供了 `std::numeric_limits<T>`：把这些数值性质放进一个**按类型查询的模板接口**里。

## `std::numeric_limits<T>` 是什么

`std::numeric_limits<T>` 定义在 `<limits>` 中。

**接口声明：**

```cpp
#include <limits>

namespace std {
template <class T>
class numeric_limits;
}
```

它不是普通运行时对象，而是一个模板类。常见使用方式是：

```cpp
std::numeric_limits<int>::max();
std::numeric_limits<long long>::lowest();
std::numeric_limits<double>::epsilon();
std::numeric_limits<float>::has_infinity;
```

核心思想是：

- 类型 `T` 放在模板参数里。
- 类型的最大值、最小值、精度、浮点特性等信息通过静态成员访问。
- 标准库为所有基础算术类型提供特化。

## 支持哪些类型

标准库为基础算术类型提供 `numeric_limits` 特化。

| 类型类别 | 类型 |
| --- | --- |
| 布尔类型 | `bool` |
| 字符类型 | `char`, `signed char`, `unsigned char`, `wchar_t`, `char8_t`, `char16_t`, `char32_t` |
| 有符号整数 | `short`, `int`, `long`, `long long` |
| 无符号整数 | `unsigned short`, `unsigned int`, `unsigned long`, `unsigned long long` |
| 浮点数 | `float`, `double`, `long double` |

对于 `const int`、`volatile int`、`const volatile int` 这类 cv 限定类型，`std::numeric_limits` 的结果与未限定的 `int` 等价。

也就是说：

```cpp
std::numeric_limits<const int>::max();
std::numeric_limits<int>::max();
```

这两个查询表达的是同一个数值类型边界。

## 最常用接口

### 类型边界函数

这些静态成员函数都返回 `T` 类型的值。

| 接口 | 返回类型 | 含义 |
| --- | --- | --- |
| `min()` | `T` | 整数类型返回最小值；浮点类型返回最小正正规数 |
| `lowest()` | `T` | 返回该类型能表示的最低有限值，也就是最负的值 |
| `max()` | `T` | 返回该类型能表示的最大有限值 |
| `epsilon()` | `T` | 浮点类型中 `1` 和下一个可表示值之间的差 |
| `round_error()` | `T` | 最大舍入误差 |
| `infinity()` | `T` | 正无穷大，需要 `has_infinity == true` |
| `quiet_NaN()` | `T` | quiet NaN，需要 `has_quiet_NaN == true` |
| `signaling_NaN()` | `T` | signaling NaN，需要 `has_signaling_NaN == true` |
| `denorm_min()` | `T` | 最小正非正规数 |

最容易混淆的是 `min()` 和 `lowest()`。

- 对整数来说，`min()` 和 `lowest()` 通常相同。
- 对浮点来说，`min()` 不是最负数，而是**最小正正规数**。
- 如果想要“最小的有限值”，优先使用 `lowest()`。

```cpp
#include <iostream>
#include <limits>

int main() {
    std::cout << "double::min(): "
              << std::numeric_limits<double>::min() << '\n';

    std::cout << "double::lowest(): "
              << std::numeric_limits<double>::lowest() << '\n';

    std::cout << "double::max(): "
              << std::numeric_limits<double>::max() << '\n';
}
```

### 类型性质常量

这些成员常量用于描述类型特征。

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `is_specialized` | `bool` | 是否为该类型提供了 `numeric_limits` 特化 |
| `is_signed` | `bool` | 是否是有符号类型 |
| `is_integer` | `bool` | 是否是整数类型 |
| `is_exact` | `bool` | 是否能精确表示数值，整数通常为 `true`，浮点通常为 `false` |
| `is_bounded` | `bool` | 是否有有限表示范围 |
| `is_modulo` | `bool` | 溢出是否按模运算处理，典型例子是无符号整数 |
| `radix` | `int` | 数值表示使用的基数，整数通常是 `2` |
| `digits` | `int` | 能无误表示的 radix 位数 |
| `digits10` | `int` | 能无误表示的十进制有效位数 |
| `max_digits10` | `int` | 保证浮点文本 round-trip 所需的十进制位数 |

浮点相关特性如下：

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `has_infinity` | `bool` | 是否支持正无穷 |
| `has_quiet_NaN` | `bool` | 是否支持 quiet NaN |
| `has_signaling_NaN` | `bool` | 是否支持 signaling NaN |
| `has_denorm` | `std::float_denorm_style` | 是否支持非正规数 |
| `has_denorm_loss` | `bool` | 是否检测非正规数精度损失 |
| `is_iec559` | `bool` | 是否符合 IEC 559，也就是通常说的 IEEE 754 浮点 |
| `round_style` | `std::float_round_style` | 浮点舍入风格 |
| `traps` | `bool` | 算术运算是否可能触发 trap |
| `tinyness_before` | `bool` | 是否在舍入前检测 tiny 值 |
| `min_exponent` | `int` | radix 意义下的最小正规指数 |
| `min_exponent10` | `int` | 十进制意义下的最小正规指数 |
| `max_exponent` | `int` | radix 意义下的最大指数 |
| `max_exponent10` | `int` | 十进制意义下的最大指数 |

## `has_denorm` 和 `round_style`

`has_denorm` 的类型是 `std::float_denorm_style`。

| 枚举值 | 含义 |
| --- | --- |
| `std::denorm_absent` | 不支持非正规数 |
| `std::denorm_present` | 支持非正规数 |
| `std::denorm_indeterminate` | 是否支持由实现决定 |

`round_style` 的类型是 `std::float_round_style`。

| 枚举值 | 含义 |
| --- | --- |
| `std::round_indeterminate` | 舍入风格无法确定 |
| `std::round_toward_zero` | 向 0 舍入 |
| `std::round_to_nearest` | 向最近值舍入 |
| `std::round_toward_infinity` | 向正无穷方向舍入 |
| `std::round_toward_neg_infinity` | 向负无穷方向舍入 |

## 常见使用场景

### 初始化最值 DP

在动态规划或最短路里，经常需要一个足够大的初始值。

```cpp
#include <algorithm>
#include <limits>
#include <vector>

long long min_path_sum(const std::vector<int>& costs) {
    constexpr long long kInf = std::numeric_limits<long long>::max() / 4;
    std::vector<long long> dp(costs.size() + 1, kInf);
    dp[0] = 0;

    for (std::size_t i = 0; i < costs.size(); i++) {
        dp[i + 1] = std::min(dp[i + 1], dp[i] + costs[i]);
    }

    return dp[costs.size()];
}
```

这里不用 `max()` 的完整值，而是除以 `4`，是为了给后续加法留出空间，避免 `kInf + cost` 溢出。

### 模板代码里查询类型边界

C 宏很难服务模板代码，而 `std::numeric_limits<T>` 正适合泛型场景。

```cpp
#include <limits>

/**
 * @brief 返回类型 T 的最大有限值。
 *
 * @tparam T 需要查询边界的算术类型。
 * @return T 类型能够表示的最大有限值。
 */
template <typename T>
T get_type_max() {
    static_assert(std::numeric_limits<T>::is_specialized,
                  "T 必须是 numeric_limits 支持的类型");
    return std::numeric_limits<T>::max();
}
```

这个接口不关心 `T` 是 `int`、`long long` 还是 `double`，都能通过同一套模板写法获取最大值。

### 判断浮点类型是否支持无穷和 NaN

```cpp
#include <iostream>
#include <limits>

int main() {
    if constexpr (std::numeric_limits<double>::has_infinity) {
        const double inf = std::numeric_limits<double>::infinity();
        std::cout << "double 支持无穷大: " << inf << '\n';
    }

    if constexpr (std::numeric_limits<double>::has_quiet_NaN) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::cout << "double 支持 quiet NaN: " << nan << '\n';
    }
}
```

在常见平台上，输出通常类似：

```shell
double 支持无穷大: inf
double 支持 quiet NaN: nan
```

### 按类型选择更安全的初始值

```cpp
#include <limits>

/**
 * @brief 为最小化问题生成一个安全的正无穷哨兵值。
 *
 * @tparam T 用作代价的数值类型。
 * @return 如果 T 支持无穷，返回正无穷；否则返回最大值的四分之一。
 */
template <typename T>
T make_inf_for_minimize() {
    static_assert(std::numeric_limits<T>::is_specialized,
                  "T 必须是 numeric_limits 支持的类型");

    if constexpr (std::numeric_limits<T>::has_infinity) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max() / static_cast<T>(4);
    }
}
```

这个例子体现了 `numeric_limits` 的一个重要用途：**不仅查询边界，还能根据类型性质选择不同策略**。

## 常见坑

### `min()` 不是浮点最小值

对浮点类型来说：

```cpp
std::numeric_limits<double>::min();
```

返回的是最小正正规数，而不是最负的 double。

如果你要初始化最大值搜索，应该写：

```cpp
double best = std::numeric_limits<double>::lowest();
```

### `unsigned` 的 `min()` 是 0

```cpp
std::numeric_limits<unsigned int>::min();
```

返回 `0`。无符号类型没有负数。

### `is_modulo` 不等于所有整数都安全溢出

无符号整数通常是按模运算溢出，所以 `is_modulo` 为 `true`。

但有符号整数溢出在 C++ 中通常是未定义行为，不能依赖它“自然回绕”。

### `quiet_NaN()` 和 `signaling_NaN()` 不适合当普通错误码

NaN 适合表达浮点计算里的非法结果或缺失值，但不要把它当成所有错误处理的替代品。

工程代码里更清晰的方式通常是：

- `std::optional<T>` 表示可能没有值。
- 返回错误码或异常表示操作失败。
- NaN 只用于确实属于浮点数值域的问题。

## 和 C 宏的对应关系

| C 宏 | C++ `numeric_limits` |
| --- | --- |
| `INT_MIN` | `std::numeric_limits<int>::min()` |
| `INT_MAX` | `std::numeric_limits<int>::max()` |
| `LLONG_MIN` | `std::numeric_limits<long long>::min()` |
| `LLONG_MAX` | `std::numeric_limits<long long>::max()` |
| `UINT_MAX` | `std::numeric_limits<unsigned int>::max()` |
| `FLT_MIN` | `std::numeric_limits<float>::min()` |
| `FLT_MAX` | `std::numeric_limits<float>::max()` |
| `FLT_EPSILON` | `std::numeric_limits<float>::epsilon()` |
| `DBL_MIN` | `std::numeric_limits<double>::min()` |
| `DBL_MAX` | `std::numeric_limits<double>::max()` |
| `DBL_EPSILON` | `std::numeric_limits<double>::epsilon()` |

注意：浮点里的 `FLT_MIN`、`DBL_MIN` 和 `numeric_limits<T>::min()` 一样，表示**最小正正规数**，不是最负值。

## 总结

`std::numeric_limits<T>` 可以看作 C++ 对 C 极值宏的一次类型化升级。

- C 的 `INT_MAX`、`DBL_MAX` 直观简单，但不适合模板和泛型代码。
- C++ 的 `std::numeric_limits<T>` 把数值边界和类型性质统一放在模板接口里。
- 写 DP、最短路、数值计算、泛型算法时，优先使用 `std::numeric_limits<T>`。
- 对浮点类型要特别记住：`min()` 是最小正正规数，真正的最小有限值要用 `lowest()`。
