---
title: 模板学习笔记一：基础语法
date: 2026-06-26
tags: [CPP, STL]
summary: 介绍模板的基础语法和场景设计
---

# 现代 C++ 模板基础语法：让类型和常量参与设计

这章从已经知道 `template <typename T>` 开始，不讨论递归模板元编程和复杂的 SFINAE。目标是能够读懂、设计并调用 C++17/20 中常见的模板接口。

模板不是“生成任意复杂代码的魔法”。更实用的理解是：**把类型、常量和少量策略作为接口的一部分，在编译期选择正确的数据布局、实现或检查。**

本文所有示例默认使用 C++20；标有 C++17 的部分可用 `-std=c++17` 编译。代码片段彼此独立，按需补上 `#include` 即可。

```shell
g++ -std=c++20 -Wall -Wextra -Wpedantic example.cpp
```

## 这一章学完应该会什么

- 识别类型模板参数、常量模板参数和模板模板参数，并知道各自何时合适。
- 用模板默认参数、别名模板和变量模板减少重复代码。
- 用 `constexpr`、`consteval`、`static_assert`、`if constexpr` 表达编译期逻辑。
- 区分完全特化、偏特化和重载；知道函数模板没有偏特化。
- 看懂 `typename` 与 `template` 两个容易造成报错的关键字。
- 正确使用 CTAD（类模板实参推导）、推导指引、花括号初始化和 `std::in_place`。
- 知道为什么模板定义通常必须写在头文件中。

## 一张地图：模板参数能表达什么

| 参数类别 | 写法 | 表达的内容 | 典型用途 |
| --- | --- | --- | --- |
| 类型参数 | `typename T` | 元素类型、策略类型、容器类型 | `std::vector<T>` |
| 常量参数 | `std::size_t N`、`auto V` | 长度、开关、枚举值、字符串 | `std::array<T, N>` |
| 模板模板参数 | `template <typename...> class C` | 一个可被套用的模板 | 通用容器适配器 |

类型参数回答“处理什么类型”，常量参数回答“编译期已知的配置是什么”。二者一起使用，才能写出像 `std::array<T, N>` 这样既有元素类型、又有静态大小的接口。

```cpp
#include <array>
#include <cstddef>

template <typename T, std::size_t N>
struct FixedBuffer {
    std::array<T, N> values{};

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return N;
    }
};

constexpr FixedBuffer<int, 4> ids{};
static_assert(ids.size() == 4);
```

`N` 不是运行时成员变量。`FixedBuffer<int, 4>` 与 `FixedBuffer<int, 8>` 是两个不同的类型，因此编译器可以据此确定对象大小、展开循环或拒绝不匹配的接口。

## 类型模板参数：`typename` 和 `class`

对于模板参数，`typename` 与 `class` 完全等价。现代代码里任选一种并保持一致即可；本文用 `typename` 强调它是一个类型位置。

```cpp
template <typename T>
constexpr T larger(T left, T right) {
    return left < right ? right : left;
}

static_assert(larger(3, 5) == 5);
```

模板函数通常能从实参推导 `T`：`larger(3, 5)` 会推导为 `larger<int>(3, 5)`。混合类型时推导不会主动替你找“共同类型”，下面的调用会失败。

```cpp
// larger(3, 2.5);  // 错误：同一个 T 不能同时推导为 int 和 double。

const auto value = larger<double>(3, 2.5);  // 显式指定 T 后可以。
```

当接口本来就允许两个类型不同，应把它们写成不同模板参数，而不是逼迫调用者指定一个共同类型。

```cpp
#include <type_traits>

template <typename Left, typename Right>
constexpr auto add(Left left, Right right) {
    using Result = std::common_type_t<Left, Right>;
    return static_cast<Result>(left) + static_cast<Result>(right);
}

static_assert(add(3, 2.5) == 5.5);
```

这里 `Left` 和 `Right` 描述输入，`Result` 描述输出。命名稍长，但类型语义清楚，错误信息也更好读。

## 常量模板参数：把配置放到类型里

C++20 将过去常说的“非类型模板参数”改称为**常量模板参数**。它接受编译期常量，而不只接受整数。

### 整数、枚举、指针等常见写法

```cpp
#include <array>
#include <cstddef>

enum class Layout {
    row_major,
    column_major,
};

template <typename T, std::size_t Rows, std::size_t Cols, Layout Storage>
struct Matrix {
    std::array<T, Rows * Cols> data{};

    constexpr T& operator()(std::size_t row, std::size_t col) {
        if constexpr (Storage == Layout::row_major) {
            return data[row * Cols + col];
        } else {
            return data[col * Rows + row];
        }
    }
};

using RowMajorMatrix = Matrix<float, 3, 4, Layout::row_major>;
using ColumnMajorMatrix = Matrix<float, 3, 4, Layout::column_major>;
```

`Storage` 是类型的一部分，因而 `if constexpr` 可以在编译期只保留对应分支。这里的 `row_major` 和 `column_major` 不是运行时选项：两种布局各自是一种 `Matrix` 类型。

如果布局真的需要在运行中切换，就应把它写成对象状态或使用运行时多态；不要为了模板而把所有配置硬塞进模板参数。

### `auto` 常量模板参数

当参数的类型不重要或可由实参决定时，C++17 可以使用 `auto`。

```cpp
template <auto Value>
struct Constant {
    static constexpr auto value = Value;
};

static_assert(Constant<42>::value == 42);
static_assert(Constant<'x'>::value == 'x');
```

它特别适合把“值”包装成可参与模板组合的类型。标准库的 `std::integral_constant` 与 `std::bool_constant` 就是这一思路的经典实现；多数业务代码不需要自己造一个。

### C++20：把固定字符串作为模板参数

C++20 允许满足结构类型要求的类对象作为常量模板参数。固定字符串是一个直观例子：

```cpp
#include <cstddef>
#include <string_view>

template <std::size_t N>
struct FixedString {
    char value[N]{};

    constexpr FixedString(const char (&text)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            value[i] = text[i];
        }
    }

    [[nodiscard]] constexpr std::string_view view() const {
        return {value, N - 1};  // 排除结尾的 '\\0'。
    }
};

template <FixedString Name>
struct Command {
    static constexpr auto name = Name;
};

using StartCommand = Command<"start">;
static_assert(StartCommand::name.view() == "start");
```

这个能力很有趣，但不要急着把运行时字符串改成模板字符串。它适用于命令名、编译期注册表、格式串检查等**确实需要编译期身份**的场景；普通文本仍应使用 `std::string` 或 `std::string_view`。

## 模板默认参数：把常用选择变成默认值

模板默认参数和函数默认参数很像，但有一条更重要的规则：**默认模板参数后面的参数通常也要有默认值。**

```cpp
#include <array>
#include <cstddef>

template <typename T, std::size_t Capacity = 64>
struct SmallBuffer {
    std::array<T, Capacity> data{};
};

SmallBuffer<int> ids;              // Capacity 为 64。
SmallBuffer<int, 256> large_ids;   // 改变容量。
```

默认参数适合表示“绝大多数调用者不需要关心，但少数调用者可以覆写”的策略。`std::vector<T, Allocator = std::allocator<T>>` 正是这个模式。

默认参数不是配置文件。若可选项已经多到调用点难以阅读，应考虑配置结构体、具名工厂函数，或把不同职责拆开。

## 模板模板参数：接收一个模板，而非一个类型

模板模板参数用于“这个参数本身也应当是模板”的场景。它在泛型容器适配器里偶尔有用，但在日常业务代码中远少于前两类参数。

```cpp
#include <deque>
#include <vector>

template <typename T, template <typename, typename> class Sequence>
class Queue {
public:
    void push(const T& value) {
        mValues.push_back(value);
    }

    [[nodiscard]] T& front() {
        return mValues.front();
    }

private:
    Sequence<T, std::allocator<T>> mValues;
};

Queue<int, std::deque> queue;
queue.push(42);
```

这里 `Sequence` 接收两个类型参数，正好匹配 `std::deque<T, Allocator>`。模板模板参数的“形状匹配”规则曾经比较严格，容易把接口写得脆弱。优先在确实需要由调用者选择容器模板时使用；只需接收一个已经构造好的容器类型时，写 `typename Container` 更简单。

## 别名模板和变量模板：给重复类型与常量起名字

### 别名模板：`using` 的模板版本

别名模板不创建新类型，也不包装对象；它只是为一个复杂类型起一个可参数化的名字。

```cpp
#include <memory>
#include <vector>

template <typename T>
using UniqueVector = std::vector<std::unique_ptr<T>>;

UniqueVector<int> numbers;
numbers.push_back(std::make_unique<int>(42));
```

它比宏安全得多，也比 `typedef` 更适合模板。标准库中的 `std::remove_reference_t<T>`、`std::enable_if_t<...>` 都是别名模板。

### 变量模板：类型相关的常量

变量模板适合“每个类型都有一个常量”的情况。

```cpp
#include <numbers>

template <typename T>
inline constexpr T circle_ratio_v = static_cast<T>(3.14159265358979323846L);

static_assert(circle_ratio_v<float> > 3.14F);
static_assert(circle_ratio_v<double> > 3.14);

// C++20 标准库已提供更完整的版本。
static_assert(std::numbers::pi_v<double> > 3.14);
```

`inline constexpr` 中的 `inline` 很关键：C++17 起，它让头文件中的变量模板可以被多个翻译单元包含而不产生重复定义。

变量模板还常用于让 `static_assert` 在 `if constexpr` 的无效分支里按类型延迟触发：

```cpp
template <typename>
inline constexpr bool always_false_v = false;

template <typename T>
void serialize(const T&) {
    static_assert(always_false_v<T>, "serialize: 当前类型没有序列化实现");
}
```

注意不能直接写 `static_assert(false, "...")`。它在模板定义被读到时就失败，根本等不到某个 `T` 被实例化。

## `constexpr`：让代码可以在编译期执行

`constexpr` 的意思不是“必然在编译期执行”，而是“当实参与上下文允许时，可以在编译期执行”。同一个 `constexpr` 函数可以同时服务编译期和运行时。

```cpp
#include <array>
#include <cstddef>

template <typename T, std::size_t N>
constexpr T dot(const std::array<T, N>& left, const std::array<T, N>& right) {
    T result{};
    for (std::size_t i = 0; i < N; ++i) {
        result += left[i] * right[i];
    }
    return result;
}

constexpr std::array left{1, 2, 3};
constexpr std::array right{4, 5, 6};
static_assert(dot(left, right) == 32);  // 编译期求值。

int main() {
    std::array runtime_left{1, 2, 3};
    std::array runtime_right{4, 5, 6};
    return dot(runtime_left, runtime_right);  // 运行时调用同一函数。
}
```

模板与 `constexpr` 很适合组合：模板参数给出类型和固定规模，`constexpr` 让算法在有条件时提前计算。不要为了“全编译期”牺牲可读性；在运行时数据上调用 `constexpr` 函数也完全正常。

### `consteval`：必须在编译期执行

C++20 的 `consteval` 用于不允许出现运行时调用的函数，常见于编译期校验或生成固定配置。

```cpp
#include <cstddef>

consteval std::size_t checked_capacity(int capacity) {
    if (capacity <= 0) {
        throw "capacity 必须为正数";
    }
    return static_cast<std::size_t>(capacity);
}

template <typename T, std::size_t Capacity>
struct RingBuffer {
    T data[Capacity]{};
};

using EventQueue = RingBuffer<int, checked_capacity(128)>;

// int runtime_capacity = 128;
// RingBuffer<int, checked_capacity(runtime_capacity)> invalid;  // 错误：实参不是编译期常量。
```

`consteval` 比 `constexpr` 强：每一次调用都必须产生常量表达式。它不是 `constexpr` 的通用替代品，只有调用者必须在编译期给出答案时才使用。

### 不要混淆 `const`、`constexpr` 与 `constinit`

| 关键字 | 解决的问题 |
| --- | --- |
| `const` | 对象初始化后不能通过该名字修改；不保证是编译期常量。 |
| `constexpr` | 对象是常量表达式，或函数在条件允许时可参与常量求值。 |
| `constinit` | 静态/线程存储期对象必须静态初始化；不表示对象不可修改，也不使函数变成编译期函数。 |

`constinit` 主要处理跨翻译单元的初始化顺序问题，不是模板学习的核心。看到它时记住：它关心的是“何时初始化”，不是“能否修改”。

## `static_assert`：把接口约束写进编译器

模板的错误信息可能很长。`static_assert` 能在离问题最近的地方给调用者一条明确诊断。

```cpp
#include <cstddef>

template <typename T, std::size_t Alignment>
struct AlignedBlock {
    static_assert(Alignment != 0, "Alignment 不能为 0");
    static_assert((Alignment & (Alignment - 1)) == 0,
                  "Alignment 必须是 2 的幂");

    alignas(Alignment) T value{};
};

AlignedBlock<int, 64> block;
// AlignedBlock<int, 48> invalid;  // 会给出第二条断言信息。
```

好的断言检查的是模板接口的不变量，例如固定大小、对齐、策略组合是否合法。它不应该替代正常的运行时输入校验：用户输入的文件大小、网络消息长度等运行时数据仍要在运行时检查。

## `if constexpr`：按类型或配置选择实现

C++17 的 `if constexpr` 是现代模板里最重要的工具之一。条件在编译期求值，未选中的分支会被丢弃，因此其中可以出现对当前模板参数不合法的代码。

```cpp
#include <iostream>
#include <string_view>
#include <type_traits>

template <typename T>
void print_value(const T& value) {
    if constexpr (std::is_same_v<T, std::string_view>) {
        std::cout << "text: " << value << '\\n';
    } else if constexpr (std::is_integral_v<T>) {
        std::cout << "integer: " << value << '\\n';
    } else {
        std::cout << "other: " << value << '\\n';
    }
}

int main() {
    print_value(42);
    print_value(std::string_view{"hello"});
}
```

这里的类型萃取（`std::is_same_v`、`std::is_integral_v`）先把它当作“回答编译期真/假问题的标准库工具”。下一章再系统学习它们。

更能体现 `if constexpr` 价值的例子是调用只对某类类型存在的成员：

```cpp
#include <type_traits>

template <typename T>
constexpr auto absolute_value(T value) {
    if constexpr (std::is_unsigned_v<T>) {
        return value;  // 无符号数不需要处理负数。
    } else {
        return value < 0 ? -value : value;
    }
}

static_assert(absolute_value(-7) == 7);
static_assert(absolute_value(7U) == 7U);
```

普通 `if` 不行：两个分支都会参与编译，模板实例化时仍可能报错。`if constexpr` 不是运行时优化提示，而是**形成不同实现**的语法。

### 用 `if constexpr` 做小范围分派，不要做万能函数

当分支只有两三种且共享大部分流程时，它很清楚；分支增长到许多类型、每个分支都有独立职责时，应该转为重载、偏特化或后续要学的 concepts。否则单个模板会变成难维护的“类型大杂烩”。

## 参数包与折叠表达式：处理任意个参数

参数包（parameter pack）就是模板世界里的“零个或多个”。它能在一次调用里接收任意数量、甚至任意类型的参数；**数量与每个参数的类型都在编译期确定**。它不是 `std::vector`，不能在运行时再 `push_back` 一个参数。

C++11 有参数包，但常需要递归模板逐个处理。C++17 的折叠表达式（fold expression）让“把同一种操作作用于每个实参”有了直接语法，因此现代 C++ 代码通常不再为此手写递归。

这一节有两件彼此相关但不同的事：

- **包展开**：把 `Ts...` 或 `values...` 展开成一串类型、基类、函数实参或初始化器。
- **折叠表达式**：把包里的每个元素用同一个二元运算符连起来，例如 `a + b + c`、`p1 && p2 && p3`。

### 两种包：模板参数包与函数参数包

最常见的函数模板同时声明两种包：`Ts` 是类型包，`values` 是函数参数包。二者的位置一一对应。

```cpp
#include <cstddef>
#include <string_view>

template <typename... Ts>
constexpr std::size_t count_arguments(Ts&&... values) {
    // values 只用于让编译器从调用实参推导 Ts；此函数只关心数量。
    return sizeof...(Ts);
}

static_assert(count_arguments(7, 3.5, std::string_view{"ok"}) == 3);
static_assert(count_arguments() == 0);
```

调用 `count_arguments(7, 3.5, std::string_view{"ok"})` 时，概念上可以这样看：

```cpp
// Ts...     -> int, double, std::string_view
// values... -> 7,   3.5,    std::string_view{"ok"}
```

`sizeof...(Ts)` 和 `sizeof...(values)` 都返回包中元素的个数；在这个函数里它们恒相等。写类型相关逻辑时习惯使用 `sizeof...(Ts)`，写参数相关逻辑时用 `sizeof...(values)`，可读性更好。

`Ts&&... values` 中的 `Ts&&` 在这里是转发引用。它允许每个参数独立保留左值或右值属性。转发的细节会在“值类别与完美转发”章节展开；本节先把它当成参数包最常见的接收方式。

### 不要按前后位置背 `...`

`...` 的位置看起来飘忽，是因为它服务于三种不同的语法：**声明一个包、查询包长度、展开一个模式**。先判断自己正在做哪件事，再看对应写法；不要试图用“总在 `Ts` 前面/后面”记忆。

| 你要做的事 | 写法 | `...` 的角色 |
| --- | --- | --- |
| 声明一个类型参数包 | `template <typename... Ts>` | 模板参数声明语法规定 `...` 写在参数名 `Ts` 前。 |
| 声明一个函数参数包 | `void f(Ts... values)` | 函数参数声明语法规定 `...` 写在变量名 `values` 前；`Ts` 提供每个参数的类型。 |
| 查询包的元素个数 | `sizeof...(Ts)` | `sizeof...` 是一个专门的语言运算符，固定写成这个形状。 |
| 把模式重复为逗号分隔的列表 | `std::tuple<Ts...>` | `Ts` 是待重复的模式，`...` 写在模式后面。 |

因此下面这段并不是同一条规则被“反着写了四次”：

```cpp
#include <tuple>

template <typename... Ts>     // 声明：Ts 是类型包。
void inspect(Ts... values) {  // 声明：values 是与 Ts 对应的值包。
    constexpr auto count = sizeof...(Ts);  // 查询：Ts 里有几个类型。
    using Result = std::tuple<Ts...>;       // 展开：得到 tuple<T0, T1, ...>。
}
```

假设调用 `inspect(10, 2.5)`，可以把它摊平为：

```cpp
// Ts...     是 [int, double]，一个类型列表。
// values... 是 [10,  2.5]，    一个值列表。
// count     是 2。
// Result    是 std::tuple<int, double>。
```

这里的核心关系是：`Ts...` 与 `values...` 是**两个不同的包**，但 `values` 被声明为 `Ts... values`，所以第 `i` 个值的类型对应第 `i` 个 `Ts`。类型包只能在“需要类型”的位置使用；值包只能在“需要表达式或函数实参”的位置使用。

### 只有“展开模式”时，才记住模式后面跟 `...`

包本身不能单独变成一条普通表达式；要写一个包含包的**模式**，再在模式后放 `...`。编译器会为包中每个元素复制该模式。这里的“模式”可以只是 `Ts` 或 `values`，也可以是包着它们的更大表达式。

```cpp
// 假设 Ts... = [int, double]，values... = [10, 2.5]

std::tuple<Ts...>                       // std::tuple<int, double>
target(values...)                       // target(10, 2.5)
std::array<double, 2>{static_cast<double>(values)...}  // {10.0, 2.5}
std::forward<Ts>(values)...             // forward<int>(10), forward<double>(2.5)
```

最后一行尤其重要：模式是完整的 `std::forward<Ts>(values)`，不是只有 `Ts` 或 `values`。一次包展开可以同时使用多个包；所有参与同一次展开的包长度必须相同。这里二者都来自同一次函数调用，长度天然相同。

实际读代码时可以这样判断：

1. 看到 `sizeof...`：这是**数数量**，别把它当作展开。
2. 看到 `typename... Name` 或 `Type... name`：这是**声明包**，位置由声明语法决定。
3. 其余在 `<>`、`()`、`{}`、继承列表或表达式中出现的 `pattern...`：找到 `...` 左边完整的模式，然后把它对包的每个元素复制一遍。

例如 `std::tuple<std::vector<Ts>...>` 的完整模式是 `std::vector<Ts>`，而不是 `Ts`；因此它展开为一组 `vector` 类型。

```cpp
#include <tuple>
#include <vector>

template <typename... Ts>
using TupleOfValues = std::tuple<Ts...>;

template <typename... Ts>
using TupleOfVectors = std::tuple<std::vector<Ts>...>;

using Values = TupleOfValues<int, double, bool>;
// 展开为 std::tuple<int, double, bool>

using Batches = TupleOfVectors<int, double, bool>;
// 展开为 std::tuple<std::vector<int>, std::vector<double>, std::vector<bool>>
```

第二个例子里，`std::vector<Ts>` 是模式，`...` 让它对每个 `T` 重复一次。参数包可在多个语法位置展开：

| 展开位置 | 写法 | 展开结果示意 |
| --- | --- | --- |
| 模板实参 | `std::tuple<Ts...>` | `std::tuple<int, double>` |
| 复合类型 | `std::tuple<std::vector<Ts>...>` | `tuple<vector<int>, vector<double>>` |
| 函数实参 | `target(std::forward<Ts>(values)...)` | `target(v1, v2, v3)` |
| 花括号初始化 | `std::array<T, N>{values...}` | `{v1, v2, v3}` |
| 基类列表 | `struct X : Fs...` | `struct X : F1, F2` |
| `using` 声明 | `using Fs::operator()...` | 分别引入每个 `operator()` |

一个实用的“展开为构造参数”例子如下。它接收任意种可用于构造 `T` 的参数，每个参数对应一个元素，而不是把全部参数交给一个对象的构造函数。

```cpp
#include <string>
#include <utility>
#include <vector>

template <typename T, typename... Args>
std::vector<T> make_vector(Args&&... args) {
    std::vector<T> values;
    values.reserve(sizeof...(Args));

    // 对每个 arg 展开一次 emplace_back。
    (static_cast<void>(values.emplace_back(std::forward<Args>(args))), ...);
    return values;
}

int main() {
    const auto names = make_vector<std::string>("alice", "bob", "carol");
}
```

上例中的 `std::forward<Args>(args)...` 和 `values.emplace_back(...)` 都是包展开；最后的逗号折叠负责按顺序调用这些 `emplace_back`。`reserve(sizeof...(Args))` 不是必须的模板语法，但它知道元素数量已在调用时确定，因此可以避免扩容。

### 包展开不等于折叠

下面两个函数看起来都用了 `...`，做的事却不同：

```cpp
#include <array>
#include <type_traits>

template <typename... Ts>
constexpr auto make_common_array(Ts... values) {
    static_assert(sizeof...(Ts) > 0, "至少需要一个元素");

    using ValueType = std::common_type_t<Ts...>;
    return std::array<ValueType, sizeof...(Ts)>{
        static_cast<ValueType>(values)...};  // 包展开：得到一个初始化列表。
}

template <typename... Ts>
constexpr auto sum(Ts... values) {
    static_assert(sizeof...(Ts) > 0, "sum 至少需要一个参数");
    return (values + ...);  // 折叠：得到一个加法表达式。
}

static_assert(make_common_array(1, 2.5).size() == 2);
static_assert(sum(1, 2, 3, 4) == 10);
```

- `static_cast<ValueType>(values)...` 变成 `static_cast<ValueType>(v1), static_cast<ValueType>(v2), ...`，每个结果各占一个数组元素。
- `(values + ...)` 则变成单个表达式 `v1 + (v2 + (... + vn))`，最终只得到一个和。

如果你的目标是“生成多个类型或多个实参”，先想包展开；如果目标是“把多个值合成一个值或依次执行操作”，再想折叠表达式。

### 四种折叠表达式

折叠表达式只支持规定的一批二元运算符，并且整体必须写在括号内。`pack` 表示参数包，`init` 表示一个初始值。

| 写法 | 名称 | 三个元素 `a, b, c` 时的展开 |
| --- | --- | --- |
| `(... op pack)` | 一元左折叠 | `((a op b) op c)` |
| `(pack op ...)` | 一元右折叠 | `(a op (b op c))` |
| `(init op ... op pack)` | 二元左折叠 | `(((init op a) op b) op c)` |
| `(pack op ... op init)` | 二元右折叠 | `(a op (b op (c op init)))` |

加法和乘法通常满足结合律，左右折叠看不出区别；减法、除法、字符串/路径拼接的策略、流插入等就可能不同。用减法观察最直观：

```cpp
template <typename... Ts>
constexpr auto subtract_left(Ts... values) {
    static_assert(sizeof...(Ts) > 0, "至少需要一个参数");
    return (... - values);
}

template <typename... Ts>
constexpr auto subtract_right(Ts... values) {
    static_assert(sizeof...(Ts) > 0, "至少需要一个参数");
    return (values - ...);
}

static_assert(subtract_left(20, 3, 2) == 15);   // (20 - 3) - 2
static_assert(subtract_right(20, 3, 2) == 19);  // 20 - (3 - 2)
```

二元折叠的 `init` 通常用来提供数学单位元或一个累积器初值：

```cpp
template <typename... Ts>
constexpr auto sum_or_zero(Ts... values) {
    return (0 + ... + values);  // ((0 + v1) + v2) + ...
}

template <typename... Ts>
constexpr auto product_or_one(Ts... values) {
    return (1 * ... * values);  // ((1 * v1) * v2) * ...
}

static_assert(sum_or_zero() == 0);
static_assert(sum_or_zero(2, 3) == 5);
static_assert(product_or_one() == 1);
static_assert(product_or_one(2, 3, 4) == 24);
```

`init` 不是随便补一个值。它会参与类型推导和运算：`0 + ...` 对无符号大整数或自定义数值类型未必是最合适的初值。对泛型算法，先明确零值/单位元该由谁提供；必要时让调用者传入初值，或使用该类型的 `T{}`。

### 哪些运算符可以折叠

可折叠的是大多数二元运算符，例如算术运算、位运算、比较、移位、逻辑运算、逗号，以及复合赋值。`?:`、普通赋值 `=`、成员访问 `.`、下标 `[]` 和函数调用 `()` 不能直接作为 fold operator。

标准允许的运算符集合为：

```cpp
+  -  *  /  %  ^  &  |  <<  >>
+= -= *= /= %= ^= &= |= <<= >>=
== != < > <= >= && || , .* ->*
```

若 `init` 或包元素本身是复杂表达式，请明确加括号，别让优先级遮住折叠结构：

```cpp
// 正确：把 1 * 2 作为二元右折叠的初始值。
return (values + ... + (1 * 2));
```

### 空参数包：何时自然成立，何时必须拒绝

空包不是异常状态。日志函数、批量执行函数和“所有条件都成立”这类接口本来就应允许零个参数。

对于**一元**折叠，空包只有三个运算符有语言规定的身份值。这里不用表格，避免 `||` 被 Markdown 误判为表格分隔符：

- `(... && values)` 的结果是 `true`：没有反例，因此“全部为真”成立。
- `(... || values)` 的结果是 `false`：没有真值，因此“至少一个为真”不成立。
- `(values, ...)` 或 `(..., values)` 的结果是 `void()`：没有动作可执行。

其他一元折叠，例如 `(values + ...)`，在空包上是非法的。若数学或业务上有明确初值，使用二元折叠；若没有，就用 `static_assert` 拒绝空调用。

```cpp
template <typename... Bs>
constexpr bool all(Bs... values) {
    return (... && static_cast<bool>(values));
}

template <typename... Bs>
constexpr bool any(Bs... values) {
    return (... || static_cast<bool>(values));
}

static_assert(all());
static_assert(all(true, 1, 42));
static_assert(!all(true, false, true));
static_assert(!any());
static_assert(any(false, 0, true));
```

### 结合方向不等于求值顺序

这是参数包最容易埋坑的一点。左折叠/右折叠描述的是**表达式树如何结合**，不自动保证每个子表达式按从左到右执行。

对于 `+`、`*` 等运算符，不要让每个包元素都修改共享状态：

```cpp
// 不要依赖下面调用的执行先后顺序。
// return (next_value() + ...);
```

需要明确顺序时，优先使用内建逗号运算符折叠；每一个逗号左侧都先执行，再执行右侧。将左侧转换为 `void` 能避免用户类型意外重载逗号运算符。

```cpp
#include <functional>
#include <utility>

template <typename... Fs>
void run_all(Fs&&... functions) {
    (static_cast<void>(std::invoke(std::forward<Fs>(functions))), ...);
}

int main() {
    int state = 0;

    run_all(
        [&] { state = state * 10 + 1; },
        [&] { state = state * 10 + 2; },
        [&] { state = state * 10 + 3; });

    // state 一定是 123。
}
```

`&&` 和 `||` 也有明确的从左到右短路语义，因此适合“所有校验均通过”或“任一方案成功”的接口：

```cpp
#include <cassert>
#include <functional>
#include <utility>

template <typename... Predicates>
bool all_of(Predicates&&... predicates) {
    return (... && std::invoke(std::forward<Predicates>(predicates)));
}

int main() {
    int calls = 0;

    const bool passed = all_of(
        [&] { ++calls; return true; },
        [&] { ++calls; return false; },
        [&] { ++calls; return true; });  // 不会被调用。

    assert(!passed);
    assert(calls == 2);
}
```

这也是为什么不能把 `&&`/`||` 的折叠机械替换成 `&`/`|`：后两者是位运算，不会短路。

### 一个可靠的“逐个执行”模板

打印、注册回调、批量销毁、依次 `emplace_back` 都属于“对每个元素执行一次副作用”的场景。逗号折叠是最直接、也最容易审核的工具：

```cpp
#include <iostream>
#include <utility>

template <typename... Ts>
void print_lines(Ts&&... values) {
    (static_cast<void>(
         std::cout << std::forward<Ts>(values) << '\\n'),
     ...);
}

int main() {
    print_lines("id", 42, 3.5);
}
```

展开后的代码等价于依次执行三次输出。它能自然接受空包：`print_lines()` 什么也不做。

这里把整个输出表达式转换为 `void`，再进行逗号折叠，是一个值得记住的工程习惯：它明确表达“我只关心副作用”，同时固定使用内建逗号运算符。

### 类型包的常见标准库用法：`tuple` 和 `variant` 访问器

参数包不只用于函数实参。`std::tuple<Ts...>` 本身就是“按位置保存一组异构类型”的标准库类型。下面的 `capture_values` 把任意个输入按值捕获到 `tuple` 中：

```cpp
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

template <typename... Ts>
auto capture_values(Ts&&... values) {
    // decay_t 去掉引用、cv 限定和数组函数类型的特殊形式，令 tuple 拥有自己的值。
    return std::tuple<std::decay_t<Ts>...>{std::forward<Ts>(values)...};
}

int main() {
    const auto record = capture_values(7, "ready", std::string{"done"});
    static_assert(std::tuple_size_v<decltype(record)> == 3);
}
```

它与 `std::forward_as_tuple` 的设计目标不同：这里选择按值保存，所以 `record` 不会引用临时对象。后者保存的是引用，生命周期约束更严格，等学习转发与引用折叠时再使用。

另一个经常见到包展开的模式是把多个 lambda 合成一个重载集。它特别适合 `std::visit`：

```cpp
#include <string>
#include <utility>
#include <variant>

template <typename... Fs>
struct Overload : Fs... {
    using Fs::operator()...;
};

// C++17 需要这个推导指引，才能写 Overload{lambda1, lambda2}。
template <typename... Fs>
Overload(Fs...) -> Overload<Fs...>;

int main() {
    std::variant<int, std::string> result{
        std::in_place_type<std::string>, "done"};

    std::visit(
        Overload{
            [](int value) { /* 处理整数 */ },
            [](const std::string& text) { /* 处理文本 */ }},
        result);
}
```

`struct Overload : Fs...` 将每个 lambda 类型都展开为一个基类；`using Fs::operator()...` 再逐个引入它们的调用运算符，于是这个对象拥有多个重载版本。这个例子说明：包展开不仅是“多传几个参数”，还可以组合类型结构。

### 常量参数包：`std::index_sequence`

前面的 `Ts...` 是类型包。`std::size_t... Indices` 则是常量参数包，常用于给异构 `tuple` 生成 `0, 1, 2, ...` 这样的编译期下标。

```cpp
#include <cstddef>
#include <iostream>
#include <tuple>
#include <utility>

template <typename Tuple, std::size_t... Indices>
void print_tuple_impl(const Tuple& values, std::index_sequence<Indices...>) {
    (static_cast<void>(
         std::cout << std::get<Indices>(values) << '\\n'),
     ...);
}

template <typename... Ts>
void print_tuple(const std::tuple<Ts...>& values) {
    print_tuple_impl(values, std::index_sequence_for<Ts...>{});
}

int main() {
    print_tuple(std::tuple{42, 3.5, "ready"});
}
```

执行过程是：

1. `std::index_sequence_for<Ts...>` 根据 `Ts...` 生成 `std::index_sequence<0, 1, 2>`。
2. `print_tuple_impl` 中的 `Indices...` 变成 `0, 1, 2`。
3. `std::get<Indices>(values)...` 或示例中的输出模式因而能分别访问每个位置。

`index_sequence` 是读懂标准库泛型代码的重要桥梁：类型包说明“有几种元素类型”，索引包让代码可以在编译期逐一访问这些位置。

当只是把一个 tuple 的元素交给可调用对象时，C++17 的 `std::apply` 往往更简洁，不必手写下标包：

```cpp
#include <iostream>
#include <tuple>

int main() {
    const auto values = std::tuple{42, 3.5, "ready"};

    std::apply(
        [](const auto&... items) {
            (static_cast<void>(std::cout << items << '\\n'), ...);
        },
        values);
}
```

`std::apply` 在内部做了与 `index_sequence` 类似的事。先优先使用它；只有需要在多个位置复用下标、构造新的索引映射或实现 tuple 工具时，才直接写 `index_sequence`。

### 什么时候不该用参数包

参数包的长度是调用表达式的一部分，适合少量、固定、异构或需要编译期分派的输入。以下需求通常该用范围或容器：

- 元素数量在运行时才知道，例如读取文件的所有行。
- 对同类型大数据逐个处理，例如一百万个 `int`。
- 需要按索引插入、删除、排序、筛选或多次遍历。

例如 `sum(1, 2, 3)` 适合演示语法；真实的长序列求和应使用 `std::accumulate` 或 C++20 ranges 算法。模板不会让运行时数据神奇地变成编译期参数。

### 常见错误与检查顺序

| 现象 | 原因与处理方式 |
| --- | --- |
| `sizeof...(T)` 报错 | 只有包才能写 `sizeof...`；单个类型参数 `T` 应写 `sizeof(T)`。 |
| `return values + ...;` 报语法错 | 折叠表达式整体必须有括号：`return (values + ...);`。 |
| `sum()` 无法编译 | `+` 的一元折叠没有空包身份值；加初值 `(0 + ... + values)` 或用断言拒绝。 |
| 副作用顺序不稳定 | 不要在 `+`、`*` 等折叠的每项里改共享状态；改用 `&&`、逻辑或折叠，或转换为 `void` 的逗号折叠。 |
| `std::forward<T>(value)...` 报错或丢失移动语义 | `T` 与 `value` 必须是同一对参数包；只有转发引用接收到的参数才应这样转发。 |
| 想把每个 lambda 都当作一个重载 | 使用 `Overload : Fs...` 和 `using Fs::operator()...`，不要为每个数量手写一个类。 |

### 这一节的设计判断

- **合成一个值**：用折叠，例如总和、全部通过、任一命中。
- **按顺序执行多个动作**：用转换为 `void` 的逗号折叠。
- **转交任意个参数**：用 `target(std::forward<Ts>(values)...)` 包展开。
- **保存异构的值**：考虑 `std::tuple<std::decay_t<Ts>...>`；只要同类型动态序列，就用 `std::vector`。
- **访问 tuple 的每个位置**：优先 `std::apply`；确需位置编号时再用 `std::index_sequence`。
- **需要运行时可变数量**：不要使用参数包，改用容器、迭代器或 range。

掌握到这里就足够覆盖 C++17/20 项目中绝大多数参数包代码。下一次遇到 `Args&&... args` 时，先找它后面哪个模式带了 `...`；那一处就是参数真正被展开、转发或折叠的地方。

## 完全特化与偏特化：为一类模板实参换实现

可以把通用模板看作默认实现，把特化看作“某个类型模式需要特殊实现”。

### 完全特化：针对一个确切实参

```cpp
#include <string_view>

template <typename T>
struct TypeLabel {
    static constexpr std::string_view value = "unknown";
};

template <>
struct TypeLabel<int> {
    static constexpr std::string_view value = "int";
};

static_assert(TypeLabel<double>::value == "unknown");
static_assert(TypeLabel<int>::value == "int");
```

`template <>` 表示没有待推导的模板参数了。完全特化也可用于函数模板和变量模板，但函数模板通常更应该使用重载，后面会说明。

### 偏特化：针对一类有共同形状的类型

偏特化只适用于**类模板和变量模板**。它保留一部分待推导参数，同时指定一个更具体的类型模式。

```cpp
#include <cstddef>
#include <string_view>

template <typename T>
struct StorageKind {
    static constexpr std::string_view value = "普通对象";
};

template <typename T>
struct StorageKind<T*> {
    static constexpr std::string_view value = "指针";
};

template <typename T, std::size_t N>
struct StorageKind<T[N]> {
    static constexpr std::string_view value = "内建数组";
};

static_assert(StorageKind<int>::value == "普通对象");
static_assert(StorageKind<int*>::value == "指针");
```

上例故意只用类型模式做分类。真实设计里，`T*` 并不能说明对象是否拥有资源，因此不要因为“指针特化很方便”就让模板替你猜所有权。所有权应使用 `std::unique_ptr<T>`、`std::shared_ptr<T>` 或明确的接口语义表达。

更贴近实际的例子是为 `std::vector<T, Allocator>` 提供序列化策略：

```cpp
#include <string>
#include <vector>

template <typename T>
struct Serializer {
    static std::string describe(const T&) {
        return "一个普通值";
    }
};

template <typename T, typename Allocator>
struct Serializer<std::vector<T, Allocator>> {
    static std::string describe(const std::vector<T, Allocator>& values) {
        return "vector, size=" + std::to_string(values.size());
    }
};
```

`Serializer<std::vector<int>>` 会匹配偏特化，其中 `T` 推导为 `int`，`Allocator` 推导为 `std::allocator<int>`。

### 为什么函数模板不能偏特化

下面这种写法是非法的：

```cpp
template <typename T>
void inspect(T value);

// template <typename T>
// void inspect<T*>(T* value);  // 错误：函数模板不支持偏特化。
```

函数模板用**重载**代替偏特化：

```cpp
#include <iostream>

template <typename T>
void inspect(const T&) {
    std::cout << "普通对象\\n";
}

template <typename T>
void inspect(T*) {
    std::cout << "指针\\n";
}

int main() {
    int number = 42;
    inspect(number);
    inspect(&number);
}
```

选择建议很简单：

- 需要为一个**类型家族的数据表示或静态成员**改变实现，用类模板偏特化。
- 需要为不同实参形式选择一个**函数操作**，优先函数重载。
- 只有确实针对一个完全确定的类型，才考虑完全特化。

### 特化不是“运行时分支”

模板特化在编译期就确定。例如 `Serializer<int>` 和 `Serializer<std::vector<int>>` 是不同类型，对应的实现已经在编译时选好。它不能根据网络包内容、配置文件或用户点击来切换。

## 类模板实参推导：CTAD

函数模板通常从函数实参推导类型；C++17 开始，类模板也可以在构造对象时推导实参，称为 CTAD（class template argument deduction，类模板实参推导）。

```cpp
#include <array>
#include <type_traits>

std::array ids{1, 2, 3, 4};  // 推导为 std::array<int, 4>。
static_assert(std::is_same_v<decltype(ids), std::array<int, 4>>);
```

这不是 `auto` 的另一种拼写。`std::array` 仍是类模板；只是编译器根据构造实参替你补上了 `<int, 4>`。

### 推导指引：规定“从什么实参得到什么类型”

当构造参数不足以推出类模板实参时，可以写 deduction guide（推导指引）。下面让整数构造的 `Meters` 默认以 `double` 保存：

```cpp
#include <type_traits>

template <typename T>
struct Meters {
    T value;
};

Meters(int) -> Meters<double>;

Meters distance{12};
static_assert(std::is_same_v<decltype(distance), Meters<double>>);
static_assert(std::is_same_v<decltype(distance.value), double>);
```

推导指引只决定 `Meters<...>` 的模板实参；对象随后仍按正常构造/聚合初始化规则初始化。

### CTAD 的边界

CTAD 适合让局部变量少写明显的类型，例如 `std::pair{key, value}`、`std::lock_guard lock{mutex}`、`std::array{1, 2, 3}`。

公开接口、返回类型、成员类型中，显式写出模板实参往往更容易读，也能避免推导到意外类型。尤其要警惕 `std::vector items{10}`：它表示一个只含元素 `10` 的 `vector<int>`，并不是含十个元素的容器。后者应写成 `std::vector<int> items(10)`。

## 花括号初始化、`std::initializer_list` 与模板推导

花括号本身通常没有类型。因此，普通的 `T` 参数无法从 `{1, 2, 3}` 推导：

```cpp
template <typename T>
void consume(T value);

// consume({1, 2, 3});  // 错误：花括号初始化列表本身不是一个 T。
```

如果接口的语义就是“接收一组同类型初值”，请明确写 `std::initializer_list<T>`：

```cpp
#include <initializer_list>

template <typename T>
constexpr T sum_list(std::initializer_list<T> values) {
    T result{};
    for (const T value : values) {
        result += value;
    }
    return result;
}

static_assert(sum_list({1, 2, 3, 4}) == 10);
```

`std::initializer_list` 很适合构造和短暂传参，不适合把底层元素的指针或引用保存到对象中。列表的底层数组生命周期很短；若类需要持有元素，应复制或移动到自己的容器。

### `std::in_place`：直接在目标对象内部构造

`std::optional` 和 `std::variant` 是模板库里经常见到 `std::in_place` 的地方。它表示“把后续参数直接用于构造内部对象”，避免先构造一个临时对象，也能消除 `variant` 多个候选类型时的歧义。

```cpp
#include <optional>
#include <string>
#include <variant>
#include <vector>

std::optional<std::vector<int>> values{std::in_place, 3, 7};
// 内部 vector 直接按 vector(3, 7) 构造，得到 {7, 7, 7}。

std::variant<int, std::string> result{
    std::in_place_type<std::string>, 4, 'x'};
// 明确选择 std::string，并在内部构造 "xxxx"。
```

这是一种很好的接口设计信号：模板类内部承载的具体类型由模板参数决定，而对象的构造参数由调用点提供。

## 依赖名：为什么有时必须写 `typename` 和 `template`

当名称依赖模板参数时，编译器在模板定义阶段不知道它究竟是类型、静态成员还是成员模板。`typename` 和 `template` 是你提供的消歧义信息。

### `typename`：告诉编译器这是一种类型

```cpp
#include <vector>

template <typename Container>
void append_default(Container& values) {
    typename Container::value_type value{};
    values.push_back(value);
}

int main() {
    std::vector<int> values;
    append_default(values);
}
```

`Container::value_type` 依赖于 `Container`。没有 `typename`，编译器不能确定 `value_type` 是类型还是静态数据成员。

在模板参数列表、基类列表等“语法已经要求类型”的位置通常不需要写 `typename`；但在普通声明中，见到 `T::Something` 并且它是类型时，应条件反射地考虑加上它。

### `template`：告诉编译器这是成员模板

```cpp
template <typename T>
struct Box {
    template <typename U>
    void assign(U value) {
        mValue = static_cast<T>(value);
    }

    T mValue{};
};

template <typename T>
void reset(Box<T>& box) {
    box.template assign<int>(0);
}
```

`box.assign < int` 在语法上也可能被读作比较表达式，所以依赖对象后调用成员模板时写成 `box.template assign<int>(0)`。

这两条规则只在名称依赖模板参数时出现。`std::vector<int>::value_type` 已经是确定类型，不需要写 `typename`。

## 模板为什么通常放在头文件

普通非模板函数可以在 `.cpp` 中定义，然后链接器把它和调用者连接起来。模板不同：编译器需要在看到 `use<int>()` 的位置同时看到模板定义，才能生成 `use<int>` 的具体代码。

```cpp
// math.hpp
#pragma once

template <typename T>
T square(T value) {
    return value * value;
}
```

```cpp
// main.cpp
#include "math.hpp"

int main() {
    return square(5);
}
```

所以模板定义通常写在 `.hpp`，或者写在被 `.hpp` 包含的 `.inl` 中。把定义只放到 `.cpp`，再在其他翻译单元使用，常常会得到 `undefined reference` 一类链接错误。

有一个例外：如果库作者明确知道只支持哪些实参类型，可以在 `.cpp` 中进行显式实例化，例如 `template class Matrix<float, 4, 4, Layout::row_major>;`。这是减少编译时间和控制二进制体积的进阶技巧，先知道它存在即可。

## 一个小型综合例子：固定容量的栈

这个例子把类型参数、常量参数、`constexpr`、`if constexpr`、默认参数和 `static_assert` 放在一起。它不是要取代 `std::vector`，而是展示每项语法在接口中负责什么。

```cpp
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

template <typename T, std::size_t Capacity, bool CheckBounds = true>
class StaticStack {
    static_assert(Capacity > 0, "StaticStack 的 Capacity 必须大于 0");

public:
    [[nodiscard]] constexpr bool empty() const noexcept {
        return mSize == 0;
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return mSize;
    }

    constexpr void push(const T& value) {
        check_can_push();
        mValues[mSize++] = value;
    }

    constexpr T pop() {
        check_can_pop();
        return mValues[--mSize];
    }

private:
    constexpr void check_can_push() const {
        if constexpr (CheckBounds) {
            if (mSize == Capacity) {
                throw std::out_of_range("StaticStack 已满");
            }
        }
    }

    constexpr void check_can_pop() const {
        if constexpr (CheckBounds) {
            if (mSize == 0) {
                throw std::out_of_range("StaticStack 为空");
            }
        }
    }

    std::array<T, Capacity> mValues{};
    std::size_t mSize = 0;
};

constexpr int evaluate() {
    StaticStack<int, 4> stack;
    stack.push(3);
    stack.push(5);
    return stack.pop() + stack.pop();
}

static_assert(evaluate() == 8);

int main() {
    StaticStack<int, 1024, false> unchecked_stack;
    unchecked_stack.push(42);
}
```

这里有两层检查：

- `Capacity > 0` 是类型本身永远不应违反的规则，适合 `static_assert`。
- 栈是否已满取决于对象的当前状态，仍是运行时条件；但 `CheckBounds` 是编译期开关，`if constexpr` 能让关闭检查的实例完全没有边界检查分支。

设计时最重要的是先问：**这个差异是否应当成为类型的一部分？** 固定容量和检查策略可能值得；栈当前有多少元素显然不值得。

## 这一章的常见误区

| 误区 | 更准确的理解 |
| --- | --- |
| `constexpr` 函数一定在编译期运行。 | 只有实参与上下文都满足常量求值条件时才会在编译期运行。 |
| `if constexpr` 只是更快的 `if`。 | 它会在编译期丢弃未选分支，能改变一个模板实例中哪些代码需要合法。 |
| 函数模板也能像类模板一样偏特化。 | 函数模板没有偏特化，用重载、`if constexpr` 或后续的 concepts。 |
| `T*` 特化能判断所有权。 | 指针类型不携带所有权语义；应借助智能指针或明确命名表达。 |
| CTAD 总能推导出我想要的类型。 | 它只按构造函数/推导指引工作；公开接口要警惕隐式推导。 |
| `{...}` 可以自动推导任何 `T`。 | 花括号没有独立类型；需要 `std::initializer_list<T>` 或明确的构造接口。 |
| 模板实现和普通函数一样放 `.cpp` 就行。 | 实例化点通常必须看见定义，因此模板大多定义在头文件。 |

## 练习

不要直接看答案，先自己实现并让下面的断言通过。

1. 写一个 `Clamp<T, Min, Max>`，其中 `Min` 和 `Max` 是常量模板参数。用 `static_assert(Min <= Max)` 检查配置，并提供 `static constexpr T apply(T value)`。
2. 写一个 `IsPointerLike<T>` 变量模板，默认值为 `false`；对 `T*` 和 `std::unique_ptr<T>` 偏特化为 `true`。思考：它为什么仍不能表达共享所有权？
3. 写一个 `make_array` 函数模板，接收任意个同类或可转换参数，返回 `std::array`；空参数包应被拒绝。提示：需要参数包、`sizeof...`、`std::common_type_t` 与折叠/花括号初始化。
4. 为 `Bytes<T>` 写一个推导指引：传入 `int` 时推导为 `Bytes<std::size_t>`，传入 `double` 时推导为 `Bytes<double>`。比较显式构造与 CTAD 调用点的可读性。
5. 把 `StaticStack` 的边界策略由 `bool CheckBounds` 改成策略类型 `Checked` / `Unchecked`。先只用 `if constexpr` 实现；策略类真正有行为时，再考虑把行为移入类型。

## 复习清单

- 类型变化用 `typename T`，编译期配置变化用常量模板参数。
- `using Name = ...` 用于别名模板，`inline constexpr` 变量模板用于类型相关常量。
- 用 `constexpr` 提供可常量求值的实现；只有必须强制编译期时才用 `consteval`。
- 用 `static_assert` 表达模板实参不变量，用 `if constexpr` 表达编译期分派。
- 类模板和变量模板可偏特化；函数模板使用重载。
- CTAD 是便利工具，不是让类型消失；花括号初始化会影响推导与重载选择。
- 看到依赖名 `T::X`，先判断 `X` 是不是类型；看到依赖对象调用成员模板，考虑加 `template`。

下一章适合学习标准库类型萃取与类型变换：如何用 `std::is_same_v`、`std::is_integral_v`、`std::remove_cvref_t`、`std::decay_t`、`std::is_invocable_v` 把“类型条件”写得简洁而可维护。
