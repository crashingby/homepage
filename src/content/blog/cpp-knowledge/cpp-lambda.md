---
title: C++ Lambda 表达式：从 C++11 到 C++20
date: 2026-08-30
tags: [CPP, C++20, Lambda]
summary: 系统介绍 Lambda 表达式的语法演进、闭包类型、捕获与生命周期、拷贝移动语义，以及 C++20 下的常见用法和陷阱。
---

# C++ Lambda 表达式：从语法糖到闭包对象

Lambda 表达式用于在需要可调用对象的位置就地定义行为。它常出现在 STL 算法、回调、异步任务、访问器和局部泛型代码中。

理解 Lambda 不能只停留在 `[](...) { ... }` 的写法上。更可靠的心智模型是：

> **每个 Lambda 表达式都会创建一个唯一、匿名的类类型；表达式求值后得到这个类的对象，捕获通常成为对象状态，函数体成为 `operator()`。**

这个匿名类叫作**闭包类型（closure type）**，它的对象叫作**闭包对象（closure object）**。一旦接受这个模型，按值捕获、`mutable`、对象大小、拷贝与移动、函数指针转换和生命周期问题都会变得容易解释。

本文讨论的语言边界是 **C++20**。

## 先记住这些结论

- `[]` 必不可少，它不仅是语法标记，也是**捕获列表**；空的 `[]` 表示没有捕获能力。
- `[x]` 在创建闭包对象时保存 `x` 的值，`[&x]` 保存对原对象的引用语义；二者的首要差异是**状态所有权与生命周期**，而不只是能否修改。
- 按值捕获默认不能在函数体中修改，因为生成的 `operator()` 默认是 `const`；`mutable` 修改的是闭包中的副本，不会让原变量跟着变化。
- `[=]` 在 C++20 中不会按值复制当前对象。它隐式捕获的是 `this` 指针，且这种隐式捕获方式从 C++20 起被弃用；需要明确写 `[this]`、`[=, this]` 或 `[*this]`。
- `[*this]` 从 C++17 起复制当前对象；它不是移动捕获。需要移动时使用 C++14 的初始化捕获，例如 `[self = std::move(*this)]`。
- 闭包能否拷贝或移动取决于其捕获成员。捕获 `std::unique_ptr` 的闭包通常只能移动，不能拷贝。
- C++20 只有**语法上没有任何 lambda-capture 的 Lambda** 才能默认构造和赋值。`[=] { return 1; }` 虽未实际使用外部变量，仍不等价于 `[] { return 1; }`。
- 只有无捕获 Lambda 才能隐式转换为函数指针；有捕获闭包需要额外状态，普通函数指针无处保存这些状态。
- 按引用捕获不会延长对象生命周期，按值捕获指针、引用包装器或 `std::string_view` 也不会自动拥有其指向的数据。
- 捕获本身不提供线程同步。多个线程通过引用捕获访问同一对象时，仍需互斥量、原子变量或其他同步手段。

## 从 C++11 到 C++20 的语法演进

| 标准 | 与 Lambda 直接相关的主要变化 | 解决的问题 |
| --- | --- | --- |
| C++11 | 引入 Lambda、显式/默认捕获、`mutable`、尾置返回类型、无捕获 Lambda 到函数指针的转换 | 就地创建带少量状态的函数对象 |
| C++14 | 泛型 Lambda、初始化捕获、移动捕获、放宽返回类型推导 | 让 Lambda 支持泛型参数和自定义捕获初始化 |
| C++17 | `constexpr` Lambda、`[*this]` 按值捕获当前对象 | 支持编译期调用和安全复制当前对象 |
| C++20 | 显式模板形参列表、约束、`consteval`、无捕获闭包可默认构造/赋值、Lambda 可出现在不求值语境、初始化捕获包展开、`[=, this]`、结构化绑定捕获 | 使 Lambda 更接近完整的局部函数对象模板，并改善其类型能力 |

### C++11：基础 Lambda 与捕获

C++11 奠定了基本形态：捕获列表、形参列表、说明符、返回类型和函数体。

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    std::vector<int> values{4, 1, 3, 2};
    const bool descending = true;

    // descending 在创建 comparator 时被复制进闭包对象。
    const auto comparator = [descending](int left, int right) -> bool {
        return descending ? left > right : left < right;
    };

    std::sort(values.begin(), values.end(), comparator);

    for (const int value : values) {
        std::cout << value << ' ';
    }
}
```

C++11 的返回类型推导比较受限。省略尾置返回类型时，简单的单条 `return` 容易推导；复杂控制流通常需要显式写出 `-> ReturnType`。

```cpp
const auto normalize = [](int value) -> int {
    if (value < 0) {
        return -value;
    }
    return value;
};
```

### C++14：泛型 Lambda 与初始化捕获

C++14 允许在形参中使用 `auto`。这种 Lambda 的闭包类型拥有一个**函数调用运算符模板**，每种实参类型会实例化对应的 `operator()`。

```cpp
#include <string>
#include <type_traits>
#include <utility>

int main() {
    const auto add = [](const auto& left, const auto& right) {
        return left + right;
    };

    static_assert(add(2, 3) == 5);
    const std::string text = add(std::string{"C++"}, std::string{"20"});
    static_assert(std::is_same_v<decltype(text), const std::string>);
}
```

初始化捕获（init-capture，也常称广义捕获）允许在捕获列表中创建一个新的闭包成员。它可以改名、计算初值，也可以把只移动类型转入闭包。

```cpp
#include <memory>
#include <utility>

int main() {
    auto resource = std::make_unique<int>(42);

    // 创建 task 时执行 std::move；resource 随后变为空。
    auto task = [owned_value = std::move(resource)]() {
        return *owned_value;
    };

    if (resource != nullptr) {
        return 1;  // 不应执行。
    }
    return task() == 42 ? 0 : 1;
}
```

C++14 还统一并放宽了返回类型推导。多个 `return` 可以参与推导，但推导结果必须一致；花括号初始化列表不能单独提供可推导类型。

```cpp
const auto absolute = [](int value) {
    if (value < 0) {
        return -value;
    }
    return value;  // 两个分支都是 int。
};

// const auto invalid = [](bool first) {
//     if (first) {
//         return 1;     // int
//     }
//     return 2.0;       // double：错误，推导结果不一致。
// };

// const auto also_invalid = [] {
//     return {1, 2, 3}; // 错误：不能从 braced-init-list 推导返回类型。
// };
```

### C++17：`constexpr` 与 `[*this]`

C++17 允许 Lambda 参与常量表达式。满足 `constexpr` 函数要求时，即使不显式写 `constexpr`，调用运算符也可以是 `constexpr`。

```cpp
constexpr auto square = [](int value) {
    return value * value;
};

static_assert(square(6) == 36);
```

显式写 `constexpr` 可以直接表达设计意图：

```cpp
constexpr auto clamp_to_byte = [](int value) constexpr {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return value;
};

static_assert(clamp_to_byte(300) == 255);
```

C++17 还引入 `[*this]`，用于把当前对象复制进闭包。它解决了只捕获 `this` 指针时，闭包可能比原对象活得更久的问题。

```cpp
#include <string>
#include <utility>

class Request {
public:
    explicit Request(std::string path) : mPath(std::move(path)) {}

    [[nodiscard]] auto makeReader() const {
        // 闭包保存整个 Request 的副本，不依赖原 Request 的生命周期。
        return [*this] {
            return mPath;
        };
    }

private:
    std::string mPath;
};
```

### C++20：Lambda 更接近局部函数模板

C++20 可以在 `[]` 后显式书写模板形参，从而为类型命名、表达多个形参之间的关系，并添加约束。

```cpp
#include <concepts>
#include <type_traits>

int main() {
    const auto same_type_max = []<typename T>(T left, T right) {
        return left < right ? right : left;
    };

    static_assert(same_type_max(3, 5) == 5);
    // same_type_max(3, 5.0);  // 错误：两个参数要求推导为同一个 T。

    const auto twice = []<std::integral T>(T value) {
        return static_cast<T>(value + value);
    };

    static_assert(twice(21) == 42);
    static_assert(std::is_same_v<decltype(twice(2L)), long>);
}
```

C++20 新增的其他关键能力会在后文分别展开：

- 无捕获闭包可以默认构造，也可以拷贝/移动赋值。
- Lambda 可以直接出现在 `decltype` 等不求值语境中。
- 无捕获闭包可作为常量模板参数。
- 初始化捕获可以展开参数包。
- 可以显式写 `[=, this]`；与此同时，`[=]` 对 `this` 的隐式捕获被弃用。
- 可以捕获结构化绑定引入的名字。
- 可以用 `consteval` 声明必须在编译期调用的 Lambda。

## C++20 完整语法拆解

常用语法可以概括为：

```cpp
[捕获列表]
<模板形参列表> requires 约束
(函数形参列表)
mutable constexpr或consteval noexcept说明
-> 返回类型
requires 尾部约束
{
    函数体
}
```

除 `[]` 和函数体外，其余部分都可以按规则省略。换成真实代码如下：

```cpp
#include <concepts>
#include <type_traits>

int main() {
    int offset = 3;

    auto transform = [offset]<typename T>(T value) mutable noexcept
        -> T
        requires std::is_arithmetic_v<T>
    {
        // mutable 允许修改闭包对象中的 offset 副本。
        ++offset;
        return static_cast<T>(value + offset);
    };

    return transform(10) == 14 ? 0 : 1;
}
```

各部分的含义如下：

- **捕获列表**：决定闭包需要从外层环境保存什么状态，以及保存值还是引用语义。
- **模板形参列表**：C++20 新增；让泛型 Lambda 可以显式命名类型和常量模板参数。
- **函数形参列表**：与普通函数类似，可以使用默认实参和参数包。
- **`mutable`**：移除生成的 `operator()` 上默认的 `const` 限定，允许修改按值捕获的成员。
- **`constexpr`**：声明调用运算符为 `constexpr`。满足条件时也可能隐式成为 `constexpr`。
- **`consteval`**：C++20 新增；声明调用运算符为立即函数，每次可能求值的调用都必须产生编译期常量。
- **`noexcept`**：把异常说明应用到调用运算符；它还会影响无捕获 Lambda 转换后的函数指针类型。
- **尾置返回类型**：在需要保留引用、解决推导歧义或明确接口时使用。
- **`requires`**：可以约束函数调用运算符模板；最常见的写法位于形参列表之后。

`constexpr` 与 `consteval` 不能同时出现。C++20 也不能在 Lambda 上写 `static`；静态调用运算符是 C++23 才加入的能力。

最短写法 `[] { return 42; }` 等价于省略了空形参列表：

```cpp
const auto answer = [] { return 42; };
const auto explicit_answer = []() { return 42; };
```

一旦要写 `mutable`、`noexcept`、尾置返回类型等 Lambda declarator 内容，在 C++20 中就应保留 `()`：

```cpp
auto counter = [count = 0]() mutable noexcept -> int {
    return ++count;
};
```

## 本质：编译器生成的闭包类型

### 一个近似展开

下面的 Lambda：

```cpp
int base = 10;
int calls = 0;

auto add = [base, &calls](int value) mutable {
    ++calls;
    ++base;
    return base + value;
};
```

可以用下面的手写函数对象建立心智模型：

```cpp
class ApproximateClosure {
public:
    ApproximateClosure(int base, int& calls)
        : mBase(base), mCalls(calls) {}

    int operator()(int value) {
        ++mCalls;
        ++mBase;
        return mBase + value;
    }

private:
    int mBase;    // 近似表示按值捕获。
    int& mCalls;  // 近似表示按引用捕获。
};
```

这只是便于理解的近似实现，不是标准要求编译器必须生成的布局。特别要注意：

- 真正的闭包类型是**唯一、匿名、非联合类类型**，程序不能直接写出其编译器内部名称。
- 捕获成员的名字、声明顺序和按引用捕获的具体表示方式没有暴露给程序。
- 标准允许实现改变闭包的大小、对齐、是否 trivially copyable（可平凡复制）和是否 standard-layout（标准布局）。不要依赖猜测出的内存布局。
- 闭包类型不是聚合类型，不能把它当成普通聚合体自行初始化成员。
- 非泛型 Lambda 生成普通的 `operator()`；泛型 Lambda 生成 `operator()` 模板。
- `operator()` 是 `public inline` 成员函数，默认带 `const`，使用 `mutable` 后不带 `const`。

### 每个 Lambda 表达式都有不同类型

即使两个 Lambda 的文本完全相同，它们也是两个不同的表达式，因此闭包类型不同。

```cpp
#include <type_traits>

int main() {
    const auto first = [](int value) { return value + 1; };
    const auto second = [](int value) { return value + 1; };

    static_assert(!std::is_same_v<decltype(first), decltype(second)>);

    // 同一个对象的副本当然仍是同一种闭包类型。
    const auto first_copy = first;
    static_assert(std::is_same_v<decltype(first), decltype(first_copy)>);
}
```

需要保存具体 Lambda 类型时，通常使用 `auto` 或 `decltype`：

```cpp
const auto predicate = [](int value) { return value > 0; };
using Predicate = decltype(predicate);

Predicate predicate_copy = predicate;
```

### 捕获通常就是闭包的状态

按值捕获一般对应闭包中的未命名非静态数据成员。闭包对象越多，捕获的大对象就可能被复制越多次。

```cpp
#include <array>
#include <cstddef>

int main() {
    const std::array<int, 1024> lookup_table{};

    // 通常会把整个数组复制进闭包；随后复制闭包还会继续复制它。
    const auto by_value = [lookup_table](std::size_t index) {
        return lookup_table[index];
    };

    // 只保留引用语义，但调用期间 lookup_table 必须仍然存活。
    const auto by_reference = [&lookup_table](std::size_t index) {
        return lookup_table[index];
    };

    return by_value(0) + by_reference(0);
}
```

不要假定无捕获闭包的 `sizeof` 为 `0`。C++ 对象必须有可区分的地址，空闭包通常占至少一个字节，具体大小仍由实现决定。

## 捕获：值、引用与所有权

### 捕获写法速查

| 写法 | 含义 | 主要风险 |
| --- | --- | --- |
| `[]` | 不捕获局部自动变量 | 无外部状态可用 |
| `[x]` | 按值捕获 `x` | 复制成本；可能只是浅拷贝 |
| `[&x]` | 按引用捕获 `x` | `x` 可能先于闭包销毁 |
| `[=]` | 对被使用的局部变量默认按值捕获 | 捕获范围不直观；对 `this` 的行为容易误解 |
| `[&]` | 对被使用的局部变量默认按引用捕获 | 闭包逃逸时容易产生悬空引用 |
| `[=, &x]` | 默认按值，`x` 例外按引用 | 同时存在两种生命周期语义 |
| `[&, x]` | 默认按引用，`x` 例外按值 | 同时存在两种生命周期语义 |
| `[name = expr]` | 用 `expr` 初始化新的按值捕获成员 | 类型按 `auto` 规则推导 |
| `[&name = expr]` | 创建新的引用捕获别名 | 被引用对象必须足够长寿 |
| `[this]` | 复制 `this` 指针 | 不延长当前对象生命周期 |
| `[*this]` | 复制当前对象 | 复制成本；要求当前对象可复制 |

默认捕获不是“立刻捕获作用域中的所有变量”。通常只有在 Lambda 中被需要的实体才会隐式捕获。不过捕获集合按语言规则从语法确定，不能把编译器优化掉某个成员理解为没有捕获语义。

### 按值捕获是创建时的快照

```cpp
#include <cassert>

int main() {
    int value = 10;
    const auto snapshot = [value] {
        return value;
    };

    value = 20;

    assert(snapshot() == 10); // 闭包保存的是创建时的值。
    assert(value == 20);
}
```

按值捕获并不等于深拷贝。捕获裸指针、智能指针、容器、视图时，实际所有权由该类型自己的复制语义决定。

```cpp
#include <string>
#include <string_view>

auto make_dangerous_view() {
    std::string text = "temporary";

    // string_view 按值复制了，但它仍不拥有 text 的字符数组。
    return [view = std::string_view{text}] {
        return view;
    };
} // text 在这里销毁；返回闭包中的 view 已悬空。
```

### 按引用捕获观察原对象

```cpp
#include <cassert>

int main() {
    int value = 10;
    const auto observer = [&value] {
        return value;
    };

    value = 20;
    assert(observer() == 20);
}
```

`const auto observer` 并不会让被引用的 `value` 变成 `const`。闭包的 `operator() const` 约束的是闭包自身，不是闭包引用的外部对象。

```cpp
int value = 0;

const auto increment = [&value] {
    ++value;  // 合法：修改的是闭包之外的对象。
};

increment();
```

### 捕获引用变量时要区分两层引用

如果外层变量本身是引用，按值捕获保存的是它所引用对象的值；按引用捕获继续引用原对象。

```cpp
#include <cassert>

int main() {
    int value = 10;
    int& alias = value;

    auto copied_value = [alias]() mutable {
        ++alias;  // 修改闭包中的 int 副本。
        return alias;
    };

    const auto referenced_value = [&alias] {
        ++alias;  // 修改 alias 绑定的原始 value。
        return alias;
    };

    assert(copied_value() == 11);
    assert(value == 10);
    assert(referenced_value() == 11);
    assert(value == 11);
}
```

### `mutable` 只放宽闭包自身的常量性

默认生成的调用运算符近似如下：

```cpp
ReturnType operator()(Parameters...) const;
```

所以按值捕获的成员默认只能读取。添加 `mutable` 后，调用运算符近似变成非 `const` 成员函数：

```cpp
#include <cassert>

int main() {
    int original = 0;

    auto next = [count = original]() mutable {
        return ++count;
    };

    assert(next() == 1);
    assert(next() == 2);
    assert(original == 0); // 修改的始终是闭包成员。
}
```

`mutable` 不能移除捕获对象类型本身的 `const`。简单捕获会保留源变量的类型，而初始化捕获按 `auto` 规则推导，通常会丢弃顶层 `const`：

```cpp
int main() {
    const int value = 10;

    auto simple_capture = [value]() mutable {
        // ++value; // 错误：闭包成员的类型仍是 const int。
        return value;
    };

    auto init_capture = [copy = value]() mutable {
        // copy 按 auto 规则推导为 int，顶层 const 被移除。
        return ++copy;
    };

    return simple_capture() + init_capture();
}
```

另一个容易忽略的结果是：`mutable` Lambda 不能通过 `const` 闭包对象调用，因为它只有非 `const operator()`。

```cpp
auto mutable_lambda = [count = 0]() mutable {
    return ++count;
};

const auto const_closure = mutable_lambda;
// const_closure(); // 错误：不能在 const 对象上调用非 const operator()。
```

### 初始化捕获的类型近似按 `auto` 推导

初始化捕获不是先创建一个隐藏局部变量再复制一次。可以把 `[name = expr]` 理解为闭包直接拥有一个由 `auto` 推导、用 `expr` 初始化的成员。

```cpp
#include <functional>

int main() {
    int value = 1;

    auto by_value = [copy = value]() mutable {
        return ++copy;
    };

    auto by_alias = [&alias = value] {
        return ++alias;
    };

    auto by_reference_wrapper = [wrapped = std::ref(value)] {
        return ++wrapped.get();
    };

    by_value();             // 不修改 value。
    by_alias();             // value 变成 2。
    by_reference_wrapper(); // value 变成 3。
    return value == 3 ? 0 : 1;
}
```

`std::reference_wrapper` 本身虽然按值存进闭包，但仍只是引用语义，不延长原对象生命周期。

### 数组按值捕获会复制元素

简单按值捕获数组时，闭包保存数组成员，并按下标递增顺序直接初始化各元素；它不会像普通表达式那样自动退化为指针。

```cpp
#include <cassert>

int main() {
    int values[3]{1, 2, 3};

    const auto snapshot = [values] {
        return values[0] + values[1] + values[2];
    };

    values[0] = 100;
    assert(snapshot() == 6);
}
```

### 全局变量和静态局部变量不需要捕获

捕获主要针对具有自动存储期的局部实体。全局变量和静态局部变量有自己的存储期，Lambda 可以直接按普通名字查找规则访问它们。

```cpp
int global_count = 0;

void update_counts() {
    static int local_static_count = 0;

    const auto update = [] {
        ++global_count;
        ++local_static_count;
    };

    update();
}
```

这里不是“按引用捕获了全局变量”，而是根本没有捕获它们。

### C++20：捕获结构化绑定

```cpp
#include <cassert>
#include <utility>

int main() {
    const auto [key, value] = std::pair{7, 35};

    const auto sum = [key, value] {
        return key + value;
    };

    assert(sum() == 42);
}
```

为了让所有权和生命周期清晰，工程代码中优先按值捕获结构化绑定；若要按引用观察被分解对象，先确认编译器对相关 C++20 缺陷报告的实现情况，并保证底层对象存活。

### 混合默认捕获的语法规则

```cpp
int first = 1;
int second = 2;

auto mostly_value = [=, &second] {
    ++second;
    return first + second;
};

auto mostly_reference = [&, first] {
    ++second;
    return first + second;
};

// [&, &second] {}; // 错误：默认已经按引用，例外项应改为按值。
// [=, first] {};   // 错误：默认已经按值，不能再重复写按值 simple-capture。
```

代码审查时通常优先显式列出少量捕获，例如 `[threshold, &result]`。它能让闭包持有什么状态一眼可见，也能减少函数后续修改时意外扩大捕获集合。

## 泛型 Lambda 与 C++20 模板形参

### `auto` 形参会生成模板形参

```cpp
const auto less = [](const auto& left, const auto& right) {
    return left < right;
};

static_assert(less(1, 2));
static_assert(less(1.5, 2.5));
```

它可近似理解为闭包中存在：

```cpp
struct ApproximateGenericClosure {
    template <typename Left, typename Right>
    bool operator()(const Left& left, const Right& right) const {
        return left < right;
    }
};
```

每个 `auto` 通常对应一个独立的模板参数，因此 `[](auto left, auto right)` 默认允许左右两边类型不同。

### 显式模板形参可以表达类型关系

```cpp
#include <cstddef>
#include <span>

int main() {
    const auto first_or = []<typename T>(std::span<const T> values, T fallback) {
        return values.empty() ? fallback : values.front();
    };

    const int values[]{1, 2, 3};
    return first_or(std::span<const int>{values}, 0) == 1 ? 0 : 1;
}
```

如果只使用 `auto`，经常需要写 `decltype(parameter)` 才能重新引用参数类型。显式模板形参让类型关系更直观：

```cpp
#include <utility>
#include <vector>

const auto append = []<typename T>(std::vector<T>& output, T value) {
    output.push_back(std::move(value));
};
```

### 用 Concept 约束调用范围

```cpp
#include <concepts>

const auto gcd = []<std::integral T>(T left, T right) {
    while (right != 0) {
        const T remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
};

static_assert(gcd(48, 18) == 6);
// gcd(4.0, 2.0); // 错误：double 不满足 std::integral。
```

也可以使用尾部 `requires` 表达多类型关系：

```cpp
#include <concepts>

const auto add_same = []<typename Left, typename Right>(Left left, Right right)
    requires std::same_as<Left, Right>
{
    return left + right;
};

static_assert(add_same(20, 22) == 42);
```

### 完美转发与 `decltype(auto)`

泛型 Lambda 中的 `auto&&` 是转发引用。若包装另一个可调用对象，应同时转发参数并用 `decltype(auto)` 保留返回值的引用类别。

C++14 使用 `auto&&` 时，编译器会为每个 `auto` 发明一个模板形参。由于代码中没有对应的模板形参名字，转发时需要使用 `decltype(args)` 取得各参数推导后的类型：

```cpp
#include <functional>
#include <utility>

template <typename Callable>
auto make_logger_cpp14(Callable callable) {
    return [callable = std::move(callable)](auto&&... args) mutable
        -> decltype(auto)
    {
        // 这里可以在调用前后增加日志、计时或统计。
        return std::invoke(
            callable,
            std::forward<decltype(args)>(args)...
        );
    };
}
```

C++20 可以给泛型 Lambda 显式写模板形参列表。此时它与普通函数模板的 `Args&&...` 写法几乎完全一致，也不再需要通过 `decltype(args)` 间接取得类型：

```cpp
#include <functional>
#include <utility>

template <typename Callable>
auto make_logger_cpp20(Callable callable) {
    return [callable = std::move(callable)]<typename... Args>(Args&&... args) mutable
        -> decltype(auto)
    {
        // Args 由调用实参推导，所以 Args&&... 是转发引用。
        return std::invoke(
            callable,
            std::forward<Args>(args)...
        );
    };
}
```

两种写法的对应关系如下：

| C++14 泛型 Lambda | C++20 显式模板 Lambda | 含义 |
| --- | --- | --- |
| `auto&&... args` | `<typename... Args>(Args&&... args)` | 为每个实参独立推导类型和值类别 |
| `decltype(args)` | `Args` | 作为 `std::forward` 的模板实参；两者不一定是相同类型，但会产生相同的转发结果 |
| `std::forward<decltype(args)>(args)...` | `std::forward<Args>(args)...` | 恢复每个实参原来的左值或右值类别 |

例如传入一个 `int` 左值时，C++20 的 `Args` 推导为 `int&`；传入一个 `int` 右值时，`Args` 推导为 `int`。对应的 `Args&&` 经过引用折叠后分别成为 `int&` 和 `int&&`。C++14 写法不能直接使用编译器发明的模板形参名，因此选择 `decltype(args)` 配合 `std::forward` 达到相同效果。

它还可以近似理解成普通函数模板中的以下参数形式：

```cpp
#include <functional>
#include <utility>

template <typename Callable, typename... Args>
decltype(auto) invoke_target(Callable&& callable, Args&&... args) {
    return std::invoke(
        std::forward<Callable>(callable),
        std::forward<Args>(args)...
    );
}
```

这里的关键不是看到 `&&` 就认为它一定是转发引用，而是 `Args` 必须由本次调用进行模板实参推导。`const Args&&`、已经固定类型的 `SomeType&&` 都只是右值引用，不是转发引用。

如果普通 `auto` 返回一个引用表达式，通常会像变量的 `auto` 推导一样丢掉引用；需要保留引用时要显式使用 `-> decltype(auto)` 或具体引用类型。

### C++20 初始化捕获参数包

```cpp
#include <iostream>
#include <utility>

template <typename... Args>
auto make_printer(Args&&... args) {
    // 每个实参都按自己的值类别初始化一个捕获成员。
    return [...values = std::forward<Args>(args)] {
        ((std::cout << values << ' '), ...);
        std::cout << '\n';
    };
}

int main() {
    auto printer = make_printer(1, 2.5, "C++20");
    printer();
}
```

这比先把参数包塞进 `std::tuple` 再用 `std::apply` 更直接，并且可以把移动构造的对象逐个放入闭包。

## 返回类型、`noexcept` 与编译期调用

### 返回类型推导与引用保留

```cpp
#include <vector>

int main() {
    std::vector<int> values{1, 2, 3};

    const auto front_copy = [&values] {
        return values.front(); // auto 推导为 int，返回副本。
    };

    const auto front_reference = [&values]() -> int& {
        return values.front(); // 显式保留引用。
    };

    front_reference() = 42;
    return front_copy(); // 返回 42。
}
```

返回引用时，引用指向对象的生命周期必须覆盖调用方的使用时间。不要返回指向闭包内部短命状态的引用，除非闭包对象本身明确保持存活。

### `noexcept` 是调用接口的一部分

```cpp
#include <type_traits>

int main() {
    const auto safe_add = [](int left, int right) noexcept {
        return left + right;
    };

    static_assert(noexcept(safe_add(1, 2)));
    static_assert(std::is_nothrow_invocable_r_v<int, decltype(safe_add), int, int>);
}
```

`noexcept` 不会替你捕获异常。若函数体实际抛出并逃出一个 `noexcept` 调用，程序会调用 `std::terminate`。

### `constexpr` 表示可以在编译期调用

```cpp
#include <array>
#include <cstddef>
#include <type_traits>

constexpr auto sum = [](const auto& values) {
    typename std::remove_cvref_t<decltype(values)>::value_type result{};
    for (const auto value : values) {
        result += value;
    }
    return result;
};

constexpr std::array values{1, 2, 3, 4};
static_assert(sum(values) == 10);
```

`constexpr` 不表示每次调用都发生在编译期。只要传入运行时数据、且调用位置不要求常量表达式，同一个 Lambda 仍可在运行时执行。

### C++20 `consteval` 要求每次调用都在编译期完成

```cpp
constexpr auto checked_square = [](int value) consteval {
    return value * value;
};

static_assert(checked_square(8) == 64);

int runtime_value = 8;
// int result = checked_square(runtime_value);
// 错误：runtime_value 不能用于立即函数调用。
```

`consteval` 适合编译期校验和固定配置生成，不适合需要同时接受运行时数据的通用工具。

## 闭包对象能否默认构造、拷贝、移动和赋值

这是理解 Lambda 工程行为的核心。先区分四组操作：

- **默认构造**：没有现成闭包对象，仅凭类型构造一个新对象。
- **拷贝/移动构造**：用已有闭包对象创建另一个对象。
- **拷贝/移动赋值**：两个闭包对象都已经存在，再把一个状态赋给另一个。
- **调用**：执行闭包的 `operator()`；它与上述特殊成员函数是不同能力。

### C++20 规则总表

| Lambda 形式 | 默认构造 | 拷贝构造 | 移动构造 | 拷贝/移动赋值 |
| --- | --- | --- | --- | --- |
| `[] {}` | 支持 | 默认生成 | 默认生成 | 默认生成 |
| `[=] {}` 或 `[&] {}`，即使未实际使用外部变量 | 不支持 | 默认生成 | 默认生成 | 不支持 |
| `[x] {}`、`[&x] {}`、初始化捕获 | 不支持 | 默认生成 | 默认生成 | 不支持 |

“默认生成”不等于一定可用。闭包的构造能力还要看捕获成员：

- 按值捕获成员不可拷贝时，闭包的拷贝构造会被隐式删除。
- 按值捕获成员不可移动时，闭包的移动构造可能被隐式删除，或退化为可用的复制路径，具体取决于成员类型的构造函数集合。
- 按引用捕获的闭包被复制后，各副本仍引用同一个外部对象。
- C++20 中只要语法上存在 lambda-capture，拷贝赋值运算符就是删除的，也不会得到可用的移动赋值运算符；这与捕获成员自身是否可赋值无关。

### 用类型特征直接验证

```cpp
#include <type_traits>

int main() {
    int value = 42;

    // 这里故意不把闭包对象声明为 const，否则类型特征测到的会是
    // const ClosureType 自身不可赋值，而不是 ClosureType 的特殊成员函数规则。
    auto captureless = [] { return 42; };
    auto by_value = [value] { return value; };
    auto by_reference = [&value] { return value; };
    auto default_capture_only = [=] { return 42; };

    static_assert(std::is_default_constructible_v<decltype(captureless)>);
    static_assert(std::is_copy_constructible_v<decltype(captureless)>);
    static_assert(std::is_move_constructible_v<decltype(captureless)>);
    static_assert(std::is_copy_assignable_v<decltype(captureless)>);
    static_assert(std::is_move_assignable_v<decltype(captureless)>);

    static_assert(!std::is_default_constructible_v<decltype(by_value)>);
    static_assert(std::is_copy_constructible_v<decltype(by_value)>);
    static_assert(std::is_move_constructible_v<decltype(by_value)>);
    static_assert(!std::is_copy_assignable_v<decltype(by_value)>);
    static_assert(!std::is_move_assignable_v<decltype(by_value)>);

    static_assert(!std::is_default_constructible_v<decltype(by_reference)>);
    static_assert(!std::is_copy_assignable_v<decltype(by_reference)>);

    // [=] 是一个 capture-default。即使没实际捕获 value，也不是 []。
    static_assert(!std::is_default_constructible_v<decltype(default_capture_only)>);
    static_assert(!std::is_copy_assignable_v<decltype(default_capture_only)>);
}
```

在 C++11/14/17 中，即使 `[] {}` 也没有默认构造函数，并且其拷贝赋值运算符被删除。C++20 专门放宽了真正无捕获闭包的这两项限制。

### 捕获只移动资源会产生 move-only 闭包

```cpp
#include <memory>
#include <type_traits>
#include <utility>

int main() {
    auto task = [resource = std::make_unique<int>(42)] {
        return *resource;
    };

    static_assert(!std::is_copy_constructible_v<decltype(task)>);
    static_assert(std::is_move_constructible_v<decltype(task)>);

    auto moved_task = std::move(task);
    return moved_task() == 42 ? 0 : 1;
}
```

这类闭包不能放进 C++20 的 `std::function`，因为 `std::function` 的目标必须可复制构造：

```cpp
#include <functional>
#include <memory>

int main() {
    auto move_only_task = [resource = std::make_unique<int>(42)] {
        return *resource;
    };

    // std::function<int()> task = std::move(move_only_task);
    // C++20 中错误：目标闭包不可复制。
}
```

C++20 工程中可以让接口本身接收模板参数、立即执行闭包，或选择能持有 move-only callable（只移动可调用对象）的任务容器。`std::move_only_function` 是 C++23 才加入的，不能算作 C++20 方案。

## `this`、`*this` 与对象生命周期

### `[this]` 复制的是指针

```cpp
#include <functional>

class Counter {
public:
    [[nodiscard]] std::function<int()> makeReader() {
        return [this] {
            return mValue; // 实际通过 this->mValue 访问原对象。
        };
    }

private:
    int mValue = 42;
};
```

闭包只保存 `this` 指针，不拥有 `Counter`。如果返回的回调在 `Counter` 销毁后调用，就会通过悬空指针访问对象，产生未定义行为。

### C++20 中 `[=]` 仍然只隐式捕获 `this` 指针

```cpp
class Worker {
public:
    auto makeTask() {
        return [=] {
            return mValue;
        };
    }

private:
    int mValue = 42;
};
```

这里的成员访问需要 `this`。`[=]` 并没有复制 `Worker`，而是隐式复制 `this` 指针。C++20 已弃用这种隐式捕获，编译器通常会给出警告。更明确的写法是：

```cpp
class Worker {
public:
    auto makePointerBasedTask() {
        return [this] {
            return mValue; // 依赖原对象继续存活。
        };
    }

    auto makeSelfContainedTask() const {
        return [*this] {
            return mValue; // 使用 Worker 的闭包内副本。
        };
    }

private:
    int mValue = 42;
};
```

C++20 允许写 `[=, this]`，它与 `[=]` 对 `this` 的捕获语义相同，但意图更清楚且不触发这项弃用：

```cpp
auto task = [=, this] {
    return mValue;
};
```

非静态数据成员不是外层函数中的局部变量，不能直接写成 simple-capture。只需要某个成员的独立快照时，应使用初始化捕获：

```cpp
class Worker {
public:
    auto makeValueTask() const {
        // return [mValue] { return mValue; };
        // 错误：mValue 不是可以这样捕获的局部实体。

        return [value = mValue] {
            return value; // 只保存需要的 int，不捕获 this 或整个 Worker。
        };
    }

private:
    int mValue = 42;
};
```

### `[*this]` 会复制整个当前对象

```cpp
#include <cassert>

class Counter {
public:
    auto makeIndependentCounter() {
        return [*this]() mutable {
            return ++mValue; // 修改闭包中的 Counter 副本。
        };
    }

    [[nodiscard]] int value() const noexcept {
        return mValue;
    }

private:
    int mValue = 0;
};

int main() {
    Counter original;
    auto counter = original.makeIndependentCounter();

    assert(counter() == 1);
    assert(counter() == 2);
    assert(original.value() == 0);
}
```

注意以下约束：

- `[*this]` 要求当前对象能够复制构造；它始终表达复制，不会因对象是右值就自动移动。
- 在 `const` 成员函数中，`*this` 带有 `const` 限定；即使 Lambda 写了 `mutable`，也不能借此修改这个 `const` 对象副本。需要一个可变快照时，可以显式初始化捕获 `[self = *this]`，让 `auto` 推导移除顶层 `const`。
- 复制整个对象可能很昂贵，也可能复制本不该复制的句柄或共享状态。
- 即使捕获了 `*this`，对象内部的裸指针、引用和视图仍保留原本的浅层语义。
- 若只需要少数成员，通常初始化捕获所需值更精确，例如 `[id = mId, path = mPath]`。

需要明确移动当前对象时，可以使用初始化捕获：

```cpp
#include <utility>

class MoveOnlyJob {
public:
    MoveOnlyJob() = default;
    MoveOnlyJob(const MoveOnlyJob&) = delete;
    MoveOnlyJob& operator=(const MoveOnlyJob&) = delete;
    MoveOnlyJob(MoveOnlyJob&&) noexcept = default;
    MoveOnlyJob& operator=(MoveOnlyJob&&) noexcept = default;

    auto intoTask() && {
        // 只能在右值对象上调用，把整个对象移入闭包。
        return [self = std::move(*this)]() mutable {
            self.run();
        };
    }

private:
    void run() {}
};
```

### 异步回调优先显式表达共享所有权

如果对象由 `std::shared_ptr` 管理，可捕获强引用延长生命周期，或捕获 `std::weak_ptr` 避免形成循环引用。

```cpp
#include <memory>

class Session : public std::enable_shared_from_this<Session> {
public:
    auto makeSafeCallback() {
        const std::weak_ptr<Session> weak_self = weak_from_this();

        return [weak_self] {
            if (const auto self = weak_self.lock()) {
                self->handle();
            }
            // 对象已销毁时不执行，避免通过悬空 this 访问。
        };
    }

private:
    void handle() {}
};
```

是否捕获 `shared_ptr` 还是 `weak_ptr` 是所有权设计问题：前者保证执行时对象存在，但可能形成引用环；后者不延长生命周期，调用时必须检查 `lock()`。

## Lambda、函数指针与 `std::function`

### 为什么只有无捕获 Lambda 能转换

普通函数指针只保存代码地址，没有位置保存 `x`、`this` 或其他闭包状态。因此只有**没有 lambda-capture** 的 Lambda 才提供到兼容函数指针的隐式转换。

```cpp
#include <cassert>

using BinaryOperation = int (*)(int, int);

int main() {
    const auto add = [](int left, int right) {
        return left + right;
    };

    BinaryOperation operation = add;
    assert(operation(20, 22) == 42);

    const int offset = 1;
    const auto add_offset = [offset](int value) {
        return value + offset;
    };

    // int (*invalid)(int) = add_offset;
    // 错误：函数指针无法携带 offset。
}
```

一元 `+` 常用于显式触发非泛型无捕获 Lambda 的函数指针转换：

```cpp
const auto square = [](int value) noexcept {
    return value * value;
};

using FunctionPointer = int (*)(int) noexcept;
FunctionPointer pointer = +square;
```

`noexcept` 调用运算符会得到指向 `noexcept` 函数的指针。泛型无捕获 Lambda 也提供函数指针转换模板，由目标函数指针类型决定实例化哪个签名：

```cpp
const auto identity = [](auto value) {
    return value;
};

int (*int_identity)(int) = identity;
double (*double_identity)(double) = identity;
```

### `[=] {}` 不等于真正的无捕获 Lambda

下面的闭包虽然没有保存任何局部变量，但语法上存在 capture-default，因此不提供函数指针转换：

```cpp
int main() {
    const auto truly_captureless = [] { return 42; };
    const auto has_capture_default = [=] { return 42; };

    int (*function_pointer)() = truly_captureless;
    (void)function_pointer;

    // int (*invalid)() = has_capture_default;
    // 错误：它的 lambda-introducer 含有 lambda-capture。
}
```

### Lambda、函数指针和 `std::function` 的关系

三者都能表示“可以调用的东西”，但保存信息的方式不同：

| 来源与目标 | 是否支持 | 核心条件 |
| --- | --- | --- |
| 无捕获 Lambda → 函数指针 | 支持隐式转换 | 参数、返回值和 `noexcept` 类型兼容 |
| 有捕获 Lambda → 函数指针 | 不支持 | 函数指针没有空间保存闭包状态 |
| Lambda → `std::function<R(Args...)>` | 支持 | C++20 中闭包必须可复制构造，并且能按目标签名调用 |
| 函数指针 → `std::function<R(Args...)>` | 支持 | 函数指针签名兼容 |
| `std::function` → 函数指针 | 不提供通用转换 | 包装器可能保存闭包、函数对象或其他状态 |

`std::function<R(Args...)>` 是一个 **类型擦除（type erasure）** 包装器。它把不同的具体可调用类型统一成固定签名，而不是把有捕获 Lambda 变成普通函数指针。

```cpp
#include <cassert>
#include <functional>

int main() {
    const int offset = 10;

    // 闭包类型的真实名字不同，但都可以被擦除成 int(int)。
    std::function<int(int)> add_offset = [offset](int value) {
        return value + offset;
    };

    std::function<int(int)> square = [](int value) {
        return value * value;
    };

    assert(add_offset(5) == 15);
    assert(square(5) == 25);
}
```

构造 `std::function` 时，标准库通过模板构造函数保存闭包对象，并记录一组用于调用、复制和销毁该对象的操作。调用方只看见统一的 `R(Args...)` 接口，不再知道具体闭包类型。

### 无捕获 Lambda 存入 `std::function` 时不一定先转成函数指针

无捕获 Lambda 同时满足两条路径：可以直接作为闭包对象存入 `std::function`，也可以先转换成函数指针再存入。两者调用结果相同，但 `std::function` 内部保存的目标类型不同。

```cpp
#include <cassert>
#include <functional>

using BinaryOperation = int (*)(int, int);

int main() {
    auto add = [](int left, int right) {
        return left + right;
    };

    std::function<int(int, int)> stores_closure = add;
    std::function<int(int, int)> stores_pointer = +add;

    // target<T>() 只有在 T 与内部目标的真实类型完全相同时才返回非空指针。
    assert(stores_closure.target<decltype(add)>() != nullptr);
    assert(stores_closure.target<BinaryOperation>() == nullptr);

    assert(stores_pointer.target<BinaryOperation>() != nullptr);
    assert(stores_pointer.target<decltype(add)>() == nullptr);
}
```

通常不需要主动写一元 `+`。只有接口确实要求函数指针、需要检查准确的 `target<T>()` 类型，或有其他明确的 ABI（应用二进制接口）要求时，才应强制执行这次转换。

### `std::function` 不能普遍反向转换为函数指针

```cpp
#include <functional>

using UnaryOperation = int (*)(int);

int main() {
    std::function<int(int)> operation = [](int value) {
        return value + 1;
    };

    // UnaryOperation pointer = operation;
    // 错误：std::function 没有到函数指针的通用转换。
}
```

原因是 `std::function` 可能保存有捕获闭包，其行为依赖对象状态。只有明确知道它内部保存的目标恰好是某种函数指针时，才能用 `target<T>()` 检查并取出：

```cpp
#include <cassert>
#include <functional>

int increment(int value) {
    return value + 1;
}

using UnaryOperation = int (*)(int);

int main() {
    std::function<int(int)> operation = &increment;

    if (const auto stored_pointer = operation.target<UnaryOperation>()) {
        // stored_pointer 的类型是 UnaryOperation*，还要再解引用一次取得函数指针。
        assert((*stored_pointer)(41) == 42);
    }
}
```

`target<T>()` 不是“尝试转换为 `T`”，而是检查内部目标的**准确类型**。如果 `std::function` 直接存入的是无捕获闭包，查询函数指针类型仍会得到 `nullptr`。

### 固定签名会选择泛型 Lambda 的一种调用方式

泛型 Lambda 有模板化的 `operator()`。存入 `std::function<int(int)>` 后，对外只暴露 `int(int)` 这一种调用签名：

```cpp
#include <cassert>
#include <functional>

int main() {
    const auto identity = [](auto value) {
        return value;
    };

    std::function<int(int)> int_identity = identity;
    std::function<double(double)> double_identity = identity;

    assert(int_identity(42) == 42);
    assert(double_identity(2.5) == 2.5);
}
```

这里不是泛型 Lambda 被永久改变了，而是两个 `std::function` 分别要求它能以对应参数和返回类型调用。若目标签名不兼容，构造会在编译期失败。

“兼容”不要求闭包的形参和返回类型逐字相同。调用过程中允许发生普通的隐式转换：

```cpp
#include <cassert>
#include <functional>

int main() {
    // std::function 对外接收 int，Lambda 实际接收 double。
    std::function<double(int)> half = [](double value) {
        return value / 2.0;
    };

    // Lambda 返回 double，std::function 的目标签名把结果转换为 int。
    std::function<int(double)> truncate = [](double value) {
        return value;
    };

    assert(half(5) == 2.5);
    assert(truncate(3.9) == 3);
}
```

这有时很方便，但也可能隐藏窄化、精度损失或昂贵的临时对象构造。回调接口应尽量让 `std::function` 签名与 Lambda 的实际意图一致。

### C++20 的 `std::function` 要求目标可复制

捕获 `std::unique_ptr` 后，闭包通常只能移动。即使使用 `std::move` 构造 `std::function`，C++20 仍要求目标类型本身可以复制构造：

```cpp
#include <functional>
#include <memory>
#include <utility>

int main() {
    auto move_only_task = [resource = std::make_unique<int>(42)] {
        return *resource;
    };

    // std::function<int()> task = std::move(move_only_task);
    // C++20 中错误：闭包不可复制构造。
}
```

`std::move` 只能决定本次构造尽量移动目标，不能消除 `std::function` 自身需要支持复制的接口要求。C++20 中可以改用模板接口、专用的 move-only 任务包装器，或者重新设计捕获状态的所有权。标准库的 `std::move_only_function` 到 C++23 才加入。

### 复制 `std::function` 会复制其中的闭包状态

```cpp
#include <cassert>
#include <functional>

int main() {
    auto counter = [count = 0]() mutable {
        return ++count;
    };

    std::function<int()> first = counter;
    std::function<int()> second = first;

    // first 和 second 各自拥有一份 count，从复制时的相同状态独立变化。
    assert(first() == 1);
    assert(first() == 2);
    assert(second() == 1);
}
```

如果闭包按引用捕获外部对象，复制包装器只会复制引用语义，多个包装器仍访问同一个对象：

```cpp
#include <cassert>
#include <functional>

int main() {
    int count = 0;

    std::function<void()> first = [&count] {
        ++count;
    };
    std::function<void()> second = first;

    first();
    second();
    assert(count == 2);
}
```

因此复制 `std::function` 是复制闭包，还是继续共享状态，最终仍由捕获成员自身的复制语义决定。

如果明确不想复制闭包，可以让 `std::function` 保存 `std::reference_wrapper`：

```cpp
#include <cassert>
#include <functional>

int main() {
    auto counter = [count = 0]() mutable {
        return ++count;
    };

    std::function<int()> counter_view = std::ref(counter);

    assert(counter_view() == 1);
    assert(counter() == 2); // 两次调用访问的是同一个闭包对象。
}
```

这会把所有权责任反转给调用者：`counter_view` 不拥有 `counter`，因此 `counter` 必须比包装器的每次调用活得更久。除非确实需要共享同一个闭包状态，否则直接让 `std::function` 持有自己的闭包副本通常更安全。

### `const std::function` 不代表目标没有内部状态变化

`std::function::operator()` 在 C++20 中可以通过 `const std::function` 调用，但它仍能调用一个 `mutable` Lambda：

```cpp
#include <cassert>
#include <functional>

int main() {
    const std::function<int()> counter = [count = 0]() mutable {
        return ++count;
    };

    assert(counter() == 1);
    assert(counter() == 2);
}
```

这意味着包装器的 `const` 不能证明调用是纯函数，也不能证明多线程并发调用安全。若多个线程共享同一个带可变状态的 `std::function`，仍需检查目标闭包的同步策略。

### 空 `std::function` 必须先检查

默认构造、赋值 `nullptr` 或移动后的 `std::function` 可能为空。它支持显式布尔检查；直接调用空包装器会抛出 `std::bad_function_call`。

```cpp
#include <cassert>
#include <functional>

int main() {
    std::function<int(int)> operation;
    assert(!operation);

    if (operation) {
        return operation(42);
    }

    try {
        operation(42);
    } catch (const std::bad_function_call&) {
        // 预期路径：空 std::function 不能调用。
    }
}
```

若接口允许“没有回调”，调用前检查 `if (callback)`；若回调不能为空，最好在构造或注册阶段建立不变量，而不是每次调用时才发现。

### `std::function` 不会修复引用和 `this` 的生命周期

```cpp
#include <functional>

std::function<int()> make_dangling_callback() {
    int local_value = 42;

    return [&local_value] {
        return local_value;
    };
} // local_value 已销毁，std::function 中的闭包保存着悬空引用。
```

类型擦除只隐藏类型，不改变捕获语义：

- `[&local]` 仍要求 `local` 在每次调用时存活。
- `[this]` 仍不延长当前对象生命周期。
- 按值捕获 `std::string_view`、`std::span` 或裸指针仍不拥有底层数据。
- 捕获 `std::shared_ptr` 会延长生命周期，也可能形成引用环；需要时使用 `std::weak_ptr`。

### 返回引用时必须显式保证被引用对象存活

```cpp
#include <cassert>
#include <functional>

int main() {
    int value = 42;

    std::function<int&()> get_value = [&value]() -> int& {
        return value;
    };

    get_value() = 100;
    assert(value == 100);
}
```

在 C++20 中尤其不要把返回临时值的 Lambda 塞进引用返回签名，例如 `std::function<const int&()>`。即使某些写法能通过编译，返回的引用也可能在调用结束时立即悬空。引用返回接口应让 Lambda 显式写出引用返回类型，并返回一个生命周期足够长的对象。

### `noexcept` 信息会被 C++20 的 `std::function` 擦除

```cpp
#include <functional>

int main() {
    const auto safe_task = []() noexcept {};
    std::function<void()> erased_task = safe_task;

    static_assert(noexcept(safe_task()));
    static_assert(!noexcept(erased_task()));
}
```

C++20 的 `std::function<void()>` 签名没有保留目标 Lambda 的 `noexcept`。如果调用方必须在类型层面知道“绝不抛异常”，函数指针的 `noexcept` 类型、具体闭包类型或专门设计的包装器会更合适。

### C 风格回调需要函数指针和上下文指针

许多 C API 使用“函数指针 + `void*` 上下文”传递回调。`std::function` 不能直接转换为这种函数指针，但无捕获 Lambda 可以作为桥接函数，并通过上下文指针访问状态：

```cpp
#include <cassert>

using Callback = void (*)(void*, int);

void invoke_callback(Callback callback, void* context) {
    callback(context, 42);
}

struct State {
    int total = 0;
};

int main() {
    State state;

    invoke_callback(
        +[](void* context, int value) {
            auto* state = static_cast<State*>(context);
            state->total += value;
        },
        &state
    );

    assert(state.total == 42);
}
```

这里仍由调用者保证 `context` 指向的对象在回调期间存活。若 C API 不提供上下文指针，就无法安全、通用地把任意有状态 `std::function` 塞进一个普通函数指针。

### 何时使用 `std::function`

- **适合使用**：需要把多种不同 callable 放进同一字段或容器、回调要在运行时替换、公共接口需要稳定的 `R(Args...)` 签名。
- **优先保留具体类型**：局部变量直接用 `auto`，泛型算法或热路径使用模板参数，通常更容易内联，也不会产生类型擦除开销。
- **考虑成本**：`std::function` 通常需要间接调用；较大的目标还可能发生动态内存分配。实现可以使用小对象优化，但标准不保证某个具体闭包一定不分配。
- **检查语义**：确认目标可复制、包装器非空、捕获对象仍存活、复制包装器后的状态共享方式符合预期。

## C++20 的类型级能力

### Lambda 可以出现在不求值语境

C++20 之前不能直接在 `decltype` 中写新的 Lambda 表达式。C++20 可以这样定义类型：

```cpp
#include <set>

using Descending = decltype([](int left, int right) {
    return left > right;
});

int main() {
    // C++20 的无捕获闭包可默认构造，因此 set 可以创建比较器。
    std::set<int, Descending> values;
    values.insert(1);
    values.insert(3);
    values.insert(2);
    return *values.begin() == 3 ? 0 : 1;
}
```

即使文本相同，两个独立的 `decltype([] {})` 仍是两个不同类型：

```cpp
#include <type_traits>

static_assert(!std::is_same_v<decltype([] {}), decltype([] {})>);
```

### 无捕获闭包可作为常量模板参数

C++20 的无捕获闭包满足相应的结构类型要求，可以把闭包对象作为常量模板参数：

```cpp
#include <array>
#include <cstddef>

template <auto Predicate, typename T, std::size_t N>
constexpr bool all_match(const std::array<T, N>& values) {
    for (const T& value : values) {
        if (!Predicate(value)) {
            return false;
        }
    }
    return true;
}

constexpr std::array values{2, 4, 6};

static_assert(all_match<[](int value) { return value % 2 == 0; }>(values));
```

带捕获列表内容的闭包不具备这项资格。尤其不要把“没有实际捕获成员的 `[=]`”误认为 `[]`。

## 选择 `auto`、模板接口还是 `std::function`

保存或传递 Lambda 时，常见选择有三种：

| 方式 | 是否保留具体类型 | 典型成本 | 适用场景 |
| --- | --- | --- | --- |
| `auto` | 是 | 通常最小 | 局部变量、返回类型可推导 |
| 模板形参 | 是 | 通常可内联，无类型擦除 | 泛型算法、立即调用或存入模板对象 |
| `std::function<R(Args...)>` | 否，执行类型擦除 | 可能间接调用和动态分配；目标必须可复制 | 需要稳定签名、运行时替换不同 callable |

如果调用方只是立刻调用，优先使用模板接口：

```cpp
#include <utility>

template <typename Callable>
void run_twice(Callable&& callable) {
    callable();
    callable();
}

int main() {
    int count = 0;
    run_twice([&count] {
        ++count;
    });
    return count == 2 ? 0 : 1;
}
```

确实需要统一运行时类型时再使用 `std::function`。下面这个工厂函数把具体闭包类型擦除为稳定的日志回调签名：

```cpp
#include <functional>
#include <iostream>
#include <string>
#include <utility>

std::function<void(const std::string&)> make_logger(std::string prefix) {
    return [prefix = std::move(prefix)](const std::string& message) {
        std::cout << prefix << message << '\n';
    };
}
```

`std::function` 复制的是其中的目标闭包，不会自动把引用捕获改成值捕获，也不会修复悬空引用。

## 常见实用模式

### STL 算法中的局部策略

```cpp
#include <algorithm>
#include <string>
#include <vector>

int main() {
    std::vector<std::string> names{"Lambda", "C++", "closure"};

    std::sort(names.begin(), names.end(), [](const std::string& left,
                                             const std::string& right) {
        // 长度相同时再按字典序排序，保证严格弱序关系清楚。
        if (left.size() != right.size()) {
            return left.size() < right.size();
        }
        return left < right;
    });
}
```

比较器必须满足严格弱序。不要写 `left <= right`，因为元素与自身比较时必须返回 `false`。

### 有状态计数器

```cpp
#include <cassert>

int main() {
    auto next_id = [id = 0]() mutable {
        return ++id;
    };

    assert(next_id() == 1);
    assert(next_id() == 2);

    auto copied_counter = next_id;
    assert(copied_counter() == 3); // 从复制时的状态独立继续。
    assert(next_id() == 3);        // 原闭包也从自己的状态继续。
}
```

复制有状态闭包会复制当前状态，而不是让两个闭包自动共享状态。若需要共享计数器，应明确捕获共享对象，并考虑线程安全。

### `std::variant` 的重载访问器

```cpp
#include <iostream>
#include <string>
#include <variant>

template <typename... Callables>
struct Overloaded : Callables... {
    using Callables::operator()...;
};

template <typename... Callables>
Overloaded(Callables...) -> Overloaded<Callables...>;

int main() {
    const std::variant<int, std::string> value = std::string{"lambda"};

    std::visit(
        Overloaded{
            [](int number) {
                std::cout << "整数：" << number << '\n';
            },
            [](const std::string& text) {
                std::cout << "字符串：" << text << '\n';
            },
        },
        value
    );
}
```

这里通过继承多个闭包类型并引入各自的 `operator()`，得到一个重载集合。`Overloaded` 自身是手写类模板，不是单个 Lambda 获得了多个函数体。

### C++20 内的递归 Lambda

Lambda 初始化表达式中还不能通过变量名引用正在初始化的 Lambda。可以把闭包自身作为显式参数传入：

```cpp
#include <cassert>

int main() {
    const auto factorial = [](auto&& self, int value) -> int {
        if (value <= 1) {
            return 1;
        }
        return value * self(self, value - 1);
    };

    assert(factorial(factorial, 5) == 120);
}
```

这种模式没有 `std::function` 的类型擦除，也支持泛型递归。C++20 不支持 C++23 的显式对象形参，因此还不能写更新标准中的 `this Self&& self` 形式。

## 生命周期、并发与性能陷阱

### 不要返回引用局部变量的闭包

```cpp
auto make_dangling_reader() {
    int local_value = 42;

    return [&local_value] {
        return local_value;
    };
} // local_value 已销毁，返回闭包中的引用悬空。
```

创建闭包本身是合法的，错误通常直到闭包逃出作用域并被调用才暴露。更安全的版本按值捕获：

```cpp
auto make_reader() {
    int local_value = 42;
    return [local_value] {
        return local_value;
    };
}
```

### 异步任务不要随手使用 `[&]`

```cpp
#include <future>

std::future<int> start_task() {
    int local_value = 42;

    // local_value 被复制进任务，任务不依赖函数栈帧继续存在。
    return std::async(std::launch::async, [local_value] {
        return local_value;
    });
}
```

如果写成 `[&local_value]`，异步线程可能在 `start_task` 返回、局部变量销毁之后才访问它。

### 捕获不会消除数据竞争

```cpp
#include <atomic>
#include <thread>
#include <vector>

int main() {
    std::atomic<int> count{0};

    const auto increment = [&count] {
        count.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(increment);
    }
    for (auto& worker : workers) {
        worker.join();
    }

    return count.load(std::memory_order_relaxed) == 4 ? 0 : 1;
}
```

如果这里把 `std::atomic<int>` 换成普通 `int`，多个线程并发执行 `++count` 会形成数据竞争。Lambda 只是访问路径，不是同步原语。

### 控制闭包大小和复制成本

- 大对象按值捕获会增大闭包，并让算法、线程库或类型擦除包装器复制更多数据。
- 按引用捕获可减少复制，但把生命周期责任交给调用者；不能只为省几个字节就忽略悬空风险。
- 捕获 `std::shared_ptr` 复制成本不等于复制整个对象，但会进行引用计数操作，并改变对象生命周期。
- 捕获范围过大的 `[=]` 或 `[&]` 会隐藏依赖。显式捕获更利于审查所有权、线程安全和对象大小。
- 捕获成员的声明与初始化顺序不应被程序依赖；标准没有把闭包布局当作稳定接口。
- 是否内联与是否写成 Lambda 没有绝对关系。保留具体闭包类型通常更利于优化，而 `std::function` 等类型擦除可能引入间接调用。

## 常见错误与判断方法

### 误区：`[=]` 会复制当前对象

错误。C++20 中 `[=]` 对成员访问仍是隐式复制 `this` 指针，而且这种写法已弃用。需要对象副本时写 `[*this]`。

### 误区：按值捕获一定安全

错误。按值捕获 `int`、`std::string` 往往拥有独立状态；按值捕获裸指针、`std::string_view`、`std::span`、`std::reference_wrapper` 仍可能悬空。

### 误区：`mutable` 会修改外部变量

错误。它只让闭包的非 `const operator()` 可以修改按值捕获成员。引用捕获是否能修改外部对象取决于被引用对象本身的类型，与 `mutable` 无直接关系。

### 误区：能移动闭包，就一定能放进 `std::function`

错误。C++20 的 `std::function` 要求目标可复制构造。捕获 `std::unique_ptr` 的 move-only 闭包不满足要求。

### 误区：两个完全相同的 Lambda 可以互相赋值

错误。两个表达式产生不同闭包类型，连赋值双方的类型都不同：

```cpp
auto first = [] { return 1; };
auto second = [] { return 1; };

// first = second; // 错误：两者是不同类型。
```

同一无捕获闭包类型的两个对象在 C++20 才可以赋值：

```cpp
auto first = [] { return 1; };
auto second = first;
second = first; // C++20 合法。
```

### 误区：复制引用捕获闭包会复制被引用对象

错误。闭包副本继续指向同一个外部对象：

```cpp
#include <cassert>

int main() {
    int value = 0;
    const auto increment = [&value] { ++value; };
    const auto increment_copy = increment;

    increment();
    increment_copy();
    assert(value == 2);
}
```

### 误区：`[=]` 和 `[&]` 一定捕获作用域内全部变量

错误。默认捕获提供的是隐式捕获规则，而不是无条件把每个局部变量塞进闭包。仍然不应依赖具体大小或优化结果来反推标准语义。

## C++20 使用清单

编写或审查 Lambda 时，可以依次检查：

- **是否会逃出当前作用域？** 会逃出时重点检查所有引用捕获、`this`、指针和视图的生命周期。
- **所有权是否清楚？** 需要独立状态时按值或移动捕获；需要观察共享状态时再使用引用、`shared_ptr` 或 `weak_ptr`。
- **是否真的需要默认捕获？** 捕获项不多时优先显式列出，减少隐式依赖。
- **是否误把 `[=]` 当成 `[*this]`？** C++20 中应明确表达 `this` 的生命周期策略。
- **闭包需要被复制吗？** STL、线程库、任务队列和 `std::function` 对 callable 的构造要求不同；用类型特征或接口约束验证。
- **捕获对象是否只移动？** 捕获 `unique_ptr`、锁或其他独占资源后，闭包往往不再可复制。
- **是否需要 `mutable`？** 只读闭包保持默认 `const operator()`；真正需要内部可变状态时再添加。
- **返回值是否需要保留引用？** 普通 `auto` 返回会丢弃引用时，使用显式引用返回类型或 `decltype(auto)`。
- **比较器是否满足严格弱序？** 排序比较器通常使用 `<`，不要使用 `<=`。
- **并发访问是否同步？** 捕获方式不替代互斥量和原子操作。
- **是否跨越了 C++20 边界？** `static` Lambda、显式对象形参和 `std::move_only_function` 都属于 C++23，不应出现在纯 C++20 代码中。

## 参考资料

- [C++20 工作草案 N4861：Lambda expressions](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4861.pdf)
- [N3648：C++14 初始化捕获措辞](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3648.html)
- [N3649：C++14 泛型 Lambda](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3649.html)
- [N3582：返回类型推导与 Lambda 返回推导放宽](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3582.html)
- [P0018R3：C++17 按值捕获 `*this`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html)
- [N4487：`constexpr` Lambda](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4487.pdf)
- [P2131R0：C++17 到 C++20 的变化汇总](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2131r0.html)
- [P0428R2：C++20 泛型 Lambda 的模板头](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0428r2.pdf)
- [P0624R2：C++20 无捕获闭包可默认构造和赋值](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0624r2.pdf)
- [P0315R4：C++20 不求值语境中的 Lambda](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0315r4.pdf)
- [P0780R2：C++20 初始化捕获中的参数包展开](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0780r2.html)
- [P0409R2：C++20 允许 `[=, this]`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0409r2.html)
- [P0806R2：C++20 弃用 `[=]` 对 `this` 的隐式捕获](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0806r2.html)
- [P1091R3：C++20 结构化绑定与 Lambda 捕获改进](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1091r3.html)
- [P1073R3：C++20 立即函数与 `consteval`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1073r3.html)
