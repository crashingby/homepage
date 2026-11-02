---
title: CuTe Layout 笔记
date: 2026-07-13
tags: [CUDA, CuTe, CUTLASS, Layout, GPU 编程]
summary: 整理 CuTe Layout 的核心模型、整数与 tuple 抽象、make_layout / make_shape / make_stride、层次化访问、坐标映射、布局拼接、分组与展平，并对照 CUTLASS CuTe 源码解释实现。
---

# CuTe Layout 笔记

本文整理 CuTe 的 **Layout**。它是 CuTe 里非常核心的抽象：一个 Layout 建立了从 **坐标空间（coordinate space）** 到 **索引空间（index space）** 的映射。

用一句话概括：

$$
\text{Layout}(\text{coord}) \rightarrow \text{index}
$$

你给它逻辑坐标，比如 $(i, j)$，它返回一个线性索引，比如 `i + j * M` 或 `i * N + j`。这样算法逻辑就可以和内存布局分离：

- 同一个 kernel 可以通过更换 Layout，从列主序切到行主序。
- 同一个 Tensor 可以通过 Layout 表示交织布局、层次化布局、分块布局。
- 线程如何访问数据、tile 如何映射到内存，都可以转化成 Layout 的组合和操作。

CuTe 的强大之处不只是“可以描述布局”，而是有一套 **布局代数（algebra of layouts）**。Layout 可以拼接、分组、展平、切片、组合。复杂的“哪个线程读哪块数据”最后会变成一串布局变换。

这篇文章主要对照这些源码文件：

| 源码文件 | 主要内容 |
| --- | --- |
| `cutlass/include/cute/numeric/integral_constant.hpp` | `C<v>`、`Int<N>`、`_1` 这类静态整数。 |
| `cutlass/include/cute/container/tuple.hpp` | `cute::tuple`、`make_tuple`、`get<I>`。 |
| `cutlass/include/cute/int_tuple.hpp` | `IntTuple` 上的 `rank`、`depth`、`shape` 等递归工具。 |
| `cutlass/include/cute/stride.hpp` | `LayoutLeft`、`LayoutRight`、`compact_major`。 |
| `cutlass/include/cute/layout.hpp` | `Layout`、`make_shape`、`make_stride`、`make_layout`、布局访问和布局操作。 |
| `cutlass/include/cute/algorithm/tuple_algorithms.hpp` | `append`、`prepend`、`replace`、`group`、`flatten` 等 tuple 结构操作。 |

## 核心模型

一个 Layout 由两个 `IntTuple` 组成：

$$
L = (\text{Shape}, \text{Stride})
$$

其中：

- **Shape** 描述逻辑坐标空间。
- **Stride** 描述每个自然坐标分量对应的线性索引增量。

如果自然坐标是：

$$
c = (c_0, c_1, \dots, c_{r-1})
$$

stride 是：

$$
s = (s_0, s_1, \dots, s_{r-1})
$$

那么最简单的线性索引公式是：

$$
L(c) = \sum_{k=0}^{r-1} c_k \cdot s_k
$$

对于层次化 layout，公式仍然成立，只是坐标和 stride 本身也可能是嵌套 tuple。例如：

$$
\text{Shape} = (3, (2, 3)), \quad \text{Stride} = (3, (12, 1))
$$

自然坐标：

$$
c = (i, (j, k))
$$

对应索引：

$$
L(i,(j,k)) = i \cdot 3 + j \cdot 12 + k \cdot 1
$$

CuTe 的关键点是：Layout 不一定只接受自然坐标。它还可以接受 1D 坐标、扁平 2D 坐标等兼容坐标，再先转成自然坐标，最后和 stride 做乘加。

整体流程是：

```mermaid
flowchart LR
    A["输入坐标<br>1D / 2D / h-D"] --> B["idx2crd / 坐标正规化"]
    B --> C["自然坐标<br>congruent with Shape"]
    C --> D["crd2idx"]
    D --> E["线性索引"]
```

## 基本类型

### 整数：动态整数和静态整数

CuTe 大量同时使用 **动态整数** 和 **静态整数**。

| 类型 | 例子 | 值何时知道 | 用途 |
| --- | --- | --- | --- |
| 动态整数 | `int`、`size_t`、`uint16_t` | 运行时 | 灵活，适合运行时 shape。 |
| 静态整数 | `Int<8>{}`、`_8{}`、`C<8>{}` | 编译期 | 编译器可常量传播、循环展开、消除分支。 |

源码里静态整数的核心是 `C<v>`：

```cpp
template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  CUTE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};
```

`template <auto v>` 是 C++17 的非类型模板参数推导。它让 `v` 的类型由编译器推导：

```cpp
C<10>      // v 是 int
C<10ull>   // v 是 unsigned long long
```

重点是：`v` 是类型的一部分。`C<1>` 和 `C<2>` 是两个不同类型，不只是两个不同值。

CuTe 再定义：

```cpp
template <int v>
using Int = C<v>;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _3 = Int<3>;
using _4 = Int<4>;
```

所以常见写法：

```cpp
Int<8>{}
_8{}
```

本质上都是把数值编码进类型系统。

CuTe 还提供了一组 traits：

| trait | 含义 |
| --- | --- |
| `cute::is_std_integral<T>` | 是否为普通 C++ 整数类型。 |
| `cute::is_integral<T>` | 是否为普通整数或 CuTe 静态整数。 |
| `cute::is_static<T>` | 是否完全由类型决定，通常用来判断是否为空类型。 |
| `cute::is_constant<N, T>` | `T` 是否为值等于 `N` 的静态整数。 |

静态整数和动态整数在代码生成上的差别可以这样理解：

| 特性 | 动态整数 `int n = 8` | 静态整数 `Int<8>{}` |
| --- | --- | --- |
| 存储位置 | 寄存器或内存 | 类型系统 / 编译期常量 |
| 计算时间 | 运行时 | 编译期 |
| 优化空间 | 需要保守生成代码 | 可做常量传播、展开、死代码消除 |
| 在 CuTe 中的作用 | 支持运行时 shape | 让 layout 变换尽量在编译期化简 |

### `cute::tuple`

`cute::tuple` 类似 `std::tuple`，但它要能在 host 和 device 上使用，并且要服务于大量编译期 layout 计算。

源码里它大致是：

```cpp
template <class... T>
struct tuple : eso::ESO_t<T...>
{
  CUTE_HOST_DEVICE constexpr
  tuple() {}

  CUTE_HOST_DEVICE constexpr
  tuple(T const&... t) : eso::ESO_t<T...>(t...) {}
};

template <class... T>
CUTE_HOST_DEVICE constexpr
tuple<T...>
make_tuple(T const&... t)
{
  return {t...};
}
```

这里 `eso::ESO_t<T...>` 是 Empty Storage Optimization，用来让静态整数这类空对象不占运行时存储。

访问 tuple 用：

```cpp
template <size_t I, class... T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(tuple<T...> const& t) noexcept
{
  static_assert(I < sizeof...(T), "Index out of range");
  return eso::getv_cr<I>(t);
}
```

所以 `cute::tuple` 的设计目标不是做一个完整 `std::tuple` 替代品，而是：**在 CUDA host/device 代码中高效承载 layout 的结构信息**。

### `IntTuple`

CuTe 把很多东西抽象成 `IntTuple`。

递归定义：

- 一个整数是 `IntTuple`。
- 一个由 `IntTuple` 组成的 `cute::tuple` 也是 `IntTuple`。

示例：

```cpp
int{2}
Int<3>{}
make_tuple(int{2}, Int<3>{})
make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})
```

CuTe 把 `IntTuple` 复用于：

- Shape
- Stride
- Step
- Coord

也就是说，`(2, 4)` 可以是 shape，也可以是 stride，也可以是 coord。语义由使用场景决定。

### `rank`、`depth`、`size`

`rank` 表示顶层 mode 数。

源码里对普通整数做了一个特殊处理：

```cpp
template <size_t I, class T,
          __CUTE_REQUIRES(cute::is_integral<cute::remove_cvref_t<T>>::value)>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(T&& t) noexcept
{
  static_assert(I == 0, "Index out of range");
  return static_cast<T&&>(t);
}
```

这使得普通整数也可以被 `get<0>` 访问。原因是 CuTe 定义：

$$
\operatorname{rank}(\text{integer}) = 1
$$

`rank` 的源码逻辑是：

```cpp
template <int... Is, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
rank(IntTuple const& t)
{
  if constexpr (sizeof...(Is) == 0) {
    if constexpr (is_tuple<IntTuple>::value) {
      return Int<tuple_size<IntTuple>::value>{};
    } else {
      return Int<1>{};
    }
  } else {
    return rank(get<Is...>(t));
  }
}
```

因此：

| 表达式 | `rank` |
| --- | --- |
| `6` | 1 |
| `(2)` | 1 |
| `(4,3)` | 2 |
| `(3,(6,2),8)` | 3 |

`depth` 表示嵌套深度。源码逻辑是：

```cpp
if constexpr (is_tuple<IntTuple>::value) {
  return Int<1>{} + cute::apply(t, [](auto const&... v){
    return cute::max(depth(v)...);
  });
} else {
  return Int<0>{};
}
```

所以：

| 表达式 | `depth` |
| --- | --- |
| `6` | 0 |
| `(2)` | 1 |
| `(4,3)` | 1 |
| `(3,(6,2),8)` | 2 |

`size(IntTuple)` 是所有叶子整数的乘积：

$$
\operatorname{size}((a,b,c)) = a \cdot b \cdot c
$$

层次化时递归展开：

$$
\operatorname{size}((3,(6,2),8)) = 3 \cdot 6 \cdot 2 \cdot 8
$$

## Shape、Stride 和 Layout

### `make_shape` 和 `make_stride`

CuTe 里 `Shape`、`Stride`、`Coord` 等其实都是 `cute::tuple` 的别名语义。构造函数放在 `layout.hpp`：

```cpp
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Shape<Ts...>
make_shape(Ts const&... t) {
  return {t...};
}

template <class... Ts>
CUTE_HOST_DEVICE constexpr
Stride<Ts...>
make_stride(Ts const&... t) {
  return {t...};
}

template <class... Ts>
CUTE_HOST_DEVICE constexpr
Coord<Ts...>
make_coord(Ts const&... t) {
  return {t...};
}
```

它们的作用不是做复杂计算，而是让 C++ 自动推导模板参数，避免你手写很长的类型：

```cpp
auto s = make_shape(Int<2>{}, 4);   // (_2,4)
auto d = make_stride(Int<1>{}, 2);  // (_1,2)
```

### `Layout`

`Layout` 的源码结构是：

```cpp
template <class Shape, class Stride = LayoutLeft::Apply<Shape> >
struct Layout
    : private cute::tuple<Shape, Stride>   // EBO for static layouts
{
  CUTE_HOST_DEVICE constexpr
  Layout(Shape const& shape = {}, Stride const& stride = {})
      : cute::tuple<Shape, Stride>(shape, stride)
  {}

  static constexpr int rank = rank_v<Shape>;

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) shape() const {
    return get<0,I...>(static_cast<cute::tuple<Shape, Stride> const&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) stride() const {
    return get<1,I...>(static_cast<cute::tuple<Shape, Stride> const&>(*this));
  }
};
```

几个关键点：

- `Layout` 本质上保存两个东西：`Shape` 和 `Stride`。
- 它私有继承 `cute::tuple<Shape, Stride>`，可以利用 EBO 减少静态布局的运行时开销。
- 默认 `Stride` 是 `LayoutLeft::Apply<Shape>`，也就是默认按 `LayoutLeft` 生成 compact stride。
- `shape<I...>()` 和 `stride<I...>()` 是层次化访问：先取第 0 / 1 个成员，再按 `I...` 进入内部结构。

可以把 Layout 的数学语义写成：

$$
L = (S, D)
$$

其中 $S$ 是 shape，$D$ 是 stride。若自然坐标 $c$ 和 $S$ congruent（全等），则：

$$
L(c) = \langle c, D \rangle
$$

这里的 $\langle\cdot,\cdot\rangle$ 是递归点积。

### `make_layout`

`make_layout` 的基础重载：

```cpp
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape, Stride const& stride)
{
  static_assert(is_tuple<Shape >::value || is_integral<Shape >::value);
  static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
  return Layout<Shape,Stride>(shape, stride);
}
```

如果只传 shape：

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape)
{
  return make_layout(shape, compact_major<LayoutLeft>(shape));
}
```

也就是默认生成 `LayoutLeft` stride。

显式指定策略：

```cpp
make_layout(shape, LayoutLeft{})
make_layout(shape, LayoutRight{})
```

多个 Layout 也可以直接组合：

```cpp
make_layout(layout0, layout1)
```

源码里会把 shape 和 stride 分别拼起来：

```cpp
return make_layout(make_shape (layout0.shape() , layout1.shape() ),
                   make_stride(layout0.stride(), layout1.stride()));
```

这就是为什么 `make_layout(a, b)` 会生成一个更高 rank 的布局，而不是做函数复合。

看一个具体例子：

```cpp
Layout a  = Layout<_3,_1>{};      // 3:1
Layout b  = Layout<_4,_3>{};      // 4:3
Layout ab = make_layout(a, b);    // (3,4):(1,3)
```

`a` 和 `b` 本来都是 Rank-1 layout：

$$
a(i) = i \cdot 1
$$

$$
b(j) = j \cdot 3
$$

`make_layout(a, b)` 之后，新的 layout 是：

$$
ab = (3,4):(1,3)
$$

它有两个 mode，坐标也变成二维坐标：

$$
ab(i,j) = i \cdot 1 + j \cdot 3
$$

例如：

```cpp
ab(0, 0)  // 0
ab(1, 0)  // 1
ab(2, 0)  // 2
ab(0, 1)  // 3
ab(1, 1)  // 4
ab(2, 1)  // 5
```

打印成二维表就是：

```text
0  3  6  9
1  4  7  10
2  5  8  11
```

所以 `make_layout(a, b)` 的含义更接近：

$$
ab(i,j) = a(i) + b(j)
$$

也就是把 `a` 和 `b` 作为两个独立 mode 拼成一个更高 rank 的 layout。

如果是函数复合，语义会是：

$$
(a \circ b)(j) = a(b(j))
$$

这完全是另一件事：输入仍然是 `b` 的坐标，输出先经过 `b`，再喂给 `a`。而且在这个例子里 `b(1)=3`，已经超出 `a` 的合法坐标范围 `[0,3)`，所以很多点上连复合都不成立。CuTe 里要表达函数复合，应看 `composition(lhs, rhs)`，不是 `make_layout(lhs, rhs)`。

### `LayoutLeft` 和 `LayoutRight`

`LayoutLeft` / `LayoutRight` 是标签类。源码里定义在 `stride.hpp`：

```cpp
struct LayoutLeft;               // Col-major layout mapping; leftmost extent has stride 1
using GenColMajor = LayoutLeft;

struct LayoutRight;              // Row-major layout mapping; rightmost extent has stride 1
using GenRowMajor = LayoutRight;
```

实际 stride 生成由 `compact_major` 完成。

#### `LayoutLeft`

`LayoutLeft` 从左到右生成排他性前缀积：

$$
d_0 = 1
$$

$$
d_i = \prod_{k=0}^{i-1} s_k
$$

例如 shape `(2,4)`：

```cpp
make_layout(make_shape(Int<2>{}, 4), LayoutLeft{})
```

生成 stride：

$$
(1, 2)
$$

这就是广义列主序：最左边的 mode stride 为 1。

#### `LayoutRight`

`LayoutRight` 从右到左生成排他性前缀积：

$$
d_{r-1} = 1
$$

$$
d_i = \prod_{k=i+1}^{r-1} s_k
$$

例如 shape `(2,4)`：

```cpp
make_layout(make_shape(Int<2>{}, 4), LayoutRight{})
```

生成 stride：

$$
(4, 1)
$$

这就是广义行主序：最右边的 mode stride 为 1。

源码里的核心差别是 `LayoutLeft` 用 `append`，`LayoutRight` 用 `prepend`：

```cpp
// LayoutLeft
return cute::make_tuple(append(get<0>(init), get<0>(result)), get<1>(result));

// LayoutRight
return cute::make_tuple(prepend(get<0>(init), get<0>(result)), get<1>(result));
```

它们不是运行时分支，而是通过标签类型选择不同模板特化。

### 创建示例


```cpp
Layout s8        = make_layout(Int<8>{});
Layout d8        = make_layout(8);
Layout s2xs4     = make_layout(make_shape(Int<2>{}, Int<4>{}));
Layout s2xd4     = make_layout(make_shape(Int<2>{}, 4));
Layout s2xd4_a   = make_layout(make_shape(Int<2>{}, 4),
                               make_stride(Int<12>{}, Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4),
                               LayoutRight{});
Layout s2xh4     = make_layout(make_shape(2, make_shape(2, 2)),
                               make_stride(4, make_stride(2, 1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

打印结果：

```text
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

`Shape:Stride` 是 CuTe 打印 Layout 的常用格式。`_N` 表示静态整数，普通数字表示动态整数。

这里要注意一个不变量：

$$
\operatorname{congruent}(\text{shape}, \text{stride}) = \text{true}
$$

也就是说 shape 和 stride 必须有相同的 tuple 轮廓。shape 的每个叶子整数，在 stride 中都要有一个对应叶子整数。

```cpp
static_assert(congruent(my_shape, my_stride));
```

## 层次化访问 API

CuTe 对 `IntTuple` 和 `Layout` 都提供了层次化访问函数。

### `get<I...>`

源码中递归 `get` 是：

```cpp
template <size_t I0, size_t I1, size_t... Is, class T>
CUTE_HOST_DEVICE constexpr
decltype(auto)
get(T&& t) noexcept
{
  return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}
```

所以：

```cpp
get<1,0>(x)
```

等价于：

```cpp
get<0>(get<1>(x))
```

公式写成：

$$
\operatorname{get}^{I_0,I_1,\dots,I_n}(x)
=
\operatorname{get}^{I_n}(\dots \operatorname{get}^{I_1}(\operatorname{get}^{I_0}(x)) \dots)
$$

### `rank<I...>`、`depth<I...>`、`shape<I...>`、`size<I...>`

这些函数都遵循同一个模式：

$$
\operatorname{rank}^{I...}(x) = \operatorname{rank}(\operatorname{get}^{I...}(x))
$$

$$
\operatorname{depth}^{I...}(x) = \operatorname{depth}(\operatorname{get}^{I...}(x))
$$

$$
\operatorname{shape}^{I...}(x) = \operatorname{shape}(\operatorname{get}^{I...}(x))
$$

$$
\operatorname{size}^{I...}(x) = \operatorname{size}(\operatorname{get}^{I...}(x))
$$

对于 Layout，源码里 `rank` / `depth` / `size` 直接转发到 shape：

```cpp
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
size(Layout<Shape,Stride> const& layout)
{
  return size(shape<Is...>(layout));
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
rank(Layout<Shape,Stride> const& layout)
{
  return rank(shape<Is...>(layout));
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
depth(Layout<Shape,Stride> const& layout)
{
  return depth(shape<Is...>(layout));
}
```

所以 `size<0>(layout)` 的意思是：取第 0 个 mode 的 shape，再计算这个 mode 的元素数。后面的 `print2D` 示例会用它作为循环边界。

## 坐标到索引

### `Layout::operator()`

Layout 的调用运算符有两条路径：

```cpp
template <class Coord>
CUTE_HOST_DEVICE constexpr
auto
operator()(Coord const& coord) const {
  if constexpr (has_underscore<Coord>::value) {
    return slice(coord, *this);
  } else {
    return crd2idx(coord, shape(), stride());
  }
}
```

如果坐标里有 `_`，表示切片：

```cpp
layout(_, 3)
```

返回一个子 Layout。

如果坐标里没有 `_`，表示普通索引：

```cpp
layout(2, 3)
```

返回线性 index。

多参数版本只是语法糖：

```cpp
template <class Coord0, class Coord1, class... Coords>
CUTE_HOST_DEVICE constexpr
auto
operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
  return operator()(make_coord(c0,c1,cs...));
}
```

没有它时你要写：

```cpp
layout(make_coord(2, 3))
```

有了它就能写：

```cpp
layout(2, 3)
```

### 点积公式

最直接的公式：

$$
\text{Index} = c_0s_0 + c_1s_1 + c_2s_2 + \dots
$$

例如 shape `(2,4)`。

列主序 stride `(1,2)`：

$$
L(1,3) = 1 \cdot 1 + 3 \cdot 2 = 7
$$

行主序 stride `(4,1)`：

$$
L(1,3) = 1 \cdot 4 + 3 \cdot 1 = 7
$$

两个坐标都得到 7，只是布局意义不同。

### 1D 坐标怎么拆成多维坐标

当你传给 Layout 一个一维坐标 $i$ 时，CuTe 会按 **colexicographical order（逆词典序 / 广义列主序）** 把它拆成多维坐标。

对于 shape：

$$
S = (L_0, L_1, L_2, \dots)
$$

拆解公式是：

$$
c_0 = i \bmod L_0
$$

$$
i_1 = \left\lfloor \frac{i}{L_0} \right\rfloor
$$

$$
c_1 = i_1 \bmod L_1
$$

继续递归。

对于 shape `(2,4)`，一维坐标 `5`：

$$
c_0 = 5 \bmod 2 = 1
$$

$$
i_1 = 5 / 2 = 2
$$

$$
c_1 = 2 \bmod 4 = 2
$$

所以：

$$
5 \mapsto (1,2)
$$

CuTe 的 `stride.hpp` 里 `crd2idx(coord, shape)` 对坐标到 index 的描述是：

```cpp
// i = c0 + s0 * (c1 + s1 * (c2 + s2 * ...))
```

并且源码里会先 flatten，再用 Horner 形式计算：

```cpp
auto flat_coord = flatten(coord);
auto flat_shape = flatten(product_like(shape, coord));
return detail::crd2idx_horner(flat_coord, flat_shape, tuple_seq<decltype(flat_shape)>{});
```

所以 `crd2idx` 不是只接受自然坐标。它会把兼容坐标正规化后再计算。

### `print2D`

为了以二维表格打印 Rank-2 Layout，可以写：

```cpp
template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const& layout) {
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m, n));
    }
    printf("\n");
  }
}
```

对于前面的布局：

`print2D(s2xs4)`，标准列主序：

```text
0    2    4    6
1    3    5    7
```

`print2D(s2xd4_a)`，带跨度的行主序：

```text
0    1    2    3
12   13   14   15
```

`print2D(s2xh4_col)`，层次化列主序：

```text
0    2    4    6
1    3    5    7
```

`print2D(s2xh4)`，非标准层次化布局：

```text
0    2    1    3
4    6    5    7
```

这里最值得注意的是 `s2xh4`：

```cpp
Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                           make_stride(4, make_stride(2, 1)));
```

它的 shape 是：

$$
(2,(2,2))
$$

它是 Rank-2，因为顶层有两个 mode。第二个 mode 自己是一个 `(2,2)` multi-mode。你仍然可以用二维坐标 `(m,n)` 访问它，因为第二个 mode 可以接收一维坐标 `n`，再内部拆成 `(n_0,n_1)`。

例如 `layout(0, 2)`：

- 顶层第 0 mode 坐标是 `0`。
- 顶层第 1 mode 坐标 `2` 被 shape `(2,2)` 拆成 `(0,1)`。
- 自然坐标是 `(0,(0,1))`。
- stride 是 `(4,(2,1))`。

所以：

$$
L(0,(0,1)) = 0 \cdot 4 + 0 \cdot 2 + 1 \cdot 1 = 1
$$

这就是输出中第一行第三列是 `1` 的原因。

### `print1D`

任意 multi-mode 都可以接收一维坐标。把整个 layout 当成 1D 遍历：

```cpp
template <class Shape, class Stride>
void print1D(Layout<Shape, Stride> const& layout) {
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

输出：

```text
print1D(s2xs4)    : 0 1 2 3 4 5 6 7
print1D(s2xd4_a)  : 0 12 1 13 2 14 3 15
print1D(s2xh4)    : 0 4 2 6 1 5 3 7
```

例如 `s2xd4_a = (_2,4):(_12,_1)`。

一维遍历先按 shape `(_2,4)` 拆坐标：

```text
0 -> (0,0) -> 0*12 + 0*1 = 0
1 -> (1,0) -> 1*12 + 0*1 = 12
2 -> (0,1) -> 0*12 + 1*1 = 1
3 -> (1,1) -> 1*12 + 1*1 = 13
```

所以得到：

```text
0 12 1 13 ...
```

### 可视化

CuTe 提供 `print_layout`：

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

`print_latex` 会生成 LaTeX 代码，方便画带颜色的 layout 图。

## 向量和矩阵示例

### 向量

Rank 等于 1 的 Layout 可以看作向量。

`8:1`：

```text
coord : 0 1 2 3 4 5 6 7
index : 0 1 2 3 4 5 6 7
```

公式：

$$
L(i) = i \cdot 1
$$

`8:2`：

```text
coord : 0 1 2 3 4 5 6 7
index : 0 2 4 6 8 10 12 14
```

公式：

$$
L(i) = i \cdot 2
$$

`((4,2)):((2,1))`：

```text
coord : 0 1 2 3 4 5 6 7
index : 0 2 4 6 1 3 5 7
```

它外面多了一层括号，所以顶层 Rank 仍然是 1。输入一维坐标 `i` 会先在内部 shape `(4,2)` 中拆：

$$
i \mapsto (i \bmod 4, \lfloor i/4 \rfloor)
$$

然后：

$$
L(i) = (i \bmod 4) \cdot 2 + \lfloor i/4 \rfloor \cdot 1
$$

`((4,2)):((1,4))`：

```text
coord : 0 1 2 3 4 5 6 7
index : 0 1 2 3 4 5 6 7
```

它虽然内部是 `(4,2)`，但作为函数等价于 `8:1`。

### 矩阵

Rank 等于 2 的 Layout 可以看作矩阵。

列主序：

```text
Shape  : (4,2)
Stride : (1,4)
```

公式：

$$
L(i,j) = i + 4j
$$

行主序：

```text
Shape  : (4,2)
Stride : (2,1)
```

公式：

$$
L(i,j) = 2i + j
$$

这里所谓 majorness（主序）只看哪个维度 stride 是 1：

- 第 0 维 stride 为 1，就是列主序。
- 第 1 维 stride 为 1，就是行主序。

层次化矩阵：

```text
Shape  : ((2,2),2)
Stride : ((4,1),2)
```

逻辑上仍然是 $4 \times 2$，因为第 0 mode 的 size 是：

$$
\operatorname{size}((2,2)) = 4
$$

如果坐标是 `(i,j)`，其中第 0 mode 的 `i` 会拆到 `(i0,i1)`：

$$
i_0 = i \bmod 2
$$

$$
i_1 = \lfloor i / 2 \rfloor
$$

索引：

$$
L(i,j) = 4i_0 + 1i_1 + 2j
$$

## 坐标兼容性

Layout 能接收的坐标集合由 Shape 决定。形状 A 与形状 B 兼容，直观上表示：A 的坐标可以被 B 接受。

兼容需要满足：

1. A 和 B 的 size 相同。
2. A 中所有坐标在 B 中都是有效坐标。

示例：

| A | B | 是否兼容 | 原因 |
| --- | --- | --- | --- |
| `24` | `32` | 否 | size 不同。 |
| `24` | `(4,6)` | 是 | 1D 坐标可被 `(4,6)` 拆成 2D 坐标。 |
| `(4,6)` | `((2,2),6)` | 是 | 第 0 mode 可以从 `4` 拆成 `(2,2)`。 |
| `((2,2),6)` | `((2,2),(3,2))` | 是 | 第 1 mode 可以从 `6` 拆成 `(3,2)`。 |
| `24` | `((2,2),(3,2))` | 是 | 1D 坐标可递归拆成自然坐标。 |
| `24` | `((2,3),4)` | 是 | size 相同，低 rank 坐标可拆入高层次 shape。 |
| `((2,3),4)` | `((2,2),(3,2))` | 否 | 分组结构不兼容。 |
| `((2,2),(3,2))` | `((2,3),4)` | 否 | 分组结构不兼容。 |
| `24` | `(24)` | 是 | 1D 坐标可进入 rank-1 tuple。 |
| `(24)` | `24` | 否 | `(24)` 是 tuple 坐标，普通 `24` 形状不能接收。 |
| `(24)` | `(4,6)` | 否 | 顶层结构不兼容。 |

为什么 `24` 兼容 `(4,6)`，反过来不行？

- shape `24` 的坐标集是 $\{0,1,\dots,23\}$。
- shape `(4,6)` 可以接收这些 1D 坐标，并按 colexicographical order 拆成 `(m,n)`。
- 但 shape `24` 是单整数 shape，不知道如何接收二维坐标 `(m,n)`。

所以兼容性不是“size 相等”这么简单，它还关心坐标结构能不能被目标 shape 接受。

## 坐标映射和索引映射

### 坐标映射：`idx2crd`

以 shape：

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
```

也就是：

$$
(3,(2,3))
$$

它可以接受：

- 1D 坐标：`16`
- 2D 坐标：`(1,5)`
- 自然坐标：`(1,(1,2))`

这些输入都可以映射到自然坐标：

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

以 1D 坐标 `16` 为例：

顶层 shape 是 `(3, (2,3))`。

第一步：

$$
i = 16
$$

$$
c_0 = 16 \bmod 3 = 1
$$

$$
i_1 = \lfloor 16/3 \rfloor = 5
$$

第二个 mode 的 shape 是 `(2,3)`，继续拆：

$$
c_1 = 5 \bmod 2 = 1
$$

$$
i_2 = \lfloor 5/2 \rfloor = 2
$$

$$
c_2 = 2 \bmod 3 = 2
$$

所以：

$$
16 \mapsto (1,(1,2))
$$

### 索引映射：`crd2idx`

布局：

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
```

即：

$$
(3,(2,3)) : (3,(12,1))
$$

自然坐标 `(i,(j,k))` 的索引：

$$
Index = i \cdot 3 + j \cdot 12 + k \cdot 1
$$

例如自然坐标 `(1,(1,2))`：

$$
Index = 1 \cdot 3 + 1 \cdot 12 + 2 \cdot 1 = 17
$$

所以这些输入都得到 `17`：

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

表格展开如下：

| `i \ (j,k)` | `(0,0)` | `(1,0)` | `(0,1)` | `(1,1)` | `(0,2)` | `(1,2)` |
| --- | --- | --- | --- | --- | --- | --- |
| `0` | 0 | 12 | 1 | 13 | 2 | 14 |
| `1` | 3 | 15 | 4 | 16 | 5 | 17 |
| `2` | 6 | 18 | 7 | 19 | 8 | 20 |

## 布局提取

CuTe 提供几种方式从复杂 layout 中提取子布局。

### `layout<I...>`

`layout<I...>` 进入层次结构内部，提取某个子 layout。源码里会分别取 shape 和 stride：

```cpp
return make_layout(get<Is...>(layout.shape()),
                   get<Is...>(layout.stride()));
```

示例：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

公式上：

$$
\operatorname{layout}^{I...}(S:D)
=
\operatorname{get}^{I...}(S) : \operatorname{get}^{I...}(D)
$$

### `select<I...>`

`select<I...>` 从顶层挑选若干 mode，重新组成 layout：

```cpp
return make_layout(select<Is...>(layout.shape()),
                   select<Is...>(layout.stride()));
```

示例：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

公式：

$$
\operatorname{select}^{I_0,\dots,I_n}(S:D)
=
(S_{I_0},\dots,S_{I_n}) : (D_{I_0},\dots,D_{I_n})
$$

### `take<B,E>`

`take<B,E>` 选取连续区间 `[B,E)`：

```cpp
template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
take(Layout<Shape,Stride> const& layout)
{
  static_assert(B < E, "take: empty range error");
  static_assert(0 <= B && E <= Layout<Shape,Stride>::rank, "take: range out of bounds");
  return make_layout(take<B,E>(layout.shape()),
                     take<B,E>(layout.stride()));
}
```

示例：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

## 拼接与组合

### 用 `make_layout` 连接多个 layout

如果把已有 Layout 传给 `make_layout`，它会把 shape 和 stride 分别包装或拼接。

示例：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

这里要注意：

- `make_layout(a,b)` 不是函数复合，而是 mode 拼接。
- `make_layout(a)` 会包一层，改变层次结构。
- `Shape` 和 `Stride` 会同步改变，保持 congruent。

### `append`、`prepend`、`replace`

先不要把它们想成 CUDA 或 Layout 的复杂概念。它们最底层就是 **tuple 结构操作**。

| 操作 | 直观含义 | 对普通 tuple 的效果 |
| --- | --- | --- |
| `append(a, x)` | 把 `x` 放到 `a` 的末尾 | `(a0,a1)` 变成 `(a0,a1,x)` |
| `prepend(a, x)` | 把 `x` 放到 `a` 的开头 | `(a0,a1)` 变成 `(x,a0,a1)` |
| `replace<N>(a, x)` | 把第 `N` 个元素替换成 `x` | `(a0,a1,a2)` 的 `replace<1>` 变成 `(a0,x,a2)` |

如果输入不是 tuple，CuTe 会先把它当作一个单元素结构处理：

```cpp
append(_3{}, _4{})      // (_3,_4)
prepend(_3{}, _4{})     // (_4,_3)
replace<0>(_3{}, _4{})  // _4
```

如果输入已经是 tuple：

```cpp
auto t = make_tuple(_2{}, _3{});

append(t, _5{})       // (_2,_3,_5)
prepend(t, _5{})      // (_5,_2,_3)
replace<1>(t, _5{})   // (_2,_5)
```

源码里 tuple 层面的 `append` 对非 tuple 会把它包装成 tuple：

```cpp
if constexpr (is_tuple<T>::value) {
  return detail::construct(a, x, make_seq<tuple_size<T>::value>{}, seq<0>{}, seq<>{});
} else {
  return cute::make_tuple(a, x);
}
```

`prepend` 的非 tuple 分支就是反过来：

```cpp
return cute::make_tuple(x, a);
```

`replace<N>` 的非 tuple 分支要求 `N == 0`：

```cpp
static_assert(N == 0);
return x;
```

因此：

- `append` / `prepend` 会改变 tuple 的长度。
- `replace` 不改变长度，只替换某个位置。
- 对非 tuple 的整数，`append` / `prepend` 会把它提升成 rank 更高的 tuple。

Layout 级别的实现是在这个 tuple 操作上再套一层：**shape 怎么改，stride 就必须用同样的结构方式改**。

```cpp
template <class ShapeA, class StrideA, class ShapeX, class StrideX>
CUTE_HOST_DEVICE constexpr
auto
append(Layout<ShapeA,StrideA> const& layout,
       Layout<ShapeX,StrideX> const& x)
{
  return make_layout(append(layout.shape(),  x.shape()),
                     append(layout.stride(), x.stride()));
}
```

`prepend` 和 `replace` 也是同样的模式：

```cpp
return make_layout(prepend(layout.shape(),  x.shape()),
                   prepend(layout.stride(), x.stride()));

return make_layout(replace<N>(layout.shape(),  x.shape()),
                   replace<N>(layout.stride(), x.stride()));
```

所以它们本质上是：

$$
\operatorname{append}(S:D, S_x:D_x)
=
\operatorname{append}(S,S_x) : \operatorname{append}(D,D_x)
$$

$$
\operatorname{prepend}(S:D, S_x:D_x)
=
\operatorname{prepend}(S,S_x) : \operatorname{prepend}(D,D_x)
$$

$$
\operatorname{replace}^N(S:D, S_x:D_x)
=
\operatorname{replace}^N(S,S_x) : \operatorname{replace}^N(D,D_x)
$$

为什么要 shape 和 stride 一起改？

因为 Layout 的不变量是：

$$
\operatorname{congruent}(\text{shape}, \text{stride}) = \text{true}
$$

shape 增加一个 mode，stride 也必须增加对应 mode；shape 替换第 `N` 个 mode，stride 也必须替换第 `N` 个 mode。否则坐标叶子和 stride 叶子无法一一对应，映射公式就不成立。

示例：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

逐个看。

`append(a,b)`：

$$
a = 3:1,\quad b = 4:3
$$

$$
\operatorname{append}(a,b)
=
\operatorname{append}(3,4) : \operatorname{append}(1,3)
=
(3,4):(1,3)
$$

它把 `b` 放在末尾，所以新坐标是 `(i,j)`，映射为：

$$
ab(i,j) = i \cdot 1 + j \cdot 3
$$

`prepend(a,b)`：

$$
\operatorname{prepend}(a,b)
=
\operatorname{prepend}(3,4) : \operatorname{prepend}(1,3)
=
(4,3):(3,1)
$$

它把 `b` 放在开头，所以新坐标是 `(j,i)`，映射为：

$$
ba(j,i) = j \cdot 3 + i \cdot 1
$$

`replace<2>(c,b)`：

```cpp
c = (3,4,(3,4)):(1,3,(1,3))
b = 4:3
```

替换第 2 个顶层 mode：

$$
\operatorname{replace}^2(c,b)
=
(3,4,4):(1,3,3)
$$

这里不是把内部 `(3,4)` 展开，也不是做乘法合并，而是把顶层第 2 个 mode 整个替换成 `b` 的 shape/stride。

## 分组与展平

分组和展平是文章里最容易只讲“怎么写”、没讲公式的部分。它们其实非常重要：**它们改变 layout 的层次结构，但不改变叶子顺序，也不改变叶子 shape/stride 的对应关系**。

### `group<B,E>`

`group<B,E>` 把顶层 `[B,E)` 这一段 mode 包成一个 multi-mode。

Layout 层源码：

```cpp
template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
group(Layout<Shape,Stride> const& layout)
{
  return make_layout(group<B,E>(layout.shape()),
                     group<B,E>(layout.stride()));
}
```

tuple 层源码注释给了例子：

```cpp
// group<2,4>(T<_1,_2,_3,_4,_5,_6>{})
//          => T<_1,_2,T<_3,_4>,_5,_6>{}
```

公式上，如果：

$$
S = (s_0, s_1, \dots, s_{r-1})
$$

$$
D = (d_0, d_1, \dots, d_{r-1})
$$

那么：

$$
\operatorname{group}^{B,E}(S)
=
(s_0,\dots,s_{B-1},(s_B,\dots,s_{E-1}),s_E,\dots,s_{r-1})
$$

stride 同理：

$$
\operatorname{group}^{B,E}(D)
=
(d_0,\dots,d_{B-1},(d_B,\dots,d_{E-1}),d_E,\dots,d_{r-1})
$$

所以：

$$
\operatorname{group}^{B,E}(S:D)
=
\operatorname{group}^{B,E}(S)
:
\operatorname{group}^{B,E}(D)
$$

示例：

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

对 `a`：

$$
S_a = (2,3,5,7)
$$

$$
D_a = (1,2,6,30)
$$

执行 `group<0,2>`：

$$
S_b = ((2,3),5,7)
$$

$$
D_b = ((1,2),6,30)
$$

注意它没有把前两个 mode 乘成 `6`，而是把 `(2,3)` 包成一个 multi-mode。内部结构仍然保留。

如果坐标是：

$$
((i,j), k, l)
$$

那么映射仍然是：

$$
L_b((i,j),k,l) = i \cdot 1 + j \cdot 2 + k \cdot 6 + l \cdot 30
$$

这和原 layout：

$$
L_a(i,j,k,l) = i \cdot 1 + j \cdot 2 + k \cdot 6 + l \cdot 30
$$

是同一个叶子级公式，只是坐标外形变了。

这就是“分组改变层级，不改变叶子映射”的含义。

### `flatten`

`flatten` 把层次结构压平到 depth 不超过 1。

Layout 层源码：

```cpp
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
flatten(Layout<Shape,Stride> const& layout)
{
  return make_layout(flatten(layout.shape()),
                     flatten(layout.stride()));
}
```

tuple 层源码：

```cpp
template <class T>
CUTE_HOST_DEVICE constexpr
auto
flatten(T const& t)
{
  if constexpr (is_tuple<T>::value) {
    if constexpr (is_flat<T>::value) {
      return t;
    } else {
      return filter_tuple(t, [](auto const& a) { return flatten_to_tuple(a); });
    }
  } else {
    return t;
  }
}
```

公式上：

$$
\operatorname{flatten}(((2,3),(5,7))) = (2,3,5,7)
$$

对于 layout：

$$
\operatorname{flatten}(((2,3),(5,7)):((1,2),(6,30)))
=
(2,3,5,7):(1,2,6,30)
$$

因此前面示例中：

```cpp
flatten(b) == a
flatten(c) == a
```

从映射角度看：

$$
L_c((i,j),(k,l)) = i + 2j + 6k + 30l
$$

flatten 后：

$$
L_f(i,j,k,l) = i + 2j + 6k + 30l
$$

叶子级乘加公式完全相同，只是输入坐标的括号结构不同。

### 分组为什么有用

分组常用于把高 rank layout 临时看成低 rank layout。

例如：

```cpp
(_2,_3,_5,_7):(_1,_2,_6,_30)
```

可以通过：

```cpp
group<0,2>
group<1,3>
```

变成：

```cpp
((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
```

这样它顶层 Rank 从 4 变成 2，可以被当成矩阵处理；但每个矩阵 mode 内部仍然保留原来的二维结构。

这就是 CuTe 里“把复杂 tensor 逻辑化地划分到线程布局上”的基础能力。

## 小结

CuTe Layout 可以按三层理解：

1. **数据结构层**：`Layout<Shape, Stride>` 私有继承 `tuple<Shape, Stride>`，Shape 和 Stride 都是 `IntTuple`。
2. **数学层**：Layout 是从坐标到索引的函数，核心是坐标正规化和递归点积。
3. **代数层**：`layout`、`select`、`take`、`append`、`prepend`、`replace`、`group`、`flatten` 都是在同步变换 Shape 和 Stride。

最重要的心智模型是：

$$
\text{Layout} = \text{Shape} : \text{Stride}
$$

$$
\text{index} = \langle \text{natural coord}, \text{stride} \rangle
$$

只要 Shape 和 Stride 的叶子结构保持对应，CuTe 就能在编译期和运行期之间自由混合，用同一套接口描述非常复杂的 GPU 数据访问模式。
