---
title: C++ CRTP 单例模式
date: 2026-06-25
tags: [C++, CRTP, 单例模式, 设计模式]
summary: 整理 C++ 单例模式的常见写法，重点解释 CRTP 如何复用单例基类，以及单例类的配置接口应该如何设计。
---

# C++ CRTP 单例模式

单例模式（Singleton）的目标是：**保证一个类型在进程内只有一个实例，并提供统一访问入口**。

在 C++ 里，最推荐先记住这条经验：

- 如果只是一个普通全局服务对象，优先考虑 **Meyers Singleton**，也就是函数内 `static` 局部变量。
- 如果有很多类都需要写成单例，可以用 **CRTP 单例基类**复用 `GetInstance()`、禁用拷贝、统一生命周期管理。
- 如果单例需要配置，优先让配置在实例构造前确定，不要让全局对象在运行中被随意修改。

## 单例解决什么问题

单例适合表达“进程内唯一资源”：

- **日志系统**：统一管理日志格式、输出目标、flush 策略。
- **配置中心**：统一读取命令行参数、配置文件、环境变量。
- **线程池或连接池**：统一管理一组昂贵资源。
- **运行时上下文**：保存设备、stream、工作目录、全局开关等信息。

它带来的便利也很明显：

- 调用方不需要到处传对象引用。
- 可以集中控制对象构造和销毁。
- 可以避免重复创建昂贵资源。

但单例也容易带来问题：

- **全局状态会增加隐式依赖**，测试和复用会变难。
- **初始化顺序需要明确**，尤其是配置型单例。
- **运行时修改配置容易产生并发问题**。

所以单例应该用在真正需要“唯一实例”的地方，不要把它当成省参数传递的万能工具。

## 最简单可靠的写法

C++11 之后，函数内 `static` 局部变量的初始化是线程安全的。这个写法通常足够好。

```cpp
#include <iostream>

/**
 * @brief 一个最小的 Meyers Singleton 示例。
 *
 * 实例会在第一次调用时构造。C++11 之后，函数内局部静态变量的初始化
 * 由语言保证线程安全。
 */
class Logger {
public:
    /**
     * @brief 返回唯一的 Logger 实例。
     *
     * @return 进程内唯一 Logger 实例的引用。
     */
    static Logger& getInstance()
    {
        static Logger instance;
        return instance;
    }

    void log(const char* message) const
    {
        std::cout << message << '\n';
    }

private:
    Logger() = default;

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};
```

使用方式：

```cpp
Logger::getInstance().log("hello singleton");
```

这个版本有几个优点：

- 不需要手动 `new/delete`。
- 不需要自己写锁。
- 不需要智能指针管理生命周期。
- 构造发生在第一次调用 `getInstance()` 时。

## 不推荐的写法

理解单例时，很有必要看一些“前人常见错误”。这些写法不一定都完全不能用，但它们暴露了单例最容易出问题的几个点：

- 何时构造。
- 谁负责析构。
- 多线程下是否只构造一次。
- 调用方是否拿到了不该拿的所有权。

### 饿汉式：程序启动时就创建

饿汉式是在程序启动阶段就创建实例：

```cpp
class Logger {
public:
    static Logger& getInstance()
    {
        return mInstance;
    }

    void log(const char* message) const;

private:
    Logger() = default;

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    static Logger mInstance;
};

Logger Logger::mInstance;
```

它的特点是：

- **线程安全通常不是问题**，因为对象在进入 `main()` 前就已经构造。
- **不支持按需创建**，即使程序从来不用 Logger，也会构造它。
- **容易遇到静态初始化顺序问题**。如果多个全局对象之间互相依赖，跨翻译单元的初始化顺序并不可靠。

饿汉式不是绝对错误，但在 C++ 里一般不如函数内 `static` 局部变量稳妥。

### 裸指针懒汉式：线程不安全且容易泄漏

懒汉式是在第一次使用时才创建实例。裸指针版本很常见，但问题最多：

```cpp
class Logger {
public:
    static Logger& getInstance()
    {
        if (mInstance == nullptr) {
            mInstance = new Logger();
        }
        return *mInstance;
    }

private:
    Logger() = default;

    static Logger* mInstance;
};

Logger* Logger::mInstance = nullptr;
```

问题是：

- **线程不安全**：两个线程可能同时看到 `mInstance == nullptr`，然后各自 `new Logger()`。
- **内存不安全**：没有明确释放逻辑时会泄漏。
- **异常安全差**：构造过程如果抛异常，状态管理容易混乱。

### 返回裸指针：调用方可能误删

如果把接口写成返回裸指针，问题会更明显：

```cpp
class Logger {
public:
    static Logger* getInstance()
    {
        if (mInstance == nullptr) {
            mInstance = new Logger();
        }
        return mInstance;
    }

private:
    Logger() = default;

    static Logger* mInstance;
};
```

调用方可能这样写：

```cpp
delete Logger::getInstance();
```

这在语法上可能成立，但语义上非常危险。单例对象的生命周期应该由单例类自己管理，调用方不应该获得释放它的权力。

如果确实要返回指针，也应该把它视为**观察指针**，并在接口设计上避免调用方误解所有权。多数情况下，返回 `Logger&` 更清楚。

### 加锁懒汉式：看起来安全但容易写错

最直接的修复方式是加锁：

```cpp
#include <memory>
#include <mutex>

class Logger {
public:
    static Logger& getInstance()
    {
        std::lock_guard<std::mutex> lock(mMutex);

        if (mInstance == nullptr) {
            mInstance.reset(new Logger());
        }

        return *mInstance;
    }

private:
    Logger() = default;

    static std::mutex mMutex;
    static std::unique_ptr<Logger> mInstance;
};
```

这个版本解决了重复构造和泄漏问题，但仍然不算优雅：

- 每次 `getInstance()` 都要加锁，即使对象已经创建。
- 静态成员需要在 `.cpp` 中定义，头文件单独使用不方便。
- 析构顺序仍然需要注意，尤其是其他全局对象在析构时访问 Logger。

如果只是想要线程安全懒加载，C++11 的局部静态变量更简单。

### 双重检查锁：C++ 里不建议手写

为了减少每次调用都加锁，有人会写双重检查：

```cpp
class Logger {
public:
    static Logger* getInstance()
    {
        if (mInstance == nullptr) {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mInstance == nullptr) {
                mInstance = new Logger();
            }
        }

        return mInstance;
    }

private:
    Logger() = default;

    static std::mutex mMutex;
    static Logger* mInstance;
};
```

这类写法在现代 C++ 里不建议手写。原因是：

- 如果没有正确使用 `std::atomic` 和内存序，可能出现可见性和重排序问题。
- 代码复杂度高，收益很小。
- C++11 之后局部静态变量已经提供了更简单的线程安全懒初始化。

学习时知道它想解决“重复加锁”的问题即可，工程里优先不用。

### `std::call_once`：可以用，但不一定必要

`std::call_once` 是标准库提供的一次性初始化工具：

```cpp
#include <memory>
#include <mutex>

class Logger {
public:
    static Logger& getInstance()
    {
        std::call_once(mInitFlag, [] {
            mInstance.reset(new Logger());
        });

        return *mInstance;
    }

private:
    Logger() = default;

    static std::once_flag mInitFlag;
    static std::unique_ptr<Logger> mInstance;
};
```

这个版本线程安全，也能保证只初始化一次。它适合初始化过程比较复杂、不能直接放进局部静态变量构造函数的场景。

但普通单例里，它仍然比 Meyers Singleton 更啰嗦：

```cpp
static Logger& getInstance()
{
    static Logger instance;
    return instance;
}
```

### `std::unique_ptr` 管理内部实例：可以，但不要返回所有权

内部用 `std::unique_ptr` 管理单例对象是可以的：

```cpp
class Logger {
public:
    static Logger& getInstance()
    {
        if (!mInstance) {
            mInstance.reset(new Logger());
        }

        return *mInstance;
    }

private:
    Logger() = default;

    static std::unique_ptr<Logger> mInstance;
};
```

但是这个版本仍然有两个注意点：

- 这里没有加锁，所以多线程第一次调用仍然不安全。
- 构造函数是私有的，`std::make_unique<Logger>()` 通常不能直接访问私有构造函数，所以经常会看到 `reset(new Logger())`。

### 返回 `std::unique_ptr`

`std::unique_ptr` 表示独占所有权。单例不应该把所有权交给调用方。

```cpp
static std::unique_ptr<Logger> getInstance()
{
    return std::move(mInstance);
}
```

这个接口语义是错误的：调用一次就把单例对象移走了，下一次再访问时内部指针可能已经为空。单例访问接口应该返回**引用**、**裸观察指针**，或者在少数需要共享生命周期的场景返回 `std::shared_ptr`。

### 返回 `std::shared_ptr`：能用但语义变复杂

如果一定要返回智能指针，`std::shared_ptr` 比 `std::unique_ptr` 更符合“共享访问”的语义：

```cpp
#include <memory>
#include <mutex>

class Logger {
public:
    static std::shared_ptr<Logger> getInstance()
    {
        static std::once_flag init_flag;
        std::call_once(init_flag, [] {
            mInstance = std::shared_ptr<Logger>(new Logger());
        });

        return mInstance;
    }

private:
    Logger() = default;

    static std::shared_ptr<Logger> mInstance;
};
```

这个写法的主要问题不是正确性，而是语义：

- 调用方拿到的是共享所有权，可能长期持有对象。
- 生命周期不再只是“单例类内部控制”，而是和外部 `shared_ptr` 副本有关。
- 如果未来要重置或替换实例，持有旧 `shared_ptr` 的调用方仍然会延长旧对象生命周期。

所以默认还是推荐返回 `T&`。只有当框架或测试确实需要共享所有权时，再考虑 `shared_ptr`。

### 推荐对比表

| 写法 | 是否懒加载 | 线程安全 | 生命周期 | 推荐程度 |
|---|---:|---:|---|---|
| 饿汉式静态成员 | 否 | 通常安全 | 静态对象，可能有初始化顺序问题 | 一般 |
| 裸指针懒汉式 | 是 | 否 | 容易泄漏或误删 | 不推荐 |
| 加锁懒汉式 | 是 | 是 | 可控但啰嗦 | 可用 |
| 双重检查锁 | 是 | 容易写错 | 复杂 | 不推荐手写 |
| `std::call_once` | 是 | 是 | 可控 | 可用 |
| Meyers Singleton | 是 | 是 | 局部静态对象自动管理 | 推荐 |
| 返回 `std::unique_ptr` | 是 | 取决于实现 | 所有权被移走 | 不推荐 |
| 返回 `std::shared_ptr` | 是 | 取决于实现 | 共享所有权复杂 | 谨慎使用 |

## CRTP 是什么

CRTP 的全称是 **Curiously Recurring Template Pattern**，中文常叫“奇异递归模板模式”。

它的形式是：

```cpp
template <typename Derived>
class Base {
};

class RealType : public Base<RealType> {
};
```

看起来像“派生类把自己作为模板参数传给基类”，所以叫递归模板模式。

CRTP 的核心价值是：**基类在编译期知道派生类的具体类型**。这让基类可以在不使用虚函数的情况下，复用一套和派生类相关的逻辑。

## CRTP 与普通继承的区别

普通动态多态依赖虚函数：

```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual void eat() = 0;
};
```

调用时通过 vtable 在运行期分派。

CRTP 是静态多态：

```cpp
#include <iostream>

/**
 * @brief CRTP 基类，把公共接口转发给派生类实现。
 *
 * @tparam Derived 继承自 Animal<Derived> 的具体派生类型。
 */
template <typename Derived>
class Animal {
public:
    void eat()
    {
        static_cast<Derived*>(this)->eatImpl();
    }
};

class Dog : public Animal<Dog> {
public:
    void eatImpl()
    {
        std::cout << "dog eat\n";
    }
};
```

使用方式：

```cpp
Dog dog;
dog.eat();
```

这里没有虚函数，`Animal<Dog>` 在编译期就知道 `Derived = Dog`，所以可以通过 `static_cast<Derived*>(this)` 调用派生类实现。

CRTP 的特点：

- **编译期绑定**：不需要 vtable，不依赖运行时动态派发。
- **适合复用模板逻辑**：基类可以为多个派生类生成同一套接口。
- **每个派生类都有独立的基类实例化**：`Base<A>` 和 `Base<B>` 是两个不同类型。
- **不适合运行时异构集合**：如果需要 `std::vector<std::unique_ptr<Base>>` 这种运行时多态，仍然要用虚函数。

## 为什么 CRTP 适合实现单例基类

如果不用 CRTP，每个单例类都要重复写：

- `getInstance()`。
- 删除拷贝构造和拷贝赋值。
- 私有构造。
- 线程安全初始化。

CRTP 可以把公共逻辑收进模板基类：

```cpp
Singleton<T>
```

其中 `T` 是真正的单例类型。这样每个派生类都会拥有自己独立的单例实例：

```cpp
class Logger : public Singleton<Logger> {};
class ConfigManager : public Singleton<ConfigManager> {};
```

`Singleton<Logger>` 和 `Singleton<ConfigManager>` 是两个不同模板实例，因此不会共享同一个静态对象。

## CRTP 单例基类

下面是一个推荐版本：返回引用，使用函数内 `static` 局部变量完成线程安全初始化。

```cpp
#pragma once

#include <utility>

/**
 * @brief CRTP 单例辅助基类，为每个派生类型创建一个唯一实例。
 *
 * @tparam Derived 具体的单例类型。派生类需要继承 Singleton<Derived>，
 * 并把 Singleton<Derived> 声明为友元。
 */
template <typename Derived>
class Singleton {
public:
    /**
     * @brief 返回 Derived 类型的唯一实例。
     *
     * 实例会在第一次调用时构造。C++11 之后，局部静态变量初始化由语言保证线程安全。
     *
     * @return 唯一 Derived 实例的引用。
     */
    static Derived& getInstance()
    {
        static Derived instance;
        return instance;
    }

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

protected:
    Singleton() = default;
    ~Singleton() = default;
};
```

派生类这样写：

```cpp
#include <iostream>

class Logger : public Singleton<Logger> {
    friend class Singleton<Logger>;

public:
    void log(const char* message) const
    {
        std::cout << message << '\n';
    }

private:
    Logger() = default;
};
```

使用方式：

```cpp
Logger::getInstance().log("hello CRTP singleton");
```

这里有几个关键点：

- `Logger` 继承 `Singleton<Logger>`，这是 CRTP 的固定写法。
- `friend class Singleton<Logger>` 让基类可以访问 `Logger` 的私有构造函数。
- `Logger()` 放在 `private`，外部无法直接创建第二个实例。
- `getInstance()` 返回引用，调用方没有所有权，不应该负责释放。

## 为什么不直接用 `std::shared_ptr`

很多示例会把单例写成：

```cpp
static std::shared_ptr<T> getInstance();
```

这不是不能用，但默认不推荐。

原因是：

- `shared_ptr` 表示共享所有权，但单例通常不需要把所有权暴露给调用者。
- 调用方拿到 `shared_ptr` 后可能长期持有，生命周期语义更复杂。
- 如果对象本来就应该活到进程结束，函数内 `static` 引用更简单。

适合返回 `shared_ptr` 的场景通常是：

- 单例对象需要被显式销毁和重建。
- 测试中需要替换实例。
- 框架要求统一使用智能指针管理组件。

否则优先返回 `T&`。

## 单例类如何配置

单例最容易踩坑的地方不是 `getInstance()`，而是**配置什么时候传进去**。

配置型单例通常有三种接口设计：

- 先 `configure(config)`，再 `getInstance()`。
- `getInstance(config)` 第一次调用时传配置。
- 实例创建后通过 `setConfig(config)` 修改配置。

一般推荐顺序是：

```cpp
configure(config) -> getInstance()
```

它的优点是初始化边界清晰，业务代码能明确看到“先配置，再取实例”。

## 推荐方案：Config + configure + getInstance

这种方案适合配置只应该在启动阶段确定，运行期间不应该随意修改的场景。

```cpp
#pragma once

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

struct LoggerConfig {
    std::string file_path;
    bool enable_console{true};
    int flush_interval_ms{1000};
};

/**
 * @brief 进程级 Logger，要求在第一次使用前完成配置。
 */
class Logger : public Singleton<Logger> {
    friend class Singleton<Logger>;

public:
    /**
     * @brief 在单例创建前设置 Logger 配置。
     *
     * @param config Logger 的启动配置。实例创建后该配置不再被替换。
     *
     * @throws std::logic_error 如果 Logger 已经创建，则抛出异常。
     */
    static void configure(LoggerConfig config)
    {
        std::lock_guard<std::mutex> lock(configMutex());

        if (isCreated()) {
            throw std::logic_error("Logger has already been created.");
        }

        pendingConfig() = std::move(config);
    }

    void log(const std::string& message) const
    {
        if (mConfig.enable_console) {
            // 真实实现里可以在这里写入控制台和文件输出端。
        }
    }

private:
    Logger()
        : mConfig(loadConfig())
    {
        isCreated() = true;
    }

    static LoggerConfig loadConfig()
    {
        std::lock_guard<std::mutex> lock(configMutex());

        if (!pendingConfig().has_value()) {
            throw std::logic_error("Logger must be configured before getInstance().");
        }

        return *pendingConfig();
    }

    static std::optional<LoggerConfig>& pendingConfig()
    {
        static std::optional<LoggerConfig> config;
        return config;
    }

    static bool& isCreated()
    {
        static bool created = false;
        return created;
    }

    static std::mutex& configMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

private:
    LoggerConfig mConfig;
};
```

使用方式：

```cpp
int main()
{
    LoggerConfig config;
    config.file_path = "app.log";
    config.enable_console = true;
    config.flush_interval_ms = 500;

    Logger::configure(std::move(config));
Logger::getInstance().log("server started");
}
```

### 三个静态状态的作用

这段代码里有三个额外的静态状态：

- `pendingConfig()`：保存**尚未消费的启动配置**。
- `isCreated()`：记录**单例实例是否已经构造过**。
- `configMutex()`：保护配置读写过程，避免多线程下 `configure()` 和 `getInstance()` 交错访问。

它们都写成函数内 `static`：

```cpp
static std::optional<LoggerConfig>& pendingConfig()
{
    static std::optional<LoggerConfig> config;
    return config;
}
```

这种写法有两个好处：

- **避免在头文件里定义模板或类静态成员带来的链接问题**。函数内 `static` 的定义留在函数体里，调用时返回同一个对象引用。
- **延迟初始化**。这些状态只会在第一次调用对应函数时创建，不会提前参与全局初始化顺序。

`pendingConfig()` 用 `std::optional<LoggerConfig>`，是为了区分两种状态：

- 还没有调用过 `configure()`，此时 `pendingConfig().has_value()` 为 `false`。
- 已经设置过配置，构造函数可以安全读取配置。

如果不用 `optional`，只保存一个默认构造的 `LoggerConfig`，就很难判断用户到底是忘记配置，还是确实想使用默认配置。

`isCreated()` 用来禁止实例创建后再次配置：

```cpp
if (isCreated()) {
    throw std::logic_error("Logger has already been created.");
}
```

这是为了避免这种危险情况：

```cpp
Logger::getInstance().log("first");
Logger::configure(new_config); // 实例已经构造，配置再写入就不会影响 mConfig
```

如果允许这种调用，调用方会以为配置已经更新，但 `Logger` 内部保存的 `mConfig` 仍然是构造时的旧配置。所以这里选择直接抛异常，让错误尽早暴露。

`configMutex()` 用来保护两类操作：

- `configure()` 写入 `pendingConfig()`，并检查 `isCreated()`。
- `loadConfig()` 读取 `pendingConfig()`，并在构造函数中完成配置消费。

它保护的是**配置初始化协议**，不是 `Logger::log()` 的运行时日志写入逻辑。也就是说，实例创建完成以后，`mConfig` 已经是对象成员，普通读取不再依赖这把锁。

这种设计的重点是：

- `LoggerConfig` 把所有配置收成一个结构体，避免 `getInstance(a, b, c, d)` 这种参数爆炸。
- `configure()` 只允许在实例创建前调用。
- `getInstance()` 是显式访问入口，业务代码读起来很清楚。
- 构造函数从 `pendingConfig()` 取配置，实例创建后配置固定。

## 方案变体：getInstance(config)

也可以把配置放进 `getInstance(config)`：

```cpp
class Runtime : public Singleton<Runtime> {
    friend class Singleton<Runtime>;

public:
    struct Config {
        int worker_count{4};
        std::string data_dir;
    };

    /**
     * @brief 返回 Runtime 单例，并在第一次调用时完成初始化。
     *
     * @param config 只在单例首次创建时生效的配置。
     * @return 进程内唯一 Runtime 实例的引用。
     */
    static Runtime& getInstance(const Config& config)
    {
        static Runtime instance(config);
        return instance;
    }

    static Runtime& getInstance()
    {
        return getInstance(Config{});
    }

private:
    explicit Runtime(Config config)
        : mConfig(std::move(config))
    {
    }

private:
    Config mConfig;
};
```

这个写法看起来方便，但有一个隐患：**只有第一次调用传入的配置会生效**。

例如：

```cpp
Runtime::getInstance(Runtime::Config{.worker_count = 8});
Runtime::getInstance(Runtime::Config{.worker_count = 16});
```

第二次的 `worker_count = 16` 不会改变已经创建好的实例。为了避免误解，如果使用这种方案，最好在文档里明确说明：

- 配置只在第一次调用时生效。
- 后续调用 `getInstance(config)` 不会重新配置。

## 方案变体：setConfig

如果配置需要运行时修改，可以提供 `setConfig()`。

```cpp
#include <mutex>
#include <string>

class ServiceRegistry : public Singleton<ServiceRegistry> {
    friend class Singleton<ServiceRegistry>;

public:
    struct Config {
        std::string endpoint;
        int timeout_ms{1000};
    };

    /**
     * @brief 替换运行时配置。
     *
     * @param config 新配置。对象会在互斥锁保护下拷贝该配置。
     */
    void setConfig(Config config)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mConfig = std::move(config);
    }

    Config config() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mConfig;
    }

private:
    ServiceRegistry() = default;

private:
    mutable std::mutex mMutex;
    Config mConfig;
};
```

这个方案适合运行时确实需要热更新的场景，但代价是：

- 所有读取配置的路径都要考虑锁或快照。
- 配置更新和正在执行的业务逻辑可能交错。
- 测试需要覆盖配置变更过程。

如果配置不需要热更新，不要为了灵活而默认提供 `setConfig()`。

## 方案变体：依赖注入

有些对象其实不应该做成单例。

例如业务服务、数据库仓库、算法组件，如果只是“用起来到处都要传很麻烦”，更适合用依赖注入：

```cpp
class UserService {
public:
    explicit UserService(Logger& logger)
        : mLogger(logger)
    {
    }

    void createUser()
    {
        mLogger.log("create user");
    }

private:
    Logger& mLogger;
};
```

这种写法的好处是：

- 依赖关系显式。
- 测试时可以传入 fake logger。
- 不会把所有组件都绑死到全局对象。

单例适合基础设施对象，依赖注入更适合业务对象。

## 完整推荐模板

如果只是想快速写一个 CRTP 单例基类，可以使用这个版本：

```cpp
#pragma once

/**
 * @brief CRTP 单例基类。
 *
 * @tparam Derived 具体的单例类型。
 */
template <typename Derived>
class Singleton {
public:
    /**
     * @brief 返回进程内唯一的 Derived 实例。
     *
     * @return 单例对象的引用。
     */
    static Derived& getInstance()
    {
        static Derived instance;
        return instance;
    }

    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

protected:
    Singleton() = default;
    ~Singleton() = default;
};
```

派生类模板：

```cpp
class MyManager : public Singleton<MyManager> {
    friend class Singleton<MyManager>;

public:
    void run()
    {
        // 在这里执行真实业务逻辑。
    }

private:
    MyManager() = default;
};
```

使用：

```cpp
MyManager::getInstance().run();
```

## 写单例时的检查清单

- **实例访问方式**：优先返回 `T&`，不要把所有权暴露给调用方。
- **生命周期**：优先使用函数内 `static`，避免手写 `new/delete`。
- **构造权限**：派生类构造函数放 `private`，CRTP 基类作为友元。
- **拷贝控制**：删除拷贝构造和拷贝赋值。
- **配置方式**：优先使用 `Config + configure + getInstance`，并限制配置只能在实例创建前设置。
- **线程安全**：C++11 局部静态初始化天然线程安全；运行时配置更新需要额外锁。
- **测试性**：如果对象需要频繁替换、mock 或按场景创建，优先考虑依赖注入，不要强行做单例。

## 核心结论

CRTP 实现单例的本质是：**把“如何成为单例”的模板逻辑放进基类，把“具体业务能力”留给派生类**。

对配置型单例来说，重点不是模板技巧，而是初始化协议：

- 配置应该集中到 `Config` 结构体里。
- 配置应该在 `getInstance()` 前完成。
- 实例创建后是否允许改配置，要明确设计，不要靠调用习惯约定。

一般工程里我更推荐：

```cpp
Logger::configure(config);
Logger::getInstance().log("message");
```

这个接口比隐式初始化更清楚，也比运行时随意 `setConfig()` 更容易维护。
