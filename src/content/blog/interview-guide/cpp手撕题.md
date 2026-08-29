# C++ 手撕题

## 线程池

面试手撕时，先把问题收敛到一个**固定大小、共享任务队列**的线程池。重点不是复刻工业级执行器，而是说明并实现：任务如何安全交接、线程如何阻塞与退出、以及调用方如何拿到结果。

### 本版本需要满足的要求

- **固定数量的工作线程**：构造线程池时创建 $W$ 个工作线程；每个线程不断从同一个任务队列中取任务执行。
- **生产者—消费者模型**：任意调用方都能通过 `submit` 生产任务；工作线程是消费者。队列为空时消费者必须阻塞等待，不能忙等占用 CPU。
- **异步结果**：`submit` 接受任意可调用对象及其参数，返回 `std::future`。返回值由 `future.get()` 取得；任务内部的异常也应由 `future.get()` 重新抛出。
- **线程安全**：多个生产者可以并发提交任务，多个消费者可以并发取任务；任务入队、出队和关闭状态都要受互斥锁保护，条件变量只负责等待和唤醒。
- **明确的停止语义**：调用 `shutdown` 或析构线程池后，不再接受新任务；已入队的任务仍然执行完，随后所有工作线程退出并被 `std::jthread` 自动 join。
- **现代 C++ 接口**：使用 `std::jthread` 管理线程、`std::invoke` 调用任务，并通过 lambda 捕获任务和参数，不使用 `std::bind`。

### 复杂度与边界

设当前排队任务数为 $N$，工作线程数为 $W$。

| 操作 | 复杂度 | 说明 |
| --- | --- | --- |
| `submit` 入队 | 均摊 $O(1)$ | `std::queue` 队尾插入，竞争锁的等待时间不计入算法复杂度。 |
| 工作线程取任务 | 均摊 $O(1)$ | 队首取出；队列为空时通过条件变量睡眠，不消耗 CPU 自旋。 |
| `shutdown` | $O(W)$ | 关闭队列并唤醒所有工作线程；析构时等待各线程结束。 |
| 额外空间 | $O(N + W)$ | 队列保存 $N$ 个待执行任务，线程对象保存 $W$ 个工作线程。 |

这个版本有意不处理下列工业化需求：有界队列与背压、优先级或延迟任务、任务取消、动态扩缩容、工作窃取、线程亲和性、异常监控与统计指标。这些需求会改变队列接口或调度模型，应在明确需求后单独设计。

### 实现

下面是一个基于阻塞队列的最小实现：`submit` 将任务放入队列，多个工作线程作为消费者取出并执行；任务结果和异常均由 `std::future` 获取。

```cpp
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @brief 线程安全的阻塞队列，负责在线程池中传递待执行任务。
 *
 * 队列关闭后不再接受新任务；消费者会先取完已有任务，再得到空结果并退出。
 *
 * @tparam T 队列中保存的元素类型。
 */
template <typename T>
class BlockingQueue {
 public:
  /**
   * @brief 向队尾添加一个元素，并唤醒一个等待中的消费者。
   *
   * @param value 要写入队列的元素；所有权会转移到队列中。
   * @throws std::runtime_error 队列已关闭时抛出。
   */
  void push(T value) {
    {
      std::lock_guard lock(mMutex);
      if (mClosed) {
        throw std::runtime_error("线程池已经停止，不能提交新任务");
      }
      mQueue.push(std::move(value));
    }
    mNotEmpty.notify_one();
  }

  /**
   * @brief 阻塞等待并取出一个元素。
   *
   * @return 取到的元素；当队列关闭且所有元素已取完时返回 `std::nullopt`。
   */
  [[nodiscard]] std::optional<T> pop() {
    std::unique_lock lock(mMutex);
    mNotEmpty.wait(lock, [this] { return mClosed || !mQueue.empty(); });

    if (mQueue.empty()) {
      return std::nullopt;
    }

    T value = std::move(mQueue.front());
    mQueue.pop();
    return value;
  }

  /**
   * @brief 关闭队列并唤醒所有消费者。
   *
   * 已入队的元素仍可被取出；之后的 `push` 会失败。
   */
  void close() {
    {
      std::lock_guard lock(mMutex);
      mClosed = true;
    }
    mNotEmpty.notify_all();
  }

 private:
  std::mutex mMutex;
  std::condition_variable mNotEmpty;
  std::queue<T> mQueue;
  bool mClosed = false;
};

/**
 * @brief 基于阻塞队列的固定大小线程池。
 *
 * `std::jthread` 在析构时自动 join；线程池析构前会关闭任务队列，
 * 因此已提交的任务会被工作线程处理完毕，再安全退出。
 */
class ThreadPool {
 public:
  using Task = std::function<void()>;

  /**
   * @brief 创建并启动指定数量的工作线程。
   *
   * @param worker_count 工作线程数量；传入 0 时使用一个工作线程。
   */
  explicit ThreadPool(std::size_t worker_count = std::thread::hardware_concurrency()) {
    if (worker_count == 0) {
      worker_count = 1;
    }

    mWorkers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
      mWorkers.emplace_back([this](std::stop_token stop_token) {
        workerLoop(stop_token);
      });
    }
  }

  /**
   * @brief 关闭任务队列，并等待工作线程结束。
   */
  ~ThreadPool() { shutdown(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  /**
   * @brief 提交一个可调用对象，并返回其异步执行结果。
   *
   * 函数对象和参数均按值保存到任务中，可安全提交临时 lambda。
   * 若任务抛出异常，异常会在调用 `future.get()` 时重新抛出。
   *
   * @tparam F 可调用对象类型。
   * @tparam Args 调用参数类型。
   * @param function 要异步执行的函数对象。
   * @param args 传给函数对象的参数。
   * @return 与函数返回类型对应的 `std::future`。
   * @throws std::runtime_error 线程池停止后提交任务时抛出。
   */
  template <typename F, typename... Args>
  auto submit(F&& function, Args&&... args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using Result = std::invoke_result_t<F, Args...>;

    auto promise = std::make_shared<std::promise<Result>>();
    std::future<Result> future = promise->get_future();
    Task task = [promise, function = std::forward<F>(function),
                 ... arguments = std::forward<Args>(args)]() mutable {
      try {
        if constexpr (std::is_void_v<Result>) {
          std::invoke(std::move(function), std::move(arguments)...);
          promise->set_value();
        } else {
          promise->set_value(
              std::invoke(std::move(function), std::move(arguments)...));
        }
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    };

    auto holder = std::make_shared<decltype(task)>(std::move(task));
    Task wrapper = [holder] { (*holder)(); };
    mTasks.push(std::move(wrapper));
    return future;
  }

  /**
   * @brief 停止接收新任务，并让工作线程处理完队列中的已有任务后退出。
   */
  void shutdown() { mTasks.close(); }

 private:
  /**
   * @brief 工作线程的消费循环。
   *
   * @param stop_token 由 `std::jthread` 注入的停止令牌；本实现以关闭队列作为退出信号，
   * 因此不会因为析构请求停止而丢弃已提交的任务。
   */
  void workerLoop([[maybe_unused]] std::stop_token stop_token) {
    while (auto task = mTasks.pop()) {
      (*task)();
    }
  }

  BlockingQueue<Task> mTasks;
  std::vector<std::jthread> mWorkers;
};

int main() {
  ThreadPool pool(4);

  auto sum = pool.submit([](int left, int right) { return left + right; }, 40, 2);
  auto message = pool.submit([] { return std::string("任务执行完成"); });

  std::cout << sum.get() << '\n';
  std::cout << message.get() << '\n';
}
```
