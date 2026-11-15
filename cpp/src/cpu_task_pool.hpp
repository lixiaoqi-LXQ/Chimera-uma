#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Chimera {

// One corpus owner supplies this pool to all query lanes. Workers use GB10's
// efficiency cores by default; an explicit CPU list can use other idle cores.
class CpuTaskPool {
 public:
    class Group {
     public:
        void wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return remaining_ == 0; });
            if (error_) std::rethrow_exception(error_);
        }
     private:
        friend class CpuTaskPool;
        void finish(std::exception_ptr error) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error && !error_) error_ = error;
            if (--remaining_ == 0) done_.notify_all();
        }
        std::mutex mutex_;
        std::condition_variable done_;
        size_t remaining_ = 0;
        std::exception_ptr error_;
    };

    // Strict decimal CSV keeps the benchmark JSON value safe and rejects
    // ranges, empty tokens, trailing commas and ambiguous partial numbers.
    static std::vector<int> parse_cpu_list(const std::string& csv) {
        std::vector<int> cpus;
        if (csv.empty()) return cpus;
        size_t first = 0;
        while (first < csv.size()) {
            const size_t comma = csv.find(',', first);
            const size_t last = comma == std::string::npos ? csv.size() : comma;
            if (first == last) throw std::invalid_argument("Empty CPU-list element");
            int cpu = 0;
            for (size_t j = first; j < last; ++j) {
                if (csv[j] < '0' || csv[j] > '9')
                    throw std::invalid_argument("CPU list requires decimal IDs separated by commas");
                cpu = cpu * 10 + (csv[j] - '0');
                if (cpu > 19) throw std::invalid_argument("CPU IDs must be in 0..19");
            }
            cpus.push_back(cpu);
            if (comma == std::string::npos) break;
            first = comma + 1;
            if (first == csv.size()) throw std::invalid_argument("Trailing CPU-list comma");
        }
        return cpus;
    }

    static void validate_cpu_selection(size_t threads, const std::vector<int>& cpus) {
        if (threads == 0 || threads > 16)
            throw std::invalid_argument("CPU collaboration requires 1..16 workers");
        if (cpus.empty()) {
            if (threads > 10)
                throw std::invalid_argument("More than 10 workers requires explicit uma_cpu_pool_cpus");
            return;
        }
        if (cpus.size() != threads)
            throw std::invalid_argument("CPU-list length must equal uma_cpu_threads");
        bool seen[20] = {};
        for (int cpu : cpus) {
            if (cpu < 0 || cpu > 19 || seen[cpu])
                throw std::invalid_argument("CPU IDs must be unique and in 0..19");
            seen[cpu] = true;
        }
    }

    explicit CpuTaskPool(size_t threads, std::vector<int> cpus = {}) {
        static constexpr int efficiency_cpus[] = {0, 10, 1, 11, 2, 12, 3, 13, 4, 14};
        validate_cpu_selection(threads, cpus);
        if (cpus.empty()) cpus.assign(efficiency_cpus, efficiency_cpus + threads);
        try {
            for (size_t i = 0; i < threads; ++i) {
                workers_.emplace_back([this, cpu = cpus[i], i] {
                    std::exception_ptr error;
                    try {
                        cpu_set_t mask;
                        CPU_ZERO(&mask);
                        CPU_SET(cpu, &mask);
                        const int status = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
                        if (status) throw std::runtime_error(
                            "Cannot bind CPU collaboration worker to CPU " + std::to_string(cpu)
                            + ": pthread error " + std::to_string(status));
                        char name[16];
                        std::snprintf(name, sizeof(name), "Chimera_cpu_%zu", i);
                        (void)pthread_setname_np(pthread_self(), name);
                    } catch (...) { error = std::current_exception(); }
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (error && !startup_error_) startup_error_ = error;
                        ++started_;
                        ready_.notify_all();
                    }
                    if (error) return;
                    worker_loop();
                });
            }
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return started_ == threads; });
            if (startup_error_) std::rethrow_exception(startup_error_);
        } catch (...) {
            stop();
            throw;
        }
    }

    ~CpuTaskPool() { stop(); }
    CpuTaskPool(const CpuTaskPool&) = delete;
    CpuTaskPool& operator=(const CpuTaskPool&) = delete;

    // Build the entire submission before publishing it. A failing allocation
    // cannot leave tasks borrowing caller storage without a returned Group.
    std::shared_ptr<Group> submit(size_t count, size_t grain,
                                 std::function<void(size_t)> function) {
        if (!grain || (count && !function))
            throw std::invalid_argument("Invalid CPU task group");
        auto group = std::make_shared<Group>();
        if (!count) return group;
        auto shared_function = std::make_shared<std::function<void(size_t)>>(std::move(function));
        std::list<Task> pending;
        for (size_t begin = 0; begin < count;) {
            const size_t end = begin + std::min(grain, count - begin);
            pending.push_back(Task{group, shared_function, begin, end});
            begin = end;
        }
        group->remaining_ = pending.size();
        const size_t wake = std::min(pending.size(), workers_.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::logic_error("CPU task pool is stopping");
            tasks_.splice(tasks_.end(), pending);
        }
        for (size_t i = 0; i < wake; ++i) ready_.notify_one();
        return group;
    }

 private:
    struct Task {
        std::shared_ptr<Group> group;
        std::shared_ptr<std::function<void(size_t)>> function;
        size_t begin, end;
    };
    void worker_loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            std::exception_ptr error;
            try {
                for (size_t i = task.begin; i < task.end; ++i) (*task.function)(i);
            } catch (...) { error = std::current_exception(); }
            task.group->finish(error);
        }
    }
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::list<Task> tasks_;
    std::vector<std::thread> workers_;
    size_t started_ = 0;
    bool stopping_ = false;
    std::exception_ptr startup_error_;
};

}  // namespace Chimera
