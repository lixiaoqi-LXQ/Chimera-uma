#include "cpu_task_pool.hpp"
#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    constexpr size_t producers = 4, count = 257;
    std::array<std::atomic<int>, producers * count> visits{};
    for (auto& visit : visits) std::atomic_init(&visit, 0);
    {
        Chimera::CpuTaskPool pool(4);
        std::array<std::thread, producers> submitters;
        for (size_t p = 0; p < producers; ++p) {
            submitters[p] = std::thread([&, p] {
                pool.submit(count, 7, [&, p](size_t i) { ++visits[p * count + i]; })->wait();
            });
        }
        for (auto& submitter : submitters) submitter.join();
        for (const auto& visit : visits) require(visit.load() == 1, "Concurrent submission lost or repeated work");

        std::atomic<int> finished{0};
        auto failing = pool.submit(31, 1, [&](size_t i) {
            ++finished;
            if (i == 9) throw std::runtime_error("expected task failure");
        });
        bool reported = false;
        try { failing->wait(); } catch (const std::runtime_error&) { reported = true; }
        require(reported && finished.load() == 31, "Failure did not drain the group before propagation");
        pool.submit(5, 1, [&](size_t) { ++finished; })->wait();
        require(finished.load() == 36, "A failed group damaged subsequent work");
        pool.submit(0, 1, {})->wait();
    }
    std::atomic<int> drained{0};
    {
        Chimera::CpuTaskPool pool(4);
        (void)pool.submit(4096, 1, [&](size_t) { ++drained; });
        // The caller releases its handle. Owner shutdown must still finish all
        // callbacks before caller-owned/corpus storage can be released.
    }
    require(drained.load() == 4096, "Pool shutdown abandoned borrowed-data callbacks");
    std::cout << "[CPU_POOL_VERIFY] concurrent_groups=passed error_drain=passed reuse=passed shutdown_drain=passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
