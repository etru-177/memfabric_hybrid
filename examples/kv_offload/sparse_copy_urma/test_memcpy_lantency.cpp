/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

namespace {
using Clock = std::chrono::steady_clock;

struct LatencyStats {
    double averageNs;
    uint64_t minNs;
    uint64_t maxNs;
    uint64_t p95Ns;
    uint64_t p99Ns;
};

struct BenchmarkResult {
    const char *mode;
    LatencyStats latency;
    double wallNsPerCopy;
    double bandwidthGiBs;
};

std::vector<size_t> MakeRandomOffsets(size_t copies, size_t stride, uint64_t seed);

class HugePageBuffer {
public:
    HugePageBuffer(size_t size, uint8_t value) : size_(size)
    {
        const long pageSize = sysconf(_SC_PAGESIZE);
        const auto alignment = static_cast<size_t>(pageSize);
        if (pageSize <= 0 || posix_memalign(reinterpret_cast<void **>(&data_), alignment, size_) != 0) {
            throw std::bad_alloc();
        }
#ifdef MADV_HUGEPAGE
        if (madvise(data_, size_, MADV_HUGEPAGE) != 0) {
            std::cerr << "warning: MADV_HUGEPAGE failed, size=" << size_ << '\n';
        }
#endif
        std::memset(data_, value, size_);
    }

    ~HugePageBuffer()
    {
        std::free(data_);
    }

    HugePageBuffer(const HugePageBuffer &) = delete;
    HugePageBuffer &operator=(const HugePageBuffer &) = delete;
    HugePageBuffer(HugePageBuffer &&other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0U;
    }
    HugePageBuffer &operator=(HugePageBuffer &&) = delete;

    uint8_t *Data() const
    {
        return data_;
    }

private:
    uint8_t *data_{nullptr};
    size_t size_{0U};
};

struct CopyBuffers {
    HugePageBuffer source;
    HugePageBuffer destination;
    std::vector<size_t> sourceOffsets;
    std::vector<size_t> destinationOffsets;

    CopyBuffers(size_t size, size_t copies, size_t stride, uint64_t seed)
        : source(size, 0x5A), destination(size, 0), sourceOffsets(MakeRandomOffsets(copies, stride, seed)),
          destinationOffsets(MakeRandomOffsets(copies, stride, seed ^ 0xD1B54A32D192ED03ULL))
    {}
};

template <size_t Bytes>
struct FixedCopier {
    __attribute__((always_inline)) inline void operator()(void *dst, const void *src) const
    {
        __builtin_memcpy(dst, src, Bytes);
    }
};

struct DynamicCopier {
    size_t bytes;

    void operator()(void *dst, const void *src) const
    {
        std::memcpy(dst, src, bytes);
    }
};

bool ParsePositiveInteger(const char *text, uint64_t &value)
{
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0) {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

uint64_t Percentile(const std::vector<uint64_t> &sorted, double percentile)
{
    const auto rank = static_cast<size_t>(std::ceil(percentile * static_cast<double>(sorted.size())));
    return sorted[std::min(rank - 1, sorted.size() - 1)];
}

LatencyStats CalculateStats(std::vector<uint64_t> latencies)
{
    std::sort(latencies.begin(), latencies.end());
    const long double total = std::accumulate(latencies.begin(), latencies.end(), static_cast<long double>(0));
    return {static_cast<double>(total / latencies.size()), latencies.front(), latencies.back(),
            Percentile(latencies, 0.95), Percentile(latencies, 0.99)};
}

void CompilerBarrier(const void *memory)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(memory) : "memory");
#else
    (void)memory;
#endif
}

inline void CpuRelax()
{
#if defined(__aarch64__)
    __asm__ __volatile__("yield" : : : "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" : : : "memory");
#else
    std::this_thread::yield();
#endif
}

std::vector<int> GetWorkerCpus(uint32_t threadCount)
{
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        return {};
    }
    std::vector<int> cpus;
    const int callerCpu = sched_getcpu();
    if (callerCpu >= 0 && CPU_ISSET(callerCpu, &affinity)) {
        cpus.push_back(callerCpu);
    }
    for (int cpu = 0; cpu < CPU_SETSIZE && cpus.size() < threadCount; ++cpu) {
        if (CPU_ISSET(cpu, &affinity) && cpu != callerCpu) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

void PinWorker(int cpu)
{
    if (cpu < 0) {
        return;
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
}

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

class DedicatedWorkers {
public:
    explicit DedicatedWorkers(uint32_t threadCount) : threadCount_(threadCount), done_(threadCount + 1U)
    {
        const auto cpus = GetWorkerCpus(threadCount_);
        if (!cpus.empty() && cpus.size() < threadCount_) {
            throw std::invalid_argument("thread count exceeds CPUs allowed by process affinity");
        }
        workers_.reserve(threadCount_);
        for (uint32_t index = 0U; index < threadCount_; ++index) {
            const int cpu = index < cpus.size() ? cpus[index] : -1;
            workers_.emplace_back(&DedicatedWorkers::WorkerLoop, this, index, cpu);
        }
    }

    ~DedicatedWorkers()
    {
        stopping_.store(true, std::memory_order_release);
        generation_.fetch_add(1U, std::memory_order_release);
        for (auto &worker : workers_) {
            worker.join();
        }
    }

    double Run(const std::function<void(uint32_t)> &work)
    {
        work_ = &work;
        done_.store(0U, std::memory_order_relaxed);
        const auto begin = Clock::now();
        generation_.fetch_add(1U, std::memory_order_release);
        while (done_.load(std::memory_order_acquire) != threadCount_ + 1U) {
            CpuRelax();
        }
        return std::chrono::duration<double, std::nano>(Clock::now() - begin).count();
    }

private:
    void WorkerLoop(uint32_t index, int cpu)
    {
        PinWorker(cpu);
        uint64_t observed = 0U;
        while (true) {
            auto generation = generation_.load(std::memory_order_acquire);
            while (generation == observed && !stopping_.load(std::memory_order_relaxed)) {
                CpuRelax();
                generation = generation_.load(std::memory_order_acquire);
            }
            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }
            observed = generation;
            (*work_)(index);
            const uint32_t finished = done_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
            if (finished == threadCount_) {
                done_.store(threadCount_ + 1U, std::memory_order_release);
            }
        }
    }

    uint32_t threadCount_;
    std::vector<std::thread> workers_;
    const std::function<void(uint32_t)> *work_{nullptr};
    std::atomic<uint64_t> generation_{0U};
    std::atomic<uint32_t> done_;
    std::atomic<bool> stopping_{false};
};

std::vector<size_t> MakeRandomOffsets(size_t copies, size_t stride, uint64_t seed)
{
    std::vector<size_t> offsets(copies);
    std::iota(offsets.begin(), offsets.end(), 0U);
    std::mt19937_64 generator(seed);
    std::shuffle(offsets.begin(), offsets.end(), generator);
    for (auto &offset : offsets) {
        offset *= stride;
    }
    return offsets;
}

std::vector<CopyBuffers> MakeBuffers(uint32_t threadCount, uint64_t rounds, size_t bytes)
{
    constexpr size_t cacheLineBytes = 64U;
    const size_t copies = static_cast<size_t>(rounds);
    const size_t stride = AlignUp(bytes, cacheLineBytes) + cacheLineBytes;
    std::vector<CopyBuffers> buffers;
    buffers.reserve(threadCount);
    for (uint32_t thread = 0U; thread < threadCount; ++thread) {
        const uint64_t seed = 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(thread) + 1U);
        buffers.emplace_back(copies * stride, copies, stride, seed);
    }
    return buffers;
}

template <typename Copier>
BenchmarkResult MeasureMemcpy(uint64_t rounds, size_t bytes, uint32_t threadCount, const Copier &copy)
{
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double nanosecondsPerSecond = 1e9;
    auto buffers = MakeBuffers(threadCount, rounds, bytes);
    DedicatedWorkers workers(threadCount);
    const auto throughputWork = [&](uint32_t thread) {
        auto &buffer = buffers[thread];
        for (uint64_t round = 0; round < rounds; ++round) {
            const size_t srcOffset = buffer.sourceOffsets[round];
            const size_t dstOffset = buffer.destinationOffsets[round];
            copy(buffer.destination.Data() + dstOffset, buffer.source.Data() + srcOffset);
            CompilerBarrier(buffer.destination.Data() + dstOffset);
        }
    };
    const double elapsedNs = workers.Run(throughputWork);

    std::vector<std::vector<uint64_t>> threadLatencies(threadCount);
    const auto latencyWork = [&](uint32_t thread) {
        auto &buffer = buffers[thread];
        auto &latencies = threadLatencies[thread];
        latencies.reserve(static_cast<size_t>(rounds));
        for (uint64_t round = 0; round < rounds; ++round) {
            const size_t srcOffset = buffer.sourceOffsets[round];
            const size_t dstOffset = buffer.destinationOffsets[round];
            const auto begin = Clock::now();
            copy(buffer.destination.Data() + dstOffset, buffer.source.Data() + srcOffset);
            const auto end = Clock::now();
            CompilerBarrier(buffer.destination.Data() + dstOffset);
            latencies.push_back(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
        }
    };
    (void)workers.Run(latencyWork);

    std::vector<uint64_t> latencies;
    const uint64_t totalCopies = rounds * threadCount;
    latencies.reserve(static_cast<size_t>(totalCopies));
    for (auto &values : threadLatencies) {
        latencies.insert(latencies.end(), values.begin(), values.end());
    }
    const double wallNsPerCopy = elapsedNs / static_cast<double>(totalCopies);
    const double bandwidth =
        static_cast<double>(totalCopies) * bytes * nanosecondsPerSecond / elapsedNs / bytesPerGiB;
    return {"random", CalculateStats(std::move(latencies)), wallNsPerCopy, bandwidth};
}

BenchmarkResult RunBenchmark(uint64_t rounds, size_t bytes, uint32_t threadCount)
{
    if (bytes == 656U) {
        const FixedCopier<656> copy{};
        return MeasureMemcpy(rounds, bytes, threadCount, copy);
    }
    const DynamicCopier copy{bytes};
    return MeasureMemcpy(rounds, bytes, threadCount, copy);
}

void PrintStats(uint64_t rounds, size_t bytes, uint32_t threadCount, const BenchmarkResult &result)
{
    constexpr int columnWidth = 14;
    constexpr int columnCount = 10;
    const auto printSeparator = [=]() {
        for (int column = 0; column < columnCount; ++column) {
            std::cout << '+' << std::string(columnWidth, '-');
        }
        std::cout << "+\n";
    };

    std::cout << "memcpy benchmark: rounds/thread=" << rounds << ", threads=" << threadCount
              << ", samples=" << rounds * threadCount << ", bytes/copy=" << bytes << '\n';
    printSeparator();
    std::cout << '|' << std::setw(columnWidth) << "mode " << '|' << std::setw(columnWidth) << "samples " << '|'
              << std::setw(columnWidth) << "bytes/copy " << '|' << std::setw(columnWidth) << "average(ns) " << '|'
              << std::setw(columnWidth) << "min(ns) " << '|'
              << std::setw(columnWidth) << "max(ns) " << '|' << std::setw(columnWidth) << "P95(ns) " << '|'
              << std::setw(columnWidth) << "P99(ns) " << '|' << std::setw(columnWidth) << "wall(ns/copy) " << '|'
              << std::setw(columnWidth) << "GiB/s " << "|\n";
    printSeparator();
    const auto &stats = result.latency;
    std::cout << '|' << std::setw(columnWidth) << result.mode << '|' << std::setw(columnWidth)
              << rounds * threadCount << '|' << std::setw(columnWidth) << bytes << '|' << std::fixed
              << std::setprecision(1) << std::setw(columnWidth) << stats.averageNs << '|' << std::setw(columnWidth)
              << stats.minNs << '|' << std::setw(columnWidth) << stats.maxNs << '|' << std::setw(columnWidth)
              << stats.p95Ns << '|' << std::setw(columnWidth) << stats.p99Ns << '|' << std::setw(columnWidth)
              << result.wallNsPerCopy << '|' << std::setprecision(2) << std::setw(columnWidth)
              << result.bandwidthGiBs << "|\n";
    printSeparator();
}
} // namespace

int main(int argc, char **argv)
{
    uint64_t rounds = 0;
    uint64_t bytes = 0;
    uint64_t threads = 1U;
    if ((argc != 3 && argc != 4) || !ParsePositiveInteger(argv[1], rounds) ||
        !ParsePositiveInteger(argv[2], bytes) || (argc == 4 && !ParsePositiveInteger(argv[3], threads)) ||
        rounds > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) || threads > 256U) {
        std::cerr << "Usage: " << argv[0] << " <rounds-per-thread> <bytes> [threads<=256]\n";
        return 1;
    }

    const size_t copyBytes = static_cast<size_t>(bytes);
    constexpr size_t cacheLineBytes = 64U;
    if (copyBytes > std::numeric_limits<size_t>::max() - 2U * cacheLineBytes ||
        rounds > std::numeric_limits<size_t>::max() / (AlignUp(copyBytes, cacheLineBytes) + cacheLineBytes) ||
        rounds > std::numeric_limits<uint64_t>::max() / threads) {
        std::cerr << "Buffer size overflow: rounds=" << rounds << ", bytes=" << bytes << ", threads=" << threads
                  << '\n';
        return 1;
    }
    const auto threadCount = static_cast<uint32_t>(threads);
    const auto result = RunBenchmark(rounds, copyBytes, threadCount);
    PrintStats(rounds, copyBytes, threadCount, result);
    return 0;
}
