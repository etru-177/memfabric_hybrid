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
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

struct CopyBuffers {
    std::vector<uint8_t> source;
    std::vector<uint8_t> destination;
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

size_t BufferOffset(uint64_t round, size_t bytes, bool streaming)
{
    return streaming ? static_cast<size_t>(round) * bytes : 0U;
}

template <typename Work>
double RunWorkers(uint32_t threadCount, const Work &work)
{
    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (uint32_t thread = 0U; thread < threadCount; ++thread) {
        workers.emplace_back([&, thread]() {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            work(thread);
        });
    }
    while (ready.load(std::memory_order_acquire) != threadCount) {
        std::this_thread::yield();
    }
    const auto begin = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }
    return std::chrono::duration<double, std::nano>(Clock::now() - begin).count();
}

std::vector<CopyBuffers> MakeBuffers(uint32_t threadCount, uint64_t rounds, size_t bytes, bool streaming)
{
    const size_t copies = streaming ? static_cast<size_t>(rounds) : 1U;
    const size_t sourceMultiplier = streaming ? 2U : 1U;
    std::vector<CopyBuffers> buffers;
    buffers.reserve(threadCount);
    for (uint32_t thread = 0U; thread < threadCount; ++thread) {
        buffers.push_back({std::vector<uint8_t>(copies * bytes * sourceMultiplier, 0x5A),
                           std::vector<uint8_t>(copies * bytes, 0)});
    }
    return buffers;
}

template <typename Copier>
BenchmarkResult MeasureMemcpy(
    uint64_t rounds, size_t bytes, uint32_t threadCount, bool streaming, const Copier &copy)
{
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double nanosecondsPerSecond = 1e9;
    auto buffers = MakeBuffers(threadCount, rounds, bytes, streaming);
    const auto throughputWork = [&](uint32_t thread) {
        auto &buffer = buffers[thread];
        for (uint64_t round = 0; round < rounds; ++round) {
            const size_t dstOffset = BufferOffset(round, bytes, streaming);
            const size_t srcOffset = streaming ? dstOffset * 2U : 0U;
            if (streaming && rounds - round > 4U) {
                __builtin_prefetch(buffer.source.data() + (static_cast<size_t>(round) + 4U) * bytes * 2U, 0, 1);
            }
            copy(buffer.destination.data() + dstOffset, buffer.source.data() + srcOffset);
            CompilerBarrier(buffer.destination.data() + dstOffset);
        }
    };
    const double elapsedNs = RunWorkers(threadCount, throughputWork);

    std::vector<std::vector<uint64_t>> threadLatencies(threadCount);
    const auto latencyWork = [&](uint32_t thread) {
        auto &buffer = buffers[thread];
        auto &latencies = threadLatencies[thread];
        latencies.reserve(static_cast<size_t>(rounds));
        for (uint64_t round = 0; round < rounds; ++round) {
            const size_t dstOffset = BufferOffset(round, bytes, streaming);
            const size_t srcOffset = streaming ? dstOffset * 2U : 0U;
            if (streaming && rounds - round > 4U) {
                __builtin_prefetch(buffer.source.data() + (static_cast<size_t>(round) + 4U) * bytes * 2U, 0, 1);
            }
            const auto begin = Clock::now();
            copy(buffer.destination.data() + dstOffset, buffer.source.data() + srcOffset);
            const auto end = Clock::now();
            CompilerBarrier(buffer.destination.data() + dstOffset);
            latencies.push_back(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
        }
    };
    (void)RunWorkers(threadCount, latencyWork);

    std::vector<uint64_t> latencies;
    const uint64_t totalCopies = rounds * threadCount;
    latencies.reserve(static_cast<size_t>(totalCopies));
    for (auto &values : threadLatencies) {
        latencies.insert(latencies.end(), values.begin(), values.end());
    }
    const double wallNsPerCopy = elapsedNs / static_cast<double>(totalCopies);
    const double bandwidth =
        static_cast<double>(totalCopies) * bytes * nanosecondsPerSecond / elapsedNs / bytesPerGiB;
    return {streaming ? "strided" : "hot", CalculateStats(std::move(latencies)), wallNsPerCopy, bandwidth};
}

std::vector<BenchmarkResult> RunBenchmarks(uint64_t rounds, size_t bytes, uint32_t threadCount)
{
    if (bytes == 656U) {
        const FixedCopier<656> copy{};
        return {MeasureMemcpy(rounds, bytes, threadCount, false, copy),
                MeasureMemcpy(rounds, bytes, threadCount, true, copy)};
    }
    const DynamicCopier copy{bytes};
    return {MeasureMemcpy(rounds, bytes, threadCount, false, copy),
            MeasureMemcpy(rounds, bytes, threadCount, true, copy)};
}

void PrintStats(uint64_t rounds, size_t bytes, uint32_t threadCount, const std::vector<BenchmarkResult> &results)
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
    for (const auto &result : results) {
        const auto &stats = result.latency;
        std::cout << '|' << std::setw(columnWidth) << result.mode << '|' << std::setw(columnWidth)
                  << rounds * threadCount << '|'
                  << std::setw(columnWidth) << bytes << '|' << std::fixed << std::setprecision(1)
                  << std::setw(columnWidth) << stats.averageNs << '|' << std::setw(columnWidth) << stats.minNs << '|'
                  << std::setw(columnWidth) << stats.maxNs << '|' << std::setw(columnWidth) << stats.p95Ns << '|'
                  << std::setw(columnWidth) << stats.p99Ns << '|' << std::setw(columnWidth) << result.wallNsPerCopy
                  << '|' << std::setprecision(2) << std::setw(columnWidth) << result.bandwidthGiBs << "|\n";
    }
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
    if (rounds > std::numeric_limits<size_t>::max() / copyBytes / 2U ||
        rounds > std::numeric_limits<uint64_t>::max() / threads) {
        std::cerr << "Buffer size overflow: rounds=" << rounds << ", bytes=" << bytes << ", threads=" << threads
                  << '\n';
        return 1;
    }
    const auto threadCount = static_cast<uint32_t>(threads);
    const auto results = RunBenchmarks(rounds, copyBytes, threadCount);
    PrintStats(rounds, copyBytes, threadCount, results);
    return 0;
}
