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
    double bandwidthGiBs;
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

template <typename Copier>
BenchmarkResult MeasureMemcpy(uint64_t rounds, size_t bytes, bool streaming, const Copier &copy)
{
    constexpr double bytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double nanosecondsPerSecond = 1e9;
    const size_t copies = streaming ? static_cast<size_t>(rounds) : 1U;
    std::vector<uint8_t> source(copies * bytes * (streaming ? 2U : 1U), 0x5A);
    std::vector<uint8_t> destination(copies * bytes, 0);
    std::vector<uint64_t> latencies;
    latencies.reserve(static_cast<size_t>(rounds));

    const auto throughputBegin = Clock::now();
    for (uint64_t round = 0; round < rounds; ++round) {
        const size_t dstOffset = BufferOffset(round, bytes, streaming);
        const size_t srcOffset = streaming ? dstOffset * 2U : 0U;
        if (streaming && rounds - round > 4U) {
            __builtin_prefetch(source.data() + (static_cast<size_t>(round) + 4U) * bytes * 2U, 0, 1);
        }
        copy(destination.data() + dstOffset, source.data() + srcOffset);
        CompilerBarrier(destination.data() + dstOffset);
    }
    const auto throughputEnd = Clock::now();

    for (uint64_t round = 0; round < rounds; ++round) {
        const size_t dstOffset = BufferOffset(round, bytes, streaming);
        const size_t srcOffset = streaming ? dstOffset * 2U : 0U;
        if (streaming && rounds - round > 4U) {
            __builtin_prefetch(source.data() + (static_cast<size_t>(round) + 4U) * bytes * 2U, 0, 1);
        }
        const auto begin = Clock::now();
        copy(destination.data() + dstOffset, source.data() + srcOffset);
        const auto end = Clock::now();
        CompilerBarrier(destination.data() + dstOffset);
        latencies.push_back(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
    }
    const double elapsedNs = std::chrono::duration<double, std::nano>(throughputEnd - throughputBegin).count();
    const double bandwidth = static_cast<double>(rounds) * bytes * nanosecondsPerSecond / elapsedNs / bytesPerGiB;
    return {streaming ? "strided" : "hot", CalculateStats(std::move(latencies)), bandwidth};
}

std::vector<BenchmarkResult> RunBenchmarks(uint64_t rounds, size_t bytes)
{
    if (bytes == 656U) {
        const FixedCopier<656> copy{};
        return {MeasureMemcpy(rounds, bytes, false, copy), MeasureMemcpy(rounds, bytes, true, copy)};
    }
    const DynamicCopier copy{bytes};
    return {MeasureMemcpy(rounds, bytes, false, copy), MeasureMemcpy(rounds, bytes, true, copy)};
}

void PrintStats(uint64_t rounds, size_t bytes, const std::vector<BenchmarkResult> &results)
{
    constexpr int columnWidth = 14;
    constexpr int columnCount = 9;
    const auto printSeparator = [=]() {
        for (int column = 0; column < columnCount; ++column) {
            std::cout << '+' << std::string(columnWidth, '-');
        }
        std::cout << "+\n";
    };

    std::cout << "memcpy benchmark: rounds=" << rounds << ", bytes/copy=" << bytes << '\n';
    printSeparator();
    std::cout << '|' << std::setw(columnWidth) << "mode " << '|' << std::setw(columnWidth) << "rounds " << '|'
              << std::setw(columnWidth) << "bytes/copy " << '|' << std::setw(columnWidth) << "average(ns) " << '|'
              << std::setw(columnWidth) << "min(ns) " << '|'
              << std::setw(columnWidth) << "max(ns) " << '|' << std::setw(columnWidth) << "P95(ns) " << '|'
              << std::setw(columnWidth) << "P99(ns) " << '|' << std::setw(columnWidth) << "GiB/s " << "|\n";
    printSeparator();
    for (const auto &result : results) {
        const auto &stats = result.latency;
        std::cout << '|' << std::setw(columnWidth) << result.mode << '|' << std::setw(columnWidth) << rounds << '|'
                  << std::setw(columnWidth) << bytes << '|' << std::fixed << std::setprecision(1)
                  << std::setw(columnWidth) << stats.averageNs << '|' << std::setw(columnWidth) << stats.minNs << '|'
                  << std::setw(columnWidth) << stats.maxNs << '|' << std::setw(columnWidth) << stats.p95Ns << '|'
                  << std::setw(columnWidth) << stats.p99Ns << '|' << std::setprecision(2) << std::setw(columnWidth)
                  << result.bandwidthGiBs << "|\n";
    }
    printSeparator();
}
} // namespace

int main(int argc, char **argv)
{
    uint64_t rounds = 0;
    uint64_t bytes = 0;
    if (argc != 3 || !ParsePositiveInteger(argv[1], rounds) || !ParsePositiveInteger(argv[2], bytes) ||
        rounds > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        std::cerr << "Usage: " << argv[0] << " <rounds> <bytes>\n";
        return 1;
    }

    const size_t copyBytes = static_cast<size_t>(bytes);
    if (rounds > std::numeric_limits<size_t>::max() / copyBytes / 2U) {
        std::cerr << "Buffer size overflow: rounds=" << rounds << ", bytes=" << bytes << '\n';
        return 1;
    }
    const auto results = RunBenchmarks(rounds, copyBytes);
    PrintStats(rounds, copyBytes, results);
    return 0;
}
