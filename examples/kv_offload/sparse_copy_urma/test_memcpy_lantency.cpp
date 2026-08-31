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

std::vector<uint64_t> MeasureMemcpy(uint64_t rounds, size_t bytes)
{
    std::vector<uint8_t> source(bytes, 0x5A);
    std::vector<uint8_t> destination(bytes, 0);
    std::vector<uint64_t> latencies;
    latencies.reserve(static_cast<size_t>(rounds));

    for (uint64_t round = 0; round < rounds; ++round) {
        const auto begin = Clock::now();
        std::memcpy(destination.data(), source.data(), bytes);
        const auto end = Clock::now();
        CompilerBarrier(destination.data());
        latencies.push_back(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
    }
    return latencies;
}

void PrintStats(uint64_t rounds, size_t bytes, const LatencyStats &stats)
{
    constexpr int columnWidth = 14;
    constexpr int columnCount = 7;
    const auto printSeparator = [=]() {
        for (int column = 0; column < columnCount; ++column) {
            std::cout << '+' << std::string(columnWidth, '-');
        }
        std::cout << "+\n";
    };

    std::cout << "memcpy latency statistics\n";
    printSeparator();
    std::cout << '|' << std::setw(columnWidth) << "rounds " << '|' << std::setw(columnWidth) << "bytes/copy " << '|'
              << std::setw(columnWidth) << "average(ns) " << '|' << std::setw(columnWidth) << "min(ns) " << '|'
              << std::setw(columnWidth) << "max(ns) " << '|' << std::setw(columnWidth) << "P95(ns) " << '|'
              << std::setw(columnWidth) << "P99(ns) " << "|\n";
    printSeparator();
    std::cout << '|' << std::setw(columnWidth) << rounds << '|' << std::setw(columnWidth) << bytes << '|' << std::fixed
              << std::setprecision(1) << std::setw(columnWidth) << stats.averageNs << '|' << std::setw(columnWidth)
              << stats.minNs << '|' << std::setw(columnWidth) << stats.maxNs << '|' << std::setw(columnWidth)
              << stats.p95Ns << '|' << std::setw(columnWidth) << stats.p99Ns << "|\n";
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

    const auto latencies = MeasureMemcpy(rounds, static_cast<size_t>(bytes));
    PrintStats(rounds, static_cast<size_t>(bytes), CalculateStats(latencies));
    return 0;
}
