/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "acc_offload.h"
#include "operators/aicpu/hybm_aggregate_urma_demo.h"

namespace py = pybind11;

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t PREFETCH_DISTANCE = 4;

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

std::vector<int> GetGatherCpus(uint32_t threadCount)
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

void PinGatherWorker(int cpu)
{
    if (cpu < 0) {
        return;
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
}

template <size_t Bytes>
__attribute__((always_inline)) inline void CopyFixed(uint8_t *__restrict dst, const uint8_t *__restrict src)
{
    __builtin_memcpy(dst, src, Bytes);
}

template <size_t Bytes>
void GatherFixed(uint8_t *__restrict dst, const uint8_t *__restrict src, uint64_t srcStride, uint32_t segmentCount)
{
    for (uint32_t index = 0; index < segmentCount; ++index) {
        if (segmentCount - index > PREFETCH_DISTANCE) {
            __builtin_prefetch(src + PREFETCH_DISTANCE * srcStride, 0, 1);
        }
        CopyFixed<Bytes>(dst, src);
        dst += Bytes;
        src += srcStride;
    }
}

void GatherDynamic(uint8_t *__restrict dst, const uint8_t *__restrict src, uint64_t srcStride, uint32_t segmentCount,
                   uint32_t segmentBytes)
{
    for (uint32_t index = 0; index < segmentCount; ++index) {
        std::memcpy(dst, src, segmentBytes);
        dst += segmentBytes;
        src += srcStride;
    }
}

void GatherSegments(uint8_t *dst, const uint8_t *src, const HybmAggregateUrmaDemoRequest &request)
{
    if (request.segmentBytes == 656U) {
        GatherFixed<656>(dst, src, request.srcStride, request.segmentCount);
    } else if (request.segmentBytes == 576U) {
        GatherFixed<576>(dst, src, request.srcStride, request.segmentCount);
    } else if (request.segmentBytes == 1152U) {
        GatherFixed<1152>(dst, src, request.srcStride, request.segmentCount);
    } else {
        GatherDynamic(dst, src, request.srcStride, request.segmentCount, request.segmentBytes);
    }
}

struct GatherTask {
    uint8_t *dst;
    const uint8_t *src;
    HybmAggregateUrmaDemoRequest request;
    uint32_t threadCount;
    const uint32_t *sourceIndices;
    const uint64_t *sourceAddresses;
    uint32_t sourcePoolSegments;
    uint64_t sourceOrdinal;
    int64_t gvaToVaOffset;
};

template <size_t Bytes>
void GatherIndexedFixed(uint8_t *dst, const uint8_t *src, const GatherTask &task, uint32_t begin, uint32_t end)
{
    for (uint32_t index = begin; index < end; ++index) {
        const uint64_t ordinal = (task.sourceOrdinal + index) % task.sourcePoolSegments;
        const uint32_t sourceIndex = task.sourceIndices[ordinal];
        if (end - index > PREFETCH_DISTANCE) {
            const uint64_t nextOrdinal = (task.sourceOrdinal + index + PREFETCH_DISTANCE) % task.sourcePoolSegments;
            __builtin_prefetch(src + static_cast<uint64_t>(task.sourceIndices[nextOrdinal]) * Bytes, 0, 1);
        }
        CopyFixed<Bytes>(dst + static_cast<uint64_t>(index) * Bytes,
                         src + static_cast<uint64_t>(sourceIndex) * Bytes);
    }
}

void GatherIndexed(const GatherTask &task, uint32_t begin, uint32_t end)
{
    if (task.request.segmentBytes == 656U) {
        GatherIndexedFixed<656>(task.dst, task.src, task, begin, end);
    } else if (task.request.segmentBytes == 576U) {
        GatherIndexedFixed<576>(task.dst, task.src, task, begin, end);
    } else if (task.request.segmentBytes == 1152U) {
        GatherIndexedFixed<1152>(task.dst, task.src, task, begin, end);
    } else {
        for (uint32_t index = begin; index < end; ++index) {
            const uint64_t ordinal = (task.sourceOrdinal + index) % task.sourcePoolSegments;
            const uint32_t sourceIndex = task.sourceIndices[ordinal];
            std::memcpy(task.dst + static_cast<uint64_t>(index) * task.request.segmentBytes,
                        task.src + static_cast<uint64_t>(sourceIndex) * task.request.segmentBytes,
                        task.request.segmentBytes);
        }
    }
}

template <size_t Bytes>
void GatherAddressesFixed(const GatherTask &task, uint32_t begin, uint32_t end)
{
    for (uint32_t index = begin; index < end; ++index) {
        const auto source = reinterpret_cast<const uint8_t *>(task.sourceAddresses[index] + task.gvaToVaOffset);
        if (end - index > PREFETCH_DISTANCE) {
            const auto next = reinterpret_cast<const uint8_t *>(
                task.sourceAddresses[index + PREFETCH_DISTANCE] + task.gvaToVaOffset);
            __builtin_prefetch(next, 0, 1);
        }
        CopyFixed<Bytes>(task.dst + static_cast<uint64_t>(index) * Bytes, source);
    }
}

void GatherAddresses(const GatherTask &task, uint32_t begin, uint32_t end)
{
    if (task.request.segmentBytes == 656U) {
        GatherAddressesFixed<656>(task, begin, end);
    } else if (task.request.segmentBytes == 576U) {
        GatherAddressesFixed<576>(task, begin, end);
    } else if (task.request.segmentBytes == 1152U) {
        GatherAddressesFixed<1152>(task, begin, end);
    } else {
        for (uint32_t index = begin; index < end; ++index) {
            const auto source = reinterpret_cast<const void *>(task.sourceAddresses[index] + task.gvaToVaOffset);
            std::memcpy(task.dst + static_cast<uint64_t>(index) * task.request.segmentBytes, source,
                        task.request.segmentBytes);
        }
    }
}

void GatherPartition(const GatherTask &task, uint32_t threadIndex)
{
    const uint32_t segmentsPerCacheLine = 64U / std::gcd(64U, task.request.segmentBytes);
    const auto partitionBoundary = [&task, segmentsPerCacheLine](uint32_t index) {
        if (index == task.threadCount) {
            return task.request.segmentCount;
        }
        const uint32_t boundary = static_cast<uint32_t>(
            static_cast<uint64_t>(task.request.segmentCount) * index / task.threadCount);
        return boundary / segmentsPerCacheLine * segmentsPerCacheLine;
    };
    const uint32_t begin = partitionBoundary(threadIndex);
    const uint32_t end = partitionBoundary(threadIndex + 1U);
    if (begin == end) {
        return;
    }
    if (task.sourceAddresses != nullptr) {
        GatherAddresses(task, begin, end);
        return;
    }
    if (task.sourceIndices != nullptr) {
        GatherIndexed(task, begin, end);
        return;
    }
    auto request = task.request;
    request.segmentCount = end - begin;
    GatherSegments(task.dst + static_cast<uint64_t>(begin) * request.segmentBytes,
                   task.src + static_cast<uint64_t>(begin) * request.srcStride, request);
}

class GatherThreadPool {
public:
    GatherThreadPool(uint32_t threadCount, const std::vector<int> &configuredCpus)
        : threadCount_(threadCount), done_(threadCount + 1U)
    {
        const auto cpus = configuredCpus.empty() ? GetGatherCpus(threadCount_) : configuredCpus;
        if (!cpus.empty() && cpus.size() < threadCount_) {
            throw std::invalid_argument("gatherThreads exceeds configured gather CPU count");
        }
        workers_.reserve(threadCount_);
        for (uint32_t index = 0U; index < threadCount_; ++index) {
            const int cpu = index < cpus.size() ? cpus[index] : -1;
            workers_.emplace_back(&GatherThreadPool::WorkerLoop, this, index, cpu);
        }
    }

    ~GatherThreadPool()
    {
        stopping_.store(true, std::memory_order_release);
        generation_.fetch_add(1U, std::memory_order_release);
        for (auto &worker : workers_) {
            worker.join();
        }
    }

    uint32_t ThreadCount() const
    {
        return threadCount_;
    }

    void Run(uint8_t *dst, const uint8_t *src, const HybmAggregateUrmaDemoRequest &request,
             const uint32_t *sourceIndices = nullptr, uint32_t sourcePoolSegments = 0U, uint64_t sourceOrdinal = 0U,
             const uint64_t *sourceAddresses = nullptr, int64_t gvaToVaOffset = 0)
    {
        while (done_.load(std::memory_order_acquire) != threadCount_ + 1U &&
               generation_.load(std::memory_order_relaxed) != 0U) {
            CpuRelax();
        }
        task_ = {dst, src, request, threadCount_, sourceIndices, sourceAddresses, sourcePoolSegments, sourceOrdinal,
                 gvaToVaOffset};
        done_.store(0U, std::memory_order_relaxed);
        generation_.fetch_add(1U, std::memory_order_release);
        while (done_.load(std::memory_order_acquire) != threadCount_ + 1U) {
            CpuRelax();
        }
    }

private:
    void WorkerLoop(uint32_t threadIndex, int cpu)
    {
        PinGatherWorker(cpu);
        uint64_t observedGeneration = 0;
        while (true) {
            auto generation = generation_.load(std::memory_order_acquire);
            while (generation == observedGeneration && !stopping_.load(std::memory_order_relaxed)) {
                CpuRelax();
                generation = generation_.load(std::memory_order_acquire);
            }
            if (stopping_.load(std::memory_order_acquire)) {
                return;
            }
            observedGeneration = generation;
            GatherPartition(task_, threadIndex);
            const uint32_t finished = done_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
            if (finished == threadCount_) {
                done_.store(threadCount_ + 1U, std::memory_order_release);
            }
        }
    }

    uint32_t threadCount_;
    std::vector<std::thread> workers_;
    GatherTask task_{};
    std::atomic<uint64_t> generation_{0};
    std::atomic<uint32_t> done_;
    std::atomic<bool> stopping_{false};
};

GatherThreadPool &GetGatherThreadPool(uint32_t threadCount, const std::vector<int> &configuredCpus)
{
    static std::mutex poolMutex;
    static std::unique_ptr<GatherThreadPool> pool;
    std::lock_guard<std::mutex> lock(poolMutex);
    if (pool == nullptr || pool->ThreadCount() != threadCount) {
        pool = std::make_unique<GatherThreadPool>(threadCount, configuredCpus);
    }
    return *pool;
}

void WaitForDoorbell(const HybmAggregateUrmaDemoMessage *message, uint64_t expectedDoorbell)
{
    while (true) {
        const uint64_t doorbell = __atomic_load_n(&message->doorbell, __ATOMIC_ACQUIRE);
        if (doorbell == expectedDoorbell) {
            return;
        }
        CpuRelax();
    }
}
} // namespace

py::tuple AggregateWaitDemo(uint64_t mailbox, uint64_t expectedDoorbell)
{
    auto *message = reinterpret_cast<HybmAggregateUrmaDemoMessage *>(mailbox);
    HybmAggregateUrmaDemoRequest request{};
    uint64_t waitNs = 0;
    {
        py::gil_scoped_release release;
        const auto begin = Clock::now();
        WaitForDoorbell(message, expectedDoorbell);
        const auto end = Clock::now();
        request = message->request;
        waitNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    return py::make_tuple(request.dstNewGva, request.readyGva, request.totalBytes, request.srcStride,
                          request.segmentCount, request.segmentBytes, waitNs);
}

uint64_t AggregateGatherRangeDemo(uint64_t source, uint64_t aggregate, uint64_t srcStride, uint32_t segmentCount,
                                  uint32_t segmentBytes, uint32_t gatherThreads,
                                  const std::vector<int> &configuredCpus)
{
    if (gatherThreads == 0U || gatherThreads > 64U) {
        throw py::value_error("gatherThreads must be in [1, 64]");
    }
    auto &threadPool = GetGatherThreadPool(gatherThreads, configuredCpus);
    HybmAggregateUrmaDemoRequest request{};
    request.srcStride = srcStride;
    request.segmentCount = segmentCount;
    request.segmentBytes = segmentBytes;
    uint64_t gatherNs = 0;
    {
        py::gil_scoped_release release;
        const auto begin = Clock::now();
        threadPool.Run(reinterpret_cast<uint8_t *>(aggregate), reinterpret_cast<const uint8_t *>(source), request);
        const auto end = Clock::now();
        gatherNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    return gatherNs;
}

uint64_t AggregateGatherIndexedDemo(uint64_t source, uint64_t aggregate, uint64_t sourceIndices,
                                    uint32_t sourcePoolSegments, uint64_t sourceOrdinal, uint32_t segmentCount,
                                    uint32_t segmentBytes, uint32_t gatherThreads,
                                    const std::vector<int> &configuredCpus)
{
    if (sourcePoolSegments == 0U || gatherThreads == 0U || gatherThreads > 64U) {
        throw py::value_error(
            "sourcePoolSegments and gatherThreads must be positive; gatherThreads must not exceed 64");
    }
    auto &threadPool = GetGatherThreadPool(gatherThreads, configuredCpus);
    HybmAggregateUrmaDemoRequest request{};
    request.segmentCount = segmentCount;
    request.segmentBytes = segmentBytes;
    uint64_t gatherNs = 0;
    {
        py::gil_scoped_release release;
        const auto begin = Clock::now();
        threadPool.Run(reinterpret_cast<uint8_t *>(aggregate), reinterpret_cast<const uint8_t *>(source), request,
                       reinterpret_cast<const uint32_t *>(sourceIndices), sourcePoolSegments, sourceOrdinal);
        const auto end = Clock::now();
        gatherNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    return gatherNs;
}

uint64_t AggregateGatherAddressesDemo(uint64_t aggregate, uint64_t sourceAddresses, int64_t gvaToVaOffset,
                                      uint32_t segmentCount, uint32_t segmentBytes, uint32_t gatherThreads,
                                      const std::vector<int> &configuredCpus)
{
    if (gatherThreads == 0U || gatherThreads > 64U) {
        throw py::value_error("gatherThreads must be in [1, 64]");
    }
    auto &threadPool = GetGatherThreadPool(gatherThreads, configuredCpus);
    HybmAggregateUrmaDemoRequest request{};
    request.segmentCount = segmentCount;
    request.segmentBytes = segmentBytes;
    uint64_t gatherNs = 0;
    {
        py::gil_scoped_release release;
        const auto begin = Clock::now();
        threadPool.Run(reinterpret_cast<uint8_t *>(aggregate), nullptr, request, nullptr, 0U, 0U,
                       reinterpret_cast<const uint64_t *>(sourceAddresses), gvaToVaOffset);
        const auto end = Clock::now();
        gatherNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    return gatherNs;
}

void DefineAccOffloadConfig(py::module_ &m)
{
    py::enum_<offload_scene_t>(m, "Scene")
        .value("LOCAL", OFFLOAD_SCENE_LOCAL)
        .value("SHARED", OFFLOAD_SCENE_SHARED)
        .export_values();

    py::class_<offload_config_t>(m, "OffloadConfig")
        .def(py::init<>())
        .def_readwrite("device_id", &offload_config_t::deviceId)
        .def_readwrite("reserve_size", &offload_config_t::reserveSize,
                       "Reserved DRAM pool size in bytes, will be aligned up to GB")
        .def_readwrite("alloc_size", &offload_config_t::allocSize,
                       "Allocated local physical DRAM size in bytes, will be aligned up to GB. "
                       "LOCAL: must equal reserve_size; SHARED: provides the actual size")
        .def_readwrite("world_size", &offload_config_t::worldSize,
                       "number of ranks in the group (multi-card shared mode)")
        .def_readwrite("rank_id", &offload_config_t::rankId, "local rank id, 0 is the server (multi-card shared mode)")
        .def_readwrite("scene", &offload_config_t::scene,
                       "memory pool scene: LOCAL=single-card, SHARED=multi-card shared");
}

void DefineAccOffloadApi(py::module_ &m)
{
    m.def("initialize", &offload_init, py::call_guard<py::gil_scoped_release>(), py::arg("config"));

    m.def("uninitialize", &offload_uninit, py::call_guard<py::gil_scoped_release>());

    m.def("malloc", &offload_malloc, py::call_guard<py::gil_scoped_release>(), py::arg("size"), py::arg("flags") = 0);

    m.def("free", &offload_free, py::call_guard<py::gil_scoped_release>(), py::arg("ptr"), py::arg("flags") = 0);

    m.def("sparse_copy", &offload_sparse_copy, py::call_guard<py::gil_scoped_release>(), py::arg("srcPtrs"),
          py::arg("dstPtrs"), py::arg("lenPtrs"), py::arg("sizePtr"), py::arg("deviceId"));

    m.def("sparse_copy_urma", &offload_sparse_copy_urma, py::call_guard<py::gil_scoped_release>(), py::arg("srcPtrs"),
          py::arg("dstPtrs"), py::arg("lenPtrs"), py::arg("listNum"), py::arg("deviceId"));

    m.def("aggregate_wait_demo", &AggregateWaitDemo, py::arg("mailbox"), py::arg("expectedDoorbell"));

    m.def("aggregate_gather_range_demo", &AggregateGatherRangeDemo, py::arg("source"), py::arg("aggregate"),
          py::arg("srcStride"), py::arg("segmentCount"), py::arg("segmentBytes"),
          py::arg("gatherThreads") = 1U, py::arg("configuredCpus") = std::vector<int>{});

    m.def("aggregate_gather_indexed_demo", &AggregateGatherIndexedDemo, py::arg("source"), py::arg("aggregate"),
          py::arg("sourceIndices"), py::arg("sourcePoolSegments"), py::arg("sourceOrdinal"),
          py::arg("segmentCount"), py::arg("segmentBytes"), py::arg("gatherThreads") = 1U,
          py::arg("configuredCpus") = std::vector<int>{});
    m.def("aggregate_gather_addresses_demo", &AggregateGatherAddressesDemo, py::arg("aggregate"),
          py::arg("sourceAddresses"), py::arg("gvaToVaOffset"), py::arg("segmentCount"),
          py::arg("segmentBytes"), py::arg("gatherThreads") = 1U,
          py::arg("configuredCpus") = std::vector<int>{});

    m.def("npu_kvcache_scatter_copy", &offload_kvcache_scatter_copy, py::call_guard<py::gil_scoped_release>(),
          py::arg("hbmKpe"), py::arg("hbmCkv"), py::arg("hbmBlockTable"), py::arg("dramBlockTable"),
          py::arg("offloadSlots"), py::arg("srcTokenIds"), py::arg("dstSlots"), py::arg("copyCounts"),
          py::arg("readyFlag"), py::arg("hbmBlockCount"), py::arg("hbmMaxBlocks"), py::arg("dramMaxBlocks"),
          py::arg("dramBlockTableRows"), py::arg("batchSize"), py::arg("layerId"), py::arg("deviceId"));

    m.def("group_pack_copy", &offload_group_pack_copy, py::call_guard<py::gil_scoped_release>(), py::arg("srcPtrs"),
          py::arg("dstPtrs"), py::arg("lenPtrs"), py::arg("numLocalExpertPtr"), py::arg("groupList"),
          py::arg("packedGroupList"), py::arg("deviceId"));
}

PYBIND11_MODULE(_pymf_acc_offload, m)
{
    auto offload = m.def_submodule("offload", "Acc Offload Module.");

    DefineAccOffloadConfig(offload);
    DefineAccOffloadApi(offload);
}

#pragma GCC diagnostic pop
