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

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
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
};

void GatherPartition(const GatherTask &task, uint32_t threadIndex)
{
    const uint32_t begin = static_cast<uint32_t>(
        static_cast<uint64_t>(task.request.segmentCount) * threadIndex / task.threadCount);
    const uint32_t end = static_cast<uint32_t>(
        static_cast<uint64_t>(task.request.segmentCount) * (threadIndex + 1U) / task.threadCount);
    if (begin == end) {
        return;
    }
    auto request = task.request;
    request.segmentCount = end - begin;
    GatherSegments(task.dst + static_cast<uint64_t>(begin) * request.segmentBytes,
                   task.src + static_cast<uint64_t>(begin) * request.srcStride, request);
}

class GatherThreadPool {
public:
    explicit GatherThreadPool(uint32_t threadCount) : threadCount_(threadCount)
    {
        for (uint32_t index = 1; index < threadCount_; ++index) {
            workers_.emplace_back(&GatherThreadPool::WorkerLoop, this, index);
        }
    }

    ~GatherThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        workCv_.notify_all();
        for (auto &worker : workers_) {
            worker.join();
        }
    }

    uint32_t ThreadCount() const
    {
        return threadCount_;
    }

    void Run(uint8_t *dst, const uint8_t *src, const HybmAggregateUrmaDemoRequest &request)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = {dst, src, request, threadCount_};
            pendingWorkers_ = static_cast<uint32_t>(workers_.size());
            ++generation_;
        }
        workCv_.notify_all();
        GatherPartition(task_, 0);
        std::unique_lock<std::mutex> lock(mutex_);
        doneCv_.wait(lock, [this]() { return pendingWorkers_ == 0U; });
    }

private:
    void WorkerLoop(uint32_t threadIndex)
    {
        uint64_t observedGeneration = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            workCv_.wait(lock, [this, observedGeneration]() {
                return stopping_ || generation_ != observedGeneration;
            });
            if (stopping_) {
                return;
            }
            observedGeneration = generation_;
            lock.unlock();
            GatherPartition(task_, threadIndex);
            lock.lock();
            if (--pendingWorkers_ == 0U) {
                doneCv_.notify_one();
            }
        }
    }

    uint32_t threadCount_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable workCv_;
    std::condition_variable doneCv_;
    GatherTask task_{};
    uint64_t generation_{0};
    uint32_t pendingWorkers_{0};
    bool stopping_{false};
};

GatherThreadPool &GetGatherThreadPool(uint32_t threadCount)
{
    static std::mutex poolMutex;
    static std::unique_ptr<GatherThreadPool> pool;
    std::lock_guard<std::mutex> lock(poolMutex);
    if (pool == nullptr || pool->ThreadCount() != threadCount) {
        pool = std::make_unique<GatherThreadPool>(threadCount);
    }
    return *pool;
}
} // namespace

py::tuple AggregateWaitAndGatherDemo(uint64_t mailbox, uint64_t source, uint64_t aggregate, uint32_t gatherThreads)
{
    if (gatherThreads == 0U || gatherThreads > 64U) {
        throw py::value_error("gatherThreads must be in [1, 64]");
    }
    auto &threadPool = GetGatherThreadPool(gatherThreads);
    auto *message = reinterpret_cast<HybmAggregateUrmaDemoMessage *>(mailbox);
    HybmAggregateUrmaDemoRequest request{};
    uint64_t waitNs = 0;
    uint64_t gatherNs = 0;
    {
        py::gil_scoped_release release;
        const auto waitBegin = Clock::now();
        while (__atomic_load_n(&message->doorbell, __ATOMIC_ACQUIRE) == 0U) {}
        const auto gatherBegin = Clock::now();
        request = message->request;
        auto *src = reinterpret_cast<const uint8_t *>(source);
        auto *dst = reinterpret_cast<uint8_t *>(aggregate);
        threadPool.Run(dst, src, request);
        const auto gatherEnd = Clock::now();
        waitNs = std::chrono::duration_cast<std::chrono::nanoseconds>(gatherBegin - waitBegin).count();
        gatherNs = std::chrono::duration_cast<std::chrono::nanoseconds>(gatherEnd - gatherBegin).count();
    }
    return py::make_tuple(request.dstNewGva, request.readyGva, request.totalBytes, waitNs, gatherNs);
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

    m.def("aggregate_wait_and_gather_demo", &AggregateWaitAndGatherDemo, py::arg("mailbox"), py::arg("source"),
          py::arg("aggregate"), py::arg("gatherThreads") = 1U);

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
