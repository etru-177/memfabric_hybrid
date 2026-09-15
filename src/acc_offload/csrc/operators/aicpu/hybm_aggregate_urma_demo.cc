/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#include "hybm_aggregate_urma_demo.h"

#include <chrono>
#include <cstddef>
#include <cstring>

#include "hybm_batch_copy_route.h"
#include "hybm_batch_transfer.h"
#include "hybm_def.h"
#include "hybm_define.h"
#include "hybm_kernel_log.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kScatterLaneCount = 6U;

void InvalidateDeviceCache(uintptr_t address)
{
    __asm__ __volatile__("dc civac, %0" : : "r"(address) : "memory");
    __asm__ __volatile__("dsb ish" : : : "memory");
}

void FlushDeviceCache(uintptr_t address)
{
    __asm__ __volatile__("dc cvac, %0" : : "r"(address) : "memory");
    __asm__ __volatile__("dsb ish" : : : "memory");
}

const ock::mf::BatchCopyRangeEntry *FindMailboxRange(const ock::mf::BatchCopyRouteTable *route, uint64_t mailbox,
                                                     uint64_t mailboxBytes)
{
    uintptr_t previousLine = 0U;
    for (uint16_t index = 0; index < route->header.rangeCount; ++index) {
        const auto *range = &route->ranges[index];
        // Adjacent 32B range entries share one 64B cache line. Do not evict it twice.
        const uintptr_t line = reinterpret_cast<uintptr_t>(range) & ~uintptr_t{63U};
        if (line != previousLine) {
            InvalidateDeviceCache(line);
            previousLine = line;
        }
        if (mailbox >= range->srcGvaBegin && mailbox <= range->srcGvaEnd &&
            mailboxBytes <= range->srcGvaEnd - mailbox) {
            return range;
        }
    }
    return nullptr;
}

uint32_t WriteRemoteRequestAndDoorbell(const ock::mf::BatchCopyPeerEntry &peer, uint64_t remote,
                                       const HybmAggregateUrmaDemoMessage &message, const uint64_t *sourceAddresses)
{
    const uint64_t addressBytes = static_cast<uint64_t>(message.request.segmentCount) * sizeof(uint64_t);
    uint64_t payloadLength = addressBytes + sizeof(HybmAggregateUrmaDemoRequest);
    void *payloadDestination = reinterpret_cast<void *>(remote);
    void *payloadSource = const_cast<uint64_t *>(sourceAddresses);
    HybmOneSideOpParam payload{};
    payload.thread = peer.thread;
    payload.channel = peer.channel;
    payload.list_num = 1U;
    payload.dst_buf_addr_list = &payloadDestination;
    payload.src_buf_addr_list = &payloadSource;
    payload.len_list = &payloadLength;
    const auto ret = HybmWriteStrict(&payload);
    if (ret != BM_OK) {
        return ret;
    }

    uint64_t doorbellLength = sizeof(message.doorbell);
    void *doorbellDestination = reinterpret_cast<void *>(remote + payloadLength);
    void *doorbellSource = const_cast<uint64_t *>(&message.doorbell);
    HybmOneSideOpParam doorbell{};
    doorbell.thread = peer.thread;
    doorbell.channel = peer.channel;
    doorbell.list_num = 1U;
    doorbell.dst_buf_addr_list = &doorbellDestination;
    doorbell.src_buf_addr_list = &doorbellSource;
    doorbell.len_list = &doorbellLength;
    return HybmWriteStrict(&doorbell);
}

void WaitForHost(const HybmAggregateUrmaDemoParam &param)
{
    do {
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(param.ready));
    } while (*param.ready != param.message->doorbell);
}

template <size_t Bytes>
__attribute__((always_inline)) inline void CopyFixed(uint8_t *__restrict destination,
                                                     const uint8_t *__restrict source)
{
    __builtin_memcpy(destination, source, Bytes);
}

template <size_t Bytes>
void ScatterFixed(const HybmAggregateUrmaDemoParam &param, const HybmAggregateUrmaDemoRequest &request)
{
    auto *destination = param.dstBase;
    const auto *source = param.dstNew;
    for (uint32_t index = 0; index < request.segmentCount; ++index) {
        CopyFixed<Bytes>(destination, source);
        source += Bytes;
        destination += request.dstStride;
    }
}

uint64_t NowNs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

HybmAggregateUrmaDemoSync *GetSync(HybmAggregateUrmaDemoTiming *timing)
{
    return reinterpret_cast<HybmAggregateUrmaDemoSync *>(timing + 1);
}

void ScatterDynamic(const HybmAggregateUrmaDemoParam &param, const HybmAggregateUrmaDemoRequest &request)
{
    for (uint32_t index = 0; index < request.segmentCount; ++index) {
        auto *destination = param.dstBase + index * request.dstStride;
        std::memcpy(destination, param.dstNew + index * request.segmentBytes, request.segmentBytes);
    }
}

void Scatter(const HybmAggregateUrmaDemoParam &param, const HybmAggregateUrmaDemoRequest &request)
{
    if (request.segmentBytes == 656U) {
        ScatterFixed<656>(param, request);
    } else if (request.segmentBytes == 576U) {
        ScatterFixed<576>(param, request);
    } else if (request.segmentBytes == 1152U) {
        ScatterFixed<1152>(param, request);
    } else {
        ScatterDynamic(param, request);
    }
}

void ScatterPartition(const HybmAggregateUrmaDemoParam &param, uint32_t laneIndex)
{
    auto request = param.message->request;
    const uint32_t begin = static_cast<uint32_t>(
        static_cast<uint64_t>(request.segmentCount) * laneIndex / kScatterLaneCount);
    const uint32_t end = static_cast<uint32_t>(
        static_cast<uint64_t>(request.segmentCount) * (laneIndex + 1U) / kScatterLaneCount);
    HybmAggregateUrmaDemoParam partition = param;
    partition.dstNew += static_cast<uint64_t>(begin) * request.segmentBytes;
    partition.dstBase += static_cast<uint64_t>(begin) * request.dstStride;
    request.segmentCount = end - begin;
    Scatter(partition, request);
}

uint32_t PublishRequest(HybmAggregateUrmaDemoParam *param)
{
    const auto *route = reinterpret_cast<const ock::mf::BatchCopyRouteTable *>(ock::mf::HYBM_BATCH_COPY_META_ADDR);
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&route->header));
    const uint64_t addressBytes = static_cast<uint64_t>(param->message->request.segmentCount) * sizeof(uint64_t);
    const uint64_t mailboxBytes = addressBytes + sizeof(HybmAggregateUrmaDemoRequest) + sizeof(uint64_t);
    const auto *range = FindMailboxRange(route, param->message->request.hostMailboxGva, mailboxBytes);
    if (range == nullptr) {
        HYBM_LOGE(BM_NOT_CONNECTED, "aggregate demo mailbox has no route, gva=0x%lx",
                  param->message->request.hostMailboxGva);
        return BM_NOT_CONNECTED;
    }
    const auto &peer = route->peers[range->peerIndex];
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&peer));
    const uint64_t remote = range->hcommVaBegin + param->message->request.hostMailboxGva - range->srcGvaBegin;
    const auto ret = WriteRemoteRequestAndDoorbell(peer, remote, *param->message, param->sourceAddresses);
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "aggregate demo request and doorbell write failed, ret=%u", ret);
    }
    return ret;
}

uint32_t WaitUntilReady(const HybmAggregateUrmaDemoSync *sync, uint32_t generation)
{
    uint32_t phase = __atomic_load_n(&sync->phase, __ATOMIC_ACQUIRE);
    while (phase != generation) {
        __asm__ __volatile__("yield" : : : "memory");
        phase = __atomic_load_n(&sync->phase, __ATOMIC_ACQUIRE);
    }
    return __atomic_load_n(&sync->error, __ATOMIC_ACQUIRE);
}

void FinishLastLane(HybmAggregateUrmaDemoParam *param, uint64_t copied)
{
    __asm__ __volatile__("dsb ish" : : : "memory");
    const uint64_t done = NowNs();
    const uint64_t begin = param->timing->requestNs;
    const uint64_t requested = param->timing->waitHostNs;
    const uint64_t ready = param->timing->scatterCopyNs;
    param->timing->requestNs = requested - begin;
    param->timing->waitHostNs = ready - requested;
    param->timing->scatterCopyNs = copied - ready;
    param->timing->scatterPublishNs = done - copied;
    param->timing->scatterNs = done - ready;
    param->timing->totalNs = done - begin;
    FlushDeviceCache(reinterpret_cast<uintptr_t>(param->timing));
}
} // namespace

extern "C" uint32_t HybmAggregateUrmaDemo(HybmAggregateUrmaDemoParam *param)
{
    auto *sync = GetSync(param->timing);
    const uint32_t laneTicket = __atomic_fetch_add(&sync->nextLane, 1U, __ATOMIC_ACQ_REL);
    const uint32_t laneIndex = laneTicket % kScatterLaneCount;
    const uint32_t generation = static_cast<uint32_t>(param->message->doorbell);
    if (laneIndex == 0U) {
        param->timing->requestNs = NowNs();
        const auto ret = PublishRequest(param);
        param->timing->waitHostNs = NowNs();
        __atomic_store_n(&sync->error, ret, __ATOMIC_RELEASE);
        if (ret != BM_OK) {
            __atomic_store_n(&sync->phase, generation, __ATOMIC_RELEASE);
            return ret;
        }
        WaitForHost(*param);
        param->timing->scatterCopyNs = NowNs();
        __atomic_store_n(&sync->phase, generation, __ATOMIC_RELEASE);
    } else {
        const auto ret = WaitUntilReady(sync, generation);
        if (ret != BM_OK) {
            return ret;
        }
    }
    ScatterPartition(*param, laneIndex);
    const uint32_t completed = __atomic_add_fetch(&sync->completedLanes, 1U, __ATOMIC_ACQ_REL);
    if (completed % kScatterLaneCount == 0U) {
        FinishLastLane(param, NowNs());
    }
    return BM_OK;
}
