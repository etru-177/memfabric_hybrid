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
constexpr uint32_t kScatterPrefetchDistance = 4U;

uint64_t ElapsedNs(Clock::time_point begin, Clock::time_point end)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

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

const ock::mf::BatchCopyRangeEntry *FindMailboxRange(const ock::mf::BatchCopyRouteTable *route, uint64_t mailbox)
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
        if (mailbox >= range->srcGvaBegin && mailbox + sizeof(HybmAggregateUrmaDemoMessage) <= range->srcGvaEnd) {
            return range;
        }
    }
    return nullptr;
}

uint32_t WriteRemoteRequestAndDoorbell(const ock::mf::BatchCopyPeerEntry &peer, uint64_t remote,
                                       const HybmAggregateUrmaDemoMessage &message)
{
    void *destinations[] = {reinterpret_cast<void *>(remote),
                            reinterpret_cast<void *>(remote + offsetof(HybmAggregateUrmaDemoMessage, doorbell))};
    void *sources[] = {const_cast<HybmAggregateUrmaDemoRequest *>(&message.request),
                       const_cast<uint64_t *>(&message.doorbell)};
    uint64_t lengths[] = {sizeof(message.request), sizeof(message.doorbell)};
    HybmOneSideOpParam write{};
    write.thread = peer.thread;
    write.channel = peer.channel;
    write.list_num = 2U;
    write.dst_buf_addr_list = destinations;
    write.src_buf_addr_list = sources;
    write.len_list = lengths;
    return HybmBatchWriteStrict(&write);
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
        if (request.segmentCount - index > kScatterPrefetchDistance) {
            __builtin_prefetch(source + kScatterPrefetchDistance * Bytes, 0, 1);
            __builtin_prefetch(destination + kScatterPrefetchDistance * request.dstStride, 1, 1);
        }
        CopyFixed<Bytes>(destination, source);
        source += Bytes;
        destination += request.dstStride;
    }
}

void ScatterDynamic(const HybmAggregateUrmaDemoParam &param, const HybmAggregateUrmaDemoRequest &request)
{
    for (uint32_t index = 0; index < request.segmentCount; ++index) {
        auto *destination = param.dstBase + index * request.dstStride;
        std::memcpy(destination, param.dstNew + index * request.segmentBytes, request.segmentBytes);
    }
}

void Scatter(const HybmAggregateUrmaDemoParam &param)
{
    const auto &request = param.message->request;
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
} // namespace

extern "C" uint32_t HybmAggregateUrmaDemo(HybmAggregateUrmaDemoParam *param)
{
    const auto begin = Clock::now();
    const auto *route = reinterpret_cast<const ock::mf::BatchCopyRouteTable *>(ock::mf::HYBM_BATCH_COPY_META_ADDR);
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&route->header));
    const auto *range = FindMailboxRange(route, param->message->request.hostMailboxGva);
    if (range == nullptr) {
        HYBM_LOGE(BM_NOT_CONNECTED, "aggregate demo mailbox has no route, gva=0x%lx",
                  param->message->request.hostMailboxGva);
        return BM_NOT_CONNECTED;
    }
    const auto &peer = route->peers[range->peerIndex];
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&peer));
    const uint64_t remote = range->hcommVaBegin + param->message->request.hostMailboxGva - range->srcGvaBegin;
    const auto ret = WriteRemoteRequestAndDoorbell(peer, remote, *param->message);
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "aggregate demo request and doorbell write failed, ret=%u", ret);
        return ret;
    }
    const auto requested = Clock::now();
    WaitForHost(*param);
    const auto ready = Clock::now();
    Scatter(*param);
    const auto copied = Clock::now();
    __asm__ __volatile__("dsb ish" : : : "memory");
    const auto done = Clock::now();
    param->timing->requestNs = ElapsedNs(begin, requested);
    param->timing->waitHostNs = ElapsedNs(requested, ready);
    param->timing->scatterCopyNs = ElapsedNs(ready, copied);
    param->timing->scatterPublishNs = ElapsedNs(copied, done);
    param->timing->scatterNs = ElapsedNs(ready, done);
    param->timing->totalNs = ElapsedNs(begin, done);
    FlushDeviceCache(reinterpret_cast<uintptr_t>(param->timing));
    return BM_OK;
}
