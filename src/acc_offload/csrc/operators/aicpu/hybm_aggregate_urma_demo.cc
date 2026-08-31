/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#include "hybm_aggregate_urma_demo.h"

#include <chrono>
#include <cstring>

#include "hybm_batch_copy_route.h"
#include "hybm_batch_transfer.h"
#include "hybm_def.h"
#include "hybm_define.h"
#include "hybm_kernel_log.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr uintptr_t kCacheLineBytes = 64U;

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

void FlushDeviceCacheRange(uintptr_t address, uint64_t bytes)
{
    const uintptr_t end = address + bytes;
    for (uintptr_t line = address & ~(kCacheLineBytes - 1U); line < end; line += kCacheLineBytes) {
        __asm__ __volatile__("dc cvac, %0" : : "r"(line) : "memory");
    }
}

void InvalidateDeviceCacheRange(uintptr_t address, uint64_t bytes)
{
    const uintptr_t end = address + bytes;
    for (uintptr_t line = address & ~(kCacheLineBytes - 1U); line < end; line += kCacheLineBytes) {
        __asm__ __volatile__("dc civac, %0" : : "r"(line) : "memory");
    }
    __asm__ __volatile__("dsb ish" : : : "memory");
}

const ock::mf::BatchCopyRangeEntry *FindMailboxRange(const ock::mf::BatchCopyRouteTable *route, uint64_t mailbox)
{
    for (uint16_t index = 0; index < route->header.rangeCount; ++index) {
        const auto *range = &route->ranges[index];
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(range));
        if (mailbox >= range->srcGvaBegin && mailbox + sizeof(HybmAggregateUrmaDemoMessage) <= range->srcGvaEnd) {
            return range;
        }
    }
    return nullptr;
}

uint32_t WriteRemote(const ock::mf::BatchCopyPeerEntry &peer, uint64_t remote, const void *local, uint64_t bytes)
{
    void *destinations[] = {reinterpret_cast<void *>(remote)};
    void *sources[] = {const_cast<void *>(local)};
    uint64_t lengths[] = {bytes};
    HybmOneSideOpParam write{};
    write.thread = peer.thread;
    write.channel = peer.channel;
    write.list_num = 1;
    write.dst_buf_addr_list = destinations;
    write.src_buf_addr_list = sources;
    write.len_list = lengths;
    return HybmBatchWrite(&write);
}

void WaitForHost(const HybmAggregateUrmaDemoParam &param)
{
    do {
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(param.ready));
    } while (*param.ready != param.message->doorbell);
}

void Scatter(const HybmAggregateUrmaDemoParam &param)
{
    const auto &request = param.message->request;
    InvalidateDeviceCacheRange(reinterpret_cast<uintptr_t>(param.dstNew), request.totalBytes);
    for (uint32_t index = 0; index < request.segmentCount; ++index) {
        auto *destination = param.dstBase + index * request.dstStride;
        std::memcpy(destination, param.dstNew + index * request.segmentBytes, request.segmentBytes);
        FlushDeviceCacheRange(reinterpret_cast<uintptr_t>(destination), request.segmentBytes);
    }
    __asm__ __volatile__("dsb ish" : : : "memory");
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
    auto ret = WriteRemote(peer, remote, &param->message->request, sizeof(param->message->request));
    if (ret == BM_OK) {
        ret = WriteRemote(peer, remote + offsetof(HybmAggregateUrmaDemoMessage, doorbell), &param->message->doorbell,
                          sizeof(param->message->doorbell));
    }
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "aggregate demo request write failed, ret=%u", ret);
        return ret;
    }
    const auto requested = Clock::now();
    WaitForHost(*param);
    const auto ready = Clock::now();
    Scatter(*param);
    const auto done = Clock::now();
    param->timing->requestNs = ElapsedNs(begin, requested);
    param->timing->waitHostNs = ElapsedNs(requested, ready);
    param->timing->scatterNs = ElapsedNs(ready, done);
    param->timing->totalNs = ElapsedNs(begin, done);
    FlushDeviceCache(reinterpret_cast<uintptr_t>(param->timing));
    return BM_OK;
}
