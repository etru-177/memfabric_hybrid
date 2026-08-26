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

#include "hybm_batch_copy.h"

#include <array>
#include <chrono>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include "hybm_batch_copy_route.h"
#include "hybm_batch_transfer.h"
#include "hybm_define.h"
#include "hybm_def.h"
#include "hybm_kernel_log.h"

namespace {
using ock::mf::BatchCopyRangeEntry;
using ock::mf::BatchCopyRouteTable;

constexpr uint64_t kCompletionAddress = ock::mf::HYBM_BATCH_COPY_META_ADDR + ock::mf::BATCH_COPY_COMPLETION_OFFSET;
constexpr auto kCompletionTimeout = std::chrono::seconds(60);

struct BatchCopyGroup {
    std::vector<void *> destinations;
    std::vector<void *> sources;
    std::vector<uint64_t> lengths;
};

using BatchCopyGroups = std::array<BatchCopyGroup, ock::mf::BATCH_COPY_MAX_PEER_COUNT>;
using BatchCopyRoundState = std::array<uint32_t, ock::mf::BATCH_COPY_MAX_PEER_COUNT>;

void InvalidateDeviceCache(uintptr_t address)
{
    __asm__ __volatile__("dc civac, %0" :: "r"(address) : "memory");
    __asm__ __volatile__("dsb ish" ::: "memory");
}

void FlushDeviceCache(uintptr_t address)
{
    __asm__ __volatile__("dc cvac, %0" :: "r"(address) : "memory");
    __asm__ __volatile__("dsb ish" :::"memory");
}

void DeviceMemoryBarrier()
{
    __asm__ __volatile__("dsb ish" ::: "memory");
}

int32_t ValidateFourInputs(const HybmBatchCopyParam *param)
{
    if (param == nullptr || param->list_num == 0U || param->dst_buf_addr_list == nullptr ||
        param->src_buf_addr_list == nullptr || param->len_list == nullptr) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy inputs, param=%p listNum=%u dst=%p src=%p len=%p",
                  static_cast<const void *>(param), param == nullptr ? 0U : param->list_num,
                  param == nullptr ? nullptr : param->dst_buf_addr_list,
                  param == nullptr ? nullptr : param->src_buf_addr_list, param == nullptr ? nullptr : param->len_list);
        return BM_INVALID_PARAM;
    }

    constexpr size_t kBytesPerItem = sizeof(void *) * 2U + sizeof(uint64_t);
    if (param->list_num > std::numeric_limits<size_t>::max() / kBytesPerItem) {
        HYBM_LOGE(BM_INVALID_PARAM, "BatchCopy list byte size overflows, listNum=%u bytesPerItem=%zu", param->list_num,
                  kBytesPerItem);
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t ValidateRouteHeader(const BatchCopyRouteTable *route)
{
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&route->header));
    const auto &header = route->header;
    if (header.magic != ock::mf::BATCH_COPY_ROUTE_MAGIC) {
        HYBM_LOGE(BM_NOT_INITIALIZED, "BatchCopy route is not published, magic=0x%x", header.magic);
        return BM_NOT_INITIALIZED;
    }
    if (header.peerCount == 0U || header.peerCount > ock::mf::BATCH_COPY_MAX_PEER_COUNT || header.rangeCount == 0U ||
        header.rangeCount > ock::mf::BATCH_COPY_MAX_RANGE_COUNT ||
        header.rangeCount > header.peerCount * ock::mf::BATCH_COPY_MAX_RANGE_PER_PEER) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy route counts, peerCount=%u rangeCount=%u", header.peerCount,
                  header.rangeCount);
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t ValidateRoutePeers(const BatchCopyRouteTable *route)
{
    for (uint16_t peerIndex = 0U; peerIndex < route->header.peerCount; ++peerIndex) {
        const auto &peer = route->peers[peerIndex];
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&peer));
        if (peer.thread == 0U || peer.channel == 0U || peer.remoteFlagAddr == 0U) {
            HYBM_LOGE(BM_NOT_CONNECTED,
                      "invalid BatchCopy peer, peerIndex=%u thread=%lu channel=%lu remoteFlagAddr=0x%lx", peerIndex,
                      peer.thread, peer.channel, peer.remoteFlagAddr);
            return BM_NOT_CONNECTED;
        }
    }
    return BM_OK;
}

int32_t ValidateRouteRange(const BatchCopyRangeEntry &range, uint16_t peerCount, uint16_t rangeIndex,
                           std::array<uint16_t, ock::mf::BATCH_COPY_MAX_PEER_COUNT> &rangeCounts)
{
    if (range.srcGvaBegin == 0U || range.srcGvaBegin >= range.srcGvaEnd || range.hcommVaBegin == 0U ||
        range.peerIndex >= peerCount) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy range, index=%u begin=0x%lx end=0x%lx hcomm=0x%lx peerIndex=%u",
                  rangeIndex, range.srcGvaBegin, range.srcGvaEnd, range.hcommVaBegin, range.peerIndex);
        return BM_INVALID_PARAM;
    }

    const uint64_t length = range.srcGvaEnd - range.srcGvaBegin;
    if (range.hcommVaBegin > std::numeric_limits<uint64_t>::max() - length ||
        ++rangeCounts[range.peerIndex] > ock::mf::BATCH_COPY_MAX_RANGE_PER_PEER) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy range span/count, index=%u peerIndex=%u length=0x%lx",
                  rangeIndex, range.peerIndex, length);
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t ValidateRouteRanges(const BatchCopyRouteTable *route)
{
    std::array<uint16_t, ock::mf::BATCH_COPY_MAX_PEER_COUNT> rangeCounts{};
    uint64_t previousEnd = 0U;
    for (uint16_t index = 0U; index < route->header.rangeCount; ++index) {
        const auto &range = route->ranges[index];
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&range));
        const auto ret = ValidateRouteRange(range, route->header.peerCount, index, rangeCounts);
        if (ret != BM_OK) {
            return ret;
        }
        if (index != 0U && range.srcGvaBegin < previousEnd) {
            HYBM_LOGE(BM_INVALID_PARAM,
                      "BatchCopy ranges overlap or are unsorted, index=%u begin=0x%lx previousEnd=0x%lx", index,
                      range.srcGvaBegin, previousEnd);
            return BM_INVALID_PARAM;
        }
        previousEnd = range.srcGvaEnd;
    }
    return BM_OK;
}

void LogRouteTableForDebug(const BatchCopyRouteTable *route)
{
    HYBM_LOGE(BM_OK, "BatchCopy route debug, magic=0x%x peerCount=%u rangeCount=%u", route->header.magic,
              route->header.peerCount, route->header.rangeCount);
    for (uint16_t index = 0U; index < route->header.peerCount; ++index) {
        const auto &peer = route->peers[index];
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&peer));
        HYBM_LOGE(BM_OK, "BatchCopy route peer, index=%u thread=%lu channel=%lu remoteFlagAddr=0x%lx", index,
                  peer.thread, peer.channel, peer.remoteFlagAddr);
    }
    for (uint16_t index = 0U; index < route->header.rangeCount; ++index) {
        const auto &range = route->ranges[index];
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&range));
        HYBM_LOGE(BM_OK,
                  "BatchCopy route range, index=%u srcGvaBegin=0x%lx srcGvaEnd=0x%lx hcommVaBegin=0x%lx peerIndex=%u",
                  index, range.srcGvaBegin, range.srcGvaEnd, range.hcommVaBegin, range.peerIndex);
    }
}

int32_t ValidatePublishedRoute(const BatchCopyRouteTable *route)
{
    auto ret = ValidateRouteHeader(route);
    if (ret != BM_OK) {
        return ret;
    }
    LogRouteTableForDebug(route);
    ret = ValidateRoutePeers(route);
    if (ret != BM_OK) {
        return ret;
    }
    ret = ValidateRouteRanges(route);
    if (ret != BM_OK) {
        return ret;
    }
    InvalidateDeviceCache(reinterpret_cast<uintptr_t>(&route->header.magic));
    if (route->header.magic != ock::mf::BATCH_COPY_ROUTE_MAGIC) {
        HYBM_LOGE(BM_NOT_INITIALIZED, "BatchCopy route was cleared during validation");
        return BM_NOT_INITIALIZED;
    }
    return BM_OK;
}

const BatchCopyRangeEntry *FindCoveringRange(const BatchCopyRouteTable *route, uint64_t source, uint64_t length)
{
    for (uint16_t index = 0U; index < route->header.rangeCount; ++index) {
        const auto *range = &route->ranges[index];
        if (source >= range->srcGvaBegin && source < range->srcGvaEnd && length <= range->srcGvaEnd - source) {
            return range;
        }
    }
    return nullptr;
}

int32_t ValidateDestination(uint64_t destination, uint64_t length, uint32_t index)
{
    if (destination == 0U || destination > std::numeric_limits<uint64_t>::max() - length) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy destination, index=%u dst=0x%lx length=0x%lx", index,
                  destination, length);
        return BM_INVALID_PARAM;
    }
    const uint64_t end = destination + length;
    if (destination < ock::mf::SVM_END_ADDR && end > ock::mf::HYBM_BATCH_COPY_META_ADDR) {
        HYBM_LOGE(BM_INVALID_PARAM,
                  "BatchCopy destination overlaps control HBM, index=%u dst=0x%lx end=0x%lx controlBegin=0x%lx "
                  "controlEnd=0x%lx",
                  index, destination, end, ock::mf::HYBM_BATCH_COPY_META_ADDR, ock::mf::SVM_END_ADDR);
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t ResolveAndAppendItem(const HybmBatchCopyParam &param, uint32_t index, const BatchCopyRouteTable *route,
                             BatchCopyGroups &groups)
{
    const uint64_t length = param.len_list[index];
    if (length == 0U) {
        return BM_OK;
    }
    const uint64_t source = reinterpret_cast<uint64_t>(param.src_buf_addr_list[index]);
    const uint64_t destination = reinterpret_cast<uint64_t>(param.dst_buf_addr_list[index]);
    auto ret = ValidateDestination(destination, length, index);
    if (ret != BM_OK) {
        return ret;
    }
    if (source == 0U || source > std::numeric_limits<uint64_t>::max() - length) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid BatchCopy source, index=%u src=0x%lx length=0x%lx", index, source, length);
        return BM_INVALID_PARAM;
    }
    const auto *range = FindCoveringRange(route, source, length);
    if (range == nullptr) {
        HYBM_LOGE(BM_NOT_CONNECTED, "BatchCopy source has no route, index=%u src=0x%lx length=0x%lx", index, source,
                  length);
        return BM_NOT_CONNECTED;
    }
    const uint64_t hcommSource = range->hcommVaBegin + (source - range->srcGvaBegin);
    auto &group = groups[range->peerIndex];
    group.destinations.push_back(reinterpret_cast<void *>(destination));
    group.sources.push_back(reinterpret_cast<void *>(hcommSource));
    group.lengths.push_back(length);
    return BM_OK;
}

int32_t ValidateAndGroupItems(const HybmBatchCopyParam &param, const BatchCopyRouteTable *route,
                              BatchCopyGroups &groups)
{
    try {
        for (uint32_t index = 0U; index < param.list_num; ++index) {
            const auto ret = ResolveAndAppendItem(param, index, route, groups);
            if (ret != BM_OK) {
                return ret;
            }
        }
    } catch (const std::bad_alloc &) {
        HYBM_LOGE(BM_MALLOC_FAILED, "allocate BatchCopy groups failed, listNum=%u", param.list_num);
        return BM_MALLOC_FAILED;
    } catch (...) {
        HYBM_LOGE(BM_ERROR, "unexpected exception while grouping BatchCopy items, listNum=%u", param.list_num);
        return BM_ERROR;
    }
    return BM_OK;
}

volatile uint64_t *GetCompletionCell(uint16_t peerIndex)
{
    return reinterpret_cast<volatile uint64_t *>(kCompletionAddress + peerIndex * sizeof(uint64_t));
}

void ClearUsedCompletionCells(const BatchCopyRoundState &roundCounts, uint16_t peerCount)
{
    for (uint16_t peerIndex = 0U; peerIndex < peerCount; ++peerIndex) {
        if (roundCounts[peerIndex] == 0U) {
            continue;
        }
        auto *cell = GetCompletionCell(peerIndex);
        *cell = 0U;
        FlushDeviceCache(reinterpret_cast<uintptr_t>(cell));
    }
    DeviceMemoryBarrier();
}

int32_t SubmitPeerGroups(const BatchCopyRouteTable *route, BatchCopyGroups &groups,
                         const BatchCopyRoundState &offsets, const BatchCopyRoundState &roundCounts)
{
    for (uint16_t peerIndex = 0U; peerIndex < route->header.peerCount; ++peerIndex) {
        auto &group = groups[peerIndex];
        const uint32_t count = roundCounts[peerIndex];
        if (count == 0U) {
            continue;
        }
        const uint32_t offset = offsets[peerIndex];
        const auto &peer = route->peers[peerIndex];
        HybmOneSideOpParam oneSide{};
        oneSide.thread = peer.thread;
        oneSide.channel = peer.channel;
        oneSide.list_num = count;
        oneSide.dst_buf_addr_list = group.destinations.data() + offset;
        oneSide.src_buf_addr_list = group.sources.data() + offset;
        oneSide.len_list = group.lengths.data() + offset;
        oneSide.remote_flag_addr = peer.remoteFlagAddr;
        oneSide.local_flag_addr = reinterpret_cast<uint64_t>(GetCompletionCell(peerIndex));
        oneSide.flag_size = sizeof(uint64_t);
        const auto ret = static_cast<int32_t>(HybmBatchRead(&oneSide));
        if (ret != BM_OK) {
            HYBM_LOGE(ret,
                      "HybmBatchRead failed for BatchCopy peer, peerIndex=%u offset=%u itemCount=%u", peerIndex,
                      offset, count);
            return ret;
        }
    }
    return BM_OK;
}

bool AllUsedPeersCompleted(const BatchCopyRoundState &roundCounts, uint16_t peerCount)
{
    for (uint16_t peerIndex = 0U; peerIndex < peerCount; ++peerIndex) {
        if (roundCounts[peerIndex] == 0U) {
            continue;
        }
        auto *cell = GetCompletionCell(peerIndex);
        InvalidateDeviceCache(reinterpret_cast<uintptr_t>(cell));
        if (*cell == 0U) {
            return false;
        }
    }
    return true;
}

int32_t WaitForPeerCompletions(const BatchCopyRoundState &roundCounts, uint16_t peerCount)
{
    const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
    uint32_t spins = 0U;
    while (!AllUsedPeersCompleted(roundCounts, peerCount)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            HYBM_LOGE(BM_TIMEOUT, "BatchCopy completion timed out, peerCount=%u", peerCount);
            return BM_TIMEOUT;
        }
        if ((++spins & 0x3FFU) == 0U) {
            std::this_thread::yield();
        }
    }
    DeviceMemoryBarrier();
    return BM_OK;
}

bool PrepareNextRound(const BatchCopyGroups &groups, uint16_t peerCount, const BatchCopyRoundState &offsets,
                      BatchCopyRoundState &roundCounts)
{
    bool hasWork = false;
    for (uint16_t peerIndex = 0U; peerIndex < peerCount; ++peerIndex) {
        const auto total = static_cast<uint32_t>(groups[peerIndex].lengths.size());
        const uint32_t remaining = total - offsets[peerIndex];
        roundCounts[peerIndex] = std::min(remaining, kHybmBatchCopyMaxRoundDescriptors);
        hasWork = hasWork || roundCounts[peerIndex] != 0U;
    }
    return hasWork;
}

int32_t SubmitInCompletionRounds(const BatchCopyRouteTable *route, BatchCopyGroups &groups)
{
    BatchCopyRoundState offsets{};
    BatchCopyRoundState roundCounts{};
    while (PrepareNextRound(groups, route->header.peerCount, offsets, roundCounts)) {
        // 只有完成标志回到本端后才进入下一轮，确保上一轮WQE已执行并释放SQ空间。
        ClearUsedCompletionCells(roundCounts, route->header.peerCount);
        auto ret = SubmitPeerGroups(route, groups, offsets, roundCounts);
        if (ret != BM_OK) {
            return ret;
        }
        ret = WaitForPeerCompletions(roundCounts, route->header.peerCount);
        if (ret != BM_OK) {
            return ret;
        }
        for (uint16_t peerIndex = 0U; peerIndex < route->header.peerCount; ++peerIndex) {
            offsets[peerIndex] += roundCounts[peerIndex];
        }
    }
    return BM_OK;
}

int32_t ExecuteBatchCopy(HybmBatchCopyParam *param)
{
    auto ret = ValidateFourInputs(param);
    if (ret != BM_OK) {
        return ret;
    }
    const auto *route = reinterpret_cast<const BatchCopyRouteTable *>(ock::mf::HYBM_BATCH_COPY_META_ADDR);
    ret = ValidatePublishedRoute(route);
    if (ret != BM_OK) {
        return ret;
    }
    BatchCopyGroups groups{};
    ret = ValidateAndGroupItems(*param, route, groups);
    if (ret != BM_OK) {
        return ret;
    }
    return SubmitInCompletionRounds(route, groups);
}
} // namespace

extern "C" uint32_t HybmBatchCopy(HybmBatchCopyParam *param)
{
    const auto ret = ExecuteBatchCopy(param);
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "HybmBatchCopy failed, ret=%d", ret);
    }
    return static_cast<uint32_t>(ret);
}
