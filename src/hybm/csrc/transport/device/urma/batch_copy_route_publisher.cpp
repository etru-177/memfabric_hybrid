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

#include <algorithm>
#include <limits>
#include <memory>
#include <new>

#include "batch_copy_route_publisher.h"
#include "dl_acl_api.h"
#include "hybm_define.h"
#include "hybm_logger.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {

namespace {
constexpr uint64_t BATCH_COPY_COMPLETION_ADDR = HYBM_BATCH_COPY_META_ADDR + BATCH_COPY_COMPLETION_OFFSET;
constexpr urma::UrmaMemTag BATCH_COPY_COMPLETION_MEM_TAG = BATCH_COPY_COMPLETION_ADDR;

Result CopyHostToDevice(uint64_t destination, const void *source, size_t size, const char *operation)
{
    const auto ret =
        DlAclApi::AclrtMemcpy(reinterpret_cast<void *>(destination), size, source, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != BM_OK) {
        BM_LOG_ERROR(operation << " failed, ret: " << ret << " dst: 0x" << std::hex << destination << std::dec
                               << " size: " << size);
    }
    return ret;
}

void LogRouteTable(uint32_t userDeviceId, const BatchCopyRouteTable &table)
{
    BM_LOG_INFO("BatchCopy route table, userDeviceId: "
                << userDeviceId << " magic: 0x" << std::hex << table.header.magic << std::dec
                << " peerCount: " << table.header.peerCount << " rangeCount: " << table.header.rangeCount);
    for (uint16_t peerIndex = 0; peerIndex < table.header.peerCount; ++peerIndex) {
        const auto &peer = table.peers[peerIndex];
        BM_LOG_INFO("BatchCopy route peer, userDeviceId: " << userDeviceId << " peerIndex: " << peerIndex
                                                           << " thread: " << peer.thread << " channel: " << peer.channel
                                                           << " remoteFlagAddr: 0x" << std::hex << peer.remoteFlagAddr
                                                           << std::dec);
    }
    for (uint16_t rangeIndex = 0; rangeIndex < table.header.rangeCount; ++rangeIndex) {
        const auto &range = table.ranges[rangeIndex];
        BM_LOG_INFO("BatchCopy route range, userDeviceId: "
                    << userDeviceId << " rangeIndex: " << rangeIndex << " peerIndex: " << range.peerIndex
                    << " srcGvaBegin: 0x" << std::hex << range.srcGvaBegin << " srcGvaEnd: 0x" << range.srcGvaEnd
                    << " hcommVaBegin: 0x" << range.hcommVaBegin << std::dec);
    }
}
} // namespace

BatchCopyRoutePublisher::BatchCopyRoutePublisher(uint32_t userDeviceId, const urma::UrmaEndpointHandle &localEndpoint,
                                                 urma::HcommTransportManager &hcommManager)
    : userDeviceId_(userDeviceId), localEndpoint_(localEndpoint), hcommManager_(hcommManager)
{}

BatchCopyRoutePublisher::~BatchCopyRoutePublisher()
{
    const auto ret = Clear();
    if (ret != BM_OK) {
        BM_LOG_ERROR("clear BatchCopy route in destructor failed, ret: " << ret << " userDeviceId: " << userDeviceId_);
    }
}

Result BatchCopyRoutePublisher::ValidatePeer(const std::vector<BatchCopyRouteSource> &sources, size_t peerIndex) const
{
    const auto &source = sources[peerIndex];
    if (source.thread == 0 || source.channel == 0 || source.remoteFlagAddr == 0 || source.ranges.empty() ||
        source.ranges.size() > BATCH_COPY_MAX_RANGE_PER_PEER) {
        BM_LOG_ERROR("invalid BatchCopy peer source, userDeviceId: "
                     << userDeviceId_ << " peerRank: " << source.peerRank << " peerIndex: " << peerIndex
                     << " thread: " << source.thread << " channel: " << source.channel << " remoteFlagAddr: 0x"
                     << std::hex << source.remoteFlagAddr << std::dec << " rangeCount: " << source.ranges.size());
        return BM_INVALID_PARAM;
    }
    for (size_t previous = 0U; previous < peerIndex; ++previous) {
        if (sources[previous].peerRank == source.peerRank) {
            BM_LOG_ERROR("duplicate BatchCopy peer rank, userDeviceId: " << userDeviceId_
                                                                         << " peerRank: " << source.peerRank);
            return BM_INVALID_PARAM;
        }
    }
    return BM_OK;
}

Result BatchCopyRoutePublisher::CollectRanges(const BatchCopyRouteSource &source, SourceRangeArray &ranges,
                                              size_t &rangeCount) const
{
    for (const auto &range : source.ranges) {
        if (range.srcGvaBegin >= range.srcGvaEnd || range.hcommVaBegin == 0U || rangeCount >= ranges.size()) {
            BM_LOG_ERROR("invalid BatchCopy source range, userDeviceId: "
                         << userDeviceId_ << " peerRank: " << source.peerRank << " srcGvaBegin: 0x" << std::hex
                         << range.srcGvaBegin << " srcGvaEnd: 0x" << range.srcGvaEnd << " hcommVaBegin: 0x"
                         << range.hcommVaBegin << std::dec << " collectedRangeCount: " << rangeCount);
            return BM_INVALID_PARAM;
        }
        const uint64_t length = range.srcGvaEnd - range.srcGvaBegin;
        if (range.hcommVaBegin > std::numeric_limits<uint64_t>::max() - length) {
            BM_LOG_ERROR("BatchCopy HCOMM range overflows, userDeviceId: "
                         << userDeviceId_ << " peerRank: " << source.peerRank << " hcommVaBegin: 0x" << std::hex
                         << range.hcommVaBegin << " length: 0x" << length << std::dec);
            return BM_INVALID_PARAM;
        }
        ranges[rangeCount++] = range;
    }
    return BM_OK;
}

Result BatchCopyRoutePublisher::ValidateSortedRanges(SourceRangeArray &ranges, size_t rangeCount) const
{
    std::sort(ranges.begin(), ranges.begin() + rangeCount,
              [](const auto &left, const auto &right) { return left.srcGvaBegin < right.srcGvaBegin; });
    for (size_t index = 1U; index < rangeCount; ++index) {
        if (ranges[index].srcGvaBegin < ranges[index - 1U].srcGvaEnd) {
            BM_LOG_ERROR("overlapping BatchCopy source ranges, userDeviceId: "
                         << userDeviceId_ << " srcGvaBegin: 0x" << std::hex << ranges[index].srcGvaBegin
                         << " previousEnd: 0x" << ranges[index - 1U].srcGvaEnd << std::dec);
            return BM_INVALID_PARAM;
        }
    }
    return BM_OK;
}

Result BatchCopyRoutePublisher::ValidateSources(const std::vector<BatchCopyRouteSource> &sources) const
{
    if (sources.empty() || sources.size() > BATCH_COPY_MAX_PEER_COUNT) {
        BM_LOG_ERROR("invalid BatchCopy peer count: " << sources.size() << " userDeviceId: " << userDeviceId_);
        return BM_INVALID_PARAM;
    }
    auto ranges = std::unique_ptr<SourceRangeArray>(new (std::nothrow) SourceRangeArray{});
    if (ranges == nullptr) {
        BM_LOG_ERROR("allocate BatchCopy source ranges failed, userDeviceId: " << userDeviceId_
                                                                                << " rangeCapacity: "
                                                                                << BATCH_COPY_MAX_RANGE_COUNT);
        return BM_ERROR;
    }
    size_t rangeCount = 0U;
    for (size_t peerIndex = 0U; peerIndex < sources.size(); ++peerIndex) {
        auto ret = ValidatePeer(sources, peerIndex);
        if (ret != BM_OK) {
            return ret;
        }
        ret = CollectRanges(sources[peerIndex], *ranges, rangeCount);
        if (ret != BM_OK) {
            return ret;
        }
    }
    return ValidateSortedRanges(*ranges, rangeCount);
}

void BatchCopyRoutePublisher::BuildRouteImage(const std::vector<BatchCopyRouteSource> &sources,
                                              BatchCopyRouteTable &table) const
{
    table.header = BatchCopyRouteHeader{};
    std::fill_n(table.peers, BATCH_COPY_MAX_PEER_COUNT, BatchCopyPeerEntry{});
    std::fill_n(table.ranges, BATCH_COPY_MAX_RANGE_COUNT, BatchCopyRangeEntry{});
    size_t rangeIndex = 0U;
    for (size_t peerIndex = 0U; peerIndex < sources.size(); ++peerIndex) {
        const auto &source = sources[peerIndex];
        table.peers[peerIndex].thread = source.thread;
        table.peers[peerIndex].channel = source.channel;
        table.peers[peerIndex].remoteFlagAddr = source.remoteFlagAddr;
        for (const auto &range : source.ranges) {
            auto &entry = table.ranges[rangeIndex++];
            entry.srcGvaBegin = range.srcGvaBegin;
            entry.srcGvaEnd = range.srcGvaEnd;
            entry.hcommVaBegin = range.hcommVaBegin;
            entry.peerIndex = static_cast<uint16_t>(peerIndex);
        }
    }
    std::sort(table.ranges, table.ranges + rangeIndex,
              [](const auto &left, const auto &right) { return left.srcGvaBegin < right.srcGvaBegin; });
    table.header.peerCount = static_cast<uint16_t>(sources.size());
    table.header.rangeCount = static_cast<uint16_t>(rangeIndex);
}

Result BatchCopyRoutePublisher::ClearMagic()
{
    const uint32_t magic = 0U;
    return CopyHostToDevice(HYBM_BATCH_COPY_META_ADDR, &magic, sizeof(magic), "clear BatchCopy route magic");
}

Result BatchCopyRoutePublisher::ClearCompletionArea()
{
    const BatchCopyCompletionArea completion{};
    return CopyHostToDevice(BATCH_COPY_COMPLETION_ADDR, &completion, sizeof(completion),
                            "clear BatchCopy completion area");
}

Result BatchCopyRoutePublisher::RegisterCompletionArea()
{
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("register BatchCopy completion failed, local endpoint is null, userDeviceId: " << userDeviceId_);
        return BM_NOT_INITIALIZED;
    }
    const urma::UrmaCommMem memory{BATCH_COPY_COMPLETION_ADDR, sizeof(BatchCopyCompletionArea),
                                   urma::UrmaMemoryType::DEVICE_HBM};
    const auto ret =
        hcommManager_.HcommMemReg(localEndpoint_, BATCH_COPY_COMPLETION_MEM_TAG, memory, &completionHandle_);
    if (ret != BM_OK) {
        BM_LOG_ERROR("register BatchCopy completion failed, ret: "
                     << ret << " userDeviceId: " << userDeviceId_ << " addr: 0x" << std::hex
                     << BATCH_COPY_COMPLETION_ADDR << std::dec << " size: " << sizeof(BatchCopyCompletionArea));
    }
    return ret;
}

Result BatchCopyRoutePublisher::UnregisterCompletionArea()
{
    if (completionHandle_ == nullptr) {
        return BM_OK;
    }
    const auto ret = hcommManager_.HcommMemUnreg(localEndpoint_, completionHandle_);
    if (ret != BM_OK) {
        BM_LOG_ERROR("unregister BatchCopy completion failed, ret: " << ret << " userDeviceId: " << userDeviceId_
                                                                     << " handle: " << completionHandle_);
        return ret;
    }
    completionHandle_ = nullptr;
    return BM_OK;
}

Result BatchCopyRoutePublisher::WriteRouteImage(const BatchCopyRouteTable &table)
{
    if (table.header.magic != 0U) {
        BM_LOG_ERROR("BatchCopy route image magic must be zero, userDeviceId: " << userDeviceId_
                                                                                << " magic: " << table.header.magic);
        return BM_INVALID_PARAM;
    }
    return CopyHostToDevice(HYBM_BATCH_COPY_META_ADDR, &table, sizeof(table), "write BatchCopy route image");
}

Result BatchCopyRoutePublisher::PublishMagic()
{
    const uint32_t magic = BATCH_COPY_ROUTE_MAGIC;
    return CopyHostToDevice(HYBM_BATCH_COPY_META_ADDR, &magic, sizeof(magic), "publish BatchCopy route magic");
}

Result BatchCopyRoutePublisher::PublishRouteImage(const std::vector<BatchCopyRouteSource> &sources)
{
    auto ret = ClearMagic();
    if (ret != BM_OK) {
        BM_LOG_ERROR("ClearMagic failed: " << ret);
        return ret;
    }
    ret = ClearCompletionArea();
    if (ret != BM_OK) {
        BM_LOG_ERROR("ClearCompletionArea failed: " << ret);
        return ret;
    }
    ret = RegisterCompletionArea();
    if (ret != BM_OK) {
        BM_LOG_ERROR("RegisterCompletionArea failed: " << ret);
        return ret;
    }
    auto table = std::unique_ptr<BatchCopyRouteTable>(new (std::nothrow) BatchCopyRouteTable{});
    if (table == nullptr) {
        BM_LOG_ERROR("allocate BatchCopy route image failed, userDeviceId: " << userDeviceId_
                                                                              << " imageSize: "
                                                                              << sizeof(BatchCopyRouteTable));
        return BM_ERROR;
    }
    BuildRouteImage(sources, *table);
    ret = WriteRouteImage(*table);
    if (ret != BM_OK) {
        BM_LOG_ERROR("WriteRouteImage failed: " << ret);
        return ret;
    }
    LogRouteTable(userDeviceId_, *table);
    return PublishMagic();
}

void BatchCopyRoutePublisher::RollbackPublish()
{
    published_ = false;
    const auto ret = Clear();
    if (ret != BM_OK) {
        BM_LOG_ERROR("rollback BatchCopy route failed, ret: " << ret << " userDeviceId: " << userDeviceId_);
    }
}

Result BatchCopyRoutePublisher::Publish(const std::vector<BatchCopyRouteSource> &sources)
{
    if (published_) {
        BM_LOG_DEBUG("already published_");
        return BM_OK;
    }
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("publish BatchCopy route failed, local endpoint is null, userDeviceId: " << userDeviceId_);
        return BM_NOT_INITIALIZED;
    }
    if (cleanupRequired_ || completionHandle_ != nullptr) {
        BM_LOG_ERROR("BatchCopy route has pending cleanup, userDeviceId: " << userDeviceId_ << " completionHandle: "
                                                                           << completionHandle_);
        return BM_ERROR;
    }
    auto ret = ValidateSources(sources);
    if (ret != BM_OK) {
        BM_LOG_ERROR("ValidateSources failed : " << ret);
        return ret;
    }
    cleanupRequired_ = true;
    ret = PublishRouteImage(sources);
    if (ret != BM_OK) {
        RollbackPublish();
        return ret;
    }
    published_ = true;
    return BM_OK;
}

Result BatchCopyRoutePublisher::Clear()
{
    if (!cleanupRequired_ && completionHandle_ == nullptr) {
        published_ = false;
        return BM_OK;
    }
    auto ret = ClearMagic();
    if (ret != BM_OK) {
        return ret;
    }
    published_ = false;
    ret = UnregisterCompletionArea();
    if (ret != BM_OK) {
        return ret;
    }
    cleanupRequired_ = false;
    return BM_OK;
}

bool BatchCopyRoutePublisher::IsPublished() const
{
    return published_;
}

} // namespace device
} // namespace transport
} // namespace mf
} // namespace ock
