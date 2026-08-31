/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dl_hcomm_api.h"
#include "hybm_logger.h"
#include "hybm_types.h"
#include "hybm_va_manager.h"
#include "hcomm_transport_manager.h"

namespace ock {
namespace mf {
namespace transport {
namespace urma {

namespace {
CommProtocol ToHcommProtocol(UrmaProtocol protocol)
{
    // 目前支持的通信协议包括：RoCE、UBC_TP、UBC_CTP、UBoE。
    if (protocol == UrmaProtocol::ROCE) {
        return COMM_PROTOCOL_ROCE;
    }
    if (protocol == UrmaProtocol::UBC_TP) {
        return COMM_PROTOCOL_UBC_TP;
    }
    if (protocol == UrmaProtocol::UBC_CTP) {
        return COMM_PROTOCOL_UBC_CTP;
    }
    if (protocol == UrmaProtocol::UBOE) {
        return COMM_PROTOCOL_UBOE;
    }
    return COMM_PROTOCOL_RESERVED;
}
constexpr int32_t HCOMM_CHANNEL_READY = 0;
constexpr int32_t HCOMM_CHANNEL_IN_PROGRESS = 1;
constexpr int32_t HCOMM_CHANNEL_FAILED = 2;
constexpr int32_t HCOMM_CHANNEL_TIMEOUT = 3;
} // namespace

Result HcommTransportManager::WaitForChannelReady(HcommChannelHandle channel, uint32_t peerRank) const
{
    constexpr auto pollInterval = std::chrono::milliseconds(1);

    int32_t channelStatus = HCOMM_CHANNEL_IN_PROGRESS;
    while (channelStatus == HCOMM_CHANNEL_IN_PROGRESS) {
        const auto ret = DlHcommApi::HcommChannelGetStatus(&channel, 1, &channelStatus);
        if (ret != 0) {
            BM_LOG_ERROR("device_urma HcommChannelGetStatus API failed, channel: " << channel << " peer: " << peerRank
                                                                                   << " ret: " << ret);
            return BM_DL_FUNCTION_FAILED;
        }
        if (channelStatus == HCOMM_CHANNEL_IN_PROGRESS) {
            std::this_thread::sleep_for(pollInterval);
        }
    }
    if (channelStatus == HCOMM_CHANNEL_READY) {
        return BM_OK;
    }
    if (channelStatus == HCOMM_CHANNEL_FAILED) {
        BM_LOG_ERROR("device_urma channel FAILED, channel: " << channel << " peer: " << peerRank
                                                             << " status: " << channelStatus);
        return BM_NOT_CONNECTED;
    }
    if (channelStatus == HCOMM_CHANNEL_TIMEOUT) {
        BM_LOG_ERROR("device_urma channel TIMEOUT, channel: " << channel << " peer: " << peerRank
                                                              << " status: " << channelStatus);
        return BM_TIMEOUT;
    }
    BM_LOG_ERROR("device_urma channel unknown status, channel: " << channel << " peer: " << peerRank
                                                                 << " status: " << channelStatus);
    return BM_DL_FUNCTION_FAILED;
}

bool GetRangeEnd(const UrmaCommMem &mem, uint64_t &end)
{
    if (mem.addr == 0 || mem.size == 0) {
        return false;
    }
    if (std::numeric_limits<uint64_t>::max() - mem.addr < mem.size) {
        return false;
    }
    end = mem.addr + mem.size;
    return true;
}

bool IsValidMem(const UrmaCommMem &mem)
{
    uint64_t end = 0;
    return (mem.type == UrmaMemoryType::HOST_DRAM || mem.type == UrmaMemoryType::DEVICE_HBM) && GetRangeEnd(mem, end);
}

EndpointDesc ToHcommEndpointDesc(const UrmaEndpointDesc &desc)
{
    EndpointDesc endpoint{};
    endpoint.protocol = ToHcommProtocol(desc.protocol);
    endpoint.commAddr.type = desc.type;
    std::memcpy(endpoint.commAddr.raws, desc.raws, sizeof(endpoint.commAddr.raws));

    endpoint.loc = desc.loc;
    return endpoint;
}

HcommCommMem ToHcommMem(const UrmaCommMem &mem)
{
    HcommCommMem hcommMem{};
    hcommMem.type = COMM_MEM_TYPE_HOST;
    if (mem.type == UrmaMemoryType::DEVICE_HBM) {
        hcommMem.type = COMM_MEM_TYPE_DEVICE;
    }
    hcommMem.addr = reinterpret_cast<void *>(mem.addr);
    hcommMem.size = mem.size;
    return hcommMem;
}

std::string MakeMemTag(UrmaMemTag memTag)
{
    return std::to_string(memTag);
}

bool SameMem(const UrmaCommMem &left, const UrmaCommMem &right)
{
    return left.addr == right.addr && left.size == right.size && left.type == right.type;
}

bool Overlaps(const UrmaCommMem &left, const UrmaCommMem &right)
{
    uint64_t leftEnd = 0;
    uint64_t rightEnd = 0;
    return left.type == right.type && GetRangeEnd(left, leftEnd) && GetRangeEnd(right, rightEnd) &&
           left.addr < rightEnd && right.addr < leftEnd;
}

bool DeserializeExportDesc(const uint8_t *memDesc, uint32_t descLen, UrmaExportDesc &desc, const uint8_t **hcommDesc,
                           uint32_t *hcommDescLen)
{
    if (memDesc == nullptr || hcommDesc == nullptr || hcommDescLen == nullptr || descLen < sizeof(UrmaExportDesc)) {
        return false;
    }
    std::memcpy(&desc, memDesc, sizeof(desc));
    const UrmaCommMem mem{desc.addr, desc.size, desc.memoryType};
    if (desc.magic != URMA_EXPORT_DESC_MAGIC || desc.version != URMA_EXPORT_DESC_VERSION ||
        desc.headerSize != sizeof(UrmaExportDesc) || !IsValidMem(mem)) {
        return false;
    }
    if (desc.hcommDescLen == 0 || descLen != sizeof(UrmaExportDesc) + desc.hcommDescLen) {
        return false;
    }
    *hcommDesc = memDesc + sizeof(UrmaExportDesc);
    *hcommDescLen = desc.hcommDescLen;
    return true;
}

Result HcomUrmaDestroyEndpoint(HcommEndpointHandle endpoint)
{
    if (endpoint == nullptr) {
        return BM_OK;
    }
    const auto ret = DlHcommApi::HcommEndpointDestroy(endpoint);
    if (ret != 0) {
        BM_LOG_ERROR("urma HcommEndpointDestroy failed, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

UrmaEndpointHandle HcommTransportManager::CreateEndpoint(const UrmaEndpointDesc &desc) const
{
    auto endpoint = std::make_shared<UrmaEndpointEntity>();
    endpoint->desc = desc;
    auto hcommDesc = ToHcommEndpointDesc(desc);
    EndpointHandle handle = nullptr;
    const auto ret = DlHcommApi::HcommEndpointCreate(&hcommDesc, &handle);
    if (ret != 0) {
        BM_LOG_ERROR("urma HcommEndpointCreate failed, ret: " << ret);
        return nullptr;
    }
    endpoint->hcommEndpoint = handle;
    return endpoint;
}

Result HcommTransportManager::HcommMemReg(const UrmaEndpointHandle &endpoint, UrmaMemTag memTag,
                                          const UrmaCommMem &mem, HcommMemHandle *memHandle)
{
    if (endpoint == nullptr || memHandle == nullptr) {
        BM_LOG_ERROR("urma HcommMemReg: endpoint or memHandle is null");
        return BM_INVALID_PARAM;
    }
    if (!IsValidMem(mem)) {
        BM_LOG_ERROR("urma HcommMemReg: invalid memory");
        return BM_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> guard(endpoint->mutex);
    auto tagIt = endpoint->tagIndex.find(memTag);
    if (tagIt != endpoint->tagIndex.end()) {
        auto entryIt = endpoint->memEntries.find(tagIt->second);
        if (entryIt == endpoint->memEntries.end()) {
            BM_LOG_ERROR("urma HcommMemReg: tag index points to non-existent memEntry");
            return BM_ERROR;
        }
        auto &entry = entryIt->second;
        if (!SameMem(entry->mem, mem)) {
            BM_LOG_ERROR("memTag conflict, memTag: " << memTag);
            return BM_ERROR;
        }
        entry->refCount++;
        endpoint->memRef++;
        *memHandle = entry->handle;
        return BM_OK;
    }

    for (const auto &item : endpoint->memEntries) {
        if (item.second != nullptr && Overlaps(item.second->mem, mem)) {
            BM_LOG_ERROR("URMA memory range overlaps an existing MR, addr: " << std::hex << mem.addr
                                                                             << " size: " << mem.size);
            return BM_ERROR;
        }
    }

    const auto hcommMem = ToHcommMem(mem);
    HcommMemHandle hcommHandle = nullptr;
    const auto tag = MakeMemTag(memTag);

    BM_LOG_INFO("urma try to register mem, addr: " << VaToStr(hcommMem.addr) << " size: " << hcommMem.size
                                                   << " memType: " << hcommMem.type);
    int ret = DlHcommApi::HcommMemReg(endpoint->hcommEndpoint, tag.c_str(), &hcommMem, &hcommHandle);
    if (ret != 0) {
        BM_LOG_ERROR("urma HcommMemReg failed, addr: " << std::hex << mem.addr << " size: " << mem.size
                                                       << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    UrmaLocalMr localMr{};
    localMr.mem = mem;
    localMr.hcommMem = hcommHandle;

    try {
        auto entry = std::make_shared<MemEntry>();
        entry->handle = localMr.hcommMem;
        entry->memTag = memTag;
        entry->mem = mem;
        entry->mr = localMr;
        entry->refCount = 1;

        auto entryInserted = endpoint->memEntries.emplace(entry->handle, entry);
        auto tagInserted = endpoint->tagIndex.emplace(memTag, entry->handle);
        if (!entryInserted.second || !tagInserted.second) {
            endpoint->memEntries.erase(entry->handle);
            endpoint->tagIndex.erase(memTag);
            int deregRet = DlHcommApi::HcommMemUnreg(endpoint->hcommEndpoint, localMr.hcommMem);
            if (deregRet != 0) {
                BM_LOG_ERROR("urma HcommMemUnreg rollback failed, ret: " << deregRet);
            }
            return BM_ERROR;
        }

        endpoint->memRef++;
        *memHandle = entry->handle;
        return BM_OK;
    } catch (...) {
        int deregRet = DlHcommApi::HcommMemUnreg(endpoint->hcommEndpoint, localMr.hcommMem);
        if (deregRet != 0) {
            BM_LOG_ERROR("urma HcommMemUnreg rollback failed, ret: " << deregRet);
        }
        return BM_MALLOC_FAILED;
    }
}

Result HcommTransportManager::HcommMemUnreg(const UrmaEndpointHandle &endpoint, HcommMemHandle memHandle)
{
    if (endpoint == nullptr || memHandle == INVALID_MEM_HANDLE) {
        BM_LOG_ERROR("urma HcommMemUnreg: endpoint is null or memHandle is invalid");
        return BM_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> guard(endpoint->mutex);
    auto entryIt = endpoint->memEntries.find(memHandle);
    if (entryIt == endpoint->memEntries.end() || entryIt->second == nullptr) {
        BM_LOG_ERROR("urma HcommMemUnreg: memEntry not found, memHandle " << memHandle);
        return BM_INVALID_PARAM;
    }

    auto entry = entryIt->second;
    if (entry->refCount > 1) {
        entry->refCount--;
        if (endpoint->memRef > 0) {
            endpoint->memRef--;
        }
        return BM_OK;
    }

    const auto ret = DlHcommApi::HcommMemUnreg(endpoint->hcommEndpoint, entry->mr.hcommMem);
    if (ret != 0) {
        BM_LOG_ERROR("urma HcommMemUnreg failed, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }

    endpoint->tagIndex.erase(entry->memTag);
    endpoint->memEntries.erase(entryIt);
    if (endpoint->memRef > 0) {
        endpoint->memRef--;
    }
    return BM_OK;
}

Result HcommTransportManager::HcommMemExport(const UrmaEndpointHandle &endpoint, HcommMemHandle memHandle,
                                             const uint8_t **memDesc, uint32_t *memDescLen)
{
    if (endpoint == nullptr || memDesc == nullptr || memDescLen == nullptr) {
        BM_LOG_ERROR("urma HcommMemExport: endpoint, memDesc, or memDescLen is null");
        return BM_INVALID_PARAM;
    }
    std::lock_guard<std::mutex> guard(endpoint->mutex);
    auto entryIt = endpoint->memEntries.find(memHandle);
    if (entryIt == endpoint->memEntries.end() || entryIt->second == nullptr) {
        return BM_INVALID_PARAM;
    }

    auto entry = entryIt->second;
    if (!entry->exportCacheValid) {
        void *hcommDesc = nullptr;
        uint32_t hcommDescLen = 0;
        BM_LOG_INFO("urma try to export memory addr: " << VaToStr(entry->mem.addr) << " size: " << entry->mem.size);
        auto ret = DlHcommApi::HcommMemExport(endpoint->hcommEndpoint, entry->mr.hcommMem, &hcommDesc, &hcommDescLen);
        if (ret != BM_OK) {
            return ret;
        }
        if (hcommDesc == nullptr || hcommDescLen == 0) {
            BM_LOG_ERROR("urma HcommMemExport: hcommDesc is null or hcommDescLen is 0 after HcommMemExport");
            return BM_ERROR;
        }

        try {
            UrmaExportDesc desc{};
            desc.magic = URMA_EXPORT_DESC_MAGIC;
            desc.version = URMA_EXPORT_DESC_VERSION;
            desc.headerSize = sizeof(UrmaExportDesc);
            desc.memoryType = entry->mem.type;
            desc.memTag = entry->memTag;
            desc.addr = entry->mem.addr;
            desc.size = entry->mem.size;
            desc.hcommDescLen = hcommDescLen;

            std::vector<uint8_t> bytes(sizeof(desc) + hcommDescLen);
            std::memcpy(bytes.data(), &desc, sizeof(desc));
            std::memcpy(bytes.data() + sizeof(desc), hcommDesc, hcommDescLen);
            entry->exportCache.swap(bytes);
            entry->exportCacheValid = true;
        } catch (...) {
            return BM_MALLOC_FAILED;
        }
    }

    *memDesc = entry->exportCache.data();
    *memDescLen = static_cast<uint32_t>(entry->exportCache.size());
    return BM_OK;
}

Result HcommTransportManager::HcommMemImport(const UrmaEndpointHandle &endpoint, const uint8_t *memDesc,
                                             uint32_t descLen, UrmaCommMem *commMem)
{
    if (endpoint == nullptr || memDesc == nullptr || commMem == nullptr) {
        BM_LOG_ERROR("urma HcommMemImport: endpoint, memDesc, or commMem is null");
        return BM_INVALID_PARAM;
    }
    UrmaExportDesc desc{};
    const uint8_t *hcommDesc = nullptr;
    uint32_t hcommDescLen = 0;
    if (!DeserializeExportDesc(memDesc, descLen, desc, &hcommDesc, &hcommDescLen)) {
        return BM_INVALID_PARAM;
    }

    HcommCommMem outMem{};
    {
        std::lock_guard<std::mutex> guard(endpoint->mutex);
        const auto ret = DlHcommApi::HcommMemImport(endpoint->hcommEndpoint, hcommDesc, hcommDescLen, &outMem);
        if (ret != 0) {
            BM_LOG_ERROR("urma HcommMemImport failed, ret: " << ret);
            return BM_DL_FUNCTION_FAILED;
        }
    }

    BM_LOG_INFO("urma import memory returned outMem (addr=" << VaToStr(outMem.addr) << " size=" << outMem.size
                                                            << " type=" << outMem.type << ")");
    UrmaMemoryType viewType =
        endpoint->desc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE ? UrmaMemoryType::DEVICE_HBM : desc.memoryType;
    if (outMem.type == COMM_MEM_TYPE_DEVICE) {
        viewType = UrmaMemoryType::DEVICE_HBM;
    } else if (outMem.type == COMM_MEM_TYPE_HOST) {
        viewType = UrmaMemoryType::HOST_DRAM;
    }
    UrmaCommMem view{reinterpret_cast<uint64_t>(outMem.addr), outMem.size, viewType};
    if (!IsValidMem(view)) {
        BM_LOG_ERROR("urma HcommMemImport returned invalid view (addr="
                     << std::hex << view.addr << " size=" << view.size << " type=" << view.type << std::dec << ")");
        return BM_DL_FUNCTION_FAILED;
    }
    *commMem = view;
    return BM_OK;
}

Result HcommTransportManager::HcommMemUnimport(const UrmaEndpointHandle &endpoint, const uint8_t *memDesc,
                                               uint32_t descLen)
{
    if (endpoint == nullptr || memDesc == nullptr) {
        BM_LOG_ERROR("urma HcommMemUnimport: endpoint or memDesc is null");
        return BM_INVALID_PARAM;
    }
    UrmaExportDesc desc{};
    const uint8_t *hcommDesc = nullptr;
    uint32_t hcommDescLen = 0;
    if (!DeserializeExportDesc(memDesc, descLen, desc, &hcommDesc, &hcommDescLen)) {
        return BM_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> guard(endpoint->mutex);
    const auto ret = DlHcommApi::HcommMemUnimport(endpoint->hcommEndpoint, hcommDesc, hcommDescLen);
    if (ret != 0) {
        BM_LOG_ERROR("urma HcommMemUnimport failed, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

} // namespace urma
} // namespace transport
} // namespace mf
} // namespace ock
