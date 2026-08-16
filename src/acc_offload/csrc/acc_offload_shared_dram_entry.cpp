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
#include "hybm_big_mem.h"
#include "smem_net_common.h"
#include "smem_store_factory.h"
#include "acc_offload_launch.h"
#include "acc_offload_shared_dram_entry.h"

namespace ock {
namespace offload {

using namespace ock::smem;

constexpr uint64_t GB = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t HOST_MEM_SLICE_SIZE = 32ULL * GB;

static uint64_t AlignUp(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1) & ~(align - 1);
}

static int32_t AllGatherAndImportPeers(const SmemGroupEnginePtr &group,
                                       const std::vector<hybm_exchange_info> &localInfos, hybm_entity_t entity,
                                       uint32_t importFlags, uint32_t rankCount, uint32_t selfRank)
{
    uint32_t infoCount = static_cast<uint32_t>(localInfos.size());
    std::vector<uint32_t> allCounts(rankCount, 0);
    auto ret = group->GroupAllGather(reinterpret_cast<const char *>(&infoCount), sizeof(uint32_t),
                                     reinterpret_cast<char *>(allCounts.data()), sizeof(uint32_t) * rankCount);
    if (ret != OFFLOAD_OK) {
        return ret;
    }
    uint32_t maxCount = *std::max_element(allCounts.cbegin(), allCounts.cend());
    if (maxCount == 0) {
        return group->GroupBarrier();
    }
    std::vector<hybm_exchange_info> sendInfos(maxCount);
    for (uint32_t i = 0; i < infoCount; i++) {
        sendInfos[i] = localInfos[i];
    }
    std::vector<hybm_exchange_info> allInfos(static_cast<size_t>(rankCount) * maxCount);
    ret = group->GroupAllGather(reinterpret_cast<const char *>(sendInfos.data()), maxCount * sizeof(hybm_exchange_info),
                                reinterpret_cast<char *>(allInfos.data()),
                                rankCount * maxCount * sizeof(hybm_exchange_info));
    if (ret != OFFLOAD_OK) {
        return ret;
    }
    std::vector<hybm_exchange_info> peerInfos;
    peerInfos.reserve(static_cast<size_t>(rankCount > 0 ? rankCount - 1 : 0) * maxCount);
    for (uint32_t r = 0; r < rankCount; r++) {
        if (r == selfRank) {
            continue;
        }
        for (uint32_t i = 0; i < maxCount; i++) {
            const auto &info = allInfos[static_cast<size_t>(r) * maxCount + i];
            if (info.descLen > 0) {
                peerInfos.push_back(info);
            }
        }
    }
    if (!peerInfos.empty()) {
        ret = hybm_import(entity, peerInfos.data(), peerInfos.size(), nullptr, importFlags);
        if (ret != OFFLOAD_OK) {
            return ret;
        }
    }
    return group->GroupBarrier();
}

int32_t AccOffloadSharedDramEntry::AllocAndExportHostSlices()
{
    const uint64_t totalSize = options_.hostVASpace;
    uint64_t remaining = totalSize;
    do {
        uint64_t sliceSize = (remaining >= HOST_MEM_SLICE_SIZE) ? HOST_MEM_SLICE_SIZE : remaining;
        uint64_t allocated = totalSize - remaining;
        OFFLOAD_LOG_INFO("alloc host slice progress: " << allocated << "/" << totalSize << ", sliceSize: " << sliceSize
                                                       << ", rankId: " << options_.rankId);
        auto memSlice = hybm_alloc_local_memory(entity_, HYBM_MEM_TYPE_HOST, sliceSize, 0);
        if (memSlice == nullptr) {
            OFFLOAD_LOG_ERROR("alloc host slice failed, allocated: " << allocated << ", sliceSize: " << sliceSize
                                                                     << ", totalSize: " << totalSize
                                                                     << ", rankId: " << options_.rankId);
            return OFFLOAD_ERROR;
        }
        hybm_exchange_info sliceInfo{};
        auto ret = hybm_export(entity_, memSlice, 0, &sliceInfo);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("export host slice failed, result: " << ret << ", sliceSize: " << sliceSize
                                                                   << ", rankId: " << options_.rankId);
            return ret;
        }
        slices_.push_back(memSlice);
        sliceInfos_.push_back(sliceInfo);
        remaining -= sliceSize;
    } while (remaining > 0);
    OFFLOAD_LOG_INFO("alloc and export host slices done, sliceCount: " << sliceInfos_.size() << ", totalSize: "
                                                                       << totalSize << ", rankId: " << options_.rankId);
    return OFFLOAD_OK;
}

int32_t AccOffloadSharedDramEntry::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (inited_) {
        return OFFLOAD_OK;
    }

    if (config.worldSize == 0) {
        OFFLOAD_LOG_ERROR("shared dram world size is 0");
        return OFFLOAD_ERROR;
    }

    int32_t ret = OFFLOAD_OK;
    do {
        ret = hybm_init(config.deviceId, 0);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("hybm_init failed, result: " << ret);
            break;
        }

        ret = AccOffloadLaunchApi::TryLoadLibrary();
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("offload launch load library failed");
            break;
        }

        constexpr int portBase = 8500;
        int port = portBase + config.deviceId / config.worldSize;
        storeUrl_ = "tcp://127.0.0.1:" + std::to_string(port);

        UrlExtraction extraction;
        if (extraction.ExtractIpPortFromUrl(storeUrl_) != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("extract ip port from url failed, storeUrl: " << storeUrl_);
            ret = OFFLOAD_ERROR;
            break;
        }

        smem_tls_config tlsCfg{};
        StoreFactory::SetTlsInfo(tlsCfg);
        bool startServer = (config.rankId == 0);
        uint16_t model = startServer ? CSM_BOTH : CSM_CLIENT;
        auto baseStore = StoreFactory::CreateStoreByUrl(storeUrl_, model, config.worldSize, config.rankId);
        if (baseStore == nullptr) {
            OFFLOAD_LOG_ERROR("create store failed, storeUrl: " << storeUrl_);
            ret = OFFLOAD_ERROR;
            break;
        }

        auto prefix = "(" + std::to_string(HYBM_ENTITY_ID_OFFLOAD_BASE) + ")_";
        auto offloadStore = StoreFactory::PrefixStore(baseStore, "OFFLOAD_");
        entryStore_ = StoreFactory::PrefixStore(offloadStore, prefix);
        if (entryStore_ == nullptr) {
            OFFLOAD_LOG_ERROR("create prefix store failed, storeUrl: " << storeUrl_);
            ret = OFFLOAD_ERROR;
            break;
        }

        SmemGroupOption groupOpt = {
            config.worldSize, config.rankId, SMEM_DEFAUT_WAIT_TIME * SECOND_TO_MILLSEC, false, nullptr, nullptr,
            nullptr,          nullptr};
        group_ = SmemNetGroupEngine::Create(entryStore_, groupOpt);
        if (group_ == nullptr) {
            OFFLOAD_LOG_ERROR("create net group failed, rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }
        ret = group_->GroupBarrier();
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("group barrier after create failed, result: " << ret << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        uint64_t alignedReserveSize = AlignUp(config.reserveSize, GB);
        uint64_t alignedAllocSize = AlignUp(config.allocSize, GB);
        options_ = {};
        options_.bmType = HYBM_TYPE_HOST_INITIATE;
        options_.memType = HYBM_MEM_TYPE_HOST;
        options_.bmDataOpType = HYBM_DOP_TYPE_MTE;
        options_.rankCount = config.worldSize;
        options_.rankId = config.rankId;
        options_.devId = config.deviceId;
        options_.maxDRAMSize = alignedReserveSize;
        options_.hostVASpace = alignedAllocSize;
        options_.role = HYBM_ROLE_PEER;
        options_.scene = HYBM_SCENE_DEFAULT;
        options_.flags = HYBM_FLAG_DRAM_MAP_HOST_VA | HYBM_FLAG_UNRESTRICTED_MEM;
        options_.dramShmFd = -1;
        options_.enable56BitsGva = false;
        bzero(options_.transUrl, sizeof(options_.transUrl));
        if (storeUrl_.size() >= sizeof(options_.transUrl)) {
            OFFLOAD_LOG_ERROR("store url too long, storeUrl: " << storeUrl_);
            ret = OFFLOAD_ERROR;
            break;
        }
        std::copy_n(storeUrl_.c_str(), storeUrl_.size(), options_.transUrl);

        entity_ = hybm_create_entity(HYBM_ENTITY_ID_OFFLOAD_BASE, &options_, 0);
        if (entity_ == nullptr) {
            OFFLOAD_LOG_ERROR("create entity failed, rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        ret = hybm_reserve_mem_space(entity_, 0);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("reserve mem failed, result: " << ret << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        ret = AllocAndExportHostSlices();
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("alloc and export host slices failed, hostVASpace: " << options_.hostVASpace
                                                                                   << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }
        ret = AllGatherAndImportPeers(group_, sliceInfos_, entity_, 0, options_.rankCount, options_.rankId);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("allgather/import slice info failed, result: " << ret << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        hybm_exchange_info entityInfo{};
        ret = hybm_export(entity_, nullptr, HYBM_FLAG_EXPORT_ENTITY, &entityInfo);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("export entity failed, result: " << ret << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }
        if (entityInfo.descLen > 0) {
            std::vector<hybm_exchange_info> entityInfos{entityInfo};
            ret = AllGatherAndImportPeers(group_, entityInfos, entity_, HYBM_FLAG_EXPORT_ENTITY, options_.rankCount,
                                          options_.rankId);
            if (ret != OFFLOAD_OK) {
                OFFLOAD_LOG_ERROR("allgather/import entity info failed, result: " << ret
                                                                                  << ", rankId: " << config.rankId);
                ret = OFFLOAD_ERROR;
                break;
            }
        }

        ret = hybm_mmap(entity_, 0);
        if (ret != OFFLOAD_OK) {
            OFFLOAD_LOG_ERROR("hybm mmap failed, result: " << ret << ", rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }

        hostGva_ = hybm_get_memory_ptr(entity_, HYBM_MEM_TYPE_HOST);
        if (hostGva_ == nullptr) {
            OFFLOAD_LOG_ERROR("get host gva failed, rankId: " << config.rankId);
            ret = OFFLOAD_ERROR;
            break;
        }
        base_ = reinterpret_cast<uint8_t *>(hostGva_) + options_.maxDRAMSize * options_.rankId;
        size_ = options_.maxDRAMSize;

        memMng_ = std::make_shared<AccOffloadMemManager>(base_, size_);
        if (memMng_ == nullptr) {
            OFFLOAD_LOG_ERROR("create mem manager failed");
            ret = OFFLOAD_ERROR;
            break;
        }
    } while (0);

    inited_ = true;
    if (ret != OFFLOAD_OK) {
        UnInitialize();
        return ret;
    }

    OFFLOAD_LOG_INFO("shared dram entry initialized, rankId: " << config.rankId << ", deviceId: " << config.deviceId
                                                               << ", base: " << reinterpret_cast<void *>(base_)
                                                               << ", size: " << size_);
    return OFFLOAD_OK;
}

void AccOffloadSharedDramEntry::UnInitialize()
{
    if (!inited_) {
        return;
    }

    uint32_t flags = 0;
    memMng_.reset();
    if (group_ != nullptr) {
        group_->GroupSnClean();
        group_ = nullptr;
    }
    if (entity_ != nullptr) {
        for (auto slice : slices_) {
            hybm_free_local_memory(entity_, slice, 1, flags);
        }
        slices_.clear();
        sliceInfos_.clear();
        hybm_unreserve_mem_space(entity_, flags);
        hybm_destroy_entity(entity_, flags);
        entity_ = nullptr;
    }
    entryStore_ = nullptr;
    if (!storeUrl_.empty()) {
        StoreFactory::DestroyStore(storeUrl_);
        storeUrl_.clear();
    }
    AccOffloadLaunchApi::CleanupLibrary();
    hybm_uninit();

    base_ = nullptr;
    hostGva_ = nullptr;
    size_ = 0;
    inited_ = false;
}

void *AccOffloadSharedDramEntry::MallocHost(size_t size)
{
    if (memMng_ == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr, malloc failed");
        return nullptr;
    }

    OFFLOAD_LOG_DEBUG("shared malloc host size: " << size);
    return memMng_->Allocate(size);
}

void AccOffloadSharedDramEntry::FreeHost(void *ptr)
{
    if (memMng_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr or ptr is nullptr, free failed");
        return;
    }

    OFFLOAD_LOG_DEBUG("shared free host ptr: " << reinterpret_cast<uint64_t>(ptr));
    memMng_->Release(ptr);
}

int32_t AccOffloadSharedDramEntry::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                              uint32_t *sizePtr, uint8_t devIdx, uint32_t flag)
{
    OFFLOAD_LOG_DEBUG("shared sparse copy, src: " << reinterpret_cast<uint64_t>(srcPtrs)
                                                  << ", dst: " << reinterpret_cast<uint64_t>(dstPtrs) << ", len: "
                                                  << reinterpret_cast<uint64_t>(lenPtrs) << ", size: " << *sizePtr
                                                  << ", devIdx: " << devIdx << ", flag: " << flag);
    return AccOffloadLaunchApi::AccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx, flag);
}

int32_t AccOffloadSharedDramEntry::GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                                 uint32_t *numLocalExpertPtr, int64_t *groupList,
                                                 int64_t *packedGroupList, uint8_t devIdx)
{
    return AccOffloadLaunchApi::AccOffloadGroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList,
                                                        packedGroupList, devIdx);
}

} // namespace offload
} // namespace ock
