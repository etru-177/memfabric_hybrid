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
#include <thread>
#include <vector>
#include "hybm_big_mem.h"
#include "hybm_data_op.h"
#include "acc_offload_local_urma_entry.h"

namespace ock {
namespace offload {

constexpr uint64_t KB = 1024ULL;
constexpr uint64_t MB = KB * 1024ULL;
constexpr uint64_t GB = MB * 1024ULL;

constexpr uint32_t REMOTE_DEVICE_RANK_ID = 0;

static uint64_t AlignUp(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1) & ~(align - 1);
}

int32_t AccOffloadLocalUrmaEntry::CreateHostEntity(const hybm_options &opts, uint32_t flags,
                                                   hybm_exchange_info *entityInfo, hybm_exchange_info *sliceInfo)
{
    entity1_ = hybm_create_entity(HYBM_ENTITY_ID_OFFLOAD_BASE, &opts, flags);
    if (entity1_ == nullptr) {
        OFFLOAD_LOG_ERROR("create host urma entity failed, bmDataOpType: " << opts.bmDataOpType);
        return OFFLOAD_ERROR;
    }
    int32_t ret = hybm_reserve_mem_space(entity1_, flags);
    if (ret != 0) {
        OFFLOAD_LOG_ERROR("reserve host urma mem failed, result: " << ret << ", maxDRAMSize: " << opts.maxDRAMSize);
        return OFFLOAD_ERROR;
    }
    if (opts.maxDRAMSize > 0) {
        slice_ = hybm_alloc_local_memory(entity1_, HYBM_MEM_TYPE_HOST, opts.hostVASpace, flags);
        if (slice_ == nullptr) {
            OFFLOAD_LOG_ERROR("alloc local host mem failed, size: " << opts.hostVASpace);
            return OFFLOAD_ERROR;
        }
    }
    base_ = reinterpret_cast<uint8_t *>(hybm_get_slice_va(entity1_, slice_));
    if (base_ == nullptr) {
        OFFLOAD_LOG_ERROR("get slice va failed, hostVASpace: " << opts.hostVASpace);
        return OFFLOAD_ERROR;
    }
    ret = hybm_export(entity1_, nullptr, HYBM_FLAG_EXPORT_ENTITY, entityInfo);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("export host urma entity info failed, result: " << ret);
        return OFFLOAD_ERROR;
    }
    ret = hybm_export(entity1_, slice_, 0, sliceInfo);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("export host urma slice info failed, result: " << ret);
        return OFFLOAD_ERROR;
    }
    return OFFLOAD_OK;
}

int32_t AccOffloadLocalUrmaEntry::CreateDeviceEntity(const hybm_options &opts, uint32_t flags,
                                                     hybm_exchange_info *entityInfo)
{
    entity2_ = hybm_create_entity(HYBM_ENTITY_ID_OFFLOAD_BASE + 1, &opts, flags);
    if (entity2_ == nullptr) {
        OFFLOAD_LOG_ERROR("create device urma entity failed, bmDataOpType: " << opts.bmDataOpType);
        return OFFLOAD_ERROR;
    }
    int32_t ret = hybm_reserve_mem_space(entity2_, flags);
    if (ret != 0) {
        OFFLOAD_LOG_ERROR("reserve device urma mem failed, result: " << ret << ", maxDRAMSize: " << opts.maxDRAMSize);
        return OFFLOAD_ERROR;
    }
    ret = hybm_export(entity2_, nullptr, HYBM_FLAG_EXPORT_ENTITY, entityInfo);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("export device urma entity info failed, result: " << ret);
        return OFFLOAD_ERROR;
    }
    return OFFLOAD_OK;
}

int32_t AccOffloadLocalUrmaEntry::CrossImportAndMap(uint32_t flags, const hybm_exchange_info &hostEntityInfo,
                                                    const hybm_exchange_info &devEntityInfo,
                                                    const hybm_exchange_info &hostSliceInfo)
{
    int32_t hostRet = OFFLOAD_OK;
    int32_t devRet = OFFLOAD_OK;
    std::thread hostThread([&]() {
        hostRet = hybm_import(entity1_, &devEntityInfo, 1, nullptr, HYBM_FLAG_EXPORT_ENTITY);
    });
    std::thread devThread([&]() {
        devRet = hybm_import(entity2_, &hostEntityInfo, 1, nullptr, HYBM_FLAG_EXPORT_ENTITY);
    });
    hostThread.join();
    devThread.join();
    if (hostRet != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("host urma import device entity info failed, result: " << hostRet);
        return OFFLOAD_ERROR;
    }
    if (devRet != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("device urma import host entity info failed, result: " << devRet);
        return OFFLOAD_ERROR;
    }
    OFFLOAD_LOG_INFO("import entity info success.");

    int32_t ret = hybm_import(entity2_, &hostSliceInfo, 1, nullptr, 0);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("device urma import host slice info failed, result: " << ret);
        return OFFLOAD_ERROR;
    }
    ret = hybm_mmap(entity2_, flags);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("device urma mmap failed, result: " << ret);
        return OFFLOAD_ERROR;
    }
    return OFFLOAD_OK;
}

hybm_options AccOffloadLocalUrmaEntry::BuildHostOptions(const offload_config_t &config, uint64_t reserveSize) const
{
    hybm_options opts{};
    opts.bmType = HYBM_TYPE_HOST_INITIATE;
    opts.memType = HYBM_MEM_TYPE_HOST;
    opts.bmDataOpType = HYBM_DOP_TYPE_HOST_URMA;
    opts.rankCount = 2;
    opts.rankId = 0;
    opts.devId = config.deviceId;
    opts.maxDRAMSize = reserveSize;
    opts.hostVASpace = reserveSize;
    opts.scene = HYBM_SCENE_DEFAULT;
    opts.flags = 0;
    opts.dramShmFd = -1;
    return opts;
}

int32_t AccOffloadLocalUrmaEntry::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (inited_) {
        return OFFLOAD_OK;
    }

    uint32_t flags = 0;
    int32_t ret = hybm_init(config.deviceId, flags);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("init hybm failed, result: " << ret << ", deviceId: " << config.deviceId);
        hybm_uninit();
        return OFFLOAD_ERROR;
    }

    uint64_t alignedReserveSize = AlignUp(config.reserveSize, GB);
    uint64_t alignedAllocSize = AlignUp(config.allocSize, GB);
    if (alignedReserveSize != alignedAllocSize) {
        OFFLOAD_LOG_ERROR("local urma requires reserveSize == allocSize, reserveSize: "
                          << config.reserveSize << ", allocSize: " << config.allocSize);
        hybm_uninit();
        return OFFLOAD_ERROR;
    }

    hybm_options options1 = BuildHostOptions(config, alignedReserveSize);
    hybm_options options2 = options1;
    options2.bmDataOpType = HYBM_DOP_TYPE_DEVICE_URMA;
    options2.hostVASpace = 0;
    options2.rankId = 1;

    hybm_exchange_info hostEntityInfo{};
    hybm_exchange_info devEntityInfo{};
    hybm_exchange_info hostSliceInfo{};
    do {
        ret = CreateDeviceEntity(options2, flags, &devEntityInfo);
        if (ret != OFFLOAD_OK) {
            break;
        }
        ret = CreateHostEntity(options1, flags, &hostEntityInfo, &hostSliceInfo);
        if (ret != OFFLOAD_OK) {
            break;
        }
        ret = CrossImportAndMap(flags, hostEntityInfo, devEntityInfo, hostSliceInfo);
    } while (0);

    size_ = alignedReserveSize;
    if (ret == OFFLOAD_OK) {
        memMng_ = std::make_shared<AccOffloadMemManager>(base_, size_);
    }

    inited_ = true;

    if (ret != OFFLOAD_OK || memMng_ == nullptr) {
        UnInitialize();
        return ret;
    }

    return ret;
}

void AccOffloadLocalUrmaEntry::UnInitialize()
{
    if (!inited_) {
        return;
    }

    uint32_t flags = 0;
    if (entity2_ != nullptr) {
        hybm_unmap(entity2_, flags);
        hybm_unreserve_mem_space(entity2_, flags);
        hybm_destroy_entity(entity2_, flags);
        entity2_ = nullptr;
    }
    if (entity1_ != nullptr) {
        if (slice_ != nullptr) {
            hybm_free_local_memory(entity1_, slice_, 1, flags);
            slice_ = nullptr;
        }
        hybm_unreserve_mem_space(entity1_, flags);
        hybm_destroy_entity(entity1_, flags);
        entity1_ = nullptr;
    }

    hybm_uninit();
    base_ = nullptr;
    size_ = 0;
    inited_ = false;
}

void *AccOffloadLocalUrmaEntry::MallocHost(size_t size)
{
    if (memMng_ == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr, malloc failed");
        return nullptr;
    }

    OFFLOAD_LOG_DEBUG("malloc host size: " << size);
    return memMng_->Allocate(size);
}

void AccOffloadLocalUrmaEntry::FreeHost(void *ptr)
{
    if (memMng_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("mem manager is nullptr or ptr is nullptr, free failed");
        return;
    }

    OFFLOAD_LOG_DEBUG("free host ptr: " << reinterpret_cast<uint64_t>(ptr));
    memMng_->Release(ptr);
}

int32_t AccOffloadLocalUrmaEntry::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                             uint8_t devIdx, uint32_t flag)
{
    if (entity1_ == nullptr || srcPtrs == nullptr || dstPtrs == nullptr || lenPtrs == nullptr || sizePtr == nullptr) {
        OFFLOAD_LOG_ERROR("invalid params, entity1_ or ptrs is nullptr");
        return OFFLOAD_ERROR;
    }

    uint32_t batchSize = *sizePtr;
    OFFLOAD_LOG_DEBUG("sparse copy, src: " << reinterpret_cast<uint64_t>(srcPtrs) << ", dst: "
                                           << reinterpret_cast<uint64_t>(dstPtrs) << ", len: "
                                           << reinterpret_cast<uint64_t>(lenPtrs) << ", size: " << batchSize
                                           << ", devIdx: " << devIdx << ", flag: " << flag);

    if (batchSize == 0) {
        return OFFLOAD_OK;
    }

    std::vector<uint64_t> dataSizes(batchSize);
    for (uint32_t i = 0; i < batchSize; i++) {
        dataSizes[i] = lenPtrs[i];
    }

    hybm_batch_raw_copy_params params{};
    params.rankId = REMOTE_DEVICE_RANK_ID;
    params.localAddrs = reinterpret_cast<void **>(dstPtrs);
    params.remoteAddrs = reinterpret_cast<void **>(srcPtrs);
    params.dataSizes = dataSizes.data();
    params.batchSize = batchSize;

    return hybm_data_batch_raw_copy(entity2_, &params, HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE);
}

int32_t AccOffloadLocalUrmaEntry::GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                                 uint32_t *numLocalExpertPtr, int64_t *groupList,
                                                 int64_t *packedGroupList, uint8_t devIdx)
{
    if (entity1_ == nullptr || srcPtrs == nullptr || dstPtrs == nullptr || lenPtrs == nullptr ||
        numLocalExpertPtr == nullptr || groupList == nullptr || packedGroupList == nullptr) {
        OFFLOAD_LOG_ERROR("invalid params, entity1_ or ptrs is nullptr");
        return OFFLOAD_ERROR;
    }

    uint32_t num = *numLocalExpertPtr;
    OFFLOAD_LOG_DEBUG("group pack copy, num: " << num << ", devIdx: " << devIdx);

    if (num == 0) {
        return OFFLOAD_OK;
    }

    std::vector<void *> sources;
    std::vector<void *> destinations;
    std::vector<uint64_t> dataSizes;
    uint32_t packedCount = 0;
    for (uint32_t i = 0; i < num; i++) {
        if (groupList[i] == 0) {
            continue;
        }
        sources.push_back(reinterpret_cast<void *>(srcPtrs[i]));
        destinations.push_back(reinterpret_cast<void *>(dstPtrs[packedCount]));
        dataSizes.push_back(lenPtrs[i]);
        packedGroupList[packedCount] = groupList[i];
        packedCount++;
    }

    if (packedCount == 0) {
        return OFFLOAD_OK;
    }

    hybm_batch_raw_copy_params params{};
    params.rankId = REMOTE_DEVICE_RANK_ID;
    params.localAddrs = destinations.data();
    params.remoteAddrs = sources.data();
    params.dataSizes = dataSizes.data();
    params.batchSize = packedCount;

    return hybm_data_batch_raw_copy(entity2_, &params, HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE);
}

} // namespace offload
} // namespace ock
