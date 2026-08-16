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
#include "acc_offload_entry_manager.h"
#include "acc_offload_local_dram_entry.h"
#include "acc_offload_shared_dram_entry.h"
#include "acc_offload_define.h"

namespace ock {
namespace offload {

AccOffloadEntryManager &AccOffloadEntryManager::Instance()
{
    static AccOffloadEntryManager instance;
    return instance;
}

AccOffloadEntryManager::~AccOffloadEntryManager()
{
    UnInitialize();
}

int32_t AccOffloadEntryManager::Initialize(const offload_config_t &config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (inited_) {
        OFFLOAD_LOG_WARN("entry manager already initialized, scene: " << static_cast<uint32_t>(scene_));
        return OFFLOAD_OK;
    }

    switch (config.scene) {
        case OFFLOAD_SCENE_LOCAL:
            entry_ = std::make_unique<AccOffloadLocalDramEntry>();
            break;
        case OFFLOAD_SCENE_SHARED:
            entry_ = std::make_unique<AccOffloadSharedDramEntry>();
            break;
        default:
            OFFLOAD_LOG_ERROR("invalid scene: " << static_cast<uint32_t>(config.scene));
            return OFFLOAD_ERROR;
    }
    scene_ = config.scene;

    auto ret = entry_->Initialize(config);
    if (ret != OFFLOAD_OK) {
        OFFLOAD_LOG_ERROR("entry initialize failed, scene: " << static_cast<uint32_t>(scene_) << ", ret: " << ret);
        entry_->UnInitialize();
        entry_.reset();
        scene_ = OFFLOAD_SCENE_LOCAL;
        return ret;
    }

    inited_ = true;
    OFFLOAD_LOG_INFO("entry manager initialized, scene: "
                     << static_cast<uint32_t>(scene_) << ", deviceId: " << config.deviceId
                     << ", reserveSize: " << config.reserveSize << ", allocSize: " << config.allocSize
                     << ", worldSize: " << config.worldSize << ", rankId: " << config.rankId);
    return OFFLOAD_OK;
}

void AccOffloadEntryManager::UnInitialize()
{
    if (!inited_) {
        return;
    }

    if (entry_ != nullptr) {
        entry_->UnInitialize();
    }
    entry_.reset();
    inited_ = false;
    OFFLOAD_LOG_INFO("entry manager uninitialized, scene: " << static_cast<uint32_t>(scene_));
}

void *AccOffloadEntryManager::MallocHost(size_t size)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, malloc failed, size: " << size);
        return nullptr;
    }
    return entry_->MallocHost(size);
}

void AccOffloadEntryManager::FreeHost(void *ptr)
{
    if (entry_ == nullptr || ptr == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null or ptr is null, free failed");
        return;
    }
    entry_->FreeHost(ptr);
}

int32_t AccOffloadEntryManager::GetDva(uint64_t hostPtr, uint64_t *dvaPtr)
{
    if (entry_ == nullptr || dvaPtr == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null or dvaPtr is null, get dva failed, hostPtr: " << hostPtr);
        return OFFLOAD_ERROR;
    }
    return entry_->GetDva(hostPtr, dvaPtr);
}

int32_t AccOffloadEntryManager::SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                           uint8_t devIdx, uint32_t flag)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, sparse copy failed");
        return OFFLOAD_ERROR;
    }
    return entry_->SparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx, flag);
}

int32_t AccOffloadEntryManager::GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                              uint32_t *numLocalExpertPtr, int64_t *groupList, int64_t *packedGroupList,
                                              uint8_t devIdx)
{
    if (entry_ == nullptr) {
        OFFLOAD_LOG_ERROR("entry is null, group pack copy failed");
        return OFFLOAD_ERROR;
    }
    return entry_->GroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, devIdx);
}

} // namespace offload
} // namespace ock
