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
#include "acc_offload.h"
#include "acc_offload_entry_manager.h"
#include "acc_offload_define.h"

using namespace ock::offload;

OFFLOAD_API int32_t offload_init(const offload_config_t &config)
{
    auto ret = AccOffloadEntryManager::Instance().Initialize(config);
    OFFLOAD_LOG_TRACE("offload_init ret: " << ret << ", deviceId: " << config.deviceId
                                           << ", reserveSize: " << config.reserveSize);
    return ret;
}

OFFLOAD_API void offload_uninit()
{
    AccOffloadEntryManager::Instance().UnInitialize();
    OFFLOAD_LOG_TRACE("offload_uninit finished");
}

OFFLOAD_API uint64_t offload_malloc(uint64_t size, uint64_t flags)
{
    (void)flags;
    auto ptr = AccOffloadEntryManager::Instance().MallocHost(size);
    if (ptr == nullptr) {
        OFFLOAD_LOG_ERROR("offload_malloc failed, size:" << size);
        return 0;
    }

    return reinterpret_cast<uint64_t>(ptr);
}

OFFLOAD_API void offload_free(uint64_t ptr, uint64_t flags)
{
    (void)flags;
    AccOffloadEntryManager::Instance().FreeHost(reinterpret_cast<void *>(ptr));
}

OFFLOAD_API int32_t offload_sparse_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t sizePtr,
                                        uint16_t deviceId)
{
    auto srcPtrs = reinterpret_cast<uint64_t *>(srcPtr);
    auto dstPtrs = reinterpret_cast<uint64_t *>(dstPtr);
    auto lenPtrs = reinterpret_cast<uint32_t *>(lenPtr);
    auto sizePtr_ = reinterpret_cast<uint32_t *>(sizePtr);

    return AccOffloadEntryManager::Instance().SparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr_, deviceId);
}

OFFLOAD_API int32_t offload_group_pack_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr,
                                            uint64_t numLocalExpertPtr, uint64_t groupListPtr,
                                            uint64_t packedGroupListPtr, uint16_t deviceId)
{
    auto srcPtrs = reinterpret_cast<uint64_t *>(srcPtr);
    auto dstPtrs = reinterpret_cast<uint64_t *>(dstPtr);
    auto lenPtrs = reinterpret_cast<uint32_t *>(lenPtr);
    auto numLocalExpertPtrs = reinterpret_cast<uint32_t *>(numLocalExpertPtr);
    auto groupList = reinterpret_cast<int64_t *>(groupListPtr);
    auto packedGroupList = reinterpret_cast<int64_t *>(packedGroupListPtr);

    return AccOffloadEntryManager::Instance().GroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtrs, groupList,
                                                            packedGroupList, deviceId);
}
