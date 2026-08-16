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

OFFLOAD_API int32_t offload_get_dva(uint64_t hostPtr, uint64_t *dvaPtr)
{
    if (dvaPtr == nullptr) {
        OFFLOAD_LOG_ERROR("offload_get_dva: dvaPtr is null");
        return OFFLOAD_ERROR;
    }
    if (hostPtr == 0) {
        OFFLOAD_LOG_ERROR("offload_get_dva: invalid null hostPtr");
        return OFFLOAD_ERROR;
    }
    return AccOffloadEntryManager::Instance().GetDva(hostPtr, dvaPtr);
}

OFFLOAD_API int32_t offload_sparse_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t sizePtr,
                                        uint16_t deviceId, uint32_t flag)
{
    /* the internal chain narrows the device index to uint8_t, reject values
     * that would be silently truncated instead of failing later. */
    if (deviceId > UINT8_MAX) {
        OFFLOAD_LOG_ERROR("invalid deviceId " << deviceId << ", exceeds uint8_t range");
        return OFFLOAD_ERROR;
    }

    if (flag > 1) {
        OFFLOAD_LOG_ERROR("invalid flag " << flag << ", expect 0 (sparse copy) or 1 (varlen copy)");
        return OFFLOAD_ERROR;
    }

    /* the addresses refer to device memory, validate numerically only */
    if (srcPtr == 0 || dstPtr == 0 || lenPtr == 0 || sizePtr == 0) {
        OFFLOAD_LOG_ERROR("invalid null array address, src: " << srcPtr << ", dst: " << dstPtr << ", len: " << lenPtr
                                                              << ", size: " << sizePtr);
        return OFFLOAD_ERROR;
    }

    auto srcPtrs = reinterpret_cast<uint64_t *>(srcPtr);
    auto dstPtrs = reinterpret_cast<uint64_t *>(dstPtr);
    auto lenPtrs = reinterpret_cast<uint32_t *>(lenPtr);
    auto sizePtr_ = reinterpret_cast<uint32_t *>(sizePtr);

    return AccOffloadEntryManager::Instance().SparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr_,
                                                         static_cast<uint8_t>(deviceId), flag);
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
