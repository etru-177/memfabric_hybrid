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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_H

#include <cstddef>
#include <cstdint>
#include "acc_offload.h"

namespace ock {
namespace offload {

class AccOffloadEntry {
public:
    virtual ~AccOffloadEntry() = default;

    virtual int32_t Initialize(const offload_config_t &config) = 0;

    virtual void UnInitialize() = 0;

    virtual void *MallocHost(size_t size) = 0;

    virtual void FreeHost(void *ptr) = 0;

    virtual int32_t GetDva(uint64_t hostPtr, uint64_t *dvaPtr)
    {
        (void)hostPtr;
        (void)dvaPtr;
        return -1; /* not supported by this scene */
    }

    virtual int32_t SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                               uint8_t devIdx, uint32_t flag) = 0;

    virtual int32_t GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *numLocalExpertPtr,
                                  int64_t *groupList, int64_t *packedGroupList, uint8_t devIdx) = 0;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_ENTRY_H
