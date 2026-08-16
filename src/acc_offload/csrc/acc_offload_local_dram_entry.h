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
#ifndef MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H
#define MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H

#include <mutex>
#include <memory>
#include "hybm_def.h"
#include "acc_offload.h"
#include "acc_offload_entry.h"
#include "acc_offload_mem_manager.h"

namespace ock {
namespace offload {

class AccOffloadLocalDramEntry : public AccOffloadEntry {
public:
    AccOffloadLocalDramEntry() = default;
    ~AccOffloadLocalDramEntry() override
    {
        UnInitialize();
    };

    AccOffloadLocalDramEntry(const AccOffloadLocalDramEntry &) = delete;
    AccOffloadLocalDramEntry &operator=(const AccOffloadLocalDramEntry &) = delete;

    int32_t Initialize(const offload_config_t &config) override;

    void UnInitialize() override;

public:
    void *MallocHost(size_t size) override;

    void FreeHost(void *ptr) override;

    int32_t GetDva(uint64_t hostPtr, uint64_t *dvaPtr) override;

    int32_t SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, uint8_t devIdx,
                       uint32_t flag) override;

    int32_t GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *numLocalExpertPtr,
                          int64_t *groupList, int64_t *packedGroupList, uint8_t devIdx) override;

private:
    std::mutex mutex_;
    bool inited_ = false;
    hybm_entity_t entity_ = nullptr;
    hybm_mem_slice_t slice_ = nullptr;
    uint8_t *base_ = nullptr;
    uint64_t size_ = 0;
    std::shared_ptr<AccOffloadMemManager> memMng_;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_LOCAL_DRAM_OFFLOAD_ENTRY_H
