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
#ifndef MEMFABRIC_HYBRID_ACC_LOCAL_URMA_OFFLOAD_ENTRY_H
#define MEMFABRIC_HYBRID_ACC_LOCAL_URMA_OFFLOAD_ENTRY_H

#include <mutex>
#include <memory>
#include "hybm_def.h"
#include "acc_offload.h"
#include "acc_offload_entry.h"
#include "acc_offload_mem_manager.h"

namespace ock {
namespace offload {

class AccOffloadLocalUrmaEntry : public AccOffloadEntry {
public:
    AccOffloadLocalUrmaEntry() = default;
    ~AccOffloadLocalUrmaEntry() override
    {
        UnInitialize();
    };

    AccOffloadLocalUrmaEntry(const AccOffloadLocalUrmaEntry &) = delete;
    AccOffloadLocalUrmaEntry &operator=(const AccOffloadLocalUrmaEntry &) = delete;

    int32_t Initialize(const offload_config_t &config) override;

    void UnInitialize() override;

public:
    void *MallocHost(size_t size) override;

    void FreeHost(void *ptr) override;

    int32_t SparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                       uint8_t devIdx, uint32_t flag) override;

    int32_t GroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *numLocalExpertPtr,
                          int64_t *groupList, int64_t *packedGroupList, uint8_t devIdx) override;

private:
    int32_t CreateHostEntity(const hybm_options &opts, uint32_t flags, hybm_exchange_info *entityInfo,
                             hybm_exchange_info *sliceInfo);
    int32_t CreateDeviceEntity(const hybm_options &opts, uint32_t flags, hybm_exchange_info *entityInfo);
    int32_t CrossImportAndMap(uint32_t flags, const hybm_exchange_info &hostEntityInfo,
                              const hybm_exchange_info &devEntityInfo, const hybm_exchange_info &hostSliceInfo);
    hybm_options BuildHostOptions(const offload_config_t &config, uint64_t reserveSize) const;

    std::mutex mutex_;
    bool inited_ = false;
    hybm_entity_t entity1_ = nullptr;
    hybm_entity_t entity2_ = nullptr;
    hybm_mem_slice_t slice_ = nullptr;
    uint8_t *base_ = nullptr;
    uint64_t size_ = 0;
    std::shared_ptr<AccOffloadMemManager> memMng_;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_LOCAL_URMA_OFFLOAD_ENTRY_H
