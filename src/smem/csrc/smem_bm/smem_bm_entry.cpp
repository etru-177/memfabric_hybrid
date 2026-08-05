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
#include "smem_bm_entry.h"

#include <chrono>

#include "hybm_def.h"
#include "hybm_big_mem.h"
#include "hybm_data_op.h"
#include "mf_env_define.h"
#include "mf_env_util.h"
#include "smem_store_factory.h"
#include "mf_fault_injection_point.h"
#include "mf_num_util.h"
#include "smem_store_factory.h"

namespace ock {
namespace smem {
Result SmemBmEntry::AllocDramMemBySlice(hybm_entity_t entity, uint64_t totalSize, uint32_t flags)
{
    constexpr uint64_t dramSliceSize = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    SM_LOG_INFO("alloc dram mem by 32GB slice start, totalSize: " << totalSize);
    uint64_t remaining = totalSize;
    uint64_t allocated = 0;
    while (remaining > 0) {
        uint64_t sliceSize = (remaining >= dramSliceSize) ? dramSliceSize : remaining;
        SM_LOG_INFO("alloc dram slice progress: " << allocated << "/" << totalSize << ", sliceSize: " << sliceSize);
        auto memSlice = hybm_alloc_local_memory(entity, HYBM_MEM_TYPE_HOST, sliceSize, flags);
        if (memSlice == nullptr) {
            SM_LOG_ERROR("alloc host mem slice failed, allocated: " << allocated << " sliceSize: " << sliceSize
                                                                    << " totalSize: " << totalSize);
            return SM_ERROR;
        }
        slices_.push_back(memSlice);

        hybm_exchange_info sliceInfo{};
        auto ret = hybm_export(entity, memSlice, flags, &sliceInfo);
        if (ret != 0) {
            SM_LOG_ERROR("hybm export host slice failed, allocated: " << allocated << " sliceSize: " << sliceSize
                                                                      << " result: " << ret);
            return SM_ERROR;
        }
        sliceInfos_.push_back(sliceInfo);
        remaining -= sliceSize;
        allocated += sliceSize;
    }
    realDRAMSize_ = allocated;
    SM_LOG_INFO("alloc dram mem by 32GB slice done, totalSize: " << totalSize);
    return SM_OK;
}

Result SmemBmEntry::AllocDramMemBestEffort(hybm_entity_t entity, uint64_t totalSize, uint32_t flags)
{
    constexpr uint64_t dramSliceSize = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    SM_LOG_INFO("alloc dram mem best effort start, totalSize: " << totalSize);
    uint64_t allocated = 0;
    uint64_t sliceIdx = 0;
    while (allocated < totalSize) {
        uint64_t sliceSize = dramSliceSize;
        if (allocated + sliceSize > totalSize) {
            sliceSize = totalSize - allocated;
        }
        SM_LOG_INFO("alloc dram slice progress: " << allocated << "/" << totalSize << ", sliceSize: " << sliceSize);
        auto memSlice = hybm_alloc_local_memory(entity, HYBM_MEM_TYPE_HOST, sliceSize, flags);
        if (memSlice == nullptr) {
            SM_LOG_INFO("alloc dram mem best effort stopped, allocated: " << allocated << " sliceCount: " << sliceIdx);
            break;
        }
        slices_.push_back(memSlice);

        hybm_exchange_info sliceInfo{};
        auto ret = hybm_export(entity, memSlice, flags, &sliceInfo);
        if (ret != 0) {
            SM_LOG_ERROR("hybm export host slice failed at slice: " << sliceIdx << " result: " << ret);
            return SM_ERROR;
        }
        sliceInfos_.push_back(sliceInfo);
        allocated += sliceSize;
        sliceIdx++;
    }
    realDRAMSize_ = allocated;
    SM_LOG_INFO("alloc dram mem best effort done, " << allocated << "/" << totalSize);
    return SM_OK;
}

Result SmemBmEntry::AllocDramMem(hybm_entity_t entity, const hybm_options &options, uint32_t flags)
{
    if (options.maxDRAMSize == 0) {
        return SM_OK;
    }
    if (options.flags & SMEM_BM_FLAG_DRAM_BEST_EFFORT) {
        return AllocDramMemBestEffort(entity, options.hostVASpace, flags);
    }
    if (options.flags & SMEM_BM_FLAG_DRAM_MAP_HOST_VA) {
        return AllocDramMemBySlice(entity, options.hostVASpace, flags);
    }
    auto slice = hybm_alloc_local_memory(entity, HYBM_MEM_TYPE_HOST, options.hostVASpace, flags);
    if (slice == nullptr) {
        SM_LOG_ERROR("alloc local host mem failed, size: " << options.hostVASpace);
        return SM_ERROR;
    }

    slices_.push_back(slice);

    hybm_exchange_info dramSliceInfo{};
    auto ret = hybm_export(entity, slice, flags, &dramSliceInfo);
    if (ret != 0) {
        SM_LOG_ERROR("hybm export host slice failed, result: " << ret);
        return SM_ERROR;
    }
    sliceInfos_.push_back(dramSliceInfo);
    realDRAMSize_ = options.hostVASpace;
    return SM_OK;
}
int32_t SmemBmEntry::Initialize(const hybm_options &options)
{
    if (inited_) {
        return SM_OK;
    }
    uint32_t flags = 0;
    hybm_entity_t entity = nullptr;
    hybm_mem_slice_t slice = nullptr;
    Result ret = SM_ERROR;

    SM_VALIDATE_RETURN(CheckRankConfigConsistency(options), "check rank config consistency failed", SM_INVALID_PARAM);
    SM_LOG_ERROR_RETURN_IT_IF_NOT_OK(CreateGlobalTeam(options.rankCount, options.rankId), "create global team failed");
    if (!executorService_.Start()) {
        SM_LOG_ERROR("executor service start failed");
        if (globalGroup_ != nullptr && globalGroup_->IsJoined()) {
            (void)globalGroup_->GroupLeave();
        }
        globalGroup_ = nullptr;
        executorService_.Stop();
        return SM_ERROR;
    }
    executorService_.SetThreadName("batch-copy");

    do {
        auto entityId = Id() + HYBM_ENTITY_ID_BM_BASE;
        entity = hybm_create_entity(entityId, &options, flags);
        if (entity == nullptr) {
            SM_LOG_ERROR("hybm_create_entity failed, entityId: " << entityId << " rankId: " << options.rankId
                                                                 << " flags: " << flags);
            ret = SM_ERROR;
            break;
        }

        ret = hybm_reserve_mem_space(entity, flags);
        if (ret != 0) {
            SM_LOG_ERROR("hybm_reserve_mem_space failed, entityId: " << entityId << " ret: " << ret);
            hybm_destroy_entity(entity, flags);
            ret = SM_ERROR;
            break;
        }
        entity_ = entity;

        hybm_exchange_info hbmSliceInfo{};
        if (options.maxHBMSize > 0) {
            slice = hybm_alloc_local_memory(entity, HYBM_MEM_TYPE_DEVICE, options.deviceVASpace, flags);
            if (slice == nullptr) {
                SM_LOG_ERROR("alloc local device mem failed, size: " << options.deviceVASpace);
                ret = SM_ERROR;
                break;
            }
            slices_.push_back(slice);

            ret = hybm_export(entity, slice, flags, &hbmSliceInfo);
            if (ret != 0) {
                SM_LOG_ERROR("hybm export device slice failed, result: " << ret);
                break;
            }
            sliceInfos_.push_back(hbmSliceInfo);
            realHBMSize_ = options.deviceVASpace;
        }

        ret = AllocDramMem(entity, options, flags);
        if (ret != SM_OK) {
            SM_LOG_ERROR("alloc dram mem failed, result: " << ret);
            break;
        }

        bzero(&entityInfo_, sizeof(hybm_exchange_info));
        ret = hybm_export(entity, nullptr, HYBM_FLAG_EXPORT_ENTITY, &entityInfo_);
        if (ret != 0) {
            SM_LOG_ERROR("hybm entity export failed, result: " << ret);
            break;
        }
    } while (0);

    if (ret != 0) {
        inited_ = true; // ensure Uninitialize will execute
        Uninitialize();
        globalGroup_ = nullptr;
        return ret;
    }

    coreOptions_ = options;
    hostGva_ = hybm_get_memory_ptr(entity, HYBM_MEM_TYPE_HOST);
    deviceGva_ = hybm_get_memory_ptr(entity, HYBM_MEM_TYPE_DEVICE);
    inited_ = true;
    return 0;
}

void SmemBmEntry::Uninitialize()
{
    executorService_.Stop();
    if (!inited_) {
        return;
    }
    if (entity_ == nullptr) {
        return;
    }
    // Perform a graceful group leave so that peer ranks can synchronously clean up
    // their imported state via LeaveHandle(). This must happen before the local entity
    // is destroyed because the leave callback on the peer side still references entity_.
    if (globalGroup_ != nullptr && globalGroup_->IsJoined()) {
        auto ret = globalGroup_->GroupLeave();
        if (ret != SM_OK) {
            SM_LOG_WARN("unable to group leave during uninitialize, ret: " << ret);
        }
    }
    // Stop the group engine listen thread before releasing the entity.
    globalGroup_ = nullptr;

    uint32_t flags = 0;
    for (auto slice : slices_) {
        hybm_free_local_memory(entity_, slice, 1, flags);
    }
    slices_.clear();
    sliceInfos_.clear();
    realDRAMSize_ = 0;
    realHBMSize_ = 0;
    for (auto &pair : registedSlice_) {
        hybm_free_local_memory(entity_, pair.second.second, 1, flags);
    }
    registedSlice_.clear();
    hybm_unreserve_mem_space(entity_, flags);
    hybm_destroy_entity(entity_, flags);
    entity_ = nullptr;
    inited_ = false;
}

Result SmemBmEntry::GroupOpBarrier(int32_t input, std::string logTag)
{
    std::vector<std::pair<int, int>> errList;
    int32_t ret = globalGroup_->GroupGatherResult(input, errList);
    if (ret != SM_OK) {
        SM_LOG_ERROR(logTag << " failed, result: " << ret);
        return ret;
    }
    if (!errList.empty()) {
        std::string tmp;
        for (auto &p : errList) {
            tmp += std::to_string(p.first) + ":" + std::to_string(p.second) + ",";
        }
        SM_LOG_WARN(logTag << " ret barrier, get remote result " << tmp);
        return SM_ERROR;
    }
    return SM_OK;
}

Result SmemBmEntry::JoinHandle(uint32_t rk)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_LOG_INFO("do join func, local_rk: " << options_.rank << " receive_rk: " << rk
                                           << ", rank size is: " << globalGroup_->GetRankSize());

    uint32_t unitSize = sizeof(hybm_exchange_info);
    std::string localInfo;
    if (rk == options_.rank) {
        localInfo = std::string((char *)&entityInfo_, sizeof(hybm_exchange_info));
        for (auto sliceInfo : sliceInfos_) {
            if (sliceInfo.descLen > 0) {
                localInfo += std::string((char *)&sliceInfo, sizeof(hybm_exchange_info));
            }
        }
    }
    std::unordered_map<uint32_t, std::string> allInfo;
    std::vector<uint32_t> joined;
    int32_t ret = globalGroup_->GroupGatherPrefixKey(rk, localInfo, allInfo);
    SM_VALIDATE_RETURN(ret == SM_OK, "gather prefix info failed, ret:" << ret, ret);
    hybm_exchange_info info;
    std::vector<hybm_exchange_info> entityInfos;
    std::vector<hybm_exchange_info> sliceInfos;

    for (auto &it : allInfo) {
        if (it.first == options_.rank) {
            continue;
        }
        if (it.second.length() % unitSize != 0) {
            SM_LOG_ERROR("receive exchange info size is invalid!, size:" << it.second.length() << " rank:" << it.first);
            ret = SM_INVALID_PARAM;
            goto join_exit;
        }
        uint32_t num = it.second.length() / unitSize;
        joined.push_back(it.first);
        for (uint32_t i = 0; i < num; i++) {
            (void)std::copy_n(it.second.c_str() + i * unitSize, unitSize, (char *)&info);
            if (i == 0) {
                entityInfos.push_back(info);
            } else {
                sliceInfos.push_back(info);
            }
        }
    }

    if (!entityInfos.empty()) {
        ret = hybm_import(entity_, entityInfos.data(), entityInfos.size(), nullptr, HYBM_FLAG_EXPORT_ENTITY);
        if (ret != SM_OK) {
            SM_LOG_ERROR("hybm import entity failed, result: " << ret << " local_rank:" << options_.rank);
            goto join_exit;
        }
    }

    if (!sliceInfos.empty()) {
        ret = hybm_import(entity_, sliceInfos.data(), sliceInfos.size(), nullptr, 0);
        if (ret != SM_OK) {
            SM_LOG_ERROR("hybm import slice failed, result: " << ret << " local_rank:" << options_.rank);
            goto join_exit;
        }
    }

    ret = GroupOpBarrier(ret, "barrier before mmap");
    if (ret != SM_OK) {
        goto rollback_exit;
    }

    FIP_START(MMAP, &ret)
    ret = hybm_mmap(entity_, 0);
    FIP_END;
    if (ret != SM_OK) {
        SM_LOG_ERROR("hybm mmap failed, result: " << ret);
    }

join_exit:
    ret = GroupOpBarrier(ret, "barrier after mmap");
    if (ret != SM_OK) {
        goto rollback_exit;
    }

    SM_LOG_DEBUG("end join func, local_rk: " << options_.rank << " receive_rk: " << rk << " receive_info_num:"
                                             << allInfo.size() << ", rank size is: " << globalGroup_->GetRankSize());
    InvokeEventCb(rk, SMEM_GROUP_EVENT_JOIN);
    return SM_OK;

rollback_exit:
    for (auto &rks : joined) {
        hybm_remove_imported(entity_, rks, 0);
    }
    return ret;
}

Result SmemBmEntry::UpdateHandle(uint32_t rk)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_LOG_INFO("do update func, local_rk: " << options_.rank << " receive_rk: " << rk
                                             << ", rank size is: " << globalGroup_->GetRankSize());

    uint32_t unitSize = sizeof(hybm_exchange_info);
    std::string xinfo;
    if (rk == options_.rank) {
        xinfo = std::string((char *)&sliceInfos_.back(), sizeof(hybm_exchange_info));
    }

    int32_t ret = globalGroup_->GroupBarrierPrefixKey(rk, xinfo);
    SM_VALIDATE_RETURN(ret == SM_OK, "barrier prefix info failed, ret:" << ret, ret);
    if (rk != options_.rank) {
        hybm_exchange_info info;
        if (xinfo.length() % unitSize != 0) {
            SM_LOG_ERROR("receive exchange info size is invalid!, size:" << xinfo.length() << " rank:" << rk);
            ret = SM_INVALID_PARAM;
            goto update_exit;
        }
        uint32_t num = xinfo.length() / unitSize;
        for (uint32_t i = 0; i < num; i++) {
            (void)std::copy_n(xinfo.c_str() + i * unitSize, unitSize, (char *)&info);
            ret = hybm_import(entity_, &info, 1U, nullptr, (i == 0 ? HYBM_FLAG_EXPORT_ENTITY : 0));
            if (ret != SM_OK) {
                SM_LOG_ERROR("hybm import failed, result: " << ret << " remote_rank:" << rk
                                                            << " local_rank:" << options_.rank);
                goto update_exit;
            }
        }

        FIP_START(MMAP, &ret)
        ret = hybm_mmap(entity_, 0);
        FIP_END;
        if (ret != SM_OK) {
            SM_LOG_ERROR("hybm mmap failed, result: " << ret);
        }
    }

update_exit:
    ret = GroupOpBarrier(ret, "barrier update");
    if (ret != SM_OK) {
        return ret;
    }

    SM_LOG_DEBUG("end update func, local_rk: " << options_.rank << " receive_rk: " << rk
                                               << ", rank size is: " << globalGroup_->GetRankSize());
    return SM_OK;
}

Result SmemBmEntry::LeaveHandle(uint32_t rk)
{
    SM_LOG_INFO("do leave func, receive_rk: " << rk);
    if (!inited_) {
        SM_LOG_INFO("bm not inited, skip leave");
        return SM_NOT_INITIALIZED;
    }
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    auto ret = hybm_remove_imported(entity_, rk, 0);
    if (ret != 0) {
        SM_LOG_ERROR("hybm_remove_imported (leave) failed, remoteRank: " << rk << " ret: " << ret);
        return SM_ERROR;
    }
    InvokeEventCb(rk, SMEM_GROUP_EVENT_LEAVE);
    return SM_OK;
}

void SmemBmEntry::InvokeEventCb(uint32_t rankId, smem_bm_group_event_t event)
{
    std::unique_lock<std::mutex> locker{eventCbMutex_};
    auto cb = eventCb_;
    auto ctx = eventCbCtx_;
    locker.unlock();

    if (cb != nullptr) {
        (*cb)(reinterpret_cast<void *>(this), rankId, event, ctx);
    }
}

Result SmemBmEntry::Join(uint32_t flags)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    const uint32_t groupJoinTimeoutSec =
        mf::MfEnvUtil::GetOptionalUintOrDefault(mf::env::MF_GROUP_JOIN_MAX_TIMEOUT, MF_GROUP_JOIN_DEFAULT_TIMEOUT);
    SM_LOG_INFO("group join timeout sec: " << groupJoinTimeoutSec);
    auto start_time = std::chrono::steady_clock::now();
    // Track store connection state: if the store was ever disconnected since
    // we started joining, restart the clock once it reconnects. Time spent
    // with a dead leader cannot be used to complete the join.
    bool wasDisconnected = !globalGroup_->GetStoreConnectStatus();
    uint32_t resetCount = 0;
    while (true) {
        bool connected = globalGroup_->GetStoreConnectStatus();
        if (wasDisconnected && connected && resetCount < 1) {
            SM_LOG_DEBUG("store reconnected after disconnect, resetting join timer. rank: " << options_.rank);
            start_time = std::chrono::steady_clock::now();
            ++resetCount;
        }
        wasDisconnected = !connected;

        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (duration >= groupJoinTimeoutSec) {
            SM_LOG_ERROR("join timeout. rank: " << options_.rank << ", elapsed: " << duration << "s");
            return SM_ERROR;
        }
        auto ret = globalGroup_->GroupJoin();
        if (ret == SM_INNER_BUSY) {
            sleep(1U);
            continue;
        }
        SM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "join failed, ret: " << ret);
        SM_LOG_INFO("join success. rank: " << options_.rank);
        return SM_OK;
    }
}

Result SmemBmEntry::Update(uint32_t flags)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    const uint32_t retryTime =
        mf::MfEnvUtil::GetOptionalUintOrDefault(mf::env::MF_GROUP_RETRY_TIME, SMEM_GROUP_RETRY_TIME);
    for (uint32_t i = 0; i < retryTime; i++) {
        auto ret = globalGroup_->GroupUpdate();
        if (ret == SM_INNER_BUSY) {
            sleep(1U); // sleep 1s
            continue;
        }
        SM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "update failed, ret: " << ret);
        SM_LOG_INFO("update success. rank:" << options_.rank);
        return SM_OK;
    }

    SM_LOG_ERROR("update timeout. rank:" << options_.rank << " retryTime:" << retryTime);
    return SM_ERROR;
}

Result SmemBmEntry::Leave(uint32_t flags)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    auto ret = globalGroup_->GroupLeave();
    SM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "leave failed, ret: " << ret);

    return SM_OK;
}

Result SmemBmEntry::ExtendLocalMem(smem_bm_mem_type memType, uint64_t size)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_ASSERT_RETURN(memType == SMEM_MEM_TYPE_DEVICE || memType == SMEM_MEM_TYPE_HOST, SM_INVALID_PARAM);
    SM_ASSERT_RETURN(size > 0, SM_INVALID_PARAM);
    std::lock_guard<std::mutex> lock(mutex_);
    // 1.alloc slice
    auto hybmMemType = memType == SMEM_MEM_TYPE_DEVICE ? HYBM_MEM_TYPE_DEVICE : HYBM_MEM_TYPE_HOST;
    auto slice = hybm_alloc_local_memory(entity_, hybmMemType, size, 0);
    if (slice == nullptr) {
        SM_LOG_ERROR("Failed to alloc memory, memType:" << memType << " size:" << size);
        return SM_ERROR;
    }
    // 2.export slice
    hybm_exchange_info info{};
    auto ret = hybm_export(entity_, slice, 0, &info);
    if (ret != 0) {
        SM_LOG_ERROR("Failed to export slice:" << slice << " memType:" << memType << " size:" << size);
        hybm_free_local_memory(entity_, slice, 1, 0);
        return ret;
    }
    slices_.push_back(slice);
    sliceInfos_.push_back(info);
    // 3.group update
    const uint32_t retryTime =
        mf::MfEnvUtil::GetOptionalUintOrDefault(mf::env::MF_GROUP_RETRY_TIME, SMEM_GROUP_RETRY_TIME);
    for (uint32_t i = 0; i < retryTime; i++) {
        auto ret = globalGroup_->GroupUpdate();
        if (ret == SM_INNER_BUSY) {
            sleep(1U); // sleep 1s
            continue;
        }
        SM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "update failed, ret: " << ret);
        SM_LOG_INFO("update success. rank:" << options_.rank);
        if (memType == SMEM_MEM_TYPE_DEVICE) {
            realHBMSize_ += size;
        } else {
            realDRAMSize_ += size;
        }
        return SM_OK;
    }
    SM_LOG_ERROR("group update timeout. rank:" << options_.rank);
    slices_.pop_back();
    sliceInfos_.pop_back();
    hybm_free_local_memory(entity_, slice, 1, 0);
    return SM_ERROR;
}

Result SmemBmEntry::SetEventListener(smem_bm_group_event_cb cb, void *context)
{
    SM_ASSERT_RETURN(cb != nullptr, SM_INVALID_PARAM);
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    std::unique_lock<std::mutex> locker{eventCbMutex_};
    eventCb_ = cb;
    eventCbCtx_ = context;
    return SM_OK;
}

static hybm_data_copy_direction directMap[SMEM_MEM_TYPE_BUTT + 1][SMEM_MEM_TYPE_BUTT + 1] = {
    {HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE,
     HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST, HYBM_DATA_COPY_DIRECTION_BUTT},
    {HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE,
     HYBM_LOCAL_HOST_TO_GLOBAL_HOST, HYBM_DATA_COPY_DIRECTION_BUTT},
    {HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE, HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST, HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE,
     HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST, HYBM_DATA_COPY_DIRECTION_BUTT},
    {HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE,
     HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, HYBM_DATA_COPY_DIRECTION_BUTT},
    {HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_DATA_COPY_DIRECTION_BUTT,
     HYBM_DATA_COPY_DIRECTION_BUTT, HYBM_DATA_COPY_DIRECTION_BUTT},
};

smem_bm_mem_type SmemBmEntry::GetHybmMemTypeFromGva(const void *addr, uint64_t size)
{
    if (AddrInHostGva(addr, size)) {
        return SMEM_MEM_TYPE_HOST;
    }
    if (AddrInDeviceGva(addr, size)) {
        return SMEM_MEM_TYPE_DEVICE;
    }
    return SMEM_MEM_TYPE_BUTT;
}

Result SmemBmEntry::CheckJoined() const
{
    SM_VALIDATE_RETURN(globalGroup_ != nullptr && globalGroup_->IsJoined(), "not joined the net group yet",
                       SM_NOT_STARTED);
    return SM_OK;
}

hybm_data_copy_direction SmemBmEntry::TransToHybmDirection(const smem_bm_copy_type &smemDirect, const void *src,
                                                           uint64_t srcSize, const void *dest, uint64_t destSize)
{
    smem_bm_mem_type srcMemType = GetHybmMemTypeFromGva(src, srcSize);
    smem_bm_mem_type destMemType = GetHybmMemTypeFromGva(dest, destSize);
    switch (smemDirect) {
        case SMEMB_COPY_L2G:
            srcMemType = SMEM_MEM_TYPE_LOCAL_DEVICE;
            break;
        case SMEMB_COPY_G2L:
            destMemType = SMEM_MEM_TYPE_LOCAL_DEVICE;
            break;
        case SMEMB_COPY_G2H:
            destMemType = SMEM_MEM_TYPE_LOCAL_HOST;
            break;
        case SMEMB_COPY_H2G:
            srcMemType = SMEM_MEM_TYPE_LOCAL_HOST;
            break;
        case SMEMB_COPY_L2GH:
            srcMemType = SMEM_MEM_TYPE_LOCAL_DEVICE;
            // dest is already determined by GetHybmMemTypeFromGva (global host)
            break;
        case SMEMB_COPY_GH2L:
            // src is already determined by GetHybmMemTypeFromGva (global host)
            destMemType = SMEM_MEM_TYPE_LOCAL_DEVICE;
            break;
        case SMEMB_COPY_GH2H:
            destMemType = SMEM_MEM_TYPE_LOCAL_HOST;
            break;
        case SMEMB_COPY_H2GH:
            srcMemType = SMEM_MEM_TYPE_LOCAL_HOST;
            // dest is already determined by GetHybmMemTypeFromGva (global host)
            break;
        case SMEMB_COPY_G2G:
        default:
            break;
    }

    return directMap[srcMemType][destMemType];
}

Result SmemBmEntry::DataCopy(const void *src, void *dest, uint64_t size, smem_bm_copy_type t, void *stream,
                             uint32_t flags)
{
    SM_VALIDATE_RETURN(src != nullptr, "invalid param, src is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(dest != nullptr, "invalid param, dest is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(size != 0, "invalid param, size is 0", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(t < SMEMB_COPY_BUTT, "invalid param, type invalid: " << t, SM_INVALID_PARAM);
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_RETURN_IT_IF_NOT_OK(CheckJoined());

    if (!(flags & SMEM_BM_FLAG_USE_EXTERNAL_STREAM)) {
        stream = nullptr;
    } else {
        flags &= ~(SMEM_BM_FLAG_USE_EXTERNAL_STREAM);
    }

    hybm_data_copy_direction direct =
        (t == SMEMB_COPY_AUTO) ? HYBM_DATA_COPY_DIRECTION_AUTO : TransToHybmDirection(t, src, size, dest, size);
    if (direct == HYBM_DATA_COPY_DIRECTION_BUTT) {
        SM_LOG_ERROR("Failed to trans to hybm direct, smem direct: " << t << " src: " << src << " dest: " << dest);
        return SM_INVALID_PARAM;
    }

    hybm_copy_params copyParams = {const_cast<void *>(src), dest, size};
    auto ret = hybm_data_copy(entity_, &copyParams, direct, stream, flags);
    return ret == BM_NOT_CONNECTED ? SMEM_NOT_CONNECTED : ret;
}

Result SmemBmEntry::Wait()
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    return hybm_wait(entity_);
}

uint32_t SmemBmEntry::GetRankIdByGva(void *gva)
{
    if (AddrInHostGva(gva, 1UL)) {
        return ((uint64_t)gva - (uint64_t)hostGva_) / coreOptions_.maxDRAMSize;
    }

    if (AddrInDeviceGva(gva, 1UL)) {
        return ((uint64_t)gva - (uint64_t)deviceGva_) / coreOptions_.maxHBMSize;
    }
    return UINT32_MAX;
}

Result SmemBmEntry::RegisterMem(uint64_t addr, uint64_t size)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = registedSlice_.find(addr);
    if (iter != registedSlice_.end()) {
        if (iter->second.first != size) {
            SM_LOG_ERROR("RegisterMem size_mismatch: addr=0x" << std::hex << addr << std::dec << " new_size=" << size
                                                              << " existing_size=" << iter->second.first);
            return SM_ERROR;
        }
        SM_LOG_WARN("RegisterMem skip_dup: addr=0x" << std::hex << addr << std::dec << " size=" << size);
        return SM_OK;
    }
    auto slice = hybm_register_local_memory(entity_, reinterpret_cast<void *>(addr), size, 0);
    if (slice != nullptr) {
        registedSlice_.emplace(addr, std::make_pair(size, slice));
        SM_LOG_DEBUG("RegisterMem ok: addr=0x" << std::hex << addr << std::dec << " size=" << size);
        return SM_OK;
    }
    SM_LOG_ERROR("RegisterMem fail: addr=0x" << std::hex << addr << std::dec << " size=" << size);
    return SM_ERROR;
}

Result SmemBmEntry::UnRegisterMem(uint64_t addr)
{
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = registedSlice_.find(addr);
    if (iter == registedSlice_.end()) {
        SM_LOG_WARN("UnRegisterMem skip_notfound: addr=0x" << std::hex << addr);
        return SM_OK;
    }
    auto sz = iter->second.first;
    auto ret = hybm_free_local_memory(entity_, iter->second.second, 1, 0);
    if (ret != 0) {
        SM_LOG_ERROR("UnRegisterMem free_fail: addr=0x" << std::hex << addr << std::dec << " size=" << sz
                                                        << " ret=" << ret);
        return SM_ERROR;
    }
    registedSlice_.erase(iter);
    SM_LOG_DEBUG("UnRegisterMem ok: addr=0x" << std::hex << addr << std::dec << " size=" << sz);
    return SM_OK;
}

Result SmemBmEntry::DataCopyBatch(smem_batch_copy_params *params, smem_bm_copy_type t, uint32_t flags)
{
    SM_VALIDATE_RETURN(params->sources != nullptr, "invalid param, src is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(params->destinations != nullptr, "invalid param, dest is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(params->batchSize != 0, "invalid param, size is 0", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(params->dataSizes != nullptr, "invalid param, size is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(t < SMEMB_COPY_BUTT, "invalid param, type invalid: " << t, SM_INVALID_PARAM);
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_RETURN_IT_IF_NOT_OK(CheckJoined());

    if (!(flags & SMEM_BM_FLAG_USE_EXTERNAL_STREAM)) {
        params->stream = nullptr;
    } else {
        flags &= ~(SMEM_BM_FLAG_USE_EXTERNAL_STREAM);
    }

    hybm_data_copy_direction direct = (t == SMEMB_COPY_AUTO)
                                          ? HYBM_DATA_COPY_DIRECTION_AUTO
                                          : TransToHybmDirection(t, params->sources[0], params->dataSizes[0],
                                                                 params->destinations[0], params->dataSizes[0]);
    if (direct == HYBM_DATA_COPY_DIRECTION_BUTT) {
        SM_LOG_ERROR("Failed to trans to hybm direct, smem direct: " << t << " src: " << params->sources[0]
                                                                     << " dest: " << params->destinations[0]);
        return SM_INVALID_PARAM;
    }
    hybm_batch_copy_params copyParams = {params->sources, params->destinations, params->dataSizes, params->batchSize};
    return hybm_data_batch_copy(entity_, &copyParams, direct, params->stream, flags);
}

Result SmemBmEntry::DataCopyBatchConcurrent(smem_batch_copy_params *params, smem_bm_copy_type t, uint32_t flags,
                                            smem_batch_copy_result *results)
{
    SM_VALIDATE_RETURN(params->sources != nullptr, "invalid param, src is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(params->destinations != nullptr, "invalid param, dest is NULL", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(params->batchSize != 0, "invalid param, size is 0", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(results != nullptr, "results is null", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(results->results, "results inner pointer is null", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(results->batchSize == params->batchSize, "result batch size invalid", SM_INVALID_PARAM);
    SM_VALIDATE_RETURN(t < SMEMB_COPY_BUTT, "invalid param, type invalid: " << t, SM_INVALID_PARAM);
    SM_ASSERT_RETURN(inited_, SM_NOT_INITIALIZED);
    SM_RETURN_IT_IF_NOT_OK(CheckJoined());

    std::mutex finishMutex;
    std::condition_variable finishCond;
    uint32_t finishedCount = 0;
    for (auto i = 0U; i < params->batchSize; i++) {
        auto submitSuccess =
            executorService_.Execute([this, &finishedCount, &finishMutex, &finishCond, i, t, params, flags, results]() {
                hybm_copy_params singleParam{};
                singleParam.src = params->sources[i];
                singleParam.dest = params->destinations[i];
                singleParam.dataSize = params->dataSizes[i];
                auto direct = (t == SMEMB_COPY_AUTO)
                                  ? HYBM_DATA_COPY_DIRECTION_AUTO
                                  : TransToHybmDirection(t, params->sources[i], params->dataSizes[i],
                                                         params->destinations[i], params->dataSizes[i]);
                auto ret = hybm_data_copy(entity_, &singleParam, direct, params->stream, flags);
                SM_LOG_DEBUG("copy index: " << i << ", result:" << ret);
                results->results[i] = (ret == BM_NOT_CONNECTED) ? SMEM_NOT_CONNECTED : ret;

                std::unique_lock<std::mutex> locker{finishMutex};
                if (++finishedCount >= params->batchSize) {
                    locker.unlock();
                    finishCond.notify_one();
                }
                SM_LOG_DEBUG("copy index: " << i << ", run exit:");
            });
        if (!submitSuccess) {
            std::unique_lock<std::mutex> locker{finishMutex};
            ++finishedCount;
            results->results[i] = SM_ERROR;
        }
    }

    std::unique_lock<std::mutex> locker{finishMutex};
    finishCond.wait(locker, [&]() { return finishedCount >= params->batchSize; });
    locker.unlock();
    auto hasSuccess =
        std::any_of(results->results, results->results + results->batchSize, [](int r) { return r == 0; });
    auto hasFail = std::any_of(results->results, results->results + results->batchSize, [](int r) { return r != 0; });
    SM_LOG_DEBUG("has success = " << hasSuccess << ", has failed = " << hasFail);
    if (!hasFail) {
        return SM_OK;
    }

    if (!hasSuccess) {
        return SM_ERROR;
    }

    return SM_PARTIAL_FAILED;
}

Result SmemBmEntry::CreateGlobalTeam(uint32_t rankSize, uint32_t rankId)
{
    SmemGroupChangeCallback joinFunc = std::bind(&SmemBmEntry::JoinHandle, this, std::placeholders::_1);
    SmemGroupChangeCallback updateFunc = std::bind(&SmemBmEntry::UpdateHandle, this, std::placeholders::_1);
    SmemGroupChangeCallback leaveFunc = std::bind(&SmemBmEntry::LeaveHandle, this, std::placeholders::_1);
    SmemGroupOption opt = {rankSize,  rankId,   options_.controlOperationTimeout * SECOND_TO_MILLSEC,
                           true,      joinFunc, updateFunc,
                           leaveFunc, leaveFunc};
    SmemGroupEnginePtr group = SmemNetGroupEngine::Create(_configStore, opt);
    SM_VALIDATE_RETURN(group != nullptr,
                       "SmemNetGroupEngine::Create failed, rankSize: " << rankSize << " rankId: " << rankId, SM_ERROR);

    globalGroup_ = group;
    return SM_OK;
}

bool SmemBmEntry::AddrInHostGva(const void *address, uint64_t size)
{
    if (hostGva_ == nullptr) {
        return false;
    }

    auto totalSize = coreOptions_.maxDRAMSize * coreOptions_.rankCount;
    if ((const uint8_t *)address + size > (const uint8_t *)hostGva_ + totalSize) {
        return false;
    }

    if ((const uint8_t *)address < (const uint8_t *)hostGva_) {
        return false;
    }

    return true;
}

bool SmemBmEntry::AddrInDeviceGva(const void *address, uint64_t size)
{
    if (deviceGva_ == nullptr) {
        return false;
    }

    auto totalSize = coreOptions_.maxHBMSize * coreOptions_.rankCount;
    if ((const uint8_t *)address + size > (const uint8_t *)deviceGva_ + totalSize) {
        return false;
    }

    if ((const uint8_t *)address < (const uint8_t *)deviceGva_) {
        return false;
    }

    return true;
}

bool SmemBmEntry::CheckRankConfigConsistency(const hybm_options &options) const
{
    SmemBmConsistencyConfig localConfig{options};
    const auto localConfigAddr = static_cast<const uint8_t *>(static_cast<const void *>(&localConfig));
    const std::string key = "check_rank_config_consistency";

    std::vector<uint8_t> expectData;
    std::vector<uint8_t> existData;
    std::vector<uint8_t> localConfData;
    localConfData.insert(localConfData.end(), localConfigAddr, localConfigAddr + sizeof(localConfig));

    auto ret = _configStore->Cas(key, expectData, localConfData, existData);
    if (ret == SUCCESS) {
        SM_LOG_DEBUG("first set for key: " << key << ", config: " << localConfig.ToStr());
        return true;
    }

    if (ret != RESTORE) {
        SM_LOG_ERROR("CAS for key: " << key << " failed: " << ret);
        return false;
    }

    // CAS return RESTORE
    if (existData.size() != sizeof(localConfig)) {
        SM_LOG_ERROR("CAS for key: " << key << "Expected size of localConfig(" << sizeof(localConfig) << ") but got "
                                     << existData.size());
        return false;
    }

    auto existConfig = static_cast<const SmemBmConsistencyConfig *>(static_cast<const void *>(existData.data()));
    if (existConfig->maxDRAMSize != localConfig.maxDRAMSize) {
        SM_LOG_ERROR("exist Config maxDRAMSize:" << existConfig->maxDRAMSize << " != " << localConfig.maxDRAMSize);
        return false;
    }

    if (existConfig->maxHBMSize != localConfig.maxHBMSize) {
        SM_LOG_ERROR("exist Config maxHBMSize:" << existConfig->maxHBMSize << " != " << localConfig.maxHBMSize);
        return false;
    }

    if (existConfig->enable56BitsGva != localConfig.enable56BitsGva) {
        SM_LOG_ERROR("exist Config enable56BitsGva:" << existConfig->enable56BitsGva
                                                     << " != " << localConfig.enable56BitsGva);
        return false;
    }

    SM_LOG_DEBUG("compare for key: " << key << ", config: " << localConfig.ToStr() << " matches.");
    return true;
}
} // namespace smem
} // namespace ock
