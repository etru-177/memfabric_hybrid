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
#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "hybm_networks_common.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_va_manager.h"
#include "hybm_dev_user_legacy_segment.h"

namespace ock {
namespace mf {
constexpr uint8_t MAX_DEVICE_COUNT = 16;
HybmDevUserLegacySegment::HybmDevUserLegacySegment(const MemSegmentOptions &options, int eid) noexcept
    : HybmDevLegacySegment{options, eid}
{}

HybmDevUserLegacySegment::~HybmDevUserLegacySegment()
{
    UnReserveMemorySpace();
}

Result HybmDevUserLegacySegment::ValidateOptions() noexcept
{
    return BM_OK;
}

Result HybmDevUserLegacySegment::ReserveMemorySpace(void **address) noexcept
{
    BM_ASSERT_LOG_AND_RETURN(address != nullptr, "address is nullptr", BM_INVALID_PARAM);
    BM_ASSERT_LOG_AND_RETURN(options_.enable56BitsGva == true,
                             "options_.enable56BitsGva = " << options_.enable56BitsGva, BM_INVALID_PARAM);
    BM_ASSERT_LOG_AND_RETURN(options_.rankId < options_.rankCnt,
                             "rank(" << options_.rankId << ") but total " << options_.rankCnt, BM_INVALID_PARAM);

    totalVirtualSize_ = options_.rankCnt * options_.maxSize;
    auto gvaInfo = HybmVaManager::GetInstance().AllocReserveGva(options_.rankId, totalVirtualSize_, 0,
                                                                HYBM_MEM_TYPE_DEVICE, options_.enable56BitsGva, true);
    BM_ASSERT_LOG_AND_RETURN(gvaInfo.va[HVM_GVA] > 0, "gvaInfo.va[HVM_GVA] = " << gvaInfo.va[HVM_GVA], BM_ERROR);
    globalVirtualAddress_ = (uint8_t *)reinterpret_cast<void *>(gvaInfo.va[HVM_GVA]);
    lvaBase_ = globalVirtualAddress_ + options_.maxSize * options_.rankId;
    return BM_OK;
}

Result HybmDevUserLegacySegment::UnReserveMemorySpace() noexcept
{
    BM_LOG_INFO("un-reserve memory space.");
    if (!memNames_.empty() && options_.shared) {
        for (auto &name : memNames_) {
            DlAclApi::RtIpcDestroyMemoryName(name.c_str());
        }
        BM_LOG_INFO("Finish to destroy memory names.");
    } else {
        BM_LOG_INFO("Sender does not need to destroy memory names.");
    }
    memNames_.clear();
    CloseMemory();
    return BM_OK;
}

Result HybmDevUserLegacySegment::AllocLocalMemory(uint64_t size, MemSlicePtr &slice) noexcept
{
    BM_LOG_ERROR("HybmDevUserLegacySegment NOT SUPPORT AllocLocalMemory");
    return BM_NOT_SUPPORTED;
}

Result HybmDevUserLegacySegment::RegisterMemory(const void *addr, uint64_t size, MemSlicePtr &slice) noexcept
{
    if (addr == nullptr || size == 0) {
        BM_LOG_ERROR("input address parameter is invalid.");
        return BM_INVALID_PARAM;
    }

    for (auto &it : registerSlices_) {
        if (it.second.slice->vAddress_ == reinterpret_cast<uint64_t>(addr)) {
            BM_LOG_ERROR("this addr has registered, addr: 0x" << std::hex << addr);
            return BM_ERROR;
        }
    }

    /* host memory (e.g. a dram pool owned by another component) takes a dedicated
     * path: RtIpcSetMemoryName below only accepts device memory. */
    hybm_mem_type memType = HYBM_MEM_TYPE_BUTT;
    auto typeRet = HybmVaManager::GetInstance().GetLocalMemoryType(reinterpret_cast<uint64_t>(addr), memType);
    if (typeRet == BM_OK && memType == HYBM_MEM_TYPE_HOST) {
        return RegisterHostMemory(addr, size, slice);
    }

    char name[DEVICE_SHM_NAME_SIZE + 1U]{};
    int32_t ret;
    if (options_.shared) {
        ret = DlAclApi::RtIpcSetMemoryName(addr, size, name, sizeof(name));
        if (ret != 0) {
            BM_LOG_ERROR("RtIpcSetMemoryName failed, ret: " << ret << " addr: 0x" << std::hex << addr
                                                            << " size: " << size);
            return BM_DL_FUNCTION_FAILED;
        }
    }
    std::unique_lock<std::mutex> uniqueLock{mutex_};
    for (auto &remoteDev : importedDeviceInfo_) {
        if (!options_.shared ||
            !CanSdmaReaches(remoteDev.second.superPodId, remoteDev.second.serverId, remoteDev.second.devicePhyId)) {
            continue;
        }
        ret = DlAclApi::RtSetIpcMemorySuperPodPid(name, remoteDev.second.sdid, (int *)&remoteDev.second.pid, 1);
        if (ret != 0) {
            BM_LOG_ERROR("set shm(" << name << ") for sdid=" << remoteDev.second.sdid << " pid=" << remoteDev.second.pid
                                    << " failed: " << ret);
            DlAclApi::RtIpcDestroyMemoryName(name);
            return BM_DL_FUNCTION_FAILED;
        }
        BM_LOG_INFO("set shm(" << name << ") for sdid=" << remoteDev.second.sdid << " pid=" << remoteDev.second.pid
                               << " success.");
    }

    uint64_t gva = reinterpret_cast<uint64_t>(lvaBase_) + allocatedSize_;
    slice = std::make_shared<MemSlice>(sliceCount_++, HYBM_MEM_TYPE_DEVICE, MEM_PT_TYPE_SVM, gva,
                                       reinterpret_cast<uint64_t>(addr), size);
    ret = HybmVaManager::GetInstance().AddVaInfo({gva, slice->vAddress_, slice->vAddress_, size, HYBM_MEM_TYPE_DEVICE},
                                                 options_.rankId, true);
    if (ret != 0) {
        BM_LOG_ERROR("AddVaInfo failed, size: " << size << " ret: " << ret);
        if (options_.shared) {
            DlAclApi::RtIpcDestroyMemoryName(name);
        }
        slice = nullptr;
        return ret;
    }

    memNames_.emplace_back(name);
    registerSlices_.emplace(slice->index_, RegisterSlice{slice, name});
    allocatedSize_ += size;
    uniqueLock.unlock();
    return BM_OK;
}

Result HybmDevUserLegacySegment::RegisterHostMemory(const void *addr, uint64_t size, MemSlicePtr &slice) noexcept
{
    void *dva = nullptr;
    bool selfRegistered = false; /* device mapping created here (vs reused from a vmm segment) */
    bool needDeviceVa = false;
#if defined(ASCEND_NPU)
    needDeviceVa = (options_.dataOpType &
                    (HYBM_DOP_TYPE_DEVICE_RDMA | HYBM_DOP_TYPE_DEVICE_URMA | HYBM_DOP_TYPE_DEVICE_UBOE)) != 0U;
    if (needDeviceVa) {
        /* memory allocated by a vmm segment (e.g. an offload dram pool) already carries a
         * device mapping (dva == hva via HalMemMap+SetAccess): reuse it instead of
         * registering a second mapping, which the driver rejects. */
        auto existDva = HybmVaManager::GetInstance().TransformVa(reinterpret_cast<uint64_t>(addr), HVM_HVA, HVM_DVA);
        if (existDva != 0) {
            dva = reinterpret_cast<void *>(existDva);
            BM_LOG_INFO("reuse existing device mapping, addr: 0x" << std::hex << addr << " dva: 0x" << existDva);
        } else {
            auto regRet =
                DlHalApi::HalHostRegister(const_cast<void *>(addr), size, HOST_MEM_MAP_DEV, logicDeviceId_, &dva);
            if (regRet != 0) {
                BM_LOG_ERROR("HalHostRegister failed, ret: " << regRet << " addr: 0x" << std::hex << addr
                                                             << " size: " << size);
                return regRet;
            }
            selfRegistered = true;
        }
    }
#endif

    std::unique_lock<std::mutex> uniqueLock{mutex_};
    uint64_t gva = reinterpret_cast<uint64_t>(lvaBase_) + allocatedSize_;
    slice = std::make_shared<MemSlice>(sliceCount_++, HYBM_MEM_TYPE_HOST, MEM_PT_TYPE_SVM, gva,
                                       reinterpret_cast<uint64_t>(addr), size);
    /* gva-only registration: the dva/hva maps already hold the allocator's record for this
     * memory (e.g. the offload vmm pool); a full insert would overlap and be rejected.
     * GVA-map lookups (rank/direction classification) still work via the new gva below,
     * while HVA->DVA translation keeps hitting the allocator's original record. */
    auto ret = HybmVaManager::GetInstance().AddVaInfo(
        {gva, reinterpret_cast<uint64_t>(dva), reinterpret_cast<uint64_t>(addr), size, HYBM_MEM_TYPE_HOST},
        options_.rankId, true);
    if (ret != 0) {
        BM_LOG_ERROR("AddVaInfo failed, ret: " << ret << " addr: 0x" << std::hex << addr << " size: " << size);
        slice = nullptr;
        return ret;
    }

    /* host slice: no ipc name, no sdma whitelist; peers classify it via the
     * padding_[0] memType marker in the export info (with shared=false device
     * slices carry empty names too, so the name alone must not be used). */
    registerSlices_.emplace(slice->index_, RegisterSlice{slice, std::string()});
    if (selfRegistered) {
        hostDvaRegIdx_.insert(slice->index_);
    }
    allocatedSize_ += size;
    uniqueLock.unlock();
    BM_LOG_INFO("register host memory success, addr: 0x" << std::hex << addr << " size: " << size << " gva: 0x" << gva
                                                         << " dva: 0x" << reinterpret_cast<uint64_t>(dva));
    return BM_OK;
}

Result HybmDevUserLegacySegment::ReleaseSliceMemory(const MemSlicePtr &slice) noexcept
{
    if (slice == nullptr) {
        BM_LOG_ERROR("input slice is nullptr.");
        return BM_INVALID_PARAM;
    }

    auto pos = registerSlices_.find(slice->index_);
    if (pos == registerSlices_.end()) {
        BM_LOG_ERROR("release slice : " << slice->index_ << " not exist.");
        return BM_INVALID_PARAM;
    }

    /* host slice (registered via RegisterHostMemory): release the device mapping and
     * va info instead of the ipc name. Discriminate by memory type, not by the empty
     * name — with shared=false device slices carry empty names too. Only mappings
     * created at registration are unregistered here; vmm-pool mappings (reused at
     * registration) are owned by their allocator. */
    if (slice->GetMemoryType() == HYBM_MEM_TYPE_HOST) {
#if defined(ASCEND_NPU)
        if (hostDvaRegIdx_.erase(slice->index_) > 0) {
            DlHalApi::HalHostUnregisterEx(reinterpret_cast<void *>(slice->vAddress_), logicDeviceId_, HOST_MEM_MAP_DEV);
        }
#endif
        HybmVaManager::GetInstance().RemoveOneVaInfo(slice->gva_);
        registerSlices_.erase(pos);
        return BM_OK;
    }

    if (options_.shared) {
        auto ret = DlAclApi::RtIpcDestroyMemoryName(pos->second.name.c_str());
        if (ret != 0) {
            BM_LOG_ERROR("destroy memory name failed: " << ret);
            return BM_DL_FUNCTION_FAILED;
        }
    }

    registerSlices_.erase(pos);
    return BM_OK;
}

Result HybmDevUserLegacySegment::Export(std::string &exInfo) noexcept
{
    HbmExportDeviceInfo info;
    info.devicePhyId = devicePhyId_;
    info.rankId = options_.rankId;
    info.pid = HybmDevLegacySegment::pid_;
    HybmDevLegacySegment::GetDeviceInfo(info.sdid, info.serverId, info.superPodId);
    auto ret = LiteralExInfoTranslater<HbmExportDeviceInfo>{}.Serialize(info, exInfo);
    if (ret != BM_OK) {
        BM_LOG_ERROR("export info failed: " << ret);
        return BM_ERROR;
    }

    BM_LOG_DEBUG("export device info(sdid=" << sdid_ << " pid=" << pid_ << " rank=" << info.rankId
                                            << " deviceId=" << devicePhyId_ << ")");
    return BM_OK;
}

Result HybmDevUserLegacySegment::Export(const MemSlicePtr &slice, std::string &exInfo) noexcept
{
    auto pos = registerSlices_.find(slice->index_);
    if (pos == registerSlices_.end()) {
        BM_LOG_ERROR("release slice : " << slice->index_ << " not exist.");
        return BM_INVALID_PARAM;
    }

    uint32_t sdId;
    UserHbmExportSliceInfo info;
    // 多trans实例场景下,同一实例不同进程的gva_start不相同,所以仅传递offset
    info.gvaOffset = pos->second.slice->gva_ - reinterpret_cast<uint64_t>(globalVirtualAddress_);
    info.address = pos->second.slice->vAddress_;
    info.size = pos->second.slice->size_;
    info.devicePhyId = static_cast<uint32_t>(devicePhyId_);
    info.rankId = options_.rankId;
    HybmDevLegacySegment::GetDeviceInfo(sdId, info.serverId, info.superPodId);
    std::copy_n(pos->second.name.c_str(), std::min(pos->second.name.size(), sizeof(info.name) - 1), info.name);
    /* memType marker in the padding: with shared=false device slices and host slices
     * both carry an empty ipc name, so the importer cannot tell them apart by name.
     * 1 = host memory, 0 = device memory. Legacy peers ignore the padding bytes, so
     * the wire format stays compatible. */
    info.padding_[0] = (pos->second.slice->GetMemoryType() == HYBM_MEM_TYPE_HOST) ? 1 : 0;
    auto ret = LiteralExInfoTranslater<UserHbmExportSliceInfo>{}.Serialize(info, exInfo);
    if (ret != BM_OK) {
        BM_LOG_ERROR("export info failed: " << ret);
        return BM_ERROR;
    }

    BM_LOG_DEBUG("export slice success. addr:" << VaToStr(info.address) << " size:" << VaToStr(info.size)
                                               << " name:" << info.name << " rank:" << options_.rankId);
    return BM_OK;
}

Result HybmDevUserLegacySegment::GetExportSliceSize(size_t &size) noexcept
{
    size = sizeof(UserHbmExportSliceInfo);
    return BM_OK;
}

void HybmDevUserLegacySegment::RollbackIpcMemory(void *addresses[], uint32_t count) noexcept
{
    for (uint32_t j = 0; j < count; j++) {
        if (options_.shared && addresses[j] != nullptr) {
            DlAclApi::RtIpcCloseMemory(addresses[j]);
        }
    }
}

Result HybmDevUserLegacySegment::Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept
{
    if (allExInfo.empty()) {
        return BM_OK;
    }

    Result ret = BM_ERROR;
    uint32_t index = 0U;
    for (auto &info : allExInfo) {
        MemSlicePtr rms;
        auto magic = *reinterpret_cast<const uint64_t *>(info.data());
        if (magic == ENTITY_EXPORT_INFO_MAGIC) {
            ret = ImportDeviceInfo(info);
        } else if (magic == HBM_SLICE_EXPORT_INFO_MAGIC) {
            ret = ImportSliceInfo(info, rms);
        } else {
            BM_LOG_ERROR("invalid import magic : " << magic);
            ret = BM_INVALID_PARAM;
        }
        if (addresses == nullptr) {
            if (ret != BM_OK) {
                break;
            }
            // kv trans addresses is null need continue
            continue;
        }
        if (ret != BM_OK) {
            // rollback
            RollbackIpcMemory(addresses, index);
            break;
        }

        void *address = nullptr;
        if (rms != nullptr) {
            address = (void *)(ptrdiff_t)(rms->gva_);
        }
        addresses[index++] = address;
    }

    return ret;
}

Result HybmDevUserLegacySegment::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
{
    std::unique_lock<std::mutex> uniqueLock{mutex_};
    for (auto rankId : ranks) {
        importedDeviceInfo_.erase(rankId);
        RemoveSliceInfo(rankId);
    }
    uniqueLock.unlock();
    return BM_OK;
}

void HybmDevUserLegacySegment::RemoveSliceInfo(const uint32_t rankId) noexcept
{
    // Clear Imported SliceInfo
    auto it = rankToRemoteSlices_.find(rankId);
    if (it == rankToRemoteSlices_.end()) {
        return;
    }
    auto &remoteSliceVec = it->second;
    for (auto &remoteSlice : remoteSliceVec) {
        registerAddrs_.erase(reinterpret_cast<void *>(static_cast<ptrdiff_t>(remoteSlice->vAddress_)));
        HybmVaManager::GetInstance().RemoveOneVaInfo(remoteSlice->gva_);
        auto rIt = remoteSlices_.find(remoteSlice->index_);
        if (rIt == remoteSlices_.end()) {
            continue;
        }
        auto sIt = importedSliceInfo_.find(rIt->second.name);
        if (sIt == importedSliceInfo_.end()) {
            remoteSlices_.erase(remoteSlice->index_);
            continue;
        }
        auto &sliceInfo = sIt->second;
        /* synthetic host keys ("host_gva_*") have no ipc memory to close */
        if (options_.shared && CanSdmaReaches(sliceInfo.superPodId, sliceInfo.serverId, sliceInfo.devicePhyId) &&
            rIt->second.name.rfind("host_gva_", 0) != 0) {
            void *address = reinterpret_cast<void *>(static_cast<ptrdiff_t>(remoteSlice->vAddress_ << 16 >> 16));
            BM_LOG_INFO("RtIpcCloseMemory start address="
                        << address
                        << ", vAddress_ = " << reinterpret_cast<void *>(static_cast<ptrdiff_t>(remoteSlice->vAddress_))
                        << ", deviceId=" << devicePhyId_ << ", sliceInfo.devicePhyId=" << sliceInfo.devicePhyId
                        << ", sliceInfo.rankId=" << sliceInfo.rankId);
            auto ret = DlAclApi::RtIpcCloseMemory(address);
            if (ret != 0) {
                BM_LOG_WARN("Unable to close memory, address="
                            << address << ", vAddress_"
                            << reinterpret_cast<void *>(static_cast<ptrdiff_t>(remoteSlice->vAddress_))
                            << ", deviceId=" << devicePhyId_ << ", sliceInfo.devicePhyId=" << sliceInfo.devicePhyId
                            << ", sliceInfo.rankId=" << sliceInfo.rankId << ", ret:" << ret
                            << ", This may affect future memory registration.");
            }
        }
        BM_LOG_INFO("RemoveSliceInfo, rankId=" << rankId << ", remoteSlice->index_=" << remoteSlice->index_
                                               << ",slice name " << rIt->second.name);
        importedSliceInfo_.erase(rIt->second.name);
        remoteSlices_.erase(remoteSlice->index_);
    }
    rankToRemoteSlices_.erase(rankId);
}

Result HybmDevUserLegacySegment::Mmap() noexcept
{
    /* By design this segment has no deferred import work (imports complete
     * inside ImportSliceInfo), so BM_NOT_SUPPORTED is the expected outcome
     * and the caller (MemEntityDefault::Mmap) tolerates it. Keep the log at
     * WARN to avoid alarming operators on a normal path. */
    BM_LOG_WARN("HybmDevUserLegacySegment NOT SUPPORT Mmap");
    return BM_NOT_SUPPORTED;
}

MemSlicePtr HybmDevUserLegacySegment::GetMemSlice(hybm_mem_slice_t slice, bool quiet) const noexcept
{
    MemSlicePtr target;
    auto index = MemSlice::GetIndexFrom(slice);
    auto pos = registerSlices_.find(index);
    if (pos != registerSlices_.end()) {
        target = pos->second.slice;
    } else if ((pos = remoteSlices_.find(index)) != remoteSlices_.end()) {
        target = pos->second.slice;
    } else {
        if (quiet) {
            BM_LOG_DEBUG("cannot get slice: " << slice);
        } else {
            BM_LOG_ERROR("cannot get slice: " << slice);
        }
        return nullptr;
    }

    if (!target->ValidateId(slice)) {
        return nullptr;
    }

    return target;
}

Result HybmDevUserLegacySegment::Unmap() noexcept
{
    BM_LOG_INFO("HybmDevUserLegacySegment NOT SUPPORT Unmap");
    return BM_NOT_SUPPORTED;
}

Result HybmDevUserLegacySegment::ImportDeviceInfo(const std::string &info) noexcept
{
    HbmExportDeviceInfo deviceInfo;
    LiteralExInfoTranslater<HbmExportDeviceInfo> translator;
    auto ret = translator.Deserialize(info, deviceInfo);
    if (ret != 0) {
        BM_LOG_ERROR("deserialize device info failed: " << ret);
        return ret;
    }

    if (deviceInfo.devicePhyId >= MAX_DEVICE_COUNT) {
        BM_LOG_ERROR("Invalid deviceInfo device id: " << deviceInfo.devicePhyId
                                                      << ", max: " << static_cast<int>(MAX_DEVICE_COUNT));
        return BM_ERROR;
    }

    if (deviceInfo.devicePhyId != static_cast<uint32_t>(devicePhyId_) &&
        !enablePeerDevices_.test(deviceInfo.devicePhyId)) {
        auto ret = EnableRemotePeerAccess(deviceInfo.devicePhyId);
        if (ret != BM_OK) {
            return ret;
        }
        enablePeerDevices_.set(deviceInfo.devicePhyId);
        BM_LOG_DEBUG("enable peer access for : " << deviceInfo.devicePhyId);
    }
    std::unique_lock<std::mutex> uniqueLock{mutex_};
    if (options_.shared) {
        for (auto &it : registerSlices_) {
            if (it.second.name.empty()) {
                continue; /* host slices carry no ipc name and need no sdma whitelist */
            }
            ret =
                DlAclApi::RtSetIpcMemorySuperPodPid(it.second.name.c_str(), deviceInfo.sdid, (int *)&deviceInfo.pid, 1);
            if (ret != 0) {
                BM_LOG_ERROR("RtSetIpcMemorySuperPodPid failed: " << ret);
                return BM_DL_FUNCTION_FAILED;
            }
            BM_LOG_DEBUG("set whitelist for shm(" << it.second.name << ") sdid=" << deviceInfo.sdid
                                                  << " pid=" << deviceInfo.pid << " rank=" << deviceInfo.rankId
                                                  << " devId=" << deviceInfo.devicePhyId);
        }
    }

    importedDeviceInfo_.emplace(deviceInfo.rankId, deviceInfo);
    uniqueLock.unlock();
    return BM_OK;
}

Result HybmDevUserLegacySegment::ImportSliceInfo(const std::string &info, MemSlicePtr &remoteSlice) noexcept
{
    UserHbmExportSliceInfo sliceInfo;
    LiteralExInfoTranslater<UserHbmExportSliceInfo> translator;
    auto ret = translator.Deserialize(info, sliceInfo);
    if (ret != 0) {
        BM_LOG_ERROR("deserialize slice info failed: " << ret);
        return ret;
    }

    if (sliceInfo.devicePhyId >= MAX_DEVICE_COUNT) {
        BM_LOG_ERROR("Invalid sliceInfo device id: " << sliceInfo.devicePhyId
                                                     << ", max: " << static_cast<int>(MAX_DEVICE_COUNT));
        return BM_ERROR;
    }

    void *address = nullptr;
    std::unique_lock<std::mutex> uniqueLock{mutex_};
    if (sliceInfo.padding_[0] == 1) {
        /* Host slice exported with the padding memType marker: no local mapping
         * exists on this side. Peers reach this memory cross-node through the GVA
         * reconstructed from gvaOffset (fabric/GVA addressing over URMA),
         * never through a locally-usable device address — registering the
         * exporter's DVA here would poison the local VA table with a foreign
         * address. Keep vAddress_ null; the GVA-only VA entry registered
         * below is sufficient for address classification and URMA writes. */
        address = nullptr;
    } else if (options_.shared && CanSdmaReaches(sliceInfo.superPodId, sliceInfo.serverId, sliceInfo.devicePhyId)) {
        if (sliceInfo.devicePhyId != static_cast<uint32_t>(devicePhyId_) &&
            !enablePeerDevices_.test(sliceInfo.devicePhyId)) {
            auto ret = EnableRemotePeerAccess(sliceInfo.devicePhyId);
            if (ret != BM_OK) {
                return ret;
            }
            enablePeerDevices_.set(sliceInfo.devicePhyId);
            BM_LOG_DEBUG("enable peer access for : " << sliceInfo.devicePhyId);
        }

        ret = DlAclApi::RtIpcOpenMemory(&address, sliceInfo.name);
        if (ret != 0) {
            BM_LOG_ERROR("IpcOpenMemory(" << sliceInfo.name << ") failed:" << ret << ",sdid=" << sdid_
                                          << ", pid=" << pid_ << ", deviceId=" << devicePhyId_
                                          << ", sliceInfo.devicePhyId=" << sliceInfo.devicePhyId);
            return BM_DL_FUNCTION_FAILED;
        }
        BM_LOG_INFO("IpcOpenMemory(" << sliceInfo.name << ") success, sdid=" << sdid_ << ", pid=" << pid_
                                     << ", deviceId=" << devicePhyId_
                                     << ", sliceInfo.devicePhyId=" << sliceInfo.devicePhyId);
        registerAddrs_.insert(address);
    }

    /* padding memType marker (1 = host) decides the remote slice type: with
     * shared=false device slices also carry empty ipc names, so the name
     * alone cannot classify them. */
    auto remoteMemType = (sliceInfo.padding_[0] == 1) ? HYBM_MEM_TYPE_HOST : HYBM_MEM_TYPE_DEVICE;
    remoteSlice = std::make_shared<MemSlice>(sliceCount_++, remoteMemType, MEM_PT_TYPE_SVM,
                                             sliceInfo.gvaOffset + reinterpret_cast<uint64_t>(globalVirtualAddress_),
                                             reinterpret_cast<uint64_t>(address), sliceInfo.size);
    rankToRemoteSlices_[sliceInfo.rankId].push_back(remoteSlice);
    /* host slices carry no ipc name; use a unique synthetic key to avoid map collisions */
    std::string sliceKey =
        (sliceInfo.padding_[0] == 1) ? ("host_gva_" + VaToStr(remoteSlice->gva_)) : std::string(sliceInfo.name);
    remoteSlices_.emplace(remoteSlice->index_, RegisterSlice{remoteSlice, sliceKey});
    importedSliceInfo_.emplace(sliceKey, sliceInfo);
    /* [DRAM] probe: the peer-side reconstruction that smem writes will use.
     * reconstructed gva = gvaOffset + LOCAL segment GVA base. Cross-check
     * against the exporter's [DRAM] export(...) line: if the two nodes' GVA
     * bases differ, gvaOffset must encode that difference. */
    if (sliceInfo.padding_[0] == 1) {
        BM_LOG_WARN("[DRAM] import(host) exporterAddr:0x"
                    << std::hex << sliceInfo.address << " gvaOffset:0x" << sliceInfo.gvaOffset << " localSegGvaBase:0x"
                    << reinterpret_cast<uint64_t>(globalVirtualAddress_) << " -> reconstructedGva:0x"
                    << remoteSlice->gva_ << " size:0x" << std::dec << sliceInfo.size << " rank:" << sliceInfo.rankId);
    }
    uniqueLock.unlock();
    auto memType = remoteMemType;
    ret = HybmVaManager::GetInstance().AddVaInfoFromExternal(
        {remoteSlice->gva_, remoteSlice->vAddress_, 0, remoteSlice->size_, memType}, options_.rankId, sliceInfo.rankId);
    BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "ret = " << ret, ret);
    return BM_OK;
}

void HybmDevUserLegacySegment::CloseMemory() noexcept
{
    if (options_.shared) {
        for (auto &addr : registerAddrs_) {
            if (DlAclApi::RtIpcCloseMemory(addr) != 0) {
                BM_LOG_WARN("Unable to close memory. This may affect future memory registration.");
            }
        }
    }
    for (auto &it : registerSlices_) {
#if defined(ASCEND_NPU)
        /* drop device mappings created at registration; reused vmm-pool mappings
         * stay with their allocator */
        if (hostDvaRegIdx_.erase(it.second.slice->index_) > 0) {
            DlHalApi::HalHostUnregisterEx(reinterpret_cast<void *>(it.second.slice->vAddress_), logicDeviceId_,
                                          HOST_MEM_MAP_DEV);
        }
#endif
        HybmVaManager::GetInstance().RemoveOneVaInfo(it.second.slice->gva_);
    }
    registerAddrs_.clear();
    if (globalVirtualAddress_ != nullptr) {
        HybmVaManager::GetInstance().FreeReserveGva(reinterpret_cast<uint64_t>(globalVirtualAddress_));
    }
    globalVirtualAddress_ = lvaBase_ = nullptr;
    totalVirtualSize_ = 0;
    BM_LOG_INFO("close memory finish.");
}

bool HybmDevUserLegacySegment::MemoryInRange(const void *begin, uint64_t size) const noexcept
{
    if (begin < globalVirtualAddress_) {
        return false;
    }

    if (reinterpret_cast<const uint8_t *>(begin) + size > globalVirtualAddress_ + totalVirtualSize_) {
        return false;
    }

    return true;
}

bool HybmDevUserLegacySegment::CheckSdmaReaches(uint32_t rankId) const noexcept
{
    auto pos = importedDeviceInfo_.find(rankId);
    if (pos == importedDeviceInfo_.end()) {
        return false;
    }

    uint32_t sdId;
    uint32_t serverId;
    uint32_t superPodId;
    HybmDevLegacySegment::GetDeviceInfo(sdId, serverId, superPodId);

    if (pos->second.serverId == serverId) {
        return true;
    }

    if (pos->second.superPodId == invalidSuperPodId || superPodId == invalidSuperPodId) {
        return false;
    }

    return pos->second.superPodId == superPodId;
}

} // namespace mf
} // namespace ock
