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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>

#include "dl_hal_api.h"
#include "hybm_def.h"
#include "hybm_va_manager.h"

namespace ock {
namespace mf {

uint8_t HybmVaManager::directionLut[BIT_LUT_SIZE];

void HybmVaManager::InitDirectionLut()
{
    for (int except = 0; except < BIT_LUT_SIZE; except++) {
        uint8_t d = HYBM_DATA_COPY_DIRECTION_BUTT;
        for (int i = 0; i < HYBM_DATA_COPY_DIRECTION_BUTT - 1; i++) {
            if ((dirMask[i] & except) == dirMask[i]) {
                d = static_cast<uint8_t>(i);
                break;
            }
        }
        directionLut[except] = d;
    }
}

static constexpr uint64_t MB_OFFSET = 20UL;
Result HybmVaManager::Initialize(AscendSocType socType) noexcept
{
#if defined(ASCEND_NPU)
    if (socType == AscendSocType::ASCEND_UNKNOWN) {
        BM_LOG_ERROR("soc type is unknown.");
        return BM_INVALID_PARAM;
    }
#endif
    soc_ = socType;
    return BM_OK;
}

Result HybmVaManager::AddVaInfo(const AllocatedGvaInfo &info, bool onlyGva)
{
    if (info.base.size == 0) {
        BM_LOG_INFO("gva:" << info.base.va[HVM_GVA] << " size:0, skip add va info");
        return BM_OK;
    }
    if (info.base.memType >= HYBM_MEM_TYPE_BUTT) {
        BM_LOG_ERROR("AddVaInfo failed: invalid memType=" << info.base.memType);
        return BM_ERROR;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (uint32_t i = 0; i < HVM_BUTT; i++) {
        if (onlyGva && (i != HVM_GVA)) {
            continue;
        }
        if (info.base.va[i] != 0) {
            auto old = CheckOverlap(info.base.va[i], info.base.size, i);
            if (old.first && old.second != info) {
                BM_LOG_ERROR("AddVaInfo failed: address overlap. gva="
                             << VaToStr(info.base.va[0]) << " va:" << VaToStr(info.base.va[i])
                             << ", size=" << VaToStr(info.base.size) << ", vaType=" << i
                             << ", localRankId=" << info.localRankId << ", importedRankId=" << info.importedRankId
                             << ", memType=" << info.base.memType << ", old: " << old.second);
                return BM_ERROR;
            }
        }
    }

    for (uint32_t i = 0; i < HVM_BUTT; i++) {
        if (onlyGva && (i != HVM_GVA)) {
            continue;
        }
        if (info.base.va[i] != 0) {
            allocatedMap_[i][info.base.va[i]] = info;
        }
    }
    return BM_OK;
}

Result HybmVaManager::AddVaInfoFromExternal(const BaseAllocatedGvaInfo &baseInfo, uint32_t localRankId)
{
    return AddVaInfoFromExternal(baseInfo, localRankId, INVALID_RANK_ID);
}

Result HybmVaManager::AddVaInfoFromExternal(const BaseAllocatedGvaInfo &baseInfo, uint32_t localRankId,
                                            uint32_t importedRankId)
{
    AllocatedGvaInfo info(baseInfo, localRankId, importedRankId);
    const auto ret = AddVaInfo(info);
    BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "ret = " << ret, ret);
    BM_LOG_DEBUG("AddVaInfoFromExternal success: " << info);
    return BM_OK;
}

Result HybmVaManager::AddVaInfo(const BaseAllocatedGvaInfo &baseInfo, uint32_t localRankId, bool onlyGva)
{
    AllocatedGvaInfo info(baseInfo, localRankId);
    const auto ret = AddVaInfo(info, onlyGva);
    BM_ASSERT_LOG_AND_RETURN(ret == BM_OK, "ret = " << ret, ret);
    BM_LOG_DEBUG("AddVaInfo success: " << info);
    return BM_OK;
}

void HybmVaManager::RemoveOneVaInfo(uint64_t va, uint32_t type)
{
    if (va == 0 || type >= HVM_BUTT) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = allocatedMap_[type].find(va);
    if (it == allocatedMap_[type].end()) {
        BM_LOG_WARN("Unable to RemoveOneVaInfo: address not found. va=" << VaToStr(va) << " type=" << type);
        return;
    }
    const AllocatedGvaInfo &info = it->second;
    BM_LOG_DEBUG("Removing VaInfo: " << info);
    for (uint32_t i = 0; i < HVM_BUTT; i++) {
        if (i == type) {
            continue;
        }
        auto its = allocatedMap_[i].find(info.base.va[i]);
        if (its != allocatedMap_[i].end() && its->second.base.va[i] == info.base.va[i]) {
            allocatedMap_[i].erase(its);
        }
    }
    allocatedMap_[type].erase(it);
    BM_LOG_DEBUG("RemoveOneGvaInfo success: gva=" << VaToStr(va) << " type=" << type);
}

uint64_t HybmVaManager::TransformVa(uint64_t va, uint32_t inputType, uint32_t outputType)
{
    BM_VALIDATE_RETURN(va > 0 && inputType < HVM_BUTT && outputType < HVM_BUTT,
                       "input is invalid, va=" << VaToStr(va) << " Itype=" << inputType << " Otype=" << outputType, 0);

    if (inputType == outputType) {
        return va;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (allocatedMap_[inputType].empty()) {
        BM_LOG_WARN("No allocated spaces found for " << inputType << ". input_va=" << VaToStr(va));
        return 0;
    }
    auto it = allocatedMap_[inputType].upper_bound(va);
    if (it != allocatedMap_[inputType].begin()) {
        --it;
        if (it->second.Contains(va, inputType) && it->second.base.va[outputType] != 0) {
            // Calculate the corresponding GVA
            uint64_t offset = va - it->second.base.va[inputType];
            uint64_t outputVa = it->second.base.va[outputType] + offset;
            BM_LOG_DEBUG("GetGvaByVa range mapping: input_va=" << VaToStr(va) << " -> output_va=" << VaToStr(outputVa)
                                                               << " (offset=" << VaToStr(offset) << ")");
            return outputVa;
        }
    }
    BM_LOG_DEBUG("TransformVa: no mapping found for va=" << VaToStr(va));
    return 0;
}

hybm_mem_type HybmVaManager::GetGvaMemType(uint64_t va)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (allocatedMap_[HVM_GVA].empty()) {
        return HYBM_MEM_TYPE_BUTT;
    }
    auto it = allocatedMap_[HVM_GVA].upper_bound(va);
    if (it != allocatedMap_[HVM_GVA].begin()) {
        --it;
    }
    if (it->second.Contains(va, HVM_GVA)) {
        BM_LOG_DEBUG("GetMemType: va=" << VaToStr(va) << " memType=" << it->second.base.memType);
        return it->second.base.memType;
    }
    BM_LOG_DEBUG("GetMemType: va=" << VaToStr(va) << " not found, returning default type");
    return HYBM_MEM_TYPE_BUTT;
}

std::pair<uint32_t, bool> HybmVaManager::GetRankByGva(uint64_t gva)
{
    if (gva == 0) {
        BM_LOG_WARN("GetRankByGva: va=0 is invalid");
        return {0, false};
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (allocatedMap_[HVM_GVA].empty()) {
        BM_LOG_DEBUG("No allocated spaces found.");
        return {0, false};
    }
    auto it = allocatedMap_[HVM_GVA].upper_bound(gva);
    if (it != allocatedMap_[HVM_GVA].begin()) {
        --it;
    }
    if (it->second.Contains(gva, HVM_GVA)) {
        BM_LOG_DEBUG("GetRankByGva: va=" << VaToStr(gva) << " rankId=" << it->second.RankId());
        return {it->second.RankId(), true};
    }
    BM_LOG_DEBUG("GetRankByGva: va=" << VaToStr(gva) << " not found");
    return {0, false};
}

AddrType HybmVaManager::ClassifyAddress(const uint64_t va)
{
    uint8_t mask = ClassifyAddressMask(va);
    if (mask & BIT_LOCAL_HOST) {
        return LOCAL_HOST;
    }
    if (mask & BIT_GLOBAL_HOST) {
        return GLOBAL_HOST;
    }
    if (mask & BIT_LOCAL_DEVICE) {
        return LOCAL_DEVICE;
    }
    if (mask & BIT_GLOBAL_DEVICE) {
        return GLOBAL_DEVICE;
    }
    return LOCAL_HOST;
}

Result HybmVaManager::GetLocalMemoryType(uint64_t va, hybm_mem_type &memType) const noexcept
{
    memType = HYBM_MEM_TYPE_BUTT;
    if (va == 0) {
        BM_LOG_ERROR("GetLocalMemoryType failed: va=0 is invalid");
        return BM_INVALID_PARAM;
    }

    if (soc_ == ASCEND_950) {
        DVattribute attr{};
        auto ret = DlHalApi::DrvMemGetAttribute(static_cast<DVdeviceptr>(va), &attr);
        if (ret != BM_OK) {
            BM_LOG_ERROR("GetLocalMemoryType failed: interface=DrvMemGetAttribute ret=" << ret
                                                                                        << ", va=" << VaToStr(va));
            return ret;
        }
        memType = (attr.memType == DV_MEM_SVM_DEVICE || attr.memType == DV_MEM_LOCK_DEV ||
                   attr.memType == DV_MEM_LOCK_DEV_DVPP)
                      ? HYBM_MEM_TYPE_DEVICE
                      : HYBM_MEM_TYPE_HOST;
        return BM_OK;
    }

    memType = (va >= HYBM_HBM_START_ADDR && va < HYBM_HBM_END_ADDR) ? HYBM_MEM_TYPE_DEVICE : HYBM_MEM_TYPE_HOST;
    return BM_OK;
}

uint8_t HybmVaManager::ClassifyAddressMask(const uint64_t va)
{
    auto r = QueryAddr(va);
    // 已 import → 单 GLOBAL
    if (r.inAllocGva && r.importedRankId != INVALID_RANK_ID) {
        return (r.memType == HYBM_MEM_TYPE_DEVICE) ? BIT_GLOBAL_DEVICE : BIT_GLOBAL_HOST;
    }
    // reservedMap_ → GVA，已在 allocatedMap_ 中才设有效位
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        if (!reservedMap_[HVM_GVA].empty()) {
            auto it = reservedMap_[HVM_GVA].upper_bound(va);
            if (it != reservedMap_[HVM_GVA].begin()) {
                --it;
            }
            if (it->second.Contains(va)) {
                if (!r.inAllocGva) {
                    return 0; // 未 join → 无效
                }
                if (it->second.memType == HYBM_MEM_TYPE_DEVICE) {
                    return BIT_LOCAL_DEVICE | BIT_GLOBAL_DEVICE;
                } else {
                    return BIT_LOCAL_HOST | BIT_GLOBAL_HOST;
                }
            }
        }
    }

    hybm_mem_type memType;
    auto ret = GetLocalMemoryType(va, memType);
    if (ret != BM_OK) {
        return 0; // 无效
    }
    return (memType == HYBM_MEM_TYPE_DEVICE) ? BIT_LOCAL_DEVICE : BIT_LOCAL_HOST;
}

hybm_data_copy_direction HybmVaManager::InferCopyDirection(uint64_t srcVa, uint64_t dstVa)
{
    auto src = static_cast<int>(ClassifyAddress(srcVa));
    auto dst = static_cast<int>(ClassifyAddress(dstVa));
    if (src >= 0 && src < ADDRESS_CATEGORY_BUTT && dst >= 0 && dst < ADDRESS_CATEGORY_BUTT) {
        auto dir = COPY_DIRECTION_TABLE[src][dst];
        if (dir != HYBM_DATA_COPY_DIRECTION_BUTT) {
            return dir;
        }
        // 方向表 BUTT 兜底：L2G 和 H2GD 已在表中，仅 LOCAL→LOCAL 需要
        if (src == LOCAL_HOST && dst == LOCAL_HOST) {
            // 检查哪个地址在 managed 范围内，就让它作为被检查的一方
            if (dstVa >= HYBM_GVM_START_ADDR && dstVa < HYBM_GVM_END_ADDR) {
                return HYBM_LOCAL_HOST_TO_GLOBAL_HOST; // g_checkMap[0]={false,true} 查 dest
            }
            if (srcVa >= HYBM_GVM_START_ADDR && srcVa < HYBM_GVM_END_ADDR) {
                return HYBM_GLOBAL_HOST_TO_LOCAL_HOST; // g_checkMap[6]={true,false} 查 src
            }
        }
    }
    return HYBM_DATA_COPY_DIRECTION_BUTT;
}

bool HybmVaManager::IsValidAddr(uint64_t va)
{
    auto r = QueryAddr(va);
    return r.inAllocGva;
}

ReservedGvaInfo HybmVaManager::AllocReserveGva(uint32_t localRankId, uint64_t size, uint64_t localSize,
                                               hybm_mem_type memType, bool enable56BitsGva, bool isTrans)
{
    ReservedGvaInfo result;
    uint32_t t = HVM_DVA; // reserve device va all
    BM_VALIDATE_RETURN(size != 0, "size must > 0.", result);
    BM_VALIDATE_RETURN(localSize != 0 || enable56BitsGva, "local size must > 0 when 56-bit GVA is not enabled.",
                       result);

    std::unique_lock<std::shared_mutex> lock(mutex_);
    uint64_t lva = 0;
    if (localSize > 0) {
        if (soc_ == ASCEND_950) {
            localSize += GB; // A5被底软额外占用了1G
        }
        lva = AllocReserveLvaInner(localRankId, localSize, t);
        if (lva == 0) {
            return result;
        }
    }

    uint64_t gva = lva;
    if (enable56BitsGva) {
        uint64_t startAddr = HYBM_56BITS_GVA_START_ADDR;
        uint64_t endAddr = HYBM_56BITS_GVA_END_ADDR;
        if (isTrans) {
            startAddr = HYBM_TRANS_GVA_START_ADDR;
            endAddr = HYBM_TRANS_GVA_END_ADDR;
        }
        uint64_t upperLimit = endAddr - startAddr;
        if (size > upperLimit) {
            BM_LOG_ERROR("Failed to reserve size:" << size << ", upper limit size:" << upperLimit);
            return result;
        }

        BM_LOG_DEBUG("enable56BitsGva searching in GVA range " << VaToStr(startAddr) << "-" << VaToStr(endAddr));
        // Find free space
        auto [freeAddr, found] = FindFreeSpace(startAddr, endAddr, size, HVM_GVA);
        if (!found) {
            BM_LOG_ERROR("AllocReserveGva failed: no free space found for size=" << VaToStr(size));
            return result;
        }
        gva = freeAddr;
    }

    result.va[HVM_GVA] = gva;
    result.va[t] = lva;
    result.size = size;
    result.memType = memType;
    result.localRankId = localRankId;
    reservedMap_[HVM_GVA][result.va[HVM_GVA]] = result;
    // 启用 56 位 GVA 时 lvaSize 与 gvaSize 不一致
    auto lvaResult = result;
    lvaResult.size = localSize;
    reservedMap_[t][lvaResult.va[t]] = lvaResult;
    BM_LOG_DEBUG("AllocReserveGva success: " << result);
    return result;
}

ReservedGvaInfo HybmVaManager::AllocReserveLva(uint32_t localRankId, uint64_t size, uint32_t type,
                                               hybm_mem_type memType)
{
    ReservedGvaInfo result;
    BM_VALIDATE_RETURN(size != 0, "size must > 0.", result);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto lva = AllocReserveLvaInner(localRankId, size, type);
    if (lva == 0) {
        return result;
    }
    result.va[type] = lva;
    result.size = size;
    result.memType = memType;
    result.localRankId = localRankId;
    reservedMap_[type][result.va[type]] = result;
    BM_LOG_DEBUG("AllocReserveLva success: " << result);
    return result;
}

uint64_t HybmVaManager::AllocReserveLvaInner(uint32_t localRankId, uint64_t size, uint32_t type)
{
    if (size == 0) {
        BM_LOG_ERROR("AllocReserveLva failed: size=0");
        return 0;
    }
    uint64_t startAddr = HYBM_GVM_START_ADDR;
    uint64_t endAddr = HYBM_GVM_END_ADDR;
    const char *gvaLayout = std::getenv("MF_GVA_LAYOUT");
    const bool useA5Gva = soc_ == ASCEND_950 || (gvaLayout != nullptr && std::strcmp(gvaLayout, "A5") == 0);
    if (useA5Gva) {
        startAddr = HYBM_GVM_START_ADDR_A5;
        endAddr = HYBM_GVM_END_ADDR_A5;
        if (soc_ != ASCEND_950) {
            BM_LOG_INFO("Use A5 GVA layout from MF_GVA_LAYOUT");
        }
    }

    uint64_t upperLimit = endAddr - startAddr;
    if (size > upperLimit) {
        BM_LOG_ERROR("Failed to reserve size:" << size << ", upper limit size:" << upperLimit);
        return 0;
    }

    BM_LOG_DEBUG("AllocReserveLva, searching in HOST range " << VaToStr(startAddr) << "-" << VaToStr(endAddr));
    // Find free space
    auto [freeAddr, found] = FindFreeSpace(startAddr, endAddr, size, type);
    if (!found) {
        BM_LOG_ERROR("AllocReserveLva failed: no free space found for size=" << VaToStr(size));
        return 0;
    }

    BM_LOG_DEBUG("AllocReserveLva success: " << VaToStr(freeAddr) << " size: " << size);
    return freeAddr;
}

void HybmVaManager::FreeReserveGva(uint64_t addr)
{
    if (addr == 0) {
        BM_LOG_WARN("Unable to FreeReserveGva: invalid addr=0");
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = reservedMap_[HVM_GVA].find(addr);
    if (it == reservedMap_[HVM_GVA].end()) {
        BM_LOG_WARN("Unable to FreeReserveGva: reserved space not found at addr=" << VaToStr(addr));
        return;
    }
    const ReservedGvaInfo &info = it->second;
    BM_LOG_DEBUG("FreeReserveGva: " << info);
    auto type = (info.va[HVM_DVA] != 0 ? HVM_DVA : HVM_HVA);
    reservedMap_[type].erase(info.va[type]);
    reservedMap_[HVM_GVA].erase(it);
    BM_LOG_DEBUG("FreeReserveGva success: addr=" << VaToStr(addr));
}

void HybmVaManager::FreeReserveLva(uint64_t addr, uint32_t type)
{
    if (addr == 0 || (type != HVM_DVA && type != HVM_HVA)) {
        BM_LOG_WARN("Unable to FreeReserveLva: invalid addr:" << addr << ", type:" << type);
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = reservedMap_[type].find(addr);
    if (it == reservedMap_[type].end()) {
        BM_LOG_WARN("FreeReserveDva failed: reserved space not found at addr=" << VaToStr(addr));
        return;
    }
    const ReservedGvaInfo &info = it->second;
    BM_LOG_DEBUG("FreeReserveLva: " << info);
    reservedMap_[type].erase(addr);
    BM_LOG_DEBUG("FreeReserveLva success: addr=" << VaToStr(addr));
}

void HybmVaManager::DumpReservedGvaInfo() const
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    BM_LOG_DEBUG("Total reserved spaces: " << reservedMap_[HVM_GVA].size());
    if (reservedMap_[HVM_GVA].empty()) {
        BM_LOG_WARN("No reserved spaces found.");
        return;
    }
    int index = 1;
    uint64_t totalSizeMB = 0;
    for (const auto &pair : reservedMap_[HVM_GVA]) {
        const ReservedGvaInfo &info = pair.second;
        BM_LOG_DEBUG(index << ". " << info);
        totalSizeMB += (info.size >> MB_OFFSET);
        index++;
    }
    BM_LOG_DEBUG("Total reserved size: " << totalSizeMB << "MB");
}

void HybmVaManager::DumpAllocatedGvaInfo() const
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    BM_LOG_DEBUG("Total allocated spaces: " << allocatedMap_[HVM_GVA].size());
    if (allocatedMap_[HVM_GVA].empty()) {
        BM_LOG_DEBUG("No allocated spaces found.");
        return;
    }
    int index = 1;
    uint64_t totalSizeMB = 0;
    for (const auto &pair : allocatedMap_[HVM_GVA]) {
        const AllocatedGvaInfo &info = pair.second;
        BM_LOG_DEBUG(index << ". " << info);
        totalSizeMB += (info.base.size >> MB_OFFSET);
        index++;
    }
    BM_LOG_DEBUG("Total allocated size: " << totalSizeMB << "MB");
    std::map<hybm_mem_type, uint64_t> sizeByType;
    std::map<hybm_mem_type, size_t> countByType;
    for (const auto &pair : allocatedMap_[HVM_GVA]) {
        const AllocatedGvaInfo &info = pair.second;
        sizeByType[info.base.memType] += (info.base.size >> MB_OFFSET);
        countByType[info.base.memType]++;
    }
    for (const auto &pair : sizeByType) {
        hybm_mem_type memType = pair.first;
        uint64_t sizeMB = pair.second;
        size_t count = countByType[memType];
        BM_LOG_DEBUG(memType << ": " << count << " allocations, total size: " << sizeMB << "MB");
    }
}

std::pair<bool, AllocatedGvaInfo> HybmVaManager::CheckOverlap(const uint64_t va, const uint64_t size, uint32_t type)
{
    const uint64_t end = va + size;
    if (allocatedMap_[type].empty()) {
        return {false, {}};
    }
    auto it = allocatedMap_[type].upper_bound(va);
    if (it != allocatedMap_[type].end()) {
        uint64_t st = it->first;
        uint64_t ed = it->second.base.size + st;
        if ((st <= va && va < ed) || (va <= st && st < end)) {
            return {true, it->second};
        }
    }

    if (it != allocatedMap_[type].begin()) {
        it--;
        uint64_t st = it->first;
        uint64_t ed = it->second.base.size + st;
        if ((st <= va && va < ed) || (va <= st && st < end)) {
            return {true, it->second};
        }
    }
    return {false, {}};
}

std::pair<uint64_t, bool> HybmVaManager::FindFreeSpace(uint64_t start, uint64_t end, uint64_t size, uint32_t type)
{
    if (size == 0 || start >= end || size > (end - start)) {
        BM_LOG_ERROR("FindFreeSpace: invalid parameters. start=" << VaToStr(start) << ", end=" << VaToStr(end)
                                                                 << ", size=" << VaToStr(size));
        return {0, false};
    }
    std::vector<std::pair<uint64_t, uint64_t>> usedRanges;
    for (const auto &pair : reservedMap_[type]) {
        const ReservedGvaInfo &info = pair.second;
        if (info.va[type] >= start && info.va[type] < end) {
            usedRanges.emplace_back(info.va[type], info.va[type] + info.size);
        }
    }
    BM_LOG_DEBUG("FindFreeSpace: found " << usedRanges.size() << " used ranges in target area");
    if (usedRanges.empty()) {
        BM_LOG_DEBUG("FindFreeSpace: no used ranges, using start address " << VaToStr(start));
        return {start, true};
    }
    std::sort(usedRanges.begin(), usedRanges.end());
    uint64_t current = start;
    for (const auto &range : usedRanges) {
        if (range.first > current) {
            uint64_t freeSize = range.first - current;
            if (freeSize >= size) {
                BM_LOG_DEBUG("FindFreeSpace: found free space at " << VaToStr(current)
                                                                   << ", size=" << VaToStr(freeSize));
                return {current, true};
            }
            BM_LOG_DEBUG("FindFreeSpace: free space at " << VaToStr(current) << " too small (size=" << VaToStr(freeSize)
                                                         << ", needed=" << VaToStr(size) << ")");
        }
        if (range.second > current) {
            current = range.second;
        }
    }
    BM_LOG_DEBUG("FindFreeSpace: current:" << VaToStr(current) << " end:" << VaToStr(end) << " size:" << VaToStr(size));
    if (current <= end && size <= end - current) {
        BM_LOG_DEBUG("FindFreeSpace: found free space at end, addr=" << VaToStr(current));
        return {current, true};
    }
    BM_LOG_ERROR("FindFreeSpace: no suitable free space found, size: " << VaToStr(size));
    return {0, false};
}

std::pair<AllocatedGvaInfo, bool> HybmVaManager::FindAllocByVa(uint64_t va, uint32_t type) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (allocatedMap_[type].empty()) {
        BM_LOG_DEBUG("No allocated spaces found.");
        return {AllocatedGvaInfo{}, false};
    }
    auto it = allocatedMap_[type].upper_bound(va);
    if (it != allocatedMap_[type].begin()) {
        --it;
    }
    if (it->second.Contains(va, type)) {
        return {it->second, true};
    }
    return {AllocatedGvaInfo{}, false};
}

size_t HybmVaManager::GetAllocCount() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t count = allocatedMap_[HVM_GVA].size();
    BM_LOG_DEBUG("GetAllocCount: " << count);
    return count;
}

size_t HybmVaManager::GetReservedCount() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    size_t count = reservedMap_[HVM_GVA].size();
    BM_LOG_DEBUG("GetReservedCount: " << count);
    return count;
}

void HybmVaManager::ClearAll()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    BM_LOG_DEBUG("Alloc count before clear: " << allocatedMap_[HVM_GVA].size());
    BM_LOG_DEBUG("Reserved count before clear: " << reservedMap_[HVM_GVA].size());
    for (uint32_t i = 0; i < HVM_BUTT; i++) {
        allocatedMap_[i].clear();
        reservedMap_[i].clear();
    }
}
} // namespace mf
} // namespace ock
