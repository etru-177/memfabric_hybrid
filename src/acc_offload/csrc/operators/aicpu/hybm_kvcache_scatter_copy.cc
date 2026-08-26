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
#include "hybm_kvcache_scatter_copy.h"

#include <new>
#include <vector>

#include "hybm_batch_copy.h"
#include "hybm_def.h"
#include "hybm_kernel_log.h"

namespace {
constexpr uint64_t kTokensPerBlock = 128U;
constexpr uint64_t kBlockShift = 7U;
constexpr uint64_t kBlockMask = kTokensPerBlock - 1U;
constexpr uint64_t kKpeTokenBytes = 128U;
constexpr uint64_t kCkvTokenBytes = 1024U;
constexpr uint64_t kCkvLayerBytes = kTokensPerBlock * kCkvTokenBytes;
constexpr uint64_t kLayerStrideBytes = kTokensPerBlock * (kCkvTokenBytes + kKpeTokenBytes);
constexpr uint64_t kCopyCapacity = 16384U;
// 与HybmBatchCopy完成轮次保持一致；每轮完成后再复用SQ，兼顾大批量性能和队列安全。
constexpr uint32_t kDescriptorCapacity = kHybmBatchCopyMaxRoundDescriptors;

struct DescriptorBatch {
    std::vector<void *> destinations;
    std::vector<void *> sources;
    std::vector<uint64_t> lengths;

    DescriptorBatch()
    {
        // 复用堆上缓冲，避免增大批量后占用过多AICPU栈空间。
        destinations.reserve(kDescriptorCapacity);
        sources.reserve(kDescriptorCapacity);
        lengths.reserve(kDescriptorCapacity);
    }

    uint32_t Count() const
    {
        return static_cast<uint32_t>(lengths.size());
    }

    void Clear()
    {
        destinations.clear();
        sources.clear();
        lengths.clear();
    }
};

struct CopyAddress {
    void *ckvDst{nullptr};
    void *ckvSrc{nullptr};
    void *kpeDst{nullptr};
    void *kpeSrc{nullptr};
};

int32_t ValidateArgs(const HybmKvcacheScatterCopyParam *param)
{
    if (param == nullptr || param->hbmKpe == nullptr || param->hbmCkv == nullptr || param->hbmBlockTable == nullptr ||
        param->dramBlockTable == nullptr || param->offloadSlots == nullptr || param->srcTokenIds == nullptr ||
        param->dstSlots == nullptr || param->copyCounts == nullptr || param->hbmBlockCount == 0U ||
        param->hbmMaxBlocks == 0U || param->dramMaxBlocks == 0U || param->dramBlockTableRows == 0U ||
        param->batchSize == 0U || param->layerId < 0) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid KvcacheScatterCopy arguments, param=%p", static_cast<const void *>(param));
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

void WaitReady(const int32_t *readyFlag)
{
    if (readyFlag == nullptr) {
        return;
    }
    while (__atomic_load_n(readyFlag, __ATOMIC_ACQUIRE) != 1) {}
}

int32_t CalculateCopyAddress(const HybmKvcacheScatterCopyParam &param, uint64_t batch, uint64_t item,
                             CopyAddress &address)
{
    const int32_t offloadSlot = param.offloadSlots[batch];
    const int32_t srcToken = param.srcTokenIds[item];
    const int32_t dstToken = param.dstSlots[item];
    if (offloadSlot < 0 || static_cast<uint64_t>(offloadSlot) >= param.dramBlockTableRows || srcToken < 0 ||
        dstToken < 0) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid scatter index, batch=%lu slot=%d srcToken=%d dstToken=%d", batch,
                  offloadSlot, srcToken, dstToken);
        return BM_INVALID_PARAM;
    }
    const uint64_t srcBlock = static_cast<uint64_t>(srcToken) >> kBlockShift;
    const uint64_t dstBlock = static_cast<uint64_t>(dstToken) >> kBlockShift;
    if (srcBlock >= param.dramMaxBlocks || dstBlock >= param.hbmMaxBlocks) {
        HYBM_LOGE(BM_INVALID_PARAM, "scatter block out of range, batch=%lu srcBlock=%lu dstBlock=%lu", batch, srcBlock,
                  dstBlock);
        return BM_INVALID_PARAM;
    }
    const int32_t physicalBlock = param.hbmBlockTable[batch * param.hbmMaxBlocks + dstBlock];
    if (physicalBlock < 0 || static_cast<uint64_t>(physicalBlock) >= param.hbmBlockCount) {
        HYBM_LOGE(BM_INVALID_PARAM, "HBM block out of range, batch=%lu physicalBlock=%d hbmBlockCount=%lu", batch,
                  physicalBlock, param.hbmBlockCount);
        return BM_INVALID_PARAM;
    }
    const uint64_t srcOffset = static_cast<uint64_t>(srcToken) & kBlockMask;
    const uint64_t dstOffset = static_cast<uint64_t>(dstToken) & kBlockMask;
    const uint64_t blockGva = param.dramBlockTable[static_cast<uint64_t>(offloadSlot) * param.dramMaxBlocks + srcBlock];
    const uint64_t layerBase = blockGva + static_cast<uint64_t>(param.layerId) * kLayerStrideBytes;
    const uint64_t hbmToken = static_cast<uint64_t>(physicalBlock) * kTokensPerBlock + dstOffset;
    address.ckvDst = static_cast<uint8_t *>(param.hbmCkv) + hbmToken * kCkvTokenBytes;
    address.ckvSrc = reinterpret_cast<void *>(layerBase + srcOffset * kCkvTokenBytes);
    address.kpeDst = static_cast<uint8_t *>(param.hbmKpe) + hbmToken * kKpeTokenBytes;
    address.kpeSrc = reinterpret_cast<void *>(layerBase + kCkvLayerBytes + srcOffset * kKpeTokenBytes);
    return BM_OK;
}

void AppendDescriptor(DescriptorBatch &batch, void *destination, void *source, uint64_t length)
{
    batch.destinations.push_back(destination);
    batch.sources.push_back(source);
    batch.lengths.push_back(length);
}

int32_t SubmitBatch(DescriptorBatch &batch)
{
    if (batch.Count() == 0U) {
        return BM_OK;
    }
    HybmBatchCopyParam copyParam{batch.Count(), batch.destinations.data(), batch.sources.data(), batch.lengths.data()};
    (void)copyParam;
    const auto ret = static_cast<int32_t>(HybmBatchCopy(&copyParam));
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "HybmBatchCopy failed for KvcacheScatterCopy, descriptorCount=%u ret=%d", batch.Count(), ret);
        return ret;
    }
    batch.Clear();
    return BM_OK;
}

int32_t CopyBatch(const HybmKvcacheScatterCopyParam &param, uint64_t batchIndex, DescriptorBatch &descriptors)
{
    const int32_t copyCount = param.copyCounts[batchIndex];
    if (copyCount < 0 || static_cast<uint64_t>(copyCount) > kCopyCapacity) {
        HYBM_LOGE(BM_INVALID_PARAM, "scatter copy count out of range, batch=%lu copyCount=%d", batchIndex, copyCount);
        return BM_INVALID_PARAM;
    }
    for (int32_t copy = 0; copy < copyCount; ++copy) {
        CopyAddress address{};
        const uint64_t item = batchIndex * kCopyCapacity + static_cast<uint64_t>(copy);
        auto ret = CalculateCopyAddress(param, batchIndex, item, address);
        if (ret != BM_OK) {
            return ret;
        }
        AppendDescriptor(descriptors, address.ckvDst, address.ckvSrc, kCkvTokenBytes);
        AppendDescriptor(descriptors, address.kpeDst, address.kpeSrc, kKpeTokenBytes);
        if (descriptors.Count() == kDescriptorCapacity && (ret = SubmitBatch(descriptors)) != BM_OK) {
            return ret;
        }
    }
    return BM_OK;
}

int32_t RunScatterCopy(const HybmKvcacheScatterCopyParam &param)
{
    try {
        DescriptorBatch descriptors{};
        for (uint64_t batch = 0U; batch < param.batchSize; ++batch) {
            const auto ret = CopyBatch(param, batch, descriptors);
            if (ret != BM_OK) {
                return ret;
            }
        }
        return SubmitBatch(descriptors);
    } catch (const std::bad_alloc &) {
        HYBM_LOGE(BM_MALLOC_FAILED, "allocate KvcacheScatterCopy descriptor batch failed");
        return BM_MALLOC_FAILED;
    } catch (...) {
        HYBM_LOGE(BM_ERROR, "unexpected exception while building KvcacheScatterCopy descriptors");
        return BM_ERROR;
    }
}
} // namespace

extern "C" uint32_t HybmKvcacheScatterCopy(HybmKvcacheScatterCopyParam *param)
{
    auto ret = ValidateArgs(param);
    if (ret == BM_OK) {
        WaitReady(param->readyFlag);
        ret = RunScatterCopy(*param);
    }
    if (ret != BM_OK) {
        HYBM_LOGE(ret, "HybmKvcacheScatterCopy failed, ret=%d", ret);
    }
    return static_cast<uint32_t>(ret);
}
