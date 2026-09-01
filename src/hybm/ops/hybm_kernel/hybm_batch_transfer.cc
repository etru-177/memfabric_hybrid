/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#include "hybm_batch_transfer.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "hybm_batch_copy_route.h"
#include "hybm_def.h"
#include "hybm_kernel_log.h"

extern "C" {
__attribute__((weak)) int32_t HcommBatchModeStart(const char *batchTag);
__attribute__((weak)) int32_t HcommBatchModeEnd(const char *batchTag);
__attribute__((weak)) int32_t HcommReadOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel, void *dst,
                                                const void *src, uint64_t len);
__attribute__((weak)) int32_t HcommWriteOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel,
                                                 void *dst, const void *src, uint64_t len);
__attribute__((weak)) int32_t HcommChannelFenceOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel);
__attribute__((weak)) int32_t HcommBatchTransferOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel,
                                                         const ock::mf::HcommBatchTransferDesc *transferDescs,
                                                         uint32_t transferDescNum);
}

namespace {
constexpr uint32_t kMaxBatchSize = 1000;
constexpr const char *kBatchTag = "HybmKernel";
constexpr int32_t kHcommNotSupported = 5; // HCCL_E_NOT_SUPPORT ABI value.

bool IsNotSupported(int32_t ret)
{
    return ret == BM_NOT_SUPPORTED || ret == BM_NOT_SUPPORT_FUNC || ret == BM_UNDER_API_UNLOAD ||
           ret == kHcommNotSupported;
}

bool IsMarkerOnly(const HybmOneSideOpParam *param)
{
    return param->list_num == 0 && param->dst_buf_addr_list == nullptr && param->src_buf_addr_list == nullptr &&
           param->len_list == nullptr && param->remote_flag_addr != 0 && param->local_flag_addr != 0 &&
           param->flag_size != 0;
}

int32_t BatchModeStart(const char *batchTag)
{
    if (HcommBatchModeStart == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommBatchModeStart(batchTag);
}

int32_t BatchModeEnd(const char *batchTag)
{
    if (HcommBatchModeEnd == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommBatchModeEnd(batchTag);
}

int32_t ReadOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel, void *dst, const void *src,
                     uint64_t len)
{
    if (HcommReadOnThread == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommReadOnThread(thread, channel, dst, src, len);
}

int32_t WriteOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel, void *dst, const void *src,
                      uint64_t len)
{
    if (HcommWriteOnThread == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommWriteOnThread(thread, channel, dst, src, len);
}

int32_t ChannelFenceOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel)
{
    if (HcommChannelFenceOnThread == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommChannelFenceOnThread(thread, channel);
}

int32_t BatchTransferOnThread(ock::mf::ThreadHandle thread, ock::mf::ChannelHandle channel,
                              const ock::mf::HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum)
{
    if (HcommBatchTransferOnThread == nullptr) {
        return BM_NOT_SUPPORTED;
    }
    return HcommBatchTransferOnThread(thread, channel, transferDescs, transferDescNum);
}

uint32_t CheckParam(const HybmOneSideOpParam *param)
{
    if (param == nullptr || param->thread == 0 || param->channel == 0) {
        HYBM_LOGE(BM_INVALID_PARAM, "invalid HybmOneSideOpParam, param=%p thread=%lu channel=%lu", (void *)param,
                  param ? param->thread : 0, param ? param->channel : 0);
        return BM_INVALID_PARAM;
    }

    if (param->list_num == 0) {
        if (IsMarkerOnly(param)) {
            return BM_OK;
        }
        HYBM_LOGE(BM_INVALID_PARAM,
                  "invalid marker-only: list_num=0 dst=%p src=%p len=%p remote=0x%lx local=0x%lx fsize=%u",
                  param->dst_buf_addr_list, param->src_buf_addr_list, param->len_list, param->remote_flag_addr,
                  param->local_flag_addr, param->flag_size);
        return BM_INVALID_PARAM;
    }

    if (param->dst_buf_addr_list == nullptr || param->src_buf_addr_list == nullptr || param->len_list == nullptr) {
        HYBM_LOGE(BM_INVALID_PARAM, "incomplete data param: list_num=%u dst=%p src=%p len=%p", param->list_num,
                  param->dst_buf_addr_list, param->src_buf_addr_list, param->len_list);
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t TransferWithBatch(bool isRead, HybmOneSideOpParam *param)
{
    std::vector<ock::mf::HcommBatchTransferDesc> descs(param->list_num);
    for (uint32_t idx = 0; idx < param->list_num; ++idx) {
        auto &desc = descs[idx];
        desc.transType = isRead ? ock::mf::HCOMM_TRANSFER_TYPE_READ : ock::mf::HCOMM_TRANSFER_TYPE_WRITE;
        if (isRead) {
            desc.transferInfo.read.len = param->len_list[idx];
            desc.transferInfo.read.dst = param->dst_buf_addr_list[idx];
            desc.transferInfo.read.src = param->src_buf_addr_list[idx];
        } else {
            desc.transferInfo.write.len = param->len_list[idx];
            desc.transferInfo.write.dst = param->dst_buf_addr_list[idx];
            desc.transferInfo.write.src = param->src_buf_addr_list[idx];
        }
    }

    uint32_t offset = 0;
    while (offset < param->list_num) {
        const uint32_t batchSize = std::min(kMaxBatchSize, param->list_num - offset);
        const int32_t ret = BatchTransferOnThread(param->thread, param->channel, descs.data() + offset, batchSize);
        if (ret != BM_OK) {
            if (IsNotSupported(ret)) {
                HYBM_LOGI("HcommBatchTransferOnThread not supported, ret=%d", ret);
                return BM_NOT_SUPPORTED;
            }
            HYBM_LOGE(BM_ERROR, "HcommBatchTransferOnThread failed, thread=%lu channel=%lu offset=%u batch=%u ret=%d",
                      param->thread, param->channel, offset, batchSize, ret);
            return ret;
        }
        offset += batchSize;
    }
    return BM_OK;
}

uint32_t TransferWithSingle(bool isRead, HybmOneSideOpParam *param)
{
    for (uint32_t idx = 0; idx < param->list_num; ++idx) {
        const int32_t ret = isRead ? ReadOnThread(param->thread, param->channel, param->dst_buf_addr_list[idx],
                                                  param->src_buf_addr_list[idx], param->len_list[idx])
                                   : WriteOnThread(param->thread, param->channel, param->dst_buf_addr_list[idx],
                                                   param->src_buf_addr_list[idx], param->len_list[idx]);
        if (ret != BM_OK) {
            if (isRead) {
                HYBM_LOGE(BM_ERROR, "HcommReadOnThread failed, idx=%u, dst=%p, src=%p, len=%" PRIu64 ", ret=%d", idx,
                          param->dst_buf_addr_list[idx], param->src_buf_addr_list[idx], param->len_list[idx], ret);
            } else {
                HYBM_LOGE(BM_ERROR, "HcommWriteOnThread failed, idx=%u, dst=%p, src=%p, len=%" PRIu64 ", ret=%d", idx,
                          param->dst_buf_addr_list[idx], param->src_buf_addr_list[idx], param->len_list[idx], ret);
            }
            return BM_ERROR;
        }
    }
    return BM_OK;
}

uint32_t HybmBatchTransferTask(bool isRead, HybmOneSideOpParam *param, bool allowSingleFallback)
{
    const int32_t batchRet = TransferWithBatch(isRead, param);
    if (batchRet == BM_NOT_SUPPORTED) {
        if (!allowSingleFallback) {
            return BM_NOT_SUPPORTED;
        }
        return TransferWithSingle(isRead, param);
    }
    if (batchRet != BM_OK) {
        return BM_ERROR;
    }
    return BM_OK;
}

uint32_t HybmBatchTransfer(bool isRead, HybmOneSideOpParam *param, bool allowSingleFallback)
{
    HYBM_LOGD("HybmBatchTransfer start, isRead=%d, ", isRead);
    uint32_t ret = CheckParam(param);
    if (ret != BM_OK) {
        return ret;
    }

    ret = static_cast<uint32_t>(BatchModeStart(kBatchTag));
    if (ret != BM_OK && !IsNotSupported(static_cast<int32_t>(ret))) {
        HYBM_LOGE(BM_ERROR, "HcommBatchModeStart failed, batchTag=%s ret=%u", kBatchTag, ret);
        return BM_ERROR;
    }

    if (param->list_num > 0) {
        ret = HybmBatchTransferTask(isRead, param, allowSingleFallback);
        if (ret != BM_OK) {
            (void)BatchModeEnd(kBatchTag);
            return ret == BM_NOT_SUPPORTED ? ret : BM_ERROR;
        }
    }

    ret = static_cast<uint32_t>(ChannelFenceOnThread(param->thread, param->channel));
    if (ret != BM_OK) {
        HYBM_LOGE(BM_ERROR, "HcommChannelFenceOnThread failed, thread=%lu channel=%lu ret=%u", param->thread,
                  param->channel, ret);
        (void)BatchModeEnd(kBatchTag);
        return BM_ERROR;
    }

    if (param->remote_flag_addr != 0 && param->flag_size != 0) {
        ret = static_cast<uint32_t>(ReadOnThread(
            param->thread, param->channel, reinterpret_cast<void *>(static_cast<uintptr_t>(param->local_flag_addr)),
            reinterpret_cast<void *>(static_cast<uintptr_t>(param->remote_flag_addr)), param->flag_size));
        if (ret != BM_OK) {
            HYBM_LOGE(BM_ERROR,
                      "remote flag read failed, thread=%lu channel=%lu localFlag=0x%lx remoteFlag=0x%lx "
                      "flagSize=%u ret=%u",
                      param->thread, param->channel, param->local_flag_addr, param->remote_flag_addr, param->flag_size,
                      ret);
            (void)BatchModeEnd(kBatchTag);
            return BM_ERROR;
        }
    }

    ret = static_cast<uint32_t>(BatchModeEnd(kBatchTag));
    if (ret != BM_OK && !IsNotSupported(static_cast<int32_t>(ret))) {
        HYBM_LOGE(BM_ERROR, "HcommBatchModeEnd failed, batchTag=%s ret=%u", kBatchTag, ret);
        return BM_ERROR;
    }
    return BM_OK;
}
} // namespace

extern "C" {
uint32_t HybmBatchWrite(HybmOneSideOpParam *param)
{
    const uint32_t ret = HybmBatchTransfer(false, param, true);
    if (ret != BM_OK) {
        HYBM_LOGE(BM_ERROR, "HybmBatchWrite failed, ret=%u", ret);
        return BM_ERROR;
    }
    return ret;
}

uint32_t HybmBatchWriteStrict(HybmOneSideOpParam *param)
{
    if (param != nullptr && param->list_num > kMaxBatchSize) {
        HYBM_LOGE(BM_INVALID_PARAM, "strict batch write exceeds capacity, listNum=%u capacity=%u", param->list_num,
                  kMaxBatchSize);
        return BM_INVALID_PARAM;
    }
    const uint32_t ret = HybmBatchTransfer(false, param, false);
    if (ret != BM_OK && ret != BM_NOT_SUPPORTED) {
        HYBM_LOGE(BM_ERROR, "HybmBatchWriteStrict failed, ret=%u", ret);
    }
    return ret;
}

uint32_t HybmBatchRead(HybmOneSideOpParam *param)
{
    const uint32_t ret = HybmBatchTransfer(true, param, true);
    if (ret != BM_OK) {
        HYBM_LOGE(BM_ERROR, "HybmBatchRead failed, ret=%u", ret);
        return BM_ERROR;
    }
    return ret;
}
}
