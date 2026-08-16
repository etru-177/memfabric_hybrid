/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
 */

#include "acc_offload_varlen_copy.h"

extern "C" __global__ __aicore__ void OffloadVarlenCopyOps(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
{
    /* MIX_AIV_1_0 (NOT AIV_ONLY): torch aclnn vector ops (torch.where,
     * Tensor.copy_, cast) in the decode graph execute on the COMPUTE
     * queue as MIX kernels — measured, not assumed: every AIV_ONLY
     * configuration scored 0.70-0.76 on gsm8k while every MIX
     * configuration scored 0.9 (bs1), because an AIV_ONLY copy races
     * those compute-queue torch ops during replay. The copy must share
     * the compute queue so stream FIFO orders it against both the torch
     * ops and SFA. */
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    OffloadVarlenCopyKernel<uint8_t> op;
    op.Init(inputs, outputs, lens, size);
    op.Process();
}

extern "C" void OffloadOpsVarlenCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr,
                                     uint32_t blockDim, void *stream)
{
    uint8_t *inputs = reinterpret_cast<uint8_t *>(srcPtrs);
    uint8_t *outputs = reinterpret_cast<uint8_t *>(dstPtrs);
    uint8_t *lens = reinterpret_cast<uint8_t *>(lenPtrs);
    uint8_t *size = reinterpret_cast<uint8_t *>(sizePtr);

    OffloadVarlenCopyOps<<<blockDim, nullptr, stream>>>(inputs, outputs, lens, size);
}
