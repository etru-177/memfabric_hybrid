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

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "acc_offload_operators.h"

extern "C" {
void AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, uint8_t devIdx)
{
#if defined(ACC_SOC_VERSION_A5)
    constexpr uint32_t blockDim = 64;
#else
    constexpr uint32_t blockDim = 48;
#endif
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);

    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream]() -> int {
        OffloadOpsSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, blockDim, npuStream);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("acc_sparse_copy", callback);
}

void AccOffloadGroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *numLocalExpertPtr,
                             int64_t *groupList, int64_t *packedGroupList, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);

    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, npuStream]() -> int {
        OffloadOpsGroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, npuStream);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("acc_group_pack_copy", callback);
}
}
