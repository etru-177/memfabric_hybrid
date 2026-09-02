/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#include "acc_offload_aggregate_urma_scatter.h"

extern "C" __global__ __aicore__ void AggregateUrmaScatterOps(GM_ADDR dstNew, GM_ADDR dstBase,
                                                               uint32_t segmentCount, uint32_t segmentBytes,
                                                               uint64_t dstStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AggregateUrmaScatterKernel kernel(&pipe);
    kernel.Init(dstNew, dstBase, segmentCount, segmentBytes, dstStride);
    kernel.Process();
}

extern "C" void OffloadOpsAggregateUrmaScatter(void *dstNew, void *dstBase, uint32_t segmentCount,
                                                uint32_t segmentBytes, uint64_t dstStride, void *stream)
{
    constexpr uint32_t blockDim = 32U;
    AggregateUrmaScatterOps<<<blockDim, nullptr, stream>>>(reinterpret_cast<uint8_t *>(dstNew),
                                                           reinterpret_cast<uint8_t *>(dstBase), segmentCount,
                                                           segmentBytes, dstStride);
}
