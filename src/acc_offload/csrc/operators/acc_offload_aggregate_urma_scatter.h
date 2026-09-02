/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#ifndef ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
#define ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr uint32_t AGGREGATE_SCATTER_UB_BYTES = 64U * 1024U;

template<AscendC::HardEvent event>
HYBM_AICORE_KERNEL void AggregateScatterSync(int32_t eventId)
{
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class AggregateUrmaScatterKernel {
public:
    HYBM_AICORE_KERNEL AggregateUrmaScatterKernel() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR dstNew, GM_ADDR dstBase, uint32_t segmentCount, uint32_t segmentBytes,
                                 uint64_t dstStride)
    {
        segmentCount_ = segmentCount;
        segmentBytes_ = segmentBytes;
        dstStride_ = dstStride;
        source_ = reinterpret_cast<__gm__ uint8_t *>(dstNew);
        destination_ = reinterpret_cast<__gm__ uint8_t *>(dstBase);
        blockIndex_ = AscendC::GetBlockIdx();
        blockCount_ = AscendC::GetBlockNum();
    }

    HYBM_AICORE_KERNEL void Process()
    {
        const uint32_t segmentsPerBlock = (segmentCount_ + blockCount_ - 1U) / blockCount_;
        const uint32_t begin = blockIndex_ * segmentsPerBlock;
        const uint32_t candidateEnd = begin + segmentsPerBlock;
        const uint32_t end = candidateEnd < segmentCount_ ? candidateEnd : segmentCount_;
        for (uint32_t index = begin; index < end; ++index) {
            CopySegment(source_ + static_cast<uint64_t>(index) * segmentBytes_,
                        destination_ + static_cast<uint64_t>(index) * dstStride_);
        }
    }

private:
    HYBM_AICORE_KERNEL void CopyGmToUb(__ubuf__ uint8_t *destination, __gm__ uint8_t *source, uint32_t bytes)
    {
        AscendC::LocalTensor<uint8_t> local;
        AscendC::GlobalTensor<uint8_t> global;
        AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
        AscendC::DataCopyPadExtParams<uint8_t> padParams{};
        local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        local.address_.bufferAddr = reinterpret_cast<uint64_t>(destination);
        global.SetGlobalBuffer(source);
        AscendC::DataCopyPad(local, global, copyParams, padParams);
    }

    HYBM_AICORE_KERNEL void CopyUbToGm(__gm__ uint8_t *destination, __ubuf__ uint8_t *source, uint32_t bytes)
    {
        AscendC::LocalTensor<uint8_t> local;
        AscendC::GlobalTensor<uint8_t> global;
        AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
        local.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        local.address_.bufferAddr = reinterpret_cast<uint64_t>(source);
        global.SetGlobalBuffer(destination);
        AscendC::DataCopyPad(global, local, copyParams);
    }

    HYBM_AICORE_KERNEL void CopySegment(__gm__ uint8_t *source, __gm__ uint8_t *destination)
    {
        for (uint32_t offset = 0U; offset < segmentBytes_; offset += AGGREGATE_SCATTER_UB_BYTES) {
            const uint32_t remaining = segmentBytes_ - offset;
            const uint32_t copyBytes = remaining < AGGREGATE_SCATTER_UB_BYTES ? remaining : AGGREGATE_SCATTER_UB_BYTES;
            CopyGmToUb(nullptr, source + offset, copyBytes);
            AggregateScatterSync<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            CopyUbToGm(destination + offset, nullptr, copyBytes);
            AggregateScatterSync<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __gm__ uint8_t *source_;
    __gm__ uint8_t *destination_;
    uint64_t dstStride_;
    uint32_t segmentCount_;
    uint32_t segmentBytes_;
    uint32_t blockIndex_;
    uint32_t blockCount_;
};

#endif // ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
