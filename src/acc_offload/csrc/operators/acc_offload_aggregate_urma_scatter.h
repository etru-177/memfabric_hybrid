/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#ifndef ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
#define ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H

#include "aicpu/hybm_aggregate_urma_demo.h"
#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr uint32_t AGGREGATE_SCATTER_UB_BYTES = 16U * 1024U;

class AggregateUrmaScatterKernel {
public:
    HYBM_AICORE_KERNEL AggregateUrmaScatterKernel() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR message, GM_ADDR dstNew, GM_ADDR dstBase)
    {
        auto *request = &reinterpret_cast<__gm__ HybmAggregateUrmaDemoMessage *>(message)->request;
        segmentCount_ = request->segmentCount;
        segmentBytes_ = request->segmentBytes;
        dstStride_ = request->dstStride;
        source_ = reinterpret_cast<__gm__ uint8_t *>(dstNew);
        destination_ = reinterpret_cast<__gm__ uint8_t *>(dstBase);
        blockIndex_ = AscendC::GetBlockIdx();
        blockCount_ = AscendC::GetBlockNum();
        pipe_.InitBuffer(copyQueue_, 1, AGGREGATE_SCATTER_UB_BYTES);
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
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    HYBM_AICORE_KERNEL void CopySegment(__gm__ uint8_t *source, __gm__ uint8_t *destination)
    {
        for (uint32_t offset = 0U; offset < segmentBytes_; offset += AGGREGATE_SCATTER_UB_BYTES) {
            const uint32_t remaining = segmentBytes_ - offset;
            const uint32_t copyBytes = remaining < AGGREGATE_SCATTER_UB_BYTES ? remaining : AGGREGATE_SCATTER_UB_BYTES;
            AscendC::GlobalTensor<uint8_t> sourceGlobal;
            AscendC::GlobalTensor<uint8_t> destinationGlobal;
            sourceGlobal.SetGlobalBuffer(source + offset, copyBytes);
            destinationGlobal.SetGlobalBuffer(destination + offset, copyBytes);
            sourceGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            AscendC::DataCopyExtParams copyParams(1, copyBytes, 0, 0, 0);
            AscendC::DataCopyPadExtParams<uint8_t> padParams{};
            auto local = copyQueue_.AllocTensor<uint8_t>();
            AscendC::DataCopyPad(local, sourceGlobal, copyParams, padParams);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<uint8_t>();
            AscendC::DataCopyPad(destinationGlobal, local, copyParams);
            copyQueue_.FreeTensor(local);
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> copyQueue_;
    __gm__ uint8_t *source_;
    __gm__ uint8_t *destination_;
    uint64_t dstStride_;
    uint32_t segmentCount_;
    uint32_t segmentBytes_;
    uint32_t blockIndex_;
    uint32_t blockCount_;
};

#endif // ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
