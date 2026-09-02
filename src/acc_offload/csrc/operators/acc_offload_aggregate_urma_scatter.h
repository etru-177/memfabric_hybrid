/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#ifndef ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
#define ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

class AggregateUrmaScatterKernel {
public:
    HYBM_AICORE_KERNEL explicit AggregateUrmaScatterKernel(AscendC::TPipe *pipe) : pipe_(pipe) {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR dstNew, GM_ADDR dstBase, uint32_t segmentCount, uint32_t segmentBytes,
                                 uint64_t dstStride, uint32_t probeMode)
    {
        segmentCount_ = segmentCount;
        segmentBytes_ = segmentBytes;
        dstStride_ = dstStride;
        probeMode_ = probeMode;
        source_ = reinterpret_cast<__gm__ uint8_t *>(dstNew);
        destination_ = reinterpret_cast<__gm__ uint8_t *>(dstBase);
        blockIndex_ = AscendC::GetBlockIdx();
        blockCount_ = AscendC::GetBlockNum();
        const uint32_t alignedBytes = (segmentBytes_ + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, alignedBytes);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        if (probeMode_ == 1U) {
            return;
        }
        const uint32_t segmentsPerBlock = (segmentCount_ + blockCount_ - 1U) / blockCount_;
        const uint32_t begin = blockIndex_ * segmentsPerBlock;
        const uint32_t candidateEnd = begin + segmentsPerBlock;
        const uint32_t end = candidateEnd < segmentCount_ ? candidateEnd : segmentCount_;
        if (begin >= end) {
            return;
        }
        if (probeMode_ == 2U) {
            CopyIn(begin);
            auto local = copyQueue_.DeQue<uint8_t>();
            copyQueue_.FreeTensor(local);
            return;
        }
        if (probeMode_ == 3U) {
            auto local = copyQueue_.AllocTensor<uint8_t>();
            copyQueue_.EnQue<uint8_t>(local);
            CopyOut(begin);
            return;
        }
        CopyIn(begin);
        for (uint32_t index = begin; index < end; ++index) {
            if (index + 1U < end) {
                CopyIn(index + 1U);
            }
            CopyOut(index);
        }
    }

private:
    HYBM_AICORE_KERNEL void CopyIn(uint32_t index)
    {
        AscendC::GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(source_ + static_cast<uint64_t>(index) * segmentBytes_);
        auto local = copyQueue_.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams params{1, segmentBytes_, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(local, source, params, pad);
        copyQueue_.EnQue<uint8_t>(local);
    }

    HYBM_AICORE_KERNEL void CopyOut(uint32_t index)
    {
        AscendC::GlobalTensor<uint8_t> destination;
        destination.SetGlobalBuffer(destination_ + static_cast<uint64_t>(index) * dstStride_);
        auto local = copyQueue_.DeQue<uint8_t>();
        AscendC::DataCopyExtParams params{1, segmentBytes_, 0, 0, 0};
        AscendC::DataCopyPad(destination, local, params);
        copyQueue_.FreeTensor(local);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> copyQueue_;
    __gm__ uint8_t *source_;
    __gm__ uint8_t *destination_;
    uint64_t dstStride_;
    uint32_t segmentCount_;
    uint32_t segmentBytes_;
    uint32_t probeMode_;
    uint32_t blockIndex_;
    uint32_t blockCount_;
};

#endif // ACC_OFFLOAD_AGGREGATE_URMA_SCATTER_H
