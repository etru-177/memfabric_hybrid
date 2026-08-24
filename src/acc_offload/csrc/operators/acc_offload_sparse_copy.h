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

#ifndef ACC_OFFLOAD_SPARSE_COPY_H
#define ACC_OFFLOAD_SPARSE_COPY_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr int64_t UB_ONCE_SIZE = 176 * 1024;
constexpr int64_t UB_DOUBLE_BUFFER_SIZE = 88 * 1024;

template<typename T>
class OffloadMoeSparseCopyKernel {
public:
    HYBM_AICORE_KERNEL OffloadMoeSparseCopyKernel() {}

    HYBM_AICORE_KERNEL void Init(AscendC::TPipe *pipe, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
    {
        pipe_ = pipe;
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();
        size_ = *(reinterpret_cast<__gm__ uint32_t *>(size));
        inputs_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
        outputs_ = reinterpret_cast<__gm__ uint64_t *>(outputs);
        lens_ = reinterpret_cast<__gm__ uint32_t *>(lens);
        pipe_->InitBuffer(bindQueue_, 1, UB_ONCE_SIZE);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        uint64_t totalBytes = 0;
        for (uint32_t i = 0; i < size_; i++) {
            totalBytes += static_cast<uint64_t>(lens_[i]) * sizeof(T);
        }
        uint64_t base = totalBytes / aivNum_;
        uint64_t rem = totalBytes % aivNum_;
        uint64_t sliceBytes = base + (aivIndex_ < rem ? 1 : 0);
        if (sliceBytes == 0) {
            return;
        }
        uint64_t startByte = aivIndex_ * base + (aivIndex_ < rem ? aivIndex_ : rem);
        uint64_t endByte = startByte + sliceBytes;

        uint64_t curByte = 0;
        for (uint32_t i = 0; i < size_; i++) {
            uint64_t entryBytes = static_cast<uint64_t>(lens_[i]) * sizeof(T);
            uint64_t entryEnd = curByte + entryBytes;
            if (entryEnd > startByte && curByte < endByte) {
                uint64_t ovStart = (curByte > startByte) ? curByte : startByte;
                uint64_t ovEnd = (entryEnd < endByte) ? entryEnd : endByte;
                uint32_t offsetElems = static_cast<uint32_t>(ovStart - curByte) / sizeof(T);
                uint32_t copyElems = static_cast<uint32_t>(ovEnd - ovStart) / sizeof(T);
                auto inputPtr = reinterpret_cast<__gm__ T *>(inputs_[i]);
                auto outputPtr = reinterpret_cast<__gm__ T *>(outputs_[i]);
                inputGm_.SetGlobalBuffer(inputPtr, lens_[i]);
                outputGm_.SetGlobalBuffer(outputPtr, lens_[i]);
                CpGM2GM(copyElems, offsetElems);
            }
            curByte = entryEnd;
            if (curByte >= endByte) {
                break;
            }
        }
    }

private:
    HYBM_AICORE_KERNEL void CpGM2GM(uint32_t lenElems, uint32_t offsetElems)
    {
        uint32_t leftBytes = lenElems * sizeof(T);
        uint32_t times = 0;
        uint32_t preCopyNum = UB_ONCE_SIZE / sizeof(T);
        AscendC::DataCopyPadExtParams<T> padParams;
        while (leftBytes > 0) {
            uint32_t curCopyBytes = (leftBytes > UB_ONCE_SIZE) ? UB_ONCE_SIZE : leftBytes;
            AscendC::LocalTensor<T> local = bindQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyParams(1, curCopyBytes, 0, 0, 0);
            AscendC::DataCopyPad(local, inputGm_[offsetElems + times * preCopyNum], dataCopyParams, padParams);
            bindQueue_.EnQue(local);
            local = bindQueue_.DeQue<T>();
            AscendC::DataCopyPad(outputGm_[offsetElems + times * preCopyNum], local, dataCopyParams);
            bindQueue_.FreeTensor(local);
            leftBytes = (leftBytes > UB_ONCE_SIZE) ? leftBytes - UB_ONCE_SIZE : 0;
            times++;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> bindQueue_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint32_t aivNum_;
    uint32_t aivIndex_;
    uint32_t size_;
    __gm__ uint64_t *inputs_;
    __gm__ uint64_t *outputs_;
    __gm__ uint32_t *lens_;
};

template<typename T>
class OffloadKVSparseCopyKernel {
public:
    HYBM_AICORE_KERNEL OffloadKVSparseCopyKernel() {}

    HYBM_AICORE_KERNEL void Init(AscendC::TPipe *pipe, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
    {
        pipe_ = pipe;
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();

        size_ = *(reinterpret_cast<__gm__ uint32_t *>(size));
        inputs_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
        outputs_ = reinterpret_cast<__gm__ uint64_t *>(outputs);
        lens_ = reinterpret_cast<__gm__ uint32_t *>(lens);

        pipe_->InitBuffer(bindQueue_, 2, UB_DOUBLE_BUFFER_SIZE);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        uint32_t entry;
        EntryCtx cur;
        if (!FirstEntry(entry, cur)) {
            return;
        }
        CopyIn(cur);
        while (true) {
            EntryCtx next;
            bool hasNext = NextEntry(entry, next);
            if (hasNext) {
                CopyIn(next);
            }
            CopyOut(cur);
            if (!hasNext) {
                break;
            }
            cur = next;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    struct EntryCtx {
        uint64_t src = 0;
        uint64_t dst = 0;
        uint32_t bytes = 0;
    };

    HYBM_AICORE_KERNEL bool LoadEntry(uint32_t entry, EntryCtx &ctx)
    {
        ctx.src = inputs_[entry];
        ctx.dst = outputs_[entry];
        ctx.bytes = lens_[entry] * static_cast<uint32_t>(sizeof(T));
        return ctx.bytes > 0;
    }

    HYBM_AICORE_KERNEL bool FirstEntry(uint32_t &entry, EntryCtx &ctx)
    {
        entry = aivIndex_;
        if (entry >= size_) {
            return false;
        }
        if (LoadEntry(entry, ctx)) {
            return true;
        }
        return NextEntry(entry, ctx);
    }

    HYBM_AICORE_KERNEL bool NextEntry(uint32_t &entry, EntryCtx &ctx)
    {
        while ((entry += aivNum_) < size_) {
            if (LoadEntry(entry, ctx)) {
                return true;
            }
        }
        return false;
    }

    HYBM_AICORE_KERNEL void CopyIn(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<T> local = bindQueue_.AllocTensor<T>();
        AscendC::GlobalTensor<T> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ctx.src), ctx.bytes / static_cast<uint32_t>(sizeof(T)));
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPadExtParams<T> padParams{};
        AscendC::DataCopyPad(local, srcGm, copyParams, padParams);
        bindQueue_.EnQue(local);
    }

    HYBM_AICORE_KERNEL void CopyOut(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<T> local = bindQueue_.DeQue<T>();
        AscendC::GlobalTensor<T> dstGm;
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ctx.dst), ctx.bytes / static_cast<uint32_t>(sizeof(T)));
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPad(dstGm, local, copyParams);
        bindQueue_.FreeTensor(local);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> bindQueue_;
    uint32_t aivIndex_;
    uint32_t aivNum_;
    uint32_t size_;
    __gm__ uint64_t *inputs_;
    __gm__ uint64_t *outputs_;
    __gm__ uint32_t *lens_;
};

template<typename T>
class OffloadSparseCopyKernel {
public:
    HYBM_AICORE_KERNEL OffloadSparseCopyKernel() {}

    HYBM_AICORE_KERNEL void Init(AscendC::TPipe *pipe, GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
    {
        aivNum_ = AscendC::GetBlockNum();
        size_ = *(reinterpret_cast<__gm__ uint32_t *>(size));
        if (size_ >= aivNum_) {
            kvCopy_.Init(pipe, inputs, outputs, lens, size);
        } else {
            moeCopy_.Init(pipe, inputs, outputs, lens, size);
        }
    }

    HYBM_AICORE_KERNEL void Process()
    {
        if (size_ >= aivNum_) {
            kvCopy_.Process();
        } else {
            moeCopy_.Process();
        }
    }

private:
    uint32_t size_;
    uint32_t aivNum_;
    OffloadKVSparseCopyKernel<T> kvCopy_;
    OffloadMoeSparseCopyKernel<T> moeCopy_;
};

#endif // ACC_OFFLOAD_SPARSE_COPY_H
