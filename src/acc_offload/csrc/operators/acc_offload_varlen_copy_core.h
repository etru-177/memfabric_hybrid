/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
 */

#ifndef ACC_OFFLOAD_VARLEN_COPY_CORE_H
#define ACC_OFFLOAD_VARLEN_COPY_CORE_H

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

/* A5 gives each vector core 256KB of UB; TPipe reserves a small system area,
 * so 240KB is the safe user ceiling for one full-batch tensor. */
constexpr int64_t UB_ONCE_SIZE = 240 * 1024;

/* Upper bound of copy entries per launch; a larger count copies nothing to avoid
 * wild reads when the size buffer contains garbage. */
constexpr uint32_t SPARSE_COPY_MAX_ENTRIES = 1U << 20;

/*
 * Core implementation of the batched variable-length copy (DRAM<->HBM gather):
 * entry i copies lens[i] elements from srcPtrs[i] to dstPtrs[i]. No K/V split
 * assumption, even and odd entry counts are fully copied. All src/dst/len/size
 * buffers live in GM; 64-bit pointers must be read directly from GM (scalar
 * reads), never staged through UB (LocalTensor::GetValue corrupts 64-bit
 * values on A5).
 *
 * Copy strategy (tuned for the DRAM->HBM swap-in shape: ~656B entries,
 * tens of thousands per layer):
 *  - entry count >= core count: entries are assigned to cores by interleaving
 *    (i % blockNum), so any entry count spreads over all cores and uneven
 *    lengths stay balanced;
 *  - entry count < core count: byte-balanced split of the total payload across
 *    cores (few large entries are parallelized instead of pinning each entry
 *    to one core);
 *  - every segment is chunked into slot-sized units (<= 120KB) that flow
 *    through a two-slot ping-pong pipeline: the GM->UB copy-in (MTE2) of
 *    unit k runs while the UB->GM copy-out (MTE3) of unit k-1 is still in
 *    flight, hiding the per-unit round-trip latency that dominated the old
 *    serial per-entry loop;
 *  - each unit keeps its own queue round trip (Alloc/EnQue/DeQue/Free) on
 *    its own slot tensor: a single sync covering multiple staged GM->UB
 *    copies corrupted the vector core (507035) on A5, so units are never
 *    packed into one shared UB tensor.
 *
 * The former optional completion-notification tail (notify flag + the
 * OffloadWaitFlagKernel consumer) was retired: as a MIX_AIV_1_0 op this
 * kernel shares the compute queue with its consumers, so the queue FIFO
 * already orders completion and the device-side wait never spun on real
 * work (verified band-preserving on gsm8k after removal; see sglang
 * hisparse debug notes).
 *
 * Both the OffloadSparseCopyKernel and OffloadVarlenCopyKernel entry classes
 * share this core; they exist as separate symbols so each aicore entry keeps
 * a distinct kernel name.
 */
template<typename T>
class OffloadVarlenCopyCore {
public:
    HYBM_AICORE_KERNEL OffloadVarlenCopyCore() {}

    HYBM_AICORE_KERNEL void Init(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size)
    {
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();

        uint32_t entryCount = *(reinterpret_cast<__gm__ uint32_t *>(size));
        /* entry count sanity guard: zero or over-limit copies nothing. */
        size_ = (entryCount > SPARSE_COPY_MAX_ENTRIES) ? 0 : entryCount;
        inputs_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
        outputs_ = reinterpret_cast<__gm__ uint64_t *>(outputs);
        lens_ = reinterpret_cast<__gm__ uint32_t *>(lens);
        /* two 120KB slots stay within the 240KB user UB ceiling */
        pipe_.InitBuffer(queA_, 1, SLOT_BYTES);
        pipe_.InitBuffer(queB_, 1, SLOT_BYTES);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        if (size_ >= aivNum_) {
            ProcessByEntry();
        } else {
            ProcessByBytes();
        }
        FlushPipe();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    HYBM_AICORE_KERNEL void ProcessByEntry()
    {
        for (uint32_t i = aivIndex_; i < size_; i += aivNum_) {
            auto src = reinterpret_cast<__gm__ T *>(inputs_[i]);
            auto dst = reinterpret_cast<__gm__ T *>(outputs_[i]);
            uint32_t len = lens_[i] * sizeof(T);
            for (uint32_t off = 0; off < len; off += SLOT_BYTES) {
                uint32_t cur = (len - off > SLOT_BYTES) ? SLOT_BYTES : (len - off);
                PushUnit(src + off, dst + off, cur);
            }
        }
    }

    HYBM_AICORE_KERNEL void ProcessByBytes()
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
                uint32_t offsetBytes = static_cast<uint32_t>(ovStart - curByte);
                uint32_t copyBytes = static_cast<uint32_t>(ovEnd - ovStart);
                auto src = reinterpret_cast<__gm__ T *>(inputs_[i]);
                auto dst = reinterpret_cast<__gm__ T *>(outputs_[i]);
                for (uint32_t off = 0; off < copyBytes; off += SLOT_BYTES) {
                    uint32_t cur = (copyBytes - off > SLOT_BYTES) ? SLOT_BYTES : (copyBytes - off);
                    PushUnit(src + offsetBytes + off, dst + offsetBytes + off, cur);
                }
            }
            curByte = entryEnd;
            if (curByte >= endByte) {
                break;
            }
        }
    }

    /* Pipeline writer: commit the previously staged unit, then stage the
     * new one into the other slot, so its copy-in overlaps the committed
     * unit's copy-out. Pending slot is always turn_ ^ 1. */
    HYBM_AICORE_KERNEL void PushUnit(__gm__ T *src, __gm__ T *dst, uint32_t len)
    {
        if (staged_) {
            CommitPending();
        }
        StageSlot(turn_, src, len);
        pendDst_ = dst;
        pendLen_ = len;
        staged_ = true;
        turn_ ^= 1;
    }

    HYBM_AICORE_KERNEL void FlushPipe()
    {
        if (staged_) {
            CommitPending();
        }
        staged_ = false;
    }

    /* Issue the GM->UB copy-in of one unit into the given slot. */
    HYBM_AICORE_KERNEL void StageSlot(uint32_t slot, __gm__ T *src, uint32_t len)
    {
        auto &que = (slot == 0) ? queA_ : queB_;
        AscendC::LocalTensor<T> local = que.AllocTensor<T>();
        inputGm_.SetGlobalBuffer(src, len);
        /* All slice operands are named lvalues: operator[] returns
         * temporaries that cannot bind to DataCopyPad's non-const
         * reference parameters on this AscendC version. */
        AscendC::GlobalTensor<T> srcGm = inputGm_[0];
        AscendC::DataCopyExtParams cpIn(1, (int32_t)len, 0, 0, 0);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(local, srcGm, cpIn, padParams);
        que.EnQue(local);
    }

    /* Drain the pending slot (waits only its own copy-in event) and issue
     * its UB->GM copy-out. */
    HYBM_AICORE_KERNEL void CommitPending()
    {
        auto &que = (turn_ ^ 1) == 0 ? queA_ : queB_;
        AscendC::LocalTensor<T> local = que.DeQue<T>();
        outputGm_.SetGlobalBuffer(pendDst_, pendLen_);
        AscendC::GlobalTensor<T> dstGm = outputGm_[0];
        AscendC::LocalTensor<T> out = local[0];
        AscendC::DataCopyExtParams cpOut(1, (int32_t)pendLen_, 0, 0, 0);
        AscendC::DataCopyPadExtParams<T> padParams;
        AscendC::DataCopyPad(dstGm, out, cpOut);
        que.FreeTensor(local);
    }

private:
    static constexpr uint32_t SLOT_BYTES = UB_ONCE_SIZE / 2; /* 120KB per slot */
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> queA_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> queB_;
    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;
    uint32_t aivNum_;
    uint32_t aivIndex_;
    uint32_t size_;
    uint32_t turn_ = 0; /* slot index the NEXT unit stages into */
    bool staged_ = false;
    __gm__ T *pendDst_ = nullptr;
    uint32_t pendLen_ = 0;
    __gm__ uint64_t *inputs_;
    __gm__ uint64_t *outputs_;
    __gm__ uint32_t *lens_;
};

#endif // ACC_OFFLOAD_VARLEN_COPY_CORE_H
