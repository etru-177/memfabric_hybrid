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

#ifndef MEM_FABRIC_HYBRID_ACC_OFFLOAD_HYBM_BATCH_COPY_H
#define MEM_FABRIC_HYBRID_ACC_OFFLOAD_HYBM_BATCH_COPY_H

#include <cstddef>
#include <cstdint>

// 单个完成轮次预留一半SQ空间，避免数据WQE和末尾完成标志WQE在执行前覆盖尚未消费的队列项。
constexpr uint32_t kHybmBatchCopyMaxRoundDescriptors = 16000U;

struct HybmBatchCopyParam {
    uint32_t list_num;
    void **dst_buf_addr_list;
    void **src_buf_addr_list;
    uint64_t *len_list;
};

static_assert(offsetof(HybmBatchCopyParam, list_num) == 0x00U);
static_assert(offsetof(HybmBatchCopyParam, dst_buf_addr_list) == 0x08U);
static_assert(offsetof(HybmBatchCopyParam, src_buf_addr_list) == 0x10U);
static_assert(offsetof(HybmBatchCopyParam, len_list) == 0x18U);
static_assert(sizeof(HybmBatchCopyParam) == 0x20U);

extern "C" uint32_t HybmBatchCopy(HybmBatchCopyParam *param);

#endif // MEM_FABRIC_HYBRID_ACC_OFFLOAD_HYBM_BATCH_COPY_H
