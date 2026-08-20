#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ctypes
from _pymf_acc_offload import offload

sparse_copy_impl = offload.sparse_copy
sparse_copy_urma_impl = offload.sparse_copy_urma
npu_kvcache_scatter_copy_impl = offload.npu_kvcache_scatter_copy
group_pack_copy_impl = offload.group_pack_copy


def empty(sizes, dtype=None, pin_memory=False):
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    numel = 1
    for size in sizes:
        numel *= size
    element_size = numel * torch.tensor([], dtype=dtype).element_size()
    ptr = offload.malloc(element_size)
    if ptr == 0:
        raise Exception("malloc failed")
    buf = (ctypes.c_int8 * element_size).from_address(ptr)
    return torch.frombuffer(buf, dtype=dtype).reshape(sizes)


def sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId):
    return sparse_copy_impl(
        srcPtrs.data_ptr(), dstPtrs.data_ptr(), lenPtrs.data_ptr(), sizePtr.data_ptr(), deviceId.index
    )


def sparse_copy_urma(src_ptrs, dst_ptrs, len_ptrs, list_num, device):
    return sparse_copy_urma_impl(
        src_ptrs.data_ptr(), dst_ptrs.data_ptr(), len_ptrs.data_ptr(), int(list_num), device.index
    )


def npu_kvcache_scatter_copy(
    hbm_k_rope,
    hbm_kv_cache,
    dram_k_rope,
    dram_kv_cache,
    hbm_block_table,
    dram_block_table,
    offload_slots,
    src_token_ids,
    dst_slots,
    copy_counts,
    ready_flag=None,
    layer_id=0,
):
    del dram_k_rope, dram_kv_cache
    ret = npu_kvcache_scatter_copy_impl(
        hbm_k_rope.data_ptr(), hbm_kv_cache.data_ptr(), hbm_block_table.data_ptr(), dram_block_table.data_ptr(),
        offload_slots.data_ptr(), src_token_ids.data_ptr(), dst_slots.data_ptr(), copy_counts.data_ptr(),
        0 if ready_flag is None else ready_flag.data_ptr(), hbm_k_rope.shape[0], hbm_block_table.shape[1],
        dram_block_table.shape[1], dram_block_table.shape[0], copy_counts.shape[0], int(layer_id),
        hbm_k_rope.device.index,
    )
    if ret != 0:
        raise RuntimeError(f"npu_kvcache_scatter_copy failed: ret={ret}")


def group_pack_copy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, deviceId):
    return group_pack_copy_impl(
        srcPtrs.data_ptr(),
        dstPtrs.data_ptr(),
        lenPtrs.data_ptr(),
        numLocalExpertPtr.data_ptr(),
        groupList.data_ptr(),
        packedGroupList.data_ptr(),
        deviceId.index,
    )
