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

copy_impl = offload.copy
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


def sparse_copy_inner(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId, flag=0):
    """Batched variable-length copy. flag=0 (default) runs the sparse_copy
    kernel, flag=1 runs the varlen_copy kernel.
    """
    return copy_impl(
        srcPtrs.data_ptr(), dstPtrs.data_ptr(), lenPtrs.data_ptr(), sizePtr.data_ptr(), deviceId.index, flag
    )


def sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId):
    return sparse_copy_inner(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId, flag=0)


def varlen_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId):
    return sparse_copy_inner(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId, flag=1)


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
