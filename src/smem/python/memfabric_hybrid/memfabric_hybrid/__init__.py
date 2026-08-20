#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import ctypes

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(current_dir, 'lib')
for lib in ["libmf_hybm_core.so", "libmf_smem.so"]:
    ctypes.CDLL(os.path.join(libs_path, lib), mode=ctypes.RTLD_GLOBAL)

offload_lib_path = os.path.join(libs_path, "libmf_acc_offload.so")
offload_lib_loaded = os.path.isfile(offload_lib_path)
if offload_lib_loaded:
    ctypes.CDLL(offload_lib_path, mode=ctypes.RTLD_GLOBAL)

# Preload optional dlopen-ed libraries (may not be packaged depending on build options)
optional_libs = ["libboundscheck.so", "libhcom.so", "libetcd_client_v3.so"]
for lib in optional_libs:
    try:
        ctypes.CDLL(os.path.join(libs_path, lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


def get_include_path():
    package_dir = os.path.dirname(__file__)
    include_path = os.path.join(package_dir, 'include')
    return os.path.abspath(include_path)


def get_lib_path():
    package_dir = os.path.dirname(__file__)
    lib_path = os.path.join(package_dir, 'lib')
    return os.path.abspath(lib_path)


from _pymf_transfer import TransferEngine, TransferOpcode, create_config_store
from _pymf_hybrid import (
    bm,
    shm,
    initialize,
    uninitialize,
    set_log_level,
    set_extern_logger,
    get_last_err_msg,
    set_conf_store_tls,
    set_conf_store_tls_key,
    get_and_clear_last_err_msg,
)
__all__ = [
    'TransferEngine',
    'TransferOpcode',
    'create_config_store',
    'bm',
    'shm',
    'initialize',
    'uninitialize',
    'set_log_level',
    'set_extern_logger',
    'get_last_err_msg',
    'set_conf_store_tls',
    'set_conf_store_tls_key',
    'get_and_clear_last_err_msg',
]
__all__ += ['get_include_path', 'get_lib_path']

if offload_lib_loaded:
    from _pymf_acc_offload import offload
    from mf_acc_offload import empty, sparse_copy, sparse_copy_urma, npu_kvcache_scatter_copy, group_pack_copy

    offload.empty = empty
    offload.sparse_copy = sparse_copy
    offload.sparse_copy_urma = sparse_copy_urma
    offload.npu_kvcache_scatter_copy = npu_kvcache_scatter_copy
    offload.group_pack_copy = group_pack_copy
    __all__.append('offload')

# Import-time provisioning of HYBM AICPU Kernel (silent skip/install).
from memfabric_hybrid._provision import provision  # noqa: E402

provision()
