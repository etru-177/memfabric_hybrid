#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

import argparse
import ctypes
import os
import socket
import time

from urma_example_common import (
    DEFAULT_ENV_FILE,
    HBM_GVA_MAX_BYTES,
    HOST_RANK,
    NPU_RANK,
    STORE_PORT,
    WORLD_SIZE,
    _config,
)


CONTROL_BYTES = 3 * 4096
POOL_ALIGN = 2 << 20


class Request(ctypes.Structure):
    _fields_ = [
        ("host_mailbox_gva", ctypes.c_uint64),
        ("dst_new_gva", ctypes.c_uint64),
        ("ready_gva", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
        ("src_stride", ctypes.c_uint64),
        ("dst_stride", ctypes.c_uint64),
        ("segment_count", ctypes.c_uint32),
        ("segment_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    ]


class Message(ctypes.Structure):
    _fields_ = [("request", Request), ("doorbell", ctypes.c_uint64), ("padding", ctypes.c_uint8 * 56)]


class Timing(ctypes.Structure):
    _fields_ = [
        ("request_ns", ctypes.c_uint64),
        ("wait_host_ns", ctypes.c_uint64),
        ("scatter_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("padding", ctypes.c_uint8 * 32),
    ]


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def make_layout(count, segment_bytes):
    total = count * segment_bytes
    stride = 2 * segment_bytes
    source = 4096
    aggregate = align_up(source + count * stride, 4096)
    dst_new = CONTROL_BYTES
    dst_base = align_up(dst_new + total, 4096)
    pool_bytes = align_up(max(aggregate + total, dst_base + count * stride), POOL_ALIGN)
    return total, stride, source, aggregate, dst_new, dst_base, pool_bytes


def load_env(path):
    with open(path, encoding="utf-8") as env_file:
        for raw in env_file:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("export "):
                line = line[7:]
            name, value = line.split("=", 1)
            os.environ[name] = value.strip().strip("'\"")


def configure(role, env_file):
    load_env(env_file)
    physical = os.environ["MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID"]
    visible = [item.strip() for item in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
    runtime_device = visible.index(physical)
    if role == "host":
        os.environ["MF_LOCAL_DRAM_VALIDATION_ROLE"] = "host"
        os.environ["MF_HYBM_RDMA_SWAP_SPACE_SIZE"] = "0"
    else:
        os.environ.pop("MF_LOCAL_DRAM_VALIDATION_ROLE", None)
    import torch

    torch.npu.set_device(runtime_device)
    return runtime_device


def create_handle(args, bm, rank, runtime_device, pool_bytes):
    config = _config(bm, rank, args.head_ip)
    assert bm.initialize(f"tcp://{args.head_ip}:{args.store_port}", WORLD_SIZE, runtime_device, config) == 0
    is_host = rank == HOST_RANK
    handle = bm.create2(
        id=0,
        local_dram_size=pool_bytes if is_host else 0,
        max_dram_size=pool_bytes,
        local_hbm_size=0 if is_host else pool_bytes,
        max_hbm_size=HBM_GVA_MAX_BYTES,
        data_op_type=bm.BmDataOpType.HOST_DEVICE_URMA,
        enable_56bits_gva=False,
    )
    assert handle.join() == 0
    return handle


def run_host(args, handle, bm, listener, layout):
    total, stride, source_offset, aggregate_offset, _, _, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    host_va = handle.gva_to_va(host_gva, bm.BmMemType.LOCAL_HOST)
    mailbox = host_va
    source = host_va + source_offset
    aggregate = host_va + aggregate_offset
    ctypes.memset(mailbox, 0, ctypes.sizeof(Message))
    for index in range(args.segments):
        ctypes.memset(source + index * stride, index & 0xFF, args.segment_bytes)

    from _pymf_acc_offload import offload

    conn, _ = listener.accept()
    with conn:
        conn.sendall(b"R")
        dst_new_gva, ready_gva, request_bytes, wait_ns, gather_ns = offload.aggregate_wait_and_gather_demo(
            mailbox, source, aggregate
        )
        write_begin = time.perf_counter_ns()
        assert handle.copy_data(aggregate, dst_new_gva, request_bytes, bm.BmCopyType.H2G, 0) == 0
        write_end = time.perf_counter_ns()
        assert handle.copy_data(mailbox + Message.doorbell.offset, ready_gva, 8, bm.BmCopyType.H2G, 0) == 0
        ready_end = time.perf_counter_ns()
        conn.recv(1)
    print(
        f"host bytes={total} wait_us={wait_ns / 1e3:.3f} gather_us={gather_ns / 1e3:.3f} "
        f"write_us={(write_end - write_begin) / 1e3:.3f} ready_us={(ready_end - write_end) / 1e3:.3f}"
    )


def copy_to_hbm(handle, bm, source, destination, size):
    assert handle.copy_data(source, destination, size, bm.BmCopyType.H2G, 0) == 0


def run_npu(args, handle, bm, runtime_device, layout):
    total, stride, _, _, dst_new_offset, dst_base_offset, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    hbm_gva = handle.peer_rank_ptr(NPU_RANK, bm.BmMemType.DEVICE)
    hbm_va = handle.gva_to_va(hbm_gva, bm.BmMemType.LOCAL_DEVICE)
    message = Message(Request(host_gva, hbm_gva + dst_new_offset, hbm_gva + 4096, total, stride, stride,
                              args.segments, args.segment_bytes, 0), 1)
    zero = ctypes.c_uint64(0)
    timing = Timing()
    copy_to_hbm(handle, bm, ctypes.addressof(message), hbm_gva, ctypes.sizeof(message))
    copy_to_hbm(handle, bm, ctypes.addressof(zero), hbm_gva + 4096, ctypes.sizeof(zero))
    copy_to_hbm(handle, bm, ctypes.addressof(timing), hbm_gva + 8192, ctypes.sizeof(timing))

    with socket.create_connection((args.head_ip, args.ctrl_port)) as conn:
        conn.recv(1)
        library = ctypes.CDLL(os.path.join(os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"],
                                          "libmf_hybm_accoffload.so"))
        launch = library.AccOffloadAggregateUrmaDemo
        launch.argtypes = [ctypes.c_uint64] * 5 + [ctypes.c_uint16]
        launch.restype = ctypes.c_int32
        launch_begin = time.perf_counter_ns()
        assert launch(hbm_va, hbm_va + 4096, hbm_va + dst_new_offset, hbm_va + dst_base_offset, hbm_va + 8192,
                      runtime_device) == 0
        launch_end = time.perf_counter_ns()
        assert handle.copy_data(hbm_gva + 8192, ctypes.addressof(timing), ctypes.sizeof(timing),
                                bm.BmCopyType.G2H, 0) == 0
        conn.sendall(b"D")
    print(
        f"aicpu bytes={total} request_us={timing.request_ns / 1e3:.3f} "
        f"wait_host_us={timing.wait_host_ns / 1e3:.3f} scatter_us={timing.scatter_ns / 1e3:.3f} "
        f"e2e_us={timing.total_ns / 1e3:.3f} launch_sync_us={(launch_end - launch_begin) / 1e3:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="AICPU initiated Host aggregate URMA demo")
    parser.add_argument("--role", required=True, choices=("host", "device"))
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--store-port", type=int, default=STORE_PORT)
    parser.add_argument("--ctrl-port", type=int, default=9878)
    parser.add_argument("--segments", type=int, default=4096)
    parser.add_argument("--segment-bytes", type=int, default=2048)
    args = parser.parse_args()

    rank = HOST_RANK if args.role == "host" else NPU_RANK
    runtime_device = configure(args.role, args.env_file)
    layout = make_layout(args.segments, args.segment_bytes)
    listener = None
    if rank == HOST_RANK:
        listener = socket.create_server(("0.0.0.0", args.ctrl_port))
    import memfabric_hybrid as mf
    from memfabric_hybrid import bm

    assert mf.initialize() == 0
    handle = create_handle(args, bm, rank, runtime_device, layout[-1])
    if rank == HOST_RANK:
        run_host(args, handle, bm, listener, layout)
        listener.close()
    else:
        run_npu(args, handle, bm, runtime_device, layout)
    handle.leave()
    handle.destroy()
    bm.uninitialize(0)
    mf.uninitialize()


if __name__ == "__main__":
    main()
