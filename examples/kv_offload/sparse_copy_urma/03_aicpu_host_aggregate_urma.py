#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

import argparse
import ctypes
import json
import math
import multiprocessing
import os
import socket
import tempfile
import time
import traceback

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
DEFAULT_COUNTS = sorted({base * scale for base in (100, 200, 300, 400) for scale in (1, 2, 4, 8, 16, 32, 64)})
DEFAULT_SIZES = (576, 656, 1152, 8192)


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
        ("scatter_copy_ns", ctypes.c_uint64),
        ("scatter_publish_ns", ctypes.c_uint64),
        ("scatter_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("padding", ctypes.c_uint8 * 16),
    ]


CONTROL_STORAGE_BYTES = 2 * 4096 + ctypes.sizeof(Timing)
PACKED_CONTROL_COPY_BYTES = 4096 + 64


class HostAffinityError(ValueError):
    pass


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def parse_cpu_list(value):
    cpus = set()
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        if not bounds[0]:
            raise ValueError("empty CPU range")
        begin = int(bounds[0])
        end = int(bounds[-1])
        if begin < 0 or end < begin:
            raise ValueError(f"invalid CPU range: {item}")
        cpus.update(range(begin, end + 1))
    return cpus


def configure_host_affinity(cpu_list):
    value = cpu_list or os.environ.get("MF_LOCAL_DRAM_AFFINITY_CPUS", "")
    if not value or value == "unavailable":
        return
    try:
        cpus = parse_cpu_list(value)
        os.sched_setaffinity(0, cpus)
    except (OSError, ValueError) as error:
        raise HostAffinityError(str(error)) from error
    print(f"Host CPU affinity: {','.join(str(cpu) for cpu in sorted(cpus))}")


def summarize(values):
    ordered = sorted(values)
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    p99 = ordered[math.ceil(len(ordered) * 0.99) - 1]
    return sum(ordered) / len(ordered), ordered[0], ordered[-1], p95, p99


def print_table(title, headers, rows):
    widths = [max(len(str(header)), *(len(str(row[index])) for row in rows)) for index, header in enumerate(headers)]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(title)
    print(separator)
    print("| " + " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(str(value).rjust(widths[index]) for index, value in enumerate(row)) + " |")
    print(separator)


def print_timing_summary(title, stage_samples, stage_bytes):
    rows = []
    for stage, samples in stage_samples.items():
        average, minimum, maximum, p95, p99 = summarize(samples)
        byte_count = stage_bytes.get(stage)
        bandwidth = "-" if byte_count is None else f"{byte_count * 1e9 / average / (1024 ** 3):.3f}"
        rows.append((stage, f"{average / 1e3:.3f}", f"{minimum / 1e3:.3f}", f"{maximum / 1e3:.3f}",
                     f"{p95 / 1e3:.3f}", f"{p99 / 1e3:.3f}", bandwidth))
    print_table(title, ("stage", "avg(us)", "min(us)", "max(us)", "P95(us)", "P99(us)", "GiB/s"), rows)


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


def configure(role, env_file, host_cpu_list=None, force_host_nic_plugin=False):
    load_env(env_file)
    if role == "host":
        configure_host_affinity(host_cpu_list)
        os.environ["HCOMM_NIC_PLUGIN_FORCE_LOAD"] = "1" if force_host_nic_plugin else "0"
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


def timed_copy(handle, bm, source, destination, size):
    begin = time.perf_counter_ns()
    assert handle.copy_data(source, destination, size, bm.BmCopyType.H2G, 0) == 0
    return time.perf_counter_ns() - begin


def fill_source_pattern(source, stride, segment_count, segment_bytes):
    for index in range(segment_count):
        ctypes.memset(source + index * stride, index & 0xFF, segment_bytes)


def fill_destination_poison(destination, stride, segment_count, segment_bytes):
    ctypes.memset(destination, 0xA5, (segment_count - 1) * stride + segment_bytes)
    for index in range(segment_count):
        ctypes.memset(destination + index * stride, (index & 0xFF) ^ 0xFF, segment_bytes)


def run_host_round(args, handle, bm, offload, mailbox, source, aggregate, expected_doorbell):
    result = offload.aggregate_wait_demo(mailbox, expected_doorbell)
    dst_new_gva, ready_gva, total_bytes, src_stride, segment_count, segment_bytes, _ = result
    expected_total = args.segments * args.segment_bytes
    expected_stride = 2 * args.segment_bytes
    layout_matches = (total_bytes, src_stride, segment_count, segment_bytes) == (
        expected_total, expected_stride, args.segments, args.segment_bytes)
    if not layout_matches:
        raise RuntimeError("device request layout does not match host arguments")
    work_begin = time.perf_counter_ns()
    gather_ns = offload.aggregate_gather_range_demo(
        source, aggregate, src_stride, segment_count, segment_bytes, args.gather_threads
    )
    write_ns = timed_copy(handle, bm, aggregate, dst_new_gva, total_bytes)
    work_ns = time.perf_counter_ns() - work_begin
    return ready_gva, total_bytes, gather_ns, write_ns, work_ns


def signal_ready(handle, bm, mailbox, ready_gva):
    begin = time.perf_counter_ns()
    assert handle.copy_data(mailbox + Message.doorbell.offset, ready_gva, 8, bm.BmCopyType.H2G, 0) == 0
    return time.perf_counter_ns() - begin


def run_host(args, handle, bm, listener, layout):
    total, stride, source_offset, aggregate_offset, _, _, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    host_va = handle.gva_to_va(host_gva, bm.BmMemType.LOCAL_HOST)
    mailbox = host_va
    source = host_va + source_offset
    aggregate = host_va + aggregate_offset
    ctypes.memset(mailbox, 0, ctypes.sizeof(Message))
    fill_source_pattern(source, stride, args.segments, args.segment_bytes)

    from _pymf_acc_offload import offload

    conn, _ = listener.accept()
    stage_names = ("gather", "URMA write", "host total")
    stages = {name: [] for name in stage_names}
    offload.aggregate_gather_range_demo(source, aggregate, stride, 0, args.segment_bytes, args.gather_threads)
    with conn:
        conn.sendall(b"R")
        for round_index in range(args.rounds):
            expected_doorbell = round_index + 1
            ready_gva, _, gather_ns, write_ns, work_ns = run_host_round(
                args, handle, bm, offload, mailbox, source, aggregate, expected_doorbell
            )
            ready_ns = signal_ready(handle, bm, mailbox, ready_gva)
            stages["gather"].append(gather_ns)
            stages["URMA write"].append(write_ns)
            stages["host total"].append(work_ns + ready_ns)
        if conn.recv(1) != b"D":
            raise RuntimeError("device exited before finishing all rounds")
    if getattr(args, "result_file", None):
        return stages
    stage_bytes = {name: total for name in stages}
    print_timing_summary(
        f"Host summary: rounds={args.rounds}, gather_threads={args.gather_threads}, bytes/round={total}",
        stages,
        stage_bytes,
    )


def copy_to_hbm(handle, bm, source, destination, size):
    assert handle.copy_data(source, destination, size, bm.BmCopyType.H2G, 0) == 0


def run_direct_host(args, handle, bm, listener, layout):
    _, stride, source_offset, _, _, _, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    host_va = handle.gva_to_va(host_gva, bm.BmMemType.LOCAL_HOST)
    fill_source_pattern(host_va + source_offset, stride, args.segments, args.segment_bytes)
    with listener.accept()[0] as conn:
        conn.sendall(b"R")
        if conn.recv(1) != b"D":
            raise RuntimeError("direct device exited before finishing all rounds")
    return {}


def direct_copy_lists(host_gva, hbm_va, layout, count, size):
    _, stride, source_offset, _, _, dst_offset, _ = layout
    # Source is a routed GVA; destination is the local device VA, not its GVA.
    return ([host_gva + source_offset + index * stride for index in range(count)],
            [hbm_va + dst_offset + index * stride for index in range(count)], [size] * count)


def run_direct_npu(args, handle, bm, runtime_device, layout):
    import torch

    total, stride, _, _, _, dst_offset, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    hbm_gva = handle.peer_rank_ptr(NPU_RANK, bm.BmMemType.DEVICE)
    hbm_va = handle.gva_to_va(hbm_gva, bm.BmMemType.LOCAL_DEVICE)
    lists = direct_copy_lists(host_gva, hbm_va, layout, args.segments, args.segment_bytes)
    tensors = [torch.tensor(values, dtype=torch.int64, device=f"npu:{runtime_device}") for values in lists]
    torch.npu.synchronize()
    library = ctypes.CDLL(os.path.join(os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"],
                                      "libmf_hybm_accoffload.so"))
    launch = library.AccOffloadSparseCopyUrma
    launch.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_uint32, ctypes.c_uint16]
    launch.restype = ctypes.c_int32
    pointers = [tensor.data_ptr() for tensor in tensors]
    span = (args.segments - 1) * stride + args.segment_bytes
    readback = (ctypes.c_uint8 * span)() if args.verify else None
    poison = (ctypes.c_uint8 * span)() if args.verify else None
    stages = {"launch sync": []}
    with socket.create_connection((args.head_ip, args.ctrl_port)) as conn:
        if conn.recv(1) != b"R":
            raise RuntimeError("direct host did not publish source readiness")
        for round_index in range(args.rounds):
            if args.verify:
                fill_destination_poison(ctypes.addressof(poison), stride, args.segments, args.segment_bytes)
                copy_to_hbm(handle, bm, ctypes.addressof(poison), hbm_gva + dst_offset, span)
            begin = time.perf_counter_ns()
            ret = launch(*pointers, args.segments, runtime_device)
            elapsed = time.perf_counter_ns() - begin
            if ret != 0:
                raise RuntimeError(f"direct batch read failed: round={round_index}, ret={ret}")
            stages["launch sync"].append(elapsed)
            if args.verify:
                verify_scatter(handle, bm, hbm_gva, dst_offset, args, round_index, readback)
        conn.sendall(b"D")
    if not getattr(args, "result_file", None):
        print_timing_summary("Direct batch read", stages, {"launch sync": total})
        if args.verify:
            print(f"Direct verification: PASS ({args.rounds} rounds)")
    return stages


def make_device_control(request):
    storage = (ctypes.c_uint8 * CONTROL_STORAGE_BYTES)()
    message = Message.from_buffer(storage, 0)
    message.request = request
    ready = ctypes.c_uint64.from_buffer(storage, 4096)
    timing = Timing.from_buffer(storage, 8192)
    return storage, message, ready, timing


def stage_device_control(handle, bm, control, hbm_gva):
    copy_to_hbm(handle, bm, ctypes.addressof(control), hbm_gva, PACKED_CONTROL_COPY_BYTES)


def verify_scatter(handle, bm, hbm_gva, dst_base_offset, args, round_index, readback):
    span = (args.segments - 1) * (2 * args.segment_bytes) + args.segment_bytes
    assert handle.copy_data(hbm_gva + dst_base_offset, ctypes.addressof(readback), span,
                            bm.BmCopyType.G2H, 0) == 0
    actual = memoryview(readback).cast("B")
    stride = 2 * args.segment_bytes
    for index in range(args.segments):
        begin = index * stride
        expected = index & 0xFF
        mismatch = next((offset for offset, value in enumerate(actual[begin:begin + args.segment_bytes])
                         if value != expected), None)
        if mismatch is not None:
            raise RuntimeError(
                f"scatter mismatch: round={round_index}, segment={index}, offset={mismatch}, "
                f"expected=0x{expected:02x}, actual=0x{actual[begin + mismatch]:02x}"
            )


def run_npu(args, handle, bm, runtime_device, layout):
    total, stride, _, _, dst_new_offset, dst_base_offset, _ = layout
    host_gva = handle.peer_rank_ptr(HOST_RANK, bm.BmMemType.HOST)
    hbm_gva = handle.peer_rank_ptr(NPU_RANK, bm.BmMemType.DEVICE)
    hbm_va = handle.gva_to_va(hbm_gva, bm.BmMemType.LOCAL_DEVICE)
    request = Request(host_gva, hbm_gva + dst_new_offset, hbm_gva + 4096, total, stride, stride,
                      args.segments, args.segment_bytes, 0)
    control, message, ready, timing = make_device_control(request)
    timing_enabled = args.device_timing_every > 0
    scatter_span = (args.segments - 1) * stride + args.segment_bytes
    readback = (ctypes.c_uint8 * scatter_span)() if args.verify else None
    poison = (ctypes.c_uint8 * scatter_span)() if args.verify else None
    stages = {"launch sync": []}
    if timing_enabled:
        stages.update({"scatter copy": [], "publish barrier": [], "scatter total": [], "AICPU e2e": []})
    with socket.create_connection((args.head_ip, args.ctrl_port)) as conn:
        conn.recv(1)
        library = ctypes.CDLL(os.path.join(os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"],
                                          "libmf_hybm_accoffload.so"))
        launch = library.AccOffloadAggregateUrmaDemo
        launch.argtypes = [ctypes.c_uint64] * 5 + [ctypes.c_uint16]
        launch.restype = ctypes.c_int32
        for round_index in range(args.rounds):
            if args.verify:
                fill_destination_poison(ctypes.addressof(poison), stride, args.segments, args.segment_bytes)
                copy_to_hbm(handle, bm, ctypes.addressof(poison), hbm_gva + dst_base_offset, scatter_span)
            message.doorbell = round_index + 1
            ready.value = 0
            stage_device_control(handle, bm, control, hbm_gva)
            launch_begin = time.perf_counter_ns()
            assert launch(hbm_va, hbm_va + 4096, hbm_va + dst_new_offset, hbm_va + dst_base_offset,
                          hbm_va + 8192, runtime_device) == 0
            launch_end = time.perf_counter_ns()
            stages["launch sync"].append(launch_end - launch_begin)
            if args.verify:
                verify_scatter(handle, bm, hbm_gva, dst_base_offset, args, round_index, readback)
            timing_due = timing_enabled and (round_index + 1) % args.device_timing_every == 0
            if timing_due or (timing_enabled and round_index + 1 == args.rounds):
                assert handle.copy_data(hbm_gva + 8192, ctypes.addressof(timing), ctypes.sizeof(timing),
                                        bm.BmCopyType.G2H, 0) == 0
                stages["scatter copy"].append(timing.scatter_copy_ns)
                stages["publish barrier"].append(timing.scatter_publish_ns)
                stages["scatter total"].append(timing.scatter_ns)
                stages["AICPU e2e"].append(timing.total_ns)
        conn.sendall(b"D")
    if getattr(args, "result_file", None):
        return stages
    stage_bytes = {"launch sync": total, "scatter copy": total, "scatter total": total, "AICPU e2e": total}
    timing_samples = len(stages["AICPU e2e"]) if timing_enabled else 0
    print_timing_summary(
        f"Device summary: rounds={args.rounds}, timing_samples={timing_samples}, bytes/round={total}",
        stages,
        stage_bytes,
    )
    if args.verify:
        print(f"Scatter verification: PASS ({args.rounds} rounds, {args.segments} segments/round)")


def parse_args():
    parser = argparse.ArgumentParser(description="AICPU initiated Host aggregate URMA demo")
    parser.add_argument("--mode", choices=("aggregate", "direct"), default="aggregate",
                        help="aggregate via Host, or direct AICPU batch read into final HBM")
    parser.add_argument("--role", choices=("host", "device"), help=argparse.SUPPRESS)
    parser.add_argument("--head-ip", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--store-port", type=int, default=STORE_PORT, help=argparse.SUPPRESS)
    parser.add_argument("--ctrl-port", type=int, default=9878, help=argparse.SUPPRESS)
    parser.add_argument("--segments", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--segment-bytes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--case-timeout", type=float, default=600,
                        help="maximum seconds per case, including initialization and cleanup")
    parser.add_argument("--gather-threads", type=int, default=1)
    parser.add_argument("--host-cpus", help="Host process CPU list, for example 48-63")
    plugin_group = parser.add_mutually_exclusive_group()
    plugin_group.add_argument("--force-host-nic-plugin", dest="force_host_nic_plugin", action="store_true",
                              help=argparse.SUPPRESS)
    plugin_group.add_argument("--no-host-nic-plugin", dest="force_host_nic_plugin", action="store_false",
                              help="use the built-in Host URMA path instead of forcing the HCOMM NIC plugin")
    parser.set_defaults(force_host_nic_plugin=True)
    parser.add_argument("--device-timing-every", type=int, default=1,
                        help="fetch AICPU timing every N rounds; 0 disables timing G2H")
    parser.add_argument("--verify", action="store_true",
                        help="read back and verify every scatter round; excluded from launch timing")
    args = parser.parse_args()
    if min(args.segments) <= 0 or min(args.segment_bytes) <= 0 or args.rounds <= 0:
        parser.error("--segments, --segment-bytes and --rounds must be positive")
    if not 1 <= args.gather_threads <= 64:
        parser.error("--gather-threads must be in [1, 64]")
    if args.device_timing_every < 0:
        parser.error("--device-timing-every must be non-negative")
    if not math.isfinite(args.case_timeout) or args.case_timeout <= 0:
        parser.error("--case-timeout must be finite and positive")
    if args.role and (len(args.segments) != 1 or len(args.segment_bytes) != 1):
        parser.error("manual --role requires one --segments and one --segment-bytes value")
    return args


def run_role(args, listener=None):
    rank = HOST_RANK if args.role == "host" else NPU_RANK
    runtime_device = configure(args.role, args.env_file, args.host_cpus, args.force_host_nic_plugin)
    layout = make_layout(args.segments, args.segment_bytes)
    if rank == HOST_RANK and listener is None:
        listener = socket.create_server(("0.0.0.0", args.ctrl_port))
    import memfabric_hybrid as mf
    from memfabric_hybrid import bm

    assert mf.initialize() == 0
    handle = create_handle(args, bm, rank, runtime_device, layout[-1])
    if rank == HOST_RANK:
        host_runner = run_direct_host if args.mode == "direct" else run_host
        stages = host_runner(args, handle, bm, listener, layout)
        listener.close()
    else:
        device_runner = run_direct_npu if args.mode == "direct" else run_npu
        stages = device_runner(args, handle, bm, runtime_device, layout)
    handle.leave()
    handle.destroy()
    bm.uninitialize(0)
    mf.uninitialize()
    if getattr(args, "result_file", None):
        with open(args.result_file, "w", encoding="utf-8") as output:
            json.dump({name: sum(samples) / len(samples) for name, samples in stages.items()}, output)


def case_worker(args, log_path, listener=None):
    # Fresh interpreter: never fork an initialized Torch/NPU runtime.
    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            run_role(args, listener)
        except BaseException:
            traceback.print_exc(file=log)
            log.flush()
            raise
        finally:
            if listener is not None:
                listener.close()


def wait_workers(processes, timeout):
    deadline = time.monotonic() + timeout
    while any(process.is_alive() for process in processes):
        for process in processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"case exceeded {timeout:g} seconds")
        time.sleep(0.1)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{process.name} exited with code {process.exitcode}")


def run_case(args, size, count, directory):
    context = multiprocessing.get_context("spawn")
    processes = []
    # Pass the LIVE listener through spawn; the Host never rebinds this port.
    # Use the same wildcard bind scope as the manual Host path.
    ctrl = socket.create_server(("0.0.0.0", 0))
    try:
        # BM currently accepts a port, not an existing listening socket.
        # Unlike the control socket, its bind cannot be transferred here.
        with socket.create_server(("0.0.0.0", 0)) as store:
            ports = store.getsockname()[1], ctrl.getsockname()[1]
        for role in ("host", "device"):
            worker = argparse.Namespace(**vars(args))
            worker.role, worker.segment_bytes, worker.segments = role, size, count
            worker.head_ip = "127.0.0.1"
            worker.store_port, worker.ctrl_port = ports
            worker.result_file = os.path.join(directory, f"{role}.json")
            process = context.Process(target=case_worker,
                                      args=(worker, os.path.join(directory, f"{role}.log"),
                                            ctrl if role == "host" else None),
                                      name=role)
            process.start()
            processes.append(process)
        wait_workers(processes, args.case_timeout)
        results = {}
        for role in ("host", "device"):
            with open(os.path.join(directory, f"{role}.json"), encoding="utf-8") as result:
                results.update(json.load(result))
        return results
    finally:
        ctrl.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def print_case_errors(directory):
    for role in ("host", "device"):
        path = os.path.join(directory, f"{role}.log")
        print(f"--- {path} (last 8 KiB) ---", flush=True)
        try:
            with open(path, "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                print(log.read().decode("utf-8", errors="replace"), flush=True)
        except OSError as error:
            print(f"Cannot read log: {error}", flush=True)


def run_suite(args):
    directory = tempfile.mkdtemp(prefix="mf_aggregate_suite_")
    rows = []
    sizes, counts = sorted(set(args.segment_bytes)), sorted(set(args.segments))
    print(f"rounds/case={args.rounds}, cases={len(sizes) * len(counts)}, logs={directory}", flush=True)
    try:
        for size in sizes:
            for count in counts:
                case_dir = os.path.join(directory, f"{size}B_{count}")
                os.mkdir(case_dir)
                print(f"Testing {size}B x {count} ...", flush=True)
                try:
                    result = run_case(args, size, count, case_dir)
                except Exception as error:
                    print_case_errors(case_dir)
                    raise RuntimeError(f"case {size}B x {count} failed; inspect {case_dir}: {error}") from error
                rows.append((size, count, f"{size * count / (1024 ** 2):.3f}",
                             *(f"{result[name] / 1000:.3f}" if name in result else "-" for name in
                               ("launch sync", "host total", "gather", "URMA write")),
                             f"{result['scatter total'] / 1000:.3f}" if "scatter total" in result else "-",
                             f"{size * count * 1e9 / result['launch sync'] / (1024 ** 3):.3f}"))
    finally:
        if rows:
            print_table(f"{args.mode} copy summary (per-round means; E2E = launch sync; "
                        f"verify={'PASS' if args.verify else 'OFF'})",
                        ("bytes/pkt", "packets", "MiB", "E2E(us)", "host(us)", "gather(us)",
                         "write(us)", "scatter(us)", "E2E GiB/s"), rows)
        print(f"Full Host/Device logs and mean timings: {directory}", flush=True)


def main():
    args = parse_args()
    if args.role:
        args.segments, args.segment_bytes = args.segments[0], args.segment_bytes[0]
        run_role(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
