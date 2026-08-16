#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""PD 分离 KV offload 多对多端到端示例(URMA: N 张 Prefill 卡并发直写 M 张 Decode 卡的 DRAM 池).

流程:
  - Decode: 一条命令按 --npu-id 0,1 拉起 M 进程 M 卡, 每卡一个 DRAM 池按 writer 数
    划分 region(每 writer 128MiB); rank0 兼任 config store 与 meta TCPStore server,
    各 rank 布局(region 数/池基址/unique_id)自动发布, 免手工传地址;
  - Prefill: 一条命令按 --npu-id 0,1,2,3 拉起 N 进程 N 卡, 卡 r 把自己的 region r
    逐档(128M/32M/4M/128K)并发直写全部 M 张卡; 起跑屏障对齐 N 个 writer, 按
    (目标卡, 档位)汇总聚合写带宽;
  - Decode: 轮询全部 region 尾标 -> 结束屏障等全部 writer 计时完成 -> 逐 region
    校验 -> region0 逐档 sparse_copy(DVA->HBM) 提升并统计带宽.

数据公式按 (decode 卡, writer 卡) 区分, 写错卡或错位都无法通过校验.
平台: DEVICE_URMA 仅支持 ASCEND_950 (A5), 本脚本仅在 A5 上测试过; 见同目录 README.md.
卡数上限: prefill writer 与 decode 接收卡各 <= 10(目标场景 8x8).
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time
from datetime import timedelta
from queue import Empty

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from memfabric_hybrid import TransferEngine, offload, set_log_level

OP = "DEVICE_URMA"
MAX_BYTES = 128 << 20  # 每个 writer region 128 MiB
SIZES = [128 << 20, 32 << 20, 4 << 20, 128 << 10]  # 写入: 递减序(大->小)
PROMOTE_SIZES = list(reversed(SIZES))  # 提升: 递增序(小->大)
GB = 1 << 30
WAIT_TIMEOUT_S = 300
MAX_CARDS = 10  # 每侧卡数上限(variant 编码约束)

META_LAYOUT_KEY = "trans_offload_multi/layout/{}"  # 每个 decode rank 的布局
META_GLOBAL_KEY = "trans_offload_multi/num_decode"  # decode rank0 发布 decode 卡数
BARRIER_KEY = "trans_offload_multi/writer_barrier"  # writer 起跑屏障计数
WRITER_DONE_KEY = "trans_offload_multi/writer_done"  # writer 计时全部完成的结束屏障计数
META_TIMEOUT = timedelta(seconds=600)


def size_magic(size: int) -> int:
    """每档独立的完成标志: 0x5A5A5A0X, X 为档位序号(1..4)."""
    return 0x5A5A5A00 + SIZES.index(size) + 1


def fmt_size(size: int) -> str:
    return f"{size >> 20}MiB" if size >= (1 << 20) else f"{size >> 10}KiB"


def data_variant(decode_rank: int, writer_rank: int) -> int:
    """region 数据公式编号, (decode 卡, writer 卡) 唯一且 < 251 不折叠."""
    return 1 + MAX_CARDS * decode_rank + writer_rank


def build_expected(variant: int) -> np.ndarray:
    """按位置生成 128M 期望数据, 四个档位尾部植入各自 MAGIC."""
    pos = np.arange(MAX_BYTES, dtype=np.uint64)
    arr = ((pos * 7 + 3 + variant) % 251).astype(np.uint8)
    for size in SIZES:
        arr[size - 4 : size] = np.frombuffer(size_magic(size).to_bytes(4, "little"), dtype=np.uint8)
    return arr


def read_hva(addr: int, nbytes: int) -> np.ndarray:
    """经 HVA (offload 池的 host 虚拟地址) 直接读 DRAM 内容."""
    return np.frombuffer(ctypes.string_at(addr, nbytes), dtype=np.uint8).copy()


def bench_npu(fn, num_warmups: int = 2, num_tests: int = 10) -> float:
    """NPU event 计时: warmup + 每轮清 L2 + device event 多轮计时, 返回平均耗时(秒)."""
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device="npu")
    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(num_tests):
        cache.zero_()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # ms -> s
    return sum(times) / len(times)


def meta_store_server(port: int):
    """Decode 侧 TCPStore server(bind 0.0.0.0): 布局发布通道, 自动地址交换."""
    return dist.TCPStore(host_name="0.0.0.0", port=port, is_master=True, timeout=META_TIMEOUT)


def meta_store_client(host: str, port: int, timeout: timedelta = META_TIMEOUT):
    """Prefill/Decode 子进程侧 TCPStore client."""
    return dist.TCPStore(host_name=host, port=port, is_master=False, timeout=timeout)


def store_wait_counter(store, key: str, expected: int, timeout_s: float = 300.0) -> None:
    """只读等待 key 计数到达 expected(不自增; add(key,0) 兼容 key 尚不存在的首轮等待):
    轮询 get 判断到齐, 超时打印当前计数并抛错.
    """
    store.add(key, 0)
    deadline = time.time() + timeout_s
    while True:
        cur = int(store.get(key).decode())
        if cur >= expected:
            return
        if time.time() > deadline:
            print(f"[ERROR] store counter wait timeout: key={key}, expected={expected}, got={cur}")
            raise TimeoutError(f"store counter wait {key} timeout")
        time.sleep(0.05)


def store_barrier(store, key: str, world_size: int, timeout_s: float = 300.0) -> None:
    """TCPStore 计数器屏障: 各进程对 key 原子 +1, 计数到齐后同时放行."""
    store.add(key, 1)
    store_wait_counter(store, key, world_size, timeout_s)


def child_unique_id(base: str, rank: int) -> str:
    """基于 ip:port 基址为子进程分配独立 unique_id(port+rank)."""
    idx = base.rfind(":")
    if idx > 0 and base[idx + 1 :].isdigit():
        return f"{base[:idx]}:{int(base[idx + 1 :]) + rank}"
    return base


def init_engine(args, uid: str, role: str, npu_id: int, server_role: str, retry_s: int):
    """初始化 TransferEngine; retry_s>0 时(如 decode rank>=1 等 rank0 建服)重试,
    彻底失败抛错.
    """
    deadline = time.time() + retry_s
    while True:
        engine = TransferEngine()
        try:
            ret = engine.initialize(
                args.store_url, uid, role, npu_id, getattr(TransferEngine.TransDataOpType, OP), server_role
            )
        except Exception as exc:
            ret, err = -1, exc
        else:
            err = None
        if ret == 0:
            return engine
        if time.time() >= deadline:
            raise RuntimeError(
                f"engine initialize failed: uid={uid}, role={role}, server_role={server_role}, ret={ret}, err={err}"
            )
        time.sleep(2)


def timed_write(engine, dst_id: str, src: int, dst: int, size: int, tag: str) -> float:
    """单目标同步写, NPU event 多轮计时并打印该档带宽; 返回平均耗时(秒)."""
    tests = 3 if size >= (32 << 20) else 10
    elapsed = bench_npu(lambda: engine.batch_transfer_sync_write(dst_id, [src], [dst], [size]), num_tests=tests)
    print(
        f"[Prefill{tag}] write {size / 1e6:7.1f} MB: "
        f"{elapsed * 1e3:.1f} ms(avg of {tests}) => {size / elapsed / 1e9:.3f} GB/s",
        flush=True,
    )
    return elapsed


def join_procs(procs, timeout_s: float = 300.0) -> bool:
    """带超时兜底的 join: 整体 deadline 内未退出的进程 terminate, 避免父进程无限等待."""
    deadline = time.time() + timeout_s
    for p in procs:
        p.join(max(0.0, deadline - time.time()))
    stuck = [p for p in procs if p.is_alive()]
    if not stuck:
        return True
    print(f"[ERROR] process(es) stuck (>{timeout_s:.0f}s), terminating: pids {[p.pid for p in stuck]}")
    for p in stuck:
        p.terminate()
    for p in procs:
        p.join(5)
    return False


def publish_layout(store, rank: int, num_regions: int, dram_base: int, uid: str) -> None:
    """注册完成后发布本 decode rank 的布局("|" 分隔)."""
    store.set(META_LAYOUT_KEY.format(rank), f"{num_regions}|{dram_base}|{uid}")


def fetch_layout(store, rank: int):
    """取回第 rank 个 decode 布局, 解析失败打印错误并返回 None."""
    try:
        num_regions, dram_base, uid = store.get(META_LAYOUT_KEY.format(rank)).decode().split("|")
        return {"num_regions": int(num_regions), "dram_base": int(dram_base), "unique_id": uid}
    except (ValueError, IndexError) as exc:
        print(f"[ERROR] parse layout of decode rank {rank} failed: {exc}")
        return None


def announce_all_layouts(store, world_size: int) -> None:
    """decode rank0: 等全部 rank 布局就位(get 阻塞等)后发布 decode 卡数, Prefill
    才放行, 保证开始写时所有 decode rank 均已完成内存注册.
    """
    for m in range(1, world_size):
        store.get(META_LAYOUT_KEY.format(m))
    store.set(META_GLOBAL_KEY, str(world_size))


def init_offload_pool(npu_id: int, nbytes: int):
    """offload 分配 DRAM 池(URMA_POOL: HVA/DVA 分离), reserve/alloc 向上 GB 对齐."""
    reserve = max(GB, (nbytes + GB - 1) // GB * GB)
    cfg = offload.OffloadConfig()
    cfg.device_id = npu_id
    cfg.reserve_size = reserve
    cfg.alloc_size = reserve
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.LOCAL
    cfg.flags = offload.OFFLOAD_FLAG_URMA_POOL
    assert offload.initialize(cfg) == 0, "offload.initialize failed"
    dram_base = offload.malloc(nbytes, 0)
    assert dram_base != 0, "offload.malloc failed"
    dram_dva = offload.get_dva(dram_base)
    assert dram_dva != 0, "offload.get_dva failed"
    return dram_base, dram_dva


def wait_all_magic(num_regions: int, dram_base: int, tag: str) -> bool:
    """轮询全部 (region, size) 尾标直到就位(或超时)."""
    specs = [(r, s) for r in range(num_regions) for s in SIZES]
    pending = list(specs)
    deadline = time.time() + WAIT_TIMEOUT_S
    print(f"[Decode{tag}] waiting for {len(specs)} magic marks ...", flush=True)
    while pending:
        for spec in list(pending):
            r, size = spec
            tail = read_hva(dram_base + r * MAX_BYTES + size - 4, 4)
            if int.from_bytes(tail.tobytes(), "little") == size_magic(size):
                pending.remove(spec)
                print(f"[Decode{tag}] w{r} {fmt_size(size)} received", flush=True)
        if pending and time.time() > deadline:
            print(f"[Decode{tag}] FAILED: timeout, missing: {[f'w{r}/{fmt_size(s)}' for r, s in pending]}")
            return False
        if pending:
            time.sleep(0.2)
    return True


def verify_regions(dram_base: int, num_regions: int, decode_rank: int, tag: str) -> bool:
    """逐 region 校验直写内容(经 HVA 读), 公式含 decode/writer rank."""
    ok = True
    for r in range(num_regions):
        got = read_hva(dram_base + r * MAX_BYTES, MAX_BYTES)
        good = np.array_equal(got, build_expected(data_variant(decode_rank, r)))
        print(f"[Decode{tag}] w{r} receive via DRAM write ... {'OK' if good else 'FAILED'}", flush=True)
        ok = ok and good
    return ok


def promote_region0(npu_id: int, hbm_buf, dram_base: int, dram_dva: int, tag: str) -> bool:
    """region0 逐档 sparse_copy(DVA->HBM) 提升并统计带宽."""
    dev = torch.device("npu", npu_id)
    hbm_addr = hbm_buf.data_ptr()
    for size in PROMOTE_SIZES:
        src_ptrs = torch.tensor([dram_dva], dtype=torch.int64).npu()
        dst_ptrs = torch.tensor([hbm_addr], dtype=torch.int64).npu()
        lens = torch.tensor([size], dtype=torch.int32).npu()
        cnt = torch.tensor(1, dtype=torch.int32).npu()

        def do_promote():
            ret = offload.sparse_copy(src_ptrs, dst_ptrs, lens, cnt, dev)
            assert ret == 0, f"offload.sparse_copy failed: {ret}"

        tests = 3 if size >= (32 << 20) else 10
        elapsed = bench_npu(do_promote, num_tests=tests)
        ok = np.array_equal(hbm_buf[:size].cpu().numpy(), read_hva(dram_base, size))
        print(
            f"[Decode{tag}] promote {size / 1e6:7.1f} MB: {'OK' if ok else 'FAILED'}, "
            f"{elapsed * 1e3:.1f} ms(avg of {tests}) => {size / elapsed / 1e9:.3f} GB/s",
            flush=True,
        )
        if not ok:
            return False
    return True


def finish_decode(
    engine, npu_id: int, hbm_buf, dram_base: int, dram_dva: int, num_regions: int, decode_rank: int, tag: str
) -> int:
    """逐 region 校验 -> region0 提升 -> 清理."""
    if not verify_regions(dram_base, num_regions, decode_rank, tag):
        return 1
    if not promote_region0(npu_id, hbm_buf, dram_base, dram_dva, tag):
        return 1
    engine.destroy()
    offload.free(dram_base, 0)
    offload.uninitialize()
    print(f"[Decode{tag}] trans_offload_e2e_multi (writers={num_regions}): ALL PASS", flush=True)
    return 0


def decode_main(rank: int, npu_id: int, world_size: int, args, store) -> int:
    """单个 Decode rank: engine + DRAM 池注册 -> 发布布局 -> 等写完 -> 校验 -> 提升."""
    set_log_level(args.log_level)
    torch.npu.set_device(npu_id)
    tag = f" d{rank}" if world_size > 1 else ""
    n = args.num_writers
    uid = child_unique_id(args.src_unique_id, rank)
    # rank0 任 config store server, 其余 rank 为 client 需等 rank0 就绪故带重试
    server_role = "Decode" if rank == 0 else "Prefill"
    engine = init_engine(args, uid, "Decode", npu_id, server_role, 0 if rank == 0 else 180)
    print(f"[Decode{tag}] engine init ok, uid={uid}, store={args.store_url}, op={OP}, writers={n}", flush=True)

    dram_base, dram_dva = init_offload_pool(npu_id, n * MAX_BYTES)
    hbm_buf = torch.zeros(MAX_BYTES, dtype=torch.uint8, device="npu")  # 提升目标区
    assert engine.batch_register_memory([dram_base], [n * MAX_BYTES]) == 0, "batch_register failed"

    publish_layout(store, rank, n, dram_base, uid)
    if rank == 0:
        announce_all_layouts(store, world_size)
        print("[Decode] layouts published, start Prefill now", flush=True)

    if not wait_all_magic(n, dram_base, tag):
        return 1
    try:  # 结束屏障: 等全部 writer 计时完成, 保证拆通道时没有在途写
        store_wait_counter(store, WRITER_DONE_KEY, n)
    except TimeoutError:
        print(f"[Decode{tag}] FAILED: writer-done wait timeout, writers={n}")
        return 1
    time.sleep(1.0)  # 留出写完全落地的余量
    return finish_decode(engine, npu_id, hbm_buf, dram_base, dram_dva, n, rank, tag)


def decode_child(rank: int, npu_id: int, world_size: int, args) -> None:
    """Decode 子进程入口: 连父进程的 meta TCPStore 后进入主流程."""
    holder = {"store": meta_store_client("127.0.0.1", args.meta_port)}
    rc = decode_main(rank, npu_id, world_size, args, holder["store"])
    del holder["store"]  # 尽早断开, 避免连接残留拖住父进程 server 析构
    if rc != 0:
        raise SystemExit(rc)


def run_decode(args) -> int:
    """Decode 侧: 父进程持有 meta TCPStore, 按 --npu-id 拉起 M 进程 M 卡."""
    npu_ids = [int(x) for x in args.npu_id.split(",") if x.strip() != ""]
    assert npu_ids, "--npu-id got no valid id"
    assert len(npu_ids) <= MAX_CARDS, f"at most {MAX_CARDS} decode cards"
    assert 1 <= args.num_writers <= MAX_CARDS, f"--num-writers must be in [1, {MAX_CARDS}]"
    holder = {"store": meta_store_server(args.meta_port)}
    if len(npu_ids) == 1:
        rc = decode_main(0, npu_ids[0], 1, args, holder["store"])
        del holder["store"]
        return rc
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=decode_child, args=(r, nid, len(npu_ids), args)) for r, nid in enumerate(npu_ids)]
    for p in procs:
        p.start()
    ok = join_procs(procs)
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    del holder["store"]
    if not ok:
        return 1
    if bad:
        print(f"[ERROR] decode process(es) failed: exitcodes {bad}")
        return 1
    return 0


def prefill_child(rank: int, npu_id: int, world_size: int, args, layouts, result_q) -> None:
    """Prefill 子进程: 一卡一进程, 把自己的 region 逐档并发直写全部 M 张 Decode 卡."""
    set_log_level(args.log_level)
    torch.npu.set_device(npu_id)
    tag = f" w{rank}" if world_size > 1 else ""
    engine = init_engine(args, child_unique_id(args.src_unique_id, rank), "Prefill", npu_id, "Decode", 0)
    # 每个 decode 卡一份源 chunk(公式含 decode rank, 写错卡校验必失败)
    chunks = [torch.from_numpy(build_expected(data_variant(m, rank))).npu() for m in range(len(layouts))]
    assert engine.batch_register_memory([c.data_ptr() for c in chunks], [MAX_BYTES] * len(chunks)) == 0

    if world_size > 1:  # 起跑屏障: 对齐 N 个 writer 的写起点, 聚合带宽才可比
        holder = {"store": meta_store_client(args.dst_unique_id.rsplit(":", 1)[0], args.meta_port)}
        store_barrier(holder["store"], BARRIER_KEY, world_size)
        del holder["store"]

    for size in SIZES:  # 逐档(大->小)逐 decode 卡, N 个 writer 同窗并发
        for m, lay in enumerate(layouts):
            dst = lay["dram_base"] + rank * MAX_BYTES
            elapsed = timed_write(engine, lay["unique_id"], chunks[m].data_ptr(), dst, size, tag)
            if result_q is not None:
                result_q.put((m, size, elapsed))

    # 结束屏障: 最后一发 sync write 已返回, 计时全部完成 -> 计数 +1, Decode 等满
    # 才拆通道; client 短连短用(60s 连接超时), 用完立即释放
    host = args.dst_unique_id.rsplit(":", 1)[0]
    holder = {"store": meta_store_client(host, args.meta_port, timedelta(seconds=60))}
    holder["store"].add(WRITER_DONE_KEY, 1)
    del holder["store"]
    engine.destroy()


def aggregate_results(rows, num_writers: int) -> None:
    """按 (decode 卡, 档位) 汇总: 聚合带宽 = writers * size / 最慢卡耗时."""
    by_case = {}
    for decode_rank, size, elapsed in rows:
        by_case.setdefault((decode_rank, size), []).append(elapsed)
    if not by_case:
        return
    print("[Prefill] ==== aggregate: writers hitting each decode card concurrently ====", flush=True)
    order = {s: i for i, s in enumerate(SIZES)}
    for (decode_rank, size), elist in sorted(by_case.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        worst = max(elist)
        print(
            f"[Prefill] write {fmt_size(size):>7} -> d{decode_rank} x{num_writers}: "
            f"slowest {worst * 1e3:7.1f} ms => aggregate "
            f"{num_writers * size / worst / 1e9:7.3f} GB/s",
            flush=True,
        )


def run_prefill(args) -> int:
    """Prefill 侧: 取全部 decode 布局 -> 每卡一个子进程并发写 -> 汇总聚合带宽."""
    npu_ids = [int(x) for x in args.npu_id.split(",") if x.strip() != ""]
    assert npu_ids, "--npu-id got no valid id"
    assert len(npu_ids) <= MAX_CARDS, f"at most {MAX_CARDS} writer cards"
    host = args.dst_unique_id.rsplit(":", 1)[0]
    holder = {"store": meta_store_client(host, args.meta_port)}
    try:
        num_decode = int(holder["store"].get(META_GLOBAL_KEY).decode())
        layouts = [fetch_layout(holder["store"], m) for m in range(num_decode)]
    except Exception as exc:  # TCPStore 连接/等待失败: Decode 未先启动等
        print(f"[ERROR] fetch layouts from {host}:{args.meta_port} failed: {exc}; ensure Decode is started first")
        return 1
    del holder["store"]  # 布局已取回, 尽早断开
    if any(lay is None for lay in layouts):
        return 1
    if len(npu_ids) != layouts[0]["num_regions"]:
        print(f"[ERROR] writer count {len(npu_ids)} != decode --num-writers {layouts[0]['num_regions']}")
        return 1
    print(f"[Prefill] layout ok: decode cards={num_decode}, writers={len(npu_ids)}, npus={npu_ids}", flush=True)

    if len(npu_ids) == 1:
        prefill_child(0, npu_ids[0], 1, args, layouts, None)
        return 0
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=prefill_child, args=(r, nid, len(npu_ids), args, layouts, result_q))
        for r, nid in enumerate(npu_ids)
    ]
    for p in procs:
        p.start()
    if not join_procs(procs):
        return 1
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        print(f"[ERROR] writer process(es) failed: exitcodes {bad}")
        return 1
    rows = []
    while True:
        try:
            rows.append(result_q.get_nowait())
        except Empty:
            break
    aggregate_results(rows, len(npu_ids))
    return 0


EPILOG = """\
usage (start Decode first; Prefill card count must equal Decode --num-writers):
  Decode node (2 cards, expects 4 writers):
    python trans_offload_e2e_multi.py --role Decode \\
        --store-url tcp://<DECODE_IP>:9900 --src-unique-id <DECODE_IP>:9901 \\
        --num-writers 4 --npu-id 0,1
  Prefill node (4 writer cards):
    python trans_offload_e2e_multi.py --role Prefill \\
        --store-url tcp://<DECODE_IP>:9900 --src-unique-id <PREFILL_IP>:9902 \\
        --dst-unique-id <DECODE_IP>:9901 --npu-id 0,1,2,3

notes:
  - addresses auto-exchange via a TCPStore (--meta-port, default 9950); Decode
    rank0 also hosts the config store; child unique_ids are <ip>:<port+rank>,
    so when both roles run on ONE node keep the two --src-unique-id port bases
    at least max(M, N) apart (e.g. decode :9901, prefill :9921)
  - Decode waits for all writers to finish (writer_done counter) before
    verify/promote/destroy, so channels are never torn down mid-write
  - DEVICE_URMA requires ASCEND_950 (A5) and a URMA-enabled build, tested on
    A5 only; needs MEMFABRIC_HYBRID_EXTEND_LIB_PATH at libmf_hybm_accoffload.so
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="multi-to-multi PD KV offload e2e: N prefill cards concurrently write their "
        "own region into the DRAM pool of each of M decode cards via DEVICE_URMA "
        "(full mesh); layouts auto-exchanged via TCPStore",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--role", required=True, choices=["Decode", "Prefill"])
    parser.add_argument("--store-url", required=True, help="config store url, the Decode rank0 acts as store server")
    parser.add_argument(
        "--src-unique-id",
        required=True,
        help="unique id of this node, <ip>:<port>; child processes use <ip>:<port+rank> so each process stays unique",
    )
    parser.add_argument("--dst-unique-id", help="[Prefill only] unique id of the Decode node")
    parser.add_argument("--npu-id", default="0", help="comma-separated NPU ids, one process per id (e.g. 0,1,2,3)")
    parser.add_argument(
        "--num-writers", type=int, default=1, help="[Decode only] number of concurrent prefill writer cards to expect"
    )
    parser.add_argument("--meta-port", type=int, default=9950, help="TCPStore port for automatic layout exchange")
    parser.add_argument("--log-level", type=int, default=2, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    set_log_level(args.log_level)
    if args.role == "Decode":
        rc = run_decode(args)
    else:
        assert args.dst_unique_id, "dst-unique-id required for Prefill role"
        rc = run_prefill(args)
    print(f"[{args.role}] trans_offload_e2e_multi all done (rc={rc})", flush=True)
    # 资源均已在各路径显式释放; os._exit 绕过解释器关闭阶段的 C++ 析构, 避免退出挂死
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
