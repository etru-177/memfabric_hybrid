#!/usr/bin/env python
# coding=utf-8
"""varlen_copy (AIV DRAM->HBM gather) micro-benchmark.

NOTE: tested on ASCEND_950 (A5) only.

This script isolates the varlen_copy operator (batched variable-length
pointer copy) and sweeps the two most likely bottlenecks:
  entry parallelism  : one big entry vs. N equal entries (AIV core fan-out)
  swap-in shape      : many small scattered segments, mimicking the real
                       top-k gather (miss tokens x layers) of the PD DRAM
                       offload design.

Usage (on the A5 node):
  python varlen_copy_perf.py --mode single
  python varlen_copy_perf.py --mode split                  # split x1/x4/x16/x64 only
  python varlen_copy_perf.py --mode swapin                 # 100000 x 656B fixed (single-token gather)
  python varlen_copy_perf.py --mode swapin --merge-run 8   # host-side coalesced shape (8 tokens/seg)
  python varlen_copy_perf.py --mode swapin --segments 1024 --seg-bytes-min 2048 --seg-bytes-max 8192
  python varlen_copy_perf.py --mode accuracy               # per-shape data correctness
  python varlen_copy_perf.py --npu-id 0,2,3 --mode swapin  # multiple NPUs, sequential
  python varlen_copy_perf.py --mode all
"""

import argparse
import ctypes
import multiprocessing as mp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # required: registers the npu backend

from memfabric_hybrid import offload, set_log_level

POOL_RESERVE = 1 << 30
SIZES = [128 << 20, 32 << 20, 4 << 20, 128 << 10]
SPLIT_COUNTS = [1, 4, 16, 64]
RULE_HEAVY = "=" * 78
RULE_LIGHT = "-" * 78
RESULTS = []  # (tag, entries, totalBytes, elapsedS, bwGBs) rows of THIS process
NPU_TAG = ""  # "[npu N] " prefix for correctness lines; set by run_on_npu
BARRIER_KEY = "varlen_copy_perf/barrier"  # + "/{round}": one key per timed config


def banner(title: str) -> None:
    print(f"\n{RULE_LIGHT}\n {title}\n{RULE_LIGHT}")


def print_results(npu_id, lock=None) -> None:
    """One dump per process at the end of its run. `lock` is created in the
    parent BEFORE spawn and passed through Process args (the only way it is
    shared across spawn children); None in single-process runs.
    """
    if not RESULTS:
        return
    if lock is not None:
        lock.acquire()
    try:
        print(f"\n[npu {npu_id}] results:")
        print(f"  {'config':<40}  {'entries':>7}  {'total':>9}  {'time':>10}  {'GB/s':>8}")
        for tag, n, total, elapsed, bw in RESULTS:
            print(f"  {tag:<40}  {n:>7}  {total / 1e6:>8.1f}MB  {elapsed * 1e3:>8.2f}ms  {bw:>8.2f}")
    finally:
        if lock is not None:
            lock.release()


def bench_npu(fn, num_warmups=2, num_tests=10, flush_l2=True, sync=None) -> float:
    """NPU event timing (same method as trans_offload_e2e_multi.py). `sync` (if
    given) is a one-shot barrier invoked AFTER warmup drained the device:
    every NPU process enters its timed rounds at the same instant, so
    concurrent DRAM reads truly overlap per config.
    """
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device="npu") if flush_l2 else None
    for _ in range(num_warmups):
        fn()
    torch.npu.synchronize()
    if sync is not None:
        sync()
    times = []
    for _ in range(num_tests):
        if cache is not None:
            cache.zero_()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return sum(times) / len(times)


def init_offload_pool(npu_id: int):
    """offload URMA_POOL DRAM pool, returns (hva, dva)."""
    cfg = offload.OffloadConfig()
    cfg.device_id = npu_id
    cfg.reserve_size = POOL_RESERVE
    cfg.alloc_size = POOL_RESERVE
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.LOCAL
    cfg.flags = offload.OFFLOAD_FLAG_URMA_POOL
    assert offload.initialize(cfg) == 0, "offload.initialize failed"
    hva = offload.malloc(POOL_RESERVE, 0)
    assert hva != 0, "offload.malloc failed"
    dva = offload.get_dva(hva)
    assert dva != 0, "offload.get_dva failed"
    return hva, dva


def prepare_copy_args(srcs, dsts, lens, dev):
    """Build the NPU descriptor tensors ONCE; reused by every timed round."""
    src_ptrs = torch.tensor(srcs, dtype=torch.int64).to(dev)
    dst_ptrs = torch.tensor(dsts, dtype=torch.int64).to(dev)
    len_ptrs = torch.tensor(lens, dtype=torch.int32).to(dev)
    cnt = torch.tensor(len(lens), dtype=torch.int32).to(dev)
    return src_ptrs, dst_ptrs, len_ptrs, cnt


def submit_copy(src_ptrs, dst_ptrs, len_ptrs, cnt, dev) -> None:
    """Submit one batched varlen copy: entries are (src[i] -> dst[i], lens[i])."""
    ret = offload.varlen_copy(src_ptrs, dst_ptrs, len_ptrs, cnt, dev)
    assert ret == 0, f"offload.varlen_copy failed: {ret}"


def run_varlen_copy(srcs, dsts, lens, dev) -> None:
    """One-shot prepare + submit (verify/accuracy paths, outside timing)."""
    submit_copy(*prepare_copy_args(srcs, dsts, lens, dev), dev)


def bench_and_report(tag, srcs, dsts, lens, dev, rounds, sync=None) -> float:
    """Time one entry set. Descriptors are built before the event window, so
    tensor building and the H2D descriptor transfers (same stream as the
    kernel) never inflate the measured time; the events tightly bracket only
    the varlen_copy call. Rows go to RESULTS and the process prints them all
    at once when it finishes.
    """
    total = int(sum(lens))
    ptrs = prepare_copy_args(srcs, dsts, lens, dev)

    def fn():
        submit_copy(*ptrs, dev)

    elapsed = bench_npu(fn, num_tests=rounds, sync=sync)
    bw = total / elapsed / 1e9
    RESULTS.append((tag, len(lens), total, elapsed, bw))
    return bw


def bench_single(dva, hbm_addr, dev, rounds, sync=None):
    for size in SIZES:
        bench_and_report(
            f"single {size >> 20}MiB" if size >= (1 << 20) else f"single {size >> 10}KiB",
            [dva],
            [hbm_addr],
            [size],
            dev,
            rounds,
            sync,
        )


def bench_split(dva, hbm_addr, dev, rounds, sync=None):
    total = 128 << 20
    for n in SPLIT_COUNTS:
        seg = total // n
        srcs = [dva + i * seg for i in range(n)]
        dsts = [hbm_addr + i * seg for i in range(n)]
        bench_and_report(f"split x{n}", srcs, dsts, [seg] * n, dev, rounds, sync)


def gen_merged_runs(dva, hbm_addr, args, rng):
    """Merged-run shape: consecutive selected tokens coalesce into segments
    (the host-side _co_segments merge of the real swap-in path), so the entry
    count drops by ~run_len while total bytes stay the same.
    """
    lens, srcs, dsts = [], [], []
    left = args.tokens
    while left > 0:
        k = min(args.merge_run, left)
        seg = k * args.item_bytes
        src = dva + rng.randrange(0, POOL_RESERVE - seg - 32)
        dst = hbm_addr + rng.randrange(0, (128 << 20) - seg - 32)
        srcs.append(src - src % 32)  # 32B-align segment bases, len stays exact
        dsts.append(dst - dst % 32)
        lens.append(seg)
        left -= k
    return srcs, dsts, lens


def bench_swapin(dva, hbm_addr, dev, args, rounds, sync=None):
    rounds = args.swapin_rounds if args.swapin_rounds > 0 else rounds
    rng = random.Random(args.seed)
    if args.merge_run > 0:
        srcs, dsts, lens = gen_merged_runs(dva, hbm_addr, args, rng)
        tag = f"swapin merge-run={args.merge_run} x{args.item_bytes}B"
        bench_and_report(tag, srcs, dsts, lens, dev, rounds, sync)
        return
    n, lo, hi = args.segments, args.seg_bytes_min, args.seg_bytes_max
    fixed = lo == hi  # single-token gather shape: one fixed-size entry per token
    lens, srcs, dsts = [], [], []
    for i in range(n):
        seg = lo if fixed else rng.randrange(lo, hi + 1)
        src = dva + rng.randrange(0, POOL_RESERVE - seg - 32)
        dst = hbm_addr + rng.randrange(0, (128 << 20) - seg - 32)
        srcs.append(src - src % 32)  # 32B-align segment bases, keep exact len
        dsts.append(dst - dst % 32)
        lens.append(seg)
    if fixed:
        tag = f"swapin {n}segs x{lo}B fixed"
    else:
        tag = f"swapin {n}segs x{lo // 1024}-{hi // 1024}KiB"
    bench_and_report(tag, srcs, dsts, lens, dev, rounds, sync)


def verify(hva, hbm_buf, dva, hbm_addr, dev) -> None:
    """Sanity check: one 4MiB copy, compare DRAM vs promoted HBM content."""
    run_varlen_copy([dva], [hbm_addr], [4 << 20], dev)
    torch.npu.synchronize()
    src = np.frombuffer(ctypes.string_at(hva, 4 << 20), dtype=np.uint8)
    dst = hbm_buf[: 4 << 20].cpu().numpy()
    print(f"{NPU_TAG}[verify] 4MiB promote data match: {'OK' if np.array_equal(src, dst) else 'FAILED'}")


def _fill_src(hva, dva, srcs, lens) -> None:
    """Write a position-derived pattern into each DRAM source segment. srcs
    are device (DVA) addresses; in URMA_POOL mode the DVA mapping is not
    host-writable, so fill through the pool HVA at the same in-pool offset.
    """
    for src, ln in zip(srcs, lens):
        pos = np.arange(ln, dtype=np.uint64)
        arr = ((pos * 7 + src) % 251).astype(np.uint8)
        host_ptr = hva + (src - dva)
        ctypes.memmove(host_ptr, arr.tobytes(), ln)


def _check_dst(hbm_np, hbm_addr, srcs, dsts, lens) -> int:
    """Compare each promoted HBM segment against its source pattern."""
    bad = 0
    for src, dst, ln in zip(srcs, dsts, lens):
        off = dst - hbm_addr
        pos = np.arange(ln, dtype=np.uint64)
        expect = ((pos * 7 + src) % 251).astype(np.uint8)
        if not np.array_equal(hbm_np[off : off + ln], expect):
            bad += 1
    return bad


def bench_accuracy(hva, dva, hbm_addr, hbm_buf, dev, args):
    """Full-shape correctness: patterned sources -> varlen_copy -> compare dst."""
    banner("[mode accuracy] per-shape data verification")
    print(f"  {'case':<40}  {'entries':>7}  {'total':>9}  {'result':>8}")
    rng = random.Random(args.seed)
    span_src = POOL_RESERVE - (1 << 20)

    # Both src and dst are laid out sequentially with random gaps: ANY overlap
    # (src: later fill overwrites an earlier segment's pattern; dst: later copy
    # overwrites an earlier copy) fails the check for reasons unrelated to
    # kernel correctness. Gaps keep the scatter shape sparse.
    def gen_case(cnt, lo, hi):
        lens = [rng.randrange(lo, hi + 1) for _ in range(cnt)]
        total = sum(lens)
        dsts = []
        cursor = 0
        for ln in lens:
            cursor += rng.randrange(0, 4096)
            cursor = (cursor + 31) & ~31  # MTE needs 32B-aligned segment BASES
            dsts.append(hbm_addr + cursor)
            cursor += ln
        assert cursor < (128 << 20), "accuracy dst overflow"
        srcs = []
        src_gap = max(64, (span_src - total) // (cnt + 1))
        sc = rng.randrange(0, max(1, span_src - total - cnt * src_gap))
        for ln in lens:
            sc += rng.randrange(0, src_gap)
            sc = (sc + 31) & ~31  # 32B-align segment base, keep exact len
            srcs.append(dva + sc)
            sc += ln
        return srcs, dsts, lens

    run = max(args.merge_run, 1)  # merge-run=0 -> single-token shape
    n0 = max(args.tokens // 10, 1)
    cases = [
        ("large >UB (256KB x3)", 3, 256 * 1024, 256 * 1024),  # CopyLargeEntry path
        ("fixed 656B x1000 (non-32B len)", 1000, 656, 656),  # pad semantics
        ("batch-boundary 351/352/353", 1056, 640, 640),  # BATCH_MAX edges
        ("mixed 32B..40KB x2000", 2000, 32, 40 * 1024),  # cross-batch mixes
        (f"merge-run={run} x{n0}tok", max(n0 // run, 1), run * args.item_bytes, run * args.item_bytes),
    ]
    for name, cnt, lo, hi in cases:
        srcs, dsts, lens = gen_case(cnt, lo, hi)
        _fill_src(hva, dva, srcs, lens)
        run_varlen_copy(srcs, dsts, lens, dev)
        torch.npu.synchronize()
        hbm_np = hbm_buf.cpu().numpy()
        bad = _check_dst(hbm_np, hbm_addr, srcs, dsts, lens)
        total = int(sum(lens))
        verdict = "OK" if bad == 0 else f"FAILED({bad})"
        print(f"{NPU_TAG}{name:<40}  {cnt:>7}  {total / 1e6:>8.1f}MB  {verdict:>8}")


def make_sync(store, world_size: int):
    """Build the per-config barrier callable handed to bench_npu. Each call
    uses a fresh round-scoped key (fixed key would trip on stale counters
    from the previous config: add() keeps accumulating). Returns None for
    single-process runs so bench_npu skips it entirely.
    """
    if store is None:
        return None
    round_no = [0]

    def sync():
        key = f"{BARRIER_KEY}/{round_no[0]}"
        round_no[0] += 1
        store.add(key, 1)
        deadline = time.time() + 300.0
        while int(store.get(key).decode()) < world_size:
            if time.time() > deadline:
                raise TimeoutError(f"barrier {key} timeout after 300s")
        time.sleep(0)  # yield once after the barrier passes

    return sync


def run_on_npu(npu_id, args, rank=0, world_size=1, lock=None, sync_port=None) -> None:
    """Process entry: full mode suite on one NPU (pool init -> verify ->
    modes -> uninit). Progress prints are suppressed; all timed rows are
    dumped once at the end under `lock` (shared across spawn children).
    `sync_port` (multi-NPU runs) is the parent's barrier TCPStore port: the
    child builds its OWN client connection (TCPStore objects cannot be
    pickled across spawn, so the store itself is never passed in).
    """
    global NPU_TAG
    dev = torch.device("npu", npu_id)
    torch.npu.set_device(npu_id)
    NPU_TAG = f"[npu {npu_id}] "

    hva, dva = init_offload_pool(npu_id)
    hbm_buf = torch.zeros(2 * (128 << 20), dtype=torch.uint8, device="npu")
    hbm_addr = hbm_buf.data_ptr()
    verify(hva, hbm_buf, dva, hbm_addr, dev)

    # per-config barrier: bench_npu invokes it after its warmup drained the
    # device, so every process enters each config's timed rounds together
    # (busy-poll TCPStore counter, us-scale release skew; no HCCL at all)
    store = None
    if sync_port is not None:
        store = dist.TCPStore(host_name="127.0.0.1", port=sync_port, is_master=False, timeout=timedelta(seconds=600))
    sync = make_sync(store, world_size)

    modes = ("single", "split", "swapin") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode == "single":
            bench_single(dva, hbm_addr, dev, args.rounds, sync)
        elif mode == "split":
            bench_split(dva, hbm_addr, dev, args.rounds, sync)
        elif mode == "accuracy":
            bench_accuracy(hva, dva, hbm_addr, hbm_buf, dev, args)
        else:
            bench_swapin(dva, hbm_addr, dev, args, args.rounds, sync)

    if store is not None:
        del store  # drop the barrier client connection before cleanup
    offload.free(hva, 0)
    offload.uninitialize()

    print_results(npu_id, lock)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--npu-id", default="0", help="NPU id(s), comma-separated (e.g. 0,2,3); one concurrent process per NPU"
    )
    parser.add_argument("--mode", default="all", choices=["single", "split", "swapin", "accuracy", "all"])
    parser.add_argument("--rounds", type=int, default=10, help="timed rounds per config")
    parser.add_argument(
        "--swapin-rounds",
        type=int,
        default=50,
        help="timed rounds for swapin mode (0 = fall back to --rounds); "
        "higher default because swapin copies are short and jitter-prone",
    )
    parser.add_argument(
        "--segments", type=int, default=100000, help="[swapin] segment count (default: single-token gather shape)"
    )
    parser.add_argument(
        "--seg-bytes-min",
        type=int,
        default=656,
        help="[swapin] min segment bytes; min==max means fixed size (default 656B, exact)",
    )
    parser.add_argument(
        "--seg-bytes-max",
        type=int,
        default=656,
        help="[swapin] max segment bytes; min==max means fixed size (default 656B, exact)",
    )
    parser.add_argument(
        "--merge-run",
        type=int,
        default=0,
        help="[swapin] merge N consecutive tokens per segment (host-side coalesce "
        "simulation); 0 disables, else entry count ~= tokens/N",
    )
    parser.add_argument("--tokens", type=int, default=100000, help="[swapin][--merge-run] total token count")
    parser.add_argument(
        "--item-bytes",
        type=int,
        default=656,
        help="[swapin][--merge-run] KV bytes per token per layer (real GLM-5.2 shape)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--sync-port", type=int, default=29531, help="TCP port of the one-shot TCPStore start barrier (multi-NPU runs)"
    )
    args = parser.parse_args()

    npu_ids = [int(x) for x in args.npu_id.split(",") if x.strip() != ""]
    assert npu_ids, "--npu-id got no valid id"

    set_log_level(args.log_level)

    print(RULE_HEAVY)
    print(" varlen_copy (AIV DRAM -> HBM gather) micro-benchmark")
    print(
        f" npu ids: {npu_ids} (one process each), mode: {args.mode}, "
        f"rounds: {args.rounds} (swapin: {args.swapin_rounds if args.swapin_rounds > 0 else args.rounds})"
    )
    print(RULE_HEAVY)

    if len(npu_ids) == 1:
        run_on_npu(npu_ids[0], args)
    else:
        mp.set_start_method("spawn", force=True)
        lock = mp.Lock()  # created BEFORE spawn so children share it
        # parent hosts the barrier TCPStore BEFORE spawning children, so the
        # server is listening when they connect; children get only the PORT
        # (TCPStore objects cannot be pickled across spawn) and build their
        # own client connections on 127.0.0.1
        store = dist.TCPStore(
            host_name="127.0.0.1", port=args.sync_port, is_master=True, timeout=timedelta(seconds=600)
        )
        procs = [
            mp.Process(target=run_on_npu, args=(nid, args, rank, len(npu_ids), lock, args.sync_port))
            for rank, nid in enumerate(npu_ids)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        del store  # release barrier server after children are done
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        if bad:
            print(f"[ERROR] child process(es) failed: exitcodes {bad}")
            return 1

    print("\n[done] varlen_copy perf test finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
