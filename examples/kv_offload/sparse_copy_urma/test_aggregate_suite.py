#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

import contextlib
import ctypes
import importlib
import io
import multiprocessing
import os
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

demo = importlib.import_module("03_aicpu_host_aggregate_urma")


def accept_transferred_listener(listener):
    listener.settimeout(5)
    with listener, listener.accept()[0] as conn:
        conn.sendall(b"ready")


class AggregateSuiteTest(unittest.TestCase):
    def test_direct_addresses(self):
        layout = demo.make_layout(3200, 656)
        src, dst, lengths = demo.direct_copy_lists(0x10000000, 0x20000000, layout, 3200, 656)
        self.assertEqual(src[0], 0x10000000 + layout[2])
        self.assertEqual(dst[0], 0x20000000 + layout[5])
        self.assertEqual(src[-1] - src[0], 3199 * 1312)
        self.assertEqual(dst[-1] - dst[0], 3199 * 1312)
        self.assertEqual(lengths, [656] * 3200)
        self.assertLessEqual(dst[-1] + 656, 0x20000000 + layout[-1])

    def test_direct_summary_without_host_metrics(self):
        with patch("sys.argv", ["demo", "--mode", "direct", "--segments", "100", "--segment-bytes", "656"]):
            args = demo.parse_args()
        with tempfile.TemporaryDirectory() as directory, patch.object(demo.tempfile, "mkdtemp", return_value=directory):
            args.stats_file = os.path.join(directory, "stats.txt")
            with patch.object(demo, "run_case", return_value={"launch sync": [100000]}):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    demo.run_suite(args)
        self.assertIn("direct copy summary", out.getvalue())
        self.assertIn("100.000", out.getvalue())
        self.assertIn("verify=OFF", out.getvalue())

    def test_live_listener_survives_spawn(self):
        listener = socket.create_server(("0.0.0.0", 0))
        port = listener.getsockname()[1]
        child = multiprocessing.get_context("spawn").Process(target=accept_transferred_listener, args=(listener,))
        try:
            child.start()
            listener.close()
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                self.assertEqual(conn.recv(5), b"ready")
            child.join(timeout=5)
            self.assertEqual(child.exitcode, 0)
        finally:
            listener.close()
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)

    def test_error_log_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                demo.print_case_errors(directory)
            self.assertIn("Cannot read log", output.getvalue())

    def test_default_matrix(self):
        with patch("sys.argv", ["demo"]):
            args = demo.parse_args()
        self.assertEqual(args.rounds, 1000)
        self.assertEqual(args.warmup_rounds, 10)
        self.assertEqual(args.segments, sorted({b * m for b in (100, 200, 300, 400)
                                              for m in (1, 2, 4, 8, 16, 32, 64)}))
        self.assertEqual(args.segments[-1], 25600)
        self.assertTrue(args.force_host_nic_plugin)
        self.assertEqual(args.source_pool_segments, 0)

    def test_dense_source_pool_layout_and_permutation(self):
        layout = demo.make_layout(3200, 656, 262144)
        address_bytes = 3200 * ctypes.sizeof(ctypes.c_uint64)
        self.assertGreaterEqual(layout[2], ctypes.sizeof(demo.Message) + address_bytes)
        wire_bytes = address_bytes + ctypes.sizeof(demo.Request) + ctypes.sizeof(ctypes.c_uint64)
        self.assertGreaterEqual(layout[4], demo.DEVICE_ADDRESS_OFFSET + wire_bytes)
        self.assertGreaterEqual(layout[3], layout[2] + 262144 * 656)
        first = list(demo.make_source_indices(32, 2026))
        second = list(demo.make_source_indices(32, 2026))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(32)))

    def test_source_pool_must_cover_largest_case(self):
        with patch("sys.argv", ["demo", "--segments", "3200", "--segment-bytes", "656",
                                "--source-pool-segments", "100"]):
            with self.assertRaises(SystemExit):
                demo.parse_args()

    def test_warmup_rounds_must_be_non_negative(self):
        with patch("sys.argv", ["demo", "--warmup-rounds", "-1"]):
            with self.assertRaises(SystemExit):
                demo.parse_args()

    def test_measured_round_range(self):
        args = Mock(warmup_rounds=10, rounds=100)
        self.assertEqual(demo.total_iterations(args), 110)
        self.assertFalse(demo.is_measured_round(args, 9))
        self.assertTrue(demo.is_measured_round(args, 10))

    def test_host_plugin_can_be_disabled(self):
        with patch("sys.argv", ["demo", "--no-host-nic-plugin"]):
            args = demo.parse_args()
        self.assertFalse(args.force_host_nic_plugin)

    def test_force_plugin_environment_is_host_only(self):
        with patch.object(demo, "load_env"), patch.object(demo, "configure_host_affinity"):
            with patch.dict(demo.os.environ, {
                    "MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID": "0", "ASCEND_RT_VISIBLE_DEVICES": "0"}, clear=True):
                torch = Mock()
                with patch.dict("sys.modules", {"torch": torch}):
                    demo.configure("host", "unused", force_host_nic_plugin=True)
                self.assertEqual(demo.os.environ["HCOMM_NIC_PLUGIN_FORCE_LOAD"], "1")

    def test_disabled_plugin_overrides_inherited_environment(self):
        with patch.object(demo, "load_env"), patch.object(demo, "configure_host_affinity"):
            environment = {"MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID": "0", "ASCEND_RT_VISIBLE_DEVICES": "0",
                           "HCOMM_NIC_PLUGIN_FORCE_LOAD": "1"}
            with patch.dict(demo.os.environ, environment, clear=True):
                with patch.dict("sys.modules", {"torch": Mock()}):
                    demo.configure("host", "unused", force_host_nic_plugin=False)
                self.assertEqual(demo.os.environ["HCOMM_NIC_PLUGIN_FORCE_LOAD"], "0")

    def test_device_affinity_is_configured_for_device_role(self):
        with patch.object(demo, "load_env"), patch.object(demo, "configure_device_affinity") as affinity:
            environment = {"MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID": "0", "ASCEND_RT_VISIBLE_DEVICES": "0"}
            with patch.dict(demo.os.environ, environment, clear=True):
                with patch.dict("sys.modules", {"torch": Mock()}):
                    demo.configure("device", "unused", device_cpu_list="64-71")
        affinity.assert_called_once_with("64-71")

    def test_host_control_and_gather_cpus_are_disjoint(self):
        with patch.object(demo.os, "sched_setaffinity") as set_affinity:
            with patch.dict(demo.os.environ, {}, clear=True):
                demo.configure_host_affinity("0-7", "0-3")
        set_affinity.assert_called_once_with(0, {4, 5, 6, 7})
        self.assertEqual(demo.os.environ["MF_GATHER_AFFINITY_CPUS"], "0,1,2,3")

    def test_gather_cpus_must_leave_host_control_cpu(self):
        with self.assertRaisesRegex(demo.CpuAffinityError, "proper subset"):
            demo.configure_host_affinity("0-3", "0-3")

    def test_worker_failure(self):
        process = Mock(exitcode=2)
        process.name = "device"
        process.is_alive.return_value = True
        with self.assertRaisesRegex(RuntimeError, "device exited with code 2"):
            demo.wait_workers([process], 1)

    def test_worker_timeout(self):
        process = Mock(exitcode=None)
        process.is_alive.return_value = True
        with patch.object(demo.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                demo.wait_workers([process], 1)

    def test_summary_and_deduplication(self):
        with patch("sys.argv", ["demo", "--segments", "100", "100", "200", "--segment-bytes", "656"]):
            args = demo.parse_args()
        values = {"gather": [20000] * args.rounds, "URMA write": [30000] * args.rounds,
                  "request publish": [10000] * args.rounds, "scatter total": [40000] * args.rounds,
                  "AICPU e2e": [90000] * args.rounds,
                  "launch overhead": [50000] * args.rounds}
        with tempfile.TemporaryDirectory() as directory, patch.object(demo.tempfile, "mkdtemp", return_value=directory):
            args.stats_file = os.path.join(directory, "stats.txt")
            with patch.object(demo, "run_case", return_value=values) as run:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    demo.run_suite(args)
        self.assertEqual(run.call_count, 2)
        self.assertIn("100.000", out.getvalue())
        self.assertIn("150.000", out.getvalue())

    def test_suite_prints_device_overhead_in_main_table(self):
        with patch("sys.argv", ["demo", "--segments", "100", "--segment-bytes", "656"]):
            args = demo.parse_args()
        values = {"gather": [20000] * args.rounds, "URMA write": [30000] * args.rounds,
                  "request publish": [10000] * args.rounds, "scatter total": [40000] * args.rounds,
                  "AICPU e2e": [90000] * args.rounds,
                  "launch overhead": [100000] * args.rounds}
        with tempfile.TemporaryDirectory() as directory, patch.object(demo.tempfile, "mkdtemp",
                                                                       return_value=directory):
            args.stats_file = os.path.join(directory, "stats.txt")
            with patch.object(demo, "run_case", return_value=values), contextlib.redirect_stdout(io.StringIO()) as out:
                demo.run_suite(args)
        self.assertNotIn("Device overhead breakdown", out.getvalue())
        self.assertIn("request", out.getvalue())
        self.assertIn("AICPU E2E(us)", out.getvalue())
        self.assertLess(out.getvalue().index("E2E(us)"), out.getvalue().index("AICPU E2E(us)"))
        self.assertIn("launch ovh", out.getvalue())
        self.assertNotIn("P50(us)", out.getvalue())
        self.assertNotIn("Metric descriptions", out.getvalue())
        with open(args.stats_file, encoding="utf-8") as stats:
            details = stats.read()
        self.assertIn("P50(us)", details)
        self.assertIn("逐轮 request", details)

    def test_poison_and_readback(self):
        args = Mock(segments=2, segment_bytes=4)
        source = (ctypes.c_uint8 * 12)()
        readback = (ctypes.c_uint8 * 12)()
        demo.fill_destination_poison(ctypes.addressof(source), 8, 2, 4)
        bm = Mock()
        handle = Mock()

        def copy(src, dst, size, *unused):
            ctypes.memmove(dst, src, size)
            return 0

        handle.copy_data.side_effect = copy
        with self.assertRaisesRegex(RuntimeError, "scatter mismatch"):
            demo.verify_scatter(handle, bm, ctypes.addressof(source), 0, args, 0, readback)
        demo.fill_source_pattern(ctypes.addressof(source), 8, 2, 4)
        demo.verify_scatter(handle, bm, ctypes.addressof(source), 0, args, 0, readback)

    def test_device_timing_breakdown(self):
        names = ("request publish", "wait host", "scatter copy", "publish barrier", "scatter total",
                 "AICPU e2e", "launch overhead")
        stages = {name: [] for name in names}
        timing = Mock(request_ns=10, wait_host_ns=80, scatter_copy_ns=250, scatter_publish_ns=40,
                      scatter_ns=290, total_ns=400)
        demo.record_device_timing(stages, timing, 500)
        self.assertEqual(stages["request publish"], [10])
        self.assertEqual(stages["launch overhead"], [100])


if __name__ == "__main__":
    unittest.main()
