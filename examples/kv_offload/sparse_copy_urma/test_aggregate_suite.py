#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.

import contextlib
import ctypes
import importlib
import io
import multiprocessing
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
            with patch.object(demo, "run_case", return_value={"launch sync": 100000}):
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
        self.assertEqual(args.segments, sorted({b * m for b in (100, 200, 300, 400)
                                              for m in (1, 2, 4, 8, 16, 32, 64)}))
        self.assertEqual(args.segments[-1], 25600)

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
        values = {"launch sync": 100000, "host total": 50000, "gather": 20000, "URMA write": 30000}
        with tempfile.TemporaryDirectory() as directory, patch.object(demo.tempfile, "mkdtemp", return_value=directory):
            with patch.object(demo, "run_case", return_value=values) as run, contextlib.redirect_stdout(io.StringIO()) as out:
                demo.run_suite(args)
        self.assertEqual(run.call_count, 2)
        self.assertIn("100.000", out.getvalue())
        self.assertIn("50.000", out.getvalue())

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


if __name__ == "__main__":
    unittest.main()
