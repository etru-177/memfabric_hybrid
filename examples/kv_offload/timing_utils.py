#!/usr/bin/env python3
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

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def time_with_npu_event(fn) -> float:
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start_event.record()
    fn()
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / 1e3


def report_timing(rank_id: int, method: str, elapsed: float, total_bytes: int):
    bandwidth = total_bytes / elapsed if elapsed > 0 else float('inf')
    logger.info(
        f"rank_id:{rank_id} [{method}] time: {elapsed * 1e3:.3f} ms, "
        f"data: {total_bytes / (1 << 20):.2f} MiB, bandwidth: {bandwidth / (1 << 30):.2f} GiB/s"
    )
