#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import os
import re
import shlex
import stat
import sys
import tempfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = "/tmp/mf-local-dram-eid.env"
DEFAULT_TARGET = os.path.join(SCRIPT_DIR, "env")
EID_ENV_NAMES = (
    "MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID",
    "MF_LOCAL_DRAM_LOGICAL_DEVICE_ID",
    "MF_LOCAL_DRAM_TOPOLOGY",
    "MF_LOCAL_DRAM_UDMA",
    "MF_HOST_URMA_EID",
    "USE_LOCAL_EID",
)
OPTIONAL_EID_ENV_NAMES = ("MF_LOCAL_DRAM_AFFINITY_CPUS",)
EID_ENV_NAME_SET = frozenset(EID_ENV_NAMES + OPTIONAL_EID_ENV_NAMES)
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_assignment(line):
    try:
        tokens = shlex.split(line, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid quoting: {exc}") from exc
    if not tokens:
        return None
    if tokens[0] == "export":
        tokens = tokens[1:]
    if len(tokens) != 1 or "=" not in tokens[0]:
        raise ValueError("expected KEY=VALUE or export KEY=VALUE")
    name, value = tokens[0].split("=", 1)
    if not ENV_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid environment variable name: {name}")
    return name, value


def _read_eid_values(source_path):
    values = {}
    with open(source_path, "r", encoding="utf-8", newline="") as source_file:
        for line_number, line in enumerate(source_file, 1):
            try:
                assignment = _parse_assignment(line)
            except ValueError as exc:
                raise ValueError(f"invalid source line {line_number}: {exc}") from exc
            if assignment is None:
                continue
            name, value = assignment
            if name not in EID_ENV_NAME_SET:
                raise ValueError(f"unexpected environment variable at line {line_number}: {name}")
            if name in values:
                raise ValueError(f"duplicate environment variable at line {line_number}: {name}")
            values[name] = value
    missing = [name for name in EID_ENV_NAMES if name not in values]
    if missing:
        raise ValueError(f"missing EID environment variables: {','.join(missing)}")
    return values


def _replace_eid_lines(target_path, values):
    with open(target_path, "r", encoding="utf-8", newline="") as target_file:
        lines = target_file.readlines()
    updated = []
    replaced = set()
    for line_number, line in enumerate(lines, 1):
        try:
            assignment = _parse_assignment(line)
        except ValueError as exc:
            raise ValueError(f"invalid target line {line_number}: {exc}") from exc
        if assignment is None:
            updated.append(line)
            continue
        name, _ = assignment
        if name not in values:
            updated.append(line)
            continue
        if name in replaced:
            raise ValueError(f"duplicate EID environment variable in target line {line_number}: {name}")
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        updated.append(f"export {name}={shlex.quote(values[name])}{newline}")
        replaced.add(name)
    missing = [name for name in EID_ENV_NAMES if name not in replaced]
    if missing:
        raise ValueError(f"target env file is missing EID variables: {','.join(missing)}")
    return "".join(updated)


def _write_atomically(target_path, content):
    target_mode = stat.S_IMODE(os.stat(target_path).st_mode)
    temp_fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(target_path)}.",
                                          dir=os.path.dirname(target_path), text=True)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8", newline="") as temp_file:
            temp_file.write(content)
        os.chmod(temp_path, target_mode)
        os.replace(temp_path, target_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _build_parser():
    parser = argparse.ArgumentParser(description="Update the shared local validation env from EID query output")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"EID env output, default: {DEFAULT_SOURCE}")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"shared env file, default: {DEFAULT_TARGET}")
    return parser


def main():
    args = _build_parser().parse_args()
    source_path = os.path.abspath(args.source)
    target_path = os.path.abspath(args.target)
    try:
        values = _read_eid_values(source_path)
        content = _replace_eid_lines(target_path, values)
        _write_atomically(target_path, content)
    except (OSError, ValueError) as exc:
        print(f"ERROR env update failed source={source_path} target={target_path}: {exc}", file=sys.stderr)
        return 1
    print(f"Updated EID variables in {target_path} from {source_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
