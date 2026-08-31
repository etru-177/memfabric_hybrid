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

"""python api for memfabric_hybrid."""

import glob
import os
import platform
import shutil
import subprocess
import sys

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# 消除whl压缩包的时间戳差异
os.environ["SOURCE_DATE_EPOCH"] = "0"
current_version = os.getenv("MEMFABRIC_VERSION")
if not current_version:
    print("Error: MEMFABRIC_VERSION environment variable must be set.", file=sys.stderr)
is_manylinux = check_env_flag("IS_MANYLINUX", "FALSE")
xpu_type = os.getenv("XPU_TYPE", "NPU")

if xpu_type not in ("NPU", "NONE", "GPU"):
    raise ValueError("XPU_TYPE must be exactly NPU, NONE, or GPU")
if xpu_type == "NONE":
    current_version += "+cpu"
elif xpu_type == "GPU":
    current_version += "+gpu"


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../.."))
_BUILD_AICPU_ASSETS = xpu_type == "NPU" and check_env_flag("BUILD_ACC_OFFLOAD", "ON")


_AICPU_WLIST = {
    "_hybm_src/include/hybm_def.h",
    "_hybm_src/csrc/common/hybm_batch_copy_route.h",
    "_hybm_src/csrc/common/hybm_define.h",
    "_hybm_src/csrc/common/hybm_types.h",
    "_hybm_src/csrc/under_api/dl_hcomm_api.h",
    "_acc_offload_src/csrc/operators/aicpu/hybm_aggregate_urma_demo.cc",
    "_acc_offload_src/csrc/operators/aicpu/hybm_aggregate_urma_demo.h",
    "_acc_offload_src/csrc/operators/aicpu/hybm_batch_copy.cc",
    "_acc_offload_src/csrc/operators/aicpu/hybm_batch_copy.h",
    "_acc_offload_src/csrc/operators/aicpu/hybm_kvcache_scatter_copy.cc",
    "_acc_offload_src/csrc/operators/aicpu/hybm_kvcache_scatter_copy.h",
    "_util_src/mf_out_logger.h",
    "_util_src/mf_spinlock.h",
}

_WREL_SRC = [
    ("_hybm_src/include/", "src/hybm/include/"),
    ("_hybm_src/csrc/common/", "src/hybm/csrc/common/"),
    ("_hybm_src/csrc/under_api/", "src/hybm/csrc/under_api/"),
    ("_acc_offload_src/", "src/acc_offload/"),
    ("_util_src/", "src/util/csrc/"),
]


def _clean_aicpu_assets(build_lib):
    pkg_build = os.path.join(build_lib, "memfabric_hybrid")
    for subdir in ("_ops", "_hybm_src", "_acc_offload_src", "_util_src"):
        d = os.path.join(pkg_build, subdir)
        if os.path.isdir(d):
            shutil.rmtree(d)
    marker = os.path.join(pkg_build, ".hybm_aicpu_provision_wheel_only")
    if os.path.exists(marker):
        os.remove(marker)


def _copy_wlist_assets(build_lib):
    """Copy ops tree and AICPU white-list files into build_lib."""
    pkg_build = os.path.join(build_lib, "memfabric_hybrid")
    # Recursively copy entire ops directory (no whitelist filtering).
    shutil.copytree(
        os.path.join(_PROJECT_ROOT, "src/hybm/ops"),
        os.path.join(pkg_build, "_ops"),
    )
    for wl_rel in _AICPU_WLIST:
        src_rel = None
        for wprefix, spath in _WREL_SRC:
            if wl_rel.startswith(wprefix):
                inner = wl_rel[len(wprefix) :]
                src_rel = os.path.join(spath, inner)
                break
        assert src_rel is not None, "unmatched white-list entry: " + wl_rel
        dst_full = os.path.join(pkg_build, wl_rel)
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        shutil.copy2(os.path.join(_PROJECT_ROOT, src_rel), dst_full)


class _AicpuBuildPy(build_py):
    """Custom build_py that stages AICPU white-list files into build_lib."""

    def run(self):
        super().run()
        _clean_aicpu_assets(self.build_lib)
        if _BUILD_AICPU_ASSETS:
            _copy_wlist_assets(self.build_lib)
            pkg_build = os.path.join(self.build_lib, "memfabric_hybrid")
            marker = os.path.join(pkg_build, ".hybm_aicpu_provision_wheel_only")
            with open(marker, "w") as f:
                f.write("")


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


class BuildWheel(bdist_wheel):
    def run(self):
        # Force build_py even if --skip-build was given, so AICPU assets
        # are always staged and marker created in the wheel.
        if self.skip_build:
            self.run_command("build_py")
        bdist_wheel.run(self)

        auditwheel = shutil.which("auditwheel")
        if not auditwheel:
            print(
                "Warning: auditwheel is not installed. Skipping wheel repair. "
                "Please install auditwheel if repaired wheels are required.",
                file=sys.stderr,
            )
            return

        file = glob.glob(os.path.join(self.dist_dir, "*-linux_*.whl"))[0]
        if is_manylinux:
            auditwheel_cmd = [
                auditwheel,
                "-v",
                "repair",
                "--plat",
                f"manylinux_2_27_{platform.machine()}",
                "--plat",
                f"manylinux_2_28_{platform.machine()}",
                "-w",
                self.dist_dir,
                file,
            ]
        else:
            auditwheel_cmd = [
                auditwheel,
                "-v",
                "repair",
                "-w",
                self.dist_dir,
                file,
            ]
        subprocess.check_call(auditwheel_cmd)
        os.remove(file)


pkgs = find_namespace_packages()
print(pkgs)

setup(
    name="memfabric_hybrid",
    version=current_version,
    author="",
    author_email="",
    description="python api for memfabric hybrid",
    packages=find_namespace_packages(exclude=("tests*",)),
    url="https://gitcode.com/Ascend/memfabric_hybrid",
    license="Mulan PSL v2",
    python_requires=">=3.8",
    zip_safe=False,
    package_data={
        "memfabric_hybrid": [
            "_pymf_hybrid*.so",
            "_pymf_transfer*.so",
            "_pymf_acc_offload*.so",
            "lib/lib*.so",
            "include/smem/host/*.h",
            "include/smem/device/*.h",
            "include/hybm/*.h",
            "VERSION",
        ]
    },
    cmdclass={
        "build_py": _AicpuBuildPy,
        "bdist_wheel": BuildWheel,
    },
    distclass=BinaryDistribution,
)
