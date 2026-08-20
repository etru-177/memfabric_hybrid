#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -e
readonly ROOT_PATH=$(dirname $(readlink -f "$0"))
CURRENT_DIR=$(pwd)

BUILD_MODE="RELEASE"
BUILD_PYTHON="ON"
XPU_TYPE="NPU"
BUILD_TEST="OFF"
BUILD_HCOM="OFF"
BUILD_HCOM_WITH_RDMA="ON"
BUILD_HCOM_WITH_UB="OFF"
BUILD_ETCD_BACKEND="OFF"
BUILD_TOOL="cmake"
BUILD_LOCAL_DRAM_VALIDATION="OFF"
BUILD_ACC_OFFLOAD="ON"

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build_mode <mode>         Set build mode (RELEASE/DEBUG/ASAN), default: RELEASE"
    echo "  --build_python <ON/OFF>     Enable/disable Python build, default: ON"
    echo "  --xpu_type <GPU/NPU/NONE>   Set xpu dependency(GPU:CUDA, NPU:CANN), set none without xpu, default: NPU"
    echo "  --build_test <ON/OFF>       Enable/disable build and package test utilities and examples, default: OFF"
    echo "  --build_hcom <ON/OFF>       Enable/disable build and package hcom, default: OFF"
    echo "  --build_hcom_rdma <ON/OFF>  Enable/disable build and package hcom with rdma, default: ON"
    echo "  --build_hcom_ub <ON/OFF>    Enable/disable build and package hcom with ub, default: OFF"
    echo "  --build_etcd_backend <ON/OFF> Enable/disable build and package etcd backend so, default: OFF"
    echo "  --build_tool <cmake/bazel>  Set build tool (cmake/bazel), default: cmake"
    echo "  --build_local_dram_validation <ON/OFF> Enable temporary local DRAM validation, default: OFF"
    echo "  --build_acc_offload <ON/OFF> Enable/disable accelerator offload support, default: ON"
    echo "  --help                      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --build_mode DEBUG --build_python ON"
    echo ""
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --build_mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        --build_python)
            BUILD_PYTHON="$2"
            shift 2
            ;;
        --xpu_type)
            XPU_TYPE="$2"
            shift 2
            ;;
        --build_test)
            BUILD_TEST="$2"
            shift 2
            ;;
        --build_hcom)
            BUILD_HCOM="$2"
            shift 2
            ;;
        --build_hcom_rdma)
            BUILD_HCOM_WITH_RDMA="$2"
            shift 2
            ;;
        --build_hcom_ub)
            BUILD_HCOM_WITH_UB="$2"
            shift 2
            ;;
        --build_etcd_backend)
            BUILD_ETCD_BACKEND="$2"
            shift 2
            ;;
        --build_tool)
            BUILD_TOOL="$2"
            shift 2
            ;;
        --build_local_dram_validation)
            BUILD_LOCAL_DRAM_VALIDATION="$2"
            shift 2
            ;;
        --build_acc_offload)
            BUILD_ACC_OFFLOAD="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

if [[ "${BUILD_LOCAL_DRAM_VALIDATION}" != "ON" && "${BUILD_LOCAL_DRAM_VALIDATION}" != "OFF" ]]; then
    echo "Invalid --build_local_dram_validation value: ${BUILD_LOCAL_DRAM_VALIDATION}" >&2
    exit 1
fi
if [[ "${BUILD_LOCAL_DRAM_VALIDATION}" == "ON" && "${XPU_TYPE}" != "NPU" ]]; then
    echo "BUILD_LOCAL_DRAM_VALIDATION requires --xpu_type NPU" >&2
    exit 1
fi
if [[ "${BUILD_LOCAL_DRAM_VALIDATION}" == "ON" && "${BUILD_TOOL}" != "cmake" ]]; then
    echo "BUILD_LOCAL_DRAM_VALIDATION requires --build_tool cmake" >&2
    exit 1
fi
if [[ "${BUILD_ACC_OFFLOAD}" != "ON" && "${BUILD_ACC_OFFLOAD}" != "OFF" ]]; then
    echo "Invalid --build_acc_offload value: ${BUILD_ACC_OFFLOAD}" >&2
    exit 1
fi
if [[ "${XPU_TYPE}" == "NONE" ]]; then
    BUILD_ACC_OFFLOAD="OFF"
fi
if [[ "${BUILD_ACC_OFFLOAD}" == "OFF" && "${BUILD_TOOL}" != "cmake" ]]; then
    echo "BUILD_ACC_OFFLOAD=OFF currently requires --build_tool cmake" >&2
    exit 1
fi

echo "BUILD_MODE: $BUILD_MODE"
echo "BUILD_PYTHON: $BUILD_PYTHON"
echo "XPU_TYPE: $XPU_TYPE"
echo "BUILD_TEST: $BUILD_TEST"
echo "BUILD_HCOM: $BUILD_HCOM"
echo "BUILD_HCOM_RDMA: $BUILD_HCOM_WITH_RDMA"
echo "BUILD_HCOM_WITH_UB: $BUILD_HCOM_WITH_UB"
echo "BUILD_ETCD_BACKEND: $BUILD_ETCD_BACKEND"
echo "BUILD_TOOL: $BUILD_TOOL"
echo "BUILD_LOCAL_DRAM_VALIDATION: $BUILD_LOCAL_DRAM_VALIDATION"
echo "BUILD_ACC_OFFLOAD: $BUILD_ACC_OFFLOAD"

cd ${ROOT_PATH}

bash build.sh "${BUILD_MODE}" OFF OFF "${BUILD_PYTHON}" ON "${XPU_TYPE}" "${BUILD_TEST}" \
    "${BUILD_HCOM}" "${BUILD_HCOM_WITH_RDMA}" "${BUILD_HCOM_WITH_UB}" "${BUILD_ETCD_BACKEND}" \
    "${BUILD_TOOL}" "${BUILD_LOCAL_DRAM_VALIDATION}" "${BUILD_ACC_OFFLOAD}"

bash run_pkg_maker/make_run.sh "${BUILD_TEST}" "${XPU_TYPE}" "${BUILD_PYTHON}" "${BUILD_HCOM}" \
    "${BUILD_ETCD_BACKEND}" "${BUILD_ACC_OFFLOAD}"

cd ${CURRENT_DIR}
