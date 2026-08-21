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
install_flag=y
uninstall_flag=n
install_path_flag=n
install_for_all_flag=n
nocheck=n
script_dir=$(dirname $(readlink -f "$0"))
version1="none"
pkg_arch="none"
os1="none"
default_install_dir="/usr/local/memfabric_hybrid"
ascend_version="none"

function print_help() {
    echo "--install-path=<path>             Install to specific dir"
    echo "--uninstall                       Uninstall product"
    echo "--install-for-all                 Install for all user"
    echo "--no-check                        Skip check during installation"
}

function print()
{
    echo "[${1}] ${2}"
}

function get_version_in_file()
{
    if [ -f ${script_dir}/../version.info ]; then
        version1=$(awk -F ':' '$1=="Version" {gsub(/^[ \t]+|[ \t\r\n]+$/, "", $2); print $2}' ${script_dir}/../version.info)
        pkg_arch=$(awk -F ':' '$1=="Platform" {gsub(/^[ \t]+|[ \t\r\n]+$/, "", $2); print $2}' ${script_dir}/../version.info)
        os1=$(awk -F ':' '$1=="Kernel" {gsub(/^[ \t]+|[ \t\r\n]+$/, "", $2); print $2}' ${script_dir}/../version.info)
    fi
    print "INFO" "memfabric_hybrid version: ${version1} arch: ${pkg_arch} os: ${os1}"
}

function chmod_authority()
{
    chmod_file ${default_install_dir}
    chmod_file ${install_dir}
    local file_rights=$([ "${install_for_all_flag}" == "y" ] && echo 555 || echo 550)
    chmod ${file_rights} ${install_dir}/uninstall.sh
    chmod_dir ${default_install_dir} "550"
    chmod_dir ${install_dir} "550"
    local path_rights=$([ "${install_for_all_flag}" == "y" ] && echo 755 || echo 750)
    chmod ${path_rights} ${default_install_dir}
    chmod ${path_rights} ${install_dir}
}

function chmod_file()
{
    chmod_recursion ${1} "550" "file" "*.sh"
    chmod_recursion ${1} "440" "file" "*.bin"
    chmod_recursion ${1} "440" "file" "*.h"
    chmod_recursion ${1} "440" "file" "*.info"
    chmod_recursion ${1} "440" "file" "*.so"
    chmod_recursion ${1} "440" "file" "*.a"
    chmod_recursion ${1} "640" "file" "*.conf"
}

function chmod_dir()
{
    chmod_recursion ${1} ${2} "dir"
}

function chmod_recursion()
{
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ "${install_for_all_flag}" == "y" ] && echo ${rights} || echo $2)
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -name "$4" -exec chmod ${rights} {} \; 2>/dev/null
    fi
}

function parse_script_args()
{
    while true
    do
        case "$1" in
        --install-path=*)
            install_path_flag=y
            target_dir=$(echo $1 | cut -d"=" -f2-)
            target_dir=${target_dir}/memfabric_hybrid
            shift
        ;;
        --uninstall)
            uninstall_flag=y
            shift
        ;;
        --install-for-all)
            install_for_all_flag=y
            shift
        ;;
        --help)
            print_help
            exit 0
        ;;
        --no-check)
            nocheck=y
            shift
        ;;
        --*)
            shift
        ;;
        *)
            break
        ;;
        esac
    done
}

function check_owner()
{
    local cur_owner=$(whoami)
    if [ "<<XPU_TYPE>>" == "NPU" ]; then
        print "INFO" "Check ASCEND_TOOLKIT_HOME env..."
        if [ "${ASCEND_TOOLKIT_HOME}" == "" ]; then
            print "ERROR" "please check env ASCEND_TOOLKIT_HOME is set"
            exit 1
        fi

        if [ "${ASCEND_HOME_PATH}" == "" ]; then
            print "ERROR" "please check env ASCEND_HOME_PATH is set"
            exit 1
        else
            cann_path=${ASCEND_HOME_PATH}
        fi

        if [ ! -d "${cann_path}" ]; then
            print "ERROR" "can not find ${cann_path}"
            exit 1
        fi

        cann_owner=$(stat -c %U "${cann_path}")
        if [ "${cann_owner}" != "${cur_owner}" ]; then
            print "ERROR" "cur_owner is not same with CANN"
            exit 1
        fi
    fi

    if [[ "${cur_owner}" != "root" && "${install_flag}" == "y" ]]; then
        default_install_dir="${HOME}/memfabric_hybrid"
    fi

    if [ "${install_path_flag}" == "y" ]; then
        default_install_dir="${target_dir}"
    fi

    print "INFO" "Check owner success, XPU_TYPE is <<XPU_TYPE>>."
}

function delete_install_files()
{
    if [ -z "$1" ]; then
        return 0
    fi

    install_dir=$1
    print "INFO" "memfabric_hybrid $(basename $1) delete install files!"
    if [ -d ${install_dir} ]; then
        chmod -R 700 ${install_dir}
        rm -rf ${install_dir}
    elif [ -f ${install_dir} ]; then
        chmod 700 ${install_dir}
        rm -f ${install_dir}
    fi
}

function delete_latest()
{
    cd $1/..
    print "INFO" "memfabric_hybrid delete latest!"
    if [ -d "latest" ]; then
        chmod -R 700 latest
        rm -rf latest
    fi
    if [ -f "set_env.sh" ]; then
        chmod 500 set_env.sh
        rm -rf set_env.sh
    fi
}

function uninstall_process()
{
    if [ ! -d $1 ]; then
        return 0
    fi
    print "INFO" "memfabric_hybrid $(basename $1) uninstall start!"
    mf_dir=$(cd $1/..;pwd)
    delete_latest $1
    delete_install_files $1
    if [ "$2" == "y" -a -z "$(ls $mf_dir)" ]; then
        chmod -R 700 $mf_dir
        rm -rf $mf_dir
    fi
    print "INFO" "memfabric_hybrid $(basename $1) uninstall success!"
}

function uninstall()
{
    install_dir=${default_install_dir}/${version1}
    uninstall_process ${install_dir} y
}

function check_arch()
{
    # get arch
    if [ $( uname -m | grep -c -i "x86_64" ) -ne 0 ]; then
        local_arch="x86_64"
    elif [ $( uname -m | grep -c -i "aarch64" ) -ne 0 ]; then
        local_arch="aarch64"
    else
        print "ERROR" "it is not system of x86_64 or aarch64"
        exit 1
    fi

    if [ "${local_arch}" != "${pkg_arch}" ]; then
        print "ERROR" "Install failed, pkg_arch: ${pkg_arch}, os arch: ${local_arch}"
        exit 1
    fi

    local_os=$(uname -s | awk '{print tolower($0)}')
    if [ "${local_os}" != "${os1}" ]; then
        print "ERROR" "Install failed, pkg_os: ${os1}, os arch: ${local_os}"
        exit 1
    fi
}

function check_path()
{
    parentPath=$(dirname $(dirname ${default_install_dir}))
    if [ ! -d "${parentPath}" ];then
        print "ERROR" "install path ${parentPath} not exists, runpackage only support create one level of directory,need create $parentPath."
        exit 1
    fi
    username=$(whoami)
    # Run permission check as current user: $(whoami)
    if [ ! -x "${parentPath}" ]; then
        print "ERROR" "The ${username} do not have the permission to access ${parentPath}, please reset the directory to a right permission."
        exit 1
    fi

    install_dir=$1
    if [ ! -d ${install_dir} ]; then
        mkdir -p ${install_dir}
        if [ ! -d ${install_dir} ]; then
            print "ERROR" "Install failed, create ${install_dir} failed"
            exit 1
        fi
    fi
}

function get_ascend_version() {
    local soc_name=""
    # Prefer npu-smi to get the SoC name (still valid when lspci is unreliable in container/host)
    if command -v npu-smi &>/dev/null; then
        # some npu-smi versions report the chip as bare "910B3" without the "Ascend" prefix
        soc_name=$(npu-smi info 2>/dev/null | grep -oE "Ascend(9|3)[0-9]+[A-Za-z0-9]*" | head -n1)
        if [ -z "${soc_name}" ]; then
            soc_name=$(npu-smi info 2>/dev/null | grep -oE "(9|3)[0-9]{2}[A-Za-z0-9]*" | grep -E "^(910|950|310)" | head -n1)
        fi
    fi
    if [ -n "${soc_name}" ]; then
        case "${soc_name}" in
             *950*|*Ascend910_95*)
                 ascend_version="A5" ;;
             *910B*)
                 ascend_version="A2" ;;
             *910*)
                 ascend_version="A3" ;;
             *)
                ascend_version="none" ;;
        esac
        return
    fi
    # lspci fallback (PCI device IDs: d802=A2, d803=A3, d806=A5)
    if [ $(lspci 2>/dev/null | grep -ci "Processing.*Huawei.*d806") -gt 0 ]; then
        ascend_version="A5"
    elif [ $(lspci 2>/dev/null | grep -ci "Processing.*Huawei.*d803") -gt 0 ]; then
        ascend_version="A3"
    elif [ $(lspci 2>/dev/null | grep -ci "Processing.*Huawei.*d802") -gt 0 ]; then
        ascend_version="A2"
    fi
}

function install_wheel_package() {
    wheel_dir="$1"
    wheel_name="$2"
    python_version="$3"
    if [ -z ${wheel_dir} ]; then
        print "ERROR" "invalid wheel package directory, skip install wheel."
        return
    fi
    if [ -z "${wheel_name}" ]; then
        print "ERROR" "empty wheel package name, skip install wheel."
        return
    fi
    if [ -z "${python_version}" ]; then
        print "ERROR" "empty python version, skip install wheel."
        return
    fi

    wheel_package=$(find "${wheel_dir}" -name "${wheel_name}-${version1}-cp${python_version}*" -print -quit 2>/dev/null)
    if [ -z "${wheel_package}" ]; then
        # Fallback to same Python ABI even if version string in wheel does not exactly match version.info.
        wheel_package=$(find "${wheel_dir}" -name "${wheel_name}-*-cp${python_version}*" -print -quit 2>/dev/null)
    fi
    if [ -z "${wheel_package}" ]; then
        print "WARNING" "not found wheel package ${wheel_name} for python-${python_version}, skip install wheel."
        print "INFO" "available wheel packages:"
        find "${wheel_dir}" -maxdepth 1 -name "${wheel_name}-*.whl" -printf "  %f\n" 2>/dev/null
        return
    fi

    print "INFO" "install wheel package: $(basename "${wheel_package}")"
    pip3 install "${wheel_package}" --force-reinstall
}

function try_install_extend()
{
    bisheng_path=$(which bisheng 2>/dev/null)
    if [ -z "${bisheng_path}" ]; then
        print "WARNING" "bisheng Not Found, skip install extend lib."
        return
    fi

    if [ "${ascend_version}" == "none" ]; then
        print "ERROR" "cannot detect ascend chip version (got 'none'), abort install extend lib. \
Ensure npu-smi is in PATH (source CANN set_env.sh) before running .run."
        exit 1
    fi

    cce_param="--cce-aicore-arch=dav-c220"
    if [ "${ascend_version}" == "A5" ]; then
        cce_param="--cce-aicore-arch=dav-c310"
    fi

    cd ${script_dir}/../copy_extend
    bisheng -x asc hybm_copy_kernel.cpp -fPIC -shared -g -o libmf_hybm_copy_extend.so ${cce_param}
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        cp ./*.so ${install_dir}//${pkg_arch}-${os1}/lib64
        print "INFO" "install hybm extend lib success"
    else
        print "ERROR" "install hybm extend lib failed (bisheng exit ${exit_code}), maybe cann version is old, least 8.3.RC1"
        exit 1
    fi

    if [ ! -d "${script_dir}/../accoffload_operators" ]; then
        print "INFO" "accoffload_operators not found, skip install accoffload extend lib."
        return
    fi

    cd ${script_dir}/../accoffload_operators

    export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
    case "${ascend_version}" in
        A5)     mf_soc_version="ascend950pr_9599" ;;
        A3)     mf_soc_version="ascend910_9382" ;;
        A2)     print "WARNING" "A2 (Ascend910B) unsupported for accoffload DRAM offload, skip accoffload kernel build."
                return ;;
        *)      print "ERROR" "unsupported ascend chip '${ascend_version}' for accoffload kernel cmake build."
                rm -rf build_cmake
                exit 1 ;;
    esac
    rm -rf build_cmake && mkdir build_cmake
    cmake -S . -B build_cmake \
        -DSOC_VERSION=${mf_soc_version} \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
    if [ $? -ne 0 ]; then
        print "ERROR" "cmake configure accoffload kernel failed."
        rm -rf build_cmake
        exit 1
    fi
    cmake --build build_cmake -j$(nproc 2>/dev/null || echo 8)
    if [ $? -ne 0 ]; then
        print "ERROR" "cmake build accoffload kernel failed."
        rm -rf build_cmake
        exit 1
    fi
    KERN_A=build_cmake/lib/libmf_accoffload_kernel.a
    if [ ! -f "${KERN_A}" ]; then
        KERN_A=$(find build_cmake -name 'libmf_accoffload_kernel.a' -print -quit 2>/dev/null)
    fi
    if [ -z "${KERN_A}" ] || [ ! -f "${KERN_A}" ]; then
        print "ERROR" "libmf_accoffload_kernel.a not found after cmake build."
        rm -rf build_cmake
        exit 1
    fi

    python_bin=python3
    torch_dir=$(${python_bin} -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
    torch_npu_dir=$(${python_bin} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))" 2>/dev/null)
    if [ -z "${torch_dir}" ] || [ -z "${torch_npu_dir}" ]; then
        print "ERROR" "torch/torch_npu not found, cannot build accoffload extend lib."
        rm -rf build_cmake
        exit 1
    fi
    ascend_home=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

    abi_flag="-D_GLIBCXX_USE_CXX11_ABI=0"
    abi_val=$(${python_bin} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null)
    if [ "${abi_val}" == "1" ]; then
        abi_flag="-D_GLIBCXX_USE_CXX11_ABI=1"
    fi

    g++ -c -fPIC -std=c++17 -O3 -fstack-protector-strong \
        -Wno-unused-parameter -Wno-unused-function -Wunused-value -Wcast-align \
        -Wcast-qual -Wwrite-strings -Wsign-compare -Wextra \
        -fvisibility-inlines-hidden -ftrapv \
        ${abi_flag} \
        -isystem ${ascend_home}/include \
        -isystem ${ascend_home}/include/experiment/runtime/runtime/ \
        -isystem ${torch_dir}/include \
        -isystem ${torch_dir}/include/torch/csrc/api/include \
        -isystem ${torch_dir}/include/torch/csrc/utils \
        -isystem ${torch_dir}/include/c10/util \
        -isystem ${torch_dir}/include/c10/core \
        -isystem ${torch_dir}/include/ATen \
        -isystem ${torch_dir}/include/ATen/detail \
        -isystem ${torch_npu_dir}/include \
        -isystem ${torch_npu_dir}/include/torch_npu/csrc/aten \
        -isystem ${torch_npu_dir}/include/torch_npu/csrc/core/npu \
        acc_offload_operators_launch.cpp -o acc_offload_operators_launch.o
    if [ $? -ne 0 ]; then
        print "ERROR" "g++ compile acc_offload_operators_launch.cpp failed."
        rm -rf build_cmake
        rm -f acc_offload_operators_launch.o
        exit 1
    fi

    g++ -shared -fPIC -o libmf_hybm_accoffload.so \
        acc_offload_operators_launch.o \
        -Wl,--whole-archive "${KERN_A}" -Wl,--no-whole-archive \
        -L${torch_dir}/lib -ltorch -lc10 -ltorch_python \
        -L${torch_npu_dir}/lib -ltorch_npu \
        -L${ascend_home}/lib64 -lopapi \
        -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now \
        -Wl,-rpath,\$ORIGIN -Wl,-rpath,${torch_dir}/lib -Wl,-rpath,${torch_npu_dir}/lib -Wl,-rpath,${ascend_home}/lib64
    if [ $? -ne 0 ]; then
        print "ERROR" "link libmf_hybm_accoffload.so failed."
        rm -rf build_cmake
        rm -f acc_offload_operators_launch.o
        exit 1
    fi

    \cp libmf_hybm_accoffload.so ${install_dir}//${pkg_arch}-${os1}/lib64
    rm -f libmf_hybm_accoffload.so acc_offload_operators_launch.o
    rm -rf build_cmake
    print "INFO" "install accoffload extend lib success"
}

function install_to_path()
{
    install_dir=${default_install_dir}/${version1}
    if [ -d ${install_dir} ]; then
        print "INFO" "The installation directory exists, uninstall first"
    fi
    uninstall_process ${install_dir}
    check_path ${install_dir}

    cd ${install_dir}
    cp -r ${script_dir}/../${pkg_arch}-${os1} ${install_dir}/
    mkdir -p "${install_dir}/${pkg_arch}-${os1}/script"
    cp "${script_dir}/mem_scan.py" "${install_dir}/${pkg_arch}-${os1}/script/"
    cp -r ${script_dir}/../include ${install_dir}/
    cp -r ${script_dir}/uninstall.sh ${install_dir}/
    cp -r ${script_dir}/../version.info ${install_dir}/

    cd ${default_install_dir}
    ln -snf ${version1} latest

    pip_path=$(which pip3 2>/dev/null)
    if [ -z "$pip_path" ]; then
        print "WARNING" "pip3 Not Found, skip install wheel package."
        return
    fi

    wheel_dir="${install_dir}"/"${pkg_arch}"-"${os1}"/wheel
    python_version=$(python3 -c "import sys; print(''.join(map(str, sys.version_info[:2])))")

    install_wheel_package "${wheel_dir}" memfabric_hybrid "${python_version}"
}

function generate_set_env()
{
    local env_file="${default_install_dir}/set_env.sh"
    local test_dir="${default_install_dir}/latest/${pkg_arch}-${os1}/test"
    cat >"${env_file}" <<EOF
export MEMFABRIC_HYBRID_HOME_PATH=${default_install_dir}/latest
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=${default_install_dir}/latest/${pkg_arch}-${os1}/lib64
export LD_LIBRARY_PATH=${default_install_dir}/latest/${pkg_arch}-${os1}/lib64:\$LD_LIBRARY_PATH
export PATH=${default_install_dir}/latest/${pkg_arch}-${os1}/bin:\$PATH
EOF
    if [ -d "${test_dir}" ]; then
        echo "export PATH=${test_dir}:\$PATH" >>"${env_file}"
    fi
}

function install_process()
{
    if [ -n "${target_dir}" ]; then
        if [[ ! "${target_dir}" = /* ]]; then
            print "ERROR" "Install failed, [ERROR] use absolute path for --install-path argument"
            exit 1
        fi
    fi

    print "INFO" "memfabric_hybrid start install into ${default_install_dir}"
    install_to_path
    try_install_extend
    generate_set_env
}

function main()
{
    parse_script_args $*
    get_version_in_file
    get_ascend_version
    print "INFO" "found ascend env is ${ascend_version}."

    if [ "$uninstall_flag" == "y" ]; then
        uninstall
    elif [ "$install_flag" == "y" ] || [ "$install_path_flag" == "y" ]; then
        if [ "$nocheck" == "y" ]; then
            print "INFO" "skip check arch and owner."
        else
            check_arch
            check_owner
        fi

        install_process
        chmod_authority
        print "INFO" "memfabric_hybrid install success"
    fi
}

main $*
exit 0
