cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(mf_accoffload_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif ()

if (NOT SOC_VERSION)
    message(FATAL_ERROR "SOC_VERSION must be set, e.g. -DSOC_VERSION=ascend910_9382 (A3) or ascend950pr_9599 (A5)")
endif ()
message(STATUS "acc_offload: SOC_VERSION = ${SOC_VERSION}")

include(${CMAKE_CURRENT_SOURCE_DIR}/config_ascendc.cmake)

ascendc_library(mf_accoffload_kernel STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/acc_offload_sparse_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/acc_offload_group_pack_copy.cpp
)

ascendc_include_directories(mf_accoffload_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

ascendc_compile_definitions(mf_accoffload_kernel PRIVATE ASCENDC_DUMP=0)
