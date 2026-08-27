/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/

#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_LOGGER_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_LOGGER_H

#include "mf_out_logger.h"

#define OFFLOAD_LOG_DEBUG(ARGS) MF_OUT_LOG("[OFFLOAD ", ock::mf::DEBUG_LEVEL, ARGS)
#define OFFLOAD_LOG_INFO(ARGS)  MF_OUT_LOG("[OFFLOAD ", ock::mf::INFO_LEVEL, ARGS)
#define OFFLOAD_LOG_WARN(ARGS)  MF_OUT_LOG("[OFFLOAD ", ock::mf::WARN_LEVEL, ARGS)
#define OFFLOAD_LOG_ERROR(ARGS) MF_OUT_LOG("[OFFLOAD ", ock::mf::ERROR_LEVEL, ARGS)
#define OFFLOAD_LOG_TRACE(ARGS) MF_OUT_LOG("[OFFLOAD ", ock::mf::TRACE_LEVEL, ARGS)

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_LOGGER_H
