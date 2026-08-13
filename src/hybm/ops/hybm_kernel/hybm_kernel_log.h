/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MF_HYBM_OPS_HYBM_KERNEL_HYBM_KERNEL_LOG_H
#define MF_HYBM_OPS_HYBM_KERNEL_HYBM_KERNEL_LOG_H

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include "dlog_pub.h"

#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HYBM_KERNEL_MODULE_NAME static_cast<int32_t>(APP)
#define HYBM_KERNEL_LOG_HEADER  "[HYBM]"

class HybmKernelLog {
public:
    static uint64_t GetTid()
    {
#ifdef __GNUC__
        return static_cast<uint64_t>(syscall(__NR_gettid));
#else
        return static_cast<uint64_t>(getpid());
#endif
    }
};

#define HYBM_LOGE(ERROR_CODE, fmt, ...)                                                                         \
    do {                                                                                                        \
        dlog_error(HYBM_KERNEL_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %d %s" fmt, HybmKernelLog::GetTid(),      \
                   &__FUNCTION__[0U], static_cast<int32_t>(ERROR_CODE), HYBM_KERNEL_LOG_HEADER, ##__VA_ARGS__); \
    } while (false)

#define HYBM_LOGW(fmt, ...)                                                                                   \
    do {                                                                                                      \
        dlog_warn(HYBM_KERNEL_MODULE_NAME, "%" PRIu64 " %s:" fmt, HybmKernelLog::GetTid(), &__FUNCTION__[0U], \
                  ##__VA_ARGS__);                                                                             \
    } while (false)

#define HYBM_LOGI(fmt, ...)                                                                                   \
    do {                                                                                                      \
        dlog_info(HYBM_KERNEL_MODULE_NAME, "%" PRIu64 " %s:" fmt, HybmKernelLog::GetTid(), &__FUNCTION__[0U], \
                  ##__VA_ARGS__);                                                                             \
    } while (false)

#define HYBM_LOGD(fmt, ...)                                                                                    \
    do {                                                                                                       \
        dlog_debug(HYBM_KERNEL_MODULE_NAME, "%" PRIu64 " %s:" fmt, HybmKernelLog::GetTid(), &__FUNCTION__[0U], \
                   ##__VA_ARGS__);                                                                             \
    } while (false)

#ifdef __cplusplus
}
#endif

#endif // MF_HYBM_OPS_HYBM_KERNEL_HYBM_KERNEL_LOG_H
