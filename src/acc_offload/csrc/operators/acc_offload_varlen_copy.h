/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
 */

#ifndef ACC_OFFLOAD_VARLEN_COPY_H
#define ACC_OFFLOAD_VARLEN_COPY_H

#include "acc_offload_varlen_copy_core.h"

/*
 * Batched variable-length copy kernel entry. Behaviour is identical to
 * OffloadSparseCopyKernel (both share OffloadVarlenCopyCore); this class
 * exists so the varlen aicore entry carries its own kernel name.
 */
template<typename T>
class OffloadVarlenCopyKernel : public OffloadVarlenCopyCore<T> {
public:
    HYBM_AICORE_KERNEL OffloadVarlenCopyKernel() {}
};

#endif // ACC_OFFLOAD_VARLEN_COPY_H
