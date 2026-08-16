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
#ifndef MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H
#define MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H

#include <mutex>
#include "acc_offload_define.h"

namespace ock {
namespace offload {

using AccOffloadSparseCopyFunc = void (*)(uint64_t *, uint64_t *, uint32_t *, uint32_t *, uint8_t, uint32_t);

using AccOffloadGroupPackCopyFunc = void (*)(uint64_t *, uint64_t *, uint32_t *, uint32_t *, int64_t *, int64_t *,
                                             uint8_t);

class AccOffloadLaunchApi {
public:
    static int32_t TryLoadLibrary();
    static void CleanupLibrary();

    static inline int32_t AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                               uint32_t *sizePtr, uint8_t devIdx, uint32_t flag = 0)
    {
        if (pAccOffloadSparseCopy == nullptr) {
            return OFFLOAD_UNLOAD;
        }

        pAccOffloadSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, devIdx, flag);
        return OFFLOAD_OK;
    }

    static inline int32_t AccOffloadGroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs,
                                                  uint32_t *numLocalExpertPtr, int64_t *groupList,
                                                  int64_t *packedGroupList, uint8_t devIdx)
    {
        if (pAccOffloadGroupPackCopy == nullptr) {
            return OFFLOAD_UNLOAD;
        }

        pAccOffloadGroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, devIdx);
        return OFFLOAD_OK;
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *libHandle;
    static const char *gAccOffloadLibName;

    static AccOffloadSparseCopyFunc pAccOffloadSparseCopy;
    static AccOffloadGroupPackCopyFunc pAccOffloadGroupPackCopy;
};

} // namespace offload
} // namespace ock

#endif // MEMFABRIC_HYBRID_ACC_OFFLOAD_LAUNCH_H
