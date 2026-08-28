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
#ifndef __MEMFABRIC_ACC_OFFLOAD_H__
#define __MEMFABRIC_ACC_OFFLOAD_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OFFLOAD_SCENE_LOCAL = 0,      /* single-card local DRAM memory pool */
    OFFLOAD_SCENE_SHARED = 1,     /* multi-card shared DRAM memory pool */
    OFFLOAD_SCENE_LOCAL_URMA = 2, /* single-card local URMA memory pool (host+device entity pair) */
} offload_scene_t;

/**
 * @brief offload_config_t.flags bits.
 *
 * OFFLOAD_FLAG_URMA_POOL: allocate the dram pool in URMA-compatible mode
 * (conn-based segment: plain host va + an independent HalHostRegister device
 * mapping). Required when the pool is registered to smem_trans with
 * DEVICE_URMA for cross-node remote writes. In this mode AIV operators must
 * use the device address from offload_get_dva() instead of the malloc address.
 */
#define OFFLOAD_FLAG_URMA_POOL (1U << 0)

typedef struct {
    uint32_t deviceId;     /* Device ID to bind */
    uint64_t reserveSize;  /* Reserved DRAM pool size in bytes, will be aligned up to GB. */
    uint64_t allocSize;    /* Allocated local physical DRAM size in bytes, will be aligned
                                                 up to GB. LOCAL: must equal reserveSize; SHARED: provides
                                                 the actual size. */
    uint32_t worldSize;    /* number of ranks in the group (used in SHARED scene) */
    uint32_t rankId;       /* local rank id, 0 is the allocator (used in SHARED scene) */
    offload_scene_t scene; /* LOCAL: single-card pool; SHARED: multi-card shared pool */
    uint32_t flags;        /* optional flags, see OFFLOAD_FLAG_xxx; 0 by default */
} offload_config_t;

/**
 * @brief Initialize the offload module.
 *
 * This function initializes the hybm big memory entity and loads the
 * offload library for sparse copy operations.
 *
 * @param config  [in] Init config, see offload_config_t.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_init(const offload_config_t &config);

/**
 * @brief Uninitialize the offload module.
 *
 * Releases the hybm big memory entity, unloads the extend library and
 * performs cleanup. Safe to call when not initialized.
 */
void offload_uninit();

/**
 * @brief Allocate host memory from the offload memory pool.
 *
 * Allocates a contiguous block from the pre-reserved hybm host memory.
 * The returned pointer is 16-byte aligned.
 *
 * @param size  [in] Memory size in bytes.
 * @param flags [in] optional flags
 * @return Non-zero address on success, 0 on failure.
 */
uint64_t offload_malloc(uint64_t size, uint64_t flags);

/**
 * @brief Free host memory previously allocated by offload malloc.
 *
 * Returns the memory block to the offload memory pool. The pointer must
 * have been obtained from offload malloc.
 *
 * @param ptr   [in] Address returned by offload malloc.
 * @param flags [in] optional flags
 */
void offload_free(uint64_t ptr, uint64_t flags);

/**
 * @brief Get the device virtual address (DVA) of a pool address from offload malloc.
 *
 * For URMA_POOL mode pools the DVA differs from the malloc address (an
 * independent HalHostRegister device mapping); for vmm unified pools the DVA
 * equals the malloc address. AIV operators (sparse_copy etc.) must use the DVA.
 *
 * @param hostPtr  [in] Address returned by offload malloc (or an interior address of it).
 * @param dvaPtr   [out] Device virtual address corresponding to the input address.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_get_dva(uint64_t hostPtr, uint64_t *dvaPtr);

/**
 * @brief Batch copy sparse data from host to device or from device to host.
 *
 * Submits a batch of h2d or d2h copy requests. Each request copies
 * data from srcPtrs[i] to dstPtrs[i] with length lenPtrs[i]. The copy is
 * executed asynchronously on the device stream.
 *
 * @param srcPtr            [in] Array of source addresses.
 * @param dstPtr            [in] Array of destination addresses.
 * @param lenPtr            [in] Array of byte counts to copy for each pair.
 * @param sizePtr           [in] Pointer to the number of entries in the arrays above.
 * @param deviceId          [in] Device ID to perform the copy on.
 * @param flag              [in] Kernel selector: 0 = sparse copy kernel (default), 1 = varlen copy kernel.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_sparse_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t sizePtr, uint16_t deviceId,
                            uint32_t flag);

/**
 * @brief Group-pack compacted copy: compact non-zero groupList entries to front.
 *
 * inputs / outputs / lens / groupList are ALL arrays of length N (= *numLocalExpertPtr). The kernel
 * scans [0, N); for the j-th non-zero entry (original index i where groupList[i] != 0,
 * j = 0..M-1), copies inputs[i] (lenPtrs[i] bytes) to outputs[j] and writes groupList[i]
 * to packedGroupList[j]. Zero groupList entries are skipped. After the call, outputs[0..M)
 * and packedGroupList[0..M) hold the compacted results; outputs[M..N) and the tail of
 * packedGroupList are left untouched.
 *
 * @param srcPtr             [in] Base address of the uint64_t inputs[] array (length N).
 * @param dstPtr             [in] Base address of the uint64_t outputs[] array (length N).
 * @param lenPtr             [in] Base address of the uint32_t lenPtrs[] array (length N).
 * @param numLocalExpertPtr           [in] Base address of a uint32_t scalar holding N (common array length).
 * @param groupListPtr       [in] Base address of the int64_t groupList[] array (length N).
 * @param packedGroupListPtr [in] Base address of the int64_t packedGroupList[] array (length >= M).
 * @param deviceId           [in] Device ID to perform the copy on.
 * @return 0 on success, non-zero error code on failure.
 */
int32_t offload_group_pack_copy(uint64_t srcPtr, uint64_t dstPtr, uint64_t lenPtr, uint64_t numLocalExpertPtr,
                                uint64_t groupListPtr, uint64_t packedGroupListPtr, uint16_t deviceId);

#ifdef __cplusplus
}
#endif

#endif //__MEMFABRIC_ACC_OFFLOAD_H__
