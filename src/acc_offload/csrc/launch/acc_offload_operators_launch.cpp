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

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <climits>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unistd.h>

#include "acl/acl_rt.h"
#include "acc_offload_define.h"
#include "acc_offload_operators.h"
#include "hybm_aggregate_urma_demo.h"
#include "hybm_batch_copy.h"
#include "hybm_kvcache_scatter_copy.h"
#include "hybm_def.h"

namespace {
constexpr char kKernelJsonSuffix[] = "/opp/vendors/cust/op_impl/aicpu/config/libcann_hybm_kernel.json";
constexpr char kDefaultAscendPath[] = "/usr/local/Ascend/cann";
constexpr char kAggregateUrmaDemoFunctionName[] = "HybmAggregateUrmaDemo";
constexpr char kBatchCopyFunctionName[] = "HybmBatchCopy";
constexpr char kKvcacheScatterCopyFunctionName[] = "HybmKvcacheScatterCopy";
constexpr uint32_t kKernelBlockDim = 1U;
constexpr uint16_t kKernelTimeoutSeconds = 120U;

struct BatchCopyKernelCache {
    aclrtBinHandle binary{nullptr};
    aclrtFuncHandle function{nullptr};
};

std::mutex gBatchCopyKernelMutex;
std::unordered_map<uint16_t, BatchCopyKernelCache> gAggregateUrmaDemoKernelCache;
std::unordered_map<uint16_t, BatchCopyKernelCache> gBatchCopyKernelCache;
std::unordered_map<uint16_t, BatchCopyKernelCache> gKvcacheScatterCopyKernelCache;

std::string GetKernelJsonPath()
{
    const char *ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr || ascendPath[0] == '\0') {
        ascendPath = kDefaultAscendPath;
    }
    return std::string(ascendPath) + kKernelJsonSuffix;
}

int32_t ResolveKernelJsonPath(std::string &path)
{
    const auto jsonPath = GetKernelJsonPath();
    char resolvedPath[PATH_MAX] = {};
    if (realpath(jsonPath.c_str(), resolvedPath) == nullptr || access(resolvedPath, R_OK) != 0) {
        OFFLOAD_LOG_ERROR("AICPU kernel JSON is unavailable, path: " << jsonPath);
        return BM_NOT_INITIALIZED;
    }
    path = resolvedPath;
    return BM_OK;
}

int32_t LoadKernelFunction(const std::string &jsonPath, const char *functionName, BatchCopyKernelCache &cache)
{
    aclrtBinaryLoadOption loadOption{};
    loadOption.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    loadOption.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions loadOptions{&loadOption, 1U};
    auto ret = aclrtBinaryLoadFromFile(jsonPath.c_str(), &loadOptions, &cache.binary);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtBinaryLoadFromFile failed, path: " << jsonPath << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    if (cache.binary == nullptr) {
        OFFLOAD_LOG_ERROR("aclrtBinaryLoadFromFile returned null handle, path: " << jsonPath);
        return BM_DL_FUNCTION_FAILED;
    }

    ret = aclrtBinaryGetFunction(cache.binary, functionName, &cache.function);
    if (ret != ACL_SUCCESS || cache.function == nullptr) {
        OFFLOAD_LOG_ERROR("aclrtBinaryGetFunction failed, function: " << functionName << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t GetKernelFunction(uint16_t deviceId, const char *functionName,
                          std::unordered_map<uint16_t, BatchCopyKernelCache> &cacheMap, aclrtFuncHandle &function)
{
    try {
        std::lock_guard<std::mutex> guard(gBatchCopyKernelMutex);
        const auto cached = cacheMap.find(deviceId);
        if (cached != cacheMap.end()) {
            function = cached->second.function;
            return BM_OK;
        }

        std::string jsonPath;
        auto ret = ResolveKernelJsonPath(jsonPath);
        if (ret != BM_OK) {
            return ret;
        }
        BatchCopyKernelCache cache;
        ret = LoadKernelFunction(jsonPath, functionName, cache);
        if (ret != BM_OK) {
            return ret;
        }
        const auto inserted = cacheMap.emplace(deviceId, cache);
        if (!inserted.second) {
            function = inserted.first->second.function;
            return BM_OK;
        }
        function = cache.function;
        return BM_OK;
    } catch (const std::bad_alloc &) {
        OFFLOAD_LOG_ERROR("allocate AICPU kernel cache failed, deviceId: " << deviceId
                                                                           << " function: " << functionName);
        return BM_MALLOC_FAILED;
    } catch (const std::exception &exception) {
        OFFLOAD_LOG_ERROR("load AICPU kernel raised exception, deviceId: " << deviceId << " function: " << functionName
                                                                           << " error: " << exception.what());
        return BM_DL_FUNCTION_FAILED;
    } catch (...) {
        OFFLOAD_LOG_ERROR("load AICPU kernel raised unknown exception, deviceId: " << deviceId
                                                                                   << " function: " << functionName);
        return BM_DL_FUNCTION_FAILED;
    }
}

int32_t GetBatchCopyFunction(uint16_t deviceId, aclrtFuncHandle &function)
{
    return GetKernelFunction(deviceId, kBatchCopyFunctionName, gBatchCopyKernelCache, function);
}

int32_t GetAggregateUrmaDemoFunction(uint16_t deviceId, aclrtFuncHandle &function)
{
    return GetKernelFunction(deviceId, kAggregateUrmaDemoFunctionName, gAggregateUrmaDemoKernelCache, function);
}

int32_t GetKvcacheScatterCopyFunction(uint16_t deviceId, aclrtFuncHandle &function)
{
    return GetKernelFunction(deviceId, kKvcacheScatterCopyFunctionName, gKvcacheScatterCopyKernelCache, function);
}

int32_t PrepareBatchCopyKernelArgs(aclrtFuncHandle function, uint64_t srcPtrs, uint64_t dstPtrs, uint64_t lenPtrs,
                                   uint32_t listNum, aclrtArgsHandle &argsHandle)
{
    HybmBatchCopyParam param{};
    param.list_num = listNum;
    param.dst_buf_addr_list = reinterpret_cast<void **>(static_cast<uintptr_t>(dstPtrs));
    param.src_buf_addr_list = reinterpret_cast<void **>(static_cast<uintptr_t>(srcPtrs));
    param.len_list = reinterpret_cast<uint64_t *>(static_cast<uintptr_t>(lenPtrs));

    argsHandle = nullptr;
    auto ret = aclrtKernelArgsInit(function, &argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsInit failed, listNum: " << listNum << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    aclrtParamHandle paramHandle = nullptr;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(param), &paramHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsAppend failed, listNum: " << listNum << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtKernelArgsFinalize(argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsFinalize failed, listNum: " << listNum << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }

    return BM_OK;
}

int32_t LaunchBatchCopyKernel(aclrtFuncHandle function, aclrtStream stream, uint64_t srcPtrs, uint64_t dstPtrs,
                              uint64_t lenPtrs, uint32_t listNum, uint16_t deviceId)
{
    aclrtArgsHandle argsHandle = nullptr;
    auto ret = PrepareBatchCopyKernelArgs(function, srcPtrs, dstPtrs, lenPtrs, listNum, argsHandle);
    if (ret != BM_OK) {
        return ret;
    }

    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kKernelTimeoutSeconds;
    aclrtLaunchKernelCfg config{&attr, 1U};
    ret = aclrtLaunchKernelWithConfig(function, kKernelBlockDim, stream, &config, argsHandle, nullptr);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtLaunchKernelWithConfig failed, deviceId: "
                          << deviceId << " stream: " << stream << " kernel: " << kBatchCopyFunctionName
                          << " listNum: " << listNum << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtSynchronizeStream failed, deviceId: " << deviceId << " stream: " << stream
                                                                      << " kernel: " << kBatchCopyFunctionName
                                                                      << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t LaunchSparseCopyUrma(uint64_t srcPtrs, uint64_t dstPtrs, uint64_t lenPtrs, uint32_t listNum, uint16_t deviceId)
{
    try {
        c10_npu::OptionalNPUGuard npuGuard;
        npuGuard.set_index(deviceId);
        auto stream = c10_npu::getCurrentNPUStream(deviceId);
        aclrtStream npuStream = stream.stream(false);
        if (npuStream == nullptr) {
            OFFLOAD_LOG_ERROR("current NPU stream is null, deviceId: " << deviceId);
            return BM_DL_FUNCTION_FAILED;
        }

        aclrtFuncHandle function = nullptr;
        auto ret = GetBatchCopyFunction(deviceId, function);
        if (ret != BM_OK) {
            return ret;
        }
        return LaunchBatchCopyKernel(function, npuStream, srcPtrs, dstPtrs, lenPtrs, listNum, deviceId);
    } catch (const std::exception &exception) {
        OFFLOAD_LOG_ERROR("sparse_copy_urma launch raised exception, deviceId: " << deviceId
                                                                                 << " error: " << exception.what());
        return BM_ERROR;
    } catch (...) {
        OFFLOAD_LOG_ERROR("sparse_copy_urma launch raised unknown exception, deviceId: " << deviceId);
        return BM_ERROR;
    }
}

int32_t PrepareAggregateUrmaDemoArgs(aclrtFuncHandle function, HybmAggregateUrmaDemoParam param,
                                     aclrtArgsHandle &argsHandle)
{
    argsHandle = nullptr;
    auto ret = aclrtKernelArgsInit(function, &argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsInit failed for aggregate URMA demo, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    aclrtParamHandle paramHandle = nullptr;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(param), &paramHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsAppend failed for aggregate URMA demo, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtKernelArgsFinalize(argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsFinalize failed for aggregate URMA demo, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t LaunchAggregateUrmaDemoKernel(aclrtFuncHandle function, aclrtStream stream,
                                      const HybmAggregateUrmaDemoParam &param, uint16_t deviceId)
{
    aclrtArgsHandle argsHandle = nullptr;
    auto ret = PrepareAggregateUrmaDemoArgs(function, param, argsHandle);
    if (ret != BM_OK) {
        return ret;
    }
    ret = aclrtLaunchKernelWithConfig(function, kKernelBlockDim, stream, nullptr, argsHandle, nullptr);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("launch aggregate URMA demo failed, deviceId: " << deviceId << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("synchronize aggregate URMA demo failed, deviceId: " << deviceId << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t LaunchAggregateUrmaDemo(const HybmAggregateUrmaDemoParam &param, uint16_t deviceId)
{
    try {
        c10_npu::OptionalNPUGuard npuGuard;
        npuGuard.set_index(deviceId);
        auto stream = c10_npu::getCurrentNPUStream(deviceId);
        aclrtStream npuStream = stream.stream(false);
        if (npuStream == nullptr) {
            OFFLOAD_LOG_ERROR("current NPU stream is null for aggregate URMA demo, deviceId: " << deviceId);
            return BM_DL_FUNCTION_FAILED;
        }
        aclrtFuncHandle function = nullptr;
        const auto ret = GetAggregateUrmaDemoFunction(deviceId, function);
        return ret == BM_OK ? LaunchAggregateUrmaDemoKernel(function, npuStream, param, deviceId) : ret;
    } catch (const std::exception &exception) {
        OFFLOAD_LOG_ERROR("aggregate URMA demo raised exception, deviceId: " << deviceId
                                                                             << " error: " << exception.what());
        return BM_ERROR;
    } catch (...) {
        OFFLOAD_LOG_ERROR("aggregate URMA demo raised unknown exception, deviceId: " << deviceId);
        return BM_ERROR;
    }
}

int32_t PrepareKvcacheScatterCopyArgs(aclrtFuncHandle function, HybmKvcacheScatterCopyParam param,
                                      aclrtArgsHandle &argsHandle)
{
    argsHandle = nullptr;
    auto ret = aclrtKernelArgsInit(function, &argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsInit failed for kvcache scatter_copy, batchSize: " << param.batchSize
                                                                                             << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    aclrtParamHandle paramHandle = nullptr;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(param), &paramHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsAppend failed for kvcache scatter_copy, batchSize: " << param.batchSize
                                                                                               << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtKernelArgsFinalize(argsHandle);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("aclrtKernelArgsFinalize failed for kvcache scatter_copy, batchSize: " << param.batchSize
                                                                                                 << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t LaunchKvcacheScatterCopyKernel(aclrtFuncHandle function, aclrtStream stream,
                                       const HybmKvcacheScatterCopyParam &param, uint16_t deviceId)
{
    aclrtArgsHandle argsHandle = nullptr;
    auto ret = PrepareKvcacheScatterCopyArgs(function, param, argsHandle);
    if (ret != BM_OK) {
        return ret;
    }
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kKernelTimeoutSeconds;
    aclrtLaunchKernelCfg config{&attr, 1U};
    ret = aclrtLaunchKernelWithConfig(function, kKernelBlockDim, stream, &config, argsHandle, nullptr);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("launch kvcache scatter_copy failed, deviceId: " << deviceId << " batchSize: "
                                                                           << param.batchSize << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OFFLOAD_LOG_ERROR("synchronize kvcache scatter_copy failed, deviceId: " << deviceId << " ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    return BM_OK;
}

int32_t LaunchKvcacheScatterCopy(const HybmKvcacheScatterCopyParam &param, uint16_t deviceId)
{
    try {
        c10_npu::OptionalNPUGuard npuGuard;
        npuGuard.set_index(deviceId);
        auto stream = c10_npu::getCurrentNPUStream(deviceId);
        aclrtStream npuStream = stream.stream(false);
        if (npuStream == nullptr) {
            OFFLOAD_LOG_ERROR("current NPU stream is null for kvcache scatter_copy, deviceId: " << deviceId);
            return BM_DL_FUNCTION_FAILED;
        }
        aclrtFuncHandle function = nullptr;
        const auto ret = GetKvcacheScatterCopyFunction(deviceId, function);
        return ret == BM_OK ? LaunchKvcacheScatterCopyKernel(function, npuStream, param, deviceId) : ret;
    } catch (const std::exception &exception) {
        OFFLOAD_LOG_ERROR("kvcache scatter_copy raised exception, deviceId: " << deviceId
                                                                              << " error: " << exception.what());
        return BM_ERROR;
    } catch (...) {
        OFFLOAD_LOG_ERROR("kvcache scatter_copy raised unknown exception, deviceId: " << deviceId);
        return BM_ERROR;
    }
}
} // namespace

extern "C" {
int32_t AccOffloadAggregateUrmaDemo(uint64_t message, uint64_t ready, uint64_t dstNew, uint64_t dstBase,
                                    uint64_t timing, uint16_t devIdx)
{
    HybmAggregateUrmaDemoParam param{};
    param.message = reinterpret_cast<const HybmAggregateUrmaDemoMessage *>(message);
    param.ready = reinterpret_cast<volatile uint64_t *>(ready);
    param.dstNew = reinterpret_cast<uint8_t *>(dstNew);
    param.dstBase = reinterpret_cast<uint8_t *>(dstBase);
    param.timing = reinterpret_cast<HybmAggregateUrmaDemoTiming *>(timing);
    return LaunchAggregateUrmaDemo(param, devIdx);
}

void AccOffloadSparseCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *sizePtr, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);

    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream]() -> int {
        OffloadOpsSparseCopy(srcPtrs, dstPtrs, lenPtrs, sizePtr, npuStream);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("acc_sparse_copy", callback);
}

void AccOffloadGroupPackCopy(uint64_t *srcPtrs, uint64_t *dstPtrs, uint32_t *lenPtrs, uint32_t *numLocalExpertPtr,
                             int64_t *groupList, int64_t *packedGroupList, uint8_t devIdx)
{
    c10_npu::OptionalNPUGuard npuGuard;
    npuGuard.set_index(devIdx);

    auto stream = c10_npu::getCurrentNPUStream(devIdx);
    void *npuStream = stream.stream(false);

    auto callback = [srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, npuStream]() -> int {
        OffloadOpsGroupPackCopy(srcPtrs, dstPtrs, lenPtrs, numLocalExpertPtr, groupList, packedGroupList, npuStream);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("acc_group_pack_copy", callback);
}

int32_t AccOffloadSparseCopyUrma(uint64_t srcPtrs, uint64_t dstPtrs, uint64_t lenPtrs, uint32_t listNum,
                                 uint16_t devIdx)
{
    if (srcPtrs == 0U || dstPtrs == 0U || lenPtrs == 0U || listNum == 0U) {
        OFFLOAD_LOG_ERROR("invalid sparse_copy_urma launch arguments, src: 0x"
                          << std::hex << srcPtrs << " dst: 0x" << dstPtrs << " len: 0x" << lenPtrs << std::dec
                          << " listNum: " << listNum << " deviceId: " << devIdx);
        return BM_INVALID_PARAM;
    }
    return LaunchSparseCopyUrma(srcPtrs, dstPtrs, lenPtrs, listNum, devIdx);
}

int32_t AccOffloadKvcacheScatterCopy(uint64_t hbmKpe, uint64_t hbmCkv, uint64_t hbmBlockTable, uint64_t dramBlockTable,
                                     uint64_t offloadSlots, uint64_t srcTokenIds, uint64_t dstSlots,
                                     uint64_t copyCounts, uint64_t readyFlag, uint64_t hbmBlockCount,
                                     uint64_t hbmMaxBlocks, uint64_t dramMaxBlocks, uint64_t dramBlockTableRows,
                                     uint64_t batchSize, int64_t layerId, uint16_t devIdx)
{
    HybmKvcacheScatterCopyParam param{};
    param.hbmKpe = reinterpret_cast<void *>(hbmKpe);
    param.hbmCkv = reinterpret_cast<void *>(hbmCkv);
    param.hbmBlockTable = reinterpret_cast<const int32_t *>(hbmBlockTable);
    param.dramBlockTable = reinterpret_cast<const uint64_t *>(dramBlockTable);
    param.offloadSlots = reinterpret_cast<const int32_t *>(offloadSlots);
    param.srcTokenIds = reinterpret_cast<const int32_t *>(srcTokenIds);
    param.dstSlots = reinterpret_cast<const int32_t *>(dstSlots);
    param.copyCounts = reinterpret_cast<const int32_t *>(copyCounts);
    param.readyFlag = reinterpret_cast<int32_t *>(readyFlag);
    param.hbmBlockCount = hbmBlockCount;
    param.hbmMaxBlocks = hbmMaxBlocks;
    param.dramMaxBlocks = dramMaxBlocks;
    param.dramBlockTableRows = dramBlockTableRows;
    param.batchSize = batchSize;
    param.layerId = layerId;
    return LaunchKvcacheScatterCopy(param, devIdx);
}
}
