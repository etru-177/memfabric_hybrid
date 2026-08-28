/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <dirent.h>

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <sys/stat.h>
#include <unistd.h>

#define private   public
#define protected public
#include "device/urma/device_urma_eid_reader.h"
#include "device/urma/device_urma_transport_manager.h"
#include "dl_hccl_api.h"
#include "dl_hcomm_api.h"
#include "hybm_stream_manager.h"
#include "hybm_va_manager.h"
#undef private
#undef protected

using namespace ock::mf;
using namespace ock::mf::transport::device;
using ock::mf::transport::HybmTransPrepareOptions;
using ock::mf::transport::REG_MR_FLAG_DRAM;
using ock::mf::transport::REG_MR_FLAG_ACL_DRAM;
using ock::mf::transport::REG_MR_FLAG_HBM;
using ock::mf::transport::TransportMemoryKey;
using ock::mf::transport::TransportMemoryRegion;
using ock::mf::transport::TransportOptions;
using ock::mf::transport::TransportPrivateData;
using ock::mf::transport::TransportRankPrepareInfo;

namespace {
const EndpointHandle MOCK_ENDPOINT = reinterpret_cast<EndpointHandle>(0xA501UL);
const HcommMemHandle MOCK_MEM_HANDLE = reinterpret_cast<HcommMemHandle>(0xA502UL);
const HcommMemHandle MOCK_FLAG_HANDLE = reinterpret_cast<HcommMemHandle>(0xA505UL);
const aclrtBinHandle MOCK_BIN_HANDLE = reinterpret_cast<aclrtBinHandle>(0xA506UL);
const aclrtFuncHandle MOCK_READ_FUNC = reinterpret_cast<aclrtFuncHandle>(0xA507UL);
const aclrtFuncHandle MOCK_WRITE_FUNC = reinterpret_cast<aclrtFuncHandle>(0xA508UL);
const HcommMemHandle MOCK_NOTIFY_HANDLE = reinterpret_cast<HcommMemHandle>(0xA509UL);
void *const MOCK_NOTIFY = reinterpret_cast<void *>(0xA50AUL);
void *const MOCK_STREAM = reinterpret_cast<void *>(0xA50BUL);
const aclrtArgsHandle MOCK_ARGS_HANDLE = reinterpret_cast<aclrtArgsHandle>(0xA50CUL);
const aclrtParamHandle MOCK_PARAM_HANDLE = reinterpret_cast<aclrtParamHandle>(0xA50DUL);
constexpr ChannelHandle MOCK_CHANNEL = 0xA503UL;
constexpr HcommThreadHandle MOCK_THREAD = 0xA504UL;
constexpr uint64_t MOCK_LOCAL_ADDR = 0x100000UL;
constexpr uint64_t MOCK_REMOTE_ADDR = 0x200000UL;
constexpr uint64_t MOCK_NOTIFY_ADDR = 0x300000UL;
constexpr uint64_t MOCK_SIZE = 0x1000UL;
constexpr uint64_t MOCK_MEM_TAG = 7UL;
constexpr uint32_t MOCK_HCOMM_DESC_LEN = 4U;
constexpr uint32_t MOCK_NOTIFY_ID = 11U;
constexpr uint32_t MOCK_NOTIFY_LEN = sizeof(int64_t);
uint32_t g_memExportCallCount = 0;
uint32_t g_memImportCallCount = 0;
uint32_t g_memUnregCallCount = 0;
uint32_t g_kernelLaunchCallCount = 0;

struct TestHybmOneSideOpParam {
    ock::mf::ThreadHandle thread;
    ock::mf::ChannelHandle channel;
    uint32_t listNum;
    void **dstBufAddrList;
    void **srcBufAddrList;
    uint64_t *lenList;
    uint64_t remoteFlagAddr;
    uint64_t localFlagAddr;
    uint32_t flagSize;
};

struct MockcppScope {
    ~MockcppScope()
    {
        GlobalMockObject::reset();
    }
};

struct EnvVarGuard {
    explicit EnvVarGuard(const char *name) : name_(name)
    {
        const char *oldValue = std::getenv(name_.c_str());
        if (oldValue != nullptr) {
            hadOldValue_ = true;
            oldValue_ = oldValue;
        }
    }

    ~EnvVarGuard()
    {
        if (hadOldValue_) {
            (void)setenv(name_.c_str(), oldValue_.c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    std::string name_;
    bool hadOldValue_{false};
    std::string oldValue_{};
};

struct DlHcommApiFnGuard {
    hcommEndpointCreateFunc oldEndpointCreate{DlHcommApi::gHcommEndpointCreate};
    hcommEndpointDestroyFunc oldEndpointDestroy{DlHcommApi::gHcommEndpointDestroy};
    hcommMemRegFunc oldMemReg{DlHcommApi::gHcommMemReg};
    hcommMemUnregFunc oldMemUnreg{DlHcommApi::gHcommMemUnreg};
    hcommMemExportFunc oldMemExport{DlHcommApi::gHcommMemExport};
    hcommMemImportFunc oldMemImport{DlHcommApi::gHcommMemImport};
    hcommMemUnimportFunc oldMemUnimport{DlHcommApi::gHcommMemUnimport};
    hcommChannelCreateFunc oldChannelCreate{DlHcommApi::gHcommChannelCreate};
    hcommChannelDestroyFunc oldChannelDestroy{DlHcommApi::gHcommChannelDestroy};
    hcommThreadAllocFunc oldThreadAlloc{DlHcommApi::gHcommThreadAlloc};
    hcommThreadFreeFunc oldThreadFree{DlHcommApi::gHcommThreadFree};
    hcommReadOnThreadFunc oldReadOnThread{DlHcommApi::gHcommReadOnThread};
    hcommWriteOnThreadFunc oldWriteOnThread{DlHcommApi::gHcommWriteOnThread};
    hcommChannelFenceOnThreadFunc oldChannelFenceOnThread{DlHcommApi::gHcommChannelFenceOnThread};
    hcommChannelGetStatusFunc oldChannelGetStatus{DlHcommApi::gHcommChannelGetStatus};

    ~DlHcommApiFnGuard()
    {
        DlHcommApi::gHcommEndpointCreate = oldEndpointCreate;
        DlHcommApi::gHcommEndpointDestroy = oldEndpointDestroy;
        DlHcommApi::gHcommMemReg = oldMemReg;
        DlHcommApi::gHcommMemUnreg = oldMemUnreg;
        DlHcommApi::gHcommMemExport = oldMemExport;
        DlHcommApi::gHcommMemImport = oldMemImport;
        DlHcommApi::gHcommMemUnimport = oldMemUnimport;
        DlHcommApi::gHcommChannelCreate = oldChannelCreate;
        DlHcommApi::gHcommChannelDestroy = oldChannelDestroy;
        DlHcommApi::gHcommThreadAlloc = oldThreadAlloc;
        DlHcommApi::gHcommThreadFree = oldThreadFree;
        DlHcommApi::gHcommReadOnThread = oldReadOnThread;
        DlHcommApi::gHcommWriteOnThread = oldWriteOnThread;
        DlHcommApi::gHcommChannelFenceOnThread = oldChannelFenceOnThread;
        DlHcommApi::gHcommChannelGetStatus = oldChannelGetStatus;
    }
};

struct DlAclApiFnGuard {
    aclrtGetDeviceFunc oldGetDevice{DlAclApi::pAclrtGetDevice};
    aclrtCreateNotifyFunc oldCreateNotify{DlAclApi::pAclrtCreateNotify};
    aclrtGetNotifyIdFunc oldGetNotifyId{DlAclApi::pAclrtGetNotifyId};
    aclrtDestroyNotifyFunc oldDestroyNotify{DlAclApi::pAclrtDestroyNotify};
    aclrtWaitAndResetNotifyFunc oldWaitAndResetNotify{DlAclApi::pAclrtWaitAndResetNotify};
    aclrtSynchronizeStreamFunc oldSynchronizeStream{DlAclApi::pAclrtSynchronizeStream};
    aclrtBinaryLoadFromFileFunc oldBinaryLoadFromFile{DlAclApi::pAclrtBinaryLoadFromFile};
    aclrtBinaryGetFunctionFunc oldBinaryGetFunction{DlAclApi::pAclrtBinaryGetFunction};
    aclrtKernelArgsInitFunc oldKernelArgsInit{DlAclApi::pAclrtKernelArgsInit};
    aclrtKernelArgsAppendFunc oldKernelArgsAppend{DlAclApi::pAclrtKernelArgsAppend};
    aclrtKernelArgsFinalizeFunc oldKernelArgsFinalize{DlAclApi::pAclrtKernelArgsFinalize};
    aclrtLaunchKernelWithConfigFunc oldLaunchKernelWithConfig{DlAclApi::pAclrtLaunchKernelWithConfig};
    aclrtMallocFunc oldMalloc{DlAclApi::pAclrtMalloc};
    aclrtFreeFunc oldFree{DlAclApi::pAclrtFree};
    aclrtMemcpyFunc oldMemcpy{DlAclApi::pAclrtMemcpy};
    rtGetDeviceInfoFunc oldRtGetDeviceInfo{DlAclApi::pRtGetDeviceInfo};
    aclrtGetPhyDevIdByLogicDevIdFunc oldGetPhyDevIdByLogicDevId{DlAclApi::pAclrtGetPhyDevIdByLogicDevId};

    ~DlAclApiFnGuard()
    {
        DlAclApi::pAclrtGetDevice = oldGetDevice;
        DlAclApi::pAclrtCreateNotify = oldCreateNotify;
        DlAclApi::pAclrtGetNotifyId = oldGetNotifyId;
        DlAclApi::pAclrtDestroyNotify = oldDestroyNotify;
        DlAclApi::pAclrtWaitAndResetNotify = oldWaitAndResetNotify;
        DlAclApi::pAclrtSynchronizeStream = oldSynchronizeStream;
        DlAclApi::pAclrtBinaryLoadFromFile = oldBinaryLoadFromFile;
        DlAclApi::pAclrtBinaryGetFunction = oldBinaryGetFunction;
        DlAclApi::pAclrtKernelArgsInit = oldKernelArgsInit;
        DlAclApi::pAclrtKernelArgsAppend = oldKernelArgsAppend;
        DlAclApi::pAclrtKernelArgsFinalize = oldKernelArgsFinalize;
        DlAclApi::pAclrtLaunchKernelWithConfig = oldLaunchKernelWithConfig;
        DlAclApi::pAclrtMalloc = oldMalloc;
        DlAclApi::pAclrtFree = oldFree;
        DlAclApi::pAclrtMemcpy = oldMemcpy;
        DlAclApi::pRtGetDeviceInfo = oldRtGetDeviceInfo;
        DlAclApi::pAclrtGetPhyDevIdByLogicDevId = oldGetPhyDevIdByLogicDevId;
    }
};

struct DlRtApiFnGuard {
    rtGetDevResAddressFunc oldGetDevResAddress{DlRtApi::pRtGetDevResAddress};

    ~DlRtApiFnGuard()
    {
        DlRtApi::pRtGetDevResAddress = oldGetDevResAddress;
    }
};

UrmaEndpointDesc MakeEndpointDesc()
{
    UrmaEndpointDesc desc{};
    desc.devPhyId = 2UL;
    desc.superDevId = 2UL;
    desc.serverIdx = 3UL;
    desc.superPodIdx = 4UL;
    desc.protocol = UrmaProtocol::UBC_TP;
    desc.type = COMM_ADDR_TYPE_EID;
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        desc.raws[i] = static_cast<uint8_t>(i + 1);
    }
    return desc;
}

int32_t MockHcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    EXPECT_NE(endpoint, nullptr);
    EXPECT_NE(endpointHandle, nullptr);
    EXPECT_EQ(endpoint->protocol, COMM_PROTOCOL_UBC_TP);
    EXPECT_EQ(endpoint->commAddr.type, COMM_ADDR_TYPE_EID);
    EXPECT_EQ(endpoint->loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(endpoint->loc.device.devPhyId, 2U);
    EXPECT_EQ(endpoint->loc.device.superDevId, 2U);
    EXPECT_EQ(endpoint->loc.device.serverIdx, 3U);
    EXPECT_EQ(endpoint->loc.device.superPodIdx, 4U);
    *endpointHandle = MOCK_ENDPOINT;
    return BM_OK;
}

int32_t MockHcommEndpointCreateOpenDevice(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    EXPECT_NE(endpoint, nullptr);
    EXPECT_NE(endpointHandle, nullptr);
    EXPECT_EQ(endpoint->protocol, COMM_PROTOCOL_UBC_CTP);
    EXPECT_EQ(endpoint->commAddr.type, COMM_ADDR_TYPE_IP_V6);
    EXPECT_EQ(endpoint->loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(endpoint->loc.device.devPhyId, 2U);
    EXPECT_EQ(endpoint->loc.device.superDevId, 3U);
    EXPECT_EQ(endpoint->loc.device.serverIdx, 4U);
    EXPECT_EQ(endpoint->loc.device.superPodIdx, 5U);
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        EXPECT_EQ(endpoint->commAddr.raws[i], static_cast<uint8_t>(0xE0U + i));
    }
    *endpointHandle = MOCK_ENDPOINT;
    return BM_OK;
}

int32_t MockHcommEndpointCreateOpenDeviceUboe(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    EXPECT_NE(endpoint, nullptr);
    EXPECT_NE(endpointHandle, nullptr);
    EXPECT_EQ(endpoint->protocol, COMM_PROTOCOL_UBOE);
    EXPECT_EQ(endpoint->commAddr.type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(endpoint->commAddr.raws[0], 10U);
    EXPECT_EQ(endpoint->commAddr.raws[1], 10U);
    EXPECT_EQ(endpoint->commAddr.raws[2], 21U);
    EXPECT_EQ(endpoint->commAddr.raws[3], 2U);
    EXPECT_EQ(endpoint->loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(endpoint->loc.device.devPhyId, 2U);
    *endpointHandle = MOCK_ENDPOINT;
    return BM_OK;
}

int32_t MockHcommEndpointDestroy(EndpointHandle endpoint)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    return BM_OK;
}

int32_t MockHcommMemReg(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem, HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_STREQ(memTag, "7");
    EXPECT_NE(mem, nullptr);
    EXPECT_EQ(mem->type, COMM_MEM_TYPE_HOST);
    EXPECT_EQ(mem->addr, reinterpret_cast<void *>(MOCK_LOCAL_ADDR));
    EXPECT_EQ(mem->size, MOCK_SIZE);
    EXPECT_NE(memHandle, nullptr);
    *memHandle = MOCK_MEM_HANDLE;
    return BM_OK;
}

int32_t MockHcommMemUnreg(EndpointHandle endpoint, HcommMemHandle memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_EQ(memHandle, MOCK_MEM_HANDLE);
    return BM_OK;
}

int32_t MockHcommMemUnregCounted(EndpointHandle endpoint, HcommMemHandle memHandle)
{
    g_memUnregCallCount++;
    return MockHcommMemUnreg(endpoint, memHandle);
}

int32_t MockHcommMemExport(EndpointHandle endpoint, HcommMemHandle memHandle, void **memDesc, uint32_t *memDescLen)
{
    static uint8_t desc[] = {0xA5, 0x01, 0x02, 0x03};
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_EQ(memHandle, MOCK_MEM_HANDLE);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_NE(memDescLen, nullptr);
    *memDesc = desc;
    *memDescLen = sizeof(desc);
    return BM_OK;
}

int32_t MockHcommMemExportCounted(EndpointHandle endpoint, HcommMemHandle memHandle, void **memDesc,
                                  uint32_t *memDescLen)
{
    g_memExportCallCount++;
    return MockHcommMemExport(endpoint, memHandle, memDesc, memDescLen);
}

int32_t MockHcommMemRegAny(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem,
                           HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memTag, nullptr);
    EXPECT_NE(mem, nullptr);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_NE(mem->addr, nullptr);
    EXPECT_GT(mem->size, 0U);
    *memHandle = reinterpret_cast<HcommMemHandle>(reinterpret_cast<uintptr_t>(mem->addr) + 0x10U);
    return BM_OK;
}

int32_t MockHcommMemRegFailAny(EndpointHandle, const char *, const HcommCommMem *, HcommMemHandle *)
{
    return BM_ERROR;
}

int32_t MockHcommMemUnregAny(EndpointHandle endpoint, HcommMemHandle memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memHandle, nullptr);
    return BM_OK;
}

int32_t MockHcommMemRegOpenDevice(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem,
                                  HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memTag, nullptr);
    EXPECT_NE(mem, nullptr);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_EQ(mem->type, COMM_MEM_TYPE_DEVICE);
    if (mem->addr == reinterpret_cast<void *>(MOCK_NOTIFY_ADDR)) {
        EXPECT_EQ(mem->size, MOCK_NOTIFY_LEN);
        *memHandle = MOCK_NOTIFY_HANDLE;
    } else {
        EXPECT_EQ(mem->size, sizeof(int64_t));
        *memHandle = MOCK_FLAG_HANDLE;
    }
    return BM_OK;
}

int32_t MockHcommMemUnregOpenDevice(EndpointHandle endpoint, HcommMemHandle memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_TRUE(memHandle == MOCK_NOTIFY_HANDLE || memHandle == MOCK_FLAG_HANDLE || memHandle == MOCK_MEM_HANDLE);
    return BM_OK;
}

int32_t MockHcommMemExportAny(EndpointHandle endpoint, HcommMemHandle memHandle, void **memDesc, uint32_t *memDescLen)
{
    static uint8_t desc[] = {0xA5, 0x01, 0x02, 0x03};
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_NE(memDescLen, nullptr);
    *memDesc = desc;
    *memDescLen = sizeof(desc);
    return BM_OK;
}

int32_t MockHcommMemExportSecondFails(EndpointHandle endpoint, HcommMemHandle memHandle, void **memDesc,
                                      uint32_t *memDescLen)
{
    g_memExportCallCount++;
    if (g_memExportCallCount == 1U) {
        return MockHcommMemExportAny(endpoint, memHandle, memDesc, memDescLen);
    }
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_NE(memDescLen, nullptr);
    *memDesc = nullptr;
    *memDescLen = 0;
    return BM_ERROR;
}

int32_t MockHcommMemImport(EndpointHandle endpoint, const void *memDesc, uint32_t descLen, HcommCommMem *commMem)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_EQ(descLen, MOCK_HCOMM_DESC_LEN);
    EXPECT_NE(commMem, nullptr);
    commMem->type = COMM_MEM_TYPE_HOST;
    commMem->addr = reinterpret_cast<void *>(MOCK_REMOTE_ADDR);
    commMem->size = MOCK_SIZE;
    return BM_OK;
}

int32_t MockHcommMemImportInvalidType(EndpointHandle endpoint, const void *memDesc, uint32_t descLen,
                                      HcommCommMem *commMem)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_EQ(descLen, MOCK_HCOMM_DESC_LEN);
    EXPECT_NE(commMem, nullptr);
    commMem->type = COMM_MEM_TYPE_INVALID;
    commMem->addr = reinterpret_cast<void *>(MOCK_REMOTE_ADDR);
    commMem->size = MOCK_SIZE;
    return BM_OK;
}

int32_t MockHcommMemImportFail(EndpointHandle, const void *, uint32_t, HcommCommMem *)
{
    return BM_ERROR;
}

int32_t MockHcommMemImportSecondFails(EndpointHandle endpoint, const void *memDesc, uint32_t descLen,
                                      HcommCommMem *commMem)
{
    g_memImportCallCount++;
    if (g_memImportCallCount == 1U) {
        return MockHcommMemImport(endpoint, memDesc, descLen, commMem);
    }
    return BM_ERROR;
}

int32_t MockHcommMemImportSecondInvalidType(EndpointHandle endpoint, const void *memDesc, uint32_t descLen,
                                            HcommCommMem *commMem)
{
    g_memImportCallCount++;
    if (g_memImportCallCount == 1U) {
        return MockHcommMemImport(endpoint, memDesc, descLen, commMem);
    }
    return MockHcommMemImportInvalidType(endpoint, memDesc, descLen, commMem);
}

int32_t MockHcommMemUnimport(EndpointHandle endpoint, const void *memDesc, uint32_t descLen)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_EQ(descLen, MOCK_HCOMM_DESC_LEN);
    return BM_OK;
}

std::vector<uint8_t> MakeRawExportDesc(uint64_t remoteAddr = MOCK_REMOTE_ADDR, uint64_t size = MOCK_SIZE,
                                       uint64_t memTag = MOCK_MEM_TAG)
{
    UrmaExportDesc exportDesc{};
    exportDesc.headerSize = sizeof(UrmaExportDesc);
    exportDesc.memoryType = UrmaMemoryType::HOST_DRAM;
    exportDesc.addr = remoteAddr;
    exportDesc.size = size;
    exportDesc.memTag = memTag;
    exportDesc.hcommDescLen = MOCK_HCOMM_DESC_LEN;
    std::vector<uint8_t> bytes(sizeof(UrmaExportDesc) + MOCK_HCOMM_DESC_LEN, 0x5A);
    std::memcpy(bytes.data(), &exportDesc, sizeof(exportDesc));
    return bytes;
}

int32_t MockHcommChannelCreate(EndpointHandle endpoint, CommEngine engine, HcommChannelDesc *channelDescs,
                               uint32_t channelNum, ChannelHandle *channels)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_EQ(engine, COMM_ENGINE_AICPU);
    EXPECT_NE(channelDescs, nullptr);
    EXPECT_EQ(channelNum, 1U);
    EXPECT_NE(channels, nullptr);
    EXPECT_EQ(channelDescs->header.version, HCOMM_CHANNEL_VERSION);
    EXPECT_EQ(channelDescs->header.magicWord, HCOMM_CHANNEL_MAGIC_WORD);
    EXPECT_EQ(channelDescs->header.size, sizeof(HcommChannelDesc));
    EXPECT_EQ(channelDescs->channelName, nullptr);
    EXPECT_TRUE(channelDescs->exchangeAllMems);
    EXPECT_TRUE(channelDescs->remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP ||
                channelDescs->remoteEndpoint.protocol == COMM_PROTOCOL_UBC_CTP);
    *channels = MOCK_CHANNEL;
    return BM_OK;
}

int32_t MockHcommChannelCreateFail(EndpointHandle, CommEngine, HcommChannelDesc *, uint32_t, ChannelHandle *)
{
    return BM_ERROR;
}

int32_t MockHcommChannelCreateZero(EndpointHandle endpoint, CommEngine engine, HcommChannelDesc *channelDescs,
                                   uint32_t channelNum, ChannelHandle *channels)
{
    const auto ret = MockHcommChannelCreate(endpoint, engine, channelDescs, channelNum, channels);
    *channels = 0;
    return ret;
}

// 计数 mock，用于验证轮询/回滚行为
static int g_getStatusCallCount = 0;
static int g_channelDestroyCallCount = 0;
static int g_threadFreeCallCount = 0;

int32_t MockHcommChannelGetStatusReady(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    *statusList = 0UL; // READY
    return BM_OK;
}

int32_t MockHcommChannelGetStatusInProgressThenReady(const ChannelHandle *channelList, uint32_t listNum,
                                                     int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    g_getStatusCallCount++;
    if (g_getStatusCallCount <= 2UL) {
        *statusList = 1UL; // IN_PROGRESS
    } else {
        *statusList = 0UL; // READY
    }
    return BM_OK;
}

int32_t MockHcommChannelGetStatusApiFail(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    return BM_ERROR;
}

int32_t MockHcommChannelGetStatusFailed(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    *statusList = 2UL; // FAILED
    return BM_OK;
}

int32_t MockHcommChannelGetStatusTimeout(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    *statusList = 3UL; // TIMEOUT
    return BM_OK;
}

int32_t MockHcommChannelGetStatusUnknown(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList)
{
    EXPECT_NE(channelList, nullptr);
    EXPECT_EQ(listNum, 1U);
    EXPECT_EQ(channelList[0], MOCK_CHANNEL);
    EXPECT_NE(statusList, nullptr);
    *statusList = 99UL; // 未知状态
    return BM_OK;
}

// 前向声明（原 mock 定义在后）
int32_t MockHcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum);
int32_t MockHcommThreadFree(const ThreadHandle *threads, uint32_t threadNum);

int32_t MockHcommChannelDestroyCounted(const ChannelHandle *channels, uint32_t channelNum)
{
    g_channelDestroyCallCount++;
    return MockHcommChannelDestroy(channels, channelNum);
}

int32_t MockHcommThreadFreeCounted(const ThreadHandle *threads, uint32_t threadNum)
{
    g_threadFreeCallCount++;
    return MockHcommThreadFree(threads, threadNum);
}

TransportPrivateData MakePrivateData(const UrmaEndpointDesc &desc)
{
    TransportPrivateData data{};
    struct Header {
        uint32_t magic;
        uint16_t version;
        uint16_t payloadLen;
    } header{0xA5FAC003U, 1U, static_cast<uint16_t>(sizeof(UrmaEndpointDesc))};
    std::memcpy(data.key.keys, &header, sizeof(header));
    std::memcpy(reinterpret_cast<uint8_t *>(data.key.keys) + sizeof(header), &desc, sizeof(desc));
    return data;
}

TransportMemoryKey MakeImportKey(uint64_t remoteAddr, uint64_t size, uint64_t memTag)
{
    TransportMemoryKey key{};
    key.keys[0] = URMA_EXPORT_DESC_MAGIC;
    key.keys[1] = remoteAddr;
    UrmaExportDesc exportDesc{};
    exportDesc.headerSize = sizeof(UrmaExportDesc);
    exportDesc.memoryType = UrmaMemoryType::HOST_DRAM;
    exportDesc.addr = remoteAddr;
    exportDesc.size = size;
    exportDesc.memTag = memTag;
    exportDesc.hcommDescLen = MOCK_HCOMM_DESC_LEN;
    constexpr uint32_t kHeaderSlots = DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS;
    auto *payload = reinterpret_cast<uint8_t *>(&key.keys[kHeaderSlots]);
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    std::memset(payload + sizeof(exportDesc), 0, MOCK_HCOMM_DESC_LEN);
    static_assert(sizeof(UrmaExportDesc) + MOCK_HCOMM_DESC_LEN <= sizeof(key.keys) - kHeaderSlots * sizeof(uint64_t));
    return key;
}

TransportMemoryKey MakeImportKeyWithFlag(uint64_t remoteAddr, uint64_t size, uint64_t memTag)
{
    auto key = MakeImportKey(remoteAddr, size, memTag);
    auto *payload = reinterpret_cast<uint8_t *>(&key.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    UrmaExportDesc exportDesc{};
    std::memcpy(&exportDesc, payload, sizeof(exportDesc));
    exportDesc.devTransFlagDescLen = MOCK_HCOMM_DESC_LEN;
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    std::memset(payload + sizeof(exportDesc) + exportDesc.hcommDescLen, 0xF1, MOCK_HCOMM_DESC_LEN);
    constexpr uint32_t kHeaderSlots = DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS;
    static_assert(sizeof(UrmaExportDesc) + MOCK_HCOMM_DESC_LEN * 2 <=
                  sizeof(key.keys) - kHeaderSlots * sizeof(uint64_t));
    return key;
}

int32_t MockHcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum)
{
    EXPECT_NE(channels, nullptr);
    EXPECT_EQ(channelNum, 1U);
    EXPECT_EQ(channels[0], MOCK_CHANNEL);
    return BM_OK;
}

int32_t MockHcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread,
                             ThreadHandle *threads)
{
    EXPECT_EQ(engine, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(threadNum, 1U);
    EXPECT_NE(notifyNumPerThread, nullptr);
    EXPECT_NE(threads, nullptr);
    *threads = MOCK_THREAD;
    return BM_OK;
}

int32_t MockHcommThreadAllocFail(CommEngine, uint32_t, const uint32_t *, ThreadHandle *)
{
    return BM_ERROR;
}

int32_t MockHcommThreadFree(const ThreadHandle *threads, uint32_t threadNum)
{
    EXPECT_NE(threads, nullptr);
    EXPECT_EQ(threadNum, 1U);
    EXPECT_EQ(threads[0], MOCK_THREAD);
    return BM_OK;
}

int32_t MockHcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    EXPECT_EQ(thread, MOCK_THREAD);
    EXPECT_EQ(channel, MOCK_CHANNEL);
    EXPECT_EQ(dst, reinterpret_cast<void *>(MOCK_LOCAL_ADDR));
    EXPECT_EQ(src, reinterpret_cast<const void *>(MOCK_REMOTE_ADDR));
    EXPECT_EQ(len, MOCK_SIZE);
    return BM_OK;
}

int32_t MockHcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    EXPECT_EQ(thread, MOCK_THREAD);
    EXPECT_EQ(channel, MOCK_CHANNEL);
    EXPECT_EQ(dst, reinterpret_cast<void *>(MOCK_REMOTE_ADDR));
    EXPECT_EQ(src, reinterpret_cast<const void *>(MOCK_LOCAL_ADDR));
    EXPECT_EQ(len, MOCK_SIZE);
    return BM_OK;
}

int32_t MockHcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
{
    EXPECT_EQ(thread, MOCK_THREAD);
    EXPECT_EQ(channel, MOCK_CHANNEL);
    return BM_OK;
}

int32_t MockHcommEndpointCreateFail(const EndpointDesc *, EndpointHandle *)
{
    return BM_DL_FUNCTION_FAILED;
}

void MakeDirectories(const std::string &dir)
{
    if (dir.empty()) {
        return;
    }
    size_t pos = (dir.front() == '/') ? 1U : 0U;
    while ((pos = dir.find('/', pos)) != std::string::npos) {
        (void)mkdir(dir.substr(0, pos).c_str(), 0755);
        ++pos;
    }
    (void)mkdir(dir.c_str(), 0755);
}

std::string PrepareKernelJson()
{
    const std::string base = std::string(testing::TempDir()) + "device_urma_kernel_" + std::to_string(getpid());
    const std::string dir = base + "/opp/vendors/cust/op_impl/aicpu/config";
    MakeDirectories(dir);
    const std::string path = dir + "/libcann_hybm_kernel.json";
    std::ofstream json(path);
    json << "{}";
    json.close();
    (void)setenv("ASCEND_HOME_PATH", base.c_str(), 1);
    return base;
}

ock::mf::Result MockGetDeviceUrmaEid(uint32_t phyDeviceId, uint32_t rankId,
                                     std::array<uint8_t, COMM_ADDR_EID_LEN> &eidData)
{
    EXPECT_EQ(phyDeviceId, 2U);
    EXPECT_EQ(rankId, 0U);
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        eidData[i] = static_cast<uint8_t>(0xE0U + i);
    }
    return BM_OK;
}

ock::mf::Result MockGetDeviceUrmaIpAddr(uint32_t phyDeviceId, uint32_t rankId, CommAddrType &addrType,
                                        std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> &addrData)
{
    EXPECT_EQ(phyDeviceId, 2U);
    EXPECT_EQ(rankId, 0U);
    addrType = COMM_ADDR_TYPE_IP_V4;
    addrData.fill(0);
    addrData[0] = 10U;
    addrData[1U] = 10U;
    addrData[2U] = 21U;
    addrData[3U] = 2U;
    return BM_OK;
}

int32_t MockAclrtGetDevice(int32_t *deviceId)
{
    EXPECT_NE(deviceId, nullptr);
    *deviceId = 0;
    return BM_OK;
}

int32_t MockAclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t *const phyDevId)
{
    EXPECT_EQ(logicDevId, 0);
    EXPECT_NE(phyDevId, nullptr);
    *phyDevId = 2UL;
    return BM_OK;
}

int32_t MockRtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val)
{
    EXPECT_EQ(deviceId, 0U);
    EXPECT_EQ(moduleType, 0);
    EXPECT_NE(val, nullptr);
    if (infoType == INFO_TYPE_SDID) {
        *val = 3ULL;
    } else if (infoType == INFO_TYPE_SERVER_ID) {
        *val = 4ULL;
    } else if (infoType == INFO_TYPE_SUPER_POD_ID) {
        *val = 5ULL;
    } else {
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

int32_t MockAclrtCreateNotify(void **notify, uint64_t flag)
{
    EXPECT_NE(notify, nullptr);
    EXPECT_EQ(flag, 1U);
    *notify = MOCK_NOTIFY;
    return BM_OK;
}

int32_t MockAclrtGetNotifyId(void *notify, uint32_t *notifyId)
{
    EXPECT_EQ(notify, MOCK_NOTIFY);
    EXPECT_NE(notifyId, nullptr);
    *notifyId = MOCK_NOTIFY_ID;
    return BM_OK;
}

int32_t MockAclrtDestroyNotify(void *notify)
{
    EXPECT_EQ(notify, MOCK_NOTIFY);
    return BM_OK;
}

int32_t MockRtGetDevResAddress(rtDevResInfo *resInfo, rtDevResAddrInfo *addrInfo)
{
    EXPECT_NE(resInfo, nullptr);
    EXPECT_NE(addrInfo, nullptr);
    EXPECT_EQ(resInfo->resType, RT_RES_TYPE_STARS_NOTIFY_RECORD);
    EXPECT_EQ(resInfo->resId, MOCK_NOTIFY_ID);
    EXPECT_NE(addrInfo->resAddress, nullptr);
    EXPECT_NE(addrInfo->len, nullptr);
    *addrInfo->resAddress = MOCK_NOTIFY_ADDR;
    *addrInfo->len = MOCK_NOTIFY_LEN;
    return BM_OK;
}

int32_t MockAclrtMallocOk(void **ptr, size_t count, uint32_t)
{
    EXPECT_NE(ptr, nullptr);
    EXPECT_GT(count, 0U);
    *ptr = std::malloc(count);
    return (*ptr == nullptr) ? BM_ERROR : BM_OK;
}

int32_t MockAclrtFreeOk(void *ptr)
{
    std::free(ptr);
    return BM_OK;
}

int32_t MockAclrtMemcpyOk(void *dst, size_t destMax, const void *src, size_t count, uint32_t)
{
    EXPECT_NE(dst, nullptr);
    EXPECT_NE(src, nullptr);
    EXPECT_LE(count, destMax);
    std::memcpy(dst, src, count);
    return BM_OK;
}

int32_t MockAclrtMemcpyFail(void *, size_t, const void *, size_t, uint32_t)
{
    return BM_ERROR;
}

int32_t MockAclrtBinaryLoadFromFile(const char *binPath, aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    EXPECT_NE(binPath, nullptr);
    EXPECT_NE(options, nullptr);
    EXPECT_NE(binHandle, nullptr);
    EXPECT_EQ(options->numOpt, 1U);
    *binHandle = MOCK_BIN_HANDLE;
    return BM_OK;
}

int32_t MockAclrtBinaryLoadFromFileFail(const char *binPath, aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    EXPECT_NE(binPath, nullptr);
    EXPECT_NE(options, nullptr);
    EXPECT_NE(binHandle, nullptr);
    return BM_ERROR;
}

int32_t MockAclrtBinaryGetFunction(aclrtBinHandle binHandle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    EXPECT_EQ(binHandle, MOCK_BIN_HANDLE);
    EXPECT_NE(kernelName, nullptr);
    EXPECT_NE(funcHandle, nullptr);
    const std::string name(kernelName);
    *funcHandle = (name == "read_kernel" || name == "HybmBatchRead") ? MOCK_READ_FUNC : MOCK_WRITE_FUNC;
    return BM_OK;
}

int32_t MockAclrtBinaryGetFunctionFail(aclrtBinHandle, const char *, aclrtFuncHandle *)
{
    return BM_ERROR;
}

int32_t MockAclrtBinaryGetFunctionNull(aclrtBinHandle, const char *, aclrtFuncHandle *funcHandle)
{
    EXPECT_NE(funcHandle, nullptr);
    *funcHandle = nullptr;
    return BM_OK;
}

int32_t MockAclrtMallocFail(void **ptr, size_t count, uint32_t)
{
    EXPECT_NE(ptr, nullptr);
    EXPECT_GT(count, 0U);
    *ptr = nullptr;
    return BM_ERROR;
}

int32_t MockAclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    EXPECT_TRUE(funcHandle == MOCK_READ_FUNC || funcHandle == MOCK_WRITE_FUNC);
    EXPECT_NE(argsHandle, nullptr);
    *argsHandle = MOCK_ARGS_HANDLE;
    return BM_OK;
}

int32_t MockAclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
                                  aclrtParamHandle *paramHandle)
{
    EXPECT_EQ(argsHandle, MOCK_ARGS_HANDLE);
    EXPECT_NE(param, nullptr);
    EXPECT_EQ(paramSize, sizeof(TestHybmOneSideOpParam));
    EXPECT_NE(paramHandle, nullptr);
    const auto *args = static_cast<const TestHybmOneSideOpParam *>(param);
    EXPECT_EQ(args->thread, MOCK_THREAD);
    EXPECT_EQ(args->channel, MOCK_CHANNEL);
    if (args->listNum == 0) {
        // Marker-only / notify kernel launch: no data, flag fields valid
        EXPECT_EQ(args->dstBufAddrList, nullptr);
        EXPECT_EQ(args->srcBufAddrList, nullptr);
        EXPECT_EQ(args->lenList, nullptr);
        EXPECT_NE(args->remoteFlagAddr, 0U);
        EXPECT_NE(args->localFlagAddr, 0U);
        EXPECT_NE(args->flagSize, 0U);
    } else {
        // Data kernel launch: list pointers valid, flag fields are 0 (production
        // LaunchDeviceKernelBatch zero-initializes HybmOneSideOpParam and does not
        // set flag fields; LaunchDeviceKernelNotify sets them for marker-only).
        EXPECT_NE(args->dstBufAddrList, nullptr);
        EXPECT_NE(args->srcBufAddrList, nullptr);
        EXPECT_NE(args->lenList, nullptr);
        EXPECT_EQ(args->localFlagAddr, 0U);
        EXPECT_EQ(args->flagSize, 0U);
    }
    *paramHandle = MOCK_PARAM_HANDLE;
    return BM_OK;
}

int32_t MockAclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    EXPECT_EQ(argsHandle, MOCK_ARGS_HANDLE);
    return BM_OK;
}

int32_t MockAclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, void *stream,
                                        aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserved)
{
    EXPECT_TRUE(funcHandle == MOCK_READ_FUNC || funcHandle == MOCK_WRITE_FUNC);
    EXPECT_EQ(blockDim, 1U);
    EXPECT_EQ(stream, MOCK_STREAM);
    EXPECT_NE(cfg, nullptr);
    EXPECT_EQ(cfg->numAttrs, 1U);
    EXPECT_EQ(argsHandle, MOCK_ARGS_HANDLE);
    EXPECT_EQ(reserved, nullptr);
    g_kernelLaunchCallCount++;
    return BM_OK;
}

int32_t MockAclrtWaitAndResetNotify(void *notify, void *stream, uint32_t timeout)
{
    EXPECT_EQ(notify, MOCK_NOTIFY);
    EXPECT_EQ(stream, MOCK_STREAM);
    EXPECT_EQ(timeout, 60U);
    return BM_OK;
}

int32_t MockAclrtSynchronizeStream(void *stream)
{
    EXPECT_EQ(stream, MOCK_STREAM);
    return BM_OK;
}

static int32_t g_syncCallCount = 0;
int32_t MockAclrtSynchronizeStreamFirstOkThenFail(void *stream)
{
    if (g_syncCallCount++ == 0) {
        return MockAclrtSynchronizeStream(stream);
    }
    return BM_ERROR;
}

void InstallOpenDeviceMocks()
{
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreateOpenDevice;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegOpenDevice;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregOpenDevice;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlAclApi::pAclrtGetDevice = MockAclrtGetDevice;
    DlAclApi::pAclrtGetPhyDevIdByLogicDevId = MockAclrtGetPhyDevIdByLogicDevId;
    DlAclApi::pRtGetDeviceInfo = MockRtGetDeviceInfo;
    DlAclApi::pAclrtCreateNotify = MockAclrtCreateNotify;
    DlAclApi::pAclrtGetNotifyId = MockAclrtGetNotifyId;
    DlAclApi::pAclrtDestroyNotify = MockAclrtDestroyNotify;
    DlRtApi::pRtGetDevResAddress = MockRtGetDevResAddress;
    DlAclApi::pAclrtMalloc = MockAclrtMallocOk;
    DlAclApi::pAclrtFree = MockAclrtFreeOk;
    DlAclApi::pAclrtMemcpy = MockAclrtMemcpyOk;
    DlAclApi::pAclrtBinaryLoadFromFile = MockAclrtBinaryLoadFromFile;
    DlAclApi::pAclrtBinaryGetFunction = MockAclrtBinaryGetFunction;
}

void InstallKernelLaunchMocks()
{
    DlAclApi::pAclrtKernelArgsInit = MockAclrtKernelArgsInit;
    DlAclApi::pAclrtKernelArgsAppend = MockAclrtKernelArgsAppend;
    DlAclApi::pAclrtKernelArgsFinalize = MockAclrtKernelArgsFinalize;
    DlAclApi::pAclrtLaunchKernelWithConfig = MockAclrtLaunchKernelWithConfig;
    DlAclApi::pAclrtWaitAndResetNotify = MockAclrtWaitAndResetNotify;
    DlAclApi::pAclrtSynchronizeStream = MockAclrtSynchronizeStream;
}
// Helper: bundles guards for tests that do OpenDevice+Prepare+kernel launch
struct DeviceTestFixture {
    EnvVarGuard envGuard;
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;

    DeviceTestFixture() : envGuard("ASCEND_HOME_PATH") {}

    void InstallAll()
    {
        PrepareKernelJson();
        InstallOpenDeviceMocks();
        InstallKernelLaunchMocks();
        DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
        DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
        DlHcommApi::gHcommMemImport = MockHcommMemImport;
        DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
        DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
        DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
        MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
        MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
        MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));
    }

    void OpenAndPreparePeer(DeviceUrmaTransportManager &manager) const
    {
        TransportOptions opts;
        opts.rankId = 0;
        opts.rankCount = 2UL;
        opts.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
        ASSERT_EQ(manager.OpenDevice(opts), BM_OK);
        HybmTransPrepareOptions prep;
        TransportRankPrepareInfo info;
        auto pd = MakeEndpointDesc();
        pd.protocol = UrmaProtocol::UBC_CTP;
        pd.type = COMM_ADDR_TYPE_IP_V6;
        info.privateData = MakePrivateData(pd);
        info.role = HYBM_ROLE_PEER;
        info.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
        prep.options.emplace(1, std::move(info));
        ASSERT_EQ(manager.Prepare(prep), BM_OK);
    }

    void AddLocalReg(DeviceUrmaTransportManager &manager, uint64_t addr, uint64_t size, uint32_t flags,
                     uint64_t dva) const
    {
        DeviceUrmaTransportManager::LocalRegistration reg;
        reg.mr.addr = addr;
        reg.mr.size = size;
        reg.mr.flags = flags;
        reg.handle = MOCK_MEM_HANDLE;
        reg.memTag = addr;
        reg.refCount = 1U;
        reg.deviceVa = dva;
        manager.localRegistrations_.emplace(addr, reg);
    }

    void CleanupAndClose(DeviceUrmaTransportManager &manager) const
    {
        manager.localRegistrations_.clear();
        EXPECT_EQ(manager.CloseDevice(), BM_OK);
    }
};

} // namespace

TEST(DeviceUrmaTransportManagerTest, GetPrivateDataEncodesLocalEndpointDesc)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();
    manager.opened_ = true;

    const auto data = manager.GetPrivateData();
    const auto *raw = reinterpret_cast<const uint8_t *>(data.key.keys);
    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t payloadLen = 0;
    std::memcpy(&magic, raw, sizeof(magic));
    std::memcpy(&version, raw + sizeof(magic), sizeof(version));
    std::memcpy(&payloadLen, raw + sizeof(magic) + sizeof(version), sizeof(payloadLen));
    EXPECT_EQ(magic, 0xA5FAC003U);
    EXPECT_EQ(version, 1U);
    EXPECT_EQ(payloadLen, sizeof(UrmaEndpointDesc));
    manager.opened_ = false;
}

TEST(DeviceUrmaTransportManagerTest, PrepareCreatesThreadChannelAndImportsMemKeys)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    const auto peerDesc = MakeEndpointDesc();
    auto privateData = MakePrivateData(peerDesc);
    auto key = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.nic = "tcp://127.0.0.1:8000";
    info.privateData = privateData;
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {key};
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_OK);
    auto &state = manager.remoteRanks_[1];
    EXPECT_EQ(state.thread, MOCK_THREAD);
    EXPECT_EQ(state.channel, MOCK_CHANNEL);
    ASSERT_EQ(state.imports.size(), 1U);
    EXPECT_EQ(state.imports.front().memTag, MOCK_MEM_TAG);
    EXPECT_EQ(state.imports.front().addr, MOCK_REMOTE_ADDR);
}

TEST(DeviceUrmaTransportManagerTest, OpenDeviceInitializesResourcesAndCloseCleansUp)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));

    DeviceUrmaTransportManager manager;
    TransportOptions options{};
    options.rankId = 0;
    options.rankCount = 2;
    options.protocol = HYBM_DOP_TYPE_DEVICE_URMA;

    EXPECT_EQ(manager.OpenDevice(options), BM_OK);
    EXPECT_TRUE(manager.opened_);
    EXPECT_EQ(manager.rankId_, 0U);
    EXPECT_EQ(manager.rankCount_, 2U);
    EXPECT_EQ(manager.phyDeviceId_, 2U);
    EXPECT_EQ(manager.sdid_, 3U);
    EXPECT_EQ(manager.serverId_, 4U);
    EXPECT_EQ(manager.superPodId_, 5U);
    EXPECT_NE(manager.localEndpoint_, nullptr);
    EXPECT_NE(manager.devTransFlagPtr_, nullptr);
    EXPECT_EQ(manager.devTransFlagSize_, sizeof(int64_t));
    EXPECT_EQ(manager.devTransFlagHcommHandle_, MOCK_FLAG_HANDLE);
    EXPECT_TRUE(manager.deviceKernelLoaded_);

    EXPECT_EQ(manager.OpenDevice(options), BM_OK);
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
    EXPECT_FALSE(manager.opened_);
    EXPECT_EQ(manager.localEndpoint_, nullptr);
    EXPECT_EQ(manager.devTransFlagPtr_, nullptr);
    EXPECT_EQ(manager.devTransFlagHcommHandle_, nullptr);
}

TEST(DeviceUrmaTransportManagerTest, OpenDeviceRollsBackWhenFlagMemcpyFails)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlAclApi::pAclrtMemcpy = MockAclrtMemcpyFail;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));

    DeviceUrmaTransportManager manager;
    TransportOptions options{};
    options.rankId = 0;
    options.rankCount = 2;
    options.protocol = HYBM_DOP_TYPE_DEVICE_URMA;

    EXPECT_EQ(manager.OpenDevice(options), BM_ERROR);
    EXPECT_FALSE(manager.opened_);
    EXPECT_EQ(manager.localEndpoint_, nullptr);
    EXPECT_EQ(manager.devTransFlagPtr_, nullptr);
}

TEST(DeviceUrmaTransportManagerTest, OpenDeviceRejectsInvalidOptionsAndUnsupportedSoc)
{
    DeviceUrmaTransportManager manager;
    TransportOptions options{};
    options.rankId = 1;
    options.rankCount = 1;
    EXPECT_EQ(manager.OpenDevice(options), BM_INVALID_PARAM);

    MockcppScope mockcpp;
    options.rankId = 0;
    options.rankCount = 2;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_UNKNOWN));
    EXPECT_EQ(manager.OpenDevice(options), BM_NOT_SUPPORTED);
}

// OpenDevice: data_op_type=DEVICE_UBOE 时应派生为 UrmaProtocol::UBOE 并走 IP 地址读取路径。
TEST(DeviceUrmaTransportManagerTest, OpenDeviceUboeDerivesProtocolFromOptions)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreateOpenDeviceUboe;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaIpAddr).stubs().will(invoke(MockGetDeviceUrmaIpAddr));

    DeviceUrmaTransportManager manager;
    TransportOptions options{};
    options.rankId = 0;
    options.rankCount = 2;
    options.protocol = HYBM_DOP_TYPE_DEVICE_UBOE;
    EXPECT_EQ(manager.OpenDevice(options), BM_OK);
    ASSERT_NE(manager.localEndpoint_, nullptr);
    EXPECT_EQ(manager.localEndpoint_->desc.protocol, UrmaProtocol::UBOE);
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

// OpenDevice: protocol 不含 DEVICE_URMA/DEVICE_UBOE 位时应返回 BM_INVALID_PARAM。
TEST(DeviceUrmaTransportManagerTest, OpenDeviceRejectsUnsupportedProtocolBits)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));

    DeviceUrmaTransportManager manager;
    TransportOptions options{};
    options.rankId = 0;
    options.rankCount = 2;
    options.protocol = 0; // 无 DEVICE_URMA/DEVICE_UBOE 位 -> GetEndpointProtocolFromOptions 返回 RESERVED
    EXPECT_EQ(manager.OpenDevice(options), BM_INVALID_PARAM);
    EXPECT_FALSE(manager.opened_);
    EXPECT_EQ(manager.localEndpoint_, nullptr);
}

TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchAndSynchronizeUseDeviceKernel)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    InstallKernelLaunchMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));

    g_kernelLaunchCallCount = 0;
    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);

    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    constexpr uint64_t kLocalDva = 0x124000000000ULL;
    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = 0x2000U;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = kLocalDva;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    EXPECT_EQ(manager.Synchronize(1), BM_OK);

    EXPECT_EQ(manager.WriteRemote(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);

    CopyDescriptor descriptor{};
    descriptor.localAddrs = {reinterpret_cast<void *>(MOCK_LOCAL_ADDR + 0x40U),
                             reinterpret_cast<void *>(MOCK_LOCAL_ADDR + 0x80U)};
    descriptor.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x40U),
                              reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x80U)};
    descriptor.counts = {0U, 0x20U};
    EXPECT_EQ(manager.ReadRemoteBatchAsync(1, descriptor), BM_OK);
    EXPECT_EQ(manager.Synchronize(1), BM_OK);

    EXPECT_GE(g_kernelLaunchCallCount, 3U);
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, UpdateRankOptionsAndRemoveRanksReusePreparedResources)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    HybmTransPrepareOptions updateOptions{};
    TransportRankPrepareInfo updateInfo{};
    updateInfo.privateData = MakePrivateData(MakeEndpointDesc());
    updateInfo.role = HYBM_ROLE_PEER;
    updateInfo.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR + MOCK_SIZE, MOCK_SIZE, MOCK_MEM_TAG + 1U)};
    updateOptions.options.emplace(1, std::move(updateInfo));
    EXPECT_EQ(manager.UpdateRankOptions(updateOptions), BM_OK);
    ASSERT_EQ(manager.remoteRanks_[1].imports.size(), 2U);

    EXPECT_EQ(manager.RemoveRanks({1}), BM_OK);
    EXPECT_TRUE(manager.remoteRanks_.empty());
}

TEST(DeviceUrmaTransportManagerTest, PrepareRollsBackNewResourcesWhenImportFails)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommMemImport = MockHcommMemImportFail;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_DL_FUNCTION_FAILED);
    ASSERT_TRUE(manager.remoteRanks_.count(1) != 0);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
    EXPECT_TRUE(manager.remoteRanks_[1].imports.empty());
}

TEST(DeviceUrmaTransportManagerTest, PrepareRejectsRankEdgesAndChangedEndpoint)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    HybmTransPrepareOptions invalidOptions{};
    TransportRankPrepareInfo invalidInfo{};
    invalidInfo.privateData = MakePrivateData(MakeEndpointDesc());
    invalidOptions.options.emplace(2, std::move(invalidInfo));
    EXPECT_EQ(manager.Prepare(invalidOptions), BM_INVALID_PARAM);

    HybmTransPrepareOptions selfOptions{};
    TransportRankPrepareInfo selfInfo{};
    selfInfo.privateData = MakePrivateData(MakeEndpointDesc());
    selfOptions.options.emplace(0, std::move(selfInfo));
    EXPECT_EQ(manager.Prepare(selfOptions), BM_OK);

    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    options.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(options), BM_OK);

    auto changedDesc = MakeEndpointDesc();
    changedDesc.devPhyId++;
    HybmTransPrepareOptions changedOptions{};
    TransportRankPrepareInfo changedInfo{};
    changedInfo.privateData = MakePrivateData(changedDesc);
    changedOptions.options.emplace(1, std::move(changedInfo));
    EXPECT_EQ(manager.Prepare(changedOptions), BM_INVALID_PARAM);
}

TEST(DeviceUrmaTransportManagerTest, PrepareFailsWhenCreatingThreadOrChannel)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    options.options.emplace(1, std::move(info));

    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAllocFail;
    EXPECT_EQ(manager.Prepare(options), BM_ERROR);

    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreateFail;
    EXPECT_EQ(manager.Prepare(options), BM_DL_FUNCTION_FAILED);

    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreateZero;
    EXPECT_EQ(manager.Prepare(options), BM_DL_FUNCTION_FAILED);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
}

TEST(DeviceUrmaTransportManagerTest, PreparePollsInProgressUntilReady)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusInProgressThenReady;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    g_getStatusCallCount = 0;
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    options.options.emplace(1, std::move(info));

    // 无 memKeys，验证轮询行为（两次 IN_PROGRESS 后 READY），不涉及 mem import
    EXPECT_EQ(manager.Prepare(options), BM_OK);
    EXPECT_EQ(g_getStatusCallCount, 3UL); // 两次 IN_PROGRESS + 一次 READY
    EXPECT_EQ(manager.remoteRanks_[1].channel, MOCK_CHANNEL);
    EXPECT_EQ(manager.remoteRanks_[1].thread, MOCK_THREAD);
}

TEST(DeviceUrmaTransportManagerTest, PrepareGetStatusApiFailureDestroysChannelAndThread)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusApiFail;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroyCounted;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFreeCounted;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    g_channelDestroyCallCount = 0;
    g_threadFreeCallCount = 0;
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_DL_FUNCTION_FAILED);
    EXPECT_EQ(g_channelDestroyCallCount, 1);
    EXPECT_EQ(g_threadFreeCallCount, 1);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
}

TEST(DeviceUrmaTransportManagerTest, PrepareGetStatusFailedDestroysChannelAndThread)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusFailed;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroyCounted;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFreeCounted;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    g_channelDestroyCallCount = 0;
    g_threadFreeCallCount = 0;
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_NOT_CONNECTED);
    EXPECT_EQ(g_channelDestroyCallCount, 1);
    EXPECT_EQ(g_threadFreeCallCount, 1);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
}

TEST(DeviceUrmaTransportManagerTest, PrepareGetStatusTimeoutDestroysChannelAndThread)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusTimeout;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroyCounted;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFreeCounted;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    g_channelDestroyCallCount = 0;
    g_threadFreeCallCount = 0;
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_TIMEOUT);
    EXPECT_EQ(g_channelDestroyCallCount, 1);
    EXPECT_EQ(g_threadFreeCallCount, 1);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
}

TEST(DeviceUrmaTransportManagerTest, PrepareGetStatusUnknownDestroysChannelAndThread)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusUnknown;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroyCounted;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFreeCounted;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    g_channelDestroyCallCount = 0;
    g_threadFreeCallCount = 0;
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.privateData = MakePrivateData(MakeEndpointDesc());
    info.role = HYBM_ROLE_PEER;
    options.options.emplace(1, std::move(info));

    EXPECT_EQ(manager.Prepare(options), BM_DL_FUNCTION_FAILED);
    EXPECT_EQ(g_channelDestroyCallCount, 1);
    EXPECT_EQ(g_threadFreeCallCount, 1);
    EXPECT_EQ(manager.remoteRanks_[1].channel, 0U);
    EXPECT_EQ(manager.remoteRanks_[1].thread, 0U);
}

TEST(DeviceUrmaTransportManagerTest, UpdateRankOptionsFallsBackForNewRankAndChecksEdges)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommChannelGetStatus = MockHcommChannelGetStatusReady;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 3;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    HybmTransPrepareOptions invalidOptions{};
    TransportRankPrepareInfo invalidInfo{};
    invalidInfo.privateData = MakePrivateData(MakeEndpointDesc());
    invalidOptions.options.emplace(3, std::move(invalidInfo));
    EXPECT_EQ(manager.UpdateRankOptions(invalidOptions), BM_INVALID_PARAM);

    HybmTransPrepareOptions selfOptions{};
    TransportRankPrepareInfo selfInfo{};
    selfInfo.privateData = MakePrivateData(MakeEndpointDesc());
    selfOptions.options.emplace(0, std::move(selfInfo));
    EXPECT_EQ(manager.UpdateRankOptions(selfOptions), BM_OK);

    HybmTransPrepareOptions fallbackOptions{};
    TransportRankPrepareInfo fallbackInfo{};
    fallbackInfo.privateData = MakePrivateData(MakeEndpointDesc());
    fallbackOptions.options.emplace(2, std::move(fallbackInfo));
    EXPECT_EQ(manager.UpdateRankOptions(fallbackOptions), BM_OK);
    ASSERT_TRUE(manager.remoteRanks_.count(2) != 0);
    EXPECT_EQ(manager.remoteRanks_[2].thread, MOCK_THREAD);
    EXPECT_EQ(manager.remoteRanks_[2].channel, MOCK_CHANNEL);

    manager.remoteRanks_[2].channel = 0;
    EXPECT_EQ(manager.UpdateRankOptions(fallbackOptions), BM_OK);
    EXPECT_EQ(manager.remoteRanks_[2].thread, MOCK_THREAD);
    EXPECT_EQ(manager.remoteRanks_[2].channel, MOCK_CHANNEL);
}

TEST(DeviceUrmaTransportManagerTest, CloseDeviceReleasesRemoteLocalAndDeviceResources)
{
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegAny;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    DlAclApi::pAclrtDestroyNotify = MockAclrtDestroyNotify;
    DlAclApi::pAclrtFree = MockAclrtFreeOk;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.devTransFlagPtr_ = std::malloc(sizeof(int64_t));
    ASSERT_NE(manager.devTransFlagPtr_, nullptr);
    manager.devTransFlagSize_ = sizeof(int64_t);
    HcommMemHandle flagHandle = nullptr;
    const UrmaCommMem flagMem{reinterpret_cast<uint64_t>(manager.devTransFlagPtr_), sizeof(int64_t),
                              UrmaMemoryType::DEVICE_HBM};
    ASSERT_EQ(manager.manager_.HcommMemReg(manager.localEndpoint_, 1, flagMem, &flagHandle), BM_OK);
    manager.devTransFlagHcommHandle_ = flagHandle;

    // Set up a CompletionContext with notify to exercise CleanupContextLocked path
    auto ctx = std::make_shared<DeviceUrmaTransportManager::CompletionContext>();
    ctx->notify = MOCK_NOTIFY;
    ctx->notifyId = MOCK_NOTIFY_ID;
    ctx->notifyAddr = MOCK_NOTIFY_ADDR;
    ctx->notifyLen = MOCK_NOTIFY_LEN;
    HcommMemHandle notifyHandle = nullptr;
    const UrmaCommMem notifyMem{MOCK_NOTIFY_ADDR, MOCK_NOTIFY_LEN, UrmaMemoryType::DEVICE_HBM};
    ASSERT_EQ(manager.manager_.HcommMemReg(manager.localEndpoint_, MOCK_NOTIFY_ADDR, notifyMem, &notifyHandle), BM_OK);
    ctx->notifyHcommHandle = notifyHandle;
    manager.registry_.push_back(ctx);

    auto &state = manager.remoteRanks_[1];
    state.channel = MOCK_CHANNEL;
    state.thread = MOCK_THREAD;
    DeviceUrmaTransportManager::RemoteRegistration remote{};
    remote.addr = MOCK_REMOTE_ADDR;
    remote.size = MOCK_SIZE;
    remote.memTag = MOCK_MEM_TAG;
    remote.descBytes = MakeRawExportDesc(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    remote.view = {MOCK_REMOTE_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    state.imports.push_back(remote);
    state.remoteFlagAddr = MOCK_REMOTE_ADDR;
    state.remoteFlagSize = MOCK_SIZE;
    state.remoteFlagDescBytes.assign(MOCK_HCOMM_DESC_LEN, 0xF1);

    HcommMemHandle localHandle = nullptr;
    const UrmaCommMem localMem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    ASSERT_EQ(manager.manager_.HcommMemReg(manager.localEndpoint_, MOCK_MEM_TAG, localMem, &localHandle), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration local{};
    local.mr.addr = MOCK_LOCAL_ADDR;
    local.mr.size = MOCK_SIZE;
    local.mr.flags = REG_MR_FLAG_DRAM;
    local.handle = localHandle;
    local.memTag = MOCK_LOCAL_ADDR;
    local.refCount = 1U;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, local);

    EXPECT_EQ(manager.CloseDevice(), BM_OK);
    EXPECT_FALSE(manager.opened_);
    EXPECT_TRUE(manager.remoteRanks_.empty());
    EXPECT_TRUE(manager.localRegistrations_.empty());
    EXPECT_EQ(manager.localEndpoint_, nullptr);
    EXPECT_EQ(manager.devTransFlagPtr_, nullptr);
    EXPECT_EQ(manager.devTransFlagHcommHandle_, nullptr);
}

// ============================================================================
// DeviceUrmaTransportManager tests — kernel failure / sync paths
// ============================================================================

TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchRejectsZeroThread)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankCount_ = 2;
    auto &state = manager.remoteRanks_[1];
    state.channel = MOCK_CHANNEL;
    state.thread = 0; // thread is zero, channel is non-zero

    CopyDescriptor desc{};
    desc.localAddrs = {reinterpret_cast<void *>(MOCK_LOCAL_ADDR)};
    desc.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR)};
    desc.counts = {MOCK_SIZE};
    EXPECT_EQ(manager.WriteRemoteBatchAsync(1, desc), BM_NOT_CONNECTED);
    manager.opened_ = false;
}

TEST(DeviceUrmaTransportManagerTest, PrepareKernelLaunchBuffersRejectsMallocFailure)
{
    DlAclApiFnGuard guard;
    DlAclApi::pAclrtMalloc = MockAclrtMallocFail;
    DeviceUrmaTransportManager manager;
    DeviceUrmaTransportManager::DeviceTransferBuffers buffers{};
    EXPECT_EQ(manager.PrepareKernelLaunchBuffers(false, {MOCK_LOCAL_ADDR}, {MOCK_REMOTE_ADDR}, {MOCK_SIZE}, buffers),
              BM_ERROR);
}

TEST(DeviceUrmaTransportManagerTest, SynchronizeNoOpsSucceeds)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    auto &state = manager.remoteRanks_[0];
    state.channel = MOCK_CHANNEL;
    state.thread = MOCK_THREAD;
    EXPECT_EQ(manager.Synchronize(0), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, SynchronizeUnknownRankFails)
{
    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    EXPECT_NE(manager.Synchronize(0), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, SynchronizeNotOpenedFails)
{
    DeviceUrmaTransportManager manager;
    EXPECT_NE(manager.Synchronize(0), BM_OK);
}

// ============================================================================
// DeviceUrmaTransportManager tests — PendingTransfer ownership & retry
// ============================================================================

TEST(DeviceUrmaTransportManagerTest, MultipleAsyncThenSynchronizeClearsAll)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    InstallKernelLaunchMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = 0x2000U;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR + 0x100U, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->pendingTransfers.size(), 2U);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);
    EXPECT_TRUE(ctx->pendingTransfers[1].inFlight);

    EXPECT_EQ(manager.Synchronize(1), BM_OK);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
    manager.opened_ = false; // prevent double-close in destructor
}

TEST(DeviceUrmaTransportManagerTest, CrossRankAsyncSynchronizeOneClearsAll)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    InstallKernelLaunchMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 3;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    for (uint32_t peer = 1; peer <= 2; ++peer) {
        HybmTransPrepareOptions prep{};
        TransportRankPrepareInfo pi{};
        auto pd = MakeEndpointDesc();
        pd.protocol = UrmaProtocol::UBC_CTP;
        pd.type = COMM_ADDR_TYPE_IP_V6;
        pi.privateData = MakePrivateData(pd);
        pi.role = HYBM_ROLE_PEER;
        pi.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
        prep.options.emplace(peer, std::move(pi));
        ASSERT_EQ(manager.Prepare(prep), BM_OK);
    }

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = 0x2000U;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    // Launch transfers on two different ranks from same thread (same context)
    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    EXPECT_EQ(manager.ReadRemoteAsync(2, MOCK_LOCAL_ADDR + 0x100U, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);

    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->pendingTransfers.size(), 2U);

    // Synchronize rank 1 clears only rank 1's pending; rank 2 remains
    EXPECT_EQ(manager.Synchronize(1), BM_OK);
    EXPECT_EQ(ctx->pendingTransfers.size(), 1U);
    EXPECT_EQ(ctx->pendingTransfers[0].rankId, 2U);

    // Synchronize rank 2 clears the remaining pending
    EXPECT_EQ(manager.Synchronize(2), BM_OK);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, LaunchFailureAclrtFreeSuccessRemovesRecord)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));
    // Make kernel launch fail
    DlAclApi::pAclrtKernelArgsInit = MockAclrtKernelArgsInit;
    DlAclApi::pAclrtKernelArgsAppend = MockAclrtKernelArgsAppend;
    DlAclApi::pAclrtKernelArgsFinalize = [](aclrtArgsHandle) { return BM_ERROR; };

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = MOCK_SIZE;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    // Launch failure with successful AclrtFree → record removed
    EXPECT_NE(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, LaunchFailureAclrtFreeFailsRetainsDeferredRecord)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));
    // Make kernel launch fail, and AclrtFree also fail
    DlAclApi::pAclrtKernelArgsInit = MockAclrtKernelArgsInit;
    DlAclApi::pAclrtKernelArgsAppend = MockAclrtKernelArgsAppend;
    DlAclApi::pAclrtKernelArgsFinalize = [](aclrtArgsHandle) { return BM_ERROR; };
    DlAclApi::pAclrtFree = [](void *) { return BM_ERROR; };

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = MOCK_SIZE;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    EXPECT_NE(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    // Record retained (inFlight=false) with buffers for deferred cleanup
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    EXPECT_FALSE(ctx->pendingTransfers[0].inFlight);
    EXPECT_NE(ctx->pendingTransfers[0].buffers.dstList, nullptr);

    // Subsequent Synchronize retries release (AclrtFree now succeeds)
    DlAclApi::pAclrtFree = MockAclrtFreeOk;
    EXPECT_EQ(manager.Synchronize(1), BM_OK);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, AclrtSynchronizeStreamFailureRetainsInFlight)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    InstallKernelLaunchMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));
    g_syncCallCount = 0;
    DlAclApi::pAclrtSynchronizeStream = MockAclrtSynchronizeStreamFirstOkThenFail;

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = MOCK_SIZE;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->pendingTransfers.size(), 1U);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);

    // Sync failure retains inFlight
    EXPECT_NE(manager.Synchronize(1), BM_OK);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);

    // Fix stream sync and retry
    DlAclApi::pAclrtSynchronizeStream = MockAclrtSynchronizeStream;
    EXPECT_EQ(manager.Synchronize(1), BM_OK);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    manager.localRegistrations_.clear();
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, CloseRejectsInFlightAndRetriesDeferredFree)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);
    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    DeviceUrmaTransportManager::LocalRegistration localReg{};
    localReg.mr.addr = MOCK_LOCAL_ADDR;
    localReg.mr.size = MOCK_SIZE;
    localReg.mr.flags = REG_MR_FLAG_DRAM;
    localReg.deviceVa = 0;
    manager.localRegistrations_.emplace(MOCK_LOCAL_ADDR, localReg);

    // Launch a successful transfer (creates inFlight entry)
    InstallKernelLaunchMocks();
    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->pendingTransfers.size(), 1U);

    // CloseDevice releases inFlight and deferred transfers, then succeeds
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
    EXPECT_FALSE(manager.opened_);
    manager.opened_ = false; // prevent destructor double-close
}

// ============================================================================
// DeviceUrmaTransportManager tests — supplemental validation & edge cases
// ============================================================================

TEST(DeviceUrmaTransportManagerTest, GetPrivateDataBeforeEndpointReadyReturnsEmpty)
{
    DeviceUrmaTransportManager manager;
    // localEndpoint_ is null by default — GetPrivateData must not crash
    const auto data = manager.GetPrivateData();
    // Expect empty key (all zeros) — peer will reject with magic mismatch
    bool allZero = true;
    for (const auto &k : data.key.keys) {
        if (k != 0) {
            allZero = false;
            break;
        }
    }
    EXPECT_TRUE(allZero);
}

TEST(DeviceUrmaTransportManagerTest, GetPrivateDataContainsValidEndpointDesc)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    const auto expectedDesc = MakeEndpointDesc();
    manager.localEndpointDesc_ = expectedDesc;
    manager.opened_ = true;

    const auto data = manager.GetPrivateData();
    const auto *raw = reinterpret_cast<const uint8_t *>(data.key.keys);
    // Check header (magic=0xA5FAC003, version=1, payloadLen=sizeof(UrmaEndpointDesc))
    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t payloadLen = 0;
    std::memcpy(&magic, raw, sizeof(magic));
    std::memcpy(&version, raw + sizeof(magic), sizeof(version));
    std::memcpy(&payloadLen, raw + sizeof(magic) + sizeof(version), sizeof(payloadLen));
    EXPECT_EQ(magic, 0xA5FAC003U);
    EXPECT_EQ(version, 1U);
    EXPECT_EQ(payloadLen, sizeof(UrmaEndpointDesc));

    // Check payload matches expectedDesc
    UrmaEndpointDesc decodedDesc{};
    std::memcpy(&decodedDesc, raw + sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t), sizeof(UrmaEndpointDesc));
    EXPECT_EQ(decodedDesc.devPhyId, expectedDesc.devPhyId);
    EXPECT_EQ(decodedDesc.superDevId, expectedDesc.superDevId);
    EXPECT_EQ(decodedDesc.serverIdx, expectedDesc.serverIdx);
    EXPECT_EQ(decodedDesc.superPodIdx, expectedDesc.superPodIdx);
    EXPECT_EQ(decodedDesc.protocol, expectedDesc.protocol);
    EXPECT_EQ(decodedDesc.type, expectedDesc.type);
    EXPECT_EQ(std::memcmp(decodedDesc.raws, expectedDesc.raws, sizeof(decodedDesc.raws)), 0);
}

TEST(DeviceUrmaTransportManagerTest, ReadRemoteNotOpenedFails)
{
    DeviceUrmaTransportManager manager;
    EXPECT_NE(manager.ReadRemote(0, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, WriteRemoteNotOpenedFails)
{
    DeviceUrmaTransportManager manager;
    EXPECT_NE(manager.WriteRemote(0, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionNotOpenedFails)
{
    DeviceUrmaTransportManager manager;
    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_NE(manager.RegisterMemoryRegion(mr), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, UnregisterMemoryRegionUnknownAddrSucceeds)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    // UnregisterMemoryRegion for unknown addr should return BM_OK (no-op with warning)
    EXPECT_EQ(manager.UnregisterMemoryRegion(MOCK_LOCAL_ADDR), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, CloseDeviceNotOpenedSucceeds)
{
    DeviceUrmaTransportManager manager;
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, PrepareFailsOnBadPrivateDataMagic)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.localEndpointDesc_ = MakeEndpointDesc();

    // Private data with wrong magic
    TransportPrivateData badData{};
    badData.key.keys[0] = 0xDEADBEEFU;
    auto key = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    HybmTransPrepareOptions options{};
    TransportRankPrepareInfo info{};
    info.nic = "tcp://127.0.0.1:8000";
    info.privateData = badData;
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {key};
    options.options.emplace(1, std::move(info));

    EXPECT_NE(manager.Prepare(options), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionInvalidFlagsFails)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    // DRAM | HBM flags simultaneously — IsSupportedMemoryFlags returns false
    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM | REG_MR_FLAG_HBM;
    EXPECT_NE(manager.RegisterMemoryRegion(mr), BM_OK);
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionTracksRefCountAndQueryMemoryKeyExportsFlag)
{
    DlAclApiFnGuard aclGuard;
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegAny;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;
    DlHcommApi::gHcommMemExport = MockHcommMemExportAny;
    DlAclApi::pAclrtFree = MockAclrtFreeOk;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.devTransFlagHcommHandle_ = MOCK_FLAG_HANDLE;
    manager.devTransFlagPtr_ = std::malloc(sizeof(int64_t));
    ASSERT_NE(manager.devTransFlagPtr_, nullptr);
    manager.devTransFlagSize_ = sizeof(int64_t);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);
    auto conflict = mr;
    conflict.size = MOCK_SIZE + 1U;
    EXPECT_EQ(manager.RegisterMemoryRegion(conflict), BM_ERROR);
    ASSERT_EQ(manager.localRegistrations_.size(), 1U);
    EXPECT_EQ(manager.localRegistrations_[MOCK_LOCAL_ADDR].refCount, 2U);
    EXPECT_TRUE(manager.QueryHasRegistered(MOCK_LOCAL_ADDR + 0x10U, 0x20U));

    TransportMemoryKey key{};
    EXPECT_EQ(manager.QueryMemoryKey(MOCK_LOCAL_ADDR, key), BM_OK);
    EXPECT_EQ(key.keys[0], URMA_EXPORT_DESC_MAGIC);
    EXPECT_EQ(key.keys[1], MOCK_LOCAL_ADDR);
    constexpr uint32_t kHeaderSlots = DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS;
    UrmaExportDesc exportDesc{};
    std::memcpy(&exportDesc, reinterpret_cast<uint8_t *>(&key.keys[kHeaderSlots]), sizeof(exportDesc));
    EXPECT_EQ(exportDesc.devTransFlagDescLen, MOCK_HCOMM_DESC_LEN);

    EXPECT_EQ(manager.UnregisterMemoryRegion(MOCK_LOCAL_ADDR), BM_OK);
    ASSERT_EQ(manager.localRegistrations_.size(), 1U);
    EXPECT_EQ(manager.localRegistrations_[MOCK_LOCAL_ADDR].refCount, 1U);
    EXPECT_EQ(manager.UnregisterMemoryRegion(MOCK_LOCAL_ADDR), BM_OK);
    EXPECT_TRUE(manager.localRegistrations_.empty());
    // Free manually allocated devTransFlagPtr_, then nullify handles set directly
    // (not via HcommMemReg) to avoid double-cleanup in destructor.
    (void)DlAclApi::AclrtFree(manager.devTransFlagPtr_);
    manager.devTransFlagHcommHandle_ = nullptr;
    manager.devTransFlagPtr_ = nullptr;
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionPropagatesHcommRegFailure)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegFailAny;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_DL_FUNCTION_FAILED);
    EXPECT_TRUE(manager.localRegistrations_.empty());
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionRejectsInvalidRangesAndMissingEndpoint)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankCount_ = 2;

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_NOT_INITIALIZED);

    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    mr.addr = 0;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_INVALID_PARAM);
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = 0;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_INVALID_PARAM);
    mr.addr = UINT64_MAX;
    mr.size = 2;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_INVALID_PARAM);
}

TEST(DeviceUrmaTransportManagerTest, QueryMemoryKeyFailsWhenFlagHandleMissing)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegAny;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;
    DlHcommApi::gHcommMemExport = MockHcommMemExportAny;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    ASSERT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);

    TransportMemoryKey key{};
    EXPECT_EQ(manager.QueryMemoryKey(MOCK_LOCAL_ADDR, key), BM_ERROR);
}

TEST(DeviceUrmaTransportManagerTest, QueryMemoryKeyFailsWhenFlagExportFails)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegAny;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;
    DlHcommApi::gHcommMemExport = MockHcommMemExportSecondFails;

    g_memExportCallCount = 0;
    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);
    manager.devTransFlagHcommHandle_ = MOCK_FLAG_HANDLE;

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    ASSERT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);

    TransportMemoryKey key{};
    EXPECT_EQ(manager.QueryMemoryKey(MOCK_LOCAL_ADDR, key), BM_DL_FUNCTION_FAILED);
    manager.devTransFlagHcommHandle_ = nullptr;
}

TEST(DeviceUrmaTransportManagerTest, UpdateMemoryKeyRewritesAddressWhenProvided)
{
    DeviceUrmaTransportManager manager;
    TransportMemoryKey key{};
    manager.UpdateMemoryKey(key, reinterpret_cast<void *>(MOCK_REMOTE_ADDR));
    EXPECT_EQ(key.keys[1], MOCK_REMOTE_ADDR);

    manager.UpdateMemoryKey(key, nullptr);
    EXPECT_EQ(key.keys[1], MOCK_REMOTE_ADDR);
}

TEST(DeviceUrmaTransportManagerTest, ImportRemoteMemKeysImportsFlagAndSkipsDuplicateMemTag)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    auto &state = manager.remoteRanks_[1];
    state.remoteEndpointDesc = MakeEndpointDesc();
    auto key = MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {key}), BM_OK);
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {key}), BM_OK);
    ASSERT_EQ(state.imports.size(), 1U);
    EXPECT_EQ(state.imports.front().memTag, MOCK_MEM_TAG);
    EXPECT_EQ(state.remoteFlagAddr, MOCK_REMOTE_ADDR);
    EXPECT_EQ(state.remoteFlagSize, MOCK_SIZE);
    EXPECT_EQ(state.remoteFlagDescBytes.size(), MOCK_HCOMM_DESC_LEN);
}

TEST(DeviceUrmaTransportManagerTest, ImportRemoteMemKeysRejectsBadInputsAndProtocolMismatch)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    auto &state = manager.remoteRanks_[1];
    state.remoteEndpointDesc = MakeEndpointDesc();

    auto badMagic = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    badMagic.keys[0] = 0;
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {badMagic}), BM_INVALID_PARAM);

    auto zeroAddr = MakeImportKey(0, MOCK_SIZE, MOCK_MEM_TAG);
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {zeroAddr}), BM_INVALID_PARAM);

    auto badPayload = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    auto *payload = reinterpret_cast<uint8_t *>(&badPayload.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    UrmaExportDesc exportDesc{};
    std::memcpy(&exportDesc, payload, sizeof(exportDesc));
    exportDesc.hcommDescLen = 0;
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {badPayload}), BM_INVALID_PARAM);

    state.remoteEndpointDesc.protocol = UrmaProtocol::ROCE;
    auto goodKey = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {goodKey}), BM_INVALID_PARAM);
}

TEST(DeviceUrmaTransportManagerTest, ImportRemoteMemKeysRejectsMalformedPayloadsAndFlagImportFailures)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    auto &state = manager.remoteRanks_[1];
    state.remoteEndpointDesc = MakeEndpointDesc();

    auto badDescMagic = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    auto *payload = reinterpret_cast<uint8_t *>(&badDescMagic.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    UrmaExportDesc exportDesc{};
    std::memcpy(&exportDesc, payload, sizeof(exportDesc));
    exportDesc.magic = 0;
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {badDescMagic}), BM_INVALID_PARAM);

    auto badHeaderSize = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    payload = reinterpret_cast<uint8_t *>(&badHeaderSize.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    std::memcpy(&exportDesc, payload, sizeof(exportDesc));
    exportDesc.headerSize = sizeof(UrmaExportDesc) + 1U;
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {badHeaderSize}), BM_INVALID_PARAM);

    auto tooLarge = MakeImportKey(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    payload = reinterpret_cast<uint8_t *>(&tooLarge.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    std::memcpy(&exportDesc, payload, sizeof(exportDesc));
    exportDesc.hcommDescLen = sizeof(tooLarge.keys);
    exportDesc.devTransFlagDescLen = 1U;
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {tooLarge}), BM_INVALID_PARAM);

    g_memImportCallCount = 0;
    DlHcommApi::gHcommMemImport = MockHcommMemImportSecondFails;
    auto flagKey = MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG);
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {flagKey}), BM_DL_FUNCTION_FAILED);
    EXPECT_TRUE(state.imports.empty());
    EXPECT_TRUE(state.remoteFlagDescBytes.empty());

    g_memImportCallCount = 0;
    DlHcommApi::gHcommMemImport = MockHcommMemImportSecondInvalidType;
    EXPECT_EQ(manager.ImportRemoteMemKeysLocked(1, state, {flagKey}), BM_INVALID_PARAM);
    EXPECT_TRUE(state.imports.empty());
    EXPECT_TRUE(state.remoteFlagDescBytes.empty());
}

TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchRejectsMissingRankAndChannel)
{
    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankCount_ = 2;

    CopyDescriptor desc{};
    EXPECT_EQ(manager.WriteRemoteBatchAsync(1, desc), BM_OK);

    desc.localAddrs.push_back(reinterpret_cast<void *>(MOCK_LOCAL_ADDR));
    desc.globalAddrs.push_back(reinterpret_cast<void *>(MOCK_REMOTE_ADDR));
    desc.counts.push_back(MOCK_SIZE);
    EXPECT_EQ(manager.WriteRemoteBatchAsync(1, desc), BM_NOT_CONNECTED);

    auto &state = manager.remoteRanks_[1];
    EXPECT_EQ(manager.WriteRemoteBatchAsync(1, desc), BM_NOT_CONNECTED);
    state.channel = MOCK_CHANNEL;
    state.thread = MOCK_THREAD;
    EXPECT_EQ(manager.WriteRemoteBatchAsync(1, desc), BM_INVALID_PARAM);
    manager.opened_ = false;
}

TEST(DeviceUrmaTransportManagerTest, RemoteIoZeroSizeAndFindRegistrationEdges)
{
    DeviceUrmaTransportManager manager;
    EXPECT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, 0), BM_OK);

    manager.opened_ = true;
    manager.rankCount_ = 2;
    EXPECT_EQ(manager.FindLocalRegistrationLocked(0, MOCK_SIZE, nullptr), BM_INVALID_PARAM);
    EXPECT_EQ(manager.FindLocalRegistrationLocked(MOCK_LOCAL_ADDR, 0, nullptr), BM_INVALID_PARAM);
    EXPECT_EQ(manager.FindLocalRegistrationLocked(UINT64_MAX, 2, nullptr), BM_INVALID_PARAM);
    EXPECT_EQ(manager.FindRemoteRegistrationLocked(2, MOCK_REMOTE_ADDR, MOCK_SIZE, nullptr), BM_NOT_CONNECTED);
    EXPECT_EQ(manager.FindRemoteRegistrationLocked(2, MOCK_REMOTE_ADDR, 0, nullptr), BM_NOT_CONNECTED);
    manager.remoteRanks_.try_emplace(1);
    EXPECT_EQ(manager.FindRemoteRegistrationLocked(1, UINT64_MAX, 2, nullptr), BM_INVALID_PARAM);
}

TEST(DeviceUrmaTransportManagerTest, SynchronizeNotConnectedFails)
{
    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    EXPECT_EQ(manager.Synchronize(1), BM_NOT_CONNECTED);
}

TEST(DeviceUrmaTransportManagerTest, LoadDeviceKernelGetsHandlesFromPreloadedBinary)
{
    DlAclApiFnGuard guard;
    DlAclApi::pAclrtBinaryGetFunction = MockAclrtBinaryGetFunction;

    aclrtBinHandle binHandle = MOCK_BIN_HANDLE;
    DeviceFuncHandles handles{};
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_OK);
    EXPECT_EQ(binHandle, MOCK_BIN_HANDLE);
    EXPECT_EQ(handles.batchRead, MOCK_READ_FUNC);
    EXPECT_EQ(handles.batchWrite, MOCK_WRITE_FUNC);
}

TEST(DeviceUrmaTransportManagerTest, LoadDeviceKernelUsesDefaultPathWhenBinaryAlreadyLoaded)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    (void)unsetenv("ASCEND_HOME_PATH");
    DlAclApiFnGuard guard;
    DlAclApi::pAclrtBinaryGetFunction = MockAclrtBinaryGetFunction;

    aclrtBinHandle binHandle = MOCK_BIN_HANDLE;
    DeviceFuncHandles handles{};
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_OK);
    EXPECT_EQ(handles.batchRead, MOCK_READ_FUNC);
    EXPECT_EQ(handles.batchWrite, MOCK_WRITE_FUNC);
}

TEST(DeviceUrmaTransportManagerTest, LoadDeviceKernelRejectsMissingJsonAndBinaryLoadFailure)
{
    EnvVarGuard envGuard("ASCEND_HOME_PATH");
    DlAclApiFnGuard guard;
    const std::string missingBase =
        std::string(testing::TempDir()) + "missing_device_urma_kernel_" + std::to_string(getpid());
    (void)setenv("ASCEND_HOME_PATH", missingBase.c_str(), 1);

    aclrtBinHandle binHandle = nullptr;
    DeviceFuncHandles handles{};
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_FILE_NOT_ACCESS);

    PrepareKernelJson();
    DlAclApi::pAclrtBinaryLoadFromFile = MockAclrtBinaryLoadFromFileFail;
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_ERROR);
}

TEST(DeviceUrmaTransportManagerTest, LoadDeviceKernelPropagatesGetFunctionFailure)
{
    DlAclApiFnGuard guard;
    DlAclApi::pAclrtBinaryGetFunction = MockAclrtBinaryGetFunctionFail;

    aclrtBinHandle binHandle = MOCK_BIN_HANDLE;
    DeviceFuncHandles handles{};
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_ERROR);
}

TEST(DeviceUrmaTransportManagerTest, LoadDeviceKernelRejectsNullFuncAndNullReturnedHandle)
{
    DlAclApiFnGuard guard;

    aclrtBinHandle binHandle = MOCK_BIN_HANDLE;
    DeviceFuncHandles handles{};
    EXPECT_EQ(LoadDeviceKernelAndGetHandles(nullptr, "write_kernel", binHandle, handles), BM_INVALID_PARAM);

    DlAclApi::pAclrtBinaryGetFunction = MockAclrtBinaryGetFunctionNull;
    EXPECT_EQ(LoadDeviceKernelAndGetHandles("read_kernel", "write_kernel", binHandle, handles), BM_DL_FUNCTION_FAILED);
}

// ============================================================================
// DeviceUrmaTransportManager tests — DRAM DVA conversion in RegisterMemoryRegion
// ============================================================================

constexpr uint64_t MOCK_DRAM_HVA = 0xFFF000001000ULL;
constexpr uint64_t MOCK_DRAM_DVA = 0x124000000000ULL;
constexpr uint64_t MOCK_DRAM_SIZE = 0x100000ULL;

struct VaManagerGuard {
    ~VaManagerGuard()
    {
        HybmVaManager::GetInstance().ClearAll();
    }
};

int32_t MockHcommMemRegVerifyDva(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem,
                                 HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memTag, nullptr);
    EXPECT_NE(mem, nullptr);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_EQ(mem->addr, reinterpret_cast<void *>(MOCK_DRAM_DVA));
    EXPECT_EQ(mem->size, MOCK_DRAM_SIZE);
    EXPECT_EQ(mem->type, COMM_MEM_TYPE_HOST);
    *memHandle = MOCK_MEM_HANDLE;
    return BM_OK;
}

int32_t MockHcommMemRegVerifyHva(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem,
                                 HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memTag, nullptr);
    EXPECT_NE(mem, nullptr);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_EQ(mem->addr, reinterpret_cast<void *>(MOCK_DRAM_HVA));
    EXPECT_EQ(mem->size, MOCK_DRAM_SIZE);
    EXPECT_EQ(mem->type, COMM_MEM_TYPE_HOST);
    *memHandle = MOCK_MEM_HANDLE;
    return BM_OK;
}

int32_t MockHcommMemRegVerifyHbm(EndpointHandle endpoint, const char *memTag, const HcommCommMem *mem,
                                 HcommMemHandle *memHandle)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    EXPECT_NE(memTag, nullptr);
    EXPECT_NE(mem, nullptr);
    EXPECT_NE(memHandle, nullptr);
    EXPECT_EQ(mem->type, COMM_MEM_TYPE_DEVICE);
    *memHandle = MOCK_MEM_HANDLE;
    return BM_OK;
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionDramUsesDvaWhenMappingExists)
{
    VaManagerGuard vaGuard;
    DlHcommApiFnGuard hcommGuard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegVerifyDva;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;

    auto &vaManager = HybmVaManager::GetInstance();
    vaManager.ClearAll();
    vaManager.Initialize(AscendSocType::ASCEND_910B);
    BaseAllocatedGvaInfo baseInfo{};
    baseInfo.va[HVM_GVA] = 0;
    baseInfo.va[HVM_DVA] = MOCK_DRAM_DVA;
    baseInfo.va[HVM_HVA] = MOCK_DRAM_HVA;
    baseInfo.size = MOCK_DRAM_SIZE;
    baseInfo.memType = HYBM_MEM_TYPE_HOST;
    ASSERT_EQ(vaManager.AddVaInfoFromExternal(baseInfo, 0), BM_OK);

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2UL;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_DRAM_HVA;
    mr.size = MOCK_DRAM_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);

    ASSERT_EQ(manager.localRegistrations_.size(), 1U);
    const auto &reg = manager.localRegistrations_[MOCK_DRAM_HVA];
    EXPECT_EQ(reg.mr.addr, MOCK_DRAM_HVA);
    EXPECT_EQ(reg.deviceVa, MOCK_DRAM_DVA);
    EXPECT_EQ(reg.handle, MOCK_MEM_HANDLE);
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionDramFallsBackToHvaWhenNoDvaMapping)
{
    VaManagerGuard vaGuard;
    DlHcommApiFnGuard hcommGuard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegVerifyHva;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;

    auto &vaManager = HybmVaManager::GetInstance();
    vaManager.ClearAll();
    vaManager.Initialize(AscendSocType::ASCEND_910B);

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2UL;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_DRAM_HVA;
    mr.size = MOCK_DRAM_SIZE;
    mr.flags = REG_MR_FLAG_DRAM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);

    ASSERT_EQ(manager.localRegistrations_.size(), 1U);
    const auto &reg = manager.localRegistrations_[MOCK_DRAM_HVA];
    EXPECT_EQ(reg.mr.addr, MOCK_DRAM_HVA);
    EXPECT_EQ(reg.deviceVa, 0U);
    EXPECT_EQ(reg.handle, MOCK_MEM_HANDLE);
}

TEST(DeviceUrmaTransportManagerTest, RegisterMemoryRegionHbmSkipsDvaConversion)
{
    VaManagerGuard vaGuard;
    DlHcommApiFnGuard hcommGuard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemRegVerifyHbm;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregAny;

    DeviceUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0;
    manager.rankCount_ = 2UL;
    manager.localEndpoint_ = manager.manager_.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(manager.localEndpoint_, nullptr);

    TransportMemoryRegion mr{};
    mr.addr = MOCK_LOCAL_ADDR;
    mr.size = MOCK_SIZE;
    mr.flags = REG_MR_FLAG_HBM;
    EXPECT_EQ(manager.RegisterMemoryRegion(mr), BM_OK);

    ASSERT_EQ(manager.localRegistrations_.size(), 1U);
    const auto &reg = manager.localRegistrations_[MOCK_LOCAL_ADDR];
    EXPECT_EQ(reg.mr.addr, MOCK_LOCAL_ADDR);
    EXPECT_EQ(reg.deviceVa, 0U);
}

// ============================================================================
// DeviceUrmaTransportManager tests — CorrectLocalRegAddressLocked single IO
// ============================================================================

// Verify single read IO uses DVA from ACL_DRAM local registration as staged dst address.
TEST(DeviceUrmaTransportManagerTest, RemoteIoConvertsAclDramLocalAddrToDva)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_ACL_DRAM, MOCK_DRAM_DVA);

    ASSERT_EQ(manager.ReadRemoteAsync(1UL, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], MOCK_DRAM_DVA);
    EXPECT_EQ(staged[1], MOCK_REMOTE_ADDR);
    EXPECT_EQ(staged[2], MOCK_SIZE);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);

    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    EXPECT_GE(g_kernelLaunchCallCount, 1U);
    fix.CleanupAndClose(manager);
}

// A valid corrected DVA start may launch even when the corrected end range wraps.
TEST(DeviceUrmaTransportManagerTest, RemoteIoAllowsDvaRangeEndWrapWhenStartIsValid)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    // deviceVa near max: corrected start is valid, but correctedAddr+MOCK_SIZE wraps.
    constexpr uint64_t kNearMax = ~0ULL - 0x800ULL;
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, 0x2000U, REG_MR_FLAG_DRAM, kNearMax);

    ASSERT_EQ(manager.ReadRemoteAsync(1, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    EXPECT_GE(g_kernelLaunchCallCount, 1U);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);
    EXPECT_EQ(manager.Synchronize(1), BM_OK);
    EXPECT_TRUE(ctx->pendingTransfers.empty());
    fix.CleanupAndClose(manager);
}

// Single write IO: local address (src for write) corrected to DVA.
TEST(DeviceUrmaTransportManagerTest, WriteRemoteConvertsDramLocalAddrToDva)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_DRAM, MOCK_DRAM_DVA);

    ASSERT_EQ(manager.WriteRemoteAsync(1UL, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], MOCK_REMOTE_ADDR);
    EXPECT_EQ(staged[1], MOCK_DRAM_DVA);
    EXPECT_EQ(staged[2], MOCK_SIZE);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);
    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    fix.CleanupAndClose(manager);
}

// HBM local address with deviceVa==0 → keep original address unchanged.
TEST(DeviceUrmaTransportManagerTest, RemoteIoKeepsHbmLocalAddrUnchanged)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_HBM, 0);

    ASSERT_EQ(manager.ReadRemoteAsync(1UL, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], MOCK_LOCAL_ADDR);
    EXPECT_EQ(staged[1], MOCK_REMOTE_ADDR);
    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    fix.CleanupAndClose(manager);
}

// DRAM registration with deviceVa==0 (TransformVa failed): keep original HVA.
TEST(DeviceUrmaTransportManagerTest, RemoteIoDramNoDvaKeepsHva)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_DRAM, 0);

    ASSERT_EQ(manager.ReadRemoteAsync(1UL, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], MOCK_LOCAL_ADDR);
    EXPECT_EQ(staged[1], MOCK_REMOTE_ADDR);
    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    fix.CleanupAndClose(manager);
}

// Unregistered local address → RemoteIo returns BM_INVALID_PARAM.
TEST(DeviceUrmaTransportManagerTest, RemoteIoFailsForUnregisteredLocalAddr)
{
    DlHcommApiFnGuard hcommGuard;
    DlAclApiFnGuard aclGuard;
    DlRtApiFnGuard rtGuard;
    MockcppScope mockcpp;
    PrepareKernelJson();
    InstallOpenDeviceMocks();
    DlHcommApi::gHcommThreadAlloc = MockHcommThreadAlloc;
    DlHcommApi::gHcommChannelCreate = MockHcommChannelCreate;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;
    DlHcommApi::gHcommChannelDestroy = MockHcommChannelDestroy;
    DlHcommApi::gHcommThreadFree = MockHcommThreadFree;
    MOCKER(&ock::mf::DlAclApi::GetAscendSocType).stubs().will(returnValue(ock::mf::AscendSocType::ASCEND_950));
    MOCKER(&ock::mf::transport::device::GetDeviceUrmaEid).stubs().will(invoke(MockGetDeviceUrmaEid));
    MOCKER(&ock::mf::HybmStreamManager::GetThreadAclStream).stubs().will(returnValue(MOCK_STREAM));

    DeviceUrmaTransportManager manager;
    TransportOptions openOptions{};
    openOptions.rankId = 0;
    openOptions.rankCount = 2UL;
    openOptions.protocol = HYBM_DOP_TYPE_DEVICE_URMA;
    ASSERT_EQ(manager.OpenDevice(openOptions), BM_OK);

    HybmTransPrepareOptions prepareOptions{};
    TransportRankPrepareInfo info{};
    auto peerDesc = MakeEndpointDesc();
    peerDesc.protocol = UrmaProtocol::UBC_CTP;
    peerDesc.type = COMM_ADDR_TYPE_IP_V6;
    info.privateData = MakePrivateData(peerDesc);
    info.role = HYBM_ROLE_PEER;
    info.memKeys = {MakeImportKeyWithFlag(MOCK_REMOTE_ADDR, MOCK_SIZE, MOCK_MEM_TAG)};
    prepareOptions.options.emplace(1UL, std::move(info));
    ASSERT_EQ(manager.Prepare(prepareOptions), BM_OK);

    // No local registration → CorrectLocalRegAddressLocked fails
    EXPECT_EQ(manager.ReadRemoteAsync(1UL, MOCK_LOCAL_ADDR, MOCK_REMOTE_ADDR, MOCK_SIZE), BM_INVALID_PARAM);
    EXPECT_EQ(manager.CloseDevice(), BM_OK);
}

// ============================================================================
// DeviceUrmaTransportManager tests — CorrectLocalRegAddressLocked batch
// ============================================================================

// Batch with mixed DRAM+DVA and HBM local addresses: element-wise correction.
TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchCorrectsDramAndHbmElements)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);

    constexpr uint64_t kDramLocal = MOCK_LOCAL_ADDR;
    constexpr uint64_t kDramSize = 0x200U;
    constexpr uint64_t kHbmLocal = MOCK_LOCAL_ADDR + 0x10000U;
    constexpr uint64_t kHbmSize = MOCK_SIZE;
    constexpr uint64_t kDramOffset = 0x40U;
    constexpr uint64_t kHbmOffset = 0x80U;
    constexpr uint64_t kExpectedDva = MOCK_DRAM_DVA + kDramOffset;

    fix.AddLocalReg(manager, kDramLocal, kDramSize, REG_MR_FLAG_DRAM, MOCK_DRAM_DVA);
    fix.AddLocalReg(manager, kHbmLocal, kHbmSize, REG_MR_FLAG_HBM, 0);

    CopyDescriptor desc{};
    desc.localAddrs = {reinterpret_cast<void *>(kDramLocal + kDramOffset),
                       reinterpret_cast<void *>(kHbmLocal + kHbmOffset)};
    desc.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR),
                        reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x100U)};
    desc.counts = {0x100U, 0x200U};

    ASSERT_EQ(manager.ReadRemoteBatchAsync(1UL, desc), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], kExpectedDva);
    EXPECT_EQ(staged[1], kHbmLocal + kHbmOffset);
    EXPECT_EQ(staged[2], MOCK_REMOTE_ADDR);
    EXPECT_EQ(staged[3], MOCK_REMOTE_ADDR + 0x100U);
    EXPECT_EQ(staged[4], 0x100U);
    EXPECT_EQ(staged[5], 0x200U);
    EXPECT_TRUE(ctx->pendingTransfers[0].inFlight);
    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    EXPECT_GE(g_kernelLaunchCallCount, 1U);
    fix.CleanupAndClose(manager);
}

// Batch with size==0 element: skipped, no correction attempted.
TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchSkipsZeroSizeElement)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_DRAM, MOCK_DRAM_DVA);

    CopyDescriptor desc{};
    desc.localAddrs = {reinterpret_cast<void *>(MOCK_LOCAL_ADDR), reinterpret_cast<void *>(MOCK_LOCAL_ADDR + 0x100U)};
    desc.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR),
                        reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x100U)};
    desc.counts = {0U, 0x100U};

    ASSERT_EQ(manager.ReadRemoteBatchAsync(1UL, desc), BM_OK);
    ASSERT_EQ(manager.registry_.size(), 1U);
    auto *ctx = manager.registry_[0].get();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->pendingTransfers.size(), 1U);
    const auto *staged = static_cast<const uint64_t *>(ctx->pendingTransfers[0].buffers.dstList);
    ASSERT_NE(staged, nullptr);
    EXPECT_EQ(staged[0], MOCK_DRAM_DVA + 0x100U);
    EXPECT_EQ(staged[1], MOCK_REMOTE_ADDR + 0x100U);
    EXPECT_EQ(staged[2], 0x100U);
    EXPECT_EQ(manager.Synchronize(1UL), BM_OK);
    fix.CleanupAndClose(manager);
}

// Batch with unregistered local address: error, no launch.
TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchFailsWhenLocalNotRegistered)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, MOCK_SIZE, REG_MR_FLAG_DRAM, MOCK_DRAM_DVA);

    CopyDescriptor desc{};
    desc.localAddrs = {reinterpret_cast<void *>(MOCK_LOCAL_ADDR), reinterpret_cast<void *>(MOCK_LOCAL_ADDR + 0x10000U)};
    desc.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR),
                        reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x100U)};
    desc.counts = {0x100U, 0x100U};

    EXPECT_EQ(manager.ReadRemoteBatchAsync(1UL, desc), BM_INVALID_PARAM);
    EXPECT_EQ(g_kernelLaunchCallCount, 0);
    EXPECT_TRUE(manager.registry_.empty());
    fix.CleanupAndClose(manager);
}

// Batch with all size==0 returns BM_OK without pending/launch.
TEST(DeviceUrmaTransportManagerTest, RemoteIoBatchAllZeroSizeReturnsOkWithoutLaunch)
{
    g_kernelLaunchCallCount = 0;
    DeviceTestFixture fix;
    fix.InstallAll();
    DeviceUrmaTransportManager manager;
    fix.OpenAndPreparePeer(manager);
    fix.AddLocalReg(manager, MOCK_LOCAL_ADDR, 0x2000U, REG_MR_FLAG_DRAM, MOCK_DRAM_DVA);

    CopyDescriptor desc;
    desc.localAddrs = {reinterpret_cast<void *>(MOCK_LOCAL_ADDR), reinterpret_cast<void *>(MOCK_LOCAL_ADDR + 0x100U)};
    desc.globalAddrs = {reinterpret_cast<void *>(MOCK_REMOTE_ADDR),
                        reinterpret_cast<void *>(MOCK_REMOTE_ADDR + 0x100U)};
    desc.counts = {0U, 0U};

    EXPECT_EQ(manager.ReadRemoteBatchAsync(1, desc), BM_OK);
    EXPECT_EQ(g_kernelLaunchCallCount, 0);
    EXPECT_TRUE(manager.registry_.empty());
    fix.CleanupAndClose(manager);
}

// ===================== DeviceUrmaEidReader tests =====================
// Using UT_ENABLED seam (device_urma_eid_reader.h) to inject
// mock hccn_tool and mock /etc/hccn.conf paths deterministically.

namespace {

static bool RemoveDirRecursive(const std::string &path)
{
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        return false;
    }
    bool ok = true;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        const std::string fullPath = path + "/" + entry->d_name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                ok = RemoveDirRecursive(fullPath) && ok;
            } else {
                ok = (unlink(fullPath.c_str()) == 0) && ok;
            }
        }
    }
    closedir(dir);
    ok = (rmdir(path.c_str()) == 0) && ok;
    return ok;
}

class EidReaderTempDir {
public:
    EidReaderTempDir()
    {
        dir_ = testing::TempDir() + "eid_reader_" + std::to_string(getpid()) + "_" + std::to_string(rand());
        created_ = (mkdir(dir_.c_str(), 0755) == 0);
    }
    ~EidReaderTempDir()
    {
        if (created_) {
            RemoveDirRecursive(dir_);
        }
    }
    const std::string &path() const
    {
        return dir_;
    }
    bool IsCreated() const
    {
        return created_;
    }

private:
    std::string dir_;
    bool created_{false};
};

void CreateMockHccnTool(const std::string &dir, uint32_t phyDeviceId, const std::string &outputLine, int exitCode = 0)
{
    const std::string path = dir + "/hccn_tool";
    std::ofstream script(path);
    ASSERT_TRUE(script.is_open()) << "Failed to create mock hccn_tool: " << path;
    script << "#!/bin/bash\n";
    script << "if [ \"$*\" != \"-g -ip -i " << phyDeviceId << " -d bond" << phyDeviceId << "\" ]; then\n";
    script << "    exit 1\n";
    script << "fi\n";
    if (!outputLine.empty()) {
        script << "echo '" << outputLine << "'\n";
    }
    script << "exit " << exitCode << "\n";
    script.close();
    const int rc = chmod(path.c_str(), 0755);
    ASSERT_EQ(rc, 0) << "Failed to chmod mock hccn_tool: " << path;
}

void CreateHccnConf(const std::string &path, const std::vector<std::string> &lines)
{
    std::ofstream conf(path);
    ASSERT_TRUE(conf.is_open()) << "Failed to create hccn.conf: " << path;
    for (const auto &line : lines) {
        conf << line << "\n";
    }
    conf.close();
}

} // anonymous namespace

TEST(DeviceUrmaEidReaderTest, ValidToolWinsOverDifferentConfIp)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: 10.10.21.2");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.99"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 2);
}

TEST(DeviceUrmaEidReaderTest, NonZeroToolFallsBack)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: 10.10.21.2", 1);
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, MissingIpaddrFallsBack)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "other: value");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, InvalidToolIpFallsBack)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: not_an_ip");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, ValidOutputPlusNonZeroExitFallsBack)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: 10.10.21.2", 1);
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, ExactKeyArbitraryOrder)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "other: value");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_10=10.10.21.10", "address_1=10.10.21.1", "address_2=10.10.21.2"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    // Must match address_1, NOT address_10
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 1);
}

TEST(DeviceUrmaEidReaderTest, BothSourcesFail)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    // Tool not found (file does not exist at this path)
    const std::string confPath = tmp.path() + "/hccn.conf";
    // Config has key for wrong device (address_2, not address_1)
    CreateHccnConf(confPath, {"address_2=10.10.21.2"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_NE(ret, BM_OK);
}

TEST(DeviceUrmaEidReaderTest, NonexistentToolFallsBackToConfig)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    // Tool does not exist at this path -> fall back to config
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret =
        GetDeviceUrmaIpAddrFromSources(tmp.path() + "/nonexistent_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, ConfigTrimmedExactKey)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "other: value");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"  address_1 = 10.10.21.55  "});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, ConfigIpv6Parse)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "other: value");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=2001:db8::1"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V6);
    // 2001:0db8::1 in network byte order
    EXPECT_EQ(addrData[0], 0x20);
    EXPECT_EQ(addrData[1], 0x01);
    EXPECT_EQ(addrData[2], 0x0D);
    EXPECT_EQ(addrData[3], 0xB8);
    EXPECT_EQ(addrData[14], 0x00);
    EXPECT_EQ(addrData[15], 0x01);
}

TEST(DeviceUrmaEidReaderTest, ToolReturns32HexIp)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: 20010db8000000000000000000000001");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.99"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V6);
    // 2001:0db8::1 in network byte order (32 hex: 20010db8000000000000000000000001)
    EXPECT_EQ(addrData[0], 0x20);
    EXPECT_EQ(addrData[1], 0x01);
    EXPECT_EQ(addrData[2], 0x0D);
    EXPECT_EQ(addrData[3], 0xB8);
    EXPECT_EQ(addrData[14], 0x00);
    EXPECT_EQ(addrData[15], 0x01);
}

TEST(DeviceUrmaEidReaderTest, InvalidSigned32CharRejected)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    // 32 chars but starts with '+' -> IsAllHex rejects it, falls back to config
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: +0010db80000000000000000000000001");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, InvalidNonhex32CharRejected)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    // 32 chars with non-hex 'Z' -> IsAllHex rejects it, falls back to config
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.55"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[3], 55);
}

TEST(DeviceUrmaEidReaderTest, InvalidToolIpAndInvalidConfigIpFails)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    // Tool returns invalid IP -> falls back to config
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: not_an_ip");
    const std::string confPath = tmp.path() + "/hccn.conf";
    // Config has valid key but invalid IP value -> overall failure
    CreateHccnConf(confPath, {"address_1=bad_ip_value"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_NE(ret, BM_OK);
}

TEST(DeviceUrmaEidReaderTest, BothSourcesFailConfigMissing)
{
    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: invalid_ip_here");
    // Config does not exist at this path
    const std::string confPath = tmp.path() + "/nonexistent/hccn.conf";
    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_NE(ret, BM_OK);
}

TEST(DeviceUrmaEidReaderTest, ShellSafeToolPath)
{
    const std::string baseDir = testing::TempDir() + "eid_reader_space_" + std::to_string(getpid());
    const std::string toolDir = baseDir + "/sub dir";
    ASSERT_EQ(mkdir(baseDir.c_str(), 0755), 0) << "mkdir base";
    ASSERT_EQ(mkdir(toolDir.c_str(), 0755), 0) << "mkdir sub dir";
    CreateMockHccnTool(toolDir, 1, "ipaddr: 10.10.21.2");

    const std::string confPath = baseDir + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.99"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    const auto ret = GetDeviceUrmaIpAddrFromSources(toolDir + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 2); // tool wins over config

    RemoveDirRecursive(baseDir);
}

TEST(DeviceUrmaEidReaderTest, LegacyEnvVarDoesNotAffectResult)
{
    // Legacy env var MF_DEVICE_URMA_EID_FILE is no longer read;
    // set it to a nonsense path to verify it has no effect.
    EnvVarGuard envGuard("MF_DEVICE_URMA_EID_FILE");
    setenv("MF_DEVICE_URMA_EID_FILE", "/nonexistent/eid_file", 1);

    EidReaderTempDir tmp;
    ASSERT_TRUE(tmp.IsCreated()) << "Failed to create temp dir: " << tmp.path();
    CreateMockHccnTool(tmp.path(), 1, "ipaddr: 10.10.21.2");
    const std::string confPath = tmp.path() + "/hccn.conf";
    CreateHccnConf(confPath, {"address_1=10.10.21.99"});

    CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
    std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> addrData{};
    // Internal seam does not read env var; result purely from injected paths
    const auto ret = GetDeviceUrmaIpAddrFromSources(tmp.path() + "/hccn_tool", confPath, 1, 0, addrType, addrData);
    EXPECT_EQ(ret, BM_OK);
    EXPECT_EQ(addrType, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(addrData[0], 10);
    EXPECT_EQ(addrData[1], 10);
    EXPECT_EQ(addrData[2], 21);
    EXPECT_EQ(addrData[3], 2);
}
