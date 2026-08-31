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

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#define private   public
#define protected public
#include "urma/hcomm_transport_manager.h"
#undef private
#undef protected

using namespace ock::mf;
using namespace ock::mf::transport;
using namespace ock::mf::transport::urma;

namespace {
const EndpointHandle MOCK_ENDPOINT = reinterpret_cast<EndpointHandle>(0xA501UL);
const HcommMemHandle MOCK_MEM_HANDLE = reinterpret_cast<HcommMemHandle>(0xA502UL);
constexpr uint64_t MOCK_LOCAL_ADDR = 0x100000UL;
constexpr uint64_t MOCK_REMOTE_ADDR = 0x200000UL;
constexpr uint64_t MOCK_SIZE = 0x1000UL;
constexpr uint64_t MOCK_MEM_TAG = 7UL;
constexpr uint32_t MOCK_HCOMM_DESC_LEN = 4U;
uint32_t g_memExportCallCount = 0;
uint32_t g_memUnregCallCount = 0;

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

UrmaEndpointDesc MakeEndpointDesc()
{
    UrmaEndpointDesc desc{};
    desc.protocol = UrmaProtocol::UBC_TP;
    desc.type = COMM_ADDR_TYPE_EID;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.loc.device.devPhyId = 2UL;
    desc.loc.device.superDevId = 2UL;
    desc.loc.device.serverIdx = 3UL;
    desc.loc.device.superPodIdx = 4UL;
    for (uint32_t i = 0; i < COMM_ADDR_EID_LEN; ++i) {
        desc.raws[i] = static_cast<uint8_t>(i + 1);
    }
    return desc;
}

UrmaEndpointDesc MakeHostEndpointDesc()
{
    auto desc = MakeEndpointDesc();
    desc.loc = {};
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    desc.loc.host.id = 17U;
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

int32_t MockHcommEndpointDestroy(EndpointHandle endpoint)
{
    EXPECT_EQ(endpoint, MOCK_ENDPOINT);
    return BM_OK;
}

int32_t MockHcommEndpointCreateFail(const EndpointDesc *, EndpointHandle *)
{
    return BM_DL_FUNCTION_FAILED;
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
} // namespace

// ============================================================================
// HcommTransportManager tests
// ============================================================================

TEST(UrmaTransportCommonTest, PrivateDataV2RoundTripsDeviceAndHostLocations)
{
    for (const auto &expected : {MakeEndpointDesc(), MakeHostEndpointDesc()}) {
        TransportPrivateData privateData{};
        ASSERT_EQ(SerializeUrmaPrivateData(expected, privateData), BM_OK);

        UrmaEndpointDesc actual{};
        ASSERT_EQ(ParseUrmaPrivateData(privateData, actual), BM_OK);
        EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(expected)), 0);

        const auto hcommDesc = ToHcommEndpointDesc(actual);
        EXPECT_EQ(std::memcmp(&hcommDesc.loc, &expected.loc, sizeof(expected.loc)), 0);
    }
}

TEST(UrmaTransportCommonTest, PrivateDataRejectsLegacyVersionAndInvalidEndpoint)
{
    TransportPrivateData privateData{};
    ASSERT_EQ(SerializeUrmaPrivateData(MakeEndpointDesc(), privateData), BM_OK);
    auto *header = reinterpret_cast<UrmaPrivateDataDesc *>(privateData.key.keys);
    header->version = 1U;

    UrmaEndpointDesc endpoint{};
    EXPECT_EQ(ParseUrmaPrivateData(privateData, endpoint), BM_INVALID_PARAM);

    auto invalidEndpoint = MakeEndpointDesc();
    invalidEndpoint.loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
    EXPECT_EQ(SerializeUrmaPrivateData(invalidEndpoint, privateData), BM_INVALID_PARAM);
}

TEST(UrmaTransportCommonTest, WireDescriptorsFitTransportKeys)
{
    EXPECT_EQ(sizeof(UrmaExportDesc), 48U);
    EXPECT_LE(sizeof(UrmaPrivateDataDesc) + sizeof(UrmaEndpointDesc), sizeof(TransportPrivateData{}.key.keys));
    EXPECT_LE(sizeof(UrmaExportDesc), DEVICE_URMA_EXPORT_KEY_DATA_BYTES);
}

TEST(HcommTransportManagerTest, CreateEndpointCallsHcommEndpointCreate)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    EXPECT_NE(endpoint, nullptr);
}

TEST(HcommTransportManagerTest, CreateEndpointReturnsNullOnHcommFailure)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreateFail;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    EXPECT_EQ(endpoint, nullptr);
}

TEST(HcommTransportManagerTest, RegisterLocalMemoryCallsHcommMemReg)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    EXPECT_EQ(manager.HcommMemReg(endpoint, 7, mem, &memHandle), BM_OK);
    EXPECT_EQ(memHandle, MOCK_MEM_HANDLE);
}

TEST(HcommTransportManagerTest, DeregisterLocalMemoryCallsHcommMemUnreg)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnregCounted;
    g_memUnregCallCount = 0;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    EXPECT_EQ(manager.HcommMemReg(endpoint, 7, mem, &memHandle), BM_OK);
    ASSERT_EQ(memHandle, MOCK_MEM_HANDLE);

    EXPECT_EQ(manager.HcommMemUnreg(endpoint, memHandle), BM_OK);
}

TEST(HcommTransportManagerTest, ExportLocalMemoryCallsHcommMemExport)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;
    DlHcommApi::gHcommMemExport = MockHcommMemExport;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    EXPECT_EQ(manager.HcommMemReg(endpoint, 7, mem, &memHandle), BM_OK);
    ASSERT_EQ(memHandle, MOCK_MEM_HANDLE);

    const uint8_t *memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(manager.HcommMemExport(endpoint, memHandle, &memDesc, &memDescLen), BM_OK);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_GT(memDescLen, 0U);
}

TEST(HcommTransportManagerTest, ImportAndUnimportMemoryWorkflow)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;
    DlHcommApi::gHcommMemExport = MockHcommMemExport;
    DlHcommApi::gHcommMemImport = MockHcommMemImport;
    DlHcommApi::gHcommMemUnimport = MockHcommMemUnimport;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    EXPECT_EQ(manager.HcommMemReg(endpoint, 7, mem, &memHandle), BM_OK);
    ASSERT_EQ(memHandle, MOCK_MEM_HANDLE);

    const uint8_t *exportDesc = nullptr;
    uint32_t exportDescLen = 0;
    EXPECT_EQ(manager.HcommMemExport(endpoint, memHandle, &exportDesc, &exportDescLen), BM_OK);
    ASSERT_NE(exportDesc, nullptr);
    ASSERT_GT(exportDescLen, sizeof(UrmaExportDesc));

    UrmaCommMem importedView{};
    EXPECT_EQ(manager.HcommMemImport(endpoint, exportDesc, exportDescLen, &importedView), BM_OK);
    EXPECT_EQ(importedView.addr, MOCK_REMOTE_ADDR);
    EXPECT_EQ(importedView.size, MOCK_SIZE);
    EXPECT_EQ(importedView.type, UrmaMemoryType::HOST_DRAM);

    EXPECT_EQ(manager.HcommMemUnimport(endpoint, exportDesc, exportDescLen), BM_OK);
}

TEST(HcommTransportManagerTest, HcommMemRegRejectsInvalidInputsAndOverlaps)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;
    DlHcommApi::gHcommMemUnreg = MockHcommMemUnreg;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    EXPECT_EQ(manager.HcommMemReg(nullptr, MOCK_MEM_TAG, mem, &memHandle), BM_INVALID_PARAM);
    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG, mem, nullptr), BM_INVALID_PARAM);
    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG, {0, MOCK_SIZE, UrmaMemoryType::HOST_DRAM}, &memHandle),
              BM_INVALID_PARAM);

    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG, mem, &memHandle), BM_OK);
    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG, mem, &memHandle), BM_OK);
    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG,
                                  {MOCK_LOCAL_ADDR + MOCK_SIZE, MOCK_SIZE, UrmaMemoryType::HOST_DRAM}, &memHandle),
              BM_ERROR);
    EXPECT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG + 1,
                                  {MOCK_LOCAL_ADDR + 1, MOCK_SIZE, UrmaMemoryType::HOST_DRAM}, &memHandle),
              BM_ERROR);

    EXPECT_EQ(manager.HcommMemUnreg(endpoint, MOCK_MEM_HANDLE), BM_OK);
    EXPECT_EQ(manager.HcommMemUnreg(endpoint, MOCK_MEM_HANDLE), BM_OK);
    EXPECT_GE(g_memUnregCallCount, 1U);
}

TEST(HcommTransportManagerTest, HcommMemExportCachesHcommDescriptor)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;
    DlHcommApi::gHcommMemReg = MockHcommMemReg;
    DlHcommApi::gHcommMemExport = MockHcommMemExportCounted;

    g_memExportCallCount = 0;
    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    HcommMemHandle memHandle = nullptr;
    const UrmaCommMem mem{MOCK_LOCAL_ADDR, MOCK_SIZE, UrmaMemoryType::HOST_DRAM};
    ASSERT_EQ(manager.HcommMemReg(endpoint, MOCK_MEM_TAG, mem, &memHandle), BM_OK);

    const uint8_t *firstDesc = nullptr;
    const uint8_t *secondDesc = nullptr;
    uint32_t firstLen = 0;
    uint32_t secondLen = 0;
    EXPECT_EQ(manager.HcommMemExport(endpoint, memHandle, &firstDesc, &firstLen), BM_OK);
    EXPECT_EQ(manager.HcommMemExport(endpoint, memHandle, &secondDesc, &secondLen), BM_OK);
    EXPECT_EQ(g_memExportCallCount, 1U);
    EXPECT_EQ(firstDesc, secondDesc);
    EXPECT_EQ(firstLen, secondLen);
}

TEST(HcommTransportManagerTest, HcommMemImportRejectsMalformedDescAndUsesLocalEndpointType)
{
    DlHcommApiFnGuard guard;
    DlHcommApi::gHcommEndpointCreate = MockHcommEndpointCreate;
    DlHcommApi::gHcommEndpointDestroy = MockHcommEndpointDestroy;

    HcommTransportManager manager;
    const auto endpoint = manager.CreateEndpoint(MakeEndpointDesc());
    ASSERT_NE(endpoint, nullptr);

    UrmaCommMem imported{};
    const auto rawDesc = MakeRawExportDesc();
    EXPECT_EQ(manager.HcommMemImport(nullptr, rawDesc.data(), static_cast<uint32_t>(rawDesc.size()), &imported),
              BM_INVALID_PARAM);
    EXPECT_EQ(manager.HcommMemImport(endpoint, nullptr, static_cast<uint32_t>(rawDesc.size()), &imported),
              BM_INVALID_PARAM);
    EXPECT_EQ(manager.HcommMemImport(endpoint, rawDesc.data(), sizeof(UrmaExportDesc) - 1, &imported),
              BM_INVALID_PARAM);

    DlHcommApi::gHcommMemImport = MockHcommMemImportInvalidType;
    EXPECT_EQ(manager.HcommMemImport(endpoint, rawDesc.data(), static_cast<uint32_t>(rawDesc.size()), &imported),
              BM_OK);
    EXPECT_EQ(imported.type, UrmaMemoryType::DEVICE_HBM);

    DlHcommApi::gHcommMemImport = MockHcommMemImportFail;
    EXPECT_EQ(manager.HcommMemImport(endpoint, rawDesc.data(), static_cast<uint32_t>(rawDesc.size()), &imported),
              BM_DL_FUNCTION_FAILED);
}
