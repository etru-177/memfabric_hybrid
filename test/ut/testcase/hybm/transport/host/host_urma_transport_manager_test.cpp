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

#include <gtest/gtest.h>

#define private public
#include "host_urma_transport_manager.h"
#undef private

using namespace ock::mf;
using namespace ock::mf::transport;
using namespace ock::mf::transport::host;

namespace {
constexpr uint32_t REMOTE_RANK = 1U;
constexpr uint32_t NEW_REMOTE_RANK = 2U;
constexpr ChannelHandle TEST_CHANNEL = 7U;
const EndpointHandle TEST_ENDPOINT = reinterpret_cast<EndpointHandle>(0xA501UL);
constexpr uint64_t LOCAL_ADDR = 0x100000UL;
constexpr uint64_t REMOTE_ADDR = 0x200000UL;
constexpr uint64_t HCOMM_ADDR = 0x300000UL;
constexpr uint64_t TEST_SIZE = 576UL;
constexpr int32_t HCOMM_E_AGAIN = 20;
uint32_t g_writeCount = 0;
uint32_t g_fenceCount = 0;
bool g_channelCreated = false;

struct HcommSubmitGuard {
    hcommWriteOnThreadFunc oldWrite{DlHcommApi::gHcommWriteOnThread};
    hcommChannelFenceOnThreadFunc oldFence{DlHcommApi::gHcommChannelFenceOnThread};

    ~HcommSubmitGuard()
    {
        DlHcommApi::gHcommWriteOnThread = oldWrite;
        DlHcommApi::gHcommChannelFenceOnThread = oldFence;
    }
};

struct HcommPrepareGuard {
    hcommChannelCreateFunc oldChannelCreate{DlHcommApi::gHcommChannelCreate};
    hcommChannelGetStatusFunc oldChannelGetStatus{DlHcommApi::gHcommChannelGetStatus};

    ~HcommPrepareGuard()
    {
        DlHcommApi::gHcommChannelCreate = oldChannelCreate;
        DlHcommApi::gHcommChannelGetStatus = oldChannelGetStatus;
    }
};

struct HcommImportGuard {
    hcommMemImportFunc oldImport{DlHcommApi::gHcommMemImport};

    ~HcommImportGuard()
    {
        DlHcommApi::gHcommMemImport = oldImport;
    }
};

UrmaEndpointDesc MakeEndpointDesc()
{
    UrmaEndpointDesc desc{};
    desc.protocol = UrmaProtocol::UBC_CTP;
    desc.type = COMM_ADDR_TYPE_EID;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    desc.loc.host.id = REMOTE_RANK;
    desc.raws[0] = 1U;
    return desc;
}

TransportPrivateData MakePrivateData(const UrmaEndpointDesc &desc)
{
    TransportPrivateData data{};
    EXPECT_EQ(urma::SerializeUrmaPrivateData(desc, data), BM_OK);
    return data;
}

TransportMemoryKey MakeDeviceMemoryKey()
{
    TransportMemoryKey key{};
    key.keys[0] = urma::URMA_EXPORT_DESC_MAGIC;
    key.keys[1] = REMOTE_ADDR;
    urma::UrmaExportDesc exportDesc{};
    exportDesc.headerSize = sizeof(urma::UrmaExportDesc);
    exportDesc.memoryType = UrmaMemoryType::DEVICE_HBM;
    exportDesc.addr = REMOTE_ADDR;
    exportDesc.size = TEST_SIZE;
    exportDesc.hcommDescLen = 1U;
    auto *payload = reinterpret_cast<uint8_t *>(&key.keys[urma::DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    std::memcpy(payload, &exportDesc, sizeof(exportDesc));
    return key;
}

int32_t CreateChannel(EndpointHandle endpoint, CommEngine engine, HcommChannelDesc *, uint32_t count,
                      ChannelHandle *channel)
{
    EXPECT_EQ(endpoint, TEST_ENDPOINT);
    EXPECT_EQ(engine, COMM_ENGINE_CPU);
    EXPECT_EQ(count, 1U);
    EXPECT_NE(channel, nullptr);
    g_channelCreated = true;
    *channel = TEST_CHANNEL;
    return 0;
}

int32_t ChannelReady(const ChannelHandle *channels, uint32_t count, int32_t *statuses)
{
    EXPECT_EQ(count, 1U);
    EXPECT_EQ(channels[0], TEST_CHANNEL);
    *statuses = 0;
    return 0;
}

int32_t ImportDeviceMemory(EndpointHandle endpoint, const void *, uint32_t descLen, HcommCommMem *memory)
{
    EXPECT_EQ(endpoint, TEST_ENDPOINT);
    EXPECT_EQ(descLen, 1U);
    memory->addr = reinterpret_cast<void *>(HCOMM_ADDR);
    memory->size = TEST_SIZE;
    return 0;
}

int32_t WriteAgainThenSuccess(ThreadHandle, ChannelHandle channel, void *dst, const void *src, uint64_t size)
{
    EXPECT_EQ(channel, TEST_CHANNEL);
    EXPECT_EQ(dst, reinterpret_cast<void *>(HCOMM_ADDR));
    EXPECT_EQ(src, reinterpret_cast<const void *>(LOCAL_ADDR));
    EXPECT_EQ(size, TEST_SIZE);
    return g_writeCount++ == 0 ? HCOMM_E_AGAIN : 0;
}

int32_t FenceSuccess(ThreadHandle, ChannelHandle channel)
{
    EXPECT_EQ(channel, TEST_CHANNEL);
    ++g_fenceCount;
    return 0;
}
} // namespace

TEST(HostUrmaTransportManagerTest, WriteRemoteRetriesAfterQueueFull)
{
    HcommSubmitGuard guard;
    DlHcommApi::gHcommWriteOnThread = WriteAgainThenSuccess;
    DlHcommApi::gHcommChannelFenceOnThread = FenceSuccess;
    g_writeCount = 0;
    g_fenceCount = 0;

    HostUrmaTransportManager manager;
    auto &state = manager.remoteRanks_[REMOTE_RANK];
    state.channel = TEST_CHANNEL;
    HostUrmaTransportManager::RemoteRegistration registration{};
    registration.exportedAddr = REMOTE_ADDR;
    registration.size = TEST_SIZE;
    registration.view.addr = HCOMM_ADDR;
    registration.view.size = TEST_SIZE;
    state.imports.push_back(registration);

    EXPECT_EQ(manager.WriteRemoteAsync(REMOTE_RANK, LOCAL_ADDR, REMOTE_ADDR, TEST_SIZE), BM_OK);
    EXPECT_EQ(g_writeCount, 2U);
    EXPECT_EQ(g_fenceCount, 1U);
    EXPECT_TRUE(state.pending);
}

TEST(HostUrmaTransportManagerTest, QueryHasRegisteredAcceptsSubrange)
{
    HostUrmaTransportManager manager;
    HostUrmaTransportManager::LocalRegistration registration{};
    registration.mr.addr = LOCAL_ADDR;
    registration.mr.size = TEST_SIZE;
    manager.localRegistrations_.emplace(LOCAL_ADDR, registration);

    EXPECT_TRUE(manager.QueryHasRegistered(LOCAL_ADDR + 64U, TEST_SIZE - 64U));
    EXPECT_FALSE(manager.QueryHasRegistered(LOCAL_ADDR + 64U, TEST_SIZE));
}

TEST(HostUrmaTransportManagerTest, UpdateRankOptionsFallsBackToPrepareForNewPeerWithoutHoldingLock)
{
    HcommPrepareGuard guard;
    DlHcommApi::gHcommChannelCreate = CreateChannel;
    DlHcommApi::gHcommChannelGetStatus = ChannelReady;

    HostUrmaTransportManager manager;
    manager.opened_ = true;
    manager.rankId_ = 0U;
    manager.rankCount_ = 3U;
    manager.localEndpoint_ = std::make_shared<urma::UrmaEndpointEntity>();
    manager.localEndpoint_->hcommEndpoint = TEST_ENDPOINT;
    g_channelCreated = false;

    auto endpoint = MakeEndpointDesc();
    endpoint.loc.host.id = NEW_REMOTE_RANK;
    HybmTransPrepareOptions options{};
    options.options[NEW_REMOTE_RANK].privateData = MakePrivateData(endpoint);

    EXPECT_EQ(manager.UpdateRankOptions(options), BM_OK);
    EXPECT_TRUE(g_channelCreated);
    ASSERT_EQ(manager.remoteRanks_.count(NEW_REMOTE_RANK), 1U);
    EXPECT_EQ(manager.remoteRanks_[NEW_REMOTE_RANK].channel, TEST_CHANNEL);

    manager.opened_ = false;
}

TEST(HostUrmaTransportManagerTest, UpdateRankOptionsProcessesExistingDevicePeerMemoryKeysThroughPrepare)
{
    HcommImportGuard guard;
    DlHcommApi::gHcommMemImport = ImportDeviceMemory;
    HostUrmaTransportManager manager;
    auto endpoint = MakeEndpointDesc();
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    auto &state = manager.remoteRanks_[REMOTE_RANK];
    state.endpointDesc = endpoint;
    state.channel = TEST_CHANNEL;
    manager.opened_ = true;
    manager.localEndpoint_ = std::make_shared<urma::UrmaEndpointEntity>();
    manager.localEndpoint_->hcommEndpoint = TEST_ENDPOINT;
    manager.localEndpoint_->desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    HybmTransPrepareOptions options{};
    options.options[REMOTE_RANK].privateData = MakePrivateData(endpoint);
    options.options[REMOTE_RANK].memKeys.emplace_back(MakeDeviceMemoryKey());

    EXPECT_EQ(manager.UpdateRankOptions(options), BM_OK);
    ASSERT_EQ(state.imports.size(), 1U);
    EXPECT_EQ(state.imports[0].exportedAddr, REMOTE_ADDR);
    EXPECT_EQ(state.imports[0].view.addr, HCOMM_ADDR);
    EXPECT_EQ(state.imports[0].view.type, UrmaMemoryType::DEVICE_HBM);
    manager.opened_ = false;
}

TEST(HostUrmaTransportManagerTest, UpdateRankOptionsRejectsExistingPeerEndpointChange)
{
    HostUrmaTransportManager manager;
    auto endpoint = MakeEndpointDesc();
    auto &state = manager.remoteRanks_[REMOTE_RANK];
    state.endpointDesc = endpoint;
    state.channel = TEST_CHANNEL;

    auto changedEndpoint = endpoint;
    changedEndpoint.raws[0]++;
    HybmTransPrepareOptions changedOptions{};
    changedOptions.options[REMOTE_RANK].privateData = MakePrivateData(changedEndpoint);
    changedOptions.options[REMOTE_RANK].privateData.ip[0] = '1';
    EXPECT_EQ(manager.UpdateRankOptions(changedOptions), BM_NOT_SUPPORTED);
}
