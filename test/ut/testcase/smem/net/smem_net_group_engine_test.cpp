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
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "smem_net_group_engine.h"
#undef private
#include "smem_store_factory.h"

using namespace mockcpp;

class ConfigStoreManagerMock : public ock::smem::ConfigStoreManager {
public:
    ConfigStoreManagerMock() = default;
    ~ConfigStoreManagerMock() override = default;

    ock::smem::Result Set(const std::string &key, const std::vector<uint8_t> &value) noexcept override
    {
        setCount++;
        return setResult_;
    }

    ock::smem::Result GetReal(const std::string &key, std::vector<uint8_t> &value, int64_t timeoutMs) noexcept override
    {
        getCount++;
        if (getResult_ == ock::smem::SM_OK && !mockGetValue.empty()) {
            value = mockGetValue;
        }
        return getResult_;
    }

    ock::smem::Result QueryAlive(uint32_t rank, uint32_t &alive) noexcept override
    {
        alive = true;
        return 0;
    }

    ock::smem::Result PrefixGet(const std::string &key,
                                std::unordered_map<std::string, std::string> &value) noexcept override
    {
        return 0;
    }

    ock::smem::Result Add(const std::string &key, int64_t increment, int64_t &value) noexcept override
    {
        addCount++;
        value = addValue;
        return addResult_;
    }

    ock::smem::Result Remove(const std::string &key, bool printKeyNotExist) noexcept override
    {
        removeCount++;
        return removeResult_;
    }

    ock::smem::Result Append(const std::string &key, const std::vector<uint8_t> &value,
                             uint64_t &newSize) noexcept override
    {
        appendCount++;
        newSize = appendNewSize;
        return appendResult_;
    }

    ock::smem::Result Cas(const std::string &key, const std::vector<uint8_t> &expect, const std::vector<uint8_t> &value,
                          std::vector<uint8_t> &exists) noexcept override
    {
        casCount++;
        return casResult_;
    }

    ock::smem::Result
    Watch(const std::string &key,
          const std::function<void(int result, const std::string &, const std::vector<uint8_t> &)> &notify,
          uint32_t &wid) noexcept override
    {
        watchCount++;
        wid = watchWid;
        return watchResult_;
    }

    ock::smem::Result Watch(ock::smem::WatchRankType type,
                            const std::function<void(ock::smem::WatchRankType, uint32_t)> &notify,
                            uint32_t &wid) noexcept override
    {
        watchRankCount++;
        wid = watchWid;
        return watchResult_;
    }

    ock::smem::Result Unwatch(uint32_t wid) noexcept override
    {
        unwatchCount++;
        return unwatchResult_;
    }

    ock::smem::Result Write(const std::string &key, const std::vector<uint8_t> &value,
                            const uint32_t offset) noexcept override
    {
        writeCount++;
        return writeResult_;
    }

    std::string GetCompleteKey(const std::string &key) noexcept override
    {
        getCompleteKeyCount++;
        return getCompleteKeyValue;
    }

    std::string GetCommonPrefix() noexcept override
    {
        return commonPrefix;
    }

    ock::smem::SmRef<ock::smem::ConfigStore> GetCoreStore() noexcept override
    {
        return ock::smem::SmRef<ock::smem::ConfigStore>(this);
    }

    void RegisterReconnectHandler(ock::smem::ConfigStoreReconnectHandler callback) noexcept override
    {
        reconnectHandler = callback;
    }
    ock::smem::Result ReConnectAfterBroken(int reconnectRetryTimes) noexcept override
    {
        return ock::smem::SM_OK;
    }
    bool GetConnectStatus() noexcept override
    {
        if (connectStatusTrueCount > 0) {
            connectStatusTrueCount--;
            return true;
        }
        return connectStatus;
    }
    void SetConnectStatus(bool status) noexcept override
    {
        connectStatus = status;
    }
    void RegisterClientBrokenHandler(const ock::smem::ConfigStoreClientBrokenHandler &handler) noexcept override
    {
        clientBrokenHandlerCount++;
    }
    void RegisterServerBrokenHandler(const ock::smem::ConfigStoreServerBrokenHandler &handler) noexcept override {}

    uint64_t setCount{0};
    uint64_t getCount{0};
    uint64_t addCount{0};
    uint64_t removeCount{0};
    uint64_t appendCount{0};
    uint64_t casCount{0};
    uint64_t watchCount{0};
    uint64_t watchRankCount{0};
    uint64_t unwatchCount{0};
    uint64_t writeCount{0};
    uint64_t getCompleteKeyCount{0};
    uint64_t clientBrokenHandlerCount{0};

    ock::smem::Result setResult_{ock::smem::SM_OK};
    ock::smem::Result getResult_{ock::smem::SM_OK};
    ock::smem::Result addResult_{ock::smem::SM_OK};
    ock::smem::Result removeResult_{ock::smem::SM_OK};
    ock::smem::Result appendResult_{ock::smem::SM_OK};
    ock::smem::Result casResult_{ock::smem::SM_OK};
    ock::smem::Result watchResult_{ock::smem::SM_OK};
    ock::smem::Result unwatchResult_{ock::smem::SM_OK};
    ock::smem::Result writeResult_{ock::smem::SM_OK};

    bool connectStatus{true};
    size_t connectStatusTrueCount{0};

    int64_t addValue{0};
    uint64_t appendNewSize{0};
    uint32_t watchWid{0};
    std::string getCompleteKeyValue{"test_key"};
    std::string commonPrefix{"/test/"};
    std::vector<uint8_t> mockGetValue;
    ock::smem::ConfigStoreReconnectHandler reconnectHandler{nullptr};

    void Reset()
    {
        setCount = 0;
        getCount = 0;
        addCount = 0;
        removeCount = 0;
        appendCount = 0;
        casCount = 0;
        watchCount = 0;
        watchRankCount = 0;
        unwatchCount = 0;
        writeCount = 0;
        getCompleteKeyCount = 0;
        clientBrokenHandlerCount = 0;

        setResult_ = ock::smem::SM_OK;
        getResult_ = ock::smem::SM_OK;
        addResult_ = ock::smem::SM_OK;
        removeResult_ = ock::smem::SM_OK;
        appendResult_ = ock::smem::SM_OK;
        casResult_ = ock::smem::SM_OK;
        watchResult_ = ock::smem::SM_OK;
        unwatchResult_ = ock::smem::SM_OK;
        writeResult_ = ock::smem::SM_OK;

        connectStatus = true;
        connectStatusTrueCount = 0;

        addValue = 0;
        appendNewSize = 0;
        watchWid = 0;
        getCompleteKeyValue = "test_key";
        mockGetValue.clear();
        reconnectHandler = nullptr;
    }
};

class SmemNetGroupEngineMockTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        mockStoreManager_ = new ConfigStoreManagerMock();
        storePtr_ = ock::smem::StorePtr(mockStoreManager_);

        option_.rankSize = 3UL;
        option_.rank = 0;
        option_.timeoutMs = 1000ULL;
        option_.dynamic = false;
        option_.joinCb = nullptr;
        option_.updateCb = nullptr;
        option_.leaveCb = nullptr;
        option_.linkDownCb = nullptr;
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        mockStoreManager_->Reset();
        storePtr_ = nullptr;
    }

    ConfigStoreManagerMock *mockStoreManager_;
    ock::smem::StorePtr storePtr_;
    ock::smem::SmemGroupOption option_;
};

TEST_F(SmemNetGroupEngineMockTest, Create)
{
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    EXPECT_NE(group, nullptr);
}

TEST_F(SmemNetGroupEngineMockTest, IsJoined)
{
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    EXPECT_NE(group, nullptr);

    EXPECT_TRUE(group->IsJoined());
}

TEST_F(SmemNetGroupEngineMockTest, GetLocalRank)
{
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    EXPECT_NE(group, nullptr);

    EXPECT_EQ(group->GetLocalRank(), option_.rank);
}

TEST_F(SmemNetGroupEngineMockTest, GetRankSize)
{
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    EXPECT_NE(group, nullptr);

    EXPECT_EQ(group->GetRankSize(), option_.rankSize);
}

TEST_F(SmemNetGroupEngineMockTest, GroupSnClean)
{
    mockStoreManager_->removeResult_ = ock::smem::SM_OK;

    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    EXPECT_NE(group, nullptr);

    group->GroupSnClean();
}

// === Tests for commit dc686c2 new functionality ===

TEST_F(SmemNetGroupEngineMockTest, GetStoreConnectStatusReturnsStoreStatus)
{
    mockStoreManager_->getResult_ = ock::smem::SM_OK;

    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    ASSERT_NE(group, nullptr);

    // GetStoreConnectStatus delegates to store_->GetConnectStatus().
    EXPECT_TRUE(group->GetStoreConnectStatus());
}

TEST_F(SmemNetGroupEngineMockTest, ReconnectHandlerSkipsAfterGroupDestroyed)
{
    option_.dynamic = true;
    mockStoreManager_->watchWid = UINT32_MAX;

    {
        auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
        ASSERT_NE(group, nullptr);
        ASSERT_NE(mockStoreManager_->reconnectHandler, nullptr);
    }

    EXPECT_EQ(mockStoreManager_->reconnectHandler(), ock::smem::SM_OK);
    EXPECT_EQ(mockStoreManager_->casCount, 0);
}

// === Tests for "fix exit problem when link is broken" (GroupLeave abort paths) ===
// watchWid = UINT32_MAX keeps the listener threads looping on ReWatchEvent (no Cas)
// so casCount only reflects what GroupLeave itself submits.

// Branch 1: the config store link is already broken when GroupLeave starts.
// The Cas retry loop must be skipped entirely — no Cas call, fast SM_ERROR return.
TEST_F(SmemNetGroupEngineMockTest, GroupLeaveReturnsErrorWhenStoreAlreadyDisconnected)
{
    option_.dynamic = true;
    mockStoreManager_->watchWid = UINT32_MAX;
    mockStoreManager_->connectStatus = false; // store already disconnected

    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    ASSERT_NE(group, nullptr);
    group->joined_ = true; // simulate an already-joined dynamic group

    EXPECT_EQ(group->GroupLeave(), ock::smem::SM_ERROR);
    EXPECT_FALSE(group->joined_.load());
    EXPECT_EQ(mockStoreManager_->casCount, 0);
}

// Branch 2: groupStoped_ is set (e.g. the destructor is racing with GroupLeave).
// The retry loop must abort on the first iteration, before any Cas call.
TEST_F(SmemNetGroupEngineMockTest, GroupLeaveAbortsWhenGroupStoppedDuringLoop)
{
    option_.dynamic = true;
    mockStoreManager_->watchWid = UINT32_MAX;

    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    ASSERT_NE(group, nullptr);
    group->joined_ = true;
    group->groupStoped_ = true;

    EXPECT_EQ(group->GroupLeave(), ock::smem::SM_ERROR);
    EXPECT_FALSE(group->joined_.load());
    EXPECT_EQ(mockStoreManager_->casCount, 0);
}

// Regression guard: bm scenario (useClientBrokenHandler=false) must NOT register
// ClientBrokenHandler. Otherwise a server link break triggers LeaveHandle on every
// peer rank, destroying GVA imports and causing -6 read failures after meta recovery.
TEST_F(SmemNetGroupEngineMockTest, CreateDoesNotRegisterClientBrokenHandlerByDefault)
{
    option_.useClientBrokenHandler = false;
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(mockStoreManager_->clientBrokenHandlerCount, 0U);
}

TEST_F(SmemNetGroupEngineMockTest, CreateRegistersClientBrokenHandlerWhenUseClientBrokenHandlerIsTrue)
{
    option_.useClientBrokenHandler = true;
    option_.linkDownCb = [](uint32_t) { return ock::smem::SM_OK; };
    auto group = ock::smem::SmemNetGroupEngine::Create(storePtr_, option_);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(mockStoreManager_->clientBrokenHandlerCount, 1U);
}
