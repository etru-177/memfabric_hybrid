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
#include <vector>

#include "hybm_data_op.h"
#include "hybm_common_include.h"
#include "hybm_entity_factory.h"
#include "hybm_va_manager.h"
#include "hybm_ex_info_transfer.h"

using namespace ock::mf;

inline void RegisterMockEntity(hybm_entity_t entity, EngineImplPtr mockEntity)
{
    static std::map<hybm_entity_t, EngineImplPtr> &mockEntities = *new std::map<hybm_entity_t, EngineImplPtr>();
    mockEntities[entity] = mockEntity;
}

inline void ClearMockEntities()
{
    static std::map<hybm_entity_t, EngineImplPtr> &mockEntities = *new std::map<hybm_entity_t, EngineImplPtr>();
    mockEntities.clear();
}

class MockMemEntity : public MemEntityDefault {
public:
    explicit MockMemEntity(int32_t id = 0) noexcept : MemEntityDefault(id) {}
    ~MockMemEntity() override = default;

    int32_t Initialize(const hybm_options *options) noexcept override
    {
        return BM_OK;
    }
    void UnInitialize() noexcept override {}

    int32_t ReserveMemorySpace() noexcept override
    {
        return BM_OK;
    }
    int32_t UnReserveMemorySpace() noexcept override
    {
        return BM_OK;
    }
    void *GetReservedMemoryPtr(hybm_mem_type memType) noexcept override
    {
        return nullptr;
    }

    int32_t AllocLocalMemory(uint64_t size, hybm_mem_type mType, uint32_t flags,
                             hybm_mem_slice_t &slice) noexcept override
    {
        return BM_OK;
    }
    int32_t RegisterLocalMemory(const void *ptr, uint64_t size, uint32_t flags,
                                hybm_mem_slice_t &slice) noexcept override
    {
        return BM_OK;
    }
    int32_t FreeLocalMemory(hybm_mem_slice_t slice, uint32_t flags) noexcept override
    {
        return BM_OK;
    }

    int32_t ExportEntityExchangeInfo(ExchangeInfoWriter &desc, uint32_t flags) noexcept override
    {
        return BM_OK;
    }

    int32_t ExportSliceExchangeInfo(hybm_mem_slice_t slice, ExchangeInfoWriter &desc, uint32_t flags) noexcept override
    {
        return BM_OK;
    }
    int32_t ImportSliceExchangeInfo(const ExchangeInfoReader desc[], uint32_t count, void *addresses[],
                                    uint32_t flags) noexcept override
    {
        return BM_OK;
    }
    int32_t ImportEntityExchangeInfo(const ExchangeInfoReader desc[], uint32_t count, uint32_t flags) noexcept override
    {
        return BM_OK;
    }
    int32_t RemoveImported(const std::vector<uint32_t> &ranks) noexcept override
    {
        return BM_OK;
    }
    int32_t SetExtraContext(const void *context, uint32_t size) noexcept override
    {
        return BM_OK;
    }

    int32_t Mmap() noexcept override
    {
        return BM_OK;
    }
    void Unmap() noexcept override {}

    int32_t CopyData(hybm_copy_params &params, hybm_data_copy_direction direction, void *stream,
                     uint32_t flags) noexcept override
    {
        copyCalled = true;
        copyDirection = direction;
        copySize = params.dataSize;
        return BM_OK;
    }

    int32_t BatchCopyData(hybm_batch_copy_params &params, hybm_data_copy_direction direction, void *stream,
                          uint32_t flags) noexcept override
    {
        batchCopyCalled = true;
        batchCopyDirection = direction;
        batchCopySize = params.batchSize;
        return BM_OK;
    }

    int32_t BatchRawCopyData(hybm_batch_raw_copy_params &params,
                             hybm_data_copy_direction direction) noexcept override
    {
        batchRawCopyCalled = true;
        batchRawCopyDirection = direction;
        batchRawCopyRankId = params.rankId;
        batchRawCopySize = params.batchSize;
        return batchRawCopyRet;
    }

    int32_t Wait() noexcept override
    {
        waitCalled = true;
        return BM_OK;
    }

    bool CheckAddressInEntity(const void *ptr, uint64_t length) const noexcept override
    {
        return addressInRange;
    }

    bool SdmaReaches(uint32_t remoteRank) const noexcept override
    {
        return true;
    }

    hybm_data_op_type CanReachDataOperators(uint32_t remoteRank) const noexcept override
    {
        return static_cast<hybm_data_op_type>(0U);
    }

    bool copyCalled = false;
    bool batchCopyCalled = false;
    bool waitCalled = false;
    bool addressInRange = true;
    bool batchRawCopyCalled = false;
    hybm_data_copy_direction copyDirection = HYBM_DATA_COPY_DIRECTION_BUTT;
    hybm_data_copy_direction batchCopyDirection = HYBM_DATA_COPY_DIRECTION_BUTT;
    hybm_data_copy_direction batchRawCopyDirection = HYBM_DATA_COPY_DIRECTION_BUTT;
    uint64_t copySize = 0;
    uint32_t batchCopySize = 0;
    uint32_t batchRawCopyRankId = 0;
    uint32_t batchRawCopySize = 0;
    int32_t batchRawCopyRet = 0;
};

class HybmDataOpEntryTest : public testing::Test {
public:
    void SetUp() override
    {
        GlobalMockObject::reset();
        ClearMockEntities();
        mockEntity = std::make_shared<MockMemEntity>(1); // 使用 id=1 创建实例
        // 注册 mock 实体，使用其指针作为 key
        RegisterMockEntity(mockEntity.get(), mockEntity);

        // 直接将 mock 实体添加到 MemEntityFactory 的映射中
        auto &factory = MemEntityFactory::Instance();
        // 清除现有的映射，避免冲突
        factory.enginesFromAddress_.clear();
        factory.engines_.clear();
        // 添加 mock 实体到映射中
        factory.enginesFromAddress_[mockEntity.get()] = 1; // id=1
        factory.engines_[1] = mockEntity;

        HybmVaManager::InitDirectionLut();

        // 清除 HybmVaManager 的内部状态
        auto &vaManager = HybmVaManager::GetInstance();
        vaManager.ClearAll();
        // 预留 GVM 地址范围
        vaManager.AllocReserveGva(0, HYBM_GVM_MAX_POOL_SIZE, HYBM_GVM_MAX_POOL_SIZE, HYBM_MEM_TYPE_HOST, false);

        // 添加虚拟地址信息（GVM 范围内）
        BaseAllocatedGvaInfo info;
        uint64_t addrs[] = {HYBM_GVM_START_ADDR, HYBM_GVM_START_ADDR + 0x1000, HYBM_GVM_START_ADDR + 0x2000,
                            HYBM_GVM_START_ADDR + 0x3000};
        for (auto a : addrs) {
            info.va[HVM_GVA] = a;
            info.va[HVM_HVA] = a;
            info.size = 4096; // 4096
            info.memType = HYBM_MEM_TYPE_HOST;
            vaManager.AddVaInfoFromExternal(info, 0);
        }
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        ClearMockEntities();

        // 清除 HybmVaManager 的内部状态
        auto &vaManager = HybmVaManager::GetInstance();
        vaManager.ClearAll();
    }

protected:
    std::shared_ptr<MockMemEntity> mockEntity;
};

TEST_F(HybmDataOpEntryTest, hybm_data_copy_success)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->copyCalled);
    EXPECT_EQ(mockEntity->copySize, 1024);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_null_entity)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(nullptr, &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_null_params)
{
    auto ret = hybm_data_copy(mockEntity.get(), nullptr, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_null_src)
{
    hybm_copy_params params{};
    params.src = nullptr;
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_null_dest)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = nullptr;
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_zero_size)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 0;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_invalid_direction)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_BUTT, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_infer_success)
{
    hybm_copy_params params{};
    // src 在 device VA 范围 → ClassifyAddress 返回 LOCAL_DEVICE
    // dest=0x2000 → 返回 LOCAL_HOST
    // 方向表 [2][3] = LOCAL_DEVICE_TO_GLOBAL_HOST(2) → 有效方向
    params.src = reinterpret_cast<void *>(HYBM_DEVICE_VA_START);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->copyCalled);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_all_directions)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    // GVA+GVA(src=LH|GH, dst=LH|GH) 可匹配的方向
    std::vector<hybm_data_copy_direction> directions = {HYBM_LOCAL_HOST_TO_GLOBAL_HOST, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST,
                                                        HYBM_GLOBAL_HOST_TO_LOCAL_HOST};

    for (auto direction : directions) {
        mockEntity->copyCalled = false;
        auto ret = hybm_data_copy(mockEntity.get(), &params, direction, nullptr, 0);
        EXPECT_EQ(ret, 0);
        EXPECT_TRUE(mockEntity->copyCalled);
        EXPECT_EQ(mockEntity->copyDirection, direction);
    }
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_success)
{
    void *sources[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                        reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *destinations[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                             reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_copy_params params{};
    params.sources = sources;
    params.destinations = destinations;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->batchCopyCalled);
    EXPECT_EQ(mockEntity->batchCopyDirection, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(mockEntity->batchCopySize, 2);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_entity)
{
    hybm_batch_copy_params params{};
    auto ret = hybm_data_batch_copy(nullptr, &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_params)
{
    auto ret = hybm_data_batch_copy(mockEntity.get(), nullptr, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_sources)
{
    hybm_batch_copy_params params{};
    params.sources = nullptr;
    params.destinations = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_destinations)
{
    hybm_batch_copy_params params{};
    params.sources = reinterpret_cast<void **>(0x1000);
    params.destinations = nullptr;
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_data_sizes)
{
    hybm_batch_copy_params params{};
    params.sources = reinterpret_cast<void **>(0x1000);
    params.destinations = reinterpret_cast<void **>(0x2000);
    params.dataSizes = nullptr;
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_zero_batch_size)
{
    hybm_batch_copy_params params{};
    params.sources = reinterpret_cast<void **>(0x1000);
    params.destinations = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 0;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_invalid_direction)
{
    hybm_batch_copy_params params{};
    params.sources = reinterpret_cast<void **>(0x1000);
    params.destinations = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_BUTT, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_success)
{
    void *localAddrs[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                           reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *remoteAddrs[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                            reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_raw_copy_params params{};
    params.rankId = 1;
    params.localAddrs = localAddrs;
    params.remoteAddrs = remoteAddrs;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->batchRawCopyCalled);
    EXPECT_EQ(mockEntity->batchRawCopyDirection, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(mockEntity->batchRawCopyRankId, 1U);
    EXPECT_EQ(mockEntity->batchRawCopySize, 2U);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_null_entity)
{
    hybm_batch_raw_copy_params params{};
    auto ret = hybm_data_batch_raw_copy(nullptr, &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_null_params)
{
    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), nullptr, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_null_local_addrs)
{
    hybm_batch_raw_copy_params params{};
    params.localAddrs = nullptr;
    params.remoteAddrs = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_null_remote_addrs)
{
    hybm_batch_raw_copy_params params{};
    params.localAddrs = reinterpret_cast<void **>(0x1000);
    params.remoteAddrs = nullptr;
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_null_data_sizes)
{
    hybm_batch_raw_copy_params params{};
    params.localAddrs = reinterpret_cast<void **>(0x1000);
    params.remoteAddrs = reinterpret_cast<void **>(0x2000);
    params.dataSizes = nullptr;
    params.batchSize = 2;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_zero_batch_size)
{
    hybm_batch_raw_copy_params params{};
    params.localAddrs = reinterpret_cast<void **>(0x1000);
    params.remoteAddrs = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 0;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_raw_copy_invalid_direction)
{
    hybm_batch_raw_copy_params params{};
    params.localAddrs = reinterpret_cast<void **>(0x1000);
    params.remoteAddrs = reinterpret_cast<void **>(0x2000);
    params.dataSizes = reinterpret_cast<uint64_t *>(0x3000);
    params.batchSize = 2;

    auto ret = hybm_data_batch_raw_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_BUTT);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_auto_infer_success)
{
    void *sources[2] = {reinterpret_cast<void *>(HYBM_DEVICE_VA_START),
                        reinterpret_cast<void *>(HYBM_DEVICE_VA_START + 0x2000)};
    void *destinations[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                             reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_copy_params params{};
    params.sources = sources;
    params.destinations = destinations;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->batchCopyCalled);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_null_item)
{
    void *sources[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR), nullptr};
    void *destinations[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                             reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_copy_params params{};
    params.sources = sources;
    params.destinations = destinations;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_wait_success)
{
    auto ret = hybm_wait(mockEntity.get());
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->waitCalled);
}

TEST_F(HybmDataOpEntryTest, hybm_wait_null_entity)
{
    auto ret = hybm_wait(nullptr);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_stream_parameter)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, 0);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_flags_parameter)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0xFFFFFFFF);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_stream_and_flags)
{
    void *sources[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                        reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *destinations[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                             reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_copy_params params{};
    params.sources = sources;
    params.destinations = destinations;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, 0xFFFFFFFF);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_address_alignment)
{
    std::vector<uint64_t> addresses = {0x1000, 0x1001, 0x1002, 0x1003, 0x1004, 0x1008, 0x1010};

    for (auto addr : addresses) {
        hybm_copy_params params{};
        params.src = reinterpret_cast<void *>(addr);
        params.dest = reinterpret_cast<void *>(addr + HYBM_GVM_START_ADDR);
        params.dataSize = 1024;

        auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
        EXPECT_EQ(ret, 0);
    }
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_async_flag)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, ASYNC_COPY_FLAG);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_extend_flag)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, nullptr, COPY_EXTEND_FLAG);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_combined_flags)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream,
                              ASYNC_COPY_FLAG | COPY_EXTEND_FLAG);
    EXPECT_EQ(ret, 0);
}

TEST_F(HybmDataOpEntryTest, hybm_async_copy_with_wait)
{
    // 执行异步复制
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, ASYNC_COPY_FLAG);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->copyCalled);

    // 等待操作完成
    ret = hybm_wait(mockEntity.get());
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->waitCalled);
}

TEST_F(HybmDataOpEntryTest, hybm_batch_copy_with_wait)
{
    // 执行批量复制
    void *sources[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                        reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *destinations[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                             reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t dataSizes[2] = {1024, 2048};

    hybm_batch_copy_params params{};
    params.sources = sources;
    params.destinations = destinations;
    params.dataSizes = dataSizes;
    params.batchSize = 2;

    void *stream = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x4000);
    auto ret = hybm_data_batch_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, ASYNC_COPY_FLAG);
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->batchCopyCalled);

    // 等待操作完成
    ret = hybm_wait(mockEntity.get());
    EXPECT_EQ(ret, 0);
    EXPECT_TRUE(mockEntity->waitCalled);
}

// ===== 新增加：方向校验测试 =====

TEST_F(HybmDataOpEntryTest, hybm_data_copy_direction_mismatch)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_with_gva)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mockEntity->copyDirection, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
}

TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_user_addr_fails)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(0x7f001000);
    params.dest = reinterpret_cast<void *>(0x7f002000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

// 批量拷贝+flags+stream
TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_with_flags)
{
    void *srcs[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *dsts[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t sizes[2] = {4096, 8192};
    hybm_batch_copy_params p{};
    p.sources = srcs;
    p.destinations = dsts;
    p.dataSizes = sizes;
    p.batchSize = 2; // 2
    void *stream = reinterpret_cast<void *>(0x6000);
    auto ret = hybm_data_batch_copy(mockEntity.get(), &p, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, stream, ASYNC_COPY_FLAG);
    EXPECT_EQ(ret, 0);
}

// 批量拷贝 AUTO
TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_auto_gva_pairs)
{
    void *srcs[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *dsts[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t sizes[2] = {1024, 2048};
    hybm_batch_copy_params p{};
    p.sources = srcs;
    p.destinations = dsts;
    p.dataSizes = sizes;
    p.batchSize = 2; // 2
    auto ret = hybm_data_batch_copy(mockEntity.get(), &p, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// 批量拷贝方向不匹配
TEST_F(HybmDataOpEntryTest, hybm_data_batch_copy_direction_mismatch)
{
    void *srcs[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x2000)};
    void *dsts[2] = {reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000),
                     reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x3000)};
    uint64_t sizes[2] = {1024, 2048};
    hybm_batch_copy_params p{};
    p.sources = srcs;
    p.destinations = dsts;
    p.dataSizes = sizes;
    p.batchSize = 2; // 2
    // H2GD 需要 dest=GD，但 dest 是 (LH|GH) → 不匹配
    auto ret = hybm_data_batch_copy(mockEntity.get(), &p, HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

// D2GH 方向通过（LOCAL_DEVICE → GLOBAL_HOST）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_device_to_global_host)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_DEVICE_VA_START + 0x1000);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// GH2LD 方向通过（GLOBAL_HOST → LOCAL_DEVICE）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_global_host_to_local_device)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_DEVICE_VA_START + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// D2GH 方向通过（LOCAL_DEVICE → GLOBAL_HOST）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_device_to_global_host2)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_DEVICE_VA_START);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// AUTO + 混合类型（LOCAL_DEVICE src + GVA dest）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_device_to_gva)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_DEVICE_VA_START);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mockEntity->copyDirection, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST);
}

// AUTO + 单个用户地址（src=用户, dest=GVA → 应成功）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_user_to_gva_passes)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(0x7f001000);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(mockEntity->copyDirection, HYBM_LOCAL_HOST_TO_GLOBAL_HOST);
}

// GVA→GVA 传 GH2GH 通过
TEST_F(HybmDataOpEntryTest, hybm_data_copy_gva_to_gva_gh2gh)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 2048; // 2048
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// GH2LH 通过
TEST_F(HybmDataOpEntryTest, hybm_data_copy_global_to_local_host)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, nullptr, 0);
    EXPECT_EQ(ret, 0);
}

// 方向不匹配：GVA+GVA 传 H2GD（需要dest=GD）
TEST_F(HybmDataOpEntryTest, hybm_data_copy_wrong_direction_gva_to_gva)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(HYBM_GVM_START_ADDR);
    params.dest = reinterpret_cast<void *>(HYBM_GVM_START_ADDR + 0x1000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}

// AUTO+用户地址组合拦截
TEST_F(HybmDataOpEntryTest, hybm_data_copy_auto_user_addr_combined)
{
    hybm_copy_params params{};
    params.src = reinterpret_cast<void *>(0x7f001000);
    params.dest = reinterpret_cast<void *>(0x7f002000);
    params.dataSize = 1024;
    auto ret = hybm_data_copy(mockEntity.get(), &params, HYBM_DATA_COPY_DIRECTION_AUTO, nullptr, 0);
    EXPECT_EQ(ret, BM_INVALID_PARAM);
}
