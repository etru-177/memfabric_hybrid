/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#define private public
#include "hybm_data_op_device_urma.h"
#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "hybm_define.h"
#include "hybm_rbtree_range_pool.h"
#include "hybm_transport_manager.h"
#undef private

using namespace ock::mf;

namespace {
constexpr uint32_t LOCAL_RANK = 0;
constexpr uint32_t REMOTE_RANK = 1;
constexpr uint64_t TEST_SWAP_ALLOC_SIZE = 4096ULL;

int32_t MockAclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, uint32_t kind)
{
    (void)kind;
    EXPECT_NE(dst, nullptr);
    EXPECT_NE(src, nullptr);
    EXPECT_GE(destMax, count);
    std::memcpy(dst, src, count);
    return BM_OK;
}

int32_t MockAclrtMemcpyFailed(void *, size_t, const void *, size_t, uint32_t)
{
    return BM_ERROR;
}

int32_t MockAclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, uint32_t kind, void *)
{
    return MockAclrtMemcpy(dst, destMax, src, count, kind);
}

int32_t MockAclrtMemcpyAsyncFailed(void *, size_t, const void *, size_t, uint32_t, void *)
{
    return BM_ERROR;
}

class TransportManagerMock : public transport::TransportManager {
public:
    Result OpenDevice(const transport::TransportOptions &) override
    {
        return BM_OK;
    }
    Result CloseDevice() override
    {
        return BM_OK;
    }
    Result RegisterMemoryRegion(const transport::TransportMemoryRegion &) override
    {
        registerMemoryRegionCount++;
        return registerMemoryRegionResult;
    }
    Result UnregisterMemoryRegion(uint64_t) override
    {
        unregisterMemoryRegionCount++;
        return unregisterMemoryRegionResult;
    }
    bool QueryHasRegistered(uint64_t, uint64_t) override
    {
        queryHasRegisteredCount++;
        return queryHasRegisteredResult;
    }
    Result QueryMemoryKey(uint64_t, transport::TransportMemoryKey &) override
    {
        return BM_OK;
    }
    void UpdateMemoryKey(transport::TransportMemoryKey &, void *) override {}
    Result Prepare(const transport::HybmTransPrepareOptions &) override
    {
        return BM_OK;
    }
    Result RemoveRanks(const std::vector<uint32_t> &) override
    {
        return BM_OK;
    }
    Result Connect() override
    {
        return BM_OK;
    }
    Result AsyncConnect() override
    {
        return BM_OK;
    }
    Result WaitForConnected(int64_t) override
    {
        return BM_OK;
    }
    Result UpdateRankOptions(const transport::HybmTransPrepareOptions &) override
    {
        return BM_OK;
    }
    const std::string &GetNic() const override
    {
        return nic;
    }
    const transport::TransportPrivateData GetPrivateData() const override
    {
        return {};
    }
    Result ReadRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override
    {
        readRemoteCount++;
        EXPECT_EQ(rankId, REMOTE_RANK);
        std::memcpy(reinterpret_cast<void *>(lAddr), reinterpret_cast<const void *>(rAddr), size);
        return readRemoteResult;
    }
    Result WriteRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override
    {
        writeRemoteCount++;
        EXPECT_EQ(rankId, REMOTE_RANK);
        std::memcpy(reinterpret_cast<void *>(rAddr), reinterpret_cast<const void *>(lAddr), size);
        return writeRemoteResult;
    }
    Result ReadRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override
    {
        readRemoteAsyncCount++;
        return ReadRemote(rankId, lAddr, rAddr, size);
    }
    Result WriteRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override
    {
        writeRemoteAsyncCount++;
        return WriteRemote(rankId, lAddr, rAddr, size);
    }
    Result Synchronize(uint32_t rankId) override
    {
        synchronizeCount++;
        EXPECT_EQ(rankId, REMOTE_RANK);
        return synchronizeResult;
    }
    Result WriteRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &desc) override
    {
        writeRemoteBatchAsyncCount++;
        EXPECT_EQ(rankId, REMOTE_RANK);
        if (writeRemoteBatchAsyncResult != BM_OK) {
            return writeRemoteBatchAsyncResult;
        }
        for (size_t i = 0; i < desc.localAddrs.size(); ++i) {
            std::memcpy(desc.globalAddrs[i], desc.localAddrs[i], desc.counts[i]);
        }
        return BM_OK;
    }
    Result ReadRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &desc) override
    {
        readRemoteBatchAsyncCount++;
        EXPECT_EQ(rankId, REMOTE_RANK);
        if (readRemoteBatchAsyncResult != BM_OK) {
            return readRemoteBatchAsyncResult;
        }
        for (size_t i = 0; i < desc.localAddrs.size(); ++i) {
            std::memcpy(desc.localAddrs[i], desc.globalAddrs[i], desc.counts[i]);
        }
        return BM_OK;
    }

    std::string nic{"eth0"};
    bool queryHasRegisteredResult{true};
    uint64_t registerMemoryRegionCount{0};
    uint64_t unregisterMemoryRegionCount{0};
    uint64_t queryHasRegisteredCount{0};
    uint64_t readRemoteCount{0};
    uint64_t writeRemoteCount{0};
    uint64_t readRemoteAsyncCount{0};
    uint64_t writeRemoteAsyncCount{0};
    uint64_t readRemoteBatchAsyncCount{0};
    uint64_t writeRemoteBatchAsyncCount{0};
    uint64_t synchronizeCount{0};
    Result readRemoteResult{BM_OK};
    Result writeRemoteResult{BM_OK};
    Result registerMemoryRegionResult{BM_OK};
    Result unregisterMemoryRegionResult{BM_OK};
    Result readRemoteBatchAsyncResult{BM_OK};
    Result writeRemoteBatchAsyncResult{BM_OK};
    Result synchronizeResult{BM_OK};
};

class TestDataOpDeviceURMA : public DataOpDeviceURMA {
public:
    TestDataOpDeviceURMA(uint32_t rankId, std::shared_ptr<transport::TransportManager> tm)
        : DataOpDeviceURMA(rankId, std::move(tm))
    {}

    ~TestDataOpDeviceURMA() override
    {
        if (allocatedSwapBase_ != nullptr) {
            free(allocatedSwapBase_);
            allocatedSwapBase_ = nullptr;
        }
    }

    Result AllocSwapMemory() override
    {
        if (allocatedSwapBase_ == nullptr) {
            allocatedSwapBase_ = malloc(TEST_SWAP_ALLOC_SIZE);
        }
        urmaSwapBaseAddr_ = allocatedSwapBase_;
        return urmaSwapBaseAddr_ == nullptr ? BM_MALLOC_FAILED : BM_OK;
    }

    Result InitializeWithSwap(uint64_t swapSizeBytes)
    {
        if (inited_) {
            return BM_OK;
        }
        urmaSwapSpaceSize_ = swapSizeBytes;
        if (urmaSwapSpaceSize_ == 0) {
            inited_ = true;
            return BM_OK;
        }
        auto ret = AllocSwapMemory();
        if (ret != BM_OK) {
            return ret;
        }
        transport::TransportMemoryRegion input;
        input.addr = reinterpret_cast<uint64_t>(urmaSwapBaseAddr_);
        input.size = urmaSwapSpaceSize_;
        input.flags = transport::REG_MR_FLAG_HBM;
        if (transportManager_ != nullptr) {
            ret = transportManager_->RegisterMemoryRegion(input);
            if (ret != BM_OK) {
                FreeSwapMemory();
                return BM_MALLOC_FAILED;
            }
        }
        urmaSwapMemoryAllocator_ =
            std::make_shared<RbtreeRangePool>(static_cast<uint8_t *>(urmaSwapBaseAddr_), urmaSwapSpaceSize_);
        inited_ = true;
        return BM_OK;
    }

private:
    void *allocatedSwapBase_{nullptr};
};

struct DlAclApiCopyFnGuard {
    aclrtMemcpyFunc oldMemcpy{DlAclApi::pAclrtMemcpy};
    aclrtMemcpyAsyncFunc oldMemcpyAsync{DlAclApi::pAclrtMemcpyAsync};
    aclrtSynchronizeStreamFunc oldSynchronizeStream{DlAclApi::pAclrtSynchronizeStream};
    aclrtMemcpyBatchFunc oldMemcpyBatch{DlAclApi::pAclrtMemcpyBatch};
    aclrtFreeHostFunc oldFreeHost{DlAclApi::pAclrtFreeHost};

    ~DlAclApiCopyFnGuard()
    {
        DlAclApi::pAclrtMemcpy = oldMemcpy;
        DlAclApi::pAclrtMemcpyAsync = oldMemcpyAsync;
        DlAclApi::pAclrtSynchronizeStream = oldSynchronizeStream;
        DlAclApi::pAclrtMemcpyBatch = oldMemcpyBatch;
        DlAclApi::pAclrtFreeHost = oldFreeHost;
    }
};

int32_t MockAclrtSynchronizeStream(void *)
{
    return BM_OK;
}

int32_t MockAclrtMemcpyBatch(void **dsts, size_t *destMax, void **srcs, size_t *sizes, size_t numBatches,
                             aclrtMemcpyBatchAttr *, size_t *, size_t, size_t *)
{
    for (size_t i = 0; i < numBatches; ++i) {
        EXPECT_GE(destMax[i], sizes[i]);
        std::memcpy(dsts[i], srcs[i], sizes[i]);
    }
    return BM_OK;
}

int32_t MockAclrtFreeHost(void *)
{
    return BM_OK;
}

void ExpectLocalDataCopy(DataOpDeviceURMA &dataOp, hybm_data_copy_direction direction)
{
    char src[16] = "local_copy";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp.DataCopy(params, direction, options), BM_OK);
    EXPECT_EQ(std::memcmp(dst, src, sizeof(src)), 0);
}

void ExpectLocalBatchCopy(DataOpDeviceURMA &dataOp, hybm_data_copy_direction direction)
{
    char src0[8] = "aa";
    char src1[8] = "bb";
    char dst0[8] = {};
    char dst1[8] = {};
    void *sources[2] = {src0, src1};
    void *destinations[2] = {dst0, dst1};
    uint64_t sizes[2] = {sizeof(src0), sizeof(src1)};
    hybm_batch_copy_params params{sources, destinations, sizes, 2};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp.BatchDataCopy(params, direction, options), BM_OK);
    EXPECT_EQ(std::memcmp(dst0, src0, sizeof(src0)), 0);
    EXPECT_EQ(std::memcmp(dst1, src1, sizeof(src1)), 0);
}
} // namespace

class HybmDataOpDeviceUrmaTest : public testing::Test {
public:
    void SetUp() override
    {
        tm = std::make_shared<TransportManagerMock>();
        dataOp = std::make_shared<TestDataOpDeviceURMA>(LOCAL_RANK, tm);
        DlAclApi::pAclrtMemcpy = MockAclrtMemcpy;
        DlAclApi::pAclrtMemcpyAsync = MockAclrtMemcpyAsync;
        DlAclApi::pAclrtSynchronizeStream = MockAclrtSynchronizeStream;
        DlAclApi::pAclrtMemcpyBatch = MockAclrtMemcpyBatch;
        DlAclApi::pAclrtFreeHost = MockAclrtFreeHost;
    }

protected:
    DlAclApiCopyFnGuard guard;
    std::shared_ptr<TransportManagerMock> tm;
    std::shared_ptr<TestDataOpDeviceURMA> dataOp;
};

TEST_F(HybmDataOpDeviceUrmaTest, InitializeIsIdempotentAndUnInitializeClearsState)
{
    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_OK);
    EXPECT_TRUE(dataOp->inited_);
    void *swapBase = dataOp->urmaSwapBaseAddr_;
    EXPECT_EQ(tm->registerMemoryRegionCount, 1U);

    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_OK);
    EXPECT_EQ(dataOp->urmaSwapBaseAddr_, swapBase);
    EXPECT_EQ(tm->registerMemoryRegionCount, 1U);

    dataOp->UnInitialize();
    EXPECT_FALSE(dataOp->inited_);
    EXPECT_EQ(tm->unregisterMemoryRegionCount, 1U);
}

TEST_F(HybmDataOpDeviceUrmaTest, InitializeReturnsMallocFailedWhenSwapRegistrationFails)
{
    tm->registerMemoryRegionResult = BM_ERROR;

    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_MALLOC_FAILED);
    EXPECT_FALSE(dataOp->inited_);
    EXPECT_EQ(dataOp->urmaSwapBaseAddr_, nullptr);
    EXPECT_EQ(tm->registerMemoryRegionCount, 1U);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyLocalDirectionsUseAclMemcpy)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    char src[16] = "hybm_urma_ut";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_STREQ(dst, src);
    std::memset(dst, 0, sizeof(dst));
    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, options), BM_OK);
    EXPECT_STREQ(dst, src);
    EXPECT_EQ(dataOp->DataCopy(params, HYBM_DATA_COPY_DIRECTION_AUTO, options), BM_INVALID_PARAM);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyAllLocalDirectionsSucceed)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    const hybm_data_copy_direction directions[] = {
        HYBM_LOCAL_HOST_TO_GLOBAL_HOST,      HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST,
        HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE,  HYBM_GLOBAL_HOST_TO_GLOBAL_HOST,  HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE,
        HYBM_GLOBAL_HOST_TO_LOCAL_HOST,      HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE, HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST,
        HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE, HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST, HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE,
    };

    for (auto direction : directions) {
        ExpectLocalDataCopy(*dataOp, direction);
    }
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyRemoteWriteAndReadUseTransportManager)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    char src[16] = "remote_write";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->writeRemoteCount, 1U);
    EXPECT_STREQ(dst, src);

    std::memset(src, 0, sizeof(src));
    std::strcpy(dst, "remote_read");
    options.srcRankId = REMOTE_RANK;
    options.destRankId = LOCAL_RANK;
    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->readRemoteCount, 1U);
    EXPECT_STREQ(src, dst);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopySafePutUsesSwapWhenLocalSourceIsNotRegistered)
{
    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src[16] = "safe_put";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_GE(tm->queryHasRegisteredCount, 1U);
    EXPECT_EQ(tm->writeRemoteCount, 1U);
    EXPECT_STREQ(dst, src);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopySafeGetUsesSwapWhenLocalDestinationIsNotRegistered)
{
    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src[16] = "safe_get";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = REMOTE_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, options), BM_OK);
    EXPECT_GE(tm->queryHasRegisteredCount, 1U);
    EXPECT_EQ(tm->readRemoteCount, 1U);
    EXPECT_STREQ(dst, src);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyRejectsRemoteToRemoteWhenLocalRankIsAbsent)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    char src[4] = {};
    char dst[4] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = 2UL;
    options.destRankId = 3UL;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_INVALID_PARAM);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyAsyncUnsupportedAndWaitSucceeds)
{
    hybm_copy_params params{};
    ExtOptions options{};
    EXPECT_EQ(dataOp->DataCopyAsync(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_ERROR);
    EXPECT_EQ(dataOp->Wait(0), BM_OK);
}

TEST_F(HybmDataOpDeviceUrmaTest, DataCopyPropagatesAclMemcpyFailure)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    DlAclApi::pAclrtMemcpy = MockAclrtMemcpyFailed;
    char src[4] = {1, 2, 3, 4};
    char dst[4] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_DL_FUNCTION_FAILED);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyRegisteredWriteAndReadSynchronizes)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    char src0[8] = "src0";
    char src1[8] = "src1";
    char dst0[8] = {};
    char dst1[8] = {};
    void *sources[2] = {src0, src1};
    void *destinations[2] = {dst0, dst1};
    uint64_t sizes[2] = {sizeof(src0), sizeof(src1)};
    hybm_batch_copy_params params{sources, destinations, sizes, 2};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->writeRemoteBatchAsyncCount, 1U);
    EXPECT_EQ(tm->synchronizeCount, 1U);
    EXPECT_STREQ(dst0, src0);
    EXPECT_STREQ(dst1, src1);

    tm->synchronizeCount = 0;
    options.srcRankId = REMOTE_RANK;
    options.destRankId = LOCAL_RANK;
    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->readRemoteBatchAsyncCount, 1U);
    EXPECT_EQ(tm->synchronizeCount, 1U);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyAllLocalDirectionsSucceed)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    const hybm_data_copy_direction directions[] = {
        HYBM_LOCAL_HOST_TO_GLOBAL_HOST,      HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE, HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST,
        HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE,  HYBM_GLOBAL_HOST_TO_GLOBAL_HOST,  HYBM_GLOBAL_HOST_TO_GLOBAL_DEVICE,
        HYBM_GLOBAL_HOST_TO_LOCAL_HOST,      HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE, HYBM_GLOBAL_DEVICE_TO_GLOBAL_HOST,
        HYBM_GLOBAL_DEVICE_TO_GLOBAL_DEVICE, HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST, HYBM_GLOBAL_DEVICE_TO_LOCAL_DEVICE,
    };

    for (auto direction : directions) {
        ExpectLocalBatchCopy(*dataOp, direction);
    }
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyUsesDefaultPathWhenLocalMemoryIsNotRegistered)
{
    EXPECT_EQ(dataOp->InitializeWithSwap(TEST_SWAP_ALLOC_SIZE), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src0[8] = "s0";
    char src1[8] = "s1";
    char dst0[8] = {};
    char dst1[8] = {};
    void *sources[2] = {src0, src1};
    void *destinations[2] = {dst0, dst1};
    uint64_t sizes[2] = {sizeof(src0), sizeof(src1)};
    hybm_batch_copy_params params{sources, destinations, sizes, 2};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->writeRemoteBatchAsyncCount, 0U);
    EXPECT_EQ(tm->writeRemoteCount, 2U);
    EXPECT_STREQ(dst0, src0);
    EXPECT_STREQ(dst1, src1);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyG2GRemoteWriteAndReadUseSingleBatch)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    char src0[8] = "g0";
    char src1[8] = "g1";
    char dst0[8] = {};
    char dst1[8] = {};
    void *sources[2] = {src0, src1};
    void *destinations[2] = {dst0, dst1};
    uint64_t sizes[2] = {sizeof(src0), sizeof(src1)};
    hybm_batch_copy_params params{sources, destinations, sizes, 2};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->writeRemoteBatchAsyncCount, 1U);
    EXPECT_EQ(tm->synchronizeCount, 1U);
    EXPECT_STREQ(dst0, src0);
    EXPECT_STREQ(dst1, src1);

    tm->synchronizeCount = 0;
    options.srcRankId = REMOTE_RANK;
    options.destRankId = LOCAL_RANK;
    std::memset(src0, 0, sizeof(src0));
    std::memset(src1, 0, sizeof(src1));
    std::strcpy(dst0, "r0");
    std::strcpy(dst1, "r1");
    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_OK);
    EXPECT_EQ(tm->readRemoteBatchAsyncCount, 1U);
    EXPECT_EQ(tm->synchronizeCount, 1U);
    EXPECT_STREQ(src0, dst0);
    EXPECT_STREQ(src1, dst1);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyPropagatesBatchAsyncFailure)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    tm->writeRemoteBatchAsyncResult = BM_ERROR;
    char src[8] = "fail";
    char dst[8] = {};
    void *sources[1] = {src};
    void *destinations[1] = {dst};
    uint64_t sizes[1] = {sizeof(src)};
    hybm_batch_copy_params params{sources, destinations, sizes, 1};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_GLOBAL_HOST_TO_GLOBAL_HOST, options), BM_ERROR);
    EXPECT_EQ(tm->writeRemoteBatchAsyncCount, 1U);
    EXPECT_EQ(tm->synchronizeCount, 0U);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyLocalAsyncFailureReturnsDlFailure)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    DlAclApi::pAclrtMemcpyAsync = MockAclrtMemcpyAsyncFailed;
    char src[8] = "async";
    char dst[8] = {};
    void *sources[1] = {src};
    void *destinations[1] = {dst};
    uint64_t sizes[1] = {sizeof(src)};
    hybm_batch_copy_params params{sources, destinations, sizes, 1};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_LOCAL_DEVICE_TO_GLOBAL_DEVICE, options), BM_DL_FUNCTION_FAILED);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyRejectsUnsupportedDirection)
{
    char src[4] = {};
    char dst[4] = {};
    void *sources[1] = {src};
    void *destinations[1] = {dst};
    uint64_t sizes[1] = {sizeof(src)};
    hybm_batch_copy_params params{sources, destinations, sizes, 1};
    ExtOptions options{};

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_DATA_COPY_DIRECTION_AUTO, options), BM_ERROR);
}

TEST_F(HybmDataOpDeviceUrmaTest, InitializeSkipsSwapAllocationWhenSwapSpaceSizeIsZero)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    EXPECT_TRUE(dataOp->inited_);
    EXPECT_EQ(dataOp->urmaSwapSpaceSize_, 0ULL);
    EXPECT_EQ(dataOp->urmaSwapBaseAddr_, nullptr);
    EXPECT_EQ(dataOp->urmaSwapMemoryAllocator_, nullptr);
    EXPECT_EQ(tm->registerMemoryRegionCount, 0U);
}

TEST_F(HybmDataOpDeviceUrmaTest, InitializeIsIdempotentWhenSwapSpaceSizeIsZero)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    EXPECT_TRUE(dataOp->inited_);
    EXPECT_EQ(tm->registerMemoryRegionCount, 0U);

    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    EXPECT_TRUE(dataOp->inited_);
    EXPECT_EQ(tm->registerMemoryRegionCount, 0U);

    dataOp->UnInitialize();
    EXPECT_FALSE(dataOp->inited_);
    EXPECT_EQ(tm->unregisterMemoryRegionCount, 0U);
}

TEST_F(HybmDataOpDeviceUrmaTest, SafePutReturnsErrorWhenSwapAllocatorIsNotInitialized)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src[16] = "no_swap_put";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_ERROR);
    EXPECT_GE(tm->queryHasRegisteredCount, 1U);
    EXPECT_EQ(tm->writeRemoteCount, 0U);
}

TEST_F(HybmDataOpDeviceUrmaTest, SafeGetReturnsErrorWhenSwapAllocatorIsNotInitialized)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src[16] = "no_swap_get";
    char dst[16] = {};
    hybm_copy_params params{src, dst, sizeof(src)};
    ExtOptions options{};
    options.srcRankId = REMOTE_RANK;
    options.destRankId = LOCAL_RANK;

    EXPECT_EQ(dataOp->DataCopy(params, HYBM_GLOBAL_HOST_TO_LOCAL_HOST, options), BM_ERROR);
    EXPECT_GE(tm->queryHasRegisteredCount, 1U);
    EXPECT_EQ(tm->readRemoteCount, 0U);
}

TEST_F(HybmDataOpDeviceUrmaTest, BatchDataCopyDefaultPathFailsWhenSwapAllocatorIsNotInitialized)
{
    EXPECT_EQ(dataOp->Initialize(), BM_OK);
    tm->queryHasRegisteredResult = false;
    char src[8] = "s0";
    char dst[8] = {};
    void *sources[1] = {src};
    void *destinations[1] = {dst};
    uint64_t sizes[1] = {sizeof(src)};
    hybm_batch_copy_params params{sources, destinations, sizes, 1};
    ExtOptions options{};
    options.srcRankId = LOCAL_RANK;
    options.destRankId = REMOTE_RANK;

    EXPECT_EQ(dataOp->BatchDataCopy(params, HYBM_LOCAL_HOST_TO_GLOBAL_HOST, options), BM_ERROR);
    EXPECT_EQ(tm->writeRemoteCount, 0U);
}
