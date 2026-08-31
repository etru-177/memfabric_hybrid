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

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <iterator>
#include <string>

#include "dl_acl_api.h"
#include "dl_hcomm_api.h"
#include "hybm_batch_transfer.h"
#include "hybm_logger.h"
#include "hybm_stream_manager.h"
#include "device_urma_eid_reader.h"
#include "hybm_va_manager.h"
#include "device_urma_transport_manager.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {

using namespace urma;

namespace {
constexpr uint32_t HCOMM_NORMAL_NOTIFY_NUM = 0;
constexpr const char *HYBM_DEVICE_FUNC_READ = "HybmBatchRead";
constexpr const char *HYBM_DEVICE_FUNC_WRITE = "HybmBatchWrite";
constexpr uint32_t HYBM_DEVICE_KERNEL_BLOCK_DIM = 1U;
constexpr uint32_t ACL_NOTIFY_FLAG_DEVICE_ONLY = 0x00000001U; // 使能该bit表示创建的Notify仅在Device上调用。
constexpr uint16_t HYBM_DEVICE_KERNEL_TIMEOUT_S = 60U;
constexpr uint32_t HYBM_NOTIFY_DEFAULT_WAIT_TIME_S = 27U * 68U;
UrmaMemoryType ToUrmaMemoryType(uint32_t flags)
{
    if (flags & REG_MR_FLAG_HBM) {
        return UrmaMemoryType::DEVICE_HBM;
    }
    return UrmaMemoryType::HOST_DRAM;
}

UrmaCommMem ToUrmaMem(const TransportMemoryRegion &mr)
{
    return UrmaCommMem{mr.addr, mr.size, ToUrmaMemoryType(mr.flags)};
}

bool IsSupportedMemoryFlags(uint32_t flags)
{
    const bool hasDram = (flags & (REG_MR_FLAG_DRAM | REG_MR_FLAG_ACL_DRAM)) != 0;
    const bool hasHbm = (flags & REG_MR_FLAG_HBM) != 0;
    return !(hasDram && hasHbm);
}

bool ContainsAddressRange(uint64_t outerAddr, uint64_t outerSize, uint64_t innerAddr, uint64_t innerSize)
{
    uint64_t outerEnd = 0;
    uint64_t innerEnd = 0;
    const UrmaCommMem outer{outerAddr, outerSize, UrmaMemoryType::HOST_DRAM};
    const UrmaCommMem inner{innerAddr, innerSize, UrmaMemoryType::HOST_DRAM};
    return GetRangeEnd(outer, outerEnd) && GetRangeEnd(inner, innerEnd) && outerAddr <= innerAddr &&
           outerEnd >= innerEnd;
}

UrmaProtocol GetEndpointProtocolFromOptions(uint32_t protocol)
{
    if (protocol & HYBM_DOP_TYPE_DEVICE_UBOE) {
        return UrmaProtocol::UBOE;
    }
    if (protocol & (HYBM_DOP_TYPE_DEVICE_URMA | HYBM_DOP_TYPE_HOST_DEVICE_URMA)) {
        return UrmaProtocol::UBC_CTP;
    }
    return UrmaProtocol::RESERVED;
}

std::string FormatEid(const std::array<uint8_t, COMM_ADDR_EID_LEN> &eidData)
{
    constexpr char HEX_DIGITS[] = "0123456789abcdef";
    std::string eid;
    eid.reserve(COMM_ADDR_EID_LEN * 2U);
    for (const auto byte : eidData) {
        eid.push_back(HEX_DIGITS[(byte >> 4U) & 0x0FU]);
        eid.push_back(HEX_DIGITS[byte & 0x0FU]);
    }
    return eid;
}

std::string FormatIpAddress(CommAddrType addrType, const uint8_t *addrData)
{
    int family = AF_UNSPEC;
    if (addrType == COMM_ADDR_TYPE_IP_V4) {
        family = AF_INET;
    } else if (addrType == COMM_ADDR_TYPE_IP_V6) {
        family = AF_INET6;
    } else {
        return "<invalid>";
    }

    char ipText[INET6_ADDRSTRLEN]{};
    if (addrData == nullptr || inet_ntop(family, addrData, ipText, sizeof(ipText)) == nullptr) {
        return "<invalid>";
    }
    return ipText;
}

bool HasBatchCopyMemoryKeys(const HybmTransPrepareOptions &options)
{
    return std::any_of(options.options.begin(), options.options.end(),
                       [](const auto &item) { return !item.second.memKeys.empty(); });
}

} // namespace

// get TLS(Thread Local Storage) bingdings
std::vector<DeviceUrmaTransportManager::ContextBinding> &DeviceUrmaTransportManager::GetTlsBindings()
{
    thread_local std::vector<ContextBinding> bindings;
    return bindings;
}

DeviceUrmaTransportManager::~DeviceUrmaTransportManager()
{
    (void)CloseDevice();
}

Result DeviceUrmaTransportManager::InitLocalDeviceInfoLocked(const TransportOptions &options)
{
    int32_t userId = -1;
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    BM_ASSERT_LOG_AND_RETURN(ret == 0 && userId >= 0,
                             "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
                             BM_DL_FUNCTION_FAILED);

    options_ = options;
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    userDeviceId_ = static_cast<uint32_t>(userId);
    int32_t phyId = 0;
    // 实测需要使用userDeviceId
    ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(userId, &phyId);
    BM_ASSERT_LOG_AND_RETURN(ret == 0, "aclrtGetPhyDevIdByLogicDevId() return=" << ret << ", userDeviceId=" << userId,
                             BM_DL_FUNCTION_FAILED);
    BM_LOG_INFO("aclrtGetPhyDevIdByLogicDevId: userId=" << userId << ", phyId=" << phyId);
    phyDeviceId_ = static_cast<uint32_t>(phyId);

    // Get device location info
    int64_t infoValue = 0;
    ret = DlAclApi::RtGetDeviceInfo(static_cast<uint32_t>(userId), 0, INFO_TYPE_SDID, &infoValue);
    BM_ASSERT_LOG_AND_RETURN(ret == 0, "RtGetDeviceInfo(INFO_TYPE_SDID) return=" << ret, BM_DL_FUNCTION_FAILED);
    sdid_ = static_cast<uint32_t>(infoValue);

    infoValue = 0;
    ret = DlAclApi::RtGetDeviceInfo(static_cast<uint32_t>(userId), 0, INFO_TYPE_SERVER_ID, &infoValue);
    BM_ASSERT_LOG_AND_RETURN(ret == 0, "RtGetDeviceInfo(INFO_TYPE_SERVER_ID) return=" << ret, BM_DL_FUNCTION_FAILED);
    serverId_ = static_cast<uint32_t>(infoValue);

    infoValue = 0;
    ret = DlAclApi::RtGetDeviceInfo(static_cast<uint32_t>(userId), 0, INFO_TYPE_SUPER_POD_ID, &infoValue);
    BM_ASSERT_LOG_AND_RETURN(ret == 0, "RtGetDeviceInfo(INFO_TYPE_SUPER_POD_ID) return=" << ret, BM_DL_FUNCTION_FAILED);
    superPodId_ = static_cast<uint32_t>(infoValue);
    BM_LOG_INFO("local device info: userId=" << userId << ", phyId=" << phyId << " sdid=" << sdid_
                                             << ", server_id=" << serverId_ << ", superpod id=" << superPodId_);

    return BM_OK;
}

Result DeviceUrmaTransportManager::InitDeviceTransferFlagLocked()
{
    // Allocate a local flag buffer on device, initialise to 1, and register with Hcomm.
    // It is exported as an Hcomm flag descriptor in the TransportMemoryKey payload,
    // so remote peers can import it and use the resulting address as remote_flag_addr.
    // NOT inserted into localRegistrations_ — not exported/imported as a regular MR.
    void *flagPtr = nullptr;
    auto ret = DlAclApi::AclrtMalloc(&flagPtr, sizeof(int64_t), static_cast<uint32_t>(ACL_MEM_MALLOC_NORMAL_ONLY));
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma AclrtMalloc for local flag buffer failed, ret: " << ret);
        return ret;
    }
    int64_t flagInit = 1;
    ret = DlAclApi::AclrtMemcpy(flagPtr, sizeof(int64_t), &flagInit, sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma AclrtMemcpy init local flag buffer failed, ret: " << ret);
        (void)DlAclApi::AclrtFree(flagPtr);
        return ret;
    }
    const UrmaCommMem flagMem{reinterpret_cast<uint64_t>(flagPtr), sizeof(int64_t), UrmaMemoryType::DEVICE_HBM};
    HcommMemHandle flagHandle = nullptr;
    ret = manager_.HcommMemReg(localEndpoint_, 1, flagMem, &flagHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma HcommMemReg for local flag buffer failed, ret: " << ret);
        (void)DlAclApi::AclrtFree(flagPtr);
        return ret;
    }
    devTransFlagPtr_ = flagPtr;
    devTransFlagSize_ = sizeof(int64_t);
    devTransFlagHcommHandle_ = flagHandle;
    BM_LOG_INFO("device_urma local flag buffer allocated and registered, addr: "
                << VaToStr(devTransFlagPtr_) << " size: " << devTransFlagSize_
                << " handle: " << devTransFlagHcommHandle_);
    return BM_OK;
}

Result DeviceUrmaTransportManager::BuildLocalEndpointDescLocked(UrmaProtocol protocol, UrmaEndpointDesc &localDesc)
{
    localDesc = {};
    localDesc.protocol = protocol;
    localDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    localDesc.loc.device.devPhyId = phyDeviceId_;
    localDesc.loc.device.superDevId = sdid_;
    localDesc.loc.device.serverIdx = serverId_;
    localDesc.loc.device.superPodIdx = superPodId_;

    if (protocol == UrmaProtocol::UBC_CTP) {
        std::array<uint8_t, COMM_ADDR_EID_LEN> eidData{};
        const auto ret = GetDeviceUrmaEid(phyDeviceId_, rankId_, eidData);
        if (ret != BM_OK) {
            return ret;
        }
        localDesc.type = COMM_ADDR_TYPE_IP_V6;
        std::memcpy(localDesc.raws, eidData.data(), COMM_ADDR_EID_LEN);
        BM_LOG_INFO("device_urma local endpoint address protocol=UBC_CTP phyDeviceId="
                    << phyDeviceId_ << " rankId=" << rankId_ << " eid=" << FormatEid(eidData));
        return BM_OK;
    }
    if (protocol == UrmaProtocol::UBOE) {
        CommAddrType addrType = COMM_ADDR_TYPE_RESERVED;
        std::array<uint8_t, sizeof(localDesc.raws)> addrData{};
        const auto ret = GetDeviceUrmaIpAddr(phyDeviceId_, rankId_, addrType, addrData);
        if (ret != BM_OK) {
            return ret;
        }
        localDesc.type = addrType;
        const size_t copyLen = (addrType == COMM_ADDR_TYPE_IP_V4) ? sizeof(struct in_addr) : sizeof(struct in6_addr);
        std::memcpy(localDesc.raws, addrData.data(), copyLen);
        BM_LOG_INFO("device_urma local endpoint address protocol=UBOE phyDeviceId="
                    << phyDeviceId_ << " rankId=" << rankId_ << " ip=" << FormatIpAddress(addrType, addrData.data()));
        return BM_OK;
    }
    BM_LOG_ERROR("device_urma unexpected protocol=" << static_cast<int>(protocol) << " phyDeviceId=" << phyDeviceId_
                                                    << " rankId=" << rankId_);
    return BM_INVALID_PARAM;
}

Result DeviceUrmaTransportManager::CreateEndpointAndInitResourcesLocked(const UrmaEndpointDesc &localDesc)
{
    auto endpoint = manager_.CreateEndpoint(localDesc);
    if (endpoint == nullptr) {
        BM_LOG_ERROR("device_urma CreateEndpoint failed, protocol=" << static_cast<int>(localDesc.protocol)
                                                                    << " phyDeviceId=" << phyDeviceId_
                                                                    << " rankId=" << rankId_);
        return BM_MALLOC_FAILED;
    }
    localEndpoint_ = endpoint;
    localEndpointDesc_ = localDesc;

    auto ret = InitDeviceTransferFlagLocked();
    if (ret != BM_OK) {
        RollbackOpenDeviceLocked();
        return ret;
    }
    ret = EnsureDeviceKernelLoadedLocked();
    if (ret != BM_OK) {
        RollbackOpenDeviceLocked();
        return ret;
    }
    return BM_OK;
}

void DeviceUrmaTransportManager::RollbackOpenDeviceLocked()
{
    if (devTransFlagHcommHandle_ != nullptr) {
        auto ret = manager_.HcommMemUnreg(localEndpoint_, devTransFlagHcommHandle_);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma RollbackOpenDevice HcommMemUnreg devTransFlag failed, ret: " << ret);
        }
    }
    if (devTransFlagPtr_ != nullptr) {
        (void)DlAclApi::AclrtFree(devTransFlagPtr_);
        devTransFlagPtr_ = nullptr;
    }
    if (localEndpoint_ != nullptr) {
        (void)HcomUrmaDestroyEndpoint(localEndpoint_->hcommEndpoint);
        localEndpoint_.reset();
    }
    localEndpointDesc_ = UrmaEndpointDesc{};
}

Result DeviceUrmaTransportManager::OpenEndpointResourcesLocked(const TransportOptions &options)
{
    const auto protocol = GetEndpointProtocolFromOptions(options.protocol);
    if (protocol == UrmaProtocol::RESERVED) {
        BM_LOG_ERROR("device_urma OpenDevice unsupported protocol bits: " << options.protocol
                                                                          << ", rankId=" << rankId_);
        return BM_INVALID_PARAM;
    }
    BM_LOG_INFO("device_urma OpenDevice protocol=" << static_cast<int>(protocol) << ", rankId=" << rankId_
                                                   << ", phyDeviceId=" << phyDeviceId_);
    UrmaEndpointDesc localDesc{};
    auto ret = BuildLocalEndpointDescLocked(protocol, localDesc);
    if (ret != BM_OK) {
        return ret;
    }
    ret = CreateEndpointAndInitResourcesLocked(localDesc);
    if (ret != BM_OK) {
        (void)RollbackOpenDeviceLocked();
        return ret;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::OpenDevice(const TransportOptions &options)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (opened_) {
        return BM_OK;
    }
    if (options.rankCount == 0 || options.rankId >= options.rankCount) {
        BM_LOG_ERROR("device_urma OpenDevice: invalid rankCount or rankId");
        return BM_INVALID_PARAM;
    }
    if (DlAclApi::GetAscendSocType() != AscendSocType::ASCEND_950) {
        BM_LOG_ERROR("device_urma is only supported on Ascend950 soc, rank: " << options.rankId);
        return BM_NOT_SUPPORTED;
    }

    static std::atomic<uint64_t> g_nextGen{1};
    auto newOwner = std::make_shared<OpenGeneration>();
    newOwner->id = g_nextGen.fetch_add(1, std::memory_order_relaxed);

    auto ret = InitLocalDeviceInfoLocked(options);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma InitLocalDeviceInfoLocked failed, rank=" << options.rankId << " ret: " << ret);
        return ret;
    }
    ret = OpenEndpointResourcesLocked(options);
    if (ret != BM_OK) {
        return ret;
    }
    owner_ = std::move(newOwner);
    opened_ = true;
    BM_LOG_INFO("device_urma OpenDevice success, rank: " << rankId_ << " rankCount: " << rankCount_
                                                         << " devPhyId: " << phyDeviceId_);
    return BM_OK;
}

Result DeviceUrmaTransportManager::EnsureDeviceKernelLoadedLocked()
{
    if (deviceKernelLoaded_) {
        return BM_OK;
    }

    auto ret = LoadDeviceKernelAndGetHandles(HYBM_DEVICE_FUNC_READ, HYBM_DEVICE_FUNC_WRITE, deviceKernelHandle_,
                                             deviceFuncHandles_);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LoadDeviceKernelAndGetHandles failed, ret: " << ret);
        return ret;
    }
    if (deviceFuncHandles_.batchRead == nullptr || deviceFuncHandles_.batchWrite == nullptr) {
        BM_LOG_ERROR("device_urma invalid device kernel function handles, read: "
                     << deviceFuncHandles_.batchRead << " write: " << deviceFuncHandles_.batchWrite);
        return BM_DL_FUNCTION_FAILED;
    }
    deviceKernelLoaded_ = true;
    return BM_OK;
}

DeviceUrmaTransportManager::CompletionContext *DeviceUrmaTransportManager::LookupOrCreateContextLocked()
{
    // Scan TLS bindings for a valid (owner matches this manager generation, context alive)
    auto &bindings = GetTlsBindings();
    for (auto &binding : bindings) {
        auto ownerSp = binding.owner.lock();
        if (!ownerSp || ownerSp != owner_) {
            continue;
        }
        auto ctxSp = binding.ctx.lock();
        if (!ctxSp) {
            continue;
        }
        return ctxSp.get();
    }

    // create and publish
    CompletionContext *newCtx = nullptr;
    auto ret = CreateAndPublishContextLocked(newCtx);
    if (ret != BM_OK) {
        return nullptr;
    }
    return newCtx;
}

Result DeviceUrmaTransportManager::CreateAndPublishContextLocked(CompletionContext *&outRaw)
{
    std::shared_ptr<CompletionContext> ctx;
    ctx = std::make_shared<CompletionContext>();
    void *stream = HybmStreamManager::GetThreadAclStream();
    if (stream == nullptr) {
        BM_LOG_ERROR("device_urma CreateAndPublishContextLocked GetThreadAclStream failed");
        return BM_DL_FUNCTION_FAILED;
    }
    ctx->stream = stream;
    auto ret = EnsureContextInitLocked(*ctx);
    if (ret != BM_OK) {
        return ret;
    }
    auto &bindings = GetTlsBindings();
    registry_.push_back(ctx);
    bindings.push_back({owner_, ctx});
    outRaw = ctx.get();
    return BM_OK;
}

Result DeviceUrmaTransportManager::EnsureContextInitLocked(CompletionContext &ctx)
{
    // Step 1: Create ACL notify (device-only)
    void *notify = nullptr;
    auto ret = DlAclApi::AclrtCreateNotify(&notify, ACL_NOTIFY_FLAG_DEVICE_ONLY);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma EnsureContextInitLocked AclrtCreateNotify failed, ret: " << ret);
        return ret;
    }
    ctx.notify = notify;

    // Step 2: Get notify id
    uint32_t notifyId = 0;
    ret = DlAclApi::AclrtGetNotifyId(notify, &notifyId);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma EnsureContextInitLocked AclrtGetNotifyId failed, ret: " << ret);
        (void)RollbackContextInitLocked(ctx);
        return ret;
    }
    ctx.notifyId = notifyId;

    // Step 3: Get device resource address for notify record
    uint64_t devAddr = 0;
    uint32_t devLen = 0;
    rtDevResInfo resInfo{};
    resInfo.dieId = 0U;
    resInfo.procType = RT_PROCESS_HCCP;
    resInfo.resType = RT_RES_TYPE_STARS_NOTIFY_RECORD;
    resInfo.resId = notifyId;
    resInfo.flag = 0U;
    rtDevResAddrInfo addrInfo{};
    addrInfo.resAddress = &devAddr;
    addrInfo.len = &devLen;
    ret = DlRtApi::RtGetDevResAddress(&resInfo, &addrInfo);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma EnsureContextInitLocked RtGetDevResAddress failed, notifyId: " << notifyId
                                                                                                 << " ret: " << ret);
        (void)RollbackContextInitLocked(ctx);
        return ret;
    }
    ctx.notifyAddr = devAddr;
    ctx.notifyLen = devLen;

    // Step 4: Register notify record address with Hcomm
    const UrmaCommMem notifyMem{devAddr, devLen, UrmaMemoryType::DEVICE_HBM};
    HcommMemHandle notifyHandle = nullptr;
    ret = manager_.HcommMemReg(localEndpoint_, ctx.notifyAddr, notifyMem, &notifyHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma EnsureContextInitLocked HcommMemReg for notify failed, ret: " << ret);
        (void)RollbackContextInitLocked(ctx);
        return ret;
    }
    ctx.notifyHcommHandle = notifyHandle;
    ctx.initialized = true;

    BM_LOG_INFO("device_urma EnsureContextInitLocked success, notifyId: " << notifyId << " addr: " << VaToStr(devAddr)
                                                                          << " len: " << devLen);
    return BM_OK;
}

void DeviceUrmaTransportManager::RollbackContextInitLocked(CompletionContext &ctx)
{
    if (ctx.notifyHcommHandle != nullptr) {
        auto ret = manager_.HcommMemUnreg(localEndpoint_, ctx.notifyHcommHandle);
        if (ret != BM_OK) {
            BM_LOG_WARN("device_urma RollbackContextInitLocked HcommMemUnreg failed, ret: " << ret);
        }
        ctx.notifyHcommHandle = nullptr;
    }
    ctx.notifyAddr = 0;
    ctx.notifyLen = 0;
    if (ctx.notify != nullptr) {
        auto ret = DlAclApi::AclrtDestroyNotify(ctx.notify);
        if (ret != BM_OK) {
            BM_LOG_WARN("device_urma RollbackContextInitLocked AclrtDestroyNotify failed, notify: "
                        << VaToStr(ctx.notify) << " ret: " << ret);
        }
        ctx.notify = nullptr;
    }
    ctx.notifyId = 0;
    ctx.initialized = false;
}

void DeviceUrmaTransportManager::CleanupContextLocked(CompletionContext &ctx)
{
    if (ctx.notifyHcommHandle != nullptr) {
        auto ret = manager_.HcommMemUnreg(localEndpoint_, ctx.notifyHcommHandle);
        if (ret != BM_OK) {
            BM_LOG_WARN("device_urma CleanupContextLocked HcommMemUnreg notify failed, ret: " << ret);
        }
        ctx.notifyHcommHandle = nullptr;
    }
    if (ctx.notify != nullptr) {
        auto ret = DlAclApi::AclrtDestroyNotify(ctx.notify);
        if (ret != BM_OK) {
            BM_LOG_WARN("device_urma CleanupContextLocked AclrtDestroyNotify failed, notify: " << VaToStr(ctx.notify)
                                                                                               << " ret: " << ret);
        }
        ctx.notify = nullptr;
    }
    ctx.notifyId = 0;
    ctx.notifyAddr = 0;
    ctx.notifyLen = 0;
    ctx.initialized = false;
}

void DeviceUrmaTransportManager::CloseDeviceCleanupResourcesLocked()
{
    for (auto &rankItem : remoteRanks_) {
        const uint32_t peerRank = rankItem.first;
        auto &state = rankItem.second;
        (void)CleanupPeerRankState(state, peerRank);
    }

    (void)CleanupLocalRegistrationsLocked();

    if (devTransFlagHcommHandle_ != nullptr) {
        (void)manager_.HcommMemUnreg(localEndpoint_, devTransFlagHcommHandle_);
        devTransFlagHcommHandle_ = nullptr;
    }

    remoteRanks_.clear();

    if (localEndpoint_ != nullptr) {
        (void)HcomUrmaDestroyEndpoint(localEndpoint_->hcommEndpoint);
        localEndpoint_.reset();
    }
    localEndpointDesc_ = UrmaEndpointDesc{};

    if (devTransFlagPtr_ != nullptr) {
        auto ret = DlAclApi::AclrtFree(devTransFlagPtr_);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma CloseDeviceCleanupResources AclrtFree devTransFlag failed, ret: " << ret);
        }
        devTransFlagPtr_ = nullptr;
    }

    deviceKernelLoaded_ = false;
}

DeviceUrmaTransportManager::CompletionContext *DeviceUrmaTransportManager::FindCurrentContextLocked() const
{
    void *stream = HybmStreamManager::GetThreadAclStream();
    if (stream == nullptr) {
        return nullptr;
    }
    auto &bindings = GetTlsBindings();
    for (auto &binding : bindings) {
        auto ownerSp = binding.owner.lock();
        if (!ownerSp || ownerSp != owner_) {
            continue;
        }
        auto ctxSp = binding.ctx.lock();
        if (!ctxSp) {
            continue;
        }
        if (ctxSp->stream != stream) {
            BM_LOG_ERROR("device_urma FindCurrentContextLocked stream mismatch, expected: "
                         << VaToStr(ctxSp->stream) << " actual: " << VaToStr(stream));
            return nullptr;
        }
        return ctxSp.get();
    }
    return nullptr;
}

Result DeviceUrmaTransportManager::ReleasePendingTransfersLocked(std::vector<PendingTransfer> &pendingTransfers)
{
    Result finalRet = BM_OK;
    auto it = pendingTransfers.begin();
    while (it != pendingTransfers.end()) {
        if (it->buffers.dstList == nullptr) {
            it = pendingTransfers.erase(it);
            continue;
        }
        auto ret = ReleaseDeviceTransferBuffers(it->buffers);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma ReleasePendingTransfersLocked ReleaseDeviceTransferBuffers failed, ret: " << ret);
            if (finalRet == BM_OK) {
                finalRet = ret;
            }
            it->inFlight = false;
            ++it;
        } else {
            it = pendingTransfers.erase(it);
        }
    }
    return finalRet;
}

void DeviceUrmaTransportManager::ExtractRankPending(std::vector<PendingTransfer> &src, uint32_t rankId,
                                                    std::vector<PendingTransfer> &dst)
{
    auto it = src.begin();
    while (it != src.end()) {
        if (it->rankId == rankId) {
            dst.push_back(std::move(*it));
            it = src.erase(it);
        } else {
            ++it;
        }
    }
}

void DeviceUrmaTransportManager::RestoreRankPending(std::vector<PendingTransfer> &src,
                                                    std::vector<PendingTransfer> &dst)
{
    if (src.empty()) {
        return;
    }
    dst.insert(dst.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
    src.clear();
}

Result DeviceUrmaTransportManager::SynchronizeContextLocked(void *notify, void *stream,
                                                            std::vector<PendingTransfer> &pendingTransfers)
{
    bool hasInFlight = false;
    for (const auto &pt : pendingTransfers) {
        if (pt.inFlight) {
            hasInFlight = true;
            break;
        }
    }

    if (hasInFlight) {
        auto notifyRet = DlAclApi::AclrtWaitAndResetNotify(notify, stream, HYBM_DEVICE_KERNEL_TIMEOUT_S);
        if (notifyRet != BM_OK) {
            BM_LOG_ERROR("device_urma SynchronizeContext AclrtWaitAndResetNotify failed, notify=" << notify << " ret="
                                                                                                  << notifyRet);
            return notifyRet;
        }
        auto syncRet = DlAclApi::AclrtSynchronizeStream(stream);
        if (syncRet != BM_OK) {
            BM_LOG_ERROR("device_urma SynchronizeContext AclrtSynchronizeStream failed, syncRet="
                         << syncRet << ", notify=" << notify << ", stream=" << stream);
            return syncRet;
        }
    }
    return ReleasePendingTransfersLocked(pendingTransfers);
}

Result DeviceUrmaTransportManager::CloseDevice()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!opened_) {
        return BM_OK;
    }
    if (routePublisher_ != nullptr) {
        const auto ret = routePublisher_->Clear();
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma CloseDevice failed to clear BatchCopy route, rank: "
                         << rankId_ << " userDeviceId: " << userDeviceId_ << " ret: " << ret);
            return ret;
        }
        routePublisher_.reset();
    }
    for (const auto &ctxSp : registry_) {
        if (!ctxSp) {
            continue;
        }
        (void)ReleasePendingTransfersLocked(ctxSp->pendingTransfers);
    }

    for (const auto &ctxSp : registry_) {
        if (!ctxSp) {
            continue;
        }
        (void)CleanupContextLocked(*ctxSp);
    }

    CloseDeviceCleanupResourcesLocked();

    registry_.clear();
    owner_.reset();
    opened_ = false;
    BM_LOG_INFO("device_urma CloseDevice success");
    return BM_OK;
}

Result DeviceUrmaTransportManager::DestroyRankChannelsAndThread(RemoteRankState &state, uint32_t peerRank)
{
    Result localResult = BM_OK;
    if (state.channel != 0) {
        auto hcommChan = state.channel;
        BM_LOG_INFO("device_urma calling HcommChannelDestroy, peerRank: " << peerRank << " channel: " << state.channel
                                                                          << " thread: " << state.thread);
        const auto ret = DlHcommApi::HcommChannelDestroy(&hcommChan, 1);
        if (ret != 0) {
            BM_LOG_ERROR("device_urma HcommChannelDestroy failed, peerRank: "
                         << peerRank << " channel: " << state.channel << " thread: " << state.thread
                         << " ret: " << ret);
            localResult = BM_DL_FUNCTION_FAILED;
        }
        state.channel = 0;
        state.channelDesc = {};
    }
    if (state.thread != 0) {
        auto hcommThread = state.thread;
        BM_LOG_INFO("device_urma HcommThreadFree, thread: " << state.thread);
        const auto ret = DlHcommApi::HcommThreadFree(&hcommThread, 1);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma HcommThreadFree failed, peerRank: " << peerRank << " thread: " << state.thread
                                                                          << " ret: " << ret);
            if (localResult == BM_OK) {
                localResult = ret;
            }
        }
        state.thread = 0;
    }
    return localResult;
}

Result DeviceUrmaTransportManager::UnimportPeerImportsAndFlag(RemoteRankState &state, uint32_t peerRank)
{
    Result localResult = BM_OK;
    auto importIt = state.imports.begin();
    while (importIt != state.imports.end()) {
        if (importIt->descBytes.empty()) {
            ++importIt;
            continue;
        }
        BM_LOG_INFO("device_urma HcommMemUnimport, peer: " << peerRank
                                                           << "descBytes.size: " << importIt->descBytes.size());
        const auto ret = manager_.HcommMemUnimport(localEndpoint_, importIt->descBytes.data(),
                                                   static_cast<uint32_t>(importIt->descBytes.size()));
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma UnimportPeerImportsAndFlag HcommMemUnimport failed, "
                         << "peerRank: " << peerRank << " descBytes.size: " << importIt->descBytes.size()
                         << " ret: " << ret);
            if (localResult == BM_OK) {
                localResult = ret;
            }
            // Keep import entry on failure — still owner-enumerable for retry
            ++importIt;
        } else {
            importIt->descBytes.clear();
            importIt = state.imports.erase(importIt);
        }
    }
    if (!state.remoteFlagDescBytes.empty()) {
        BM_LOG_INFO("device_urma HcommMemUnimport remoteFlagDescBytes, "
                    << "peerRank: " << peerRank << " remoteFlagDescBytes.size: " << state.remoteFlagDescBytes.size());
        const auto ret = DlHcommApi::HcommMemUnimport(localEndpoint_->hcommEndpoint, state.remoteFlagDescBytes.data(),
                                                      static_cast<uint32_t>(state.remoteFlagDescBytes.size()));
        if (ret != 0) {
            BM_LOG_ERROR("device_urma UnimportPeerImportsAndFlag HcommMemUnimport for flag desc failed, "
                         << "peerRank: " << peerRank << " ret: " << ret);
            if (localResult == BM_OK) {
                localResult = BM_DL_FUNCTION_FAILED;
            }
        } else {
            state.remoteFlagDescBytes.clear();
            state.remoteFlagAddr = 0;
            state.remoteFlagSize = 0;
        }
    } else {
        state.remoteFlagAddr = 0;
        state.remoteFlagSize = 0;
    }
    return localResult;
}

Result DeviceUrmaTransportManager::CleanupPeerRankState(RemoteRankState &state, uint32_t peerRank)
{
    Result localResult = BM_OK;

    const auto retChan = DestroyRankChannelsAndThread(state, peerRank);
    if (retChan != BM_OK && localResult == BM_OK) {
        localResult = retChan;
    }
    const auto retImports = UnimportPeerImportsAndFlag(state, peerRank);
    if (retImports != BM_OK && localResult == BM_OK) {
        localResult = retImports;
    }
    return localResult;
}

Result DeviceUrmaTransportManager::CleanupLocalRegistrationsLocked()
{
    Result localResult = BM_OK;
    for (auto &item : localRegistrations_) {
        const auto ret = manager_.HcommMemUnreg(localEndpoint_, item.second.handle);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma CleanupLocalRegistrationsLocked HcommMemUnreg global handle failed, "
                         << "addr: " << VaToStr(item.first) << " ret: " << ret);
            if (localResult == BM_OK)
                localResult = ret;
        }
    }
    localRegistrations_.clear();
    return localResult;
}

Result DeviceUrmaTransportManager::FindLocalRegistrationLocked(uint64_t addr, uint64_t size,
                                                               LocalRegistration *registration) const
{
    if (addr == 0) {
        BM_LOG_ERROR("device_urma FindLocalRegistrationLocked: addr is 0");
        return BM_INVALID_PARAM;
    }
    if (size == 0) {
        return BM_OK;
    }
    // Validate that (addr, size) forms a valid non-wrapping address range.
    if (!ContainsAddressRange(addr, size, addr, size)) {
        BM_LOG_ERROR("device_urma FindLocalRegistrationLocked: address range check failed, addr=" << VaToStr(addr)
                                                                                                  << " size=" << size);
        return BM_INVALID_PARAM;
    }
    for (const auto &item : localRegistrations_) {
        if (ContainsAddressRange(item.second.mr.addr, item.second.mr.size, addr, size)) {
            if (registration != nullptr) {
                *registration = item.second;
            }
            return BM_OK;
        }
    }
    return BM_INVALID_PARAM;
}

Result DeviceUrmaTransportManager::CorrectLocalRegAddressLocked(uint64_t addr, uint64_t size,
                                                                uint64_t &correctedAddr) const
{
    correctedAddr = addr;
    LocalRegistration registration{};
    auto ret = FindLocalRegistrationLocked(addr, size, &registration);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma CorrectLocalRegAddressLocked: local addr not registered, rankId: "
                     << rankId_ << " addr: " << VaToStr(addr) << " size: " << size << " ret: " << ret);
        return ret;
    }
    const bool isDram = (registration.mr.flags & (REG_MR_FLAG_DRAM | REG_MR_FLAG_ACL_DRAM)) != 0;
    if (!isDram || registration.deviceVa == 0) {
        return BM_OK;
    }
    const uint64_t offset = addr - registration.mr.addr;
    const uint64_t corrected = registration.deviceVa + offset;
    if (corrected < registration.deviceVa) {
        BM_LOG_ERROR("device_urma CorrectLocalRegAddressLocked: addr overflow, rankId: "
                     << rankId_ << " deviceVa: " << VaToStr(registration.deviceVa) << " offset: " << offset);
        return BM_INVALID_PARAM;
    }
    // Verify full [corrected, corrected + size) range does not overflow
    if (!ContainsAddressRange(corrected, size, corrected, size)) {
        BM_LOG_ERROR("device_urma CorrectLocalRegAddressLocked: range overflow, rankId: "
                     << rankId_ << " addr: " << VaToStr(addr) << " size: " << size
                     << " mr.addr: " << VaToStr(registration.mr.addr) << " deviceVa: " << VaToStr(registration.deviceVa)
                     << " offset: " << offset);
        return BM_INVALID_PARAM;
    }
    correctedAddr = corrected;
    return BM_OK;
}

Result DeviceUrmaTransportManager::FindRemoteRegistrationLocked(uint32_t rankId, uint64_t addr, uint64_t size,
                                                                RemoteRegistration *registration) const
{
    if (rankId >= rankCount_) {
        return BM_INVALID_PARAM;
    }
    if (size == 0) {
        return BM_OK;
    }
    const auto rankIt = remoteRanks_.find(rankId);
    if (rankIt == remoteRanks_.end()) {
        return BM_NOT_CONNECTED;
    }
    // Validate that (addr, size) forms a valid non-wrapping address range.
    if (!ContainsAddressRange(addr, size, addr, size)) {
        BM_LOG_ERROR("device_urma FindRemoteRegistrationLocked: address range check failed, addr=0x"
                     << std::hex << addr << " size=0x" << size << std::dec);
        return BM_INVALID_PARAM;
    }
    for (const auto &remote : rankIt->second.imports) {
        if (ContainsAddressRange(remote.addr, remote.size, addr, size)) {
            if (registration != nullptr) {
                *registration = remote;
            }
            return BM_OK;
        }
    }
    return BM_INVALID_PARAM;
}

Result DeviceUrmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion &mr)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    BM_VALIDATE_RETURN(mr.addr != 0 && mr.size != 0, "device_urma RegisterMemoryRegion: mr.addr or mr.size is 0",
                       BM_INVALID_PARAM);
    if (!IsSupportedMemoryFlags(mr.flags)) {
        BM_LOG_ERROR("device_urma RegisterMemoryRegion: unsupported memory flags: " << mr.flags);
        return BM_INVALID_PARAM;
    }
    UrmaCommMem mem = ToUrmaMem(mr);
    if (!IsValidMem(mem)) {
        BM_LOG_ERROR("device_urma RegisterMemoryRegion: invalid memory");
        return BM_INVALID_PARAM;
    }

    // localEndpoint_ must be created by OpenDevice before any memory registration
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("device_urma localEndpoint_ is null, cannot register memory");
        return BM_NOT_INITIALIZED;
    }

    auto item = localRegistrations_.find(mr.addr);
    if (item != localRegistrations_.end()) {
        if (item->second.mr.size != mr.size) {
            BM_LOG_ERROR("device_urma register memory conflict, addr: " << std::hex << mr.addr);
            return BM_ERROR;
        }
        item->second.refCount++;
        return BM_OK;
    }

    LocalRegistration registration{};
    registration.mr = mr;
    registration.memTag = mr.addr;
    registration.refCount = 1;

    if ((mr.flags & (REG_MR_FLAG_DRAM | REG_MR_FLAG_ACL_DRAM)) != 0) {
        const uint64_t dva = HybmVaManager::GetInstance().TransformVa(mr.addr, HVM_HVA, HVM_DVA);
        if (dva != 0) {
            registration.deviceVa = dva;
            mem.addr = dva;
        } else {
            BM_LOG_WARN("device_urma RegisterMemoryRegion: DRAM addr " << VaToStr(mr.addr)
                                                                       << " has no DVA mapping, using HVA");
        }
    }

    HcommMemHandle hcommHandle = nullptr;
    auto ret = manager_.HcommMemReg(localEndpoint_, registration.memTag, mem, &hcommHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma HcommMemReg failed, addr: " << VaToStr(mr.addr) << " size: " << mr.size
                                                              << " ret: " << ret);
        return ret;
    }
    registration.handle = hcommHandle;

    try {
        localRegistrations_.emplace(mr.addr, registration);
    } catch (...) {
        (void)manager_.HcommMemUnreg(localEndpoint_, hcommHandle);
        return BM_MALLOC_FAILED;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    auto item = localRegistrations_.find(addr);
    if (item == localRegistrations_.end()) {
        BM_LOG_WARN("device_urma unregister skipped for unknown addr: " << std::hex << addr);
        return BM_OK;
    }
    if (item->second.refCount > 1) {
        item->second.refCount--;
        return BM_OK;
    }

    Result finalRet = BM_OK;
    auto retUnreg = manager_.HcommMemUnreg(localEndpoint_, item->second.handle);
    if (retUnreg != BM_OK) {
        BM_LOG_ERROR("device_urma HcommMemUnreg failed for global handle, addr: " << std::hex << addr
                                                                                  << " ret: " << retUnreg);
        if (finalRet == BM_OK) {
            finalRet = retUnreg;
        }
    }
    localRegistrations_.erase(item);
    return finalRet;
}

bool DeviceUrmaTransportManager::QueryHasRegistered(uint64_t addr, uint64_t size)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return FindLocalRegistrationLocked(addr, size, nullptr) == BM_OK;
}

Result DeviceUrmaTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey &key)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    LocalRegistration registration{};
    auto ret = FindLocalRegistrationLocked(addr, 1, &registration);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma QueryMemoryKey addr is not registered: " << std::hex << addr);
        return ret;
    }

    if (registration.handle == INVALID_MEM_HANDLE) {
        BM_LOG_ERROR("device_urma QueryMemoryKey addr has no HCOMM handle: " << std::hex << addr);
        return BM_ERROR;
    }

    // Guard: devTransFlagHcommHandle_ must be valid for flag export
    if (devTransFlagHcommHandle_ == nullptr) {
        BM_LOG_ERROR("device_urma QueryMemoryKey devTransFlagHcommHandle_ is null, cannot export flag");
        return BM_ERROR;
    }

    // Export primary memory descriptor via HcommTransportManager (caches UrmaExportDesc + hcommDesc)
    const uint8_t *memDesc = nullptr;
    uint32_t memDescLen = 0;
    ret = manager_.HcommMemExport(localEndpoint_, registration.handle, &memDesc, &memDescLen);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma QueryMemoryKey HcommMemExport failed for addr: " << std::hex << addr
                                                                                   << " ret: " << ret);
        return ret;
    }

    // Parse exported desc to recover hcommDesc pointer and len
    UrmaExportDesc exportDesc{};
    const uint8_t *hcommDesc = nullptr;
    uint32_t hcommDescLen = 0;
    if (!DeserializeExportDesc(memDesc, memDescLen, exportDesc, &hcommDesc, &hcommDescLen)) {
        BM_LOG_ERROR("device_urma QueryMemoryKey DeserializeExportDesc failed");
        return BM_ERROR;
    }

    // Export flag descriptor via raw HCOMM API
    void *flagDescRaw = nullptr;
    uint32_t flagDescLenRaw = 0;
    ret = DlHcommApi::HcommMemExport(localEndpoint_->hcommEndpoint, devTransFlagHcommHandle_, &flagDescRaw,
                                     &flagDescLenRaw);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma QueryMemoryKey flag HcommMemExport failed, ret: " << ret);
        return BM_DL_FUNCTION_FAILED;
    }
    if (flagDescRaw == nullptr || flagDescLenRaw == 0) {
        BM_LOG_ERROR("device_urma QueryMemoryKey flag export returned null or zero length");
        return BM_ERROR;
    }
    const uint8_t *flagDesc = static_cast<const uint8_t *>(flagDescRaw);
    const uint32_t flagDescLen = flagDescLenRaw;

    // Serialize into key payload
    key.keys[0] = URMA_EXPORT_DESC_MAGIC;
    uint64_t gva = HybmVaManager::GetInstance().TransformVa(registration.mr.addr, HVM_HVA, HVM_GVA);
    key.keys[1] = (gva != 0) ? gva : registration.mr.addr; // 要导出gva

    UrmaExportDesc keyExportDesc = exportDesc;
    keyExportDesc.devTransFlagDescLen = flagDescLen;

    uint8_t *payload = reinterpret_cast<uint8_t *>(&key.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    constexpr uint32_t exportHeaderSize = sizeof(UrmaExportDesc);
    std::memcpy(payload, &keyExportDesc, exportHeaderSize);
    std::memcpy(payload + exportHeaderSize, hcommDesc, hcommDescLen);
    std::memcpy(payload + exportHeaderSize + hcommDescLen, flagDesc, flagDescLen);

    return BM_OK;
}

void DeviceUrmaTransportManager::UpdateMemoryKey(TransportMemoryKey &key, void *addr)
{
    if (addr != nullptr) {
        key.keys[1] = reinterpret_cast<uint64_t>(addr);
    }
}

Result DeviceUrmaTransportManager::ImportRemoteMemKeysLocked(uint32_t peerRank, RemoteRankState &state,
                                                             const std::vector<TransportMemoryKey> &memKeys)
{
    if (memKeys.empty()) {
        return BM_OK;
    }
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked localEndpoint_ is null, peer: " << peerRank);
        return BM_NOT_INITIALIZED;
    }
    const auto endpointLoc = state.remoteEndpointDesc.loc.locType;
    if (endpointLoc != ENDPOINT_LOC_TYPE_HOST && endpointLoc != ENDPOINT_LOC_TYPE_DEVICE) {
        BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked unsupported endpoint location, peer: "
                     << peerRank << " locType: " << endpointLoc);
        return BM_INVALID_PARAM;
    }
    const bool hostPeer = endpointLoc == ENDPOINT_LOC_TYPE_HOST;
    const auto expectedRemoteType = hostPeer ? UrmaMemoryType::HOST_DRAM : UrmaMemoryType::DEVICE_HBM;

    // Collect newly imported registrations in a local vector;
    // commit to state.imports only after all keys succeed.
    std::vector<RemoteRegistration> newImports;
    bool flagImportedInThisCall = false;
    auto rollbackNewImports = [&]() {
        if (flagImportedInThisCall && !state.remoteFlagDescBytes.empty()) {
            (void)DlHcommApi::HcommMemUnimport(localEndpoint_->hcommEndpoint, state.remoteFlagDescBytes.data(),
                                               static_cast<uint32_t>(state.remoteFlagDescBytes.size()));
            state.remoteFlagDescBytes.clear();
            state.remoteFlagAddr = 0;
            state.remoteFlagSize = 0;
            flagImportedInThisCall = false;
        }
        for (const auto &ni : newImports) {
            if (!ni.descBytes.empty()) {
                (void)manager_.HcommMemUnimport(localEndpoint_, ni.descBytes.data(),
                                                static_cast<uint32_t>(ni.descBytes.size()));
            }
        }
        newImports.clear();
    };

    for (const auto &key : memKeys) {
        // --- 1. Validate top-level magic ---
        if (key.keys[0] != URMA_EXPORT_DESC_MAGIC) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked invalid key magic 0x"
                         << std::hex << key.keys[0] << ", peer: " << peerRank << ", expected: 0x"
                         << URMA_EXPORT_DESC_MAGIC);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        const uint64_t remoteAddr = key.keys[1];
        if (remoteAddr == 0) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked zero addr(" << remoteAddr
                                                                            << ") in key, peer: " << peerRank);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        // --- 2. Parse UrmaExportDesc header from payload after header slots ---
        const uint8_t *raw = reinterpret_cast<const uint8_t *>(&key.keys[DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
        UrmaExportDesc exportDesc{};
        std::memcpy(&exportDesc, raw, sizeof(UrmaExportDesc));

        if (exportDesc.magic != URMA_EXPORT_DESC_MAGIC || exportDesc.version != URMA_EXPORT_DESC_VERSION) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked invalid UrmaExportDesc magic/version in key, peer: "
                         << peerRank);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }
        if (exportDesc.headerSize != sizeof(UrmaExportDesc)) {
            BM_LOG_ERROR(
                "device_urma ImportRemoteMemKeysLocked UrmaExportDesc headerSize mismatch, peer: " << peerRank);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }
        if (exportDesc.hcommDescLen == 0) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked hcommDescLen is 0, peer: " << peerRank);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }
        if (exportDesc.size == 0) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked exportDesc.size is 0, peer: " << peerRank);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        const uint64_t remoteSize = exportDesc.size;
        const uint64_t memTag = exportDesc.memTag;
        const uint32_t memDescLen = sizeof(UrmaExportDesc) + exportDesc.hcommDescLen;
        if (exportDesc.memoryType != expectedRemoteType || (hostPeer && exportDesc.addr != remoteAddr)) {
            BM_LOG_ERROR("device_urma import export descriptor mismatch, peer: "
                         << peerRank << " memTag: " << memTag << " endpointLoc: " << endpointLoc << " keyAddr: 0x"
                         << std::hex << remoteAddr << " descAddr: 0x" << exportDesc.addr << std::dec
                         << " descType: " << exportDesc.memoryType << " expectedType: " << expectedRemoteType);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }
        if (memDescLen + exportDesc.devTransFlagDescLen > DEVICE_URMA_EXPORT_KEY_DATA_BYTES) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked total payload exceeds key capacity, "
                         << "peer: " << peerRank << " memDescLen: " << memDescLen
                         << " flagDescLen: " << exportDesc.devTransFlagDescLen);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        // --- 3. Idempotency: skip if already imported (by memTag) ---
        auto it = std::find_if(state.imports.begin(), state.imports.end(),
                               [memTag](const auto &r) { return r.memTag == memTag; });
        if (it != state.imports.end()) {
            BM_LOG_DEBUG("device_urma ImportRemoteMemKeysLocked skip duplicate memTag: " << memTag
                                                                                         << ", peer: " << peerRank);
            continue;
        }

        // --- 4. Protocol compatibility check before import ---
        if (state.remoteEndpointDesc.protocol != localEndpoint_->desc.protocol) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked protocol mismatch, peer: "
                         << peerRank << " remote protocol: " << state.remoteEndpointDesc.protocol
                         << " local protocol: " << localEndpoint_->desc.protocol);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        // --- 5. HcommMemImport using global localEndpoint_ ---
        UrmaCommMem view{};
        auto ret = manager_.HcommMemImport(localEndpoint_, raw, memDescLen, &view);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked HcommMemImport failed, memTag: "
                         << memTag << " peer: " << peerRank << " ret: " << ret);
            rollbackNewImports();
            return ret;
        }

        if (view.type != UrmaMemoryType::DEVICE_HBM || (hostPeer && view.addr != remoteAddr) ||
            view.size < remoteSize) {
            BM_LOG_ERROR("device_urma import local view mismatch, peer: "
                         << peerRank << " memTag: " << memTag << " endpointLoc: " << endpointLoc << " keyAddr: 0x"
                         << std::hex << remoteAddr << " viewAddr: 0x" << view.addr << std::dec
                         << " keySize: " << remoteSize << " viewSize: " << view.size << " viewType: " << view.type
                         << " expectedViewType: " << UrmaMemoryType::DEVICE_HBM);
            (void)manager_.HcommMemUnimport(localEndpoint_, raw, memDescLen);
            rollbackNewImports();
            return BM_INVALID_PARAM;
        }

        // --- 6. Build RemoteRegistration ---
        RemoteRegistration remote{};
        remote.addr = remoteAddr;
        remote.size = remoteSize;
        remote.memTag = memTag;
        remote.exportedMemoryType = exportDesc.memoryType;
        remote.descBytes.assign(raw, raw + memDescLen);
        remote.view = view;
        newImports.emplace_back(std::move(remote));

        // --- 7. Import flag desc from UrmaExportDesc (if present and not yet imported for this peer) ---
        if (exportDesc.devTransFlagDescLen > 0 && !flagImportedInThisCall && state.remoteFlagAddr == 0) {
            const uint8_t *flagRaw = raw + sizeof(UrmaExportDesc) + exportDesc.hcommDescLen;
            HcommCommMem flagOutMem{};
            const auto flagRet = DlHcommApi::HcommMemImport(localEndpoint_->hcommEndpoint, flagRaw,
                                                            exportDesc.devTransFlagDescLen, &flagOutMem);
            if (flagRet != 0) {
                BM_LOG_ERROR("device_urma ImportRemoteMemKeysLocked HcommMemImport for flag failed, peer: "
                             << peerRank << " ret: " << flagRet);
                rollbackNewImports();
                return BM_DL_FUNCTION_FAILED;
            }
            state.remoteFlagAddr = reinterpret_cast<uint64_t>(flagOutMem.addr);
            state.remoteFlagSize = flagOutMem.size;
            state.remoteFlagDescBytes.assign(flagRaw, flagRaw + exportDesc.devTransFlagDescLen);
            flagImportedInThisCall = true;
            BM_LOG_INFO("device_urma ImportRemoteMemKeysLocked imported flag, peer: "
                        << peerRank << " flagAddr: " << VaToStr(state.remoteFlagAddr)
                        << " flagSize: " << state.remoteFlagSize << " descLen: " << exportDesc.devTransFlagDescLen);
        }

        BM_LOG_INFO("device_urma ImportRemoteMemKeysLocked imported mem, peer: "
                    << peerRank << " memTag: " << memTag << " addr: " << VaToStr(remoteAddr) << " size: " << remoteSize
                    << " view: " << VaToStr(view.addr));
    }

    state.imports.insert(state.imports.end(), newImports.begin(), newImports.end());
    return BM_OK;
}

bool DeviceUrmaTransportManager::IsBatchCopyRouteEnabledLocked() const
{
    return (options_.protocol & HYBM_DOP_TYPE_HOST_DEVICE_URMA) != 0U;
}

Result DeviceUrmaTransportManager::ValidateBatchCopyPeerLocked(uint32_t peerRank, const RemoteRankState &state,
                                                               uint32_t endpointLocType) const
{
    const bool validEndpoint =
        state.hasEndpointDesc && static_cast<uint32_t>(state.remoteEndpointDesc.loc.locType) == endpointLocType;
    const bool validResources = state.thread != 0 && state.channel != 0 && state.remoteFlagAddr != 0 &&
                                state.remoteFlagSize == sizeof(uint64_t) && !state.remoteFlagDescBytes.empty();
    if (!validEndpoint || !validResources) {
        BM_LOG_ERROR("device_urma invalid BatchCopy peer resources, peer: "
                     << peerRank << " locType: " << state.remoteEndpointDesc.loc.locType
                     << " expectedLocType: " << endpointLocType << " hasEndpointDesc: " << state.hasEndpointDesc
                     << " thread: " << state.thread << " channel: " << state.channel << " flagAddr: 0x" << std::hex
                     << state.remoteFlagAddr << std::dec << " flagSize: " << state.remoteFlagSize);
        return BM_INVALID_PARAM;
    }
    if (state.imports.empty() || state.imports.size() > BATCH_COPY_MAX_RANGE_PER_PEER) {
        BM_LOG_ERROR("device_urma invalid BatchCopy range count, peer: " << peerRank
                                                                         << " count: " << state.imports.size());
        return BM_INVALID_PARAM;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::BuildBatchCopyRouteSourceLocked(uint32_t peerRank, const RemoteRankState &state,
                                                                   uint32_t endpointLocType,
                                                                   BatchCopyRouteSource &source) const
{
    const auto expectedRemoteType =
        endpointLocType == ENDPOINT_LOC_TYPE_HOST ? UrmaMemoryType::HOST_DRAM : UrmaMemoryType::DEVICE_HBM;
    constexpr auto expectedLocalViewType = UrmaMemoryType::DEVICE_HBM;
    auto ret = ValidateBatchCopyPeerLocked(peerRank, state, endpointLocType);
    if (ret != BM_OK) {
        return ret;
    }

    source = {};
    source.peerRank = peerRank;
    source.thread = state.thread;
    source.channel = state.channel;
    source.remoteFlagAddr = state.remoteFlagAddr;
    try {
        source.ranges.reserve(state.imports.size());
        for (const auto &registration : state.imports) {
            uint64_t gvaEnd = 0;
            uint64_t hcommEnd = 0;
            const UrmaCommMem gvaMem{registration.addr, registration.size, registration.exportedMemoryType};
            const UrmaCommMem hcommMem{registration.view.addr, registration.size, registration.view.type};
            const bool invalid =
                registration.exportedMemoryType != expectedRemoteType ||
                registration.view.type != expectedLocalViewType || registration.view.addr == 0 ||
                registration.view.size < registration.size || !GetRangeEnd(gvaMem, gvaEnd) ||
                !GetRangeEnd(hcommMem, hcommEnd) ||
                (endpointLocType == ENDPOINT_LOC_TYPE_HOST && registration.view.addr != registration.addr);
            if (invalid) {
                BM_LOG_ERROR("device_urma invalid BatchCopy memory range, peer: "
                             << peerRank << " memTag: " << registration.memTag << " gva: 0x" << std::hex
                             << registration.addr << " view: 0x" << registration.view.addr << std::dec
                             << " size: " << registration.size << " viewSize: " << registration.view.size
                             << " remoteType: " << registration.exportedMemoryType
                             << " expectedRemoteType: " << expectedRemoteType << " viewType: " << registration.view.type
                             << " expectedViewType: " << expectedLocalViewType);
                return BM_INVALID_PARAM;
            }
            source.ranges.push_back({registration.addr, gvaEnd, registration.view.addr});
        }
    } catch (...) {
        BM_LOG_ERROR("device_urma allocate BatchCopy route ranges failed, peer: " << peerRank << " rangeCount: "
                                                                                  << state.imports.size());
        return BM_MALLOC_FAILED;
    }
    return BM_OK;
}

Result
DeviceUrmaTransportManager::BuildBatchCopyRouteSourcesForPeersLocked(const std::vector<uint32_t> &peerRanks,
                                                                     std::vector<BatchCopyRouteSource> &sources) const
{
    if (peerRanks.empty() || peerRanks.size() > BATCH_COPY_MAX_PEER_COUNT || peerRanks.size() != remoteRanks_.size()) {
        BM_LOG_ERROR("device_urma invalid BatchCopy peer set, rank: " << rankId_ << " requested: " << peerRanks.size()
                                                                      << " prepared: " << remoteRanks_.size());
        return BM_INVALID_PARAM;
    }
    std::vector<uint32_t> sortedRanks = peerRanks;
    std::sort(sortedRanks.begin(), sortedRanks.end());
    sources.clear();
    try {
        sources.reserve(sortedRanks.size());
    } catch (...) {
        BM_LOG_ERROR("device_urma allocate BatchCopy peer sources failed, rank: " << rankId_ << " peerCount: "
                                                                                  << sortedRanks.size());
        return BM_MALLOC_FAILED;
    }

    bool hasRouteLocation = false;
    uint32_t routeLocation = 0;
    for (const auto peerRank : sortedRanks) {
        const auto stateIt = remoteRanks_.find(peerRank);
        if (stateIt == remoteRanks_.end()) {
            BM_LOG_ERROR("device_urma BatchCopy peer is not prepared, rank: " << rankId_ << " peer: " << peerRank);
            return BM_NOT_CONNECTED;
        }
        const auto location = static_cast<uint32_t>(stateIt->second.remoteEndpointDesc.loc.locType);
        if (location != ENDPOINT_LOC_TYPE_HOST && location != ENDPOINT_LOC_TYPE_DEVICE) {
            BM_LOG_ERROR("device_urma invalid BatchCopy endpoint location, rank: " << rankId_ << " peer: " << peerRank
                                                                                   << " locType: " << location);
            return BM_INVALID_PARAM;
        }
        if (hasRouteLocation && location != routeLocation) {
            BM_LOG_ERROR("device_urma mixed BatchCopy endpoint locations, rank: " << rankId_ << " peer: " << peerRank
                                                                                  << " expected: " << routeLocation
                                                                                  << " actual: " << location);
            return BM_INVALID_PARAM;
        }
        routeLocation = location;
        hasRouteLocation = true;
        BatchCopyRouteSource source{};
        const auto ret = BuildBatchCopyRouteSourceLocked(peerRank, stateIt->second, routeLocation, source);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma build BatchCopy route source failed, rank: " << rankId_ << " peer: " << peerRank
                                                                                   << " ret: " << ret);
            return ret;
        }
        sources.emplace_back(std::move(source));
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::BuildBatchCopyRouteSourcesLocked(const HybmTransPrepareOptions &options,
                                                                    std::vector<BatchCopyRouteSource> &sources) const
{
    std::vector<uint32_t> peerRanks;
    for (const auto &item : options.options) {
        if (item.first == rankId_) {
            continue;
        }
        peerRanks.push_back(item.first);
    }
    if (peerRanks.empty()) {
        BM_LOG_ERROR("device_urma BatchCopy route has no remote peer, rank: " << rankId_);
        return BM_INVALID_PARAM;
    }
    return BuildBatchCopyRouteSourcesForPeersLocked(peerRanks, sources);
}

Result DeviceUrmaTransportManager::BuildBatchCopyRouteSourcesLocked(std::vector<BatchCopyRouteSource> &sources) const
{
    std::vector<uint32_t> peerRanks;
    try {
        peerRanks.reserve(remoteRanks_.size());
        for (const auto &item : remoteRanks_) {
            peerRanks.push_back(item.first);
        }
    } catch (...) {
        BM_LOG_ERROR("device_urma allocate BatchCopy peer rank list failed, rank: " << rankId_ << " peerCount: "
                                                                                    << remoteRanks_.size());
        return BM_MALLOC_FAILED;
    }
    return BuildBatchCopyRouteSourcesForPeersLocked(peerRanks, sources);
}

Result DeviceUrmaTransportManager::TryPublishBatchCopyRouteLocked(const HybmTransPrepareOptions &options)
{
    if (!IsBatchCopyRouteEnabledLocked()) {
        BM_LOG_DEBUG("No need publish route");
        return BM_OK;
    }
    if (routePublisher_ != nullptr && routePublisher_->IsPublished()) {
        BM_LOG_DEBUG("BatchCopy route is immutable after the first successful publication.");
        return BM_OK;
    }
    if (!HasBatchCopyMemoryKeys(options)) {
        BM_LOG_DEBUG("memoryKeys is empty.");
        return BM_OK;
    }
    for (const auto &item : options.options) {
        if (item.first != rankId_ && item.second.memKeys.empty()) {
            BM_LOG_ERROR("device_urma BatchCopy route peer memory keys are incomplete, rank: " << rankId_ << " peer: "
                                                                                               << item.first);
            return BM_NOT_CONNECTED;
        }
    }
    std::vector<BatchCopyRouteSource> sources;
    auto ret = BuildBatchCopyRouteSourcesLocked(options, sources);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma build BatchCopy route sources failed, rank: " << rankId_ << " userDeviceId: "
                                                                                << userDeviceId_ << " ret: " << ret);
        return ret;
    }
    try {
        if (routePublisher_ == nullptr) {
            routePublisher_ = std::make_unique<BatchCopyRoutePublisher>(userDeviceId_, localEndpoint_, manager_);
        }
    } catch (...) {
        BM_LOG_ERROR("device_urma allocate BatchCopy route publisher failed, rank: " << rankId_ << " userDeviceId: "
                                                                                     << userDeviceId_);
        return BM_MALLOC_FAILED;
    }
    ret = routePublisher_->Publish(sources);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma publish BatchCopy route failed, rank: " << rankId_ << " userDeviceId: "
                                                                          << userDeviceId_ << " ret: " << ret);
    }
    return ret;
}

Result DeviceUrmaTransportManager::Prepare(const HybmTransPrepareOptions &options)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    BM_VALIDATE_RETURN(localEndpoint_ != nullptr,
                       "evice_urma Prepare failed: localEndpoint_ is null (OpenDevice may not have completed)",
                       BM_NOT_INITIALIZED);
    for (const auto &item : options.options) {
        const uint32_t peerRank = item.first;
        if (peerRank >= rankCount_) {
            BM_LOG_ERROR("device_urma Prepare invalid peerRank: " << peerRank << " rankCount: " << rankCount_);
            return BM_INVALID_PARAM;
        }
        if (peerRank == rankId_) {
            BM_LOG_WARN("device_urma Prepare skipping self rank: " << peerRank);
            continue;
        }
        auto &state = remoteRanks_[peerRank];

        // 1. Parse peer UrmaEndpointDesc from privateData
        UrmaEndpointDesc peerDesc{};
        auto ret = ParseUrmaPrivateData(item.second.privateData, peerDesc);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma Prepare failed to parse peer endpoint desc, peer: " << peerRank);
            return ret;
        }

        bool resourcesCreatedInThisCall = false;

        // 2. Idempotency: if thread and channel already exist, compare with stored remoteEndpointDesc.
        //    If unchanged → skip resource creation; if changed → reject (no hot-replace).
        //    Either way, fall through to mem import — do NOT continue.
        if (state.channel != 0 && state.thread != 0) {
            if (std::memcmp(&state.remoteEndpointDesc, &peerDesc, sizeof(UrmaEndpointDesc)) != 0) {
                BM_LOG_ERROR(
                    "device_urma Prepare peer endpoint desc changed but hot-replace not supported, peer: " << peerRank);
                return BM_INVALID_PARAM;
            }
            BM_LOG_INFO("device_urma Prepare reusing existing channel/thread for peer: " << peerRank);
            // resourcesCreatedInThisCall remains false — do not destroy channel/thread on import failure
        } else {
            // 3. Convert peer UrmaEndpointDesc → HCOMM EndpointDesc for HcommChannelDesc.remoteEndpoint
            EndpointDesc hcommRemoteEndpoint = ToHcommEndpointDesc(peerDesc);

            // 4. Build a single HcommChannelDesc for this peer
            HcommChannelDesc channelDesc{};
            HcommChannelDescInit(&channelDesc, 1); // not in DlHcommApi namespace, direct C function
            channelDesc.role = (rankId_ > peerRank) ? HCOMM_SOCKET_ROLE_CLIENT : HCOMM_SOCKET_ROLE_SERVER;
            channelDesc.remoteEndpoint = hcommRemoteEndpoint;
            channelDesc.notifyNum = HCOMM_NORMAL_NOTIFY_NUM;
            channelDesc.exchangeAllMems = true; // 填true, 不用管memHandles了, remoteEndpoint要填对
            if (localEndpoint_->desc.protocol == UrmaProtocol::UBOE) {
                // CRITICAL: HcommChannelDescInit sets union to 0xFF garbage values.
                // Must zero the entire union before setting ubAttr to avoid:
                // - queueNum=0xFFFFFFFF (4B QPs → OOM)
                // - retryCnt=0xFFFFFFFF (impossible retries → timeout)
                // - tc/sl=0xFF (invalid QoS → init failure)
                std::memset(channelDesc.raws, 0, sizeof(channelDesc.raws));
                // sqDepth合法范围[16,256]且需为2的幂，0会被CheckUbAttr拒绝；128对齐hcomm MS模式默认值
                channelDesc.ubAttr.sqDepth = 128;
            }

            // 5. Allocate one thread per peer (use temporary variable for safe rollback)
            HcommThreadHandle threadHandle = 0;
            ret = DlHcommApi::HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &HCOMM_NORMAL_NOTIFY_NUM, &threadHandle);
            if (ret != BM_OK) {
                BM_LOG_ERROR("device_urma Prepare HcommThreadAlloc failed, peer: "
                             << peerRank << " engine: " << COMM_ENGINE_AICPU_TS << " ret: " << ret);
                return ret;
            }
            if (threadHandle == 0) {
                BM_LOG_ERROR("device_urma Prepare HcommThreadAlloc returned invalid thread, peer: " << peerRank);
                return BM_DL_FUNCTION_FAILED;
            }

            // 6. Create one channel per peer (temporary variable for safe rollback)
            HcommChannelHandle channelHandle = 0;
            BM_LOG_INFO("device_urma Prepare HcommChannelCreate peerRank: " << peerRank << ", channelDesc.role: "
                                                                            << channelDesc.role);
            ret = DlHcommApi::HcommChannelCreate(localEndpoint_->hcommEndpoint, COMM_ENGINE_AICPU, &channelDesc, 1,
                                                 &channelHandle);
            if (ret != 0) {
                BM_LOG_ERROR("device_urma Prepare HcommChannelCreate failed, peer: " << peerRank << " ret: " << ret);
                auto rollbackThread = threadHandle;
                // No stream sync needed: thread was just allocated, no RemoteIO launched during Prepare.
                (void)DlHcommApi::HcommThreadFree(&rollbackThread, 1);
                return BM_DL_FUNCTION_FAILED;
            }
            if (channelHandle == 0) {
                BM_LOG_ERROR("device_urma Prepare HcommChannelCreate returned invalid channel, peer: " << peerRank);
                auto rollbackThread = threadHandle;
                // No stream sync needed: thread was just allocated, no RemoteIO launched during Prepare.
                (void)DlHcommApi::HcommThreadFree(&rollbackThread, 1);
                return BM_DL_FUNCTION_FAILED;
            }

            // 7. Persist new resources to state
            state.remoteEndpointDesc = peerDesc;
            state.hasEndpointDesc = true;
            state.thread = threadHandle;
            state.channelDesc = channelDesc;
            state.channel = channelHandle;
            resourcesCreatedInThisCall = true;

            BM_LOG_INFO("device_urma Prepare created channel/thread, peer: " << peerRank << " thread: " << threadHandle
                                                                             << " channel: " << channelHandle);

            // 7.5 Wait for channel ready before importing mem keys
            ret = manager_.WaitForChannelReady(channelHandle, peerRank);
            if (ret != BM_OK) {
                BM_LOG_ERROR("device_urma Prepare WaitForChannelReady failed, peer: " << peerRank << " ret: " << ret);
                auto rbRet = DlHcommApi::HcommChannelDestroy(&channelHandle, 1);
                if (rbRet != 0) {
                    BM_LOG_ERROR("device_urma Prepare rollback HcommChannelDestroy failed, "
                                 << "channel: " << channelHandle << " peer: " << peerRank << " ret: " << rbRet);
                }
                auto trRet = DlHcommApi::HcommThreadFree(&threadHandle, 1);
                if (trRet != 0) {
                    BM_LOG_ERROR("device_urma Prepare rollback HcommThreadFree failed, "
                                 << "thread: " << threadHandle << " peer: " << peerRank << " ret: " << trRet);
                }
                remoteRanks_.erase(peerRank);
                return ret;
            }
        }

        // 8. Import remote memory keys (always, even if channel/thread were reused)
        ret = ImportRemoteMemKeysLocked(peerRank, state, item.second.memKeys);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma Prepare ImportRemoteMemKeysLocked failed, peer: " << peerRank);
            if (resourcesCreatedInThisCall) {
                // Rollback the channel/thread that were just created in this call
                BM_LOG_WARN("device_urma Prepare rolling back newly created channel/thread for peer: " << peerRank);
                if (state.channel != 0) {
                    auto chanRollback = state.channel;
                    // No stream sync needed: channel was just created, no RemoteIO launched during Prepare.
                    (void)DlHcommApi::HcommChannelDestroy(&chanRollback, 1);
                    state.channel = 0;
                }
                if (state.thread != 0) {
                    auto threadRollback = state.thread;
                    (void)DlHcommApi::HcommThreadFree(&threadRollback, 1);
                    state.thread = 0;
                }
                state.channelDesc = {};
                state.remoteEndpointDesc = UrmaEndpointDesc{};
                state.hasEndpointDesc = false;
            }
            // If resourcesCreatedInThisCall is false, ImportRemoteMemKeysLocked already
            // rolled back its own partial imports; we leave pre-existing channel/thread intact.
            return ret;
        }

        BM_LOG_INFO("device_urma Prepare success, peer: " << peerRank << " thread: " << state.thread << " channel: "
                                                          << state.channel << " imports: " << state.imports.size());
    }
    return TryPublishBatchCopyRouteLocked(options);
}

Result DeviceUrmaTransportManager::RemoveRankLocked(uint32_t rankId)
{
    auto rankIt = remoteRanks_.find(rankId);
    if (rankIt == remoteRanks_.end()) {
        BM_LOG_WARN("device_urma RemoveRankLocked rank not found, localRank: "
                    << rankId_ << " peerRank: " << rankId << " remoteRanksSize: " << remoteRanks_.size());
        return BM_OK;
    }
    auto &state = rankIt->second;
    BM_LOG_INFO("device_urma RemoveRankLocked found entry, localRank: "
                << rankId_ << " peerRank: " << rankId << " channel: " << state.channel << " thread: " << state.thread);
    Result finalRet = DestroyRankChannelsAndThread(state, rankId);

    const auto retImports = UnimportPeerImportsAndFlag(state, rankId);
    if (retImports != BM_OK && finalRet == BM_OK) {
        finalRet = retImports;
    }
    remoteRanks_.erase(rankIt);
    return finalRet;
}

bool DeviceUrmaTransportManager::IsAnyRegistryContextPendingForRank(uint32_t rankId) const
{
    for (const auto &ctxSp : registry_) {
        if (!ctxSp) {
            continue;
        }
        for (const auto &pt : ctxSp->pendingTransfers) {
            if (pt.inFlight && pt.rankId == rankId) {
                return true;
            }
        }
    }
    return false;
}

Result DeviceUrmaTransportManager::RemoveRanks(const std::vector<uint32_t> &removedRanks)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_LOG_INFO("device_urma RemoveRanks called, localRank: " << rankId_ << " ranks: " << removedRanks.size()
                                                              << " opened: " << opened_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    if (routePublisher_ != nullptr && routePublisher_->IsPublished()) {
        BM_LOG_WARN("device_urma RemoveRanks is not supported after BatchCopy route publication, rank: "
                     << rankId_ << " removedCount: " << removedRanks.size()); // TODO  此处为什么会进入到 remove ranks
    }

    // Atomic pending preflight: any target rank with pending ops → reject all
    for (auto rankId : removedRanks) {
        BM_LOG_INFO("device_urma RemoveRanks checking IsAnyRegistryContextPendingForRank, rankId: " << rankId);
        if (IsAnyRegistryContextPendingForRank(rankId)) {
            BM_LOG_ERROR("device_urma RemoveRanks: rank " << rankId << " has pending ops, rejecting all");
            return BM_ERROR;
        }
    }

    Result finalRet = BM_OK;
    for (auto rankId : removedRanks) {
        BM_LOG_INFO("device_urma RemoveRanks calling RemoveRankLocked, rankId: " << rankId);
        auto ret = RemoveRankLocked(rankId);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma RemoveRanks RemoveRankLocked failed, localRank: " << rankId_ << " peerRank: "
                                                                                        << rankId << " ret: " << ret);
            if (finalRet == BM_OK) {
                finalRet = ret;
            }
        }
    }
    if (finalRet != BM_OK) {
        BM_LOG_ERROR("device_urma RemoveRanks cleanup failed, localRank: "
                     << rankId_ << " removedRanks: " << removedRanks.size() << " ret: " << finalRet);
    }
    return finalRet;
}

Result DeviceUrmaTransportManager::Connect()
{
    connected_ = true;
    return BM_OK;
}

Result DeviceUrmaTransportManager::AsyncConnect()
{
    return BM_OK;
}

Result DeviceUrmaTransportManager::WaitForConnected(int64_t timeoutNs)
{
    (void)timeoutNs;
    return BM_OK;
}

Result DeviceUrmaTransportManager::UpdateRankOptions(const HybmTransPrepareOptions &options)
{
    bool needFallback = false;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
        if (localEndpoint_ == nullptr) {
            BM_LOG_ERROR("device_urma UpdateRankOptions failed: localEndpoint_ is null");
            return BM_NOT_INITIALIZED;
        }
        for (const auto &item : options.options) {
            const uint32_t peerRank = item.first;
            if (peerRank >= rankCount_) {
                BM_LOG_ERROR("device_urma UpdateRankOptions invalid peerRank: " << peerRank
                                                                                << " rankCount: " << rankCount_);
                return BM_INVALID_PARAM;
            }
            if (peerRank == rankId_) {
                BM_LOG_WARN("device_urma UpdateRankOptions skipping self rank: " << peerRank);
                continue;
            }
            auto stateIt = remoteRanks_.find(peerRank);
            if (stateIt == remoteRanks_.end()) {
                BM_LOG_WARN("device_urma UpdateRankOptions peer rank " << peerRank
                                                                       << " not prepared yet, fallback to Prepare");
                needFallback = true;
                break;
            }
            auto &state = stateIt->second;
            if (state.channel == 0 || state.thread == 0) {
                BM_LOG_WARN("device_urma UpdateRankOptions peer rank "
                            << peerRank << " has no channel/thread, fallback to Prepare");
                needFallback = true;
                break;
            }
        }
    } // mutex_ released

    if (needFallback) {
        // Fallback to Prepare (without holding mutex_) — it will create resources and import memKeys.
        BM_LOG_INFO("device_urma UpdateRankOptions falling back to Prepare for new dynamic ranks");
        return Prepare(options);
    }

    // Only import remote memory keys without re-creating resources.
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto &item : options.options) {
        const uint32_t peerRank = item.first;
        if (peerRank >= rankCount_ || peerRank == rankId_) {
            continue; // already validated/skipped above
        }
        auto stateIt = remoteRanks_.find(peerRank);
        if (stateIt == remoteRanks_.end()) {
            continue;
        }
        auto &state = stateIt->second;
        auto ret = ImportRemoteMemKeysLocked(peerRank, state, item.second.memKeys);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma UpdateRankOptions ImportRemoteMemKeysLocked failed, peer: " << peerRank);
            return ret;
        }
        BM_LOG_INFO("device_urma UpdateRankOptions success, peer: " << peerRank);
    }
    return TryPublishBatchCopyRouteLocked(options);
}

const std::string &DeviceUrmaTransportManager::GetNic() const
{
    static const std::string emptyNic;
    return emptyNic;
}

const TransportPrivateData DeviceUrmaTransportManager::GetPrivateData() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    TransportPrivateData data{};
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("device_urma GetPrivateData called before localEndpoint_ is ready, returning empty");
        return data; // Empty data is rejected by ParseUrmaPrivateData on the peer.
    }
    const auto ret = SerializeUrmaPrivateData(localEndpointDesc_, data);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma GetPrivateData failed to serialize endpoint, rankId: " << rankId_
                                                                                         << ", ret: " << ret);
    }
    return data;
}

Result DeviceUrmaTransportManager::StageAndLaunchTransfer(CompletionContext &ctx, RemoteRankState &state, bool isRead,
                                                          const std::vector<uint64_t> &localVec,
                                                          const std::vector<uint64_t> &remoteVec,
                                                          const std::vector<uint64_t> &sizeVec, uint32_t rankId)
{
    ctx.pendingTransfers.emplace_back();
    auto &pt = ctx.pendingTransfers.back();
    pt.rankId = rankId;

    auto ret = PrepareKernelLaunchBuffers(isRead, localVec, remoteVec, sizeVec, pt.buffers);
    if (ret != BM_OK) {
        auto releaseRet = ReleaseDeviceTransferBuffers(pt.buffers);
        if (releaseRet != BM_OK) {
            BM_LOG_ERROR("device_urma StageAndLaunchTransfer ReleaseDeviceTransferBuffers failed, rank: "
                         << rankId << " ret: " << releaseRet);
            return ret;
        }
        ctx.pendingTransfers.pop_back();
        return ret;
    }

    std::lock_guard<std::mutex> lock_guard(state.rankMutex);
    ret = LaunchDeviceKernelBatch(pt.buffers, state.thread, isRead, state.channel, localVec.size());
    if (ret != BM_OK) {
        auto releaseRet = ReleaseDeviceTransferBuffers(pt.buffers);
        if (releaseRet != BM_OK) {
            BM_LOG_ERROR("device_urma StageAndLaunchTransfer ReleaseDeviceTransferBuffers failed, rank: "
                         << rankId << " ret: " << releaseRet);
            return ret;
        }
        ctx.pendingTransfers.pop_back();
        return ret;
    }

    pt.inFlight = true;
    return BM_OK;
}

Result DeviceUrmaTransportManager::RemoteIo(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size, bool write)
{
    if (size == 0) {
        return BM_OK;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    RemoteRegistration remote{};
    auto ret = FindRemoteRegistrationLocked(rankId, rAddr, size, &remote);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma remote address is not prepared, rank: " << rankId << " addr: " << std::hex << rAddr);
        return ret;
    }
    uint64_t correctedLAddr = lAddr;
    ret = CorrectLocalRegAddressLocked(lAddr, size, correctedLAddr);
    if (ret != BM_OK) {
        return ret;
    }
    auto &state = remoteRanks_[rankId];
    if (state.channel == 0 || state.thread == 0) {
        BM_LOG_ERROR("RemoteIo no channel/thread, rankId: " << rankId << " channel: " << state.channel
                                                            << " thread: " << state.thread << std::hex << " rAddr: 0x"
                                                            << rAddr << std::dec << " size: " << size);
        return BM_NOT_CONNECTED;
    }
    CompletionContext *ctx = LookupOrCreateContextLocked();
    if (ctx == nullptr) {
        BM_LOG_ERROR("device_urma RemoteIo LookupOrCreateContextLocked failed");
        return BM_ERROR;
    }
    const auto translatedRemoteAddr = remote.view.addr + (rAddr - remote.addr);
    const bool isRead = !write;

    return StageAndLaunchTransfer(*ctx, state, isRead, {correctedLAddr}, {translatedRemoteAddr}, {size}, rankId);
}

Result DeviceUrmaTransportManager::ReadRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size)
{
    auto ret = ReadRemoteAsync(rankId, lAddr, rAddr, size);
    if (ret != BM_OK) {
        return ret;
    }
    return Synchronize(rankId);
}

Result DeviceUrmaTransportManager::WriteRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size)
{
    auto ret = WriteRemoteAsync(rankId, lAddr, rAddr, size);
    if (ret != BM_OK) {
        return ret;
    }
    return Synchronize(rankId);
}

Result DeviceUrmaTransportManager::ReadRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size)
{
    return RemoteIo(rankId, lAddr, rAddr, size, false);
}

Result DeviceUrmaTransportManager::WriteRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size)
{
    return RemoteIo(rankId, lAddr, rAddr, size, true);
}

Result DeviceUrmaTransportManager::ResolveBatchIoAddressesLocked(uint32_t rankId, const CopyDescriptor &descriptor,
                                                                 std::vector<uint64_t> &localVec,
                                                                 std::vector<uint64_t> &remoteVec,
                                                                 std::vector<uint64_t> &sizeVec) const
{
    const auto &localAddrs = descriptor.localAddrs;
    const auto &globalAddrs = descriptor.globalAddrs;
    const auto &counts = descriptor.counts;
    const auto batchSize = counts.size();

    std::vector<uint64_t> tmpLocal;
    std::vector<uint64_t> tmpRemote;
    std::vector<uint64_t> tmpSize;
    tmpLocal.reserve(batchSize);
    tmpRemote.reserve(batchSize);
    tmpSize.reserve(batchSize);

    for (uint32_t i = 0; i < batchSize; ++i) {
        const uint64_t lAddr = reinterpret_cast<uint64_t>(localAddrs[i]);
        const uint64_t rAddr = reinterpret_cast<uint64_t>(globalAddrs[i]);
        const uint64_t size = counts[i];
        if (size == 0) {
            continue;
        }

        uint64_t correctedLAddr = lAddr;
        auto ret = CorrectLocalRegAddressLocked(lAddr, size, correctedLAddr);
        if (ret != BM_OK) {
            BM_LOG_DEBUG("device_urma ResolveBatchIoAddressesLocked local address correction failed, rank: "
                         << rankId << " addr: 0x" << std::hex << lAddr << " index: " << i);
            return ret;
        }
        RemoteRegistration remote{};
        ret = FindRemoteRegistrationLocked(rankId, rAddr, size, &remote);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma ResolveBatchIoAddresses remote not prepared, rank: "
                         << rankId << " addr: 0x" << std::hex << rAddr << " index: " << i);
            return ret;
        }
        const auto translatedRemoteAddr = remote.view.addr + (rAddr - remote.addr);
        tmpLocal.push_back(correctedLAddr);
        tmpRemote.push_back(translatedRemoteAddr);
        tmpSize.push_back(size);
    }
    localVec = std::move(tmpLocal);
    remoteVec = std::move(tmpRemote);
    sizeVec = std::move(tmpSize);
    return BM_OK;
}

Result DeviceUrmaTransportManager::RemoteIoBatch(uint32_t rankId, const CopyDescriptor &descriptor, bool write)
{
    const auto batchSize = descriptor.counts.size();
    if (batchSize == 0) {
        return BM_OK;
    }

    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    auto rankIt = remoteRanks_.find(rankId);
    if (rankIt == remoteRanks_.end()) {
        BM_LOG_ERROR("device_urma RemoteIoBatch rank not found: " << rankId);
        return BM_NOT_CONNECTED;
    }
    auto &state = rankIt->second;
    if (state.channel == 0 || state.thread == 0) {
        BM_LOG_ERROR("device_urma RemoteIoBatch no channel/thread, rankId: " << rankId << " channel: " << state.channel
                                                                             << " thread: " << state.thread);
        return BM_NOT_CONNECTED;
    }

    std::vector<uint64_t> localVec;
    std::vector<uint64_t> remoteVec;
    std::vector<uint64_t> sizeVec;
    auto ret = ResolveBatchIoAddressesLocked(rankId, descriptor, localVec, remoteVec, sizeVec);
    if (ret != BM_OK) {
        return ret;
    }
    if (sizeVec.empty()) {
        return BM_OK;
    }

    CompletionContext *ctx = LookupOrCreateContextLocked();
    if (ctx == nullptr) {
        BM_LOG_ERROR("device_urma RemoteIoBatch LookupOrCreateContextLocked failed");
        return BM_ERROR;
    }
    return StageAndLaunchTransfer(*ctx, state, !write, localVec, remoteVec, sizeVec, rankId);
}

Result DeviceUrmaTransportManager::WriteRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor)
{
    return RemoteIoBatch(rankId, descriptor, true);
}

Result DeviceUrmaTransportManager::ReadRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor)
{
    return RemoteIoBatch(rankId, descriptor, false);
}

aclrtFuncHandle DeviceUrmaTransportManager::GetDeviceKernelFunc(bool isRead) const
{
    return isRead ? deviceFuncHandles_.batchRead : deviceFuncHandles_.batchWrite;
}

Result DeviceUrmaTransportManager::ReleaseDeviceTransferBuffers(DeviceTransferBuffers &buffers)
{
    if (buffers.dstList == nullptr) {
        return BM_OK;
    }
    auto ret = DlAclApi::AclrtFree(buffers.dstList);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma ReleaseDeviceTransferBuffers AclrtFree failed, dstList: " << VaToStr(buffers.dstList)
                                                                                            << " ret: " << ret);
        return ret;
    }
    buffers.dstList = nullptr;
    buffers.srcList = nullptr;
    buffers.lenList = nullptr;
    return BM_OK;
}

Result DeviceUrmaTransportManager::PrepareKernelLaunchBuffers(bool isRead, const std::vector<uint64_t> &localAddrs,
                                                              const std::vector<uint64_t> &remoteAddrs,
                                                              const std::vector<uint64_t> &sizes,
                                                              DeviceTransferBuffers &outBuffers)
{
    const auto batchSize = localAddrs.size();
    const auto ptrBytes = batchSize * sizeof(void *);
    const auto lenBytes = batchSize * sizeof(uint64_t);
    const auto totalBytes = ptrBytes * 2UL + lenBytes;

    // Allocate host buffer FIRST to avoid leak if vector throws after AclrtMalloc
    std::vector<uint8_t> hostBuf;
    hostBuf.resize(totalBytes);

    auto ret = DlAclApi::AclrtMalloc(&outBuffers.dstList, totalBytes, 0);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma PrepareKernelLaunchBuffers alloc batch buffers failed, ret: " << ret);
        return ret;
    }
    outBuffers.srcList = static_cast<uint8_t *>(outBuffers.dstList) + ptrBytes;
    outBuffers.lenList = static_cast<uint8_t *>(outBuffers.dstList) + ptrBytes * 2UL;

    auto *dstBase = reinterpret_cast<void **>(hostBuf.data());
    auto *srcBase = reinterpret_cast<void **>(hostBuf.data() + ptrBytes);
    auto *lenBase = reinterpret_cast<uint64_t *>(hostBuf.data() + ptrBytes * 2UL);
    for (size_t i = 0; i < batchSize; ++i) {
        dstBase[i] = reinterpret_cast<void *>(isRead ? localAddrs[i] : remoteAddrs[i]);
        srcBase[i] = reinterpret_cast<void *>(isRead ? remoteAddrs[i] : localAddrs[i]);
        lenBase[i] = sizes[i];
    }

    ret = DlAclApi::AclrtMemcpy(outBuffers.dstList, totalBytes, hostBuf.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma PrepareKernelLaunchBuffers copy batch buffers failed, ret: " << ret);
        return ret;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::LaunchDeviceKernelBatch(const DeviceTransferBuffers &buffers,
                                                           HcommThreadHandle thread, bool isRead,
                                                           HcommChannelHandle channel, size_t batchSize)
{
    HybmOneSideOpParam args{};
    args.thread = thread;
    args.channel = channel;
    args.list_num = static_cast<uint32_t>(batchSize);
    args.dst_buf_addr_list = static_cast<void **>(buffers.dstList);
    args.src_buf_addr_list = static_cast<void **>(buffers.srcList);
    args.len_list = static_cast<uint64_t *>(buffers.lenList);
    aclrtArgsHandle argsHandle = nullptr;
    auto funcHandle = GetDeviceKernelFunc(isRead);
    auto ret = DlAclApi::AclrtKernelArgsInit(funcHandle, &argsHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelBatch AclrtKernelArgsInit failed, ret: " << ret);
        return ret;
    }
    aclrtParamHandle paramHandle = nullptr;
    ret = DlAclApi::AclrtKernelArgsAppend(argsHandle, &args, sizeof(args), &paramHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelBatch AclrtKernelArgsAppend failed, ret: " << ret);
        return ret;
    }
    ret = DlAclApi::AclrtKernelArgsFinalize(argsHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelBatch AclrtKernelArgsFinalize failed, ret: " << ret);
        return ret;
    }
    void *stream = HybmStreamManager::GetThreadAclStream();
    if (stream == nullptr) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelBatch GetThreadAclStream failed");
        return BM_DL_FUNCTION_FAILED;
    }

    aclrtLaunchKernelAttr attr{};
    attr.id = aclrtLaunchKernelAttrId::ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = HYBM_NOTIFY_DEFAULT_WAIT_TIME_S;
    aclrtLaunchKernelCfg cfg{};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    ret = DlAclApi::AclrtLaunchKernelWithConfig(funcHandle, HYBM_DEVICE_KERNEL_BLOCK_DIM, stream, &cfg, argsHandle,
                                                nullptr);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelBatch AclrtLaunchKernelWithConfig failed, kernel: "
                     << (isRead ? HYBM_DEVICE_FUNC_READ : HYBM_DEVICE_FUNC_WRITE) << " ret: " << ret);
        return ret;
    }

    ret = DlAclApi::AclrtSynchronizeStream(stream);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma launch kernel AclrtSynchronizeStream failed, ret: " << ret);
        return ret;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::LaunchDeviceKernelNotify(HcommThreadHandle thread, HcommChannelHandle channel,
                                                            uint64_t remoteFlagAddr, uint64_t notifyAddr,
                                                            uint32_t notifyLen)
{
    if (thread == 0 || channel == 0 || remoteFlagAddr == 0 || notifyAddr == 0 || notifyLen == 0) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify invalid param: thread="
                     << thread << " channel=" << channel << " remoteFlag=0x" << std::hex << remoteFlagAddr << std::dec
                     << " notifyAddr=0x" << std::hex << notifyAddr << std::dec << " notifyLen=" << notifyLen);
        return BM_INVALID_PARAM;
    }

    HybmOneSideOpParam args{};
    args.thread = thread;
    args.channel = channel;
    args.remote_flag_addr = remoteFlagAddr;
    args.local_flag_addr = notifyAddr;
    args.flag_size = notifyLen;

    aclrtArgsHandle argsHandle = nullptr;
    auto funcHandle = GetDeviceKernelFunc(true);
    auto ret = DlAclApi::AclrtKernelArgsInit(funcHandle, &argsHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify AclrtKernelArgsInit failed, ret: " << ret);
        return ret;
    }
    aclrtParamHandle paramHandle = nullptr;
    ret = DlAclApi::AclrtKernelArgsAppend(argsHandle, &args, sizeof(args), &paramHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify AclrtKernelArgsAppend failed, ret: " << ret);
        return ret;
    }
    ret = DlAclApi::AclrtKernelArgsFinalize(argsHandle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify AclrtKernelArgsFinalize failed, ret: " << ret);
        return ret;
    }
    void *stream = HybmStreamManager::GetThreadAclStream();
    if (stream == nullptr) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify GetThreadAclStream failed");
        return BM_DL_FUNCTION_FAILED;
    }

    aclrtLaunchKernelAttr attr{};
    attr.id = aclrtLaunchKernelAttrId::ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = HYBM_NOTIFY_DEFAULT_WAIT_TIME_S;
    aclrtLaunchKernelCfg cfg{};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    ret = DlAclApi::AclrtLaunchKernelWithConfig(funcHandle, HYBM_DEVICE_KERNEL_BLOCK_DIM, stream, &cfg, argsHandle,
                                                nullptr);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify AclrtLaunchKernelWithConfig failed, kernel: "
                     << HYBM_DEVICE_FUNC_READ << " ret: " << ret);
        return ret;
    }

    ret = DlAclApi::AclrtSynchronizeStream(stream);
    if (ret != BM_OK) {
        BM_LOG_ERROR("device_urma LaunchDeviceKernelNotify AclrtSynchronizeStream failed, ret: " << ret);
        return ret;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::SynchronizeRankPendingLocked(CompletionContext &ctx, RemoteRankState &state,
                                                                uint32_t rankId, bool hasInFlight)
{
    std::lock_guard<std::mutex> rankLock(state.rankMutex);
    if (hasInFlight) {
        auto ret =
            LaunchDeviceKernelNotify(state.thread, state.channel, state.remoteFlagAddr, ctx.notifyAddr, ctx.notifyLen);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma SynchronizeRankPending LaunchDeviceKernelNotify failed, rankId="
                         << rankId << " ret=" << ret);
            return ret;
        }
    }

    std::vector<PendingTransfer> rankPending;
    ExtractRankPending(ctx.pendingTransfers, rankId, rankPending);

    auto syncRet = SynchronizeContextLocked(ctx.notify, ctx.stream, rankPending);
    if (syncRet != BM_OK) {
        RestoreRankPending(rankPending, ctx.pendingTransfers);
        return syncRet;
    }
    return BM_OK;
}

Result DeviceUrmaTransportManager::Synchronize(uint32_t rankId)
{
    std::lock_guard<std::mutex> guard(mutex_);
    BM_VALIDATE_RETURN(opened_, "device_urma transport manager is not opened", BM_ERROR);
    auto rankIt = remoteRanks_.find(rankId);
    if (rankIt == remoteRanks_.end()) {
        BM_LOG_ERROR("device_urma Synchronize rank not found: " << rankId);
        return BM_NOT_CONNECTED;
    }

    CompletionContext *currentCtx = FindCurrentContextLocked();
    const uint32_t targetRankId = rankId;
    if (currentCtx == nullptr) {
        if (IsAnyRegistryContextPendingForRank(targetRankId)) {
            BM_LOG_ERROR("device_urma Synchronize no TLS context but rank " << targetRankId
                                                                            << " has pending in another context");
            return BM_ERROR;
        }
        return BM_OK;
    }

    // Check if current context has any entry for this rank and whether any are in-flight
    bool needSync = false;
    bool hasInFlight = false;
    for (const auto &pt : currentCtx->pendingTransfers) {
        if (pt.rankId == targetRankId) {
            needSync = true;
            if (pt.inFlight) {
                hasInFlight = true;
                break;
            }
        }
    }
    if (needSync) {
        return SynchronizeRankPendingLocked(*currentCtx, rankIt->second, targetRankId, hasInFlight);
    }

    // Current context has no pending for target rankId. If any OTHER context
    // has pending for target rankId → reject (foreign pending for same rank).
    for (const auto &ctxSp : registry_) {
        if (!ctxSp || ctxSp.get() == currentCtx) {
            continue;
        }
        for (const auto &pt : ctxSp->pendingTransfers) {
            if (pt.inFlight && pt.rankId == targetRankId) {
                BM_LOG_ERROR("device_urma Synchronize rank "
                             << targetRankId << " has pending in another context, stream: " << VaToStr(ctxSp->stream));
                return BM_ERROR;
            }
        }
    }
    return BM_OK;
}

} // namespace device
} // namespace transport
} // namespace mf
} // namespace ock
