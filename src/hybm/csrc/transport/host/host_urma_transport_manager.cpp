#include "host_urma_transport_manager.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <sstream>

#include "hybm_def.h"
#include "hybm_logger.h"
#include "hybm_va_manager.h"
#include "mf_env_util.h"
#include "urma/hcomm_transport_manager.h"
#include "urma/urma_transport_common.h"

namespace ock {
namespace mf {
namespace transport {
namespace host {

using urma::DeserializeExportDesc;
using urma::UrmaExportDesc;

namespace {

constexpr uint64_t HOST_TRANSFER_FLAG_SIZE = sizeof(uint64_t);
constexpr uint64_t URMA_CHANNEL_DESC_NUM = 1;
constexpr int32_t HCOMM_E_AGAIN = 20; // Aligned with HCCL_E_AGAIN without depending on HCCL headers.
constexpr uint32_t HCOMM_SUBMIT_MAX_RETRIES = 3U;
constexpr const char *ENV_HOST_URMA_EID = "MF_HOST_URMA_EID";

bool ContainsAddressRange(uint64_t outerAddr, uint64_t outerSize, uint64_t innerAddr, uint64_t innerSize)
{
    uint64_t outerEnd = 0;
    uint64_t innerEnd = 0;
    const UrmaCommMem outer{outerAddr, outerSize, UrmaMemoryType::HOST_DRAM};
    const UrmaCommMem inner{innerAddr, innerSize, UrmaMemoryType::HOST_DRAM};
    return GetRangeEnd(outer, outerEnd) && GetRangeEnd(inner, innerEnd) && outerAddr <= innerAddr &&
           outerEnd >= innerEnd;
}

struct ParsedRemoteMemKey {
    uint64_t remoteAddr{0};
    UrmaExportDesc exportDesc{};
    const uint8_t *payload{nullptr};
    uint32_t memDescLen{0};
};

static bool IsAllHex(const std::string &s)
{
    for (auto ch : s) {
        if (!std::isxdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

Result ParseRemoteMemKey(const TransportMemoryKey &memKey, uint32_t peerRank, ParsedRemoteMemKey &parsed)
{
    if (memKey.keys[0] != urma::URMA_EXPORT_DESC_MAGIC || memKey.keys[1] == 0) {
        BM_LOG_ERROR("Invalid remote memory key, peerRank: " << peerRank << " magic: 0x" << std::hex << memKey.keys[0]
                                                             << " keyAddr: 0x" << memKey.keys[1]);
        return BM_INVALID_PARAM;
    }
    parsed.remoteAddr = memKey.keys[1];
    parsed.payload = reinterpret_cast<const uint8_t *>(&memKey.keys[urma::DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    std::memcpy(&parsed.exportDesc, parsed.payload, sizeof(UrmaExportDesc));
    const auto &desc = parsed.exportDesc;
    const bool validType =
        desc.memoryType == UrmaMemoryType::HOST_DRAM || desc.memoryType == UrmaMemoryType::DEVICE_HBM;
    uint64_t exportEnd = 0;
    const UrmaCommMem exportMem{desc.addr, desc.size, desc.memoryType};
    const uint64_t totalDescLen = sizeof(UrmaExportDesc) + static_cast<uint64_t>(desc.hcommDescLen) +
                                  static_cast<uint64_t>(desc.devTransFlagDescLen);
    if (desc.magic != urma::URMA_EXPORT_DESC_MAGIC || desc.version != urma::URMA_EXPORT_DESC_VERSION ||
        desc.headerSize != sizeof(UrmaExportDesc) || !validType || desc.addr == 0 || desc.size == 0 ||
        desc.hcommDescLen == 0 || !urma::GetRangeEnd(exportMem, exportEnd) ||
        totalDescLen > urma::DEVICE_URMA_EXPORT_KEY_DATA_BYTES) {
        BM_LOG_ERROR("Invalid remote export descriptor, peerRank: "
                     << peerRank << " keyAddr: 0x" << std::hex << parsed.remoteAddr << " descAddr: 0x" << desc.addr
                     << std::dec << " size: " << desc.size << " memoryType: " << desc.memoryType
                     << " hcommDescLen: " << desc.hcommDescLen << " flagDescLen: " << desc.devTransFlagDescLen);
        return BM_INVALID_PARAM;
    }
    parsed.memDescLen = static_cast<uint32_t>(sizeof(UrmaExportDesc) + desc.hcommDescLen);
    return BM_OK;
}

Result ParseHostUrmaEid(std::array<uint8_t, COMM_ADDR_EID_LEN> &eid)
{
    const char *eidStr = std::getenv(ENV_HOST_URMA_EID);
    if (eidStr == nullptr || eidStr[0] == '\0') {
        BM_LOG_ERROR("Environment variable " << ENV_HOST_URMA_EID << " not set (32 hex chars)");
        return BM_INVALID_PARAM;
    }
    std::string val(eidStr);
    if (val.length() != COMM_ADDR_EID_LEN * 2 || !IsAllHex(val)) {
        BM_LOG_ERROR("Invalid " << ENV_HOST_URMA_EID << "='" << val << "' (expect " << (COMM_ADDR_EID_LEN * 2)
                                << " hex chars)");
        return BM_INVALID_PARAM;
    }
    for (size_t i = 0; i < COMM_ADDR_EID_LEN; i++) {
        auto byteStr = val.substr(i * 2, 2);
        eid[i] = static_cast<uint8_t>(std::strtoul(byteStr.c_str(), nullptr, 16) & 0xFF);
    }
    return BM_OK;
}

} // namespace

HostUrmaTransportManager::~HostUrmaTransportManager()
{
    (void)CloseDevice();
}

Result HostUrmaTransportManager::OpenDevice(const TransportOptions &options)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_) {
        BM_LOG_INFO("HostUrmaTransportManager already opened, rankId: " << options.rankId);
        return BM_OK;
    }
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    options_ = options;
    localNic_ = options.nic;
    auto ret = BuildLocalHostEndpointDescLocked(localEndpointDesc_);
    if (ret != BM_OK) {
        return ret;
    }
    localEndpoint_ = manager_.CreateEndpoint(localEndpointDesc_);
    if (localEndpoint_ == nullptr) {
        BM_LOG_ERROR("Failed to create Host URMA endpoint, rankId: " << rankId_);
        return BM_ERROR;
    }
    ret = InitHostTransferFlagLocked();
    if (ret != BM_OK) {
        return ret;
    }
    opened_ = true;
    BM_LOG_INFO("HostUrmaTransportManager opened, rankId: " << rankId_);
    return BM_OK;
}

Result HostUrmaTransportManager::BuildLocalHostEndpointDescLocked(UrmaEndpointDesc &endpoint) const
{
    std::array<uint8_t, COMM_ADDR_EID_LEN> eid{};
    auto ret = ParseHostUrmaEid(eid);
    if (ret != BM_OK) {
        return ret;
    }
    endpoint = {};
    endpoint.protocol = UrmaProtocol::UBC_CTP;
    endpoint.type = COMM_ADDR_TYPE_EID;
    std::memcpy(endpoint.raws, eid.data(), COMM_ADDR_EID_LEN);
    endpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpoint.loc.host.id = rankId_;
    return BM_OK;
}

Result HostUrmaTransportManager::InitHostTransferFlagLocked()
{
    hostTransFlagSize_ = HOST_TRANSFER_FLAG_SIZE;
    hostTransFlagPtr_ = std::malloc(hostTransFlagSize_);
    if (hostTransFlagPtr_ == nullptr) {
        BM_LOG_ERROR("Failed to allocate Host transfer flag, size: " << hostTransFlagSize_ << " rankId: " << rankId_);
        return BM_MALLOC_FAILED;
    }
    *static_cast<uint64_t *>(hostTransFlagPtr_) = 1;
    UrmaCommMem flagMem{};
    flagMem.addr = reinterpret_cast<uint64_t>(hostTransFlagPtr_);
    flagMem.size = hostTransFlagSize_;
    flagMem.type = UrmaMemoryType::HOST_DRAM;
    auto ret = manager_.HcommMemReg(localEndpoint_, reinterpret_cast<UrmaMemTag>(hostTransFlagPtr_), flagMem,
                                    &hostTransFlagHcommHandle_);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to register Host transfer flag, rankId: " << rankId_ << " ret: " << ret);
        std::free(hostTransFlagPtr_);
        hostTransFlagPtr_ = nullptr;
        return ret;
    }
    BM_LOG_DEBUG("Host transfer flag registered, rankId: " << rankId_ << " addr: " << std::hex << flagMem.addr);
    return BM_OK;
}

Result HostUrmaTransportManager::CloseDevice()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        return BM_OK;
    }
    Result finalRet = BM_OK;
    for (auto &[peerRank, state] : remoteRanks_) {
        auto ret = CleanupRemoteRankLocked(peerRank, state);
        if (ret != BM_OK && finalRet == BM_OK) {
            finalRet = ret;
        }
    }
    remoteRanks_.clear();

    for (auto &[addr, reg] : localRegistrations_) {
        if (reg.handle != nullptr) {
            (void)manager_.HcommMemUnreg(localEndpoint_, reg.handle);
        }
    }
    localRegistrations_.clear();

    if (hostTransFlagHcommHandle_ != nullptr) {
        (void)manager_.HcommMemUnreg(localEndpoint_, hostTransFlagHcommHandle_);
        hostTransFlagHcommHandle_ = nullptr;
    }
    if (hostTransFlagPtr_ != nullptr) {
        std::free(hostTransFlagPtr_);
        hostTransFlagPtr_ = nullptr;
    }
    hostTransFlagSize_ = 0;

    if (localEndpoint_ != nullptr) {
        (void)urma::HcomUrmaDestroyEndpoint(localEndpoint_->hcommEndpoint);
        localEndpoint_.reset();
    }
    opened_ = false;
    BM_LOG_INFO("HostUrmaTransportManager closed, rankId: " << rankId_);
    return finalRet;
}

Result HostUrmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion &mr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        BM_LOG_ERROR("HostUrmaTransportManager not opened, rankId: " << rankId_);
        return BM_NOT_INITIALIZED;
    }
    if (mr.addr == 0 || mr.size == 0) {
        BM_LOG_ERROR("Invalid memory region, addr: " << std::hex << mr.addr << " size: " << std::dec << mr.size);
        return BM_INVALID_PARAM;
    }
    if ((mr.flags & REG_MR_FLAG_DRAM) == 0U || (mr.flags & REG_MR_FLAG_HBM) != 0U) {
        BM_LOG_ERROR("Host URMA only supports DRAM memory, rankId: " << rankId_ << " addr: " << std::hex << mr.addr
                                                                     << " size: " << std::dec << mr.size << " flags: 0x"
                                                                     << std::hex << mr.flags);
        return BM_INVALID_PARAM;
    }
    auto it = localRegistrations_.find(mr.addr);
    if (it != localRegistrations_.end()) {
        if (it->second.mr.size == mr.size) {
            it->second.refCount++;
            BM_LOG_DEBUG("Memory region already registered, refCount: " << it->second.refCount << " addr: " << std::hex
                                                                        << mr.addr);
            return BM_OK;
        }
        BM_LOG_ERROR("Overlapping but different registration, addr: " << std::hex << mr.addr << " newSize: " << std::dec
                                                                      << mr.size
                                                                      << " existingSize: " << it->second.mr.size);
        return BM_INVALID_PARAM;
    }
    UrmaMemTag memTag = static_cast<UrmaMemTag>(mr.addr);
    UrmaCommMem commMem{};
    commMem.addr = mr.addr;
    commMem.size = mr.size;
    commMem.type = UrmaMemoryType::HOST_DRAM;
    HcommMemHandle handle = nullptr;
    auto ret = manager_.HcommMemReg(localEndpoint_, memTag, commMem, &handle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to register memory region, addr: " << std::hex << mr.addr << " size: " << std::dec
                                                                << mr.size << " ret: " << ret);
        return ret;
    }
    uint64_t exportedGva = 0;
    auto gvaRet = ResolveExportedGvaLocked(mr, exportedGva);
    if (gvaRet != BM_OK) {
        BM_LOG_WARN("GVA resolution failed for mr addr: " << std::hex << mr.addr
                                                          << " this mr is local-only, exportedGva=0");
    }
    LocalRegistration reg{};
    reg.mr = mr;
    reg.handle = handle;
    reg.memTag = memTag;
    reg.exportedGva = exportedGva;
    reg.refCount = 1;
    localRegistrations_.emplace(mr.addr, reg);
    BM_LOG_DEBUG("Memory region registered, addr: " << std::hex << mr.addr << " size: " << std::dec << mr.size
                                                    << " exportedGva: " << std::hex << exportedGva);
    return BM_OK;
}

Result HostUrmaTransportManager::ResolveExportedGvaLocked(const TransportMemoryRegion &mr, uint64_t &exportedGva) const
{
    exportedGva = 0;
    auto result = HybmVaManager::GetInstance().FindAllocByVa(mr.addr, HVM_GVA);
    if (!result.second) {
        return BM_OK;
    }
    const auto &vaInfo = result.first;
    if (vaInfo.base.size < mr.size || vaInfo.base.va[HVM_GVA] > mr.addr ||
        vaInfo.base.va[HVM_GVA] + vaInfo.base.size < mr.addr + mr.size) {
        return BM_OK;
    }
    if (vaInfo.base.memType != HYBM_MEM_TYPE_HOST) {
        return BM_OK;
    }
    exportedGva = mr.addr;
    return BM_OK;
}

Result HostUrmaTransportManager::FindLocalRegistrationLocked(uint64_t addr, uint64_t size,
                                                             LocalRegistration *registration) const
{
    if (addr == 0 || size == 0 || !ContainsAddressRange(addr, size, addr, size)) {
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

Result HostUrmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = localRegistrations_.find(addr);
    if (it == localRegistrations_.end()) {
        BM_LOG_ERROR("Memory region not found, addr: " << std::hex << addr);
        return BM_INVALID_PARAM;
    }
    it->second.refCount--;
    if (it->second.refCount > 0) {
        return BM_OK;
    }
    auto ret = manager_.HcommMemUnreg(localEndpoint_, it->second.handle);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to unregister memory region, addr: " << std::hex << addr << " ret: " << ret);
        return ret;
    }
    localRegistrations_.erase(it);
    return BM_OK;
}

bool HostUrmaTransportManager::QueryHasRegistered(uint64_t addr, uint64_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return FindLocalRegistrationLocked(addr, size, nullptr) == BM_OK;
}

Result HostUrmaTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey &key)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        BM_LOG_ERROR("HostUrmaTransportManager not opened");
        return BM_NOT_INITIALIZED;
    }
    auto it = localRegistrations_.find(addr);
    if (it == localRegistrations_.end()) {
        BM_LOG_ERROR("Address not registered, addr: " << std::hex << addr);
        return BM_INVALID_PARAM;
    }
    if (it->second.exportedGva == 0) {
        BM_LOG_ERROR("Address has no exported GVA, addr: " << std::hex << addr);
        return BM_INVALID_PARAM;
    }
    if (hostTransFlagHcommHandle_ == nullptr) {
        BM_LOG_ERROR("Host transfer flag not registered");
        return BM_ERROR;
    }
    // Export MR descriptor via HcommTransportManager (returns UrmaExportDesc + hcommDesc)
    const uint8_t *memDesc = nullptr;
    uint32_t memDescLen = 0;
    auto ret = manager_.HcommMemExport(localEndpoint_, it->second.handle, &memDesc, &memDescLen);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to export memory region, addr: " << std::hex << addr << " ret: " << ret);
        return ret;
    }
    // Parse export to recover hcommDesc pointer
    UrmaExportDesc exportDesc{};
    const uint8_t *hcommDesc = nullptr;
    uint32_t hcommDescLen = 0;
    if (!DeserializeExportDesc(memDesc, memDescLen, exportDesc, &hcommDesc, &hcommDescLen)) {
        BM_LOG_ERROR("DeserializeExportDesc failed for addr: " << std::hex << addr);
        return BM_ERROR;
    }
    // Export flag descriptor via raw HCOMM API
    void *flagDescRaw = nullptr;
    uint32_t flagDescLen = 0;
    ret = DlHcommApi::HcommMemExport(localEndpoint_->hcommEndpoint, hostTransFlagHcommHandle_, &flagDescRaw,
                                     &flagDescLen);
    if (ret != 0 || flagDescRaw == nullptr || flagDescLen == 0) {
        BM_LOG_ERROR("Failed to export Host transfer flag, ret: " << ret);
        return BM_ERROR;
    }
    const uint8_t *flagDesc = static_cast<const uint8_t *>(flagDescRaw);
    // Build key layout: [keys[0]=magic, keys[1]=gva, keys[2..]=UrmaExportDesc+hcommDesc+flagDesc]
    key.keys[0] = urma::URMA_EXPORT_DESC_MAGIC;
    key.keys[1] = it->second.exportedGva;
    UrmaExportDesc keyExportDesc = exportDesc;
    keyExportDesc.devTransFlagDescLen = flagDescLen;
    uint8_t *payload = reinterpret_cast<uint8_t *>(&key.keys[urma::DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS]);
    constexpr uint32_t exportHdrSize = sizeof(UrmaExportDesc);
    constexpr uint32_t maxPayloadBytes = (KEY_SIZE * 6 - urma::DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS) * sizeof(uint64_t);
    uint32_t totalBytes = exportHdrSize + hcommDescLen + flagDescLen;
    if (totalBytes > maxPayloadBytes) {
        BM_LOG_ERROR("Export descriptors too large for key slots, total: " << totalBytes);
        return BM_ERROR;
    }
    std::memcpy(payload, &keyExportDesc, exportHdrSize);
    std::memcpy(payload + exportHdrSize, hcommDesc, hcommDescLen);
    std::memcpy(payload + exportHdrSize + hcommDescLen, flagDesc, flagDescLen);
    return BM_OK;
}

void HostUrmaTransportManager::UpdateMemoryKey(TransportMemoryKey &key, void *addr)
{
    key.keys[1] = reinterpret_cast<uint64_t>(addr);
}

Result HostUrmaTransportManager::Prepare(const HybmTransPrepareOptions &options)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        BM_LOG_ERROR("HostUrmaTransportManager not opened");
        return BM_NOT_INITIALIZED;
    }
    for (const auto &[peerRank, peerInfo] : options.options) {
        auto &state = remoteRanks_[peerRank];
        bool isNew = (state.channel == 0);
        if (isNew) {
            UrmaEndpointDesc peerEndpoint{};
            auto ret = urma::ParseUrmaPrivateData(peerInfo.privateData, peerEndpoint);
            if (ret != BM_OK) {
                BM_LOG_ERROR("Failed to parse private data for peer " << peerRank);
                remoteRanks_.erase(peerRank);
                return ret;
            }
            state.endpointDesc = peerEndpoint;
            ret = PreparePeerLocked(peerRank, peerInfo, state);
            if (ret != BM_OK) {
                remoteRanks_.erase(peerRank);
                return ret;
            }
        } else {
            auto ret = ValidateInitialPeerSetLocked(options, state);
            if (ret != BM_OK) {
                return ret;
            }
        }
        if (!peerInfo.memKeys.empty()) {
            auto ret = PreparePeerMemoryKeysLocked(peerRank, peerInfo.memKeys, state);
            if (ret != BM_OK) {
                return ret;
            }
        }
    }
    return BM_OK;
}

Result HostUrmaTransportManager::PreparePeerLocked(uint32_t peerRank, const TransportRankPrepareInfo &peerInfo,
                                                   RemoteRankState &state)
{
    EndpointDesc hcommEndpoint = urma::ToHcommEndpointDesc(state.endpointDesc);
    HcommChannelDesc channelDesc{};
    auto ret = HcommChannelDescInit(&channelDesc, URMA_CHANNEL_DESC_NUM);
    if (ret != 0) {
        BM_LOG_ERROR("Failed to init channel desc for peer " << peerRank << " ret: " << ret);
        return BM_ERROR;
    }
    channelDesc.remoteEndpoint = hcommEndpoint;
    channelDesc.exchangeAllMems = true;
    channelDesc.port = 0;
    channelDesc.role = (rankId_ > peerRank) ? HCOMM_SOCKET_ROLE_CLIENT : HCOMM_SOCKET_ROLE_SERVER;
    ChannelHandle channel = 0;
    ret = DlHcommApi::HcommChannelCreate(localEndpoint_->hcommEndpoint, COMM_ENGINE_CPU, &channelDesc, 1, &channel);
    if (ret != 0) {
        BM_LOG_ERROR("Failed to create channel with peer " << peerRank << " ret: " << ret);
        return BM_ERROR;
    }
    ret = manager_.WaitForChannelReady(channel, peerRank);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Host URMA channel is not ready, rankId: " << rankId_ << " peerRank: " << peerRank
                                                                << " channel: " << channel << " ret: " << ret);
        const auto destroyRet = DlHcommApi::HcommChannelDestroy(&channel, URMA_CHANNEL_DESC_NUM);
        if (destroyRet != 0) {
            BM_LOG_ERROR("Host URMA rollback HcommChannelDestroy failed, rankId: "
                         << rankId_ << " peerRank: " << peerRank << " channel: " << channel << " ret: " << destroyRet);
        }
        return ret;
    }
    state.channelDesc = channelDesc;
    state.channel = channel;
    BM_LOG_INFO("Host URMA channel ready with peer "
                << peerRank << " channel: " << channel
                << " role: " << (channelDesc.role == HCOMM_SOCKET_ROLE_CLIENT ? "CLIENT" : "SERVER"));
    return BM_OK;
}

Result HostUrmaTransportManager::ValidateInitialPeerSetLocked(const HybmTransPrepareOptions &options,
                                                              RemoteRankState &state)
{
    (void)options;
    if (state.channel == 0) {
        BM_LOG_ERROR("Peer rank state has no channel but was previously prepared");
        return BM_ERROR;
    }
    return BM_OK;
}

Result HostUrmaTransportManager::PreparePeerMemoryKeysLocked(uint32_t peerRank,
                                                             const std::vector<TransportMemoryKey> &memKeys,
                                                             RemoteRankState &state)
{
    if (state.endpointDesc.loc.locType != ENDPOINT_LOC_TYPE_DEVICE &&
        state.endpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        BM_LOG_ERROR("Unsupported peer endpoint location, rankId: " << rankId_ << " peerRank: " << peerRank
                                                                    << " locType: " << state.endpointDesc.loc.locType);
        return BM_INVALID_PARAM;
    }
    return ImportRemoteMemKeysLocked(peerRank, memKeys, state);
}

Result HostUrmaTransportManager::ImportRemoteMemKeysLocked(uint32_t peerRank,
                                                           const std::vector<TransportMemoryKey> &memKeys,
                                                           RemoteRankState &state)
{
    bool flagImported = (state.remoteFlagAddr != 0);
    for (const auto &memKey : memKeys) {
        ParsedRemoteMemKey parsed{};
        auto ret = ParseRemoteMemKey(memKey, peerRank, parsed);
        if (ret != BM_OK) {
            return ret;
        }
        const auto &exportDesc = parsed.exportDesc;
        const auto *payload = parsed.payload;
        const uint32_t hcommDescLen = exportDesc.hcommDescLen;
        const uint32_t flagDescLen = exportDesc.devTransFlagDescLen;
        UrmaCommMem imported{};
        ret = manager_.HcommMemImport(localEndpoint_, payload, parsed.memDescLen, &imported);
        if (ret != BM_OK) {
            BM_LOG_ERROR("Failed to HcommMemImport for peer " << peerRank << " ret: " << ret << " addr: " << std::hex
                                                              << exportDesc.addr);
            return ret;
        }
        RemoteRegistration reg{};
        reg.exportedAddr = parsed.remoteAddr;
        reg.size = exportDesc.size;
        reg.memTag = exportDesc.memTag;
        reg.view = imported;
        reg.descBytes.resize(parsed.memDescLen);
        std::memcpy(reg.descBytes.data(), payload, parsed.memDescLen);
        auto valRet = ValidateImportedGva(peerRank, reg.exportedAddr, reg.size, exportDesc, reg.view);
        if (valRet != BM_OK) {
            (void)manager_.HcommMemUnimport(localEndpoint_, reg.descBytes.data(), reg.descBytes.size());
            return valRet;
        }
        state.imports.push_back(reg);
        if (exportDesc.memoryType == UrmaMemoryType::HOST_DRAM && flagDescLen > 0 && !flagImported) {
            HcommCommMem hcommFlagOut{};
            auto flagRet =
                DlHcommApi::HcommMemImport(localEndpoint_->hcommEndpoint,
                                           payload + sizeof(UrmaExportDesc) + hcommDescLen, flagDescLen, &hcommFlagOut);
            if (flagRet != 0) {
                BM_LOG_ERROR("Failed to HcommMemImport flag for peer " << peerRank << " ret: " << flagRet);
                (void)manager_.HcommMemUnimport(localEndpoint_, reg.descBytes.data(), reg.descBytes.size());
                state.imports.pop_back();
                return BM_ERROR;
            }
            state.remoteFlagAddr = reinterpret_cast<uint64_t>(hcommFlagOut.addr);
            state.remoteFlagSize = hcommFlagOut.size;
            state.remoteFlagDescBytes.resize(flagDescLen);
            std::memcpy(state.remoteFlagDescBytes.data(), payload + sizeof(UrmaExportDesc) + hcommDescLen, flagDescLen);
            flagImported = true;
            BM_LOG_INFO("Imported remote flag for peer " << peerRank << " flagAddr: " << std::hex
                                                         << state.remoteFlagAddr);
        }
    }
    return BM_OK;
}

Result HostUrmaTransportManager::ValidateImportedGva(uint32_t peerRank, uint64_t exportedAddr, uint64_t exportedSize,
                                                     const UrmaExportDesc &exportDesc, const UrmaCommMem &view) const
{
    if (view.addr == 0 || view.size == 0) {
        BM_LOG_ERROR("Import view is empty for peer " << peerRank);
        return BM_NOT_SUPPORTED;
    }
    if (exportDesc.size != exportedSize) {
        BM_LOG_ERROR("Export descriptor size mismatch for peer " << peerRank << " exportDesc.size: " << exportDesc.size
                                                                 << " exportedSize: " << exportedSize);
        return BM_NOT_SUPPORTED;
    }
    if (view.size < exportedSize) {
        BM_LOG_ERROR("Import view size too small for peer " << peerRank << " view.size: " << view.size
                                                            << " exportedSize: " << exportedSize);
        return BM_NOT_SUPPORTED;
    }
    if (exportDesc.memoryType == UrmaMemoryType::HOST_DRAM &&
        (exportDesc.addr != exportedAddr || view.addr != exportedAddr || view.type != UrmaMemoryType::HOST_DRAM)) {
        BM_LOG_ERROR("Invalid Host DRAM import for peer "
                     << peerRank << " exportedAddr: " << std::hex << exportedAddr << " descAddr: " << exportDesc.addr
                     << " viewAddr: " << view.addr << std::dec << " viewType: " << view.type);
        return BM_NOT_SUPPORTED;
    }
    if (exportDesc.memoryType == UrmaMemoryType::DEVICE_HBM && view.type != UrmaMemoryType::DEVICE_HBM) {
        BM_LOG_ERROR("Invalid Device HBM import for peer " << peerRank << " gva: " << std::hex << exportedAddr
                                                           << " viewAddr: " << view.addr << std::dec
                                                           << " viewType: " << view.type);
        return BM_NOT_SUPPORTED;
    }
    if (exportDesc.memoryType != UrmaMemoryType::HOST_DRAM && exportDesc.memoryType != UrmaMemoryType::DEVICE_HBM) {
        BM_LOG_ERROR("Unexpected memory type for peer " << peerRank << " type: " << exportDesc.memoryType);
        return BM_NOT_SUPPORTED;
    }
    return BM_OK;
}

Result HostUrmaTransportManager::ResolveRemoteAddressLocked(const RemoteRankState &state, uint64_t remoteAddr,
                                                            uint64_t size, uint64_t &hcommAddr) const
{
    for (const auto &import : state.imports) {
        if (remoteAddr >= import.exportedAddr && remoteAddr + size <= import.exportedAddr + import.size) {
            hcommAddr = import.view.addr + (remoteAddr - import.exportedAddr);
            BM_LOG_INFO("Resolved addr 0x" << std::hex << remoteAddr << " to view 0x" << hcommAddr << " exported 0x"
                                           << import.exportedAddr << " size 0x" << import.size);
            return BM_OK;
        }
    }
    BM_LOG_ERROR("Remote address not found in any import, addr: 0x" << std::hex << remoteAddr << " size: 0x" << std::dec
                                                                    << size << " imports: " << state.imports.size());
    for (const auto &import : state.imports) {
        BM_LOG_ERROR("  import exportedAddr=0x" << std::hex << import.exportedAddr << " size=0x" << import.size
                                                << " view.addr=0x" << import.view.addr);
    }
    return BM_NOT_CONNECTED;
}

Result HostUrmaTransportManager::DestroyRemoteChannelLocked(uint32_t peerRank, RemoteRankState &state)
{
    Result finalRet = BM_OK;
    if (state.channel != 0) {
        const auto channel = state.channel;
        const auto ret = DlHcommApi::HcommChannelDestroy(&state.channel, 1);
        state.channel = 0;
        if (ret != 0) {
            BM_LOG_ERROR("HcommChannelDestroy failed, rankId: " << rankId_ << " peerRank: " << peerRank
                                                                << " channel: " << channel << " ret: " << ret);
            finalRet = BM_ERROR;
        }
    }
    return finalRet;
}

Result HostUrmaTransportManager::UnimportRemoteResourcesLocked(uint32_t peerRank, RemoteRankState &state)
{
    Result finalRet = BM_OK;
    for (auto &import : state.imports) {
        auto ret = manager_.HcommMemUnimport(localEndpoint_, import.descBytes.data(), import.descBytes.size());
        if (ret != BM_OK) {
            BM_LOG_ERROR("Failed to unimport remote MR, rankId: "
                         << rankId_ << " peerRank: " << peerRank << " exportedAddr: " << std::hex << import.exportedAddr
                         << std::dec << " ret: " << ret);
            if (finalRet == BM_OK) {
                finalRet = ret;
            }
        }
    }
    state.imports.clear();
    if (!state.remoteFlagDescBytes.empty()) {
        auto ret = DlHcommApi::HcommMemUnimport(localEndpoint_->hcommEndpoint, state.remoteFlagDescBytes.data(),
                                                state.remoteFlagDescBytes.size());
        if (ret != 0) {
            BM_LOG_ERROR("Failed to unimport remote flag, rankId: " << rankId_ << " peerRank: " << peerRank
                                                                    << " flagAddr: " << std::hex << state.remoteFlagAddr
                                                                    << std::dec << " ret: " << ret);
            if (finalRet == BM_OK) {
                finalRet = BM_ERROR;
            }
        }
    }
    state.remoteFlagAddr = 0;
    state.remoteFlagSize = 0;
    state.remoteFlagDescBytes.clear();
    return finalRet;
}

Result HostUrmaTransportManager::CleanupRemoteRankLocked(uint32_t peerRank, RemoteRankState &state)
{
    Result finalRet = FenceRank(state, peerRank);
    auto ret = DestroyRemoteChannelLocked(peerRank, state);
    if (ret != BM_OK && finalRet == BM_OK) {
        finalRet = ret;
    }
    ret = UnimportRemoteResourcesLocked(peerRank, state);
    if (ret != BM_OK && finalRet == BM_OK) {
        finalRet = ret;
    }
    state.pending = false;
    return finalRet;
}

Result HostUrmaTransportManager::RemoveRankLocked(uint32_t peerRank)
{
    auto it = remoteRanks_.find(peerRank);
    if (it == remoteRanks_.end()) {
        return BM_OK;
    }
    auto ret = CleanupRemoteRankLocked(peerRank, it->second);
    remoteRanks_.erase(it);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to clean remote rank, rankId: " << rankId_ << " peerRank: " << peerRank
                                                             << " ret: " << ret);
    }
    return ret;
}

Result HostUrmaTransportManager::RemoveRanks(const std::vector<uint32_t> &removedRanks)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_) {
        BM_LOG_ERROR("HostUrmaTransportManager not opened, rankId: " << rankId_);
        return BM_NOT_INITIALIZED;
    }
    Result finalRet = BM_OK;
    for (auto peerRank : removedRanks) {
        auto ret = RemoveRankLocked(peerRank);
        if (ret != BM_OK && finalRet == BM_OK) {
            finalRet = ret;
        }
    }
    return finalRet;
}

Result HostUrmaTransportManager::Connect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    connected_ = true;
    return BM_OK;
}

Result HostUrmaTransportManager::AsyncConnect()
{
    return Connect();
}

Result HostUrmaTransportManager::WaitForConnected(int64_t timeoutNs)
{
    if (timeoutNs < 0) {
        return BM_INVALID_PARAM;
    }
    if (connected_) {
        return BM_OK;
    }
    return BM_NOT_CONNECTED;
}

Result HostUrmaTransportManager::UpdateRankOptions(const HybmTransPrepareOptions &options)
{
    bool needFallback = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[peerRank, peerInfo] : options.options) {
            auto it = remoteRanks_.find(peerRank);
            if (it == remoteRanks_.end()) {
                BM_LOG_WARN("UpdateRankOptions: peer " << peerRank << " not prepared yet, fallback to Prepare");
                needFallback = true;
                break;
            }
            if (peerInfo.privateData.ip[0] != '\0') {
                UrmaEndpointDesc newDesc{};
                auto ret = urma::ParseUrmaPrivateData(peerInfo.privateData, newDesc);
                if (ret != BM_OK) {
                    return ret;
                }
                if (std::memcmp(&it->second.endpointDesc, &newDesc, sizeof(UrmaEndpointDesc)) != 0) {
                    BM_LOG_ERROR("UpdateRankOptions: endpoint changed for peer " << peerRank);
                    return BM_NOT_SUPPORTED;
                }
            }
            if (!peerInfo.memKeys.empty()) {
                needFallback = true;
            }
        }
    }
    return needFallback ? Prepare(options) : BM_OK;
}

const std::string &HostUrmaTransportManager::GetNic() const
{
    return localNic_;
}

const TransportPrivateData HostUrmaTransportManager::GetPrivateData() const
{
    TransportPrivateData data{};
    if (!opened_) {
        BM_LOG_ERROR("HostUrmaTransportManager not opened, returning empty private data");
        return data;
    }
    auto ret = urma::SerializeUrmaPrivateData(localEndpointDesc_, data);
    if (ret != BM_OK) {
        BM_LOG_ERROR("Failed to serialize private data, ret: " << ret);
    }
    return data;
}

Result HostUrmaTransportManager::ReadRemote(uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr, uint64_t size)
{
    BM_LOG_INFO("host_urma ReadRemote rankId=" << rankId << " local=0x" << std::hex << localAddr << " remote=0x"
                                               << remoteAddr << std::dec << " size=" << size);
    return RemoteIo(rankId, localAddr, remoteAddr, size, false, true);
}

Result HostUrmaTransportManager::WriteRemote(uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr, uint64_t size)
{
    return RemoteIo(rankId, localAddr, remoteAddr, size, true, true);
}

Result HostUrmaTransportManager::ReadRemoteAsync(uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr,
                                                 uint64_t size)
{
    return RemoteIo(rankId, localAddr, remoteAddr, size, false, false);
}

Result HostUrmaTransportManager::WriteRemoteAsync(uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr,
                                                  uint64_t size)
{
    return RemoteIo(rankId, localAddr, remoteAddr, size, true, false);
}

Result HostUrmaTransportManager::RemoteIo(uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr, uint64_t size,
                                          bool write, bool synchronize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = remoteRanks_.find(rankId);
    if (it == remoteRanks_.end()) {
        BM_LOG_ERROR("RemoteIo: rank " << rankId << " not connected");
        return BM_NOT_CONNECTED;
    }
    auto &state = it->second;
    uint64_t hcommAddr = 0;
    auto ret = ResolveRemoteAddressLocked(state, remoteAddr, size, hcommAddr);
    if (ret != BM_OK) {
        return ret;
    }
    ret = SubmitRemoteIo(state, rankId, localAddr, remoteAddr, hcommAddr, size, write);
    if (ret != BM_OK) {
        return ret;
    }
    return synchronize ? FenceRank(state, rankId) : BM_OK;
}

Result HostUrmaTransportManager::SubmitRemoteIo(RemoteRankState &state, uint32_t rankId, uint64_t localAddr,
                                                uint64_t remoteAddr, uint64_t hcommAddr, uint64_t size, bool write)
{
    for (uint32_t retry = 0; retry <= HCOMM_SUBMIT_MAX_RETRIES; ++retry) {
        int32_t hcomRet = write ? DlHcommApi::HcommWriteNbi(state.channel, reinterpret_cast<void *>(hcommAddr),
                                                            reinterpret_cast<const void *>(localAddr), size)
                                : DlHcommApi::HcommReadNbi(state.channel, reinterpret_cast<void *>(localAddr),
                                                          reinterpret_cast<const void *>(hcommAddr), size);
        if (hcomRet == 0) {
            state.pending = true;
            return BM_OK;
        }
        if (hcomRet != HCOMM_E_AGAIN || retry == HCOMM_SUBMIT_MAX_RETRIES) {
            BM_LOG_ERROR("RemoteIo submit failed, rankId: " << rankId << " write: " << write << " localAddr: "
                                                            << std::hex << localAddr << " remoteAddr: " << remoteAddr
                                                            << " size: " << std::dec << size << " hcomRet: " << hcomRet
                                                            << " retries: " << retry);
            return BM_ERROR;
        }
        state.pending = true;
        auto ret = FenceRank(state, rankId);
        if (ret != BM_OK) {
            return ret;
        }
    }
    BM_LOG_ERROR("RemoteIo submit retry loop exited unexpectedly, rankId: " << rankId << " write: " << write);
    return BM_ERROR;
}

Result HostUrmaTransportManager::ReadRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor)
{
    return RemoteIoBatch(rankId, descriptor, false);
}

Result HostUrmaTransportManager::WriteRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor)
{
    return RemoteIoBatch(rankId, descriptor, true);
}

Result HostUrmaTransportManager::RemoteIoBatch(uint32_t rankId, const CopyDescriptor &descriptor, bool write)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (remoteRanks_.find(rankId) == remoteRanks_.end()) {
            BM_LOG_ERROR("RemoteIoBatch: rank " << rankId << " not connected");
            return BM_NOT_CONNECTED;
        }
    }
    if (descriptor.localAddrs.size() != descriptor.globalAddrs.size() ||
        descriptor.localAddrs.size() != descriptor.counts.size()) {
        BM_LOG_ERROR("RemoteIoBatch: vector size mismatch for rank " << rankId);
        return BM_INVALID_PARAM;
    }
    if (descriptor.localAddrs.empty()) {
        return BM_OK;
    }
    for (size_t i = 0; i < descriptor.localAddrs.size(); i++) {
        if (descriptor.counts[i] == 0) {
            continue;
        }
        auto ret =
            RemoteIo(rankId, reinterpret_cast<uint64_t>(write ? descriptor.localAddrs[i] : descriptor.globalAddrs[i]),
                     reinterpret_cast<uint64_t>(write ? descriptor.globalAddrs[i] : descriptor.localAddrs[i]),
                     descriptor.counts[i], write, false);
        if (ret != BM_OK) {
            BM_LOG_ERROR("RemoteIoBatch: item " << i << " failed for rank " << rankId);
            return ret;
        }
    }
    return BM_OK;
}

Result HostUrmaTransportManager::Synchronize(uint32_t rankId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = remoteRanks_.find(rankId);
    if (it == remoteRanks_.end()) {
        BM_LOG_ERROR("Synchronize: rank " << rankId << " not found");
        return BM_NOT_CONNECTED;
    }
    return FenceRank(it->second, rankId);
}

Result HostUrmaTransportManager::FenceRank(RemoteRankState &state, uint32_t rankId)
{
    if (!state.pending) {
        return BM_OK;
    }
    auto ret = DlHcommApi::HcommChannelFence(state.channel);
    if (ret != 0) {
        BM_LOG_ERROR("HcommChannelFence failed, rankId: " << rankId << " channel: " << state.channel
                                                           << " ret: " << ret);
        return BM_ERROR;
    }
    state.pending = false;
    return BM_OK;
}

} // namespace host
} // namespace transport
} // namespace mf
} // namespace ock
