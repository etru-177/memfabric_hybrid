#ifndef MF_HYBRID_HOST_URMA_TRANSPORT_MANAGER_H
#define MF_HYBRID_HOST_URMA_TRANSPORT_MANAGER_H

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "dl_hcomm_api.h"
#include "hybm_transport_manager.h"
#include "urma/hcomm_transport_manager.h"

namespace ock {
namespace mf {
namespace transport {
namespace host {

using urma::HcommTransportManager;
using urma::UrmaCommMem;
using urma::UrmaEndpointDesc;
using urma::UrmaEndpointHandle;
using urma::UrmaMemTag;
using urma::UrmaMemoryType;
using urma::UrmaProtocol;

class HostUrmaTransportManager final : public transport::TransportManager {
public:
    HostUrmaTransportManager() = default;
    ~HostUrmaTransportManager() override;

    Result OpenDevice(const TransportOptions &options) override;
    Result CloseDevice() override;

    Result RegisterMemoryRegion(const TransportMemoryRegion &mr) override;
    Result UnregisterMemoryRegion(uint64_t addr) override;
    bool QueryHasRegistered(uint64_t addr, uint64_t size) override;
    Result QueryMemoryKey(uint64_t addr, TransportMemoryKey &key) override;
    void UpdateMemoryKey(TransportMemoryKey &key, void *addr) override;

    Result Prepare(const HybmTransPrepareOptions &options) override;
    Result RemoveRanks(const std::vector<uint32_t> &removedRanks) override;
    Result Connect() override;
    Result AsyncConnect() override;
    Result WaitForConnected(int64_t timeoutNs) override;
    Result UpdateRankOptions(const HybmTransPrepareOptions &options) override;

    const std::string &GetNic() const override;
    const TransportPrivateData GetPrivateData() const override;

    Result ReadRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;
    Result WriteRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;
    Result ReadRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;
    Result WriteRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;
    Result WriteRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor) override;
    Result ReadRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor) override;
    Result Synchronize(uint32_t rankId) override;

private:
    struct LocalRegistration {
        TransportMemoryRegion mr{};
        HcommMemHandle handle{nullptr};
        UrmaMemTag memTag{0};
        uint64_t exportedGva{0};
        uint32_t refCount{0};
    };

    struct RemoteRegistration {
        uint64_t exportedAddr{0};
        uint64_t size{0};
        UrmaMemTag memTag{0};
        std::vector<uint8_t> descBytes{};
        UrmaCommMem view{};
    };

    struct RemoteRankState {
        UrmaEndpointDesc endpointDesc{};
        HcommChannelDesc channelDesc{};
        ChannelHandle channel{0};
        std::vector<RemoteRegistration> imports{};
        uint64_t remoteFlagAddr{0};
        uint64_t remoteFlagSize{0};
        std::vector<uint8_t> remoteFlagDescBytes{};
        bool pending{false};
    };

    Result BuildLocalHostEndpointDescLocked(UrmaEndpointDesc &endpoint) const;
    Result InitHostTransferFlagLocked();

    Result FindLocalRegistrationLocked(uint64_t addr, uint64_t size, LocalRegistration *registration) const;
    Result ResolveExportedGvaLocked(const TransportMemoryRegion &mr, uint64_t &exportedGva) const;

    Result PreparePeerLocked(uint32_t peerRank, const TransportRankPrepareInfo &peerInfo, RemoteRankState &state);
    Result ValidateInitialPeerSetLocked(const HybmTransPrepareOptions &options, RemoteRankState &state);
    Result PreparePeerMemoryKeysLocked(uint32_t peerRank, const std::vector<TransportMemoryKey> &memKeys,
                                       RemoteRankState &state);
    Result ImportRemoteMemKeysLocked(uint32_t peerRank, const std::vector<TransportMemoryKey> &memKeys,
                                     RemoteRankState &state);
    Result ValidateImportedGva(uint32_t peerRank, uint64_t exportedAddr, uint64_t exportedSize,
                               const urma::UrmaExportDesc &exportDesc, const UrmaCommMem &view) const;
    Result ResolveRemoteAddressLocked(const RemoteRankState &state, uint64_t remoteAddr, uint64_t size,
                                      uint64_t &hcommAddr) const;

    Result RemoteIo(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size, bool write, bool synchronize);
    Result SubmitRemoteIo(RemoteRankState &state, uint32_t rankId, uint64_t localAddr, uint64_t remoteAddr,
                          uint64_t hcommAddr, uint64_t size, bool write);
    Result RemoteIoBatch(uint32_t rankId, const CopyDescriptor &descriptor, bool write);
    Result FenceRank(RemoteRankState &state, uint32_t rankId);
    Result DestroyRemoteChannelLocked(uint32_t peerRank, RemoteRankState &state);
    Result UnimportRemoteResourcesLocked(uint32_t peerRank, RemoteRankState &state);
    Result CleanupRemoteRankLocked(uint32_t peerRank, RemoteRankState &state);
    Result RemoveRankLocked(uint32_t peerRank);

    mutable std::mutex mutex_{};
    bool opened_{false};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    TransportOptions options_{};
    HcommTransportManager manager_;
    UrmaEndpointHandle localEndpoint_{nullptr};
    UrmaEndpointDesc localEndpointDesc_{};
    std::string localNic_{};
    std::array<uint8_t, COMM_ADDR_EID_LEN> localEid_{};

    void *hostTransFlagPtr_{nullptr};
    uint64_t hostTransFlagSize_{0};
    HcommMemHandle hostTransFlagHcommHandle_{nullptr};

    std::map<uint64_t, LocalRegistration> localRegistrations_{};
    std::unordered_map<uint32_t, RemoteRankState> remoteRanks_{};
};

} // namespace host
} // namespace transport
} // namespace mf
} // namespace ock

#endif // MF_HYBRID_HOST_URMA_TRANSPORT_MANAGER_H
