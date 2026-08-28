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

#ifndef MF_HYBRID_DEVICE_URMA_TRANSPORT_MANAGER_H
#define MF_HYBRID_DEVICE_URMA_TRANSPORT_MANAGER_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "dl_hcomm_api.h"
#include "dl_rt_api.h"
#include "hcomm_transport_manager.h"
#include "hybm_transport_manager.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {

class DeviceUrmaTransportManager final : public transport::TransportManager {
public:
    DeviceUrmaTransportManager() = default;

    ~DeviceUrmaTransportManager() override;

    // Initialization/open/close lifecycle
    Result OpenDevice(const TransportOptions &options) override;

    Result CloseDevice() override;

    // Local/remote registration and key export/import
    Result RegisterMemoryRegion(const TransportMemoryRegion &mr) override;

    Result UnregisterMemoryRegion(uint64_t addr) override;

    bool QueryHasRegistered(uint64_t addr, uint64_t size) override;

    Result QueryMemoryKey(uint64_t addr, TransportMemoryKey &key) override;

    void UpdateMemoryKey(TransportMemoryKey &key, void *addr) override;

    // Prepare/rank/connection/private data
    Result Prepare(const HybmTransPrepareOptions &options) override;

    Result RemoveRanks(const std::vector<uint32_t> &removedRanks) override;

    Result Connect() override;

    Result AsyncConnect() override;

    Result WaitForConnected(int64_t timeoutNs) override;

    Result UpdateRankOptions(const HybmTransPrepareOptions &options) override;

    const std::string &GetNic() const override;

    const TransportPrivateData GetPrivateData() const override;

    // Data transfer/copy
    Result ReadRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;

    Result WriteRemote(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;

    Result ReadRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;

    Result WriteRemoteAsync(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size) override;

    Result WriteRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor) override;

    Result ReadRemoteBatchAsync(uint32_t rankId, const CopyDescriptor &descriptor) override;

    // Sync stream
    Result Synchronize(uint32_t rankId) override;

private:
    struct LocalRegistration {
        TransportMemoryRegion mr{};
        HcommMemHandle handle{nullptr};
        UrmaMemTag memTag{0};
        uint32_t refCount{0};
        uint64_t deviceVa{0};
    };

    struct RemoteRegistration {
        uint64_t addr{0};
        uint64_t size{0};
        uint64_t memTag{0};
        std::vector<uint8_t> descBytes{};
        UrmaCommMem view{};
    };

    struct RemoteRankState {
        std::mutex rankMutex{};
        UrmaEndpointDesc remoteEndpointDesc{};
        bool hasEndpointDesc{false};
        // 本 rank 侧用于与该 peer 通信的 channel 描述符
        HcommChannelDesc channelDesc{};
        // 本 rank 侧用于与该 peer 通信的 channel handle
        HcommChannelHandle channel{0};
        // 本 rank 侧用于与该 peer 通信的 thread
        HcommThreadHandle thread{0};
        std::vector<RemoteRegistration> imports{};
        // Remote peer's notify record address (from TransportMemoryKey header during
        // ImportRemoteMemKeysLocked). Used as remote_flag_addr in kernel launch args.
        uint64_t remoteFlagAddr{0};
        uint64_t remoteFlagSize{0};
        // Raw hcommDesc bytes for the remote flag, used for HcommMemUnimport during cleanup.
        std::vector<uint8_t> remoteFlagDescBytes{};
    };

    struct RemoteMemKey {
        uint64_t addr;
        uint64_t size;
    };

    // Device kernel launch helpers (moved from removed HcomUrmaTransportAdapter)
    struct DeviceTransferBuffers {
        void *dstList{nullptr};
        void *srcList{nullptr};
        void *lenList{nullptr};
    };

    // A pending or deferred-free device transfer; owned by CompletionContext.
    struct PendingTransfer {
        uint32_t rankId{0};
        DeviceTransferBuffers buffers{};
        bool inFlight{false};
    };

    // Per-thread async completion context (manager-ownered via registry, weak TLS binding)
    struct CompletionContext {
        void *stream{nullptr}; // non-owning ACL stream, compared at each launch/sync
        void *notify{nullptr};
        uint32_t notifyId{0};
        uint64_t notifyAddr{0};
        uint64_t notifyLen{0};
        HcommMemHandle notifyHcommHandle{nullptr};
        bool initialized{false};
        // All in-flight and deferred-free transfers; emptied by Synchronize or CloseDevice.
        std::vector<PendingTransfer> pendingTransfers{};
    };

    // Open generation identity — unique per OpenDevice call
    struct OpenGeneration {
        uint64_t id{0};
    };

    // TLS binding: weak owner + weak context, does NOT own ACL/Hcomm resources
    struct ContextBinding {
        std::weak_ptr<OpenGeneration> owner;
        std::weak_ptr<CompletionContext> ctx;
    };

    // Initialization/open/close helpers and lifecycle
    Result InitLocalDeviceInfoLocked(const TransportOptions &options);
    Result OpenEndpointResourcesLocked(const TransportOptions &options);
    Result BuildLocalEndpointDescLocked(UrmaProtocol protocol, UrmaEndpointDesc &localDesc);
    Result CreateEndpointAndInitResourcesLocked(const UrmaEndpointDesc &localDesc);
    Result InitDeviceTransferFlagLocked();
    void RollbackOpenDeviceLocked();
    Result EnsureDeviceKernelLoadedLocked();
    static Result DestroyRankChannelsAndThread(RemoteRankState &state, uint32_t peerRank);
    Result UnimportPeerImportsAndFlag(RemoteRankState &state, uint32_t peerRank);
    Result CleanupPeerRankState(RemoteRankState &state, uint32_t peerRank);
    Result CleanupLocalRegistrationsLocked();

    // Local/remote registration and key export/import
    Result FindLocalRegistrationLocked(uint64_t addr, uint64_t size, LocalRegistration *registration) const;
    Result CorrectLocalRegAddressLocked(uint64_t addr, uint64_t size, uint64_t &correctedAddr) const;
    Result FindRemoteRegistrationLocked(uint32_t rankId, uint64_t addr, uint64_t size,
                                        RemoteRegistration *registration) const;
    Result ImportRemoteMemKeysLocked(uint32_t peerRank, RemoteRankState &state,
                                     const std::vector<TransportMemoryKey> &memKeys);

    // Prepare/rank/connection/private data
    Result RemoveRankLocked(uint32_t rankId);

    // Data transfer/copy
    Result RemoteIo(uint32_t rankId, uint64_t lAddr, uint64_t rAddr, uint64_t size, bool write);
    Result RemoteIoBatch(uint32_t rankId, const CopyDescriptor &descriptor, bool write);
    Result StageAndLaunchTransfer(CompletionContext &ctx, RemoteRankState &state, bool isRead,
                                  const std::vector<uint64_t> &localVec, const std::vector<uint64_t> &remoteVec,
                                  const std::vector<uint64_t> &sizeVec, uint32_t rankId);

    // Resolve remote registration addresses and build batch transfer vectors
    Result ResolveBatchIoAddressesLocked(uint32_t rankId, const CopyDescriptor &descriptor,
                                         std::vector<uint64_t> &localVec, std::vector<uint64_t> &remoteVec,
                                         std::vector<uint64_t> &sizeVec) const;

    // TLS(Thread Local Storage) binding container access (static thread_local via function-local static)
    static std::vector<ContextBinding> &GetTlsBindings();

    // Per-thread context lifecycle: lookup via TLS binding or create new
    CompletionContext *LookupOrCreateContextLocked();
    Result CreateAndPublishContextLocked(CompletionContext *&ctx);
    Result EnsureContextInitLocked(CompletionContext &ctx);
    void RollbackContextInitLocked(CompletionContext &ctx);
    void CleanupContextLocked(CompletionContext &ctx);

    // Find current thread's context via TLS binding only (no registry scan for owner)
    CompletionContext *FindCurrentContextLocked() const;

    // Synchronize and release a specific rank's pending transfers
    Result SynchronizeContextLocked(void *notify, void *stream, std::vector<PendingTransfer> &pendingTransfers);

    // CloseDevice helpers
    void CloseDeviceCleanupResourcesLocked();

    // Check if any context in registry has pending ops for a specific rank
    bool IsAnyRegistryContextPendingForRank(uint32_t rankId) const;

    // Device kernel buffer management
    aclrtFuncHandle GetDeviceKernelFunc(bool isRead) const;
    static Result ReleaseDeviceTransferBuffers(DeviceTransferBuffers &buffers);
    static Result ReleasePendingTransfersLocked(std::vector<PendingTransfer> &pendingTransfers);
    // Move all entries matching rankId from src to dst
    static void ExtractRankPending(std::vector<PendingTransfer> &src, uint32_t rankId,
                                   std::vector<PendingTransfer> &dst);
    // Move all entries back from src to dst (for error recovery)
    static void RestoreRankPending(std::vector<PendingTransfer> &src, std::vector<PendingTransfer> &dst);
    // Marker-launch + extract + sync + release for a single rank's pending
    Result SynchronizeRankPendingLocked(CompletionContext &ctx, RemoteRankState &state, uint32_t rankId,
                                        bool hasInFlight);

    // Device kernel launch helpers
    Result PrepareKernelLaunchBuffers(bool isRead, const std::vector<uint64_t> &localAddrs,
                                      const std::vector<uint64_t> &remoteAddrs, const std::vector<uint64_t> &sizes,
                                      DeviceTransferBuffers &outBuffers);
    // Device kernel launch (builds args, configures and launches)
    Result LaunchDeviceKernelBatch(const DeviceTransferBuffers &buffers, HcommThreadHandle thread, bool isRead,
                                   HcommChannelHandle channel, size_t batchSize);
    // Device kernel launch for marker-only notify (no data transfer)
    Result LaunchDeviceKernelNotify(HcommThreadHandle thread, HcommChannelHandle channel, uint64_t remoteFlagAddr,
                                    uint64_t notifyAddr, uint32_t notifyLen);

    mutable std::shared_mutex mutex_{};
    mutable std::shared_mutex registryMutex_{};
    bool opened_{false};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    uint32_t userDeviceId_{0};
    uint32_t logicDeviceId_{0};
    uint32_t phyDeviceId_{0};
    uint32_t sdid_{0};
    uint32_t serverId_{0};
    uint32_t superPodId_{0};
    TransportOptions options_{};
    HcommTransportManager manager_;
    UrmaEndpointHandle localEndpoint_{nullptr};
    UrmaEndpointDesc localEndpointDesc_{};
    std::map<uint64_t, LocalRegistration> localRegistrations_{};
    // Device kernel launch state
    bool deviceKernelLoaded_{false};
    aclrtBinHandle deviceKernelHandle_{nullptr};
    DeviceFuncHandles deviceFuncHandles_{};
    // Local flag buffer allocated via AclrtMalloc in OpenDevice, initialised to 1.
    void *devTransFlagPtr_{nullptr};
    uint64_t devTransFlagSize_{0};
    HcommMemHandle devTransFlagHcommHandle_{nullptr};
    // Open generation identity for this manager instance
    std::shared_ptr<OpenGeneration> owner_;
    // Strong registry of all per-thread completion contexts
    std::vector<std::shared_ptr<CompletionContext>> registry_{};
    std::unordered_map<uint32_t, RemoteRankState> remoteRanks_{};
};

} // namespace device
} // namespace transport
} // namespace mf
} // namespace ock

#endif // MF_HYBRID_DEVICE_URMA_TRANSPORT_MANAGER_H
