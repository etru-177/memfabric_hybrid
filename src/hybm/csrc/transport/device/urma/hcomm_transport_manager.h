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

#ifndef MF_HYBRID_HCOMM_TRANSPORT_MANAGER_H
#define MF_HYBRID_HCOMM_TRANSPORT_MANAGER_H

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "dl_hcomm_api.h"
#include "dl_rt_api.h"
#include "hybm_transport_manager.h"
#include "hybm_types.h"
#include "load_kernel.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {

constexpr uint32_t URMA_EXPORT_DESC_MAGIC = 0xA5FAB001U;
constexpr uint16_t URMA_EXPORT_DESC_VERSION = 1U;
constexpr uint32_t DEVICE_URMA_MAX_EXPORT_KEY_LENGTH = KEY_SIZE * 6;
constexpr uint32_t DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS = 2; // 表示keys前面的几个slots被占用
constexpr uint32_t DEVICE_URMA_EXPORT_KEY_DATA_BYTES =
    (DEVICE_URMA_MAX_EXPORT_KEY_LENGTH - DEVICE_URMA_EXPORT_KEY_HEADER_SLOTS) * sizeof(uint64_t);
constexpr void *INVALID_MEM_HANDLE = nullptr;

using UrmaMemTag = uint64_t;
using HcommEndpointHandle = ock::mf::EndpointHandle;
using HcommChannelHandle = ock::mf::ChannelHandle;
using HcommThreadHandle = ock::mf::ThreadHandle;

enum UrmaProtocol {
    RESERVED = -1, /* 保留协议类型 */
    HCCS = 0,      /* HCCS协议 */
    ROCE = 1,      /* RDMA over Converged Ethernet */
    PCIE = 2,      /* PCIe协议 */
    SIO = 3,       /* SIO协议 */
    UBC_CTP = 4,   /* 华为统一总线UBC_CTP */
    UBC_TP = 5,    /* 华为统一总线UBC_TP */
    UB_MEM = 6,    /* UB_MEM协议 */
    UBOE = 7,      /* UBoE协议 */
};

enum UrmaMemoryType : uint16_t {
    HOST_DRAM = 0,
    DEVICE_HBM = 1,
    INVALID_BUTT = 2,
};

inline std::ostream &operator<<(std::ostream &os, UrmaMemoryType obj)
{
    switch (obj) {
        case HOST_DRAM:
            return os << "DRAM";
        case DEVICE_HBM:
            return os << "DEVICE";
        case INVALID_BUTT:
            return os << "BUTT";
        default:
            return os << "UNKNOWN(" << static_cast<uint16_t>(obj) << ")";
    }
}

struct UrmaEndpointDesc {
    uint32_t devPhyId{0};
    uint32_t superDevId{0};
    uint32_t serverIdx{0};
    uint32_t superPodIdx{0};
    UrmaProtocol protocol{UrmaProtocol::RESERVED};
    CommAddrType type{COMM_ADDR_TYPE_RESERVED};
    uint8_t raws[URMA_ENDPOINT_RAW_LEN]{}; // CommAddr.raws
    EndpointLocType locType{ENDPOINT_LOC_TYPE_DEVICE};
    uint32_t hostId{0};
};

struct UrmaCommMem {
    uint64_t addr{0};
    uint64_t size{0};
    UrmaMemoryType type{UrmaMemoryType::INVALID_BUTT};
};

struct UrmaLocalMr {
    UrmaCommMem mem{};
    HcommMemHandle hcommMem{nullptr};
};

struct UrmaExportDesc {
    uint32_t magic{URMA_EXPORT_DESC_MAGIC};
    uint16_t version{URMA_EXPORT_DESC_VERSION};
    uint16_t headerSize{0};
    UrmaMemoryType memoryType{UrmaMemoryType::INVALID_BUTT};
    UrmaMemTag memTag{0};
    uint64_t addr{0};
    uint64_t size{0};
    uint32_t hcommDescLen{0};
    uint32_t devTransFlagDescLen{0};
};

struct MemEntry {
    HcommMemHandle handle{INVALID_MEM_HANDLE};
    UrmaMemTag memTag{0};
    UrmaCommMem mem{};
    UrmaLocalMr mr{};
    uint32_t refCount{0};
    bool exportCacheValid{false};
    std::vector<uint8_t> exportCache{};
};

struct UrmaEndpointEntity {
    HcommEndpointHandle hcommEndpoint{nullptr};
    UrmaEndpointDesc desc{};
    mutable std::mutex mutex{};
    uint64_t memRef{0};
    std::unordered_map<UrmaMemTag, HcommMemHandle> tagIndex{};
    std::unordered_map<HcommMemHandle, std::shared_ptr<MemEntry>> memEntries{};
};

using UrmaEndpointHandle = std::shared_ptr<UrmaEndpointEntity>;

bool GetRangeEnd(const UrmaCommMem &mem, uint64_t &end);
bool IsValidMem(const UrmaCommMem &mem);
EndpointDesc ToHcommEndpointDesc(const UrmaEndpointDesc &desc);
HcommCommMem ToHcommMem(const UrmaCommMem &mem);
std::string MakeMemTag(UrmaMemTag memTag);
bool SameMem(const UrmaCommMem &left, const UrmaCommMem &right);
bool Overlaps(const UrmaCommMem &left, const UrmaCommMem &right);
bool DeserializeExportDesc(const uint8_t *memDesc, uint32_t descLen, UrmaExportDesc &desc, const uint8_t **hcommDesc,
                           uint32_t *hcommDescLen);
Result HcomUrmaDestroyEndpoint(HcommEndpointHandle endpoint);

class HcommTransportManager final {
public:
    HcommTransportManager() = default;

    UrmaEndpointHandle CreateEndpoint(const UrmaEndpointDesc &desc) const;

    Result HcommMemReg(const UrmaEndpointHandle &endpoint, UrmaMemTag memTag, const UrmaCommMem &mem,
                       HcommMemHandle *memHandle);

    Result HcommMemUnreg(const UrmaEndpointHandle &endpoint, HcommMemHandle memHandle);

    Result HcommMemExport(const UrmaEndpointHandle &endpoint, HcommMemHandle memHandle, const uint8_t **memDesc,
                          uint32_t *memDescLen);

    Result HcommMemImport(const UrmaEndpointHandle &endpoint, const uint8_t *memDesc, uint32_t descLen,
                          UrmaCommMem *commMem);

    Result HcommMemUnimport(const UrmaEndpointHandle &endpoint, const uint8_t *memDesc, uint32_t descLen);

    Result WaitForChannelReady(HcommChannelHandle channel, uint32_t peerRank,
                               std::chrono::milliseconds timeout = std::chrono::seconds(100)) const;
};

} // namespace device
} // namespace transport
} // namespace mf
} // namespace ock

#endif // MF_HYBRID_HCOMM_TRANSPORT_MANAGER_H
