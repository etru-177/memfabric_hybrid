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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers" // ignore pybind11 warning

#include "pytransfer.h"
#include <thread>
#include <chrono>
#include <cctype>
#include <iostream>
#include <unistd.h>
#include <pybind11/stl.h>
#include "mf_env_define.h"
#include "transfer_util.h"
#include "smem_ptracer.h"
#include "adapter_logger.h"

using namespace ock::adapter;

namespace py = pybind11;

static const char *PY_TRANSFER_LIB_VERSION =
    "library version: " STR2(PROJECT_VERSION_RAW) ", build time: " __DATE__ " " __TIME__
                                                  ", commit: " STR2(GIT_LAST_COMMIT);
constexpr uint64_t MAX_BATCH_COUNT = 1024 * 1024;

// static callback invoked by SMEM layer when a remote peer disconnects
void TransferAdapterPy::PeerDownCallback(const char *peerAddr, void *userData)
{
    if (userData == nullptr || peerAddr == nullptr)
        return;
    auto *self = static_cast<TransferAdapterPy *>(userData);
    self->OnLinkDownByPeerAddr(std::string(peerAddr));
}

TransferAdapterPy::TransferAdapterPy() {}

TransferAdapterPy::~TransferAdapterPy()
{
    // ensure consumer thread is stopped before destruction
    StopLinkDownConsumer();
    if (sockfd_ != -1) {
        close(sockfd_);
    }
}

int TransferAdapterPy::Initialize(const char *storeUrl, const char *uniqueId, const char *role, uint32_t deviceId,
                                  TransDataOpType dataOpType, const char *storeServerRole)
{
    if (strcmp(role, "Prefill") != 0 && strcmp(role, "Decode") != 0) {
        ADAPTER_LOG_ERROR("The value of role is invalid. Expected 'Prefill' or 'Decode.");
        return -1;
    }

    // set log level from env (SMEM OutLogger)
    const std::string &mfLevel = ock::mf::env::MF_LOG_LEVEL;
    if (mfLevel.size() == 1) {
        unsigned char c = static_cast<unsigned char>(mfLevel[0]);
        if (std::isdigit(c)) {
            int level = c - '0';
            smem_set_log_level(level);
            ock::mf::OutLogger::Instance().SetLogLevel(static_cast<ock::mf::LogLevel>(level));
        }
    }
    smem_set_conf_store_tls(false, nullptr, 0);

    // init config
    int32_t ret = smem_trans_config_init(&config_);
    ADAPTER_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "Failed to init smem_trans_config, ret=" << ret);

    config_.role = (strcmp(role, "Prefill") == 0) ? SMEM_TRANS_SENDER : SMEM_TRANS_RECEIVER;
    config_.deviceId = deviceId;
    config_.dataOpType = static_cast<smem_bm_data_op_type>(dataOpType);

    configStoreProtocol_ = GetConfigStoreProtocol(std::string(storeUrl));

    ret = smem_trans_init(&config_);
    ADAPTER_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "Failed to init smem_trans, ret=" << ret);

    bool isStoreServer = (strcmp(storeServerRole, role) == 0);
    std::string ip;
    uint16_t port = 0;
    ParseIpPortFromUniqueId(std::string(uniqueId), ip, port);
    ADAPTER_LOG_INFO("Begin to initialize trans, role=" << role << ", uniqueId=" << uniqueId << ", ip=" << ip
                                                        << ", port=" << port << ", isStoreServer=" << isStoreServer);

    if (isStoreServer) {
        return InitStoreServer(ip, port);
    }
    return InitStoreClient(ip, port);
}

int TransferAdapterPy::InitStoreServer(const std::string &ip, uint16_t port)
{
    int32_t ret = 0;
    std::string myUrl;
    if (port == 0) {
        uint16_t candidatePort = 0;
        for (uint32_t i = 0; i < PORT_SELECT_MAX_RETRY; ++i) {
            int probeFd = -1;
            candidatePort = AccFindAvailableTcpPortAdapter(probeFd);
            if (candidatePort == 0) {
                ADAPTER_LOG_ERROR("no available port for store server, ip=" << ip << ", attempt=" << i);
                return -1;
            }
            close(probeFd); // release so the real listener can bind+listen
            sessionId_ = BuildSessionId(ip, candidatePort);
            myUrl = configStoreProtocol_ + sessionId_;
            ret = smem_create_config_store(myUrl.c_str(), SMEM_STORE_SKIP_RECOVER);
            if (ret == 0) {
                break;
            }
            ADAPTER_LOG_WARN("allocate store server port after " << PORT_SELECT_MAX_RETRY << " retries, ip=" << ip);
            candidatePort = 0;
        }
        if (candidatePort == 0) {
            ADAPTER_LOG_ERROR("failed to allocate store server port after " << PORT_SELECT_MAX_RETRY
                                                                            << " retries, ip=" << ip);
            return -1;
        }
        port = candidatePort;
    } else {
        sessionId_ = BuildSessionId(ip, port);
        myUrl = configStoreProtocol_ + sessionId_;
        ret = smem_create_config_store(myUrl.c_str(), SMEM_STORE_SKIP_RECOVER);
        ADAPTER_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret,
                                              "smem_create_config_store failed, storeUrl=" << myUrl << ", ret=" << ret);
    }

    handle_ = smem_trans_create(myUrl.c_str(), sessionId_.c_str(), &config_);
    if (handle_ == nullptr) {
        ADAPTER_LOG_ERROR("smem_trans_create failed, storeUrl=" << myUrl << ", sessionId=" << sessionId_);
        return -1;
    }
    rpcPort_ = port;
    ADAPTER_LOG_INFO("initialized as store server, storeUrl=" << myUrl << ", rpcPort=" << port);
    return 0;
}

int TransferAdapterPy::InitStoreClient(const std::string &ip, uint16_t port)
{
    uint16_t candidatePort = port;
    if (candidatePort == 0) {
        candidatePort = AccFindAvailableTcpPortAdapter(sockfd_); // probe + keep sockfd_ bound (reserve for uniqueness)
        if (candidatePort == 0) {
            ADAPTER_LOG_ERROR("no available port for client session id, ip=" << ip);
            return -1;
        }
    }
    sessionId_ = BuildSessionId(ip, candidatePort);
    rpcPort_ = candidatePort;
    StartLinkDownConsumer();
    ADAPTER_LOG_INFO("initialized as store client, session=" << sessionId_ << ", rpcPort=" << candidatePort);
    return 0;
}

int TransferAdapterPy::GetRpcPort()
{
    if (rpcPort_ == 0) {
        ADAPTER_LOG_WARN("get_rpc_port returns 0; call initialize() first to get the real listening port");
    }
    ADAPTER_LOG_INFO("Get rpcPort is " << rpcPort_);
    return static_cast<int>(rpcPort_);
}

int TransferAdapterPy::TransferSyncWrite(const char *destUniqueId, uintptr_t buffer, uintptr_t peer_buffer_address,
                                         size_t length, uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    const void *srcAddress = reinterpret_cast<const void *>(buffer);
    void *destAddress = reinterpret_cast<void *>(peer_buffer_address);
    TP_TRACE_BEGIN(TP_TRANS_WRITE);
    int ret = smem_trans_write(handle, srcAddress, destUniqueId, destAddress, length, flags);
    TP_TRACE_END(TP_TRANS_WRITE, ret);
    TP_TRACE_RECORD(TP_TRANS_WRITE_SIZE, length * 1000ULL, 0);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_write error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::BatchTransferSyncWrite(const char *destUniqueId, std::vector<uintptr_t> buffers,
                                              std::vector<uintptr_t> peer_buffer_addresses, std::vector<size_t> lengths,
                                              uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    if (buffers.size() != peer_buffer_addresses.size() || buffers.size() != lengths.size() ||
        buffers.size() > UINT32_MAX) {
        ADAPTER_LOG_ERROR("Buffers, peer_buffer_addresses and lengths is not equal or too long.");
        return -1;
    }

    const size_t batchSize = buffers.size();
    std::vector<const void *> srcAddresses(batchSize);
    std::vector<void *> destAddresses(batchSize);
    std::vector<size_t> dataSizes(batchSize);
    size_t totalSize = 0;
    for (size_t i = 0; i < batchSize; ++i) {
        srcAddresses[i] = reinterpret_cast<const void *>(buffers[i]);
        destAddresses[i] = reinterpret_cast<void *>(peer_buffer_addresses[i]);
        dataSizes[i] = lengths[i];
        totalSize += lengths[i];
    }
    TP_TRACE_BEGIN(TP_TRANS_BATCH_WRITE);
    int ret = smem_trans_batch_write(handle, srcAddresses.data(), destUniqueId, destAddresses.data(), dataSizes.data(),
                                     static_cast<uint32_t>(batchSize), flags);
    TP_TRACE_END(TP_TRANS_BATCH_WRITE, ret);
    TP_TRACE_RECORD(TP_TRANS_BATCH_WRITE_SIZE, totalSize * 1000ULL, 0);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_batch_write error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::TransferSyncRead(const char *destUniqueId, uintptr_t buffer, uintptr_t peer_buffer_address,
                                        size_t length, uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    void *srcAddress = reinterpret_cast<void *>(buffer);
    const void *destAddress = reinterpret_cast<const void *>(peer_buffer_address);
    TP_TRACE_BEGIN(TP_TRANS_READ);
    int ret = smem_trans_read(handle, srcAddress, destUniqueId, destAddress, length, flags);
    TP_TRACE_END(TP_TRANS_READ, ret);
    TP_TRACE_RECORD(TP_TRANS_READ_SIZE, length * 1000ULL, 0);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_read error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::TransferAsyncReadSubmit(const char *destUniqueId, uintptr_t buffer,
                                               uintptr_t peer_buffer_address, size_t length, uintptr_t stream,
                                               uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    void *srcAddress = reinterpret_cast<void *>(buffer);
    const void *destAddress = reinterpret_cast<const void *>(peer_buffer_address);
    void *st = reinterpret_cast<void *>(stream);

    int ret = smem_trans_read_submit(handle, srcAddress, destUniqueId, destAddress, length, st, flags);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_read_submit error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::TransferAsyncWriteSubmit(const char *destUniqueId, uintptr_t buffer,
                                                uintptr_t peer_buffer_address, size_t length, uintptr_t stream,
                                                uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    const void *srcAddress = reinterpret_cast<const void *>(buffer);
    void *destAddress = reinterpret_cast<void *>(peer_buffer_address);
    void *st = reinterpret_cast<void *>(stream);

    int ret = smem_trans_write_submit(handle, srcAddress, destUniqueId, destAddress, length, st, flags);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_write_submit error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::BatchTransferSyncRead(const char *destUniqueId, std::vector<uintptr_t> buffers,
                                             std::vector<uintptr_t> peer_buffer_addresses, std::vector<size_t> lengths,
                                             uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    if (buffers.size() != peer_buffer_addresses.size() || buffers.size() != lengths.size() ||
        buffers.size() > UINT32_MAX) {
        ADAPTER_LOG_ERROR("Buffers, peer_buffer_addresses and lengths is not equal or too long.");
        return -1;
    }

    const size_t batchSize = buffers.size();
    std::vector<void *> srcAddresses(batchSize);
    std::vector<const void *> destAddresses(batchSize);
    std::vector<size_t> dataSizes(batchSize);
    size_t totalSize = 0;
    for (size_t i = 0; i < batchSize; ++i) {
        srcAddresses[i] = reinterpret_cast<void *>(buffers[i]);
        destAddresses[i] = reinterpret_cast<const void *>(peer_buffer_addresses[i]);
        dataSizes[i] = lengths[i];
        totalSize += lengths[i];
    }
    TP_TRACE_BEGIN(TP_TRANS_BATCH_READ);
    int ret = smem_trans_batch_read(handle, srcAddresses.data(), destUniqueId, destAddresses.data(), dataSizes.data(),
                                    static_cast<uint32_t>(batchSize), flags);
    TP_TRACE_END(TP_TRANS_BATCH_READ, ret);
    TP_TRACE_RECORD(TP_TRANS_BATCH_READ_SIZE, totalSize * 1000ULL, 0);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_batch_read error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::BatchTransferAsyncWriteSubmit(const char *destUniqueId, std::vector<uintptr_t> buffers,
                                                     std::vector<uintptr_t> peer_buffer_addresses,
                                                     std::vector<size_t> lengths, uintptr_t stream, uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    if (buffers.size() != peer_buffer_addresses.size() || buffers.size() != lengths.size() ||
        buffers.size() > UINT32_MAX) {
        ADAPTER_LOG_ERROR("Buffers, peer_buffer_addresses and lengths is not equal or too long.");
        return -1;
    }

    const size_t batchSize = buffers.size();
    std::vector<const void *> srcAddresses(batchSize);
    std::vector<void *> destAddresses(batchSize);
    std::vector<size_t> dataSizes(batchSize);
    void *st = reinterpret_cast<void *>(stream);

    for (size_t i = 0; i < batchSize; ++i) {
        srcAddresses[i] = reinterpret_cast<const void *>(buffers[i]);
        destAddresses[i] = reinterpret_cast<void *>(peer_buffer_addresses[i]);
        dataSizes[i] = lengths[i];
    }

    int ret = smem_trans_batch_write_submit(handle, srcAddresses.data(), destUniqueId, destAddresses.data(),
                                            dataSizes.data(), static_cast<uint32_t>(batchSize), st, flags);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_batch_write_submit error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::BatchTransferAsyncReadSubmit(const char *destUniqueId, std::vector<uintptr_t> buffers,
                                                    std::vector<uintptr_t> peer_buffer_addresses,
                                                    std::vector<size_t> lengths, uintptr_t stream, uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    if (buffers.size() != peer_buffer_addresses.size() || buffers.size() != lengths.size() ||
        buffers.size() > UINT32_MAX) {
        ADAPTER_LOG_ERROR("Buffers, peer_buffer_addresses and lengths is not equal or too long.");
        return -1;
    }

    const size_t batchSize = buffers.size();
    std::vector<void *> srcAddresses(batchSize);
    std::vector<const void *> destAddresses(batchSize);
    std::vector<size_t> dataSizes(batchSize);
    void *st = reinterpret_cast<void *>(stream);

    for (size_t i = 0; i < batchSize; ++i) {
        srcAddresses[i] = reinterpret_cast<void *>(buffers[i]);
        destAddresses[i] = reinterpret_cast<const void *>(peer_buffer_addresses[i]);
        dataSizes[i] = lengths[i];
    }

    int ret = smem_trans_batch_read_submit(handle, srcAddresses.data(), destUniqueId, destAddresses.data(),
                                           dataSizes.data(), static_cast<uint32_t>(batchSize), st, flags);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_batch_read_submit error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::BatchTransferWriteWithQuant(const char *destUniqueId, std::vector<uintptr_t> buffers,
                                                   std::vector<uintptr_t> peer_buffer_addresses,
                                                   std::vector<size_t> lengths, std::vector<uintptr_t> scale_addresses,
                                                   std::vector<uintptr_t> offset_addresses, uint32_t unit_num,
                                                   uint32_t input_type, uintptr_t stream, uint32_t flags)
{
    smem_trans_t handle = GetOrCreateConnection(destUniqueId);
    ADAPTER_ASSERT_RETURN(handle != nullptr, "handle is null", -1);

    if (buffers.size() != peer_buffer_addresses.size() || buffers.size() != lengths.size() ||
        buffers.size() > UINT32_MAX) {
        ADAPTER_LOG_ERROR("Buffers, peer_buffer_addresses and lengths is not equal or too long.");
        return -1;
    }

    if (scale_addresses.size() != 0 && scale_addresses.size() != buffers.size()) {
        ADAPTER_LOG_ERROR("scale_addresses size mismatch.");
        return -1;
    }
    if (offset_addresses.size() != 0 && offset_addresses.size() != buffers.size()) {
        ADAPTER_LOG_ERROR("offset_addresses size mismatch.");
        return -1;
    }

    uint32_t batchSize = buffers.size();
    std::vector<void *> srcAddresses(batchSize);
    std::vector<void *> destAddresses(batchSize);
    std::vector<size_t> dataSizes(batchSize);
    std::vector<float *> scaleAddresses(batchSize);
    std::vector<float *> offsetAddresses(batchSize);

    for (uint32_t i = 0; i < batchSize; ++i) {
        srcAddresses[i] = reinterpret_cast<void *>(buffers[i]);
        destAddresses[i] = reinterpret_cast<void *>(peer_buffer_addresses[i]);
        dataSizes[i] = lengths[i];
        scaleAddresses[i] = (scale_addresses.size() > 0) ? reinterpret_cast<float *>(scale_addresses[i]) : nullptr;
        offsetAddresses[i] = (offset_addresses.size() > 0) ? reinterpret_cast<float *>(offset_addresses[i]) : nullptr;
    }

    smem_trans_quant_copy_param_t param = {destUniqueId,
                                           srcAddresses.data(),
                                           destAddresses.data(),
                                           dataSizes.data(),
                                           scaleAddresses.data(),
                                           offsetAddresses.data(),
                                           batchSize,
                                           unit_num,
                                           reinterpret_cast<void *>(stream),
                                           input_type,
                                           flags};
    TP_TRACE_BEGIN(TP_TRANS_BATCH_WRITE_QUANT);
    int ret = smem_trans_batch_quant_write(handle, &param);
    TP_TRACE_END(TP_TRANS_BATCH_WRITE_QUANT, ret);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("SMEM API smem_trans_batch_quant_write error, ret=" << ret << " dest=" << destUniqueId);
    }
    return ret;
}

int TransferAdapterPy::RegisterMemory(uintptr_t buffer_addr, size_t capacity)
{
    if (handle_ == nullptr) {
        {
            std::lock_guard<std::mutex> regLock(regMemMutex_);
            for (const auto &m : registeredMems_) {
                if (m.addr == buffer_addr && m.capacity == capacity)
                    return 0;
            }
            registeredMems_.push_back({buffer_addr, capacity});
            ADAPTER_LOG_INFO("P registered memory addr=0x" << std::hex << buffer_addr << std::dec << " size="
                                                           << capacity << " (total: " << registeredMems_.size() << ")");
        }

        std::lock_guard<std::mutex> lock(connMutex_);
        for (auto &entry : connections_) {
            if (!entry.second.active || entry.second.handle == nullptr)
                continue;
            char *buffer = reinterpret_cast<char *>(buffer_addr);
            int ret = smem_trans_register_mem(entry.second.handle, buffer, capacity, 0);
            if (ret != 0) {
                ADAPTER_LOG_ERROR("broadcast register_mem to " << entry.first << " failed, ret=" << ret);
            }
        }
        return 0;
    }

    ADAPTER_ASSERT_RETURN(handle_ != nullptr, "handle_ is null", -1);
    char *buffer = reinterpret_cast<char *>(buffer_addr);
    return smem_trans_register_mem(handle_, buffer, capacity, 0);
}

int TransferAdapterPy::UnregisterMemory(uintptr_t buffer_addr)
{
    ADAPTER_ASSERT_RETURN(handle_ != nullptr, "handle_ is null", -1);
    char *buffer = reinterpret_cast<char *>(buffer_addr);
    return smem_trans_deregister_mem(handle_, buffer);
}

int TransferAdapterPy::BatchRegisterMemory(std::vector<uintptr_t> buffer_addrs, std::vector<size_t> capacities)
{
    if (buffer_addrs.size() != capacities.size()) {
        ADAPTER_LOG_ERROR("Size of buffer_addrs and capacities is not equal.");
        return -1;
    }

    const size_t count = buffer_addrs.size();
    if (count > MAX_BATCH_COUNT) {
        ADAPTER_LOG_ERROR("array size (" << count << ") exceeds limit(" << MAX_BATCH_COUNT << ")");
        return -1;
    }

    std::vector<void *> registerAddrs(count);
    for (size_t i = 0; i < count; ++i) {
        registerAddrs[i] = reinterpret_cast<void *>(buffer_addrs[i]);
    }

    if (handle_ == nullptr) {
        {
            std::lock_guard<std::mutex> regLock(regMemMutex_);
            for (size_t i = 0; i < count; ++i) {
                bool dup = false;
                for (const auto &m : registeredMems_) {
                    if (m.addr == buffer_addrs[i] && m.capacity == capacities[i]) {
                        dup = true;
                        break;
                    }
                }
                if (!dup)
                    registeredMems_.push_back({buffer_addrs[i], capacities[i]});
            }
            ADAPTER_LOG_INFO("P batch registered " << count << " memories (total: " << registeredMems_.size() << ")");
        }

        std::lock_guard<std::mutex> lock(connMutex_);
        for (auto &entry : connections_) {
            if (!entry.second.active || entry.second.handle == nullptr)
                continue;
            int ret = smem_trans_batch_register_mem(entry.second.handle, registerAddrs.data(), capacities.data(),
                                                    static_cast<uint32_t>(count), 0);
            if (ret != 0) {
                ADAPTER_LOG_ERROR("batch broadcast register_mem to " << entry.first << " failed, ret=" << ret);
            }
        }
        return 0;
    }

    // legacy mode or receiver
    ADAPTER_ASSERT_RETURN(handle_ != nullptr, "handle_ is null", -1);
    return smem_trans_batch_register_mem(handle_, registerAddrs.data(), capacities.data(), count, 0);
}

void TransferAdapterPy::TransferDestroy()
{
    StopLinkDownConsumer();

    if (handle_ == nullptr) {
        // multi-store: destroy all lazy connections
        std::lock_guard<std::mutex> lock(connMutex_);
        ADAPTER_LOG_INFO("destroying " << connections_.size() << " connections");
        for (auto &entry : connections_) {
            if (entry.second.handle != nullptr) {
                smem_trans_destroy(entry.second.handle, 0);
            }
        }
        connections_.clear();
    } else {
        // receiver or legacy single-store: destroy direct handle
        smem_trans_destroy(handle_, 0);
        handle_ = nullptr;
    }

    {
        std::lock_guard<std::mutex> regLock(regMemMutex_);
        registeredMems_.clear();
    }

    if (sockfd_ != -1) {
        close(sockfd_); // release the reserved client session port
        sockfd_ = -1;
    }
    rpcPort_ = 0;
}

void TransferAdapterPy::UnInitialize()
{
    StopLinkDownConsumer();
    if (sockfd_ != -1) {
        close(sockfd_);
        sockfd_ = -1;
    }
    rpcPort_ = 0;
    smem_trans_uninit(0);
}

// === connection management ===
std::string TransferAdapterPy::GetConfigStoreProtocol(const std::string &storeUrl)
{
    constexpr std::string::size_type protocolLen = 3; // length of "://"
    const std::string tcpProtocol = "tcp://";
    auto pos = storeUrl.find("://");
    if (pos == std::string::npos) {
        return tcpProtocol;
    }
    return storeUrl.substr(0, pos + protocolLen);
}

smem_trans_t TransferAdapterPy::GetOrCreateConnection(const std::string &sessionId)
{
    if (handle_ != nullptr) {
        return handle_;
    }

    std::unique_lock<std::mutex> lock(connMutex_);
    while (pendingConnections_.count(sessionId) != 0) {
        connCv_.wait(lock);
    }

    auto it = connections_.find(sessionId);
    if (it != connections_.end() && it->second.active) {
        return it->second.handle;
    }

    pendingConnections_.insert(sessionId);
    lock.unlock();

    // session_id format is now "IP:PORT" (no _pid suffix), the store URL prefix
    // is the session_id itself.
    std::string storeUrl = configStoreProtocol_ + sessionId;
    if (storeUrl.empty() || sessionId.empty()) {
        ADAPTER_LOG_ERROR("no store URL for session: " << sessionId << ", storeUrl=" << storeUrl);
        lock.lock();
        pendingConnections_.erase(sessionId);
        connCv_.notify_all();
        return nullptr;
    }

    ADAPTER_LOG_INFO("lazy connecting to D: " << sessionId << " via store: " << storeUrl);
    smem_trans_t handle = smem_trans_create(storeUrl.c_str(), sessionId_.c_str(), &config_);
    if (handle == nullptr) {
        ADAPTER_LOG_ERROR("lazy smem_trans_create failed for session: " << sessionId);
        lock.lock();
        pendingConnections_.erase(sessionId);
        connCv_.notify_all();
        return nullptr;
    }

    smem_trans_set_peer_down_callback(handle, TransferAdapterPy::PeerDownCallback, this);
    ReplayRegisteredMemories(handle);

    lock.lock();
    pendingConnections_.erase(sessionId);
    it = connections_.find(sessionId);
    if (it != connections_.end() && it->second.active) {
        smem_trans_destroy(handle, 0);
        ADAPTER_LOG_INFO("connection for " << sessionId << " already exists, reusing");
        connCv_.notify_all();
        return it->second.handle;
    }
    connections_[sessionId] = {handle, true};
    connCv_.notify_all();

    ADAPTER_LOG_INFO("lazy connected to D: " << sessionId << " (total connections: " << connections_.size() << ")");
    return handle;
}

void TransferAdapterPy::ReplayRegisteredMemories(smem_trans_t handle)
{
    if (handle == nullptr) {
        return;
    }
    std::vector<RegMem> snapshot;
    {
        std::lock_guard<std::mutex> lock(regMemMutex_);
        if (registeredMems_.empty()) {
            return;
        }
        snapshot = registeredMems_;
    }

    const size_t count = snapshot.size();
    std::vector<void *> addrs(count);
    std::vector<size_t> caps(count);
    for (size_t i = 0; i < count; ++i) {
        addrs[i] = reinterpret_cast<void *>(snapshot[i].addr);
        caps[i] = snapshot[i].capacity;
    }

    ADAPTER_LOG_INFO("replaying " << count << " registered memories on new connection");

    int ret = smem_trans_batch_register_mem(handle, addrs.data(), caps.data(), static_cast<uint32_t>(count), 0);
    if (ret != 0) {
        ADAPTER_LOG_ERROR("replay batch_register_mem failed, ret=" << ret);
    }
}

void TransferAdapterPy::OnLinkDownByPeerAddr(const std::string &peerAddr)
{
    ADAPTER_LOG_INFO("peer down event for addr: " << peerAddr);

    // find all connections matching this peer address prefix
    std::vector<std::string> matchedSessions;
    {
        std::lock_guard<std::mutex> lock(connMutex_);
        for (const auto &entry : connections_) {
            if (entry.first.find(peerAddr) != std::string::npos) {
                matchedSessions.push_back(entry.first);
            }
        }
    }

    if (matchedSessions.empty()) {
        ADAPTER_LOG_INFO("no active connection matching peer addr: " << peerAddr);
        return;
    }

    // push matched sessions to cleanup queue
    {
        std::lock_guard<std::mutex> lock(linkDownQueueMutex_);
        for (auto &sessionId : matchedSessions) {
            linkDownQueue_.push(sessionId);
            ADAPTER_LOG_INFO("queued link down cleanup: " << sessionId);
        }
    }
    linkDownCv_.notify_one();
}

void TransferAdapterPy::StartLinkDownConsumer()
{
    if (consumerRunning_) {
        ADAPTER_LOG_WARN("link down consumer already running");
        return;
    }
    consumerRunning_ = true;
    linkDownConsumerThread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "adpt_lkdwn");
        ADAPTER_LOG_INFO("link down consumer thread started");
        while (consumerRunning_) {
            std::string sessionId;
            {
                std::unique_lock<std::mutex> lock(linkDownQueueMutex_);
                linkDownCv_.wait(lock, [this]() { return !linkDownQueue_.empty() || !consumerRunning_; });
                if (!consumerRunning_ && linkDownQueue_.empty()) {
                    break;
                }
                if (linkDownQueue_.empty()) {
                    continue;
                }
                sessionId = std::move(linkDownQueue_.front());
                linkDownQueue_.pop();
            }

            ADAPTER_LOG_INFO("cleaning up disconnected D: " << sessionId);
            {
                std::lock_guard<std::mutex> lock(connMutex_);
                auto it = connections_.find(sessionId);
                if (it != connections_.end()) {
                    if (it->second.handle != nullptr) {
                        smem_trans_destroy(it->second.handle, 0);
                    }
                    connections_.erase(it);
                    connCv_.notify_all();
                    ADAPTER_LOG_INFO("link down cleanup done: " << sessionId << " (remaining connections: "
                                                                << connections_.size() << ")");
                } else {
                    ADAPTER_LOG_INFO("session " << sessionId << " already removed from connections");
                }
            }
        }
        ADAPTER_LOG_INFO("link down consumer thread stopped");
    });
}

void TransferAdapterPy::StopLinkDownConsumer()
{
    if (!consumerRunning_) {
        return;
    }
    ADAPTER_LOG_INFO("stopping link down consumer");
    consumerRunning_ = false;
    linkDownCv_.notify_one();
    if (linkDownConsumerThread_.joinable()) {
        linkDownConsumerThread_.join();
    }
}

void DefineAdapterFunctions(py::module_ &m)
{
    m.def("create_config_store", &pytransfer_create_config_store, py::call_guard<py::gil_scoped_release>(),
          py::arg("store_url"));

    m.def("set_log_level", &pytransfer_set_log_level, py::call_guard<py::gil_scoped_release>(), py::arg("level"), R"(
set log print level.

Arguments:
    level(int): log level, 0:debug 1:info 2:warn 3:error)");
    m.def("set_conf_store_tls", &pytransfer_set_conf_store_tls, py::call_guard<py::gil_scoped_release>(),
          py::arg("enable"), py::arg("tls_info"), R"(
set the config store tls info.
Parameters:
    enable (boolean): enable config store tls or not
        tls_info (string): tls config string
Returns:
    returns zero on success. On error, none-zero is returned.
)");
    m.doc() = PY_TRANSFER_LIB_VERSION;
}

PYBIND11_MODULE(_pymf_transfer, m)
{
    py::enum_<TransferAdapterPy::TransferOpcode> transfer_opcode(m, "TransferOpcode", py::arithmetic());
    transfer_opcode.value("Read", TransferAdapterPy::TransferOpcode::READ)
        .value("Write", TransferAdapterPy::TransferOpcode::WRITE)
        .export_values();
    py::enum_<TransferAdapterPy::TransDataOpType> transfer_type(m, "TransDataOpType", py::arithmetic());
    transfer_type.value("SDMA", TransferAdapterPy::TransDataOpType::SDMA)
        .value("DEVICE_RDMA", TransferAdapterPy::TransDataOpType::DEVICE_RDMA)
        .value("DEVICE_URMA", TransferAdapterPy::TransDataOpType::DEVICE_URMA)
        .value("DEVICE_UBOE", TransferAdapterPy::TransDataOpType::DEVICE_UBOE)
        .export_values();

    DefineAdapterFunctions(m);

    auto adaptor_cls =
        py::class_<TransferAdapterPy>(m, "TransferEngine")
            .def(py::init<>())
            .def("initialize", &TransferAdapterPy::Initialize, py::call_guard<py::gil_scoped_release>(),
                 py::arg("store_url"), py::arg("session_id"), py::arg("role"), py::arg("device_id"),
                 py::arg("data_op_type") = TransferAdapterPy::TransDataOpType::SDMA,
                 py::arg("store_server_role") = "Decode")
            .def("get_rpc_port", &TransferAdapterPy::GetRpcPort, py::call_guard<py::gil_scoped_release>())
            .def("transfer_sync_write", &TransferAdapterPy::TransferSyncWrite, py::call_guard<py::gil_scoped_release>(),
                 py::arg("dest_session"), py::arg("buffer"), py::arg("peer_buffer"), py::arg("length"),
                 py::arg("flags") = 0)
            .def("batch_transfer_sync_write", &TransferAdapterPy::BatchTransferSyncWrite,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffers"),
                 py::arg("peer_buffers"), py::arg("lengths"), py::arg("flags") = 0)
            .def("batch_transfer_async_write_submit", &TransferAdapterPy::BatchTransferAsyncWriteSubmit,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffers"),
                 py::arg("peer_buffers"), py::arg("lengths"), py::arg("stream"), py::arg("flags") = 0)
            .def("transfer_sync_read", &TransferAdapterPy::TransferSyncRead, py::call_guard<py::gil_scoped_release>(),
                 py::arg("dest_session"), py::arg("buffer"), py::arg("peer_buffer"), py::arg("length"),
                 py::arg("flags") = 0)
            .def("transfer_async_write_submit", &TransferAdapterPy::TransferAsyncWriteSubmit,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffer"),
                 py::arg("peer_buffer"), py::arg("length"), py::arg("stream"), py::arg("flags") = 0)
            .def("transfer_async_read_submit", &TransferAdapterPy::TransferAsyncReadSubmit,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffer"),
                 py::arg("peer_buffer"), py::arg("length"), py::arg("stream"), py::arg("flags") = 0)
            .def("batch_transfer_sync_read", &TransferAdapterPy::BatchTransferSyncRead,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffers"),
                 py::arg("peer_buffers"), py::arg("lengths"), py::arg("flags") = 0)
            .def("batch_transfer_async_read_submit", &TransferAdapterPy::BatchTransferAsyncReadSubmit,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffers"),
                 py::arg("peer_buffers"), py::arg("lengths"), py::arg("stream"), py::arg("flags") = 0)
            .def("batch_transfer_write_with_quant", &TransferAdapterPy::BatchTransferWriteWithQuant,
                 py::call_guard<py::gil_scoped_release>(), py::arg("dest_session"), py::arg("buffers"),
                 py::arg("peer_buffers"), py::arg("lengths"), py::arg("scale_buffers"), py::arg("offset_buffers"),
                 py::arg("unit_num"), py::arg("input_type") = 0, py::arg("stream") = 0, py::arg("flags") = 0)
            .def("register_memory", &TransferAdapterPy::RegisterMemory, py::call_guard<py::gil_scoped_release>(),
                 py::arg("buffer_addr"), py::arg("capacity"))
            .def("unregister_memory", &TransferAdapterPy::UnregisterMemory, py::call_guard<py::gil_scoped_release>(),
                 py::arg("buffer_addr"))
            .def("batch_register_memory", &TransferAdapterPy::BatchRegisterMemory,
                 py::call_guard<py::gil_scoped_release>(), py::arg("buffer_addrs"), py::arg("capacities"))
            .def("destroy", &TransferAdapterPy::TransferDestroy, py::call_guard<py::gil_scoped_release>())
            .def("unInitialize", &TransferAdapterPy::UnInitialize, py::call_guard<py::gil_scoped_release>());

    adaptor_cls.attr("TransferOpcode") = transfer_opcode;
    adaptor_cls.attr("TransDataOpType") = transfer_type;
}

#pragma GCC diagnostic pop
