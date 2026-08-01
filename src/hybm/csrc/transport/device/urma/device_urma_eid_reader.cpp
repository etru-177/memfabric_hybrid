/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <fstream>
#include <string>
#include <array>
#include <arpa/inet.h>
#include <unistd.h>

#include "hybm_logger.h"
#include "dl_hcomm_api.h"
#include "topo_reader.h"
#include "device_urma_eid_reader.h"

namespace ock {
namespace mf {
namespace transport {
namespace device {

namespace {

// Check whether pclose exit code indicates success (WIFEXITED && WEXITSTATUS == 0).
static bool IsPcloseSuccess(int pcloseStatus)
{
    if (!WIFEXITED(pcloseStatus)) {
        return false;
    }
    return WEXITSTATUS(pcloseStatus) == 0;
}

// Single-quote a shell argument for safe use in popen.
// Replaces each embedded ' with '\'' per POSIX rules.
static std::string ShellQuote(const std::string &arg)
{
    std::string quoted;
    quoted.reserve(arg.size() + 4);
    quoted += '\'';
    for (auto ch : arg) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += '\'';
    return quoted;
}

// Return true when every character in s is a valid hex digit [0-9A-Fa-f].
static bool IsAllHex(const std::string &s)
{
    for (auto ch : s) {
        if (!std::isxdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

static uint8_t HexToNibble(char ch)
{
    if (ch >= '0' && ch <= '9') {
        return static_cast<uint8_t>(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
        return static_cast<uint8_t>(ch - 'a' + 10);
    }
    return static_cast<uint8_t>(ch - 'A' + 10);
}

static Result ParseLocalEidEnv(uint32_t phyDeviceId, uint32_t rankId, std::array<uint8_t, COMM_ADDR_EID_LEN> &eidData,
                               bool &isSet)
{
    constexpr const char *kLocalEidEnv = "USE_LOCAL_EID";
    const char *value = std::getenv(kLocalEidEnv);
    isSet = (value != nullptr);
    if (!isSet) {
        return BM_OK;
    }

    const std::string eidHex(value);
    constexpr size_t kEidHexLen = COMM_ADDR_EID_LEN * 2U;
    if (eidHex.size() != kEidHexLen || !IsAllHex(eidHex)) {
        BM_LOG_ERROR("device_urma invalid " << kLocalEidEnv << ", expected " << kEidHexLen
                                            << " hexadecimal characters, received length=" << eidHex.size()
                                            << ", phyDeviceId=" << phyDeviceId << ", rankId=" << rankId);
        return BM_INVALID_PARAM;
    }

    std::array<uint8_t, COMM_ADDR_EID_LEN> parsedEid{};
    for (size_t idx = 0; idx < parsedEid.size(); ++idx) {
        parsedEid[idx] =
            static_cast<uint8_t>((HexToNibble(eidHex[idx * 2U]) << 4U) | HexToNibble(eidHex[idx * 2U + 1U]));
    }
    eidData = parsedEid;
    BM_LOG_INFO("device_urma using EID from " << kLocalEidEnv << ", phyDeviceId=" << phyDeviceId
                                              << ", rankId=" << rankId);
    return BM_OK;
}

// Try to get IP from hccn_tool at the given toolPath.
// Only calls access() for paths with '/' (absolute/relative); bare names rely on popen/PATH.
// On success returns BM_OK and fills ipStr.
static Result TryGetIpFromHccnTool(const std::string &toolPath, uint32_t phyDeviceId, uint32_t rankId,
                                   std::string &ipStr)
{
    // access() does not search PATH — skip for bare names, let popen handle PATH.
    if (toolPath.find('/') != std::string::npos && access(toolPath.c_str(), X_OK) != 0) {
        BM_LOG_WARN("device_urma hccn_tool not found or not executable: " << toolPath << ", phyDeviceId=" << phyDeviceId
                                                                          << ", rankId=" << rankId);
        return BM_ERROR;
    }
    const std::string cmd =
        ShellQuote(toolPath) + " -g -ip -i " + std::to_string(phyDeviceId) + " -d bond" + std::to_string(phyDeviceId);
    FILE *pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        BM_LOG_WARN("device_urma popen hccn_tool failed, phyDeviceId=" << phyDeviceId << ", rankId=" << rankId);
        return BM_ERROR;
    }
    static constexpr size_t LINE_BUF_SIZE = 256;
    char buf[LINE_BUF_SIZE];
    bool found = false;
    while (fgets(buf, static_cast<int>(sizeof(buf)), pipe) != nullptr) {
        const std::string line(buf);
        const auto pos = line.find("ipaddr:");
        if (pos == std::string::npos) {
            continue;
        }
        ipStr = line.substr(pos + std::strlen("ipaddr:"));
        auto trimLeft = ipStr.find_first_not_of(" \t");
        if (trimLeft != std::string::npos) {
            ipStr = ipStr.substr(trimLeft);
        }
        auto trimRight = ipStr.find_last_not_of(" \t\r\n");
        if (trimRight != std::string::npos) {
            ipStr = ipStr.substr(0, trimRight + 1U);
        }
        found = true;
        break;
    }
    const int pcloseRet = pclose(pipe);
    if (!found || ipStr.empty() || !IsPcloseSuccess(pcloseRet)) {
        BM_LOG_WARN("device_urma hccn_tool no valid ipaddr, phyDeviceId=" << phyDeviceId << ", rankId=" << rankId);
        return BM_ERROR;
    }
    BM_LOG_DEBUG("device_urma from hccn_tool, devPhyId=" << phyDeviceId << " ip=" << ipStr);
    return BM_OK;
}

// Read config file at configPath, locate address_<phyDeviceId> after trimming,
// return the trimmed value. Ignores malformed/unrelated lines.
// address_1 does not match address_10. Returns BM_OK on success.
static Result TryGetIpFromConfig(const std::string &configPath, uint32_t phyDeviceId, uint32_t rankId,
                                 std::string &ipStr)
{
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
        BM_LOG_ERROR("device_urma cannot open /etc/hccn.conf: " << configPath << ", phyDeviceId=" << phyDeviceId
                                                                << ", rankId=" << rankId);
        return BM_ERROR;
    }
    const std::string targetKey = "address_" + std::to_string(phyDeviceId);
    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty()) {
            continue;
        }
        const auto eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, eqPos);
        auto trimLeft = key.find_first_not_of(" \t");
        auto trimRight = key.find_last_not_of(" \t\r\n");
        if (trimLeft == std::string::npos || trimRight == std::string::npos) {
            continue;
        }
        key = key.substr(trimLeft, trimRight - trimLeft + 1U);
        if (key != targetKey) {
            continue;
        }
        std::string value = line.substr(eqPos + 1U);
        trimLeft = value.find_first_not_of(" \t");
        if (trimLeft == std::string::npos) {
            continue;
        }
        value = value.substr(trimLeft);
        trimRight = value.find_last_not_of(" \t\r\n");
        if (trimRight != std::string::npos) {
            value = value.substr(0, trimRight + 1U);
        }
        if (!value.empty()) {
            ipStr = value;
            BM_LOG_DEBUG("device_urma from /etc/hccn.conf, devPhyId=" << phyDeviceId << " ip=" << ipStr);
            return BM_OK;
        }
    }
    BM_LOG_ERROR("device_urma key '" << targetKey << "' not found or empty in " << configPath
                                     << ", phyDeviceId=" << phyDeviceId << ", rankId=" << rankId);
    return BM_ERROR;
}

// Parse ipStr as IPv4 / IPv6 / 32-hex-IPv6 (with character pre-validation).
// Logs at WARN level — caller decides whether this is terminal.
static Result ParseIpStr(uint32_t phyDeviceId, const std::string &ipStr, const std::string &source,
                         CommAddrType &addrType, std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> &addrData)
{
    struct in_addr ipv4Addr;
    if (inet_pton(AF_INET, ipStr.c_str(), &ipv4Addr) == 1) {
        addrType = COMM_ADDR_TYPE_IP_V4;
        addrData.fill(0);
        std::memcpy(addrData.data(), &ipv4Addr, sizeof(ipv4Addr));
        BM_LOG_DEBUG("device_urma GetDeviceUrmaIpAddr ipv4, devPhyId=" << phyDeviceId << " ip=" << ipStr);
        return BM_OK;
    }
    struct in6_addr ipv6Addr;
    if (inet_pton(AF_INET6, ipStr.c_str(), &ipv6Addr) == 1) {
        addrType = COMM_ADDR_TYPE_IP_V6;
        addrData.fill(0);
        std::memcpy(addrData.data(), &ipv6Addr, sizeof(ipv6Addr));
        BM_LOG_DEBUG("device_urma GetDeviceUrmaIpAddr ipv6, devPhyId=" << phyDeviceId << " ip=" << ipStr);
        return BM_OK;
    }
    // 32 hex digits only (reject leading signs, whitespace, non-hex chars)
    if (ipStr.length() == 32U && IsAllHex(ipStr)) {
        std::array<uint8_t, 16U> raw{};
        for (size_t i = 0; i < 16U; ++i) {
            auto byteStr = ipStr.substr(i * 2, 2);
            raw[i] = static_cast<uint8_t>(std::strtoul(byteStr.c_str(), nullptr, 16U) & 0xFF);
        }
        addrType = COMM_ADDR_TYPE_IP_V6;
        addrData.fill(0);
        std::memcpy(addrData.data(), raw.data(), raw.size());
        BM_LOG_DEBUG("device_urma GetDeviceUrmaIpAddr ipv6-hex, devPhyId=" << phyDeviceId << " ip=" << ipStr);
        return BM_OK;
    }
    BM_LOG_WARN("device_urma invalid IP address format: '" << ipStr << "' for devPhyId " << phyDeviceId
                                                           << ", source: " << source);
    return BM_INVALID_PARAM;
}

} // anonymous namespace

Result GetDeviceUrmaIpAddrFromSources(const std::string &toolPath, const std::string &configPath, uint32_t phyDeviceId,
                                      uint32_t rankId, CommAddrType &addrType,
                                      std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> &addrData)
{
    std::string ipStr;
    Result ret = TryGetIpFromHccnTool(toolPath, phyDeviceId, rankId, ipStr);
    if (ret == BM_OK) {
        ret = ParseIpStr(phyDeviceId, ipStr, "hccn_tool", addrType, addrData);
        if (ret == BM_OK) {
            return BM_OK;
        }
        BM_LOG_WARN("device_urma hccn_tool returned invalid IP: '" << ipStr << "', phyDeviceId=" << phyDeviceId
                                                                   << ", rankId=" << rankId
                                                                   << ", falling back to config");
    } else {
        BM_LOG_WARN("device_urma hccn_tool failed for phyDeviceId=" << phyDeviceId << ", rankId=" << rankId
                                                                    << ", falling back to /etc/hccn.conf");
    }
    ret = TryGetIpFromConfig(configPath, phyDeviceId, rankId, ipStr);
    if (ret == BM_OK) {
        ret = ParseIpStr(phyDeviceId, ipStr, configPath, addrType, addrData);
        if (ret != BM_OK) {
            BM_LOG_ERROR("device_urma invalid IP value in " << configPath << " for phyDeviceId=" << phyDeviceId
                                                            << ", rankId=" << rankId << ", value='" << ipStr << "'");
        }
        return ret;
    }
    BM_LOG_ERROR("device_urma both hccn_tool and /etc/hccn.conf failed, phyDeviceId=" << phyDeviceId
                                                                                      << ", rankId=" << rankId);
    return BM_ERROR;
}

Result GetDeviceUrmaEid(uint32_t phyDeviceId, uint32_t rankId, std::array<uint8_t, COMM_ADDR_EID_LEN> &eidData)
{
    bool localEidSet = false;
    const Result localEidRet = ParseLocalEidEnv(phyDeviceId, rankId, eidData, localEidSet);
    if (localEidSet || localEidRet != BM_OK) {
        return localEidRet;
    }

    RootInfo ri;
    Result ret = TopoReader::ParseRootInfo(phyDeviceId, rankId, ri);
    if (ret != BM_OK) {
        return ret;
    }
    eidData = ri.eid;
    return BM_OK;
}

Result GetDeviceUrmaIpAddr(uint32_t phyDeviceId, uint32_t rankId, CommAddrType &addrType,
                           std::array<uint8_t, URMA_ENDPOINT_RAW_LEN> &addrData)
{
    std::string toolPath;
    if (access("/usr/local/Ascend/driver/tools/hccn_tool", X_OK) == 0) {
        toolPath = "/usr/local/Ascend/driver/tools/hccn_tool";
    } else {
        toolPath = "hccn_tool";
    }
    return GetDeviceUrmaIpAddrFromSources(toolPath, "/etc/hccn.conf", phyDeviceId, rankId, addrType, addrData);
}

} // namespace device
} // namespace transport
} // namespace mf
} // namespace ock
