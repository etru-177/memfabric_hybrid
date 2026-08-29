/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of the License at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "hybm_urma_eid_query.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>

#include "hybm_logger.h"

namespace ock {
namespace mf {
namespace {

constexpr uint32_t DCMI_DSMI_MAIN_COMMAND_UB = 62U;
constexpr uint32_t DCMI_DSMI_URMA_NAME_SUB_COMMAND = 3U;
constexpr const char *DCMI_LIBRARY_NAME = "libdcmi.so";
constexpr const char *DSMI_LIBRARY_NAME = "libdrvdsmi_host.so";
constexpr const char *UBCORE_SYSFS_ROOT = "/sys/class/ubcore";
constexpr const char *UBURMA_SYSFS_ROOT = "/sys/class/uburma";
constexpr size_t MAX_UDMA_NAME = 32U;
constexpr size_t MAX_EIDS_PER_DEVICE = 32U;
constexpr size_t MAX_URMA_DEVICES = 128U;
constexpr uint32_t MAX_HOST_EIDS = 1024U;
constexpr size_t EID_HEX_LENGTH = 32U;
constexpr size_t EID_BYTE_LENGTH = 16U;
constexpr int32_t HEX_BYTE_WIDTH = 2;
constexpr uint8_t HEX_ALPHA_OFFSET = 10U;
constexpr size_t IPV4_MAPPED_PREFIX_OFFSET = 10U;
constexpr size_t IPV4_MAPPED_PREFIX_LENGTH = 2U;
constexpr size_t IPV4_MAPPED_ADDRESS_OFFSET = IPV4_MAPPED_PREFIX_OFFSET + IPV4_MAPPED_PREFIX_LENGTH;
constexpr unsigned char IPV4_MAPPED_PREFIX_VALUE = 0xFFU;
constexpr uint32_t SUPER_POD_NPU_GROUP_SIZE = 8U;
constexpr uint32_t SUPER_POD_FIRST_MESH_DIE_LAST_NPU = 3U;

constexpr uint32_t SUPER_POD_MAINBOARD_1 = 0x3U;
constexpr uint32_t SUPER_POD_MAINBOARD_2 = 0x5U;
constexpr uint32_t SUPER_POD_MAINBOARD_3 = 0x7U;
constexpr uint32_t SERVER_MAINBOARD_MIN_1 = 0x21U;
constexpr uint32_t SERVER_MAINBOARD_MAX_1 = 0x2BU;
constexpr uint32_t SERVER_MAINBOARD_MIN_2 = 0x40U;
constexpr uint32_t SERVER_MAINBOARD_MAX_2 = 0x46U;

struct EidByteInfo {
    uint8_t dieId = 0;
    bool isPg = false;
    uint8_t port = 0;
};

struct IndexedEid {
    std::string eid;
};

struct UrmaDevice {
    std::vector<std::string> eids;
};

struct UrmaEidQueryOptions {
    uint32_t physicalDeviceId = 0;
    UrmaEidTopology topology = UrmaEidTopology::AUTO;
};

struct UrmaEidQueryResult {
    uint32_t logicalDeviceId = 0;
    UrmaEidTopology topology = UrmaEidTopology::AUTO;
    int32_t meshDieId = -1;
    std::string udma;
    std::string hostEid;
    std::string deviceEid;
};

struct UrmaEidQueryError {
    Result result = BM_OK;
};

const char *TopologyName(UrmaEidTopology topology)
{
    if (topology == UrmaEidTopology::SERVER) {
        return "server";
    }
    if (topology == UrmaEidTopology::SUPER_POD) {
        return "super_pod";
    }
    return "auto";
}

bool Fail(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, UrmaEidQueryError &error, const char *stage,
          Result result, int32_t nativeCode, const std::string &detail)
{
    error.result = result;
    BM_LOG_ERROR("Query URMA EID failed, stage=" << stage << ", physicalDeviceId=" << options.physicalDeviceId
                                                 << ", logicalDeviceId=" << logicalDeviceId
                                                 << ", nativeCode=" << nativeCode << ", detail=" << detail);
    return false;
}

void Info(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, const char *stage, const std::string &detail)
{
    BM_LOG_DEBUG("Query URMA EID, stage=" << stage << ", physicalDeviceId=" << options.physicalDeviceId
                                          << ", logicalDeviceId=" << logicalDeviceId << ", detail=" << detail);
}

std::string FormatBytes(const unsigned char *bytes, size_t size)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t index = 0; index < size; ++index) {
        stream << std::setw(HEX_BYTE_WIDTH) << static_cast<unsigned int>(bytes[index]);
    }
    return stream.str();
}

bool NormalizeHexEid(const std::string &input, std::string &output)
{
    output.clear();
    output.reserve(EID_HEX_LENGTH);
    for (const char character : input) {
        if (character == ':') {
            continue;
        }
        if (!std::isxdigit(static_cast<unsigned char>(character))) {
            output.clear();
            return false;
        }
        output.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    if (output.size() != EID_HEX_LENGTH || output == std::string(EID_HEX_LENGTH, '0')) {
        output.clear();
        return false;
    }
    return true;
}

bool ParseIpv4Value(const std::string &input, uint32_t &value)
{
    if (input.empty() || input.front() == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(input.c_str(), &end, 0);
    if (errno != 0 || end == input.c_str() || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool NormalizeEid(const std::string &input, std::string &output)
{
    if (NormalizeHexEid(input, output)) {
        return true;
    }
    unsigned char raw[EID_BYTE_LENGTH] = {};
    if (inet_pton(AF_INET6, input.c_str(), raw) == 1) {
        output = FormatBytes(raw, sizeof(raw));
        return output != std::string(EID_HEX_LENGTH, '0');
    }
    uint32_t ipv4 = 0;
    if (inet_pton(AF_INET, input.c_str(), &ipv4) != 1 && !ParseIpv4Value(input, ipv4)) {
        output.clear();
        return false;
    }
    std::fill_n(raw + IPV4_MAPPED_PREFIX_OFFSET, IPV4_MAPPED_PREFIX_LENGTH, IPV4_MAPPED_PREFIX_VALUE);
    const uint32_t networkIpv4 = input.find('.') == std::string::npos ? htonl(ipv4) : ipv4;
    std::memcpy(raw + IPV4_MAPPED_ADDRESS_OFFSET, &networkIpv4, sizeof(networkIpv4));
    output = FormatBytes(raw, sizeof(raw));
    return true;
}

uint8_t HexValue(char character)
{
    if (character >= '0' && character <= '9') {
        return static_cast<uint8_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
        return static_cast<uint8_t>(character - 'a' + HEX_ALPHA_OFFSET);
    }
    return static_cast<uint8_t>(character - 'A' + HEX_ALPHA_OFFSET);
}

bool ParseEidByte(const std::string &eid, EidByteInfo &info)
{
    std::string normalized;
    if (!NormalizeEid(eid, normalized)) {
        return false;
    }
    const uint8_t byte6 = static_cast<uint8_t>((HexValue(normalized[10]) << 4U) | HexValue(normalized[11]));
    info.dieId = static_cast<uint8_t>(byte6 >> 6U);
    info.port = static_cast<uint8_t>(byte6 & 0x3FU);
    info.isPg = info.port == 0x3FU;
    return true;
}

int32_t MeshDieId(uint32_t physicalDeviceId, UrmaEidTopology topology)
{
    if (topology == UrmaEidTopology::SERVER) {
        return 1;
    }
    return physicalDeviceId % SUPER_POD_NPU_GROUP_SIZE <= SUPER_POD_FIRST_MESH_DIE_LAST_NPU ? 0 : 1;
}

bool IsServerMainboard(uint32_t mainboardId)
{
    const bool firstRange =
        mainboardId >= SERVER_MAINBOARD_MIN_1 && mainboardId <= SERVER_MAINBOARD_MAX_1 && (mainboardId % 2U == 1U);
    const bool secondRange =
        mainboardId >= SERVER_MAINBOARD_MIN_2 && mainboardId <= SERVER_MAINBOARD_MAX_2 && (mainboardId % 2U == 0U);
    return firstRange || secondRange;
}

bool IsSuperPodMainboard(uint32_t mainboardId)
{
    return mainboardId == SUPER_POD_MAINBOARD_1 || mainboardId == SUPER_POD_MAINBOARD_2 ||
           mainboardId == SUPER_POD_MAINBOARD_3;
}

union DcmiUrmaEid {
    unsigned char raw[16];
    struct {
        unsigned long subnetPrefix;
        unsigned long interfaceId;
    } in6;
};

struct DcmiUrmaEidInfo {
    DcmiUrmaEid eid;
    unsigned int eidIndex;
};

static_assert(sizeof(unsigned long) == 8U, "DCMI ABI requires a 64-bit Linux host");
static_assert(sizeof(DcmiUrmaEid) == 16U, "unexpected DCMI EID ABI size");
static_assert(sizeof(DcmiUrmaEidInfo) == 24U, "unexpected DCMI EID info ABI size");

std::string DynamicLoaderError()
{
    const char *message = dlerror();
    return message == nullptr ? "unknown dynamic-loader error" : message;
}

std::string FormatRawEid(const DcmiUrmaEid &eid)
{
    return FormatBytes(eid.raw, sizeof(eid.raw));
}

class DcmiApi {
public:
    using InitFn = int32_t (*)();
    using GetCountFn = int32_t (*)(int32_t, uint32_t *);
    using GetEidListFn = int32_t (*)(int32_t, int32_t, DcmiUrmaEidInfo *, int32_t *);
    using GetMainboardFn = int32_t (*)(int32_t, uint32_t *);
    using GetLogicalFn = int32_t (*)(uint32_t, uint32_t *);

    ~DcmiApi()
    {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    bool Load(const UrmaEidQueryOptions &options, UrmaEidQueryError &error)
    {
        const std::string path = DCMI_LIBRARY_NAME;
        handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return Fail(options, -1, error, "dcmi_load", BM_DL_FUNCTION_FAILED, 0,
                        "dlopen " + path + " failed: " + DynamicLoaderError());
        }
        init_ = LoadSymbol<InitFn>("dcmiv2_init");
        getCount_ = LoadSymbol<GetCountFn>("dcmiv2_get_urma_device_cnt");
        getEidList_ = LoadSymbol<GetEidListFn>("dcmiv2_get_eid_list_by_urma_dev_index");
        getLogical_ = LoadSymbol<GetLogicalFn>("dcmiv2_get_dev_id_by_chip_phy_id");
        if (getLogical_ == nullptr) {
            getLogical_ = LoadSymbol<GetLogicalFn>("dcmiv2_get_dev_id_from_chip_phyid");
        }
        getMainboard_ = LoadSymbol<GetMainboardFn>("dcmiv2_get_mainboard_id");
        if (init_ == nullptr || getCount_ == nullptr || getEidList_ == nullptr || getLogical_ == nullptr) {
            return Fail(options, -1, error, "dcmi_symbol", BM_DL_FUNCTION_FAILED, 0,
                        "required DCMI symbol is missing from " + path);
        }
        Info(options, -1, "dcmi_load", "loaded " + path);
        return true;
    }

    bool Initialize(const UrmaEidQueryOptions &options, UrmaEidQueryError &error) const
    {
        int32_t ret = -1;
        for (uint32_t attempt = 0; attempt < 10U; ++attempt) {
            ret = init_();
            if (ret == 0) {
                return true;
            }
            if (attempt + 1U < 10U) {
                sleep(1);
            }
        }
        return Fail(options, -1, error, "dcmi_init", BM_DL_FUNCTION_FAILED, ret,
                    "dcmiv2_init failed after 10 attempts");
    }

    bool MapPhysicalToLogical(const UrmaEidQueryOptions &options, uint32_t &logicalDeviceId,
                              UrmaEidQueryError &error) const
    {
        const int32_t ret = getLogical_(options.physicalDeviceId, &logicalDeviceId);
        if (ret != 0) {
            return Fail(options, -1, error, "physical_to_logical", BM_ERROR, ret,
                        "DCMI physical-to-logical mapping failed");
        }
        return true;
    }

    bool GetMainboard(const UrmaEidQueryOptions &options, uint32_t logicalDeviceId, uint32_t &mainboardId,
                      UrmaEidQueryError &error) const
    {
        if (getMainboard_ == nullptr) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "topology", BM_ERROR, 0,
                        "mainboard API is unavailable; specify server or super pod topology");
        }
        const int32_t ret = getMainboard_(static_cast<int32_t>(logicalDeviceId), &mainboardId);
        if (ret != 0) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "topology", BM_ERROR, ret,
                        "DCMI mainboard query failed");
        }
        return true;
    }

    bool GetDevices(const UrmaEidQueryOptions &options, uint32_t logicalDeviceId, std::vector<UrmaDevice> &devices,
                    UrmaEidQueryError &error) const
    {
        uint32_t deviceCount = 0;
        const int32_t ret = getCount_(static_cast<int32_t>(logicalDeviceId), &deviceCount);
        if (ret != 0 || deviceCount == 0U || deviceCount > MAX_URMA_DEVICES) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dcmi_eid_list", BM_ERROR, ret,
                        "invalid URMA device count=" + std::to_string(deviceCount));
        }
        devices.clear();
        for (uint32_t deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
            if (!AppendDevice(options, logicalDeviceId, deviceIndex, devices, error)) {
                return false;
            }
        }
        if (devices.empty()) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dcmi_eid_list", BM_ERROR, 0,
                        "DCMI returned no non-zero URMA EIDs");
        }
        return true;
    }

private:
    template<typename Function>
    Function LoadSymbol(const char *name) const
    {
        return reinterpret_cast<Function>(dlsym(handle_, name));
    }

    bool AppendDevice(const UrmaEidQueryOptions &options, uint32_t logicalDeviceId, uint32_t deviceIndex,
                      std::vector<UrmaDevice> &devices, UrmaEidQueryError &error) const
    {
        DcmiUrmaEidInfo buffer[MAX_EIDS_PER_DEVICE] = {};
        int32_t eidCount = static_cast<int32_t>(MAX_EIDS_PER_DEVICE);
        const int32_t ret =
            getEidList_(static_cast<int32_t>(logicalDeviceId), static_cast<int32_t>(deviceIndex), buffer, &eidCount);
        if (ret != 0 || eidCount < 0 || eidCount > static_cast<int32_t>(MAX_EIDS_PER_DEVICE)) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dcmi_eid_list", BM_ERROR, ret,
                        "invalid EID list for URMA device index=" + std::to_string(deviceIndex) +
                            " eid_count=" + std::to_string(eidCount));
        }
        UrmaDevice device;
        for (int32_t index = 0; index < eidCount; ++index) {
            std::string eid;
            if (NormalizeEid(FormatRawEid(buffer[index].eid), eid)) {
                device.eids.push_back(std::move(eid));
            }
        }
        if (!device.eids.empty()) {
            devices.push_back(std::move(device));
        }
        return true;
    }

    void *handle_ = nullptr;
    InitFn init_ = nullptr;
    GetCountFn getCount_ = nullptr;
    GetEidListFn getEidList_ = nullptr;
    GetMainboardFn getMainboard_ = nullptr;
    GetLogicalFn getLogical_ = nullptr;
};

class DsmiApi {
public:
    using GetDeviceInfoFn = int (*)(uint32_t, uint32_t, uint32_t, void *, uint32_t *);

    ~DsmiApi()
    {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    bool Load(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, UrmaEidQueryError &error)
    {
        const std::string path = DSMI_LIBRARY_NAME;
        handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return Fail(options, logicalDeviceId, error, "dsmi_load", BM_DL_FUNCTION_FAILED, 0,
                        "dlopen " + path + " failed: " + DynamicLoaderError());
        }
        getDeviceInfo_ = reinterpret_cast<GetDeviceInfoFn>(dlsym(handle_, "dsmi_get_device_info"));
        if (getDeviceInfo_ == nullptr) {
            return Fail(options, logicalDeviceId, error, "dsmi_symbol", BM_DL_FUNCTION_FAILED, 0,
                        "dsmi_get_device_info is missing from " + path);
        }
        Info(options, logicalDeviceId, "dsmi_load", "loaded " + path);
        return true;
    }

    bool GetUdmaName(const UrmaEidQueryOptions &options, uint32_t logicalDeviceId, std::string &name,
                     UrmaEidQueryError &error) const
    {
        char buffer[MAX_UDMA_NAME] = {};
        uint32_t bufferSize = static_cast<uint32_t>(sizeof(buffer));
        const int ret = getDeviceInfo_(logicalDeviceId, DCMI_DSMI_MAIN_COMMAND_UB, DCMI_DSMI_URMA_NAME_SUB_COMMAND,
                                       buffer, &bufferSize);
        if (ret != 0) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dsmi_udma", BM_ERROR, ret,
                        "dsmi_get_device_info(UB, URMA_DEV_NAME) failed");
        }
        const size_t nameLength = ::strnlen(buffer, sizeof(buffer));
        if (nameLength == sizeof(buffer)) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dsmi_udma", BM_ERROR, 0,
                        "DSMI returned a non-terminated UDMA name");
        }
        name.assign(buffer, nameLength);
        if (name.empty()) {
            return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "dsmi_udma", BM_ERROR, 0,
                        "DSMI returned an empty UDMA name");
        }
        return true;
    }

private:
    void *handle_ = nullptr;
    GetDeviceInfoFn getDeviceInfo_ = nullptr;
};

std::string Trim(const std::string &value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1U]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool ReadLine(const std::string &path, std::string &value)
{
    std::ifstream input(path);
    if (!input.is_open() || !std::getline(input, value)) {
        return false;
    }
    value = Trim(value);
    return true;
}

bool ParseUint32(const std::string &text, uint32_t &value)
{
    if (text.empty() || text.front() == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

std::string JoinPath(const std::string &parent, const std::string &child)
{
    if (!parent.empty() && parent.back() == '/') {
        return parent + child;
    }
    return parent + "/" + child;
}

bool IsSafeDeviceName(const std::string &name)
{
    if (name.empty() || name == "." || name == "..") {
        return false;
    }
    for (const char character : name) {
        if (!std::isalnum(static_cast<unsigned char>(character)) && character != '_' && character != '-' &&
            character != '.') {
            return false;
        }
    }
    return true;
}

void AppendUniquePath(const std::string &path, std::vector<std::string> &paths)
{
    for (const std::string &existing : paths) {
        if (existing == path) {
            return;
        }
    }
    paths.push_back(path);
}

void FindMatchingSysfsEntries(const std::string &root, const std::string &udma, std::vector<std::string> &matches)
{
    const std::string directPath = JoinPath(root, udma);
    std::string directName;
    if (access(directPath.c_str(), F_OK) == 0 &&
        (!ReadLine(JoinPath(directPath, "ubdev"), directName) || directName == udma)) {
        AppendUniquePath(directPath, matches);
    }
    DIR *directory = opendir(root.c_str());
    if (directory == nullptr) {
        return;
    }
    for (dirent *entry = readdir(directory); entry != nullptr; entry = readdir(directory)) {
        const std::string entryName(entry->d_name);
        if (!IsSafeDeviceName(entryName)) {
            continue;
        }
        const std::string entryPath = JoinPath(root, entryName);
        std::string ubdev;
        if (ReadLine(JoinPath(entryPath, "ubdev"), ubdev) && ubdev == udma) {
            AppendUniquePath(entryPath, matches);
        }
    }
    closedir(directory);
}

bool ResolveSysfsDevicePath(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, const std::string &udma,
                            std::string &devicePath, UrmaEidQueryError &error)
{
    if (!IsSafeDeviceName(udma)) {
        return Fail(options, logicalDeviceId, error, "host_eid_sysfs", BM_ERROR, 0,
                    "DSMI returned an unsafe UDMA name=" + udma);
    }
    const std::vector<std::string> roots = {UBCORE_SYSFS_ROOT, UBURMA_SYSFS_ROOT};
    for (const std::string &root : roots) {
        std::vector<std::string> matches;
        FindMatchingSysfsEntries(root, udma, matches);
        if (matches.size() == 1U) {
            devicePath = matches.front();
            return true;
        }
        if (matches.size() > 1U) {
            return Fail(options, logicalDeviceId, error, "host_eid_sysfs", BM_ERROR, 0,
                        "multiple UDMA sysfs devices match " + udma + " under " + root +
                            ", candidate_count=" + std::to_string(matches.size()));
        }
    }
    return Fail(options, logicalDeviceId, error, "host_eid_sysfs", BM_ERROR, 0,
                "UDMA sysfs device not found for " + udma);
}

bool ReadIndexedEid(const std::string &devicePath, uint32_t index, std::string &value)
{
    const std::string currentPath = JoinPath(devicePath, "eids/eid" + std::to_string(index));
    if (ReadLine(currentPath, value)) {
        return true;
    }
    const std::string legacyPath = JoinPath(devicePath, "eid" + std::to_string(index) + "/eid");
    return ReadLine(legacyPath, value);
}

bool ReadHostEids(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, const std::string &udma,
                  std::vector<IndexedEid> &eids, UrmaEidQueryError &error)
{
    std::string devicePath;
    if (!ResolveSysfsDevicePath(options, logicalDeviceId, udma, devicePath, error)) {
        return false;
    }
    std::string countText;
    uint32_t maxEidCount = 0;
    const std::string countPath = JoinPath(devicePath, "max_eid_cnt");
    if (!ReadLine(countPath, countText) || !ParseUint32(countText, maxEidCount) || maxEidCount == 0U ||
        maxEidCount > MAX_HOST_EIDS) {
        return Fail(options, logicalDeviceId, error, "host_eid_sysfs", BM_ERROR, 0,
                    "invalid max_eid_cnt at " + countPath + " value=" + countText);
    }
    eids.clear();
    for (uint32_t index = 0; index < maxEidCount; ++index) {
        std::string value;
        std::string eid;
        if (!ReadIndexedEid(devicePath, index, value)) {
            continue;
        }
        if (!NormalizeEid(value, eid)) {
            continue;
        }
        eids.push_back({std::move(eid)});
    }
    if (eids.empty()) {
        return Fail(options, logicalDeviceId, error, "host_eid_sysfs", BM_ERROR, 0,
                    "no valid EIDs found under " + devicePath);
    }
    Info(options, logicalDeviceId, "host_eid_sysfs",
         "read UDMA=" + udma + " valid_eid_count=" + std::to_string(eids.size()));
    return true;
}

bool FindHostEid(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, const std::vector<IndexedEid> &eids,
                 const std::string &udma, std::string &hostEid, UrmaEidQueryError &error)
{
    std::set<std::string> candidates;
    for (const IndexedEid &item : eids) {
        EidByteInfo info;
        if (ParseEidByte(item.eid, info) && info.isPg) {
            candidates.insert(item.eid);
        }
    }
    if (candidates.empty()) {
        return Fail(options, logicalDeviceId, error, "host_eid_select", BM_ERROR, 0,
                    "no CPU PG EID found for UDMA " + udma);
    }
    if (candidates.size() > 1U) {
        return Fail(options, logicalDeviceId, error, "host_eid_select", BM_ERROR, 0,
                    "multiple CPU PG EIDs found for UDMA " + udma +
                        ", candidate_count=" + std::to_string(candidates.size()));
    }
    hostEid = *candidates.begin();
    return true;
}

bool FindDeviceEid(const UrmaEidQueryOptions &options, int32_t logicalDeviceId, const std::vector<UrmaDevice> &devices,
                   UrmaEidTopology topology, int32_t meshDieId, std::string &deviceEid, UrmaEidQueryError &error)
{
    if (topology == UrmaEidTopology::AUTO || (meshDieId != 0 && meshDieId != 1)) {
        return Fail(options, logicalDeviceId, error, "device_eid_select", BM_ERROR, 0, "invalid topology or mesh die");
    }
    const int32_t nonMeshDie = 1 - meshDieId;
    std::vector<std::string> candidates;
    for (const UrmaDevice &device : devices) {
        if (topology == UrmaEidTopology::SERVER && device.eids.size() == 1U) {
            EidByteInfo info;
            if (ParseEidByte(device.eids.front(), info) && !info.isPg &&
                static_cast<int32_t>(info.dieId) == nonMeshDie) {
                candidates.push_back(device.eids.front());
            }
        }
        if (topology == UrmaEidTopology::SUPER_POD && device.eids.size() == 3U) {
            for (const std::string &eid : device.eids) {
                EidByteInfo info;
                if (ParseEidByte(eid, info) && info.isPg && static_cast<int32_t>(info.dieId) == nonMeshDie) {
                    candidates.push_back(eid);
                }
            }
        }
    }
    if (candidates.empty()) {
        return Fail(options, logicalDeviceId, error, "device_eid_select", BM_ERROR, 0,
                    "no Device EID matched topology=" + std::string(TopologyName(topology)));
    }
    if (candidates.size() > 1U) {
        return Fail(options, logicalDeviceId, error, "device_eid_select", BM_ERROR, 0,
                    "multiple Device EID candidates for topology=" + std::string(TopologyName(topology)) +
                        ", candidate_count=" + std::to_string(candidates.size()));
    }
    deviceEid = candidates.front();
    return true;
}

bool ResolveTopology(const UrmaEidQueryOptions &options, DcmiApi &dcmi, uint32_t logicalDeviceId,
                     UrmaEidQueryResult &result, UrmaEidQueryError &error)
{
    if (options.topology == UrmaEidTopology::SERVER || options.topology == UrmaEidTopology::SUPER_POD) {
        result.topology = options.topology;
        result.meshDieId = MeshDieId(options.physicalDeviceId, result.topology);
        return true;
    }
    if (options.topology != UrmaEidTopology::AUTO) {
        return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "topology", BM_INVALID_PARAM, 0,
                    "invalid topology value");
    }
    uint32_t mainboardId = 0;
    if (!dcmi.GetMainboard(options, logicalDeviceId, mainboardId, error)) {
        return false;
    }
    if (IsSuperPodMainboard(mainboardId)) {
        result.topology = UrmaEidTopology::SUPER_POD;
    } else if (IsServerMainboard(mainboardId)) {
        result.topology = UrmaEidTopology::SERVER;
    } else {
        std::ostringstream value;
        value << std::hex << mainboardId;
        return Fail(options, static_cast<int32_t>(logicalDeviceId), error, "topology", BM_ERROR, 0,
                    "unsupported mainboard id=0x" + value.str());
    }
    result.meshDieId = MeshDieId(options.physicalDeviceId, result.topology);
    return true;
}

bool Discover(const UrmaEidQueryOptions &options, UrmaEidQueryResult &result, UrmaEidQueryError &error)
{
    DcmiApi dcmi;
    if (!dcmi.Load(options, error) || !dcmi.Initialize(options, error)) {
        return false;
    }
    if (!dcmi.MapPhysicalToLogical(options, result.logicalDeviceId, error) ||
        !ResolveTopology(options, dcmi, result.logicalDeviceId, result, error)) {
        return false;
    }
    DsmiApi dsmi;
    if (!dsmi.Load(options, static_cast<int32_t>(result.logicalDeviceId), error) ||
        !dsmi.GetUdmaName(options, result.logicalDeviceId, result.udma, error)) {
        return false;
    }
    std::vector<IndexedEid> hostEids;
    std::vector<UrmaDevice> devices;
    if (!ReadHostEids(options, static_cast<int32_t>(result.logicalDeviceId), result.udma, hostEids, error) ||
        !dcmi.GetDevices(options, result.logicalDeviceId, devices, error)) {
        return false;
    }
    return FindHostEid(options, static_cast<int32_t>(result.logicalDeviceId), hostEids, result.udma, result.hostEid,
                       error) &&
           FindDeviceEid(options, static_cast<int32_t>(result.logicalDeviceId), devices, result.topology,
                         result.meshDieId, result.deviceEid, error);
}

std::mutex &QueryMutex()
{
    static std::mutex mutex;
    return mutex;
}

Result QueryUrmaEidPairImpl(const UrmaEidQueryOptions &options, UrmaEidPair &eidPair, UrmaEidQueryError &error)
{
    if (options.topology != UrmaEidTopology::AUTO && options.topology != UrmaEidTopology::SERVER &&
        options.topology != UrmaEidTopology::SUPER_POD) {
        Fail(options, -1, error, "argument_validation", BM_INVALID_PARAM, 0, "invalid topology value");
        return error.result;
    }
    std::lock_guard<std::mutex> lock(QueryMutex());
    try {
        UrmaEidQueryResult localResult;
        if (!Discover(options, localResult, error)) {
            return error.result;
        }
        UrmaEidPair localPair{std::move(localResult.hostEid), std::move(localResult.deviceEid)};
        eidPair = std::move(localPair);
        error = {};
        return BM_OK;
    } catch (const std::exception &exception) {
        Fail(options, -1, error, "internal", BM_ERROR, 0, exception.what());
        return error.result;
    } catch (...) {
        Fail(options, -1, error, "internal", BM_ERROR, 0, "unknown exception");
        return error.result;
    }
}

} // namespace

Result QueryUrmaEidPair(uint32_t physicalDeviceId, UrmaEidPair &eidPair, UrmaEidTopology topology)
{
    UrmaEidQueryOptions options;
    options.physicalDeviceId = physicalDeviceId;
    options.topology = topology;
    UrmaEidQueryError error;
    return QueryUrmaEidPairImpl(options, eidPair, error);
}

} // namespace mf
} // namespace ock
