/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * Temporary local DRAM validation helper. Remove this file after hardware sign-off.
 * Linux build:
 *   g++ -std=c++17 -O2 -Wall -Wextra -Werror examples/kv_offload/sparse_copy_urma/urma_eid_query.cpp \
 *       -ldl -o /tmp/mf_urma_eid_query
 */

#if !defined(__linux__)

#include <cstdio>

int main(int, char **)
{
    std::fprintf(stderr, "[ERROR] stage=platform ret=3 detail=the EID tool requires Linux\n");
    return 3;
}

#else

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kDsmiMainCommandUb = 62U;
constexpr uint32_t kDsmiUrmaNameSubCommand = 3U;
constexpr size_t kMaxUdmaName = 32U;
constexpr size_t kMaxEidsPerDevice = 32U;
constexpr size_t kMaxDevices = 128U;
constexpr size_t kEidHexLength = 32U;
constexpr size_t kUrmaOutputLimit = 4U * 1024U * 1024U;
constexpr size_t kDcmiCpuListBufferSize = 4096U;
constexpr uint32_t kPodNpuGroupSize = 8U;
constexpr uint32_t kPodFirstMeshDieLastNpu = 3U;

constexpr uint32_t kPodMainboard1 = 0x3U;
constexpr uint32_t kPodMainboard2 = 0x5U;
constexpr uint32_t kPodMainboard3 = 0x7U;
constexpr uint32_t kServerMainboardMin1 = 0x21U;
constexpr uint32_t kServerMainboardMax1 = 0x2BU;
constexpr uint32_t kServerMainboardMin2 = 0x40U;
constexpr uint32_t kServerMainboardMax2 = 0x46U;

enum class Topology {
    kAuto,
    kServer,
    kPod,
};

enum class OutputFormat {
    kText,
    kJson,
    kEnv,
};

struct Options {
    uint32_t physical_device_id = 0;
    Topology topology = Topology::kAuto;
    OutputFormat format = OutputFormat::kText;
    std::string dcmi_library = "libdcmi.so";
    std::string dsmi_library = "libdrvdsmi_host.so";
    std::string urma_admin_path;
    bool dump_candidates = true;
};

struct Failure {
    int code = 5;
    std::string message;
};

struct EidByteInfo {
    uint8_t die_id = 0;
    bool is_pg = false;
    int32_t port = -1;
};

struct UrmaRow {
    std::string udma;
    int32_t eid_index = -1;
    std::string eid;
};

struct UrmaDevice {
    std::vector<std::string> eids;
};

struct EidPair {
    uint32_t physical_device_id = 0;
    uint32_t logical_device_id = 0;
    Topology topology = Topology::kAuto;
    int32_t mesh_die_id = -1;
    std::string udma;
    std::string host_eid;
    std::string device_eid;
    std::string affinity_cpus = "unavailable";
};

void Log(const char *level, const Options &options, int32_t logical_device_id, const char *stage,
         int ret, const std::string &detail)
{
    std::fprintf(stderr,
                 "[EID][%s] stage=%s physical_device_id=%u logical_device_id=%d ret=%d detail=%s\n",
                 level, stage, options.physical_device_id, logical_device_id, ret, detail.c_str());
}

bool Fail(const Options &options, int32_t logical_device_id, Failure &failure, const char *stage, int code,
          int ret, const std::string &detail)
{
    failure.code = code;
    failure.message = detail;
    Log("ERROR", options, logical_device_id, stage, ret, detail);
    return false;
}

void Info(const Options &options, int32_t logical_device_id, const char *stage, const std::string &detail)
{
    Log("INFO", options, logical_device_id, stage, 0, detail);
}

void Warn(const Options &options, int32_t logical_device_id, const char *stage, const std::string &detail)
{
    Log("WARN", options, logical_device_id, stage, 0, detail);
}

bool StartsWithIgnoreCase(const std::string &value, const char *prefix)
{
    const size_t prefix_size = std::strlen(prefix);
    if (value.size() < prefix_size) {
        return false;
    }
    for (size_t index = 0; index < prefix_size; ++index) {
        if (std::tolower(static_cast<unsigned char>(value[index])) !=
            std::tolower(static_cast<unsigned char>(prefix[index]))) {
            return false;
        }
    }
    return true;
}

bool NormalizeEid(const std::string &input, std::string &output)
{
    output.clear();
    output.reserve(kEidHexLength);
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
    if (output.size() != kEidHexLength ||
        output == std::string(kEidHexLength, '0')) {
        output.clear();
        return false;
    }
    return true;
}

uint8_t HexValue(char character)
{
    if (character >= '0' && character <= '9') {
        return static_cast<uint8_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
        return static_cast<uint8_t>(character - 'a' + 10);
    }
    return static_cast<uint8_t>(character - 'A' + 10);
}

bool ParseEidByte(const std::string &eid, EidByteInfo &info)
{
    std::string normalized;
    if (!NormalizeEid(eid, normalized)) {
        return false;
    }
    const uint8_t byte6 = static_cast<uint8_t>((HexValue(normalized[10]) << 4U) | HexValue(normalized[11]));
    const uint8_t high_nibble = static_cast<uint8_t>(byte6 >> 4U);
    info.die_id = (high_nibble & 0x4U) != 0U ? 1U : 0U;
    info.is_pg = high_nibble == 0x3U || high_nibble == 0x7U;
    info.port = static_cast<int32_t>(byte6 & 0x0FU);
    return true;
}

std::string TopologyName(Topology topology)
{
    if (topology == Topology::kServer) {
        return "server";
    }
    if (topology == Topology::kPod) {
        return "pod";
    }
    return "auto";
}

int32_t MeshDieId(uint32_t physical_device_id, Topology topology)
{
    if (topology == Topology::kServer) {
        return 1;
    }
    return physical_device_id % kPodNpuGroupSize <= kPodFirstMeshDieLastNpu ? 0 : 1;
}

bool IsServerMainboard(uint32_t mainboard_id)
{
    const bool first_range = mainboard_id >= kServerMainboardMin1 &&
                             mainboard_id <= kServerMainboardMax1 &&
                             (mainboard_id % 2U == 1U);
    const bool second_range = mainboard_id >= kServerMainboardMin2 &&
                              mainboard_id <= kServerMainboardMax2 &&
                              (mainboard_id % 2U == 0U);
    return first_range || second_range;
}

bool IsPodMainboard(uint32_t mainboard_id)
{
    return mainboard_id == kPodMainboard1 || mainboard_id == kPodMainboard2 ||
           mainboard_id == kPodMainboard3;
}

union DcmiUrmaEid {
    unsigned char raw[16];
    struct {
        unsigned long subnet_prefix;
        unsigned long interface_id;
    } in6;
};

struct DcmiUrmaEidInfo {
    DcmiUrmaEid eid;
    unsigned int eid_index;
};

#if defined(__linux__)
static_assert(sizeof(unsigned long) == 8U, "DCMI ABI requires a 64-bit Linux host");
static_assert(sizeof(DcmiUrmaEid) == 16U, "unexpected DCMI EID ABI size");
static_assert(sizeof(DcmiUrmaEidInfo) == 24U, "unexpected DCMI EID info ABI size");
#endif

std::string DynamicLoaderError()
{
    const char *message = dlerror();
    return message == nullptr ? "unknown dynamic-loader error" : message;
}

std::string FormatRawEid(const DcmiUrmaEid &eid)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t index = 0; index < 16U; ++index) {
        stream << std::setw(2) << static_cast<unsigned int>(eid.raw[index]);
    }
    return stream.str();
}

class DcmiApi {
public:
    using InitFn = int32_t (*)();
    using GetCountFn = int32_t (*)(int32_t, uint32_t *);
    using GetEidListFn = int32_t (*)(int32_t, int32_t, DcmiUrmaEidInfo *, int32_t *);
    using GetMainboardFn = int32_t (*)(int32_t, uint32_t *);
    using GetLogicalFn = int32_t (*)(uint32_t, uint32_t *);
    using GetAffinityCpuInfoFn = int32_t (*)(int32_t, char *, int32_t *);

    ~DcmiApi()
    {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    bool Load(const Options &options, int32_t logical_device_id, Failure &failure)
    {
        const std::string path = options.dcmi_library.empty() ? "libdcmi.so" : options.dcmi_library;
        handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return Fail(options, logical_device_id, failure, "dcmi_load", 3, 0,
                        "dlopen " + path + " failed: " + DynamicLoaderError());
        }
        init_ = LoadSymbol<InitFn>("dcmiv2_init");
        get_count_ = LoadSymbol<GetCountFn>("dcmiv2_get_urma_device_cnt");
        get_eid_list_ = LoadSymbol<GetEidListFn>("dcmiv2_get_eid_list_by_urma_dev_index");
        get_logical_ = LoadSymbol<GetLogicalFn>("dcmiv2_get_dev_id_by_chip_phy_id");
        if (get_logical_ == nullptr) {
            get_logical_ = LoadSymbol<GetLogicalFn>("dcmiv2_get_dev_id_from_chip_phyid");
        }
        get_mainboard_ = LoadSymbol<GetMainboardFn>("dcmiv2_get_mainboard_id");
        if (init_ == nullptr || get_count_ == nullptr || get_eid_list_ == nullptr ||
            get_logical_ == nullptr) {
            return Fail(options, logical_device_id, failure, "dcmi_symbol", 3, 0,
                        "required DCMI symbol is missing from " + path + ": " + DynamicLoaderError());
        }
        get_affinity_cpu_info_ = LoadSymbol<GetAffinityCpuInfoFn>(
            "dcmiv2_get_affinity_cpu_info_by_dev_id");
        if (get_affinity_cpu_info_ == nullptr) {
            (void)dlerror();
        }
        Info(options, logical_device_id, "dcmi_load", "loaded " + path);
        return true;
    }

    bool Initialize(const Options &options, int32_t logical_device_id, Failure &failure) const
    {
        int32_t ret = -1;
        for (int attempt = 0; attempt < 10; ++attempt) {
            ret = init_();
            if (ret == 0) {
                return true;
            }
            sleep(1);
        }
        return Fail(options, logical_device_id, failure, "dcmi_init", 3, ret,
                    "dcmiv2_init failed after 10 attempts");
    }

    bool MapPhysicalToLogical(const Options &options, uint32_t physical_device_id,
                              uint32_t &logical_device_id, Failure &failure) const
    {
        const int32_t ret = get_logical_(physical_device_id, &logical_device_id);
        if (ret != 0) {
            return Fail(options, -1, failure, "physical_to_logical", 4, ret,
                        "DCMI physical-to-logical mapping failed");
        }
        return true;
    }

    void GetAffinityCpuList(const Options &options, uint32_t logical_device_id,
                            std::string &affinity_cpus) const
    {
        affinity_cpus = "unavailable";
        if (get_affinity_cpu_info_ == nullptr) {
            Warn(options, static_cast<int32_t>(logical_device_id), "dcmi_affinity",
                 "dcmiv2_get_affinity_cpu_info_by_dev_id is unavailable");
            return;
        }
        char buffer[kDcmiCpuListBufferSize] = {};
        int32_t length = static_cast<int32_t>(sizeof(buffer));
        const int32_t ret = get_affinity_cpu_info_(static_cast<int32_t>(logical_device_id), buffer, &length);
        if (ret != 0) {
            Warn(options, static_cast<int32_t>(logical_device_id), "dcmi_affinity",
                 "DCMI affinity CPU query failed, ret=" + std::to_string(ret));
            return;
        }
        const size_t value_length = strnlen(buffer, sizeof(buffer));
        if (value_length == 0U) {
            Warn(options, static_cast<int32_t>(logical_device_id), "dcmi_affinity",
                 "DCMI affinity CPU query returned an empty cpulist");
            return;
        }
        affinity_cpus.assign(buffer, value_length);
    }

    bool GetMainboard(const Options &options, uint32_t logical_device_id,
                      uint32_t &mainboard_id, Failure &failure) const
    {
        if (get_mainboard_ == nullptr) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "topology", 4, 0,
                        "mainboard API is unavailable; pass --topology=server or --topology=pod");
        }
        const int32_t ret = get_mainboard_(static_cast<int32_t>(logical_device_id), &mainboard_id);
        if (ret != 0) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "topology", 4, ret,
                        "DCMI mainboard query failed");
        }
        return true;
    }

    bool GetDevices(const Options &options, uint32_t logical_device_id,
                    std::vector<UrmaDevice> &devices, Failure &failure) const
    {
        uint32_t device_count = 0;
        const int32_t count_ret = get_count_(static_cast<int32_t>(logical_device_id), &device_count);
        if (count_ret != 0 || device_count == 0U || device_count > kMaxDevices) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "dcmi_eid_list", 4,
                        count_ret, "invalid URMA device count=" + std::to_string(device_count));
        }
        devices.clear();
        for (uint32_t device_index = 0; device_index < device_count; ++device_index) {
            AppendDevice(options, logical_device_id, device_index, devices);
        }
        if (devices.empty()) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "dcmi_eid_list", 4, 0,
                        "DCMI returned no non-zero URMA EIDs");
        }
        return true;
    }

private:
    template <typename Function>
    Function LoadSymbol(const char *name) const
    {
        return reinterpret_cast<Function>(dlsym(handle_, name));
    }

    void AppendDevice(const Options &options, uint32_t logical_device_id, uint32_t device_index,
                      std::vector<UrmaDevice> &devices) const
    {
        DcmiUrmaEidInfo eid_buffer[kMaxEidsPerDevice] = {};
        int32_t eid_count = static_cast<int32_t>(kMaxEidsPerDevice);
        const int32_t ret = get_eid_list_(static_cast<int32_t>(logical_device_id),
                                          static_cast<int32_t>(device_index), eid_buffer, &eid_count);
        if (ret != 0 || eid_count < 0 || eid_count > static_cast<int32_t>(kMaxEidsPerDevice)) {
            Warn(options, static_cast<int32_t>(logical_device_id), "dcmi_eid_list",
                 "skip invalid URMA device index=" + std::to_string(device_index));
            return;
        }
        UrmaDevice device;
        for (int32_t eid_index = 0; eid_index < eid_count; ++eid_index) {
            std::string eid;
            if (NormalizeEid(FormatRawEid(eid_buffer[eid_index].eid), eid)) {
                device.eids.push_back(eid);
            }
        }
        if (!device.eids.empty()) {
            devices.push_back(std::move(device));
        }
    }

    void *handle_ = nullptr;
    InitFn init_ = nullptr;
    GetCountFn get_count_ = nullptr;
    GetEidListFn get_eid_list_ = nullptr;
    GetMainboardFn get_mainboard_ = nullptr;
    GetLogicalFn get_logical_ = nullptr;
    GetAffinityCpuInfoFn get_affinity_cpu_info_ = nullptr;
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

    bool Load(const Options &options, int32_t logical_device_id, Failure &failure)
    {
        const std::string path = options.dsmi_library.empty() ? "libdrvdsmi_host.so" : options.dsmi_library;
        handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return Fail(options, logical_device_id, failure, "dsmi_load", 3, 0,
                        "dlopen " + path + " failed: " + DynamicLoaderError());
        }
        get_device_info_ = reinterpret_cast<GetDeviceInfoFn>(dlsym(handle_, "dsmi_get_device_info"));
        if (get_device_info_ == nullptr) {
            return Fail(options, logical_device_id, failure, "dsmi_symbol", 3, 0,
                        "dsmi_get_device_info is missing from " + path);
        }
        Info(options, logical_device_id, "dsmi_load", "loaded " + path);
        return true;
    }

    bool GetUdmaName(const Options &options, uint32_t logical_device_id,
                     std::string &name, Failure &failure) const
    {
        char buffer[kMaxUdmaName] = {};
        uint32_t buffer_size = static_cast<uint32_t>(sizeof(buffer));
        const int ret = get_device_info_(logical_device_id, kDsmiMainCommandUb,
                                         kDsmiUrmaNameSubCommand, buffer, &buffer_size);
        if (ret != 0) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "dsmi_udma", 4, ret,
                        "dsmi_get_device_info(UB, URMA_DEV_NAME) failed");
        }
        name.assign(buffer, strnlen(buffer, sizeof(buffer)));
        if (name.empty()) {
            return Fail(options, static_cast<int32_t>(logical_device_id), failure, "dsmi_udma", 4, 0,
                        "DSMI returned an empty UDMA name");
        }
        return true;
    }

private:
    void *handle_ = nullptr;
    GetDeviceInfoFn get_device_info_ = nullptr;
};

std::string FindUrmaAdminPath(const Options &options)
{
    if (!options.urma_admin_path.empty()) {
        return options.urma_admin_path;
    }
    constexpr const char *default_path = "/usr/local/sbin/urma_admin";
    if (access(default_path, X_OK) == 0) {
        return default_path;
    }
    const char *path_value = std::getenv("PATH");
    if (path_value != nullptr) {
        std::stringstream paths(path_value);
        std::string directory;
        while (std::getline(paths, directory, ':')) {
            if (directory.empty()) {
                directory = ".";
            }
            const std::string candidate = directory + "/urma_admin";
            if (access(candidate.c_str(), X_OK) == 0) {
                return candidate;
            }
        }
    }
    return "urma_admin";
}

bool ReadOutput(int fd, std::string &output, const Options &options,
                int32_t logical_device_id, Failure &failure)
{
    char buffer[4096];
    while (true) {
        const ssize_t bytes = read(fd, buffer, sizeof(buffer));
        if (bytes == 0) {
            return true;
        }
        if (bytes < 0 && errno == EINTR) {
            continue;
        }
        if (bytes < 0) {
            return Fail(options, logical_device_id, failure, "urma_admin_read", 3, errno,
                        "read urma_admin output failed");
        }
        if (output.size() + static_cast<size_t>(bytes) > kUrmaOutputLimit) {
            return Fail(options, logical_device_id, failure, "urma_admin_read", 3, 0,
                        "urma_admin output exceeds 4 MiB");
        }
        output.append(buffer, static_cast<size_t>(bytes));
    }
}

bool RunUrmaAdmin(const Options &options, int32_t logical_device_id,
                  std::string &output, Failure &failure)
{
    const std::string path = FindUrmaAdminPath(options);
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
        return Fail(options, logical_device_id, failure, "urma_admin_pipe", 3, errno,
                    "pipe creation failed");
    }
    const pid_t child = fork();
    if (child < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return Fail(options, logical_device_id, failure, "urma_admin_fork", 3, errno,
                    "fork failed");
    }
    if (child == 0) {
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        char arg0[] = "urma_admin";
        char arg1[] = "show";
        char *args[] = {arg0, arg1, nullptr};
        if (path.find('/') != std::string::npos) {
            execv(path.c_str(), args);
        } else {
            execvp(path.c_str(), args);
        }
        _exit(127);
    }
    close(pipe_fds[1]);
    output.clear();
    const bool read_ok = ReadOutput(pipe_fds[0], output, options, logical_device_id, failure);
    close(pipe_fds[0]);
    int status = 0;
    const bool wait_ok = waitpid(child, &status, 0) == child;
    if (!read_ok) {
        return false;
    }
    if (!wait_ok) {
        return Fail(options, logical_device_id, failure, "urma_admin_wait", 3, errno, "waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return Fail(options, logical_device_id, failure, "urma_admin_exec", 3,
                    WIFEXITED(status) ? WEXITSTATUS(status) : -1,
                    "urma_admin show failed, path=" + path);
    }
    Info(options, logical_device_id, "urma_admin_exec",
         "completed path=" + path + " output_bytes=" + std::to_string(output.size()));
    return true;
}

bool ParseNonNegative(const std::string &value, int32_t &result)
{
    if (value.empty()) {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    result = static_cast<int32_t>(parsed);
    return true;
}

bool ParseUrmaRows(const Options &options, int32_t logical_device_id,
                   const std::string &output, std::vector<UrmaRow> &rows, Failure &failure)
{
    rows.clear();
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream token_stream(line);
        std::vector<std::string> tokens;
        std::string token;
        while (token_stream >> token) {
            tokens.push_back(token);
        }
        std::string udma;
        for (const std::string &candidate : tokens) {
            if (StartsWithIgnoreCase(candidate, "udma") && candidate.size() > 4U) {
                udma = candidate;
                break;
            }
        }
        if (udma.empty()) {
            continue;
        }
        for (size_t index = 0; index + 1U < tokens.size(); ++index) {
            if (!StartsWithIgnoreCase(tokens[index], "eid") ||
                tokens[index].size() <= 3U) {
                continue;
            }
            int32_t eid_index = -1;
            std::string eid;
            if (!ParseNonNegative(tokens[index].substr(3U), eid_index) ||
                !NormalizeEid(tokens[index + 1U], eid)) {
                continue;
            }
            rows.push_back({udma, eid_index, eid});
            break;
        }
    }
    if (rows.empty()) {
        return Fail(options, logical_device_id, failure, "urma_admin_parse", 4, 0,
                    "no valid UDMA/EID rows found");
    }
    return true;
}

bool FindHostEid(const Options &options, int32_t logical_device_id,
                 const std::vector<UrmaRow> &rows, const std::string &udma,
                 std::string &host_eid, Failure &failure)
{
    std::vector<std::string> candidates;
    for (const UrmaRow &row : rows) {
        if (row.udma != udma) {
            continue;
        }
        EidByteInfo info;
        if (ParseEidByte(row.eid, info) && info.is_pg) {
            candidates.push_back(row.eid);
        }
    }
    if (candidates.empty()) {
        return Fail(options, logical_device_id, failure, "host_eid_select", 4, 0,
                    "no CPU PG EID found for UDMA " + udma);
    }
    if (candidates.size() > 1U) {
        return Fail(options, logical_device_id, failure, "host_eid_select", 4, 0,
                    "multiple CPU PG EIDs found for UDMA " + udma +
                        ", candidate_count=" + std::to_string(candidates.size()));
    }
    host_eid = candidates.front();
    return true;
}

bool FindDeviceEid(const Options &options, int32_t logical_device_id,
                   const std::vector<UrmaDevice> &devices, Topology topology,
                   int32_t mesh_die_id, std::string &device_eid, Failure &failure)
{
    if (topology == Topology::kAuto || (mesh_die_id != 0 && mesh_die_id != 1)) {
        return Fail(options, logical_device_id, failure, "device_eid_select", 4, 0,
                    "invalid topology or mesh die");
    }
    const int32_t non_mesh_die = 1 - mesh_die_id;
    std::vector<std::string> candidates;
    for (const UrmaDevice &device : devices) {
        if (topology == Topology::kServer && device.eids.size() == 1U) {
            EidByteInfo info;
            if (ParseEidByte(device.eids.front(), info) && !info.is_pg &&
                static_cast<int32_t>(info.die_id) == non_mesh_die) {
                candidates.push_back(device.eids.front());
            }
        }
        if (topology == Topology::kPod && device.eids.size() == 3U) {
            for (const std::string &eid : device.eids) {
                EidByteInfo info;
                if (ParseEidByte(eid, info) && info.is_pg &&
                    static_cast<int32_t>(info.die_id) == non_mesh_die) {
                    candidates.push_back(eid);
                    break;
                }
            }
        }
    }
    if (candidates.empty()) {
        return Fail(options, logical_device_id, failure, "device_eid_select", 4, 0,
                    "no Device EID matched topology=" + TopologyName(topology));
    }
    if (candidates.size() > 1U) {
        return Fail(options, logical_device_id, failure, "device_eid_select", 4, 0,
                    "multiple Device EID candidates for topology=" + TopologyName(topology) +
                        ", candidate_count=" + std::to_string(candidates.size()));
    }
    device_eid = candidates.front();
    return true;
}

bool ResolveTopology(const Options &options, DcmiApi &dcmi, uint32_t logical_device_id,
                     EidPair &pair, Failure &failure)
{
    if (options.topology != Topology::kAuto) {
        pair.topology = options.topology;
        pair.mesh_die_id = MeshDieId(options.physical_device_id, pair.topology);
        return true;
    }
    uint32_t mainboard_id = 0;
    if (!dcmi.GetMainboard(options, logical_device_id, mainboard_id, failure)) {
        return false;
    }
    if (IsPodMainboard(mainboard_id)) {
        pair.topology = Topology::kPod;
    } else if (IsServerMainboard(mainboard_id)) {
        pair.topology = Topology::kServer;
    } else {
        std::ostringstream value;
        value << std::hex << mainboard_id;
        return Fail(options, static_cast<int32_t>(logical_device_id), failure, "topology", 4, 0,
                    "unsupported mainboard id=0x" + value.str());
    }
    pair.mesh_die_id = MeshDieId(options.physical_device_id, pair.topology);
    return true;
}

void DumpCandidates(const Options &options, int32_t logical_device_id,
                    const std::vector<UrmaRow> &rows, const std::vector<UrmaDevice> &devices)
{
    for (const UrmaRow &row : rows) {
        EidByteInfo info;
        const bool parsed = ParseEidByte(row.eid, info);
        const std::string detail = "CPU candidate udma=" + row.udma +
                                   " eid_index=" + std::to_string(row.eid_index) +
                                   " eid=" + row.eid +
                                   (parsed ? " pg=" + std::to_string(info.is_pg) : " parse=failed");
        Info(options, logical_device_id, "candidate", detail);
    }
    for (size_t device_index = 0; device_index < devices.size(); ++device_index) {
        Info(options, logical_device_id, "candidate",
             "Device candidate index=" + std::to_string(device_index) +
             " eid_count=" + std::to_string(devices[device_index].eids.size()));
    }
}

bool Discover(const Options &options, EidPair &pair, Failure &failure)
{
    pair.physical_device_id = options.physical_device_id;
    DcmiApi dcmi;
    if (!dcmi.Load(options, -1, failure) || !dcmi.Initialize(options, -1, failure)) {
        return false;
    }
    if (!dcmi.MapPhysicalToLogical(options, options.physical_device_id, pair.logical_device_id, failure)) {
        return false;
    }
    dcmi.GetAffinityCpuList(options, pair.logical_device_id, pair.affinity_cpus);
    if (!ResolveTopology(options, dcmi, pair.logical_device_id, pair, failure)) {
        return false;
    }
    DsmiApi dsmi;
    if (!dsmi.Load(options, static_cast<int32_t>(pair.logical_device_id), failure) ||
        !dsmi.GetUdmaName(options, pair.logical_device_id, pair.udma, failure)) {
        return false;
    }
    std::string admin_output;
    if (!RunUrmaAdmin(options, static_cast<int32_t>(pair.logical_device_id), admin_output, failure)) {
        return false;
    }
    std::vector<UrmaRow> rows;
    if (!ParseUrmaRows(options, static_cast<int32_t>(pair.logical_device_id),
                       admin_output, rows, failure)) {
        return false;
    }
    std::vector<UrmaDevice> devices;
    if (!dcmi.GetDevices(options, pair.logical_device_id, devices, failure)) {
        return false;
    }
    if (options.dump_candidates) {
        DumpCandidates(options, static_cast<int32_t>(pair.logical_device_id), rows, devices);
    }
    if (!FindHostEid(options, static_cast<int32_t>(pair.logical_device_id), rows,
                     pair.udma, pair.host_eid, failure) ||
        !FindDeviceEid(options, static_cast<int32_t>(pair.logical_device_id), devices,
                       pair.topology, pair.mesh_die_id, pair.device_eid, failure)) {
        return false;
    }
    return true;
}

std::string JsonEscape(const std::string &value)
{
    std::string escaped;
    for (const unsigned char character : value) {
        if (character == '\\' || character == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(static_cast<char>(character));
    }
    return escaped;
}

std::string ShellQuote(const std::string &value)
{
    std::string quoted("'");
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(character);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

void PrintPair(const EidPair &pair, OutputFormat format)
{
    if (format == OutputFormat::kEnv) {
        std::printf("export MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID=%u\n", pair.physical_device_id);
        std::printf("export MF_LOCAL_DRAM_LOGICAL_DEVICE_ID=%u\n", pair.logical_device_id);
        std::printf("export MF_LOCAL_DRAM_TOPOLOGY=%s\n", ShellQuote(TopologyName(pair.topology)).c_str());
        std::printf("export MF_LOCAL_DRAM_UDMA=%s\n", ShellQuote(pair.udma).c_str());
        std::printf("export MF_LOCAL_DRAM_AFFINITY_CPUS=%s\n", ShellQuote(pair.affinity_cpus).c_str());
        std::printf("export MF_HOST_URMA_EID=%s\n", ShellQuote(pair.host_eid).c_str());
        std::printf("export USE_LOCAL_EID=%s\n", ShellQuote(pair.device_eid).c_str());
        return;
    }
    if (format == OutputFormat::kJson) {
        std::printf(
            "{\"schema\":\"mf-local-dram-eid/v1\",\"physical_device_id\":%u,"
            "\"logical_device_id\":%u,\"topology\":\"%s\",\"mesh_die_id\":%d,"
            "\"udma\":\"%s\",\"affinity_cpus\":\"%s\",\"host_eid\":\"%s\","
            "\"device_eid\":\"%s\"}\n",
            pair.physical_device_id, pair.logical_device_id, TopologyName(pair.topology).c_str(),
            pair.mesh_die_id, JsonEscape(pair.udma).c_str(), JsonEscape(pair.affinity_cpus).c_str(),
            JsonEscape(pair.host_eid).c_str(), JsonEscape(pair.device_eid).c_str());
        return;
    }
    std::printf(
        "[EID] physical=%u logical=%u topology=%s mesh_die_id=%d udma=%s affinity_cpus=%s host_eid=%s "
        "device_eid=%s\n",
        pair.physical_device_id, pair.logical_device_id, TopologyName(pair.topology).c_str(),
        pair.mesh_die_id, pair.udma.c_str(), pair.affinity_cpus.c_str(), pair.host_eid.c_str(),
        pair.device_eid.c_str());
}

bool ParseUint32(const std::string &text, uint32_t &value)
{
    if (text.empty() || text[0] == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool ParseTopology(const std::string &text, Topology &topology)
{
    if (text == "auto") {
        topology = Topology::kAuto;
        return true;
    }
    if (text == "server") {
        topology = Topology::kServer;
        return true;
    }
    if (text == "pod") {
        topology = Topology::kPod;
        return true;
    }
    return false;
}

bool ParseFormat(const std::string &text, OutputFormat &format)
{
    if (text == "text") {
        format = OutputFormat::kText;
        return true;
    }
    if (text == "json") {
        format = OutputFormat::kJson;
        return true;
    }
    if (text == "env") {
        format = OutputFormat::kEnv;
        return true;
    }
    return false;
}

void PrintUsage(const char *program)
{
    std::printf("Usage: %s [options]\n"
                "  --device-id <N>       physical NPU id (default: 0)\n"
                "  --topology <auto|server|pod>\n"
                "  --format <text|json|env> output format (default: text)\n"
                "  --dcmi-lib <PATH>     libdcmi.so path/name\n"
                "  --dsmi-lib <PATH>     libdrvdsmi_host.so path/name\n"
                "  --urma-admin <PATH>   urma_admin executable path\n"
                "  --no-candidates       hide candidate diagnostics\n"
                "  --help                show this message\n",
                program);
}

bool TakeValue(int argc, char **argv, int &index, const char *option, std::string &value)
{
    const std::string argument(argv[index]);
    const std::string prefix = std::string(option) + "=";
    if (argument.compare(0, prefix.size(), prefix) == 0) {
        value = argument.substr(prefix.size());
        return !value.empty();
    }
    if (argument == option && index + 1 < argc) {
        value = argv[++index];
        return !value.empty();
    }
    return false;
}

enum class ParseResult {
    kNotHandled,
    kContinue,
    kHelp,
    kError,
};

ParseResult ParseValueArgument(int argc, char **argv, int &index, Options &options)
{
    std::string value;
    if (TakeValue(argc, argv, index, "--device-id", value)) {
        if (!ParseUint32(value, options.physical_device_id)) {
            Log("ERROR", options, -1, "argument_validation", 2, "invalid --device-id value=" + value);
            return ParseResult::kError;
        }
        return ParseResult::kContinue;
    }
    if (TakeValue(argc, argv, index, "--topology", value)) {
        if (!ParseTopology(value, options.topology)) {
            Log("ERROR", options, -1, "argument_validation", 2, "invalid --topology value=" + value);
            return ParseResult::kError;
        }
        return ParseResult::kContinue;
    }
    if (TakeValue(argc, argv, index, "--format", value)) {
        if (!ParseFormat(value, options.format)) {
            Log("ERROR", options, -1, "argument_validation", 2, "invalid --format value=" + value);
            return ParseResult::kError;
        }
        return ParseResult::kContinue;
    }
    if (TakeValue(argc, argv, index, "--dcmi-lib", value)) {
        options.dcmi_library = value;
        return ParseResult::kContinue;
    }
    if (TakeValue(argc, argv, index, "--dsmi-lib", value)) {
        options.dsmi_library = value;
        return ParseResult::kContinue;
    }
    if (TakeValue(argc, argv, index, "--urma-admin", value)) {
        options.urma_admin_path = value;
        return ParseResult::kContinue;
    }
    return ParseResult::kNotHandled;
}

ParseResult ParseArgument(int argc, char **argv, int &index, Options &options)
{
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
        PrintUsage(argv[0]);
        return ParseResult::kHelp;
    }
    if (argument == "--no-candidates") {
        options.dump_candidates = false;
        return ParseResult::kContinue;
    }
    const ParseResult value_result = ParseValueArgument(argc, argv, index, options);
    if (value_result != ParseResult::kNotHandled) {
        return value_result;
    }
    Log("ERROR", options, -1, "argument_validation", 2, "unknown or incomplete option=" + argument);
    return ParseResult::kError;
}

int ParseArguments(int argc, char **argv, Options &options)
{
    for (int index = 1; index < argc; ++index) {
        const ParseResult result = ParseArgument(argc, argv, index, options);
        if (result == ParseResult::kHelp) {
            return 0;
        }
        if (result == ParseResult::kError) {
            return 2;
        }
    }
    return -1;
}

}  // namespace

int main(int argc, char **argv)
{
    Options options;
    const int parse_ret = ParseArguments(argc, argv, options);
    if (parse_ret >= 0) {
        return parse_ret;
    }
    EidPair pair;
    Failure failure;
    if (!Discover(options, pair, failure)) {
        return failure.code;
    }
    PrintPair(pair, options.format);
    return 0;
}

#endif
