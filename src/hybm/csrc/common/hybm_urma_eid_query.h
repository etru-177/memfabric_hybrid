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

#ifndef MEM_FABRIC_HYBRID_HYBM_URMA_EID_QUERY_H
#define MEM_FABRIC_HYBRID_HYBM_URMA_EID_QUERY_H

#include <cstdint>
#include <string>

#include "hybm_types.h"

namespace ock {
namespace mf {

enum class UrmaEidTopology : uint8_t {
    AUTO,
    SERVER,
    SUPER_POD,
};

struct UrmaEidPair {
    std::string hostEid;
    std::string deviceEid;
};

/**
 * Query the Host and Device URMA EIDs associated with an A5 physical NPU.
 *
 * The output is updated only after all discovery and selection steps succeed.
 */
Result QueryUrmaEidPair(uint32_t physicalDeviceId, UrmaEidPair &eidPair,
                        UrmaEidTopology topology = UrmaEidTopology::AUTO);

} // namespace mf
} // namespace ock

#endif // MEM_FABRIC_HYBRID_HYBM_URMA_EID_QUERY_H
