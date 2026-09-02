/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#ifndef MEM_FABRIC_HYBRID_HYBM_AGGREGATE_URMA_DEMO_H
#define MEM_FABRIC_HYBRID_HYBM_AGGREGATE_URMA_DEMO_H

#include <cstddef>
#include <cstdint>

struct alignas(64) HybmAggregateUrmaDemoRequest {
    uint64_t hostMailboxGva;
    uint64_t dstNewGva;
    uint64_t readyGva;
    uint64_t totalBytes;
    uint64_t srcStride;
    uint64_t dstStride;
    uint32_t segmentCount;
    uint32_t segmentBytes;
    uint64_t reserved;
};

struct alignas(64) HybmAggregateUrmaDemoMessage {
    HybmAggregateUrmaDemoRequest request;
    uint64_t doorbell;
    uint8_t padding[56];
};

struct alignas(64) HybmAggregateUrmaDemoTiming {
    uint64_t requestNs;
    uint64_t waitHostNs;
    uint64_t scatterCopyNs;
    uint64_t scatterPublishNs;
    uint64_t scatterNs;
    uint64_t totalNs;
    uint8_t padding[16];
};

struct HybmAggregateUrmaDemoParam {
    const HybmAggregateUrmaDemoMessage *message;
    volatile uint64_t *ready;
    uint8_t *dstNew;
    uint8_t *dstBase;
    HybmAggregateUrmaDemoTiming *timing;
};

static_assert(sizeof(HybmAggregateUrmaDemoRequest) == 64U);
static_assert(offsetof(HybmAggregateUrmaDemoMessage, doorbell) == 64U);
static_assert(sizeof(HybmAggregateUrmaDemoMessage) == 128U);
static_assert(sizeof(HybmAggregateUrmaDemoTiming) == 64U);

extern "C" uint32_t HybmAggregateUrmaDemo(HybmAggregateUrmaDemoParam *param);

#endif // MEM_FABRIC_HYBRID_HYBM_AGGREGATE_URMA_DEMO_H
