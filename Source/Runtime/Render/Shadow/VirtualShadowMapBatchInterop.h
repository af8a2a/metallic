#pragma once

#ifdef __cplusplus
#include "VirtualShadowMaps.h"
namespace metallic::render {
#else
#define uint32_t uint
#endif

// One existing per-face dispatch/draw, retaining its local workBase and all
// resource bindings. An indirect source supplies its X group count at execution.
struct ShadowBatchJob
{
    ShadowPush push;
    uint32_t countBuffer; // UINT32_MAX uses groupCount directly.
    uint32_t countWord;
    uint32_t groupCount;
    uint32_t reserved;
};

struct VirtualShadowMapBatchPreparePush
{
    uint32_t jobs;
    uint32_t firstJob;
    uint32_t jobCount;
    uint32_t outputs;
    uint32_t outputWord;
    uint32_t groupBase;
    uint32_t mode; // 0: rectangular compute dispatch; 1: mesh multi-draw arguments.
    uint32_t reserved;
};

#ifdef __cplusplus
static_assert(sizeof(ShadowBatchJob) == 240);
static_assert(sizeof(VirtualShadowMapBatchPreparePush) == 32);
} // namespace metallic::render
#else
#undef uint32_t
#endif
