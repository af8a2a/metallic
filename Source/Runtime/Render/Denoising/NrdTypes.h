/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include "NrdSettings.h"
#include <cstdint>
#include <string>
#include <vector>
namespace metallic::render::denoising {
enum class ResourceType : uint32_t {
    IN_MV,
    IN_NORMAL_ROUGHNESS,
    IN_VIEWZ,
    IN_DIFF_CONFIDENCE,
    IN_SPEC_CONFIDENCE,
    IN_DISOCCLUSION_THRESHOLD_MIX,
    IN_BASECOLOR_METALNESS,
    IN_DIFF_RADIANCE_HITDIST,
    IN_SPEC_RADIANCE_HITDIST,
    IN_SIGNAL,
    OUT_DIFF_RADIANCE_HITDIST,
    OUT_SPEC_RADIANCE_HITDIST,
    OUT_SIGNAL,
    OUT_VALIDATION,
    TRANSIENT_POOL,
    PERMANENT_POOL,
    MAX_NUM
};
enum class DescriptorType { TEXTURE, STORAGE_TEXTURE };
enum class Format : uint32_t {
    R8_UNORM,
    R8_SNORM,
    R8_UINT,
    R8_SINT,

    RG8_UNORM,
    RG8_SNORM,
    RG8_UINT,
    RG8_SINT,

    RGBA8_UNORM,
    RGBA8_SNORM,
    RGBA8_UINT,
    RGBA8_SINT,
    RGBA8_SRGB,

    R16_UNORM,
    R16_SNORM,
    R16_UINT,
    R16_SINT,
    R16_SFLOAT,

    RG16_UNORM,
    RG16_SNORM,
    RG16_UINT,
    RG16_SINT,
    RG16_SFLOAT,

    RGBA16_UNORM,
    RGBA16_SNORM,
    RGBA16_UINT,
    RGBA16_SINT,
    RGBA16_SFLOAT,

    R32_UINT,
    R32_SINT,
    R32_SFLOAT,

    RG32_UINT,
    RG32_SINT,
    RG32_SFLOAT,

    RGB32_UINT,
    RGB32_SINT,
    RGB32_SFLOAT,

    RGBA32_UINT,
    RGBA32_SINT,
    RGBA32_SFLOAT,

    R10_G10_B10_A2_UNORM,
    R10_G10_B10_A2_UINT,
    R11_G11_B10_UFLOAT,
    R9_G9_B9_E5_UFLOAT,

    MAX_NUM
};

struct TextureDesc {
    Format format;
    uint16_t downsampleFactor;
};
struct ResourceDesc {
    DescriptorType descriptorType;
    ResourceType type;
    uint16_t indexInPool;
};
struct ShaderDefine {
    const char* name;
    const char* value;
};
struct PipelineDesc {
    std::string shaderName;
    std::vector<ShaderDefine> defines;
};
struct DispatchDesc {
    const char* name = nullptr;
    const ResourceDesc* resources = nullptr;
    uint32_t resourcesNum = 0;
    const uint8_t* constantBufferData = nullptr;
    uint32_t constantBufferDataSize = 0;
    uint16_t pipelineIndex = 0;
    uint16_t gridWidth = 0;
    uint16_t gridHeight = 0;
};
} // namespace metallic::render::denoising
