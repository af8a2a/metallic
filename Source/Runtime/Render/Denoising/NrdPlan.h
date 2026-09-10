/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include "NrdTypes.h"
#include "ml.h"
#include "ml.hlsli"
#include <array>
#include <cassert>
#include <cstring>
#include <span>
#include <vector>
#include "Shaders/Libraries/Denoising/NRD/NRDConfig.hlsli"

// The constants use the same field list on CPU and GPU. MathLib matrices are
// column-major; Bindless.hlsli declares this explicitly regardless of Slang defaults.
#define NRD_CONSTANTS_START(name) struct name {
#define NRD_CONSTANT(type, name) type name;
#define NRD_CONSTANTS_END                                                                                              \
    }                                                                                                                  \
    ;
#define NRD_INPUTS_START
#define NRD_INPUT(...)
#define NRD_INPUTS_END
#define NRD_OUTPUTS_START
#define NRD_OUTPUT(...)
#define NRD_OUTPUTS_END
#define NRD_SAMPLERS_START
#define NRD_SAMPLER(...)
#define NRD_SAMPLERS_END
#define NRD_DIFFUSE_SPECULAR "BOTH"
#define NRD_RADIANCE "RADIANCE"
#define _NRD_STRINGIFY(s) #s
#define NRD_STRINGIFY(s) _NRD_STRINGIFY(s)
#define NRD_PASS(name) beginPass(NRD_STRINGIFY(DENOISER_NAME) " - " name)
#define NRD_STAGE_SIZED(shader, defines, scale)                                                                        \
    addPass(#shader, defines, NumThreads(shader##GroupX, shader##GroupY), scale, sizeof(shader##Constants))
#define NRD_STAGE(shader, defines) NRD_STAGE_SIZED(shader, defines, 1)
// Shared dimension names used by the three reference-derived constant writers.
#define NRD_DECLARE_DIMS                                                                                               \
    [[maybe_unused]] uint16_t resourceW = commonSettings_.resourceSize[0];                                             \
    [[maybe_unused]] uint16_t resourceH = commonSettings_.resourceSize[1];                                             \
    [[maybe_unused]] uint16_t resourceWprev = commonSettings_.resourceSizePrev[0];                                     \
    [[maybe_unused]] uint16_t resourceHprev = commonSettings_.resourceSizePrev[1];                                     \
    [[maybe_unused]] uint16_t rectW = commonSettings_.rectSize[0];                                                     \
    [[maybe_unused]] uint16_t rectH = commonSettings_.rectSize[1];                                                     \
    [[maybe_unused]] uint16_t rectWprev = commonSettings_.rectSizePrev[0];                                             \
    [[maybe_unused]] uint16_t rectHprev = commonSettings_.rectSizePrev[1];

using uint = uint32_t;
namespace metallic::render::denoising {
constexpr uint16_t PERMANENT_POOL_START = 1000;
constexpr uint16_t TRANSIENT_POOL_START = 2000;
constexpr size_t CONSTANT_DATA_SIZE = 128 * 1024;
constexpr uint16_t USE_PREV_DIMS = 0xFFFF;
constexpr uint16_t IGNORE_RS = 0xFFFE;
inline uint16_t DivideUp(uint32_t x, uint16_t y)
{
    return uint16_t((x + y - 1) / y);
}
template <class T> uint16_t AsUint(T x)
{
    return static_cast<uint16_t>(x);
}
struct DenoiserData {
    uint32_t index = 0;
    struct {
        ReblurSettings reblur;
        RelaxSettings relax;
        ReferenceSettings reference;
    } settings;
    size_t dispatchOffset = 0, pingPongOffset = 0, pingPongNum = 0;
};
struct PingPong {
    size_t resourceIndex;
    uint16_t indexInPoolToSwapWith;
};
struct NumThreads {
    uint8_t width, height;
};
struct InternalDispatchDesc {
    const char* name;
    size_t resourceOffset;
    uint32_t resourcesNum, constantBufferDataSize;
    uint16_t pipelineIndex, downsampleFactor;
    NumThreads numThreads;
};

// Metallic owns the pass recipes, history rotation and constants. This class
// never loads NRD binaries or asks the SDK to produce opaque dispatches.
class NrdPlan {
public:
    NrdPlan();
    NrdPlan(const NrdPlan&) = delete;
    NrdPlan& operator=(const NrdPlan&) = delete;
    bool beginFrame(const CommonSettings& settings);
    std::span<const DispatchDesc> schedule(uint32_t denoiserIndex);
    void setReblurSettings(const ReblurSettings& s) { denoiserData_[0].settings.reblur = s; }
    void setRelaxSettings(const RelaxSettings& s) { denoiserData_[1].settings.relax = s; }
    const auto& permanentPool() const { return permanentPool_; }
    const auto& transientPool() const { return transientPool_; }
    const auto& pipelines() const { return pipelines_; }
    const auto& commonSettings() const { return commonSettings_; }

private:
    void addReblur(DenoiserData&);
    void addRelax(DenoiserData&);
    void addReference(DenoiserData&);
    void updateReblur(const DenoiserData&);
    void updateRelax(const DenoiserData&);
    void updateReference(const DenoiserData&);
    void writeReblurConstants(const ReblurSettings&, void*);
    void writeRelaxConstants(const RelaxSettings&, void*);
    void swapHistory(const DenoiserData&);
    void addResource(DescriptorType, uint16_t, uint16_t = UINT16_MAX);
    void addTransient(const TextureDesc&);
    void addPermanent(const TextureDesc& t) { permanentPool_.push_back(t); }
    void readTexture(uint16_t i, uint16_t j = UINT16_MAX) { addResource(DescriptorType::TEXTURE, i, j); }
    void writeTexture(uint16_t i, uint16_t j = UINT16_MAX) { addResource(DescriptorType::STORAGE_TEXTURE, i, j); }
    void beginPass(const char* n)
    {
        passName_ = n;
        resourceOffset_ = resources_.size();
    }
    void addPass(const char*, std::span<const ShaderDefine>, NumThreads, uint16_t, uint32_t);
    void* emitDispatch(const DenoiserData&, uint32_t);
    std::array<DenoiserData, 4> denoiserData_{};
    std::vector<TextureDesc> permanentPool_, transientPool_;
    std::vector<ResourceDesc> resources_;
    std::vector<PingPong> pingPongs_;
    std::vector<PipelineDesc> pipelines_;
    std::vector<InternalDispatchDesc> dispatches_;
    std::vector<DispatchDesc> activeDispatches_;
    std::vector<uint16_t> indexRemap_;
    CommonSettings commonSettings_ = {};
    float4x4 viewToClip_ = float4x4::Identity();
    float4x4 viewToClipPrev_ = float4x4::Identity();
    float4x4 clipToView_ = float4x4::Identity();
    float4x4 clipToViewPrev_ = float4x4::Identity();
    float4x4 worldToView_ = float4x4::Identity();
    float4x4 worldToViewPrev_ = float4x4::Identity();
    float4x4 viewToWorld_ = float4x4::Identity();
    float4x4 viewToWorldPrev_ = float4x4::Identity();
    float4x4 worldToClip_ = float4x4::Identity();
    float4x4 worldToClipPrev_ = float4x4::Identity();
    float4x4 clipToWorld_ = float4x4::Identity();
    float4x4 clipToWorldPrev_ = float4x4::Identity();
    float4x4 worldPrevToWorld_ = float4x4::Identity();
    float4 rotatorPre_ = float4::Zero();
    float4 rotator_ = float4::Zero();
    float4 rotatorPost_ = float4::Zero();
    float4 frustum_ = float4::Zero();
    float4 frustumPrev_ = float4::Zero();
    float3 cameraDelta_ = float3::Zero();
    float3 viewDirection_ = float3::Zero();
    float3 viewDirectionPrev_ = float3::Zero();
    float splitScreenPrev_ = 0.0f;
    const char* passName_ = nullptr;
    alignas(16) std::array<uint8_t, CONSTANT_DATA_SIZE> constantStorage_{};
    uint8_t* constantData_ = constantStorage_.data();
    size_t constantDataOffset_ = 0;
    size_t resourceOffset_ = 0;
    float orthoMode_ = 0.0f;
    float checkerboardResolveAccumSpeed_ = 0.0f;
    float jitterDelta_ = 0.0f;
    float timeDelta_ = 0.0f;
    float frameRateScale_ = 0.0f;
    float projectY_ = 0.0f;
    std::array<uint32_t, 4> accumulatedFrames_{};
    uint16_t transientPoolOffset_ = 0;
    uint16_t permanentPoolOffset_ = 0;
    bool isFirstUse_ = true;
};
} // namespace metallic::render::denoising
