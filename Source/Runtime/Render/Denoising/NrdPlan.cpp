/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdPlan.h"
#include <algorithm>
#include <cmath>
bool metallic::render::denoising::NrdPlan::beginFrame(const CommonSettings& commonSettings)
{
    splitScreenPrev_ = commonSettings_.splitScreen;

    memcpy(&commonSettings_, &commonSettings, sizeof(commonSettings));

    // Silently fix settings for known cases
    if (isFirstUse_) {
        commonSettings_.accumulationMode = AccumulationMode::CLEAR_AND_RESTART;
        isFirstUse_ = false;
    }

    if (commonSettings_.accumulationMode != AccumulationMode::CONTINUE) {
        splitScreenPrev_ = 0.0f;

        memcpy(commonSettings_.worldToViewMatrixPrev, commonSettings_.worldToViewMatrix,
               sizeof(commonSettings_.worldToViewMatrix));
        memcpy(commonSettings_.viewToClipMatrixPrev, commonSettings_.viewToClipMatrix,
               sizeof(commonSettings_.viewToClipMatrix));

        commonSettings_.resourceSizePrev[0] = commonSettings_.resourceSize[0];
        commonSettings_.resourceSizePrev[1] = commonSettings_.resourceSize[1];

        commonSettings_.rectSizePrev[0] = commonSettings_.rectSize[0];
        commonSettings_.rectSizePrev[1] = commonSettings_.rectSize[1];

        commonSettings_.cameraJitterPrev[0] = commonSettings_.cameraJitter[0];
        commonSettings_.cameraJitterPrev[1] = commonSettings_.cameraJitter[1];
    }

    // Reject invalid dimensions and projection data before deriving shader constants.
    bool isValid = commonSettings_.viewZScale > 0.0f;

    isValid &= commonSettings_.resourceSize[0] != 0 && commonSettings_.resourceSize[1] != 0;

    isValid &= commonSettings_.resourceSizePrev[0] != 0 && commonSettings_.resourceSizePrev[1] != 0;

    isValid &= commonSettings_.rectSize[0] != 0 && commonSettings_.rectSize[1] != 0;

    isValid &= commonSettings_.rectSizePrev[0] != 0 && commonSettings_.rectSizePrev[1] != 0;

    isValid &= ((commonSettings_.motionVectorScale[0] != 0.0f && commonSettings_.motionVectorScale[1] != 0.0f) ||
                commonSettings_.isMotionVectorInWorldSpace);

    isValid &= commonSettings_.cameraJitter[0] >= -0.5f && commonSettings_.cameraJitter[0] <= 0.5f &&
               commonSettings_.cameraJitter[1] >= -0.5f && commonSettings_.cameraJitter[1] <= 0.5f;

    isValid &= commonSettings_.cameraJitterPrev[0] >= -0.5f && commonSettings_.cameraJitterPrev[0] <= 0.5f &&
               commonSettings_.cameraJitterPrev[1] >= -0.5f && commonSettings_.cameraJitterPrev[1] <= 0.5f;

    isValid &= commonSettings_.denoisingRange > 0.0f;

    isValid &= commonSettings_.disocclusionThreshold > 0.0f;

    isValid &= commonSettings_.disocclusionThresholdAlternate > 0.0f;

    isValid &=
        NRD_SUPPORTS_VIEWPORT_OFFSET || (commonSettings_.rectOrigin[0] == 0 && commonSettings_.rectOrigin[1] == 0);

    isValid &= NRD_SUPPORTS_HISTORY_CONFIDENCE || !commonSettings_.isHistoryConfidenceAvailable;

    isValid &= NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX || !commonSettings_.isDisocclusionThresholdMixAvailable;

    isValid &= NRD_SUPPORTS_BASECOLOR_METALNESS || !commonSettings_.isBaseColorMetalnessAvailable;

    isValid &= commonSettings_.rectSize[0] <= commonSettings_.resourceSize[0] &&
               commonSettings_.rectSize[1] <= commonSettings_.resourceSize[1];
    isValid &= commonSettings_.rectSizePrev[0] <= commonSettings_.resourceSizePrev[0] &&
               commonSettings_.rectSizePrev[1] <= commonSettings_.resourceSizePrev[1];
    for (const float* matrix :
         {commonSettings_.worldToViewMatrix, commonSettings_.worldToViewMatrixPrev, commonSettings_.viewToClipMatrix,
          commonSettings_.viewToClipMatrixPrev, commonSettings_.worldPrevToWorldMatrix}) {
        for (uint32_t i = 0; i < 16; ++i)
            isValid &= std::isfinite(matrix[i]);
    }
    isValid &= commonSettings_.viewToClipMatrix[0] != 0 && commonSettings_.viewToClipMatrix[5] != 0 &&
               commonSettings_.viewToClipMatrixPrev[0] != 0 && commonSettings_.viewToClipMatrixPrev[5] != 0;
    if (!isValid)
        return false;

    // Rotators (respecting sample patterns symmetry)
    float angle1 = Sequence::Weyl1D(0.5f, commonSettings_.frameIndex) * radians(90.0f);
    rotatorPre_ = Geometry::GetRotator(angle1);

    float a0 = Sequence::Weyl1D(0.0f, commonSettings_.frameIndex * 2) * radians(90.0f);
    float a1 = Sequence::Bayer4x4(uint2(0, 0), commonSettings_.frameIndex * 2) * radians(360.0f);
    rotator_ = Geometry::CombineRotators(Geometry::GetRotator(a0), Geometry::GetRotator(a1));

    float a2 = Sequence::Weyl1D(0.0f, commonSettings_.frameIndex * 2 + 1) * radians(90.0f);
    float a3 = Sequence::Bayer4x4(uint2(0, 0), commonSettings_.frameIndex * 2 + 1) * radians(360.0f);
    rotatorPost_ = Geometry::CombineRotators(Geometry::GetRotator(a2), Geometry::GetRotator(a3));

    // Main matrices
    viewToClip_ = float4x4(float4(commonSettings_.viewToClipMatrix), float4(commonSettings_.viewToClipMatrix + 4),
                           float4(commonSettings_.viewToClipMatrix + 8), float4(commonSettings_.viewToClipMatrix + 12));

    viewToClipPrev_ =
        float4x4(float4(commonSettings_.viewToClipMatrixPrev), float4(commonSettings_.viewToClipMatrixPrev + 4),
                 float4(commonSettings_.viewToClipMatrixPrev + 8), float4(commonSettings_.viewToClipMatrixPrev + 12));

    worldToView_ =
        float4x4(float4(commonSettings_.worldToViewMatrix), float4(commonSettings_.worldToViewMatrix + 4),
                 float4(commonSettings_.worldToViewMatrix + 8), float4(commonSettings_.worldToViewMatrix + 12));

    worldToViewPrev_ =
        float4x4(float4(commonSettings_.worldToViewMatrixPrev), float4(commonSettings_.worldToViewMatrixPrev + 4),
                 float4(commonSettings_.worldToViewMatrixPrev + 8), float4(commonSettings_.worldToViewMatrixPrev + 12));

    worldPrevToWorld_ = float4x4(
        float4(commonSettings_.worldPrevToWorldMatrix), float4(commonSettings_.worldPrevToWorldMatrix + 4),
        float4(commonSettings_.worldPrevToWorldMatrix + 8), float4(commonSettings_.worldPrevToWorldMatrix + 12));

    // Convert to LH
    uint32_t flags = 0;
    DecomposeProjection(STYLE_D3D, STYLE_D3D, viewToClip_, &flags, nullptr, nullptr, frustum_.a, nullptr, nullptr);

    if (!(flags & PROJ_LEFT_HANDED)) {
        viewToClip_[2] = -viewToClip_[2];
        viewToClipPrev_[2] = -viewToClipPrev_[2];

        worldToView_.Transpose();
        worldToView_[2] = -worldToView_[2];
        worldToView_.Transpose();

        worldToViewPrev_.Transpose();
        worldToViewPrev_[2] = -worldToViewPrev_[2];
        worldToViewPrev_.Transpose();
    }

    // Compute other matrices
    viewToWorld_ = worldToView_;
    viewToWorld_.InvertOrtho();

    viewToWorldPrev_ = worldToViewPrev_;
    viewToWorldPrev_.InvertOrtho();

    const float3& cameraPosition = viewToWorld_[3].xyz;
    const float3& cameraPositionPrev = viewToWorldPrev_[3].xyz;
    float3 translationDelta = cameraPositionPrev - cameraPosition;

    // IMPORTANT: this part is mandatory needed to preserve precision by making matrices camera relative
    viewToWorld_.SetTranslation(float3::Zero());
    worldToView_ = viewToWorld_;
    worldToView_.InvertOrtho();

    viewToWorldPrev_.SetTranslation(translationDelta);
    worldToViewPrev_ = viewToWorldPrev_;
    worldToViewPrev_.InvertOrtho();

    worldToClip_ = viewToClip_ * worldToView_;
    worldToClipPrev_ = viewToClipPrev_ * worldToViewPrev_;

    clipToWorldPrev_ = worldToClipPrev_;
    clipToWorldPrev_.Invert();

    clipToView_ = viewToClip_;
    clipToView_.Invert();

    clipToViewPrev_ = viewToClipPrev_;
    clipToViewPrev_.Invert();

    clipToWorld_ = worldToClip_;
    clipToWorld_.Invert();

    float project[3];
    DecomposeProjection(STYLE_D3D, STYLE_D3D, viewToClip_, &flags, nullptr, nullptr, frustum_.a, project, nullptr);

    projectY_ = project[1];
    orthoMode_ = (flags & PROJ_ORTHO) ? -1.0f : 0.0f;

    DecomposeProjection(STYLE_D3D, STYLE_D3D, viewToClipPrev_, &flags, nullptr, nullptr, frustumPrev_.a, nullptr,
                        nullptr);

    viewDirection_ = -float3(viewToWorld_[2]);
    viewDirectionPrev_ = -float3(viewToWorldPrev_[2]);

    cameraDelta_ = float3(translationDelta.x, translationDelta.y, translationDelta.z);

    frameRateScale_ = max(33.333f / timeDelta_, 1.0f);

    float dx = abs(commonSettings_.cameraJitter[0] - commonSettings_.cameraJitterPrev[0]);
    float dy = abs(commonSettings_.cameraJitter[1] - commonSettings_.cameraJitterPrev[1]);
    jitterDelta_ = max(dx, dy);

    float FPS = frameRateScale_ * 30.0f;
    float nonLinearAccumSpeed = FPS * 0.25f / (1.0f + FPS * 0.25f);
    checkerboardResolveAccumSpeed_ = lerp(nonLinearAccumSpeed, 0.5f, jitterDelta_);

    constantDataOffset_ = 0;
    return isValid;
}
void metallic::render::denoising::NrdPlan::swapHistory(const DenoiserData& denoiserData)
{
    for (uint32_t i = 0; i < denoiserData.pingPongNum; i++) {
        PingPong& pingPong = pingPongs_[denoiserData.pingPongOffset + i];
        ResourceDesc& resource = resources_[pingPong.resourceIndex];

        std::swap(resource.indexInPool, pingPong.indexInPoolToSwapWith);
    }
}

void metallic::render::denoising::NrdPlan::addResource(DescriptorType descriptorType, uint16_t localIndex,
                                                       uint16_t indexToSwapWith)
{
    ResourceType resourceType = (ResourceType)localIndex;
    uint16_t globalIndex = 0;

    if (localIndex >= TRANSIENT_POOL_START) {
        resourceType = ResourceType::TRANSIENT_POOL;
        globalIndex = indexRemap_[localIndex - TRANSIENT_POOL_START];

        if (indexToSwapWith != uint16_t(-1)) {
            assert(indexToSwapWith >= TRANSIENT_POOL_START && indexToSwapWith < UINT16_MAX);

            indexToSwapWith = indexRemap_[indexToSwapWith - TRANSIENT_POOL_START];
            pingPongs_.push_back({resources_.size(), indexToSwapWith});
        }
    } else if (localIndex >= PERMANENT_POOL_START) {
        resourceType = ResourceType::PERMANENT_POOL;
        globalIndex = permanentPoolOffset_ + localIndex - PERMANENT_POOL_START;

        if (indexToSwapWith != uint16_t(-1)) {
            assert(indexToSwapWith >= PERMANENT_POOL_START);

            indexToSwapWith = permanentPoolOffset_ + indexToSwapWith - PERMANENT_POOL_START;
            pingPongs_.push_back({resources_.size(), indexToSwapWith});
        }
    }

    resources_.push_back({descriptorType, resourceType, globalIndex});
}

void metallic::render::denoising::NrdPlan::addTransient(const TextureDesc& textureDesc)
{
    // Try to find a replacement from previous denoisers
    for (uint16_t i = 0; i < transientPoolOffset_; i++) {
        // Format and dimensions must match
        const TextureDesc& t = transientPool_[i];
        if (t.format == textureDesc.format && t.downsampleFactor == textureDesc.downsampleFactor) {
            // The candidate must not be already in use in the current denoiser
            size_t j = 0;
            for (; j < indexRemap_.size(); j++) {
                if (indexRemap_[j] == i)
                    break;
            }

            // A replacement is found - reuse memory
            if (j == indexRemap_.size()) {
                indexRemap_.push_back(i);

                return;
            }
        }
    }

    // A replacement is not found - add memory
    indexRemap_.push_back((uint16_t)transientPool_.size());
    transientPool_.push_back(textureDesc);
}

void* metallic::render::denoising::NrdPlan::emitDispatch(const DenoiserData& denoiserData, uint32_t localIndex)
{
    size_t dispatchIndex = denoiserData.dispatchOffset + localIndex;
    const InternalDispatchDesc& internalDispatchDesc = dispatches_[dispatchIndex];

    // Copy data
    DispatchDesc dispatchDesc = {};
    dispatchDesc.name = internalDispatchDesc.name;
    dispatchDesc.resources = resources_.data() + internalDispatchDesc.resourceOffset;
    dispatchDesc.resourcesNum = internalDispatchDesc.resourcesNum;
    dispatchDesc.pipelineIndex = internalDispatchDesc.pipelineIndex;

    // Update constant data
    if (constantDataOffset_ + internalDispatchDesc.constantBufferDataSize > CONSTANT_DATA_SIZE) {
        assert("Constant data doesn't fit into the prealocated array!" && false);
        dispatchDesc.constantBufferData = nullptr; // TODO: better crash
    } else
        dispatchDesc.constantBufferData = constantData_ + constantDataOffset_;

    dispatchDesc.constantBufferDataSize = internalDispatchDesc.constantBufferDataSize;
    constantDataOffset_ += (internalDispatchDesc.constantBufferDataSize + 15u) & ~15u;

    // Initialize padding and optional constants deterministically.
    if (dispatchDesc.constantBufferData)
        memset((void*)dispatchDesc.constantBufferData, 0, dispatchDesc.constantBufferDataSize);

    // Update grid size
    uint16_t w = commonSettings_.rectSize[0];
    uint16_t h = commonSettings_.rectSize[1];
    uint16_t d = internalDispatchDesc.downsampleFactor;

    if (d == USE_PREV_DIMS) {
        w = commonSettings_.rectSizePrev[0];
        h = commonSettings_.rectSizePrev[1];
        d = 1;
    } else if (d == IGNORE_RS) {
        w = commonSettings_.resourceSize[0];
        h = commonSettings_.resourceSize[1];
        d = 1;
    }

    w = DivideUp(w, d);
    h = DivideUp(h, d);

    dispatchDesc.gridWidth = DivideUp(w, internalDispatchDesc.numThreads.width);
    dispatchDesc.gridHeight = DivideUp(h, internalDispatchDesc.numThreads.height);

    // Store
    activeDispatches_.push_back(dispatchDesc);

    return (void*)dispatchDesc.constantBufferData;
}

namespace metallic::render::denoising {
NrdPlan::NrdPlan()
{
    for (uint32_t index = 0; index < denoiserData_.size(); ++index) {
        auto& data = denoiserData_[index];
        data.index = index;
        data.dispatchOffset = dispatches_.size();
        data.pingPongOffset = pingPongs_.size();
        permanentPoolOffset_ = static_cast<uint16_t>(permanentPool_.size());
        transientPoolOffset_ = static_cast<uint16_t>(transientPool_.size());
        indexRemap_.clear();
        if (index == 0)
            addReblur(data);
        else if (index == 1)
            addRelax(data);
        else
            addReference(data);
        data.pingPongNum = pingPongs_.size() - data.pingPongOffset;
    }
}

void NrdPlan::addPass(const char* shader, std::span<const ShaderDefine> defines, NumThreads threads, uint16_t scale,
                      uint32_t constantSize)
{
    size_t pipeline = 0;
    for (; pipeline < pipelines_.size(); ++pipeline) {
        const auto& candidate = pipelines_[pipeline];
        if (candidate.shaderName != shader || candidate.defines.size() != defines.size())
            continue;
        bool matches = true;
        for (size_t i = 0; i < defines.size(); ++i)
            matches &= std::strcmp(candidate.defines[i].name, defines[i].name) == 0 &&
                       std::strcmp(candidate.defines[i].value, defines[i].value) == 0;
        if (matches)
            break;
    }
    if (pipeline == pipelines_.size())
        pipelines_.push_back({shader, {defines.begin(), defines.end()}});
    dispatches_.push_back({passName_, resourceOffset_, static_cast<uint32_t>(resources_.size() - resourceOffset_),
                           constantSize, static_cast<uint16_t>(pipeline), scale, threads});
}

std::span<const DispatchDesc> NrdPlan::schedule(uint32_t index)
{
    activeDispatches_.clear();
    constantDataOffset_ = 0;
    if (index >= denoiserData_.size())
        return {};
    const auto& data = denoiserData_[index];
    swapHistory(data);
    if (index == 0)
        updateReblur(data);
    else if (index == 1)
        updateRelax(data);
    else
        updateReference(data);
    return activeDispatches_;
}
} // namespace metallic::render::denoising
