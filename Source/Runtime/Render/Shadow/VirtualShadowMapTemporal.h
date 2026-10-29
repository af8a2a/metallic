#ifdef __cplusplus
#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

namespace metallic::render {
using VSMTemporalUInt = uint32_t;
using VSMTemporalViews = GPUBufferSpan;
using VSMTemporalDepth = ShaderSampledImage;
using VSMTemporalShadows = ShaderStorageImage;
using VSMTemporalHistory = ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
typealias VSMTemporalUInt = uint;
typealias VSMTemporalViews = BufferSpan<ViewConstants>;
typealias VSMTemporalDepth = ResourceHandle<Texture2D<float>>;
typealias VSMTemporalShadows = ResourceHandle<RWTexture2DArray<float>>;
typealias VSMTemporalHistory = ResourceHandle<RWTexture2D<uint2>>;
#endif

struct VirtualShadowMapTemporalParams
{
    VSMTemporalViews views;
    VSMTemporalUInt padding;
    VSMTemporalDepth depth;
    VSMTemporalShadows shadows;
    VSMTemporalHistory previousHistory, currentHistory;
    VSMTemporalUInt width, height, baseLayer, faceCount;
    VSMTemporalUInt compact, lightType, maxFrames, hasHistory;
    float lightX, lightY, lightZ;
    VSMTemporalUInt reserved;
};

#ifdef __cplusplus
static_assert(sizeof(VirtualShadowMapTemporalParams) == 80);

// One stable light owns this history, independently of its publication layer.
// The caller serializes frames and provides receiver depth in ShaderRead and
// the whole shadows array in General, with its current writes made visible.
// Recording retains descriptor allocations; public textures remain caller-owned.
class VirtualShadowMapTemporal
{
public:
    VirtualShadowMapTemporal();
    ~VirtualShadowMapTemporal();
    VirtualShadowMapTemporal(VirtualShadowMapTemporal&&) noexcept;
    VirtualShadowMapTemporal& operator=(VirtualShadowMapTemporal&&) noexcept;

    Result<> record(Device& device, CommandBuffer& commands, TextureView& depth,
        TextureView& shadows, const ViewConstants& view, uint64_t frameIndex,
        uint64_t invalidationRevision, uint64_t contentKey, const VirtualShadowMapLight& light,
        const VirtualShadowMapSettings& settings, uint32_t maxFrames, uint32_t baseLayer,
        uint32_t faceCount, bool compact, std::string& log);
    uint64_t memoryBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace metallic::render
#endif
