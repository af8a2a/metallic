#pragma once

#include "StreamLODSettings.h"

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

namespace metallic::render::builtin_pass {

inline void appendStreamRayLayout(std::vector<ComputeResourceBindingDesc>& bindings)
{
    for (const auto member : {
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages),
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable),
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamInstances),
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader),
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamChanges),
            METALLIC_RESOURCE_MEMBER(SceneResourceParameters, streamPageCount)}) {
        bindings.push_back({.binding = member, .kind = ComputeResourceBindingKind::StorageBuffer});
    }
}

inline void appendStreamRayBindings(std::vector<ComputeDispatchBinding>& bindings,
    const MeshletStreamDeferredGPUResourcesView& stream, Buffer* changes)
{
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages), .buffer = stream.pageBuffer});
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable), .buffer = stream.pageTableBuffer});
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamInstances), .buffer = stream.instanceBuffer});
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader), .buffer = stream.activeHeaderBuffer});
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, streamPageCount), .buffer = stream.paramsBuffer});
    bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamChanges), .buffer = changes});
}

// Streaming selection must use the exact camera that generates primary rays.
template<typename Push>
void describeStreamRayView(const Push& push, uint32_t displayHeight,
    const RenderGraphProperties& properties, MeshletStreamFrameDesc& view)
{
    view.width = push.width;
    view.height = push.height;
    // The runtime converts this display-pixel error to render pixels after
    // clamping and applying bias. Keep dynamic-resolution scaling there.
    view.displayHeight = displayHeight;
    view.lodPixelError = properties.value("lodPixelError", 1.5f);
    view.lodBias = properties.value("lodBias", 0.0f);
    applyStreamRayLODSettings(properties, view);
    view.camera.eye = float3(push.eye[0], push.eye[1], push.eye[2]);
    view.camera.center = float3(push.center[0], push.center[1], push.center[2]);
    view.camera.up = float3(push.upProjection[0], push.upProjection[1], push.upProjection[2]);
    view.camera.fovDegrees = push.viewport[3] * (180.0f / 3.14159265358979323846f);
    view.camera.znear = push.clipOrtho[0];
    view.camera.zfar = push.clipOrtho[1];
    view.camera.orthographic = push.upProjection[3] != 0.0f;
    view.camera.orthoHeight = push.clipOrtho[2];
}

} // namespace metallic::render::builtin_pass
