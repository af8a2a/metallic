#pragma once

#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"

namespace metallic::render {

using SceneRtxStats = vulkan::SceneRtxStats;
using ScenePartitionedRtxStats = vulkan::ScenePartitionedRtxStats;
using SceneRtxBuilder = vulkan::SceneRtxBuilder;
using ScenePartitionedRtxBuilder = vulkan::ScenePartitionedRtxBuilder;
using SceneRayQueryBindingKind = vulkan::SceneRayQueryBindingKind;
using SceneRayQueryBindingDesc = vulkan::SceneRayQueryBindingDesc;
using SceneRayQueryProgramDesc = vulkan::SceneRayQueryProgramDesc;
using SceneRayQueryDispatchBinding = vulkan::SceneRayQueryDispatchBinding;
using SceneRayQueryDispatchDesc = vulkan::SceneRayQueryDispatchDesc;
using SceneRayQueryProgram = vulkan::SceneRayQueryProgram;

} // namespace metallic::render
