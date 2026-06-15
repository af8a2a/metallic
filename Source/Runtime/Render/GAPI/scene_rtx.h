#pragma once

#include "Runtime/Render/GAPI/Vulkan/vulkan_scene_rtx.h"

namespace metallic::render {

using SceneRtxStats = vulkan::SceneRtxStats;
using SceneRtxBuilder = vulkan::SceneRtxBuilder;
using SceneRayQueryBindingKind = vulkan::SceneRayQueryBindingKind;
using SceneRayQueryBindingDesc = vulkan::SceneRayQueryBindingDesc;
using SceneRayQueryProgramDesc = vulkan::SceneRayQueryProgramDesc;
using SceneRayQueryDispatchBinding = vulkan::SceneRayQueryDispatchBinding;
using SceneRayQueryDispatchDesc = vulkan::SceneRayQueryDispatchDesc;
using SceneRayQueryProgram = vulkan::SceneRayQueryProgram;

} // namespace metallic::render
