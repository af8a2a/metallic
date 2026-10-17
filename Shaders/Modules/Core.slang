#language slang 2026
module Core;

// ComputeProgram named resource and constant parameters.
// Standalone ComputeKernel shaders import ShaderCore.
__exported import ShaderCore;
__exported import ComputeResourceTypes;

#ifndef METALLIC_INLINE_COMPUTE_ROOT
#define METALLIC_INLINE_COMPUTE_ROOT 0
#endif
#if METALLIC_INLINE_COMPUTE_ROOT != 0 && METALLIC_INLINE_COMPUTE_ROOT != 1
#error METALLIC_INLINE_COMPUTE_ROOT must be 0 or 1
#endif
#if METALLIC_INLINE_COMPUTE_ROOT
import InlineComputeRoot;
#else
import ParameterRoot;
#endif
__include "Core/ComputeResources.slang";
__include "Core/NamedResources.slang";
__include "Core/SceneResources.slang";
