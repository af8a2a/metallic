#language slang 2026
module Core;

// ComputeProgram named resource and constant parameters.
// Standalone ComputeKernel shaders import ShaderCore.
__exported import ShaderCore;
import ParameterRoot;
__include "Core/ComputeResources.slang";
__include "Core/NamedResources.slang";
