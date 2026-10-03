#language slang 2026
module Core;

// ComputeProgram named parameters and low-level slot test adapter.
// Standalone ComputeKernel shaders import ShaderCore.
__exported import ShaderCore;
import ParameterRoot;
__include "Core/ComputeResources.slang";
__include "Core/NamedResources.slang";
