#ifndef METALLIC_SHADER_TO_HUMAN_SCATTER
#define METALLIC_SHADER_TO_HUMAN_SCATTER

#include "ShaderToHuman.slang"

// The caller implements onGfxForAllScatter(int2 pxPos, float4 color), including
// clipping and destination writes. Select one invocation or disjoint output
// regions; scatter is not synchronized across shader invocations.
#include "ShaderToHuman/include/s2h_scatter.hlsl"

#endif // METALLIC_SHADER_TO_HUMAN_SCATTER
