#ifndef METALLIC_SHADER_TO_HUMAN
#define METALLIC_SHADER_TO_HUMAN

// Upstream files stay unmodified; version and licenses are in ShaderToHuman/.
// These includes declare no resources and produce no overlay until called.
#include "ShaderToHuman/include/s2h.hlsl"
#include "ShaderToHuman/include/s2h_3d.hlsl"

namespace ShaderDebug {

// ContextGather stores premultiplied linear RGBA. Keep the result linear for
// Metallic's exposure/FinalBlit chain; do not apply another sRGB conversion here.
float4 composite(float4 linearBackground, ContextGather overlay)
{
    return linearBackground * (1.0f - overlay.dstColor.a) + overlay.dstColor;
}

} // namespace ShaderDebug

#endif // METALLIC_SHADER_TO_HUMAN
