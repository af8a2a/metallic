#language slang 2026
implementing Lighting;

import ShaderCore;
using Metallic;

namespace Metallic.Lighting
{
public struct ShadowParameters
{
    public ViewConstants view;
    public GPUPunctualLight light;
    public float4 trace; // ray length, reserved, normal bias, tan(angular radius)
    public uint4 control; // enabled, celestial domain, tagged source identity, debug
    public float4 shape; // local light radius, denoise enabled, reserved, reserved
};
} // namespace Metallic.Lighting
