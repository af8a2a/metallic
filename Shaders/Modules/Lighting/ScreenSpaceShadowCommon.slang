#language slang 2026
implementing Lighting;

import Core;
using Metallic;

namespace Metallic.Lighting
{
public struct ShadowParameters
{
    public ViewConstants view;
    public GpuPunctualLight light;
    public float4 trace; // ray length, reserved, normal bias, tan(angular radius)
    public uint4 control; // enabled, reserved, stable light slot, debug
    public float4 shape; // local light radius, denoise enabled, reserved, reserved
};
} // namespace Metallic.Lighting
