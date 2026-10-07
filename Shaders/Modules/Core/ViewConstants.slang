#language slang 2026
implementing ShaderCore;

namespace Metallic
{
public struct ViewCameraConstants
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho; // near plane, far plane, orthographic height, fixed 1 for reversed Z.
};

public struct ViewConstants
{
    public ViewCameraConstants current;
    public ViewCameraConstants previous;
    public float4 jitter;
    public uint4 frame;
    public float4 outputSize;
};
} // namespace Metallic
