#language slang 2026
implementing Core;

namespace Metallic
{
public struct ViewCameraConstants
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho;
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
