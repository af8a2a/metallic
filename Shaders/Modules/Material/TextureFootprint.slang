#language slang 2026
implementing Material;

namespace Metallic.Material
{
// The cone represents one internal-render pixel's diameter. Use the larger
// axis footprint when projection aspect and render dimensions differ.
// Analytic projection slopes avoid acos(dot(ray, neighbour)) rounding to zero
// at high resolutions; the pixel diameter already includes the factor of two.
public float2 primaryTextureRayCone(
    uint2 renderSize,
    float aspect,
    float verticalFovRadians,
    float orthographicHeight,
    bool orthographic)
{
    float2 inverseSize = 1.0 / float2(max(renderSize, uint2(1u, 1u)));
    float pixelScale = max(aspect * inverseSize.x, inverseSize.y);
    if (orthographic) {
        return float2(orthographicHeight * pixelScale, 0.0);
    }
    return float2(0.0, 2.0 * tan(verticalFovRadians * 0.5) * pixelScale);
}
} // namespace Metallic.Material
