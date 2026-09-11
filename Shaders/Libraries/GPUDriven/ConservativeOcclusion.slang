#ifndef METALLIC_CONSERVATIVE_OCCLUSION_SLANG
#define METALLIC_CONSERVATIVE_OCCLUSION_SLANG

float conservativeTransformScale(float3 a, float3 b, float3 c)
{
    // Gershgorin bound on the largest eigenvalue of A^T A, also safe for shear.
    float ab = abs(dot(a, b)), ac = abs(dot(a, c)), bc = abs(dot(b, c));
    return sqrt(max(dot(a, a) + ab + ac, max(dot(b, b) + ab + bc, dot(c, c) + ac + bc)));
}

// Camera-space AABB enclosing the sphere. Project both Z endpoints: projecting
// the center and adding r/(z-r) misses the off-axis perspective displacement.
// The one-pixel guard covers raster rounding and subpixel temporal jitter.
bool conservativeSphereScreenRect(float3 center, float radius, float2 size,
    float aspect, float fov, float orthoHeight, float zNear, out float4 rect)
{
    rect = 0.0;
    if (!all(isfinite(center)) || !isfinite(radius) || radius < 0.0 ||
        any(size < 1.0) || center.z - radius <= zNear) {
        return false;
    }
    float2 lo = center.xy - radius, hi = center.xy + radius;
    float2 halfExtent;
    if (orthoHeight > 0.0) {
        halfExtent = 0.5 * orthoHeight * float2(aspect, 1.0);
    } else {
        float nearZ = center.z - radius, farZ = center.z + radius;
        lo = min(lo / nearZ, lo / farZ);
        hi = max(hi / nearZ, hi / farZ);
        halfExtent = tan(fov * 0.5) * float2(aspect, 1.0);
    }
    lo /= halfExtent;
    hi /= halfExtent;
    float2 minPixels = (float2(lo.x, -hi.y) * 0.5 + 0.5) * size - 1.0;
    float2 maxPixels = (float2(hi.x, -lo.y) * 0.5 + 0.5) * size + 1.0;
    if (!all(isfinite(minPixels)) || !all(isfinite(maxPixels)) ||
        any(maxPixels < 0.0) || any(minPixels >= size)) {
        return false;
    }
    rect = float4(clamp(minPixels, 0.0, size - 1.0), clamp(maxPixels, 0.0, size - 1.0));
    return true;
}

float conservativeProjectDepth(float viewZ, float zNear, float zFar, bool orthographic, bool reversedZ)
{
    if (orthographic) {
        return reversedZ ? (zFar - viewZ) / (zFar - zNear) : (viewZ - zNear) / (zFar - zNear);
    }
    float safeZ = abs(viewZ) > 0.000001 ? viewZ : (viewZ < 0.0 ? -0.000001 : 0.000001);
    // Avoid subtracting two nearly equal values for distant reversed-Z geometry.
    return reversedZ ? (zNear / safeZ) * ((zFar - viewZ) / (zFar - zNear))
        : (zFar / (zFar - zNear)) * (1.0 - zNear / safeZ);
}

bool conservativeDepthOccluded(float nearestDepth, float4 depths, bool reversedZ)
{
    if (!isfinite(nearestDepth) || !all(isfinite(depths)) ||
        any(depths < 0.0) || any(depths > 1.0)) { return false; }
    float bias = max(1e-7, abs(nearestDepth) * 4e-7);
    return reversedZ ? nearestDepth + bias < min(min(depths.x, depths.y), min(depths.z, depths.w))
        : nearestDepth - bias > max(max(depths.x, depths.y), max(depths.z, depths.w));
}

#endif
