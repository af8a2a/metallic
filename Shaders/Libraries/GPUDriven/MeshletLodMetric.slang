#ifndef MESHLET_LOD_METRIC_SLANG
#define MESHLET_LOD_METRIC_SLANG

// Shared by resident and streamed selection. Camera encoding is independent of
// either geometry payload: eye.w = near, forward.w = orthographic, and projection
// stores render height, tan(half FOV), ortho height, and the biased pixel threshold.
float meshletLodProjectedError(float4 sphere, float groupError,
    float4 world0, float4 world1, float4 world2, float4 world3,
    float4 eye, float4 forward, float4 projection, float worldBoundsPadding = 0.0)
{
    // sqrt(||A^T A||inf) bounds the largest singular value, including shear.
    float3 a = world0.xyz, b = world1.xyz, c = world2.xyz;
    float ab = abs(dot(a, b)), ac = abs(dot(a, c)), bc = abs(dot(b, c));
    float scale = sqrt(max(dot(a, a) + ab + ac, max(dot(b, b) + ab + bc, dot(c, c) + ac + bc)));
    float error = groupError * scale;
    if (error <= 0.0) { return 0.0; }
    if (forward.w != 0.0) { return error * projection.x / max(projection.z, 1e-6); }
    float3 center = a * sphere.x + b * sphere.y + c * sphere.z + world3.xyz - eye.xyz;
    float z = dot(center, forward.xyz);
    float radius = sphere.w * scale + error + worldBoundsPadding;
    float minZ = z - radius;
    if (minZ <= eye.w) { return 3.402823466e+38; }
    float radial = sqrt(max(dot(center, center) - z * z, 0.0)) + radius;
    float slope = radial / minZ;
    return error * (projection.x / max(2.0 * projection.y, 1e-6)) / minZ * sqrt(1.0 + slope * slope);
}

bool meshletLodGroupNeedsFine(float4 sphere, float groupError, uint level, uint flags,
    float4 world0, float4 world1, float4 world2, float4 world3,
    float4 eye, float4 forward, float4 projection, uint manualLevel)
{
    if ((flags & 1u) != 0u) { return true; }
    return manualLevel != 0xffffffffu ? level >= manualLevel :
        meshletLodProjectedError(sphere, groupError, world0, world1, world2, world3,
            eye, forward, projection) > projection.w;
}

#endif
