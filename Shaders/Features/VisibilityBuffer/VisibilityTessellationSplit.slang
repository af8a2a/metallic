// Bounded recursive red/green subdivision in source-triangle domain. Each
// task lane owns one root and a private DFS stack; only finished leaves enter
// the task payload. No global append, inter-dispatch fence or CPU readback.
static const uint kTessRootsPerTask = 16u;
static const uint kTessTasksPerCluster = 8u;
// Fixed capacity shared by task and mesh stages; the draw's push constants
// select the actual depth without changing the payload ABI or recompiling.
static const uint kTessMaxSplitDepth = 3u;
static const uint kTessLeavesPerRoot = 1u << (2u * kTessMaxSplitDepth);
static const uint kTessDomainScale = 4096u;

struct TessPatchPayload
{
    uint record, first, count, material, enabled;
    uint ends[16];
    // Reserve 12,372 bytes at every runtime depth. Scalar storage avoids
    // uint3 array-stride padding.
    uint patches[kTessRootsPerTask * kTessLeavesPerRoot * 3u];
};
groupshared TessPatchPayload g_TessPatches;

struct TessView
{
    float3 eye, forward;
    float pixelScale, nearPlane;
    bool orthographic;
};

float3 tessUnpackDomain(uint word)
{
    float2 yz = float2(word & 65535u, word >> 16u) / float(kTessDomainScale);
    return float3(1.0 - yz.x - yz.y, yz);
}

uint tessDomainMidpoint(uint a, uint b)
{
    // All coordinates remain dyadic and exact throughout the supported depth.
    return (((a & 65535u) + (b & 65535u)) >> 1u) |
        ((((a >> 16u) + (b >> 16u)) >> 1u) << 16u);
}

bool tessPointAfter(float3 a, float3 b)
{
    return a.x > b.x || (a.x == b.x && (a.y > b.y || (a.y == b.y && a.z > b.z)));
}

// Evaluate directly from the original triangle, never repeatedly normalize
// interpolated normals at split points. Canonical edge order is shared across
// source triangles with opposite winding and across different split depths.
void tessSourceDomain(float3 bary, float3 p[3], float3 n[3], float2 uv[3],
    out float3 position, out float3 normal, out float2 texcoord)
{
    uint zero = bary.z == 0.0 ? 2u : (bary.x == 0.0 ? 0u : (bary.y == 0.0 ? 1u : 3u));
    if (zero < 3u) {
        uint a = (zero + 1u) % 3u, b = (zero + 2u) % 3u;
        if (tessPointAfter(p[a], p[b])) { uint t = a; a = b; b = t; }
        position = lerp(p[a], p[b], bary[b]);
        normal = lerp(n[a], n[b], bary[b]);
        texcoord = lerp(uv[a], uv[b], bary[b]);
    } else {
        position = p[0] * bary.x + p[1] * bary.y + p[2] * bary.z;
        normal = n[0] * bary.x + n[1] * bary.y + n[2] * bary.z;
        texcoord = uv[0] * bary.x + uv[1] * bary.y + uv[2] * bary.z;
    }
    normal = tessNormal(normal, tessNormal(cross(p[1] - p[0], p[2] - p[0]), float3(0, 1, 0)));
}

float3 tessSourcePosition(uint word, float3 p[3])
{
    float3 b = tessUnpackDomain(word);
    uint zero = b.z == 0.0 ? 2u : (b.x == 0.0 ? 0u : (b.y == 0.0 ? 1u : 3u));
    if (zero < 3u) {
        uint a = (zero + 1u) % 3u, c = (zero + 2u) % 3u;
        if (tessPointAfter(p[a], p[c])) { uint t = a; a = c; c = t; }
        return lerp(p[a], p[c], b[c]);
    }
    return p[0] * b.x + p[1] * b.y + p[2] * b.z;
}

uint3 tessPatchRates(uint3 patch, float3 source[3], TessView view,
    float edgePixels, uint maximum)
{
    float3 p[3];
    for (uint i = 0u; i < 3u; ++i) { p[i] = tessSourcePosition(patch[i], source); }
    uint3 rates;
    for (uint e = 0u; e < 3u; ++e) {
        rates[e] = tessEdgeRate(p[e], p[(e + 1u) % 3u], view.eye, view.forward,
            view.pixelScale, view.nearPlane, view.orthographic, edgePixels, maximum);
    }
    return rates;
}

uint tessSplitChildren(uint3 patch, uint mask, out uint3 children[4])
{
    if (mask == 7u) {
        uint ab = tessDomainMidpoint(patch.x, patch.y);
        uint bc = tessDomainMidpoint(patch.y, patch.z);
        uint ca = tessDomainMidpoint(patch.z, patch.x);
        children[0] = uint3(patch.x, ab, ca);
        children[1] = uint3(ab, patch.y, bc);
        children[2] = uint3(ca, bc, patch.z);
        children[3] = uint3(ab, bc, ca);
        return 4u;
    }
    // Rotate to either one marked edge (AB) or two (AB, BC). No unmarked
    // exterior edge is split, so a neighbor may stop earlier without a T-junction.
    for (uint i = 0u; i < 2u && mask != 1u && mask != 3u; ++i) {
        patch = patch.yzx;
        mask = (mask >> 1u) | ((mask & 1u) << 2u);
    }
    uint ab = tessDomainMidpoint(patch.x, patch.y);
    children[0] = uint3(patch.x, ab, patch.z);
    if (mask == 1u) {
        children[1] = uint3(ab, patch.y, patch.z);
        return 2u;
    }
    uint bc = tessDomainMidpoint(patch.y, patch.z);
    children[1] = uint3(ab, bc, patch.z);
    children[2] = uint3(ab, patch.y, bc);
    return 3u;
}

uint tessBuildLeaves(uint root, float3 source[3], TessView view, float edgePixels, uint maxFactor, uint maxDepth)
{
    // A depth-first walk needs 1 + 3 * depth pending nodes, not 4^depth.
    uint4 stack[1u + 3u * kTessMaxSplitDepth];
    stack[0] = uint4(0u, kTessDomainScale, kTessDomainScale << 16u, 0u);
    uint pending = 1u, leaves = 0u;
    maxDepth = min(maxDepth, kTessMaxSplitDepth);
    while (pending != 0u) {
        uint4 node = stack[--pending];
        uint mask = 0u;
        if (node.w < maxDepth) {
            uint3 rates = tessPatchRates(node.xyz, source, view, edgePixels, maxFactor + 1u);
            mask = (rates.x > maxFactor ? 1u : 0u) | (rates.y > maxFactor ? 2u : 0u) | (rates.z > maxFactor ? 4u : 0u);
        }
        if (mask == 0u) {
            uint base = (root * kTessLeavesPerRoot + leaves++) * 3u;
            g_TessPatches.patches[base] = node.x;
            g_TessPatches.patches[base + 1u] = node.y;
            g_TessPatches.patches[base + 2u] = node.z;
        } else {
            uint3 children[4];
            uint count = tessSplitChildren(node.xyz, mask, children);
            for (uint i = count; i > 0u; --i) { stack[pending++] = uint4(children[i - 1u], node.w + 1u); }
        }
    }
    return leaves;
}

// Called by all task lanes, including lanes without a source triangle.
void tessDispatchLeaves(uint lane, uint count)
{
    bool singleWave = false;
#if GPU_DRIVEN_AMPLIFICATION_WAVE_OPS
    singleWave = WaveGetLaneCount() >= 32u;
    uint prefix = singleWave ? WavePrefixSum(count) + count : count;
#else
    uint prefix = count;
#endif
    if (lane < kTessRootsPerTask) { g_TessPatches.ends[lane] = prefix; }
    GroupMemoryBarrierWithGroupSync();
    if (!singleWave && lane == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < kTessRootsPerTask; ++i) { sum += g_TessPatches.ends[i]; g_TessPatches.ends[i] = sum; }
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(g_TessPatches.ends[kTessRootsPerTask - 1u], 1, 1, g_TessPatches);
}

uint tessPayloadRoot(TessPatchPayload payload, uint index)
{
    uint root = 0u;
    for (uint stride = 8u; stride > 0u; stride >>= 1u) {
        if (payload.ends[root + stride - 1u] <= index) { root += stride; }
    }
    return root;
}

float3 tessPatchDomain(float3 local, uint3 rates, uint3 patch, float3 source[3])
{
    float3 corners[3];
    for (uint i = 0u; i < 3u; ++i) { corners[i] = tessUnpackDomain(patch[i]); }
    uint zero = local.z < 1e-7 ? 2u : (local.x < 1e-7 ? 0u : (local.y < 1e-7 ? 1u : 3u));
    if (zero < 3u) {
        uint a = (zero + 1u) % 3u, b = (zero + 2u) % 3u;
        uint rate = rates[a], step = uint(round(local[b] * rate));
        if (tessPointAfter(tessSourcePosition(patch[a], source), tessSourcePosition(patch[b], source))) {
            uint t = a; a = b; b = t; step = rate - step;
        }
        return lerp(corners[a], corners[b], float(step) / float(rate));
    }
    return corners[0] * local.x + corners[1] * local.y + corners[2] * local.z;
}
