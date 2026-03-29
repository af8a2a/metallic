struct ShadowUniforms {
    float4x4 invViewProj;
    float4 lightDir;
    uint screenWidth;
    uint screenHeight;
    float normalBias;
    float maxRayDistance;
    uint reversedZ;
};

[[vk::push_constant]] ConstantBuffer<ShadowUniforms> uniforms;
RaytracingAccelerationStructure sceneTlas;
Texture2D<float> depthTex;
RWTexture2D<float> shadowMap;

bool isSkyDepth(float depth) {
    const float skyClear = uniforms.reversedZ != 0 ? 0.0 : 1.0;
    return abs(depth - skyClear) < 1e-6;
}

float3 reconstructWorldPosition(uint2 pixel, float depth) {
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5) / float(uniforms.screenWidth) * 2.0 - 1.0;
    ndc.y = 1.0 - (float(pixel.y) + 0.5) / float(uniforms.screenHeight) * 2.0;

    const float4 clipPos = float4(ndc, depth, 1.0);
    const float4 worldPos4 = mul(uniforms.invViewProj, clipPos);
    return worldPos4.xyz / worldPos4.w;
}

bool sampleWorldPosition(int2 pixel, out float3 worldPos) {
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(uniforms.screenWidth) ||
        pixel.y >= int(uniforms.screenHeight)) {
        worldPos = 0.0;
        return false;
    }

    const float depth = depthTex[uint2(pixel)];
    if (isSkyDepth(depth)) {
        worldPos = 0.0;
        return false;
    }

    worldPos = reconstructWorldPosition(uint2(pixel), depth);
    return true;
}

float3 estimateWorldNormal(uint2 pixel, float3 worldPos, float3 lightDir) {
    float3 leftPos;
    float3 rightPos;
    float3 upPos;
    float3 downPos;

    const bool hasLeft = sampleWorldPosition(int2(pixel) + int2(-1, 0), leftPos);
    const bool hasRight = sampleWorldPosition(int2(pixel) + int2(1, 0), rightPos);
    const bool hasUp = sampleWorldPosition(int2(pixel) + int2(0, -1), upPos);
    const bool hasDown = sampleWorldPosition(int2(pixel) + int2(0, 1), downPos);

    float3 dx = lightDir;
    bool hasDx = false;
    if (hasLeft && hasRight) {
        const float3 leftDelta = worldPos - leftPos;
        const float3 rightDelta = rightPos - worldPos;
        dx = dot(leftDelta, leftDelta) < dot(rightDelta, rightDelta) ? leftDelta : rightDelta;
        hasDx = true;
    } else if (hasLeft) {
        dx = worldPos - leftPos;
        hasDx = true;
    } else if (hasRight) {
        dx = rightPos - worldPos;
        hasDx = true;
    }

    float3 dy = lightDir;
    bool hasDy = false;
    if (hasUp && hasDown) {
        const float3 upDelta = worldPos - upPos;
        const float3 downDelta = downPos - worldPos;
        dy = dot(upDelta, upDelta) < dot(downDelta, downDelta) ? upDelta : downDelta;
        hasDy = true;
    } else if (hasUp) {
        dy = worldPos - upPos;
        hasDy = true;
    } else if (hasDown) {
        dy = downPos - worldPos;
        hasDy = true;
    }

    if (!hasDx || !hasDy) {
        return lightDir;
    }

    float3 normal = normalize(cross(dx, dy));
    if (dot(normal, lightDir) < 0.0) {
        normal = -normal;
    }
    return normal;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= uniforms.screenWidth || pixel.y >= uniforms.screenHeight) {
        return;
    }

    const float depth = depthTex[pixel];
    if (isSkyDepth(depth)) {
        shadowMap[pixel] = 1.0;
        return;
    }

    const float3 lightDir = normalize(uniforms.lightDir.xyz);
    const float3 worldPos = reconstructWorldPosition(pixel, depth);
    const float3 surfaceNormal = estimateWorldNormal(pixel, worldPos, lightDir);
    const float3 origin = worldPos + surfaceNormal * uniforms.normalBias;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = lightDir;
    ray.TMin = max(uniforms.normalBias, 0.001);
    ray.TMax = uniforms.maxRayDistance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(sceneTlas, 0, 0xFF, ray);
    while (query.Proceed()) {
    }

    const bool occluded = query.CommittedStatus() != COMMITTED_NOTHING;
    shadowMap[pixel] = occluded ? 0.0 : 1.0;
}
