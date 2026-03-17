struct ShadowUniforms {
    float4x4 invViewProj;
    float4 lightDir;
    uint screenWidth;
    uint screenHeight;
    float normalBias;
    float maxRayDistance;
    uint reversedZ;
};

ConstantBuffer<ShadowUniforms> uniforms;
RaytracingAccelerationStructure sceneTlas;
Texture2D<float> depthTex;
RWTexture2D<float> shadowMap;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= uniforms.screenWidth || pixel.y >= uniforms.screenHeight) {
        return;
    }

    const float depth = depthTex[pixel];
    const float skyClear = uniforms.reversedZ != 0 ? 0.0 : 1.0;
    if (depth == skyClear) {
        shadowMap[pixel] = 1.0;
        return;
    }

    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5) / float(uniforms.screenWidth) * 2.0 - 1.0;
    ndc.y = 1.0 - (float(pixel.y) + 0.5) / float(uniforms.screenHeight) * 2.0;

    const float4 clipPos = float4(ndc, depth, 1.0);
    const float4 worldPos4 = mul(uniforms.invViewProj, clipPos);
    const float3 worldPos = worldPos4.xyz / worldPos4.w;

    const float3 lightDir = normalize(uniforms.lightDir.xyz);
    const float3 origin = worldPos + lightDir * uniforms.normalBias;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = lightDir;
    ray.TMin = 0.001;
    ray.TMax = uniforms.maxRayDistance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(sceneTlas, 0, 0xFF, ray);
    while (query.Proceed()) {
    }

    const bool occluded = query.CommittedStatus() != COMMITTED_NOTHING;
    shadowMap[pixel] = occluded ? 0.0 : 1.0;
}
