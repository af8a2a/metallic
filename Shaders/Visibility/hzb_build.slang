struct HZBBuildUniforms {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint sourceScale;
    uint3 _pad;
};

ConstantBuffer<HZBBuildUniforms> uniforms; // buffer(0)
Texture2D<float> sourceTexture;            // texture(0)
RWTexture2D<float> destinationTexture;     // texture(1)

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= uniforms.dstWidth || dtid.y >= uniforms.dstHeight) {
        return;
    }

    uint2 baseCoord = dtid.xy * uniforms.sourceScale;
    float maxDepth = 0.0;
    for (uint y = 0; y < uniforms.sourceScale; ++y) {
        for (uint x = 0; x < uniforms.sourceScale; ++x) {
            uint2 sampleCoord = uint2(min(baseCoord.x + x, uniforms.srcWidth - 1u),
                                      min(baseCoord.y + y, uniforms.srcHeight - 1u));
            maxDepth = max(maxDepth, sourceTexture.Load(int3(sampleCoord, 0)));
        }
    }

    destinationTexture[dtid.xy] = maxDepth;
}
