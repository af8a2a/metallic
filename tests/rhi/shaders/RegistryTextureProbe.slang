import Core;
import ParameterRoot;
using Metallic;

struct RegistryTextureParams {
    ResourceHandle<RWTexture2D<uint>> image;
    BufferSpan<ResourceHandle<Texture2D<uint>>> samples;
    ResourceHandle<RWStructuredBuffer<uint>> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void registryTextureWriteMain()
{
    RegistryTextureParams params = getParameters<RegistryTextureParams>();
    resolveUniform(params.image)[uint2(0)] = 1001u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void registryTextureReadMain()
{
    RegistryTextureParams params = getParameters<RegistryTextureParams>();
    for (uint i = 0u; i < 3u; ++i) {
        resolveUniform(params.output)[i] = resolveUniform(params.samples[i]).Load(int3(0));
    }
}
