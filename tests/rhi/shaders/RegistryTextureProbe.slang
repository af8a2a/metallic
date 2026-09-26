import Core;
import ParameterRoot;
using Metallic;

struct RegistryTextureParams {
    DescriptorHandle<RWTexture2D<uint>> image;
    DescriptorHandle<Texture2D<uint>>* samples;
    DescriptorHandle<RWStructuredBuffer<uint>> output;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void registryTextureWriteMain()
{
    RegistryTextureParams params = getParameters<RegistryTextureParams>();
    resolveDescriptor(params.image)[uint2(0)] = 1001u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void registryTextureReadMain()
{
    RegistryTextureParams params = getParameters<RegistryTextureParams>();
    for (uint i = 0u; i < 3u; ++i) {
        resolveDescriptor(params.output)[i] = resolveDescriptor(params.samples[i]).Load(int3(0));
    }
}
