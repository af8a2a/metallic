import Core;
import Material;
using Metallic;
using Metallic.Material;

struct TextureFootprintProbePush
{
    uint width;
    uint height;
    uint index;
    uint orthographic;
    float aspect;
    float verticalFovRadians;
    float orthographicHeight;
    float reserved;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void textureFootprintProbeMain()
{
    TextureFootprintProbePush push = getConstants<TextureFootprintProbePush>();
    getResource<RWStructuredBuffer<float2>>(0)[push.index] = primaryTextureRayCone(
        uint2(push.width, push.height), push.aspect, push.verticalFovRadians,
        push.orthographicHeight, push.orthographic != 0u);
}
