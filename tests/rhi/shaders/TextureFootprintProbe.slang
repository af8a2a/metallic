#include "../TextureFootprintProbeParameters.h"
import Material;
using Metallic;
using Metallic.Material;

[[vk::push_constant]] ConstantBuffer<TextureFootprintProbeParameters> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void textureFootprintProbeMain()
{
    TextureFootprintProbeInput push = gProbe.input;
    if (!gProbe.output.contains(push.index)) { return; }
    gProbe.output.data[push.index] = primaryTextureRayCone(
        uint2(push.width, push.height), push.aspect, push.verticalFovRadians,
        push.orthographicHeight, push.orthographic != 0u);
}
