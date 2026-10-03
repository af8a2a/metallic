import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Material;
using Metallic.Material;

[shader("compute")]
[numthreads(3, 1, 1)]
void materialRuntimeProbeMain(uint3 id : SV_DispatchThreadID)
{
    PathTraceMaterial m = resolveBuffer<StructuredBuffer<PathTraceMaterial>>(getResourceParameters<MaterialRuntimeProbeResources>().materials)[id.x];
    RWStructuredBuffer<uint> output = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<MaterialRuntimeProbeResources>().output);
    uint cursor = id.x * 181u;
    { uint4 v = asuint(m.baseColor); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.emissive); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.params); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.textureParams); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.glassParams); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.attenuationColor); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmission); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.rtxcrHairBaseColor); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.rtxcrHairParams0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.rtxcrHairParams1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.rtxcrHairDiffuseTint); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.baseColorTexture.textureIndex, m.baseColorTexture.texCoord, m.baseColorTexture.ntcTextureSetIndex, m.baseColorTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.baseColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.baseColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.metallicRoughnessTexture.textureIndex, m.metallicRoughnessTexture.texCoord, m.metallicRoughnessTexture.ntcTextureSetIndex, m.metallicRoughnessTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.metallicRoughnessTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.metallicRoughnessTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.normalTexture.textureIndex, m.normalTexture.texCoord, m.normalTexture.ntcTextureSetIndex, m.normalTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.normalTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.normalTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.occlusionTexture.textureIndex, m.occlusionTexture.texCoord, m.occlusionTexture.ntcTextureSetIndex, m.occlusionTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.occlusionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.occlusionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.emissiveTexture.textureIndex, m.emissiveTexture.texCoord, m.emissiveTexture.ntcTextureSetIndex, m.emissiveTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.emissiveTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.emissiveTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.transmissionTexture.textureIndex, m.transmissionTexture.texCoord, m.transmissionTexture.ntcTextureSetIndex, m.transmissionTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.transmissionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.transmissionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.thicknessTexture.textureIndex, m.thicknessTexture.texCoord, m.thicknessTexture.ntcTextureSetIndex, m.thicknessTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.thicknessTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.thicknessTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.diffuseTransmissionTexture.textureIndex, m.diffuseTransmissionTexture.texCoord, m.diffuseTransmissionTexture.ntcTextureSetIndex, m.diffuseTransmissionTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.diffuseTransmissionColorTexture.textureIndex, m.diffuseTransmissionColorTexture.texCoord, m.diffuseTransmissionColorTexture.ntcTextureSetIndex, m.diffuseTransmissionColorTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specular); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.specularTexture.textureIndex, m.specularTexture.texCoord, m.specularTexture.ntcTextureSetIndex, m.specularTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = uint4(m.specularColorTexture.textureIndex, m.specularColorTexture.texCoord, m.specularColorTexture.ntcTextureSetIndex, m.specularColorTexture.ntcChannelMapping); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
#ifndef MATERIAL_PROBE_REVISION
#define MATERIAL_PROBE_REVISION 0
#endif
    output[cursor] = materialProgramId(m) + MATERIAL_PROBE_REVISION * 10u;
}
