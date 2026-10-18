module MaterialRuntimeProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Material;
using Metallic.Material;

[shader("compute")]
[numthreads(3, 1, 1)]
void materialRuntimeProbeMain(uint3 id : SV_DispatchThreadID)
{
    SceneMaterialData m = resolveBuffer<StructuredBuffer<SceneMaterialData>>(
        getResourceParameters<MaterialRuntimeProbeResources>().materials)[id.x];
    RWStructuredBuffer<uint> output = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<MaterialRuntimeProbeResources>().output);
    uint cursor = id.x * (kSceneMaterialRecordBytes / 4u + 1u);
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
    { uint2 v = uint2(m.baseColorTexture.textureIndex, m.baseColorTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.baseColorTexture.sourceLodTerm), m.baseColorTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.baseColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.baseColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.metallicRoughnessTexture.textureIndex, m.metallicRoughnessTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.metallicRoughnessTexture.sourceLodTerm), m.metallicRoughnessTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.metallicRoughnessTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.metallicRoughnessTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.normalTexture.textureIndex, m.normalTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.normalTexture.sourceLodTerm), m.normalTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.normalTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.normalTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.occlusionTexture.textureIndex, m.occlusionTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.occlusionTexture.sourceLodTerm), m.occlusionTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.occlusionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.occlusionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.emissiveTexture.textureIndex, m.emissiveTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.emissiveTexture.sourceLodTerm), m.emissiveTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.emissiveTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.emissiveTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.transmissionTexture.textureIndex, m.transmissionTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.transmissionTexture.sourceLodTerm), m.transmissionTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.transmissionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.transmissionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.thicknessTexture.textureIndex, m.thicknessTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.thicknessTexture.sourceLodTerm), m.thicknessTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.thicknessTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.thicknessTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.diffuseTransmissionTexture.textureIndex, m.diffuseTransmissionTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.diffuseTransmissionTexture.sourceLodTerm), m.diffuseTransmissionTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.diffuseTransmissionColorTexture.textureIndex, m.diffuseTransmissionColorTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.diffuseTransmissionColorTexture.sourceLodTerm), m.diffuseTransmissionColorTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.diffuseTransmissionColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specular); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.specularTexture.textureIndex, m.specularTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.specularTexture.sourceLodTerm), m.specularTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(m.specularColorTexture.textureIndex, m.specularColorTexture.texCoord); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint2 v = uint2(asuint(m.specularColorTexture.sourceLodTerm), m.specularColorTexture.sourceMipCount); for (uint i = 0; i < 2; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularColorTexture.transform0); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
    { uint4 v = asuint(m.specularColorTexture.transform1); for (uint i = 0; i < 4; ++i) { output[cursor++] = v[i]; } }
#ifndef MATERIAL_PROBE_REVISION
#define MATERIAL_PROBE_REVISION 0
#endif
    output[cursor] = materialProgramId(m) + MATERIAL_PROBE_REVISION * 10u;
}
